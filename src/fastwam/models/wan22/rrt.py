"""Relaxed Recursive Transformer (Bae et al. 2024, arXiv 2410.20672) for the Wan video expert.

CYCLE tying: L layers -> U = L / num_loops unique blocks; position l uses block l % U.
- lora_rank == 0: plain recursive model, shared blocks initialized with `init` (paper: Stepwise best).
- lora_rank  > 0: relaxed model, every position gets its own LoRA on every nn.Linear of the block,
  initialized by truncated SVD of (W_original_l - W_shared) (paper: Average + SVD best).
  Wan linears carry biases (Gemma in the paper has none), so relaxed positions also get an
  untied bias delta initialized to (b_original_l - b_shared); without it full rank cannot recover the model.
Norm weights and the AdaLN `modulation` stay fully tied, as in the paper.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int):
        super().__init__()
        self.base = base
        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features, dtype=base.weight.dtype, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=base.weight.dtype, device=base.weight.device))
        self.bias_delta = (
            nn.Parameter(torch.zeros(base.out_features, dtype=base.weight.dtype, device=base.weight.device))
            if base.bias is not None
            else None
        )

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        out = self.base(x) + nn.functional.linear(nn.functional.linear(x, self.lora_A), self.lora_B)
        if self.bias_delta is not None:
            out = out + self.bias_delta
        return out


def stepwise_sources(num_layers: int, num_unique: int) -> list[int]:
    # Intermediate layers at even intervals, first and last layer kept (paper Fig. 2).
    if num_unique == 1:
        return [0]
    return [int(round(j * (num_layers - 1) / (num_unique - 1))) for j in range(num_unique)]


def _shallow_view(module: nn.Module) -> nn.Module:
    view = copy.copy(module)
    view._modules = dict(module._modules)
    return view


def _lora_view(module: nn.Module, rank: int, lora_out: list) -> nn.Module:
    """Shallow copy of `module` sharing all parameters, with every nn.Linear wrapped in LoRALinear."""
    view = _shallow_view(module)
    for name, child in module._modules.items():
        if isinstance(child, nn.Linear):
            lora = LoRALinear(child, rank)
            view._modules[name] = lora
            lora_out.append(lora)
        elif child is not None and any(isinstance(m, nn.Linear) for m in child.modules()):
            view._modules[name] = _lora_view(child, rank, lora_out)
    return view


@torch.no_grad()
def apply_rrt(expert: nn.Module, num_loops: int, init: str, lora_rank: int, svd_init: bool) -> dict:
    blocks = list(expert.blocks)
    num_layers = len(blocks)
    if num_layers % num_loops != 0:
        raise ValueError(f"num_layers={num_layers} not divisible by num_loops={num_loops}")
    num_unique = num_layers // num_loops
    groups = [[j + u * num_unique for u in range(num_loops)] for j in range(num_unique)]

    # Keep original weights of linears for SVD residuals before shared blocks are overwritten.
    orig_linear_weights = None
    if lora_rank > 0 and svd_init:
        orig_linear_weights = [
            {
                n: (m.weight.detach().clone(), None if m.bias is None else m.bias.detach().clone())
                for n, m in b.named_modules()
                if isinstance(m, nn.Linear)
            }
            for b in blocks
        ]

    if init == "stepwise":
        sources = stepwise_sources(num_layers, num_unique)
        shared = [blocks[s] for s in sources]
    elif init == "average":
        sources = [g[0] for g in groups]
        shared = [blocks[g[0]] for g in groups]
        for j, g in enumerate(groups):
            group_params = [dict(blocks[i].named_parameters()) for i in g]
            for name, p in shared[j].named_parameters():
                p.copy_(torch.stack([gp[name].float() for gp in group_params]).mean(0).to(p.dtype))
    elif init == "lower":
        sources = list(range(num_unique))
        shared = [blocks[s] for s in sources]
    else:
        raise ValueError(f"Unknown rrt init `{init}`")

    new_blocks = []
    svd_rel_err = []
    for pos in range(num_layers):
        base = shared[pos % num_unique]
        if lora_rank == 0:
            new_blocks.append(base)
            continue
        loras = []
        view = _lora_view(base, lora_rank, loras)
        if orig_linear_weights is not None:
            base_linears = {n: m for n, m in base.named_modules() if isinstance(m, nn.Linear)}
            for lora, (name, lin) in zip(loras, base_linears.items()):
                assert lora.base is lin, name
                w_orig, b_orig = orig_linear_weights[pos][name]
                if b_orig is not None:
                    lora.bias_delta.copy_((b_orig.float() - lin.bias.float()).to(lora.bias_delta.dtype))
                residual = w_orig.float() - lin.weight.float()
                if residual.abs().max() == 0:
                    continue
                U, S, Vh = torch.linalg.svd(residual, full_matrices=False)
                r = min(lora_rank, S.numel())
                lora.lora_B[:, :r].copy_((U[:, :r] * S[:r]).to(lora.lora_B.dtype))
                lora.lora_A[:r].copy_(Vh[:r].to(lora.lora_A.dtype))
                approx = lora.lora_B.float() @ lora.lora_A.float()
                svd_rel_err.append(float((residual - approx).norm() / w_orig.float().norm()))
        new_blocks.append(view)

    expert.blocks = nn.ModuleList(new_blocks)
    unique_params = sum(p.numel() for p in expert.parameters())
    info = {
        "num_layers": num_layers,
        "num_loops": num_loops,
        "num_unique": num_unique,
        "init": init,
        "sources": sources,
        "lora_rank": lora_rank,
        "svd_init": bool(orig_linear_weights is not None),
        "expert_params_after": unique_params,
        "svd_rel_err_mean": (sum(svd_rel_err) / len(svd_rel_err)) if svd_rel_err else None,
    }
    logger.info("Applied RRT to video expert: %s", info)
    return info
