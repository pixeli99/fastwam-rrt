import copy

import torch
import torch.nn as nn

from fastwam.models.wan22.rrt import LoRALinear, apply_rrt, stepwise_sources
from fastwam.models.wan22.wan_video_dit import DiTBlock


class Expert(nn.Module):
    def __init__(self, n=6, d=16, ffn=40):
        super().__init__()
        self.blocks = nn.ModuleList([DiTBlock(d, 4, 4, ffn) for _ in range(n)])


def _linear_outputs(block, x):
    return [block.ffn(x), block.self_attn.q(x), block.self_attn.o(x), block.cross_attn.k(x), block.cross_attn.v(x)]


def test_full_rank_svd_recovers_original():
    torch.manual_seed(0)
    e = Expert()
    ref = copy.deepcopy(e)
    apply_rrt(e, num_loops=2, init="average", lora_rank=40, svd_init=True)
    x = torch.randn(3, 16)
    for pos in range(6):
        for a, b in zip(_linear_outputs(e.blocks[pos], x), _linear_outputs(ref.blocks[pos], x)):
            torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)
    # tied non-linear params: position 0 and 3 share modulation and norm3
    assert e.blocks[0].modulation is e.blocks[3].modulation
    assert e.blocks[0].norm3.weight is e.blocks[3].norm3.weight
    assert e.blocks[0].ffn[0].base is e.blocks[3].ffn[0].base
    assert e.blocks[0].ffn[0].lora_A is not e.blocks[3].ffn[0].lora_A


def test_recursive_stepwise_ties_blocks():
    torch.manual_seed(0)
    e = Expert(n=30)
    ref = copy.deepcopy(e)
    info = apply_rrt(e, num_loops=2, init="stepwise", lora_rank=0, svd_init=True)
    assert info["sources"][0] == 0 and info["sources"][-1] == 29
    for j in range(15):
        assert e.blocks[j] is e.blocks[j + 15]
        torch.testing.assert_close(e.blocks[j].ffn[0].weight, ref.blocks[info["sources"][j]].ffn[0].weight)
    n_ref = sum(p.numel() for p in ref.parameters())
    n_new = sum(p.numel() for p in e.parameters())
    assert n_new * 2 == n_ref


def test_state_dict_roundtrip_strict():
    torch.manual_seed(0)
    e = Expert()
    apply_rrt(e, num_loops=2, init="average", lora_rank=4, svd_init=True)
    sd = {k: v.clone() for k, v in e.state_dict().items()}
    f = Expert()
    apply_rrt(f, num_loops=2, init="average", lora_rank=4, svd_init=False)
    f.load_state_dict(sd, strict=True)
    x = torch.randn(3, 16)
    for pos in range(6):
        for a, b in zip(_linear_outputs(e.blocks[pos], x), _linear_outputs(f.blocks[pos], x)):
            torch.testing.assert_close(a, b)
    assert f.blocks[0].ffn[0].base is f.blocks[3].ffn[0].base


def test_shared_grad_accumulates_both_positions():
    torch.manual_seed(0)
    e = Expert()
    apply_rrt(e, num_loops=2, init="average", lora_rank=4, svd_init=True)
    x = torch.randn(3, 16)
    e.blocks[0].ffn(x).sum().backward()
    g0 = e.blocks[0].ffn[0].base.weight.grad.clone()
    e.blocks[3].ffn(x).sum().backward()
    assert not torch.equal(g0, e.blocks[3].ffn[0].base.weight.grad)
    assert e.blocks[3].ffn[0].lora_A.grad is not None


def test_layer_drop_keeps_stepwise_positions_in_all_experts():
    from fastwam.models.wan22.rrt import apply_layer_drop

    torch.manual_seed(0)
    mot = nn.Module()
    mot.mixtures = nn.ModuleDict({"video": Expert(n=30), "action": Expert(n=30, d=8, ffn=16)})
    mot.expert_order = ["video", "action"]
    mot.num_layers = 30
    orig = {n: list(mot.mixtures[n].blocks) for n in mot.expert_order}
    info = apply_layer_drop(mot, keep=15)
    assert mot.num_layers == 15 and info["sources"] == stepwise_sources(30, 15)
    for n in mot.expert_order:
        assert len(mot.mixtures[n].blocks) == 15
        assert all(b is orig[n][s] for b, s in zip(mot.mixtures[n].blocks, info["sources"]))
