# FastWAM-RRT: Recursive Video Experts for World Action Models

[![English](https://img.shields.io/badge/README-English-111111.svg)](./README.md)
[![中文](https://img.shields.io/badge/README-%E4%B8%AD%E6%96%87-d14836.svg)](./README_zh.md)

> **Core Question:** *After Fast-WAM removes test-time video denoising, do World Action Models still need 5B **unique** video parameters—or do they primarily need the **iterative computation depth** of a 30-layer backbone to align visual world priors with action generation?*

---

## 1. Motivation & Story

### From Test-Time Imagination to Structural Asymmetry
World Action Models (WAMs) couple a large pretrained video diffusion backbone with an action policy inside a layer-wise **Mixture-of-Transformers (MoT)**. Recently, **Fast-WAM** ([Yuan et al., 2026](https://arxiv.org/abs/2603.16666)) showed that WAMs do **not** need test-time future video imagination: co-training with future video prediction is essential to shape physical and spatial representations inside the video backbone, whereas at test time the action expert only needs to attend to the **first-frame KV cache** of the video expert in a single forward pass.

However, once test-time video generation is removed, a glaring **structural asymmetry** becomes the primary bottleneck:
- **Video Expert (`Wan2.2-TI2V`):** 30 layers, `hidden_dim=3072`, **5.00B parameters** (~83% of the DiT).
- **Action Expert (`ActionDiT`):** 30 layers, `hidden_dim=1024`, **1.02B parameters** (~17% of the DiT).

During training, the 5B video expert dominates GPU memory (parameters, gradients, optimizer states) and all-reduce communication. During deployment (`batch_size=1`), streaming 5B unique weights from HBM merely to prefill a single observation frame makes inference heavily **memory-bandwidth bound**.

### The MoT Depth Dilemma: Why Not Just Drop Layers?
A natural idea to compress the 5B video expert is layer pruning—e.g., keeping only 15 of the 30 layers. Yet in an MoT architecture, **video and action blocks are coupled 1-to-1 at every depth $l \in \{0, \dots, L-1\}$** via joint self-attention:

1. **Collateral Damage to the Action Expert:** Dropping 15 layers from the video expert forces the MoT to also amputate half of the lightweight `ActionDiT` (`30 → 15` layers, `1.02B → 0.52B`).
2. **Loss of Iterative Refinement Depth:** Representation extraction and cross-modal alignment in Transformers are inherently iterative. Cutting the network from 30 steps to 15 halves the number of sequential refinement steps over which action tokens can query hierarchical visual features.

### Our Idea: Loop, Don't Drop (Relaxed Recursive Video Expert)
Instead of sacrificing depth to save parameters, we **decouple computation depth from unique parameter count** by turning the Wan2.2 video expert into a **Relaxed Recursive Transformer (RRT)** ([Bae et al., 2024](https://arxiv.org/abs/2410.20672)):

```text
Standard FastWAM (30 unique layers, 5.00B video + 1.02B action):
  Video  : [ V_0 ] -> [ V_1 ] -> ... -> [ V_14 ] -> [ V_15 ] -> [ V_16 ] -> ... -> [ V_29 ]
              |          |                  |          |           |                  |     (30-layer MoT Mixed Attn)
  Action : [ A_0 ] -> [ A_1 ] -> ... -> [ A_14 ] -> [ A_15 ] -> [ A_16 ] -> ... -> [ A_29 ]

Layer Drop Control (`drop15`: 15 layers, 2.50B video + 0.52B action — loses half the depth & action capacity):
  Video  : [ V_0 ] -> [ V_2 ] -> ... -> [ V_29 ]   (15 blocks only)
              |          |                  |       (15-layer MoT Mixed Attn)
  Action : [ A_0 ] -> [ A_2 ] -> ... -> [ A_29 ]   (15 blocks only)

FastWAM-RRT (`rrt_rec` / `rrt_relax`: 15 shared video blocks looped 2x, 30-layer MoT intact):
  Video  : [ S_0 ] -> [ S_1 ] -> ... -> [ S_14 ] ==> [ S_0 (+LoRA) ] -> ... -> [ S_14 (+LoRA) ]
              |          |                  |               |                         |     (30-layer MoT Mixed Attn)
  Action : [ A_0 ] -> [ A_1 ] -> ... -> [ A_14 ] ==> [    A_15     ] -> ... -> [    A_29     ]
```

By folding $L=30$ video layers into $U=15$ unique blocks cycled twice (`pos % 15`, **CYCLE tying**):
- **30-Step Depth & Full Action Expert Preserved:** The video branch still executes 30 sequential steps and produces 30 distinct layers of KV cache, while the 30-layer `ActionDiT` (1.02B) remains completely untouched.
- **50% Video Parameter Compression (`rrt_rec`):** Using **Stepwise initialization** (uniformly sampling 15 source layers spanning layer `0` to `29`), the video expert shrinks from **5.00B → 2.50B** with zero architectural changes to `MoT`.
- **SVD Residual Relaxation (`rrt_relax`):** To preserve layer-specific knowledge from the pretrained 5B Wan2.2 checkpoint without losing weight sharing, we initialize 15 shared blocks by **group-wise averaging** $(l, l+15)$ and attach lightweight position-specific LoRA adapters ($r=512$) initialized via **truncated SVD** of the weight residual $(W_{\text{orig}, l} - W_{\text{shared}})$.
- **Exact Bias Recovery for Wan2.2:** Unlike Gemma in the original RRT paper (bias-free linears), Wan2.2 linear layers carry biases. We augment each relaxed position with an untied `bias_delta` initialized to $(b_{\text{orig}, l} - b_{\text{shared}})$, guaranteeing that full-rank SVD recovers the original pretrained model *exactly*.

---

## 2. Experimental Matrix (Controlled Ablation)

All four configurations share the exact same LIBERO training recipe (`global_batch_size = 16`, `10` epochs, `lr = 1e-4` cosine, `wd = 1e-2`, video/action flow shift `5.0`), isolating **parameter count** vs. **iterative depth** vs. **low-rank relaxation**:

| Task Config | Video Tying | MoT Depth | Init Strategy | Video Params | Action Params | Scientific Role |
| :--- | :--- | :---: | :--- | ---: | ---: | :--- |
| `libero_uncond_2cam224_1e-4` | None (Full) | 30 | Pretrained Wan2.2 | 5.00 B | 1.02 B | Upper-bound baseline |
| `libero_uncond_2cam224_1e-4_drop15` | Layer Drop (No Loop) | **15** | Stepwise (15 layers) | **2.50 B** | **0.52 B** | Non-looped control (same video params as `rrt_rec`, half depth) |
| `libero_uncond_2cam224_1e-4_rrt_rec` | **Recursive (2× Loop)** | **30** | Stepwise (15 shared) | **2.50 B** | **1.02 B** | Tests whether looping recovers 30-layer performance at 50% video size |
| `libero_uncond_2cam224_1e-4_rrt_relax` | **Relaxed RRT (2× Loop + LoRA $r=512$)** | **30** | Average + Truncated SVD | **3.84 B** | **1.02 B** | Tests whether SVD low-rank residuals close the gap to the 5B model |

---

## 3. Implementation Highlights

- **Core RRT & Layer-Drop Module:** [`src/fastwam/models/wan22/rrt.py`](./src/fastwam/models/wan22/rrt.py)
  - `apply_rrt(expert, num_loops, init, lora_rank, svd_init)`: Applies CYCLE weight tying (`stepwise`, `average`, or `lower`) and optional SVD-initialized `LoRALinear` (with `bias_delta`) per position while keeping LayerNorm and AdaLN `modulation` parameters tied.
  - `apply_layer_drop(mot, keep)`: Non-looped control that keeps the exact same Stepwise-selected positions `[0, 2, 4, 6, 8, 10, 12, 14, 17, 19, 21, 23, 25, 27, 29]` across both experts without looping.
- **Model Builder Integration:** [`src/fastwam/runtime.py`](./src/fastwam/runtime.py) (`create_fastwam` accepts `rrt` or `layer_drop` from Hydra configs and skips redundant SVD decomposition when building an uninitialized model for checkpoint evaluation).
- **Unit Tests:** [`tests/test_rrt.py`](./tests/test_rrt.py) verifies full-rank SVD exact recovery, parameter halving under Stepwise tying, strict `state_dict` save/load roundtrips, and shared-weight gradient accumulation across loop positions.

---

## 4. Quickstart

### 1) Environment & Pretrained Backbone Setup

```bash
conda create -n fastwam python=3.10 -y
conda activate fastwam
pip install -U pip
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .

# Set checkpoint directory and generate interpolated ActionDiT backbone
mkdir -p checkpoints
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"

python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16
```

### 2) Run Unit Tests

```bash
python -m pytest tests/test_rrt.py
```

### 3) Training on LIBERO (16 GPUs × Batch 1 = Global Batch 16)

```bash
# Precompute T5 text embeddings once
torchrun --standalone --nproc_per_node=8 scripts/precompute_text_embeds.py task=libero_uncond_2cam224_1e-4

# 1. Pure Recursive Video Expert (2.50B video, 2x loop, Stepwise init)
bash scripts/train_zero1.sh 16 task=libero_uncond_2cam224_1e-4_rrt_rec

# 2. Relaxed Recursive Video Expert (3.84B video, 2x loop, Average + SVD LoRA r=512)
bash scripts/train_zero1.sh 16 task=libero_uncond_2cam224_1e-4_rrt_relax

# 3. Non-Looped Layer-Drop Control (15 layers, no loop)
bash scripts/train_zero1.sh 16 task=libero_uncond_2cam224_1e-4_drop15
```

### 4) Evaluation on LIBERO

```bash
python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4_rrt_relax \
  ckpt=/path/to/checkpoints/weights/step_xxxxxx.pt \
  EVALUATION.sigma_shift=5.0 \
  MULTIRUN.num_gpus=8
```

---

## References

- **Fast-WAM:** Tianyuan Yuan, Zibin Dong, Yicheng Liu, Hang Zhao. *Fast-WAM: Do World Action Models Need Test-time Future Imagination?* ([arXiv:2603.16666](https://arxiv.org/abs/2603.16666))
- **Relaxed Recursive Transformers:** Sangmin Bae et al. *Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA.* ([arXiv:2410.20672](https://arxiv.org/abs/2410.20672))
