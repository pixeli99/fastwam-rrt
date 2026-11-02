# FastWAM-RRT：面向世界动作模型（WAM）的循环视频专家网络

[![English](https://img.shields.io/badge/README-English-111111.svg)](./README.md)
[![中文](https://img.shields.io/badge/README-%E4%B8%AD%E6%96%87-d14836.svg)](./README_zh.md)

> **核心问题：** *当 Fast-WAM 证明测试时不再需要“未来视频想象（去噪生成）”之后，世界动作模型真的还需要 **5B 独立参数** 的视频主干吗？抑或它真正需要的是 **30 层迭代计算深度（Iterative Computation Depth）** 来逐步对齐视觉世界先验与动作生成？*

---

## 1. 研究动机与故事（Motivation）

### 第一幕：从“测试时视频生成”到“结构性参数失衡”
世界动作模型（World Action Models, WAMs）通常在 **Mixture-of-Transformers (MoT)** 架构中，将大规模预训练视频扩散主干（Video Expert）与动作策略网络（Action Expert）逐层耦合。近期，**Fast-WAM** ([Yuan et al., 2026](https://arxiv.org/abs/2603.16666)) 揭示了一个关键事实：WAM 在训练阶段通过未来视频联合训练（co-training）将物理与时空先验注入视频主干至关重要，但在**测试推理时**并不需要耗时的未来视频去噪生成——Action Expert 只需对 Video Expert 提取的**首帧 KV Cache** 进行交互即可高效预测动作。

然而，当测试时的视频迭代去噪被移除后，MoT 内部极端的**参数不对称性**成为了新的核心瓶颈：
- **Video Expert (`Wan2.2-TI2V`)：** 30 层，`hidden_dim=3072`，高达 **5.00B 参数**（占整个 DiT 参数量的 **83%**）。
- **Action Expert (`ActionDiT`)：** 30 层，`hidden_dim=1024`，仅 **1.02B 参数**（仅占 **17%**）。

无论是在训练阶段（5B 模型的参数、梯度、优化器状态与多卡 All-Reduce 通信开销），还是在单帧推理部署阶段（`batch_size=1` 下仅为了做一次首帧 Prefill 就必须从 HBM 高带宽显存中流式加载 5B 独立权重，属于典型的 **Memory-bandwidth bound**），庞大的 Video Expert 都显得过于沉重。

### 第二幕：MoT 的“深度困境”——为什么不能直接砍层（Layer Drop）？
想要给 5B 的 Video Expert 减重，最直观的想法是直接剪枝层数（例如从 30 层砍到 15 层）。但在 MoT 架构下，**Video Expert 和 Action Expert 的每一层 $l \in \{0, \dots, L-1\}$ 都通过 Mixed Self-Attention 严格一一配对**。直接砍掉一半层数（`drop15`）会带来两个严重后果：

1. **误伤轻量的 Action Expert：** 由于层数必须严格对齐，砍掉 15 层 Video Expert 会迫使原本只有 1.02B 的 `ActionDiT` 也被腰斩到 15 层（`0.52B`），直接削弱了策略网络自身的表达能力。
2. **丢失特征迭代修正深度（Computation Depth）：** Transformer 的表征提取与跨模态对齐本质上是一个随深度逐步递进的迭代过程。把 30 层砍成 15 层，意味着 Action Token 查询多层级视觉特征的迭代步数直接减半。

### 第三幕：我们的方案——“循环而非丢弃”（Relaxed Recursive Video Expert）
为了在压缩参数量的同时不牺牲计算深度、也不误伤 Action Expert，我们引入了 **Relaxed Recursive Transformers (RRT)** ([Bae et al., 2024](https://arxiv.org/abs/2410.20672)) 的思想，在 Video Expert 中将**迭代计算深度**与**独立参数容量**解耦：

```text
标准 FastWAM（30 层独立权重，5.00B Video + 1.02B Action）：
  Video  : [ V_0 ] -> [ V_1 ] -> ... -> [ V_14 ] -> [ V_15 ] -> [ V_16 ] -> ... -> [ V_29 ]
              |          |                  |          |           |                  |     (30 层 MoT 联合注意力)
  Action : [ A_0 ] -> [ A_1 ] -> ... -> [ A_14 ] -> [ A_15 ] -> [ A_16 ] -> ... -> [ A_29 ]

直接砍层对照组（`drop15`：15 层无循环，2.50B Video + 0.52B Action —— 深度减半且误伤 Action）：
  Video  : [ V_0 ] -> [ V_2 ] -> ... -> [ V_29 ]   (仅保留 15 层)
              |          |                  |       (15 层 MoT 联合注意力)
  Action : [ A_0 ] -> [ A_2 ] -> ... -> [ A_29 ]   (被迫缩减为 15 层)

FastWAM-RRT（`rrt_rec` / `rrt_relax`：15 个共享 Video 块循环 2 次，完整保留 30 层 MoT 与 Action）：
  Video  : [ S_0 ] -> [ S_1 ] -> ... -> [ S_14 ] ==> [ S_0 (+LoRA) ] -> ... -> [ S_14 (+LoRA) ]
              |          |                  |               |                         |     (30 层 MoT 联合注意力)
  Action : [ A_0 ] -> [ A_1 ] -> ... -> [ A_14 ] ==> [    A_15     ] -> ... -> [    A_29     ]
```

我们将 Wan2.2 的 30 层折叠为 $U=15$ 个独立基础块，并按 **CYCLE 策略循环执行 2 次**（第 $l$ 层使用第 $l \bmod 15$ 个共享块）：
- **完整保留 30 层计算深度与 30 层 Action Expert：** Video 分支依然执行完整的 30 步递进计算并输出 30 层不同的 KV Cache；1.02B 的 `ActionDiT` 保持 30 层独立参数毫发无损。
- **纯循环压缩 50% 参数（`rrt_rec`）：** 采用 **Stepwise 均匀抽层初始化**（从第 `0` 到 `29` 层等间隔抽取 15 层保留首尾与跨度特征），在不增加任何额外模块的情况下将 Video Expert 从 **5.00B 压缩至 2.50B**。
- **截断 SVD 低秩残差松弛（`rrt_relax`）：** 纯权重共享强制第 $l$ 层与第 $l+15$ 层使用完全相同的线性变换。为了最大化继承预训练 5B 模型的层间差异先验，我们先对同组层 $(l, l+15)$ 求均值作为共享基底（`average` init），再对原始权重与共享基底的残差 $(W_{\text{orig}, l} - W_{\text{shared}})$ 进行**截断 SVD 分解**，直接初始化到每个位置独立的 LoRA 旁路（$r=512$）上。
- **针对 Wan2.2 的 Bias Delta 精确补偿：** 原 RRT 论文使用的 Gemma 没有 Linear bias，而 Wan2.2 的 Linear 层带有 bias。我们在 [`LoRALinear`](./src/fastwam/models/wan22/rrt.py) 中专门引入了不共享的 `bias_delta = b_orig - b_shared`，保证全秩 SVD 下能 **100% 无损还原**原预训练模型输出。

---

## 2. 实验设计（四组受控消融）

四个配置严格遵循完全相同的 LIBERO 训练配方（`16 GPUs × batch 1 = global_batch 16`，`10` epochs，`lr = 1e-4` cosine，`wd = 1e-2`，shift `5.0`），形成严密的对比闭环：

| 任务配置 (`task=`) | Video 权重策略 | MoT 层数 | 初始化方式 | Video 参数量 | Action 参数量 | 实验目的 |
| :--- | :--- | :---: | :--- | ---: | ---: | :--- |
| `libero_uncond_2cam224_1e-4` | 无共享（全量模型） | 30 | 预训练 Wan2.2 | 5.00 B | 1.02 B | 全量性能上限 Baseline |
| `libero_uncond_2cam224_1e-4_drop15` | 直接砍半（**不循环**） | **15** | Stepwise 抽取 15 层 | **2.50 B** | **0.52 B** | 非循环对照组（与 `rrt_rec` Video 参数完全相同，但深度减半） |
| `libero_uncond_2cam224_1e-4_rrt_rec` | **纯循环（15 层 × 循环 2 次）** | **30** | Stepwise 抽取 15 层 | **2.50 B** | **1.02 B** | 验证“相同 2.5B 参数下，循环跑满 30 层是否显著优于直接砍层” |
| `libero_uncond_2cam224_1e-4_rrt_relax` | **松弛循环（循环 2 次 + SVD LoRA $r=512$）** | **30** | 同组均值 + 截断 SVD | **3.84 B** | **1.02 B** | 验证“共享主干 + SVD 低秩残差”能否逼近甚至追平 5B 全量模型 |

---

## 3. 代码结构说明

- **RRT 与 Layer-Drop 核心实现：** [`src/fastwam/models/wan22/rrt.py`](./src/fastwam/models/wan22/rrt.py)
  - `apply_rrt(expert, num_loops, init, lora_rank, svd_init)`：支持 `stepwise`、`average`、`lower` 三种共享初始化，以及带 `bias_delta` 的截断 SVD `LoRALinear` 视图包装（Norm 与 AdaLN `modulation` 保持跨循环共享）。
  - `apply_layer_drop(mot, keep)`：非循环控制组，在 Video 和 Action 两个专家中同步保留相同的 15 个 Stepwise 层位置 `[0, 2, 4, 6, 8, 10, 12, 14, 17, 19, 21, 23, 25, 27, 29]`。
- **模型构建接入：** [`src/fastwam/runtime.py`](./src/fastwam/runtime.py)（在 `create_fastwam` 中解析 `model.rrt` 与 `model.layer_drop`，并在评测加载 checkpoint 时自动跳过耗时的预训练 SVD 初始化）。
- **单元测试：** [`tests/test_rrt.py`](./tests/test_rrt.py)（验证全秩 SVD 精确还原、参数减半、严格 `state_dict` 读写闭环及跨循环层梯度累加）。

---

## 4. 快速开始

### 1) 环境安装与预训练权重准备

```bash
conda create -n fastwam python=3.10 -y
conda activate fastwam
pip install -U pip
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .

# 设置权重目录并生成插值初始化的 ActionDiT 主干
mkdir -p checkpoints
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"

python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16
```

### 2) 运行 RRT 单元测试

```bash
python -m pytest tests/test_rrt.py
```

### 3) 启动 LIBERO 训练（16 卡 × Batch 1 = Global Batch 16）

```bash
# 预计算 T5 文本嵌入缓存
torchrun --standalone --nproc_per_node=8 scripts/precompute_text_embeds.py task=libero_uncond_2cam224_1e-4

# 1. 纯循环模型 (2.50B Video, 循环 2 次, Stepwise 初始化)
bash scripts/train_zero1.sh 16 task=libero_uncond_2cam224_1e-4_rrt_rec

# 2. 松弛循环模型 (3.84B Video, 循环 2 次, Average + 截断 SVD LoRA r=512)
bash scripts/train_zero1.sh 16 task=libero_uncond_2cam224_1e-4_rrt_relax

# 3. 非循环砍层对照组 (15 层, 不循环)
bash scripts/train_zero1.sh 16 task=libero_uncond_2cam224_1e-4_drop15
```

### 4) 运行 LIBERO 评测

```bash
python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4_rrt_relax \
  ckpt=/path/to/checkpoints/weights/step_xxxxxx.pt \
  EVALUATION.sigma_shift=5.0 \
  MULTIRUN.num_gpus=8
```

---

## 参考文献

- **Fast-WAM:** Tianyuan Yuan, Zibin Dong, Yicheng Liu, Hang Zhao. *Fast-WAM: Do World Action Models Need Test-time Future Imagination?* ([arXiv:2603.16666](https://arxiv.org/abs/2603.16666))
- **Relaxed Recursive Transformers:** Sangmin Bae et al. *Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA.* ([arXiv:2410.20672](https://arxiv.org/abs/2410.20672))
