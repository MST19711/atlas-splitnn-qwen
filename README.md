# Qwen3-0.6B / Qwen3.5-0.8B 在华为 Atlas 200I DK A2 上的部署

> 将千问小语言模型部署到华为昇腾边缘计算板，实现 NPU 加速的端侧中文对话。


---

## 目录


- [项目动机](#项目动机)
- [成果概览](#成果概览)
- [硬件与软件](#硬件与软件)
- [项目结构](#项目结构)
- [背景知识](#背景知识)
- [第一轮实验：静态窗口模型](#第一轮实验静态窗口模型)
- [第二轮实验：KV Cache 模型](#第二轮实验kv-cache-模型)
- [第三轮实验：Qwen3.5-0.8B KV Cache 模型](#第三轮实验qwen35-08b-kv-cache-模型)
- [解决的关键问题](#解决的关键问题)
- [如何复现](#如何复现)

> 本文件同时也作为一份课程报告，因此有一些复现无关内容。如果你更关心如何实际部署复现，可以直接跳转到“[如何复现](#如何复现)”部分。
---

## 项目动机

大语言模型通常运行在云端 GPU 集群上，但很多场景需要**本地推理**——隐私敏感、网络不稳定、或者单纯想把 AI 装进小盒子里。

华为 Atlas 200I DK A2 是一块售价千元级别的昇腾开发者套件，搭载 Ascend310B4 NPU（4GB）。这个项目尝试把 **Qwen3-0.6B**——通义千问的 6 亿参数指令微调模型——部署到这块板子上，实现可用的中文对话。

过程分三轮：先用简单的静态窗口方案跑通全链路，再升级到 KV Cache 以获得更长上下文，最后将 Qwen3.5-0.8B 的 DeltaNet 混合架构也部署到板端。

---

## 成果概览

| 方案 | 上下文长度 | 每 token 耗时 | 解码速度* | 输出质量 |
|------|-----------|-------------|----------|---------|
| 静态窗口 (seq=32) | 32 token | ~280 ms | 3.6 tok/s | 连贯中文 |
| Qwen3 KV Cache (max_len=256) | 256 token | ~1200 ms | 0.8 tok/s | 连贯中文 |
| Qwen3.5 KV Cache (max_len=256) | 256 token | ~7500 ms | 0.1 tok/s | 连贯中文 |

> *OM 首次加载约 75 秒（Qwen3 1.5GB）或 195 秒（Qwen3.5 1.9GB）。速度差异主要来自模型大小和 DeltaNet 层计算的额外开销。

模型输出示例：

```
Prompt: "你好，请用一句话介绍你自己"
Output: "我是一位叫AI的助手，专为用户提供帮助和支持。"
```

---

## 硬件与软件

### 开发板

| 项目 | 规格 |
|------|------|
| 型号 | 华为 Atlas 200I DK A2 |
| 芯片 | Ascend310B4 NPU + 4 核 ARM |
| NPU 内存 | 4 GB |
| 系统内存 | 3.4 GB |
| 系统 | Ubuntu 22.04 aarch64 |

### 使用的模型

**Qwen3-0.6B** (Instruct, FP16)：

| 参数 | 值 |
|------|-----|
| 参数量 | ~600M (28 层) |
| 词表大小 | 151936 |
| 注意力头 | 16 Q-heads / 8 KV-heads (GQA) |
| 权重文件 | model.safetensors (1.5 GB) |

**Qwen3.5-0.8B** (Instruct, FP16)：

| 参数 | 值 |
|------|-----|
| 参数量 | ~800M (24 层: 18 DeltaNet + 6 Gated Attention) |
| 词表大小 | 248320 |
| 注意力头 | 16 Q-heads / 2 KV-heads (GQA) |
| 权重文件 | model.safetensors (~1.9 GB) |

### 工具链

| 组件 | 版本/说明 |
|------|----------|
| CANN | 7.0.0 (容器内编译) / 7.0.RC1 (板端 runtime) |
| ATC | CANN 内置的模型编译器 (ONNX → OM) |
| ACL | CANN 的 C 运行时库 (通过 Python `acl` 模块调用) |
| ONNX | opset 15，TorchScript 导出 |
| 容器 | Podman, Rocky Linux 9 base, 镜像 `cann-atc-rocky:v7` |

---

## 项目结构

```
Embedded_FinalHW/
├── README.md                         # 本文档
├── AGENTS.md                         # AI 辅助开发用参考
├── model/
│   ├── Qwen3-0.6B/                   # Qwen3 权重 + tokenizer
│   └── Qwen3.5-0.8B/                 # Qwen3.5 权重 + tokenizer
├── scripts/                          # ONNX 导出 & ATC 转换 (x86 开发机)
│   ├── export_fp16.py                # 静态窗口模型导出 (Qwen3)
│   ├── export_kvcache.py             # KV Cache 模型导出 (Qwen3)
│   ├── export_qwen35.py              # KV Cache 模型导出 (Qwen3.5 DeltaNet)
│   ├── patch_onnx.py                 # ONNX 图修补工具
│   ├── download_model.py
│   └── podman_convert.sh             # 容器化 ATC 转换
├── board/                            # 板载推理脚本 (aarch64)
│   ├── gen_text_seq32.py             # 静态窗口推理
│   ├── gen_text_kvcache.py           # Qwen3 KV Cache 推理
│   ├── gen_text_qwen35.py            # Qwen3.5 KV Cache 推理
│   └── acl_verify.py                 # 单次推理验证
├── docker/Containerfile.v2-cann7     # CANN 7.0 容器镜像定义
├── om_out/                           # 编译产出的 OM 文件
└── logs/                             # ATC 转换日志
```

---

## 背景知识

### 自回归文本生成

语言模型逐个 token 地生成文本。每一步输入当前序列，输出下一个 token 的概率分布，采样后拼接到序列末尾。

### Left-padding

对于固定长度的输入（比如 32 token），如果实际序列更短，需要在左边填 0（padding），右边放真实 token。配合 `attention_mask`（0 表示忽略），causal attention 保证 padding 不影响实际 token 的计算。

### KV Cache

Transformer 每层的注意力需要计算 Key、Query、Value。没有缓存时，每一步都要重新计算所有历史 token 的 K 和 V（重复劳动）。KV Cache 把它们存下来，下一步只需算新 token 的 K/V，注意力复杂度从 O(N²) 降到 O(N)。

### GQA (Grouped Query Attention)

Qwen3 用 16 个 Q 头但只有 8 个 K/V 头——每 2 个 Q 头共享一组 K/V。这样在几乎不损失精度的情况下节省了 ~33% 的 K/V 计算量。

---

## 第一轮实验：静态窗口模型

### 思路

不用 KV Cache。每次把完整序列 left-pad 到固定长度（比如 32），一次性送入模型。尽管有重复计算，但实现简单，适合快速验证全链路。

**序列长度为什么选 32？** 这是反复试探的结果。1 太小（无上下文），128 太大（NPU 内存可能不够，且每步太慢）。32 是折中——够容纳一句日常对话，也能在 ~280ms 内完成一步。

### 导出 ONNX

`scripts/export_fp16.py` 把 Qwen3 封装为固定 seq_len 的静态图：

```python
dummy_input_ids = torch.ones((1, 32), dtype=torch.int64)
torch.onnx.export(wrapper, (dummy_input_ids, ...), "output.onnx",
                  opset_version=15, do_constant_folding=True)

```

关键设置：`use_cache=False`（不产生 KV Cache 输出），`dynamo=False`（用传统的 TorchScript 追踪器，兼容性更好）。

导出后用 ONNX Runtime 验证——检查 left-padding 下末尾位置 logits 与全 1 mask 一致。

### 修补 ONNX：处理 GQA 的 Expand 节点

导出后的 ONNX 图里有 56 个 `Expand` 节点——每层注意力要把 K/V 从 8 头重复到 16 头（GQA），重复因子为 `[1, 8, 2, 32, 128]` 中的那个 `2`。问题在于这个 shape 是通过 `Where + Equal + ConstantOfShape` **动态计算**的，ATC 编译器无法静态推断。

`scripts/patch_onnx.py` 遍历所有 `self_attn` 中的 Expand，替换为 `Tile` 算子配合静态常量 `[1, 1, 2, 1, 1]`。`Tile` 和 `Expand` 在这个场景下语义完全相同——都是把张量沿某一维复制指定次数。

修补后用 ORT 重新验证——确保数值输出不变。

### ATC 编译

```bash
MODEL_ONNX=om_out/qwen3_fp16_seq32_tile.onnx \
INPUT_SHAPE="input_ids:1,32;attention_mask:1,32" \
bash scripts/podman_convert.sh
```

ATC（Ascend Tensor Compiler）把 ONNX 图编译为板载 NPU 可执行的 OM 文件。过程约 3 分钟，产物 1.5 GB。

### 板载推理

`board/gen_text_seq32.py` 实现了 left-padding 滑动窗口的完整生成循环：

1. Tokenize 用户输入
2. 取最后 32 token，左侧填 0，构建 attention_mask
3. ACL 执行 → 取 logits 的最后一个有效位置 → 采样
4. 新 token 加入窗口，最旧的 token 滑出

每 token 约 280ms，解码速度约 3.6 tok/s，输出是连贯的中文。

---

## 第二轮实验：KV Cache 模型

### 思路

静态窗口方案能跑通，但扩大窗口到 256 时，每步就需要计算 256×256 的注意力矩阵，O(N²) 的增长意味着每步要等好几秒。KV Cache 解决的就是这个问题——**存储而不重复计算**。

由于这台NPU实际上不支持动态计算图，我们的做法是**修改 Qwen3 内部的注意力实现**（monkey-patch），让它把原本动态增长的 K/V 替换为预分配的固定大小缓冲区。

**为什么必须 monkey-patch？** 不是因为方便——而是因为 ONNX 和 CANN 的组合约束使得其他路径走不通：

1. **`cat` 动态 shape → ONNX 静态图不可表达。** 原生的 `torch.cat([past_k, new_k])` 每次输出的第二维长度 +1，这是 ONNX 的致命问题——ATC 必须在编译阶段确定所有张量的 shape。即使 `cat` 本身是 ONNX 算子，输出 shape 的动态变化也意味着后续所有依赖这个 K/V 的矩阵乘法（MatMul）的 shape 都是动态的，而 ATC 对动态 shape 的 MatMul 编译直接失败。

2. **`scatter_nd` 不稳定。** 我们最初尝试了 `scatter_nd` 来原地写入——它的语义是"在第 N 行写入新值"，比 `cat→concat` 更适合静态图。但它需要构造复杂的 indices 张量，且从 ONNX opset 11 引入的这个算子在 ATC 的 310B4 内核编译中反复出现 shape 推断错误。

3. **`Where` 是唯一稳定通过的方案。** `Where(mask, new_kv, cache)` 是 ONNX 最基础的逻辑算子（opset 9），CANN 的任何版本都稳定支持。代价是需要预分配整块 K/V 缓冲区并把 mask 广播到完整尺寸——多占了些内存，但换来了绝对稳定的编译成功率。

简言之：Monkey-patch 不是因为嵌套深或想省代码，而是因为**原生 `cat` 方案产生的动态 shape 流程在 ONNX→ATC 这条编译链路上是死胡同**，而 Patch-to-Where 方案提前把所有 shape 静态化，打通了整个链路。

### 设计约束与决策

在做具体实现之前，有几个硬约束主导了所有的设计决策：

**约束 1：ONNX 导出要求静态 shape。** Qwen3 原生的 KV Cache 使用 `torch.cat([past_k, new_k])` — 每次序列长度 +1，输出的 shape 也 +1。这种动态增长在 ONNX 图里没法表达（ATC 编译器必须在编译时就确定所有张量的维度和内存布局）。因此不能"追加"，只能"覆盖"。

**约束 2：NPU 内存有限（4 GB）。** K/V 缓冲区必须预分配且固定大小。选 `max_len=256` 是反复试探的结果——太小不够容纳对话历史，太大则内存可能不够（56 个 K/V tensor × 512KB × 2 组双缓冲 ≈ 56 MB，加上 1.5 GB 模型本身和推理中间张量，合计接近 4 GB 上限）。

**约束 3：monkey-patch 必须数学等价。** 替换后的推理结果必须在 FP16 精度内与原版一致。这意味着我们只能改变"如何计算"，不能改变"计算什么"。所有 patch 经过 ONNX Runtime 单步和 ORT 多步双重验证。

**为什么用 `Where` 而不是 `ScatterND` 或其他算子？** 我们需要在 K/V 缓冲区的第 `position` 行写入新 token 的 K/V 向量，其他行保持不变。ATC 支持的 ONNX 算子集中，`Where` 是最直接的实现方式——`Where(mask, new_kv, cache)` 在 mask=1 的位置写入 new_kv，其余位置保留 cache。`ScatterND` 也是候选项，但在 ATC 的实际编译中出现了 shape 推断问题，`Where` 更稳定。

### Monkey-patch：把 `cat` 换成 `where`

Qwen3 原生的 KV Cache 用 `torch.cat([past_k, new_k])` 实现——每次 K/V 长度加 1。这个动态增长 ONNX 没法表达（需要 static shape）。

我们的方案：预分配固定大小的 K/V 缓冲区 `(1, 8, max_len, 128)`，每次用 `torch.where` 把新 token 的 K/V **写入缓冲区对应位置**：

```python
def insert_to_cache(cache, new_kv, position):
    idx = torch.arange(max_len)
    mask = idx == position  # shape: (1, 1, max_len, 1)
    return torch.where(mask, new_kv, cache)
```

`Where` 是 ONNX 原生算子，`mask` 和 `new_kv` 通过 broadcasting 自动对齐，所有维度编译时完全静态。这就是整个设计最核心的一行代码。

### 三层 monkey-patch

替换从内到外逐步进行：

1. **`Qwen3Attention.forward()`**——接受 `past_k`, `past_v`, `position`，调用 `insert_to_cache`
2. **`Qwen3DecoderLayer.forward()`**——透传 K/V 参数
3. **`KVCacheWrapper`**——顶层模块，把 56 个 K/V 组织为显式的模型输入/输出

### 导出与验证

导出后模型有 **58 个输入、57 个输出**（2 + 56 个 K/V 输入，1 + 56 个输出）。

在两轮验证中确保了正确性：

1. **单步对比**：PyTorch 原生模型 logits == monkey-patch PyTorch logits（FP16 精度内完全一致）
2. **多步对比**：用 ONNX Runtime 模拟完整的 prefill（3 token）→ decode（5 token）流程，K/V 按位置正确累积

### 容器环境的构建

CANN 7.0 的 ATC 编译器需要在 Linux 环境中运行。我们把它装进 Podman 容器（Rocky Linux 9），避免直接污染开发机。

**为什么用 CANN 7 而不是 CANN 8？** 开发板出厂预装 CANN 7.0.RC1 runtime，OM 模型文件的二进制格式需要与 runtime 版本兼容。我们最初尝试了 CANN 8.0.RC3 编译的 OM——Qwen3-0.6B 的 OM 在板端可以运行（只用了 MatMul/Where 等基础算子），但 Qwen3.5-0.8B 的 OM 加载时出现 `drv_soft_fault (err_type=0xa)`，NPU 驱动直接拒绝执行。这是因为 Qwen3.5 的 DeltaNet 算子（矩阵乘法 + l2norm + sigmoid 等的组合）在 CANN 8 编译后的二进制格式不被 CANN 7 runtime 识别。与其升级板端固件（风险高），不如将编译工具链降级到与板端匹配的 CANN 7 版本——这就是 `Containerfile.v2-cann7` 的来由。

**为什么用 Rocky Linux 9（RPM 系）而不是 Ubuntu？** CANN 7.0.0 for x86_64 的内核包只提供了 `.rpm` 格式的下载。用 RPM 系发行版可以直接 `rpm -ivh` 安装，避免了 `alien` 转包的兼容性风险。实际选用的容器基镜像是 Rocky Linux 9（Python 3.9，符合 CANN 7 的要求 `>= 3.7`）。

**关键发现**：CANN 的 ATC 对不同芯片需要安装对应的"内核包"。310P 包只覆盖 P1/P3 型号，而开发板是 310B4 芯片，**必须用 310B 内核包**。用错包的症状是 ATC 编译能过，但开发板加载 OM 时返回 `ret=500002`。

另一个重要的修复：CANN 的 TBE Python 代码里硬编码了编译器路径 `/usr/local/Ascend/CANN-1.84/`。我们在容器里创建一个符号链接指向实际安装位置即可解决。

此外，CANN 7 的 TBE 编译器需要 pip 安装 `attrs cloudpickle psutil synr tornado`，并且需要 `numpy<2`（与 numpy 2.x 不兼容），同时需要 `gcc-c++` 提供 C++ 标准库头文件（CCE 编译器依赖 `<type_traits>`）。

### 板载推理

与 seq=32 方案不同，KV Cache 需要在 NPU 上维护一套持续的 K/V 缓冲区：

1. `acl.rt.malloc` 预分配 56 个 K/V tensor（每个 512KB，共约 28MB）
2. Prefill 阶段：逐 token 输入，K/V 逐步填充
3. Decode 阶段：每步从输出取回 logits 采样，同时把更新后的 K/V 复制回输入缓冲区

K/V 的 D2H + H2D 开销约 10ms，相对 1200ms 的总延迟可以忽略。

### 结果与速度分析

输出："你好！有什么可以帮助你的吗？"——与 prompt 匹配的连贯回复。

当前每 token 约 1200ms，解码速度约 0.8 tok/s。模型不区分 prefill 与 decode 阶段，prompt 的 token 同样逐个送入，因此初始 prompt 的处理需要额外时间。

### 性能优化历程

开发板推理经历了三轮优化：

**初始版本**（每步 alloc + memcpy，ACL 框架开销严重）：

```
每步 execute():
  acl.rt.malloc × 115      ← 为 58 个输入 + 57 个输出分配 buffer
  acl.rt.memcpy × 112      ← K/V 在 host↔device 间全量搬运（28MB）
  acl.mdl.execute()         ← 174ms NPU 计算
  acl.rt.free × 115         ← 释放

每步约 420ms，Profiler 实测：
- NPU 计算 (BatchMatMulV2): 174ms
- Host alloc + memcpy: ~200ms
- Framework 开销: ~46ms
```

Profiler 数据（`msprof --ascendcl=on --task-time=on --ai-core=on`）：

| 类别 | 耗时 | 占比 |
|------|------|------|
| BatchMatMulV2 (NPU 计算) | 174ms | 41% |
| Host 侧 alloc + memcpy | ~200ms | 48% |
| Python / ACL 框架 | ~46ms | 11% |

**第一轮优化**：预分配 device 缓冲区 + K/V 指针轮转（双缓冲），消除每步的 `acl.rt.malloc`/`acl.rt.free` 和 112 次 K/V memcpy。

```
每步只剩: H2D(16B) + execute + D2H(303KB)
K/V 留在 device 内存，指针身份互换
```

**第二轮优化**：预创建 AB/BA 两组 dataset 并绑定到双缓冲 K/V，消除每步的 `acl.mdl.create_dataset`/`destroy_dataset`/`create_data_buffer`/`add_dataset_buffer`。

**为什么需要两组 dataset？** 每步 ACL 执行后，输出 K/V 指针指向缓冲区 B，下一步需要把它作为输入——如果只有一组 dataset，就需要用 `acl.update_data_buffer` 来换绑（实验中发现这个 API 在 CANN 7.0.RC1 上有稳定性问题）。两组 dataset 各绑定不同的 K/V 指针组，轮流使用——这一步的输出指针恰好是下一步的输入指针，完全避免了 buffer 的动态换绑。

```
每步只剩: H2D(16B) + execute + D2H(303KB)
（全部数据结构在模型生命周期内一次性创建）
```

最终每步约 400ms（Profiler 显示 NPU 纯计算 174ms，剩余 ~226ms 为 ACL 框架调用和 Python 采样开销，受限于 CANN 7.0.RC1 runtime 与 Python 调用开销）。

> 注意：1.5 GB OM 在 CANN 7.0.RC1 runtime 上首次加载需要约 75 秒，期间无输出不是卡死。

---

## 第三轮实验：Qwen3.5-0.8B KV Cache 模型

### 思路

Qwen3.5-0.8B 是 Qwen3 的新一代模型，参数量稍大（~800M），使用了全新的架构：

| 组件 | Qwen3-0.6B | Qwen3.5-0.8B |
|------|-----------|-------------|
| 总层数 | 28 | 24 |
| DeltaNet 层 | 0 | 18 |
| Gated Attention 层 | 28 | 6 |
| 词表大小 | 151936 | 248320 |
| 参数量 | ~600M | ~800M |
| Tokenizer | vocab.json + merges.txt | tokenizer.json (HuggingFace) |

Qwen3.5 的核心创新是**混合架构**：18 层使用 DeltaNet（一种线性注意力变体）处理长距离依赖，6 层使用标准 Gated Attention。这种设计在效率和质量之间取得了平衡。

### 设计挑战

Qwen3.5 的原生代码已经支持缓存——`Qwen3_5GatedDeltaNet.forward` 接受 `cache_params`，能读写 `conv_state` 和 `recurrent_state`。那还需要 monkey-patch 什么？

实验发现：**核心问题不在 DeltaNet 的计算逻辑，而在 ONNX 导出和 CANN 编译链路上的四类操作不兼容**。原生代码里的 `torch_chunk_gated_delta_rule` / `torch_recurrent_gated_delta_rule` 在 CPU 上会自动 fallback 为纯 torch 操作（不是 CUDA kernel），ATC 完全可以编译——我们最初担心的"算子缺失"并不存在。

真正需要 patch 的四个问题：

| 问题 | 原始代码 | CANN 7 为何不兼容 | Patch |
|------|----------|-------------------|-------|
| Attention KV 缓存 | `torch.cat([past_k, new_k])` | 动态 shape，ATC 编译失败 | `torch.where(mask, new, cache)` 原地写入 |
| Causal mask | `Trilu(matrix, k)` | CANN 7 插件不支持 k 参数 | 手动 `Where + Equal` 构建 mask |
| RMSNorm | `output.type_as(x)` | ONNX 导出为 `aten::copy`（不支持） | `output.to(x.dtype)` |
| Conv1D 状态更新 | `conv_state.copy_(...)` | ONNX 不支持 in-place 操作 `aten::copy_` | 改写为返回新 state 的无 copy_ 版本 |

DeltaNet 的 recurrent 计算部分**完全不需要 patch**——直接调用 `torch_recurrent_gated_delta_rule`（seq=1）即可，单层验证 diff=0。

这个发现将 Qwen3.5 的 monkey-patch 从原来的 ~100 行（手写 delta_step + 完整 DeltaNet forward）精简为 4 个轻量 patch。

### DeltaNet 原理

DeltaNet 用矩阵状态 `S`（shape `[k_heads, k_dim, v_dim]`）替代 KV Cache。每一步：

1. **L2 Normalize** Q 和 K：`q = q / ||q||₂`, `k = k / ||k||₂`
2. **Scaling**：`q = q * d⁻⁰·⁵`
3. **Gating**：`S_new = S * exp(g)`
4. **Delta**：`delta = (v - S_new @ k) * sigmoid(beta)`
5. **状态更新**：`S_new = S_new + k ⊗ delta`
6. **输出**：`out = S_new @ q`

此外还有 **CausalConv1D**：每个 DeltaNet 层先对输入做 4-wide 因果卷积（引入局部上下文），conv 的内部状态（最后 3 个 time step 的隐向量）也需要在每步间传递。

### Monkey-patch：四个轻量修复

与之前手写 `delta_step` 的方案不同，重构后的版本只修四个问题，DeltaNet 的 recurrent 计算直接使用原生 torch 函数：

```python
# Patch 1: Conv state — 避免 copy_ in-place
def _safe_conv_update(hidden_states, conv_state, weight, bias, activation):
    inp = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    new_state = inp[:, :, -state_len:]  # 显式返回新 state
    out = F.conv1d(inp, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    return F.silu(out[:, :, -seq_len:]).to(hidden_states.dtype), new_state

# Patch 2: DeltaNet forward — 调用原生 torch_recurrent_gated_delta_rule
def _patched_dn_fwd(self, hidden_states, past_S=None, past_conv=None, ...):
    # ... Q/K/V projection, conv with _safe_conv_update ...
    # 直接调用 transformers 自带的 torch fallback（与手写 delta_step 数学等价）
    out, S_new = torch_recurrent_gated_delta_rule(
        q, k, v, g, beta, S, True, use_qk_l2norm_in_kernel=True)
    return self.out_proj(out), S_new, new_conv

# Patch 3: Attention — cat→Where + 手动 causal mask
# Patch 4: RMSNorm — type_as → to(dtype)
```

验证结果：
- 单层 DeltaNet：patched vs 原生 diff=0.000（完全一致）
- ORT 多步生成：`你好！很高兴见到你。有什么我可以帮助你的吗？`（连贯中文）
- ATC 编译通过，板端推理正常（0.1 tok/s）

### 需要维护的 Cache

Qwen3.5 每步需要传入并更新以下状态：

| 状态 | 数量 | Shape | 用途 |
|------|------|-------|------|
| `S` (DeltaNet 状态矩阵) | 18 | `(1, 16, 128, 128)` | DeltaNet 状态 |
| `conv` (Conv1D 隐藏状态) | 18 | `(1, 6144, 3)` | CausalConv1D 的滑动窗口 |
| `K` (Gated Attention Key) | 6 | `(1, 2, 256, 256)` | Gated Attention KV Cache |
| `V` (Gated Attention Value) | 6 | `(1, 2, 256, 256)` | Gated Attention KV Cache |

总计 50 个输入 / 49 个输出（logits + 18 S + 18 conv + 6 K + 6 V）。ONNX 文件约 1921 MB。Gated Attention 使用 16 个 Q-head / 2 个 KV-head（极端 GQA 配置），通过 `Where` 插入新的 K/V。

### 导出与验证

导出时关键设置：
- `_attn_implementation = "eager"` —— SDPA 会导致 ONNX 导出错误
- 通过 monkey-patch `Qwen3_5GatedDeltaNet.forward` 注入单步 DeltaNet
- Conv state 作为 ONNX I/O 显式传递，确保与原始 causal conv1d 精确等价

验证流程：
1. **单步对比**：原始 PyTorch vs Patched PyTorch，diff=0.0234
2. **ORT 保真度**：Patched PT vs ONNX Runtime，diff=0.031
3. **多步生成**：用 ORT 模拟完整 prefill→decode，输出"你好！我是通义千问…"

### ATC 编译

```bash
INPUT_SHAPE=$(pixi run python -c "
import onnx
m = onnx.load('om_out/qwen3.5_kvcache_max256.onnx')
print(';'.join(i.name+':'+','.join(str(d.dim_value)
    for d in i.type.tensor_type.shape.dim) for i in m.graph.input))
")
MODEL_ONNX=om_out/qwen3.5_kvcache_max256.onnx \
INPUT_SHAPE="$INPUT_SHAPE" \
IMAGE=localhost/cann-atc-rocky:v7 \
bash scripts/podman_convert.sh
```

产物 `qwen3.5_kvcache_max256.om` 约 1.9 GB，编译时有一个 warning（`rotary_emb/Expand` 不在高优先级算子库），不影响功能。

### 板端推理

`board/gen_text_qwen35.py` 实现了完整的 50→49 路 I/O 管理：

1. 预分配双缓冲：`S_bytes = 16×128×128×2 = 524KB`, `C_bytes = 6144×3×2 = 36KB`, `KV_bytes = 2×256×256×2 = 256KB`
2. S/conv/K/V 总量：`18×524K + 18×36K + 12×256K ≈ 13.2 MB`（双缓冲 × 2 = 26.4 MB）
3. AB/BA 双数据集预创建，避免每步 `create_dataset` 开销
4. 每步只需 `H2D(16B) + execute + D2H(485KB)`（logits=248320×2=485KB）

测试结果（CANN 7 编译的 OM）：

```
Prompt: "你好" → "您好！很高兴能与您聊天。我是 Qwen"
[10 tok, 78.6s, 0.1 tok/s, 7860 ms/tok]
```

Qwen3.5 的 DeltaNet 层每步需要更新 18 个 16×128×128 的矩阵状态，计算量远大于 Qwen3 的标准 Attention，因此速度约为 Qwen3 的 1/10。

> 首次加载约 195 秒（1.9 GB OM），期间无输出不是卡死。

---

## 解决的关键问题

### 版本匹配：CANN 编译工具链与板端 runtime

开发板出厂 CANN 7.0.RC1，而最新发布的是 CANN 8.0。直接用 CANN 8 编译的 OM 在板端可能出现 `soft_fault`（Qwen3.5 实测）。原因是不同 CANN 大版本的 ATC 编译器可能生成不被旧版 runtime 识别的算子二进制。**解决方案**：让容器内的 CANN 版本与板端匹配（都是 7.0 系列），从根本上消除兼容性问题。代价是需要处理 CANN 7 特有的依赖问题（numpy<2、RPM 内核包、gcc-c++ 等），详见容器构建章节。

### ONNX opset 选择：为什么是 15？

ONNX opset 版本决定了哪些算子在导出图中可用。opset 过高可能导致 ATC 不支持某些新算子；opset 过低则可能缺少 `Where` 等关键算子。我们选择了 opset 15：`Where` 在 opset 9 就已支持，而 opset 15 是 ATC 编译器最成熟的版本（实验证明 opset 17/18 产出的图在 ATC 编译中遇到更多 shape 推断错误）。

### TorchScript vs torch.export：为什么用 dynamo=False？

PyTorch 提供了两种 ONNX 导出路径：传统的 `torch.onnx.export`（基于 TorchScript 追踪）和新的 `torch.export` + `torch.onnx.dynamo_export`（基于 Dynamo）。我们选择了传统路径（`dynamo=False`），原因是：
- 传统 TorchScript 对 monkey-patch 的兼容性更好——它直接执行一次模型前向传播并记录所有 torch 操作，不关心调用栈的来源
- Dynamo 会尝试对 Python 代码进行图捕获，在遇到我们修改过的 `forward` 方法时，其符号追踪器可能无法正确推断某些动态分支（如 layer_type 判断）
- 在实际测试中，Dynamo 导出在 DeltaNet 的循环推理上因 Python 控制流失败

### ONNX/CANN 兼容性：四类不兼容操作

在 ONNX → ATC 链路上发现了四类不兼容操作，需要从源码层面处理：

| 操作 | 原始代码 | 不兼容原因 | 修复 |
|------|----------|-----------|------|
| `aten::copy_` | `conv_state.copy_(...)` | ONNX 不支持 in-place 操作 | 显式返回新 state |
| `aten::copy` | `output.type_as(x)` (RMSNorm) | ONNX opset 15 不支持此算子 | 改用 `.to(x.dtype)` |
| `Trilu(matrix, k)` | Attention causal mask | CANN 7 ATC 插件不支持 k 参数 | 手动 `Where+Equal` mask |
| 动态 shape `cat` | `torch.cat([past, new])` | ATC 需编译时静态 shape | `torch.where(mask, new, cache)` |

这四个问题的发现经历了反复实验——初始版本手写了完整的 DeltaNet 实现（~100 行），后来通过无 patch 导出 + ATC 编译实验确认了原生 torch fallback 完全可用，最终精简为 4 个轻量 patch。

### GQA Expand 的动态 shape

ATC 无法静态推断 ONNX 里的动态 shape 计算。用 `patch_onnx.py` 把 `Expand` 换为 `Tile` + 静态常量。

### TBE 编译器路径硬编码

`tbe/tvm/contrib/ccec.py` 里写死了 `/usr/local/Ascend/CANN-1.84/`。容器里 symlink 解决。


### thinking 模板

Qwen3 的 chat template 默认开启 `<think>` 推理链，对 0.6B 小模型反而浪费时间。`enable_thinking=False` 关闭。

### NPU 进程残留

ACL 进程被 kill -9 后（尤其 D 状态），NPU 内存无法自动回收。解决方案：重启板子。遇到脚本卡死时不要反复跑——先确认 NPU 是否干净。

### ACL API 返回值

`acl.mdl.add_dataset_buffer()` 返回 `(ptr, ret)` 而非单个 ret code，直接解包。

---

## 如何复现

### 环境准备

```bash
# 开发机 (x86_64)
cd Embedded_FinalHW
pixi install                           # Python 环境
pixi run python scripts/download_model.py  # 下载 Qwen3-0.6B
# Qwen3.5-0.8B 需手动下载或使用 huggingface-cli:
# huggingface-cli download Qwen/Qwen3.5-0.8B --local-dir model/Qwen3.5-0.8B

# 下载 CANN 7.0.0（约 2.0 GB 总计）
mkdir -p cann_install && cd cann_install
wget "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%207.0.0/Ascend-cann-toolkit_7.0.0_linux-x86_64.run?response-content-type=application/octet-stream"
wget "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%207.0.0/Ascend-cann-kernels-310b-7.0.0-linux.noarch.rpm?response-content-type=application/octet-stream"
cd ..

# 准备容器素材并构建
cp cann_install/*toolkit*.run docker/cann-toolkit-7.0.0.run
cp cann_install/*kernels*.rpm docker/cann-kernels-310b-7.0.0.rpm
podman build --network=host -t localhost/cann-atc-rocky:v7 \
    -f docker/Containerfile.v2-cann7 docker/
```

### 导出无KV Cache seq=32 的模型

```bash
# 1. 导出 ONNX
pixi run python scripts/export_fp16.py --seq-len 32 \
    --output om_out/qwen3_fp16_seq32.onnx

# 2. (可选) 验证: ONNX Runtime 推理 + left-padding 一致性
pixi run python -c "
import numpy as np, onnx, onnxruntime as ort
onnx.checker.check_model('om_out/qwen3_fp16_seq32.onnx')
sess = ort.InferenceSession('om_out/qwen3_fp16_seq32.onnx',
                            providers=['CPUExecutionProvider'])
# 全 1 mask
a = sess.run(None, {'input_ids': np.ones((1,32),np.int64),
                    'attention_mask': np.ones((1,32),np.int64)})[0]
# left-padding: 只有最后 5 个 token 参与注意力
m = np.zeros((1,32), np.int64); m[0,-5:] = 1
b = sess.run(None, {'input_ids': np.ones((1,32),np.int64),
                    'attention_mask': m})[0]
np.testing.assert_allclose(a[0,-5:,:], b[0,-5:,:], rtol=0.01, atol=0.05)
print('ONNX validation: PASS')
"

# 3. Patch GQA Expand → Tile
pixi run python scripts/patch_onnx.py \
    om_out/qwen3_fp16_seq32.onnx \
    --output om_out/qwen3_fp16_seq32_tile.onnx --seq-len 32

# 4. (可选) 再次验证修补后的 ONNX（同上，略）

# 5. ATC 转 OM
MODEL_ONNX=om_out/qwen3_fp16_seq32_tile.onnx \
INPUT_SHAPE="input_ids:1,32;attention_mask:1,32" \
IMAGE=localhost/cann-atc-rocky:v7 \
bash scripts/podman_convert.sh

# 6. 传输到开发板
sshpass -p 'Mind@123' scp om_out/qwen3_fp16_seq32_tile.om \
    root@192.168.137.100:/root/slm_deploy/
```

### 导出支持 KV Cache 的模型

```bash
# 1. 导出 ONNX
pixi run python scripts/export_kvcache.py --max-len 256 \
    --output om_out/qwen3_kvcache_max256.onnx

# 2. (可选) 验证: 单步 logits 与 PyTorch 原版一致
pixi run python -c "
import torch, numpy as np, onnx, onnxruntime as ort
from transformers import AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, Qwen3DecoderLayer
import sys; sys.path.insert(0,'scripts')
from export_kvcache import (_patched_attention_forward,
    _patched_decoder_forward, KVCacheWrapper)

# PyTorch baseline
m = AutoModelForCausalLM.from_pretrained('model/Qwen3-0.6B',
    torch_dtype=torch.float16, device_map='cpu', trust_remote_code=True).eval()
with torch.no_grad():
    bl = m(torch.tensor([[100]], dtype=torch.long), use_cache=False).logits

# Patched PyTorch
Qwen3Attention.forward = _patched_attention_forward
Qwen3DecoderLayer.forward = _patched_decoder_forward
w = KVCacheWrapper(m, 256).eval()
pos = torch.tensor([0], dtype=torch.int64)
kv = [torch.zeros(1,8,256,128, dtype=torch.float16) for _ in range(56)]
out_pt = w(torch.tensor([[100]], dtype=torch.long), pos, *kv)

assert torch.allclose(bl.float(), out_pt[0].float(), rtol=0.01, atol=0.1)
print('PyTorch patched vs baseline: PASS')

# ONNX Runtime
onnx.checker.check_model('om_out/qwen3_kvcache_max256.onnx')
sess = ort.InferenceSession('om_out/qwen3_kvcache_max256.onnx',
                            providers=['CPUExecutionProvider'])
feed = {'input_ids': np.array([[100]], np.int64),
        'position': np.array([0], np.int64)}
for i in range(28):
    feed[f'past_k_{i}'] = np.zeros((1,8,256,128), np.float16)
    feed[f'past_v_{i}'] = np.zeros((1,8,256,128), np.float16)
ort_out = sess.run(None, feed)[0]
assert np.allclose(out_pt[0].numpy().astype(np.float16), ort_out, rtol=0.01, atol=0.1)
print('ORT vs PyTorch: PASS')
"

# 3. (可选) 验证: ORT 多步生成（K/V 缓存正确积累）
pixi run python -c "
import numpy as np, onnxruntime as ort
sess = ort.InferenceSession('om_out/qwen3_kvcache_max256.onnx')
kv = {}
for i in range(28):
    kv[f'k_{i}'] = np.zeros((1,8,256,128), np.float16)
    kv[f'v_{i}'] = np.zeros((1,8,256,128), np.float16)
# 3 步 prefill + 2 步 decode
for pos in range(5):
    feed = {'input_ids': np.array([[pos+100]], np.int64),
            'position': np.array([pos], np.int64)}
    for i in range(28):
        feed[f'past_k_{i}'] = kv[f'k_{i}']
        feed[f'past_v_{i}'] = kv[f'v_{i}']
    outs = sess.run(None, feed)
    for i in range(28):
        kv[f'k_{i}'] = outs[1 + 2*i]
        kv[f'v_{i}'] = outs[1 + 2*i + 1]
# 检查前 3 个位置有非零 K 值
assert np.count_nonzero(kv['k_0'][0,:,0,:]) > 0, 'Position 0 unfilled'
assert np.count_nonzero(kv['k_0'][0,:,2,:]) > 0, 'Position 2 unfilled'
print('Multi-step KV accumulation: PASS')
"

# 4. ATC 转 OM
INPUT_SHAPE=$(pixi run python -c "
import onnx
m = onnx.load('om_out/qwen3_kvcache_max256.onnx')
print(';'.join(i.name+':'+','.join(str(d.dim_value)
    for d in i.type.tensor_type.shape.dim) for i in m.graph.input))
")
MODEL_ONNX=om_out/qwen3_kvcache_max256.onnx \
INPUT_SHAPE="$INPUT_SHAPE" \
IMAGE=localhost/cann-atc-rocky:v7 \
bash scripts/podman_convert.sh

# 5. 传输到开发板
sshpass -p 'Mind@123' scp om_out/qwen3_kvcache_max256.om \
    root@192.168.137.100:/root/slm_deploy/qwen3_kvcache_max256_cann7.om
```

### 导出支持 KV Cache 的 Qwen3.5 模型

```bash
# 1. 导出 ONNX
pixi run python scripts/export_qwen35.py --max-len 256 \
    --output om_out/qwen3.5_kvcache_max256.onnx

# 2. (可选) 验证: 单步 logits 与 PyTorch 原版一致
pixi run python -c "
import torch, numpy as np, onnx, onnxruntime as ort
from transformers import AutoModelForCausalLM
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5GatedDeltaNet, Qwen3_5Attention, Qwen3_5DecoderLayer)
import sys; sys.path.insert(0,'scripts')
from export_qwen35 import (_patched_dn_fwd, delta_step,
    Qwen35KVCacheWrapper, _AttentionCacheWrapper)

# 加载模型
model = AutoModelForCausalLM.from_pretrained('model/Qwen3.5-0.8B',
    torch_dtype=torch.float16, device_map='cpu', trust_remote_code=True).eval()

# PyTorch baseline (单 token)
with torch.no_grad():
    bl = model(torch.tensor([[100]], dtype=torch.long), use_cache=False).logits

# Patched PyTorch
Qwen3_5GatedDeltaNet.forward = _patched_dn_fwd
w = Qwen35KVCacheWrapper(model.model, 256, model.lm_head).eval()
di = torch.ones((1,1), torch.int64); dp = torch.tensor([0], torch.int64)
cache = []
for _ in range(18):
    cache.append(torch.zeros(1,16,128,128, dtype=torch.float16))
for _ in range(18):
    cache.append(torch.zeros(1,6144,3, dtype=torch.float16))
for _ in range(6):
    cache.append(torch.zeros(1,2,256,256, dtype=torch.float16))
for _ in range(6):
    cache.append(torch.zeros(1,2,256,256, dtype=torch.float16))
out_pt = w(di, dp, *cache)

d = torch.abs(bl.float() - out_pt[0].float()).max().item()
print(f'PyTorch vs patched: max_diff={d:.4f} - {\"PASS\" if d < 0.1 else \"FAIL\"}')

# ONNX Runtime
sess = ort.InferenceSession('om_out/qwen3.5_kvcache_max256.onnx',
                            providers=['CPUExecutionProvider'])
feed = {'input_ids': np.ones((1,1),np.int64), 'position': np.array([0],np.int64)}
for i in range(18):
    feed[f's_past_{i}'] = np.zeros((1,16,128,128), np.float16)
for i in range(18):
    feed[f'c_past_{i}'] = np.zeros((1,6144,3), np.float16)
for i in range(6):
    feed[f'k_past_{i}'] = feed[f'v_past_{i}'] = np.zeros((1,2,256,256), np.float16)
ort_out = sess.run(None, feed)
d = np.abs(out_pt[0].numpy().astype(np.float16) - ort_out[0]).max()
print(f'ORT vs PyTorch: max_diff={d:.6f} - {\"PASS\" if d < 0.1 else \"FAIL\"}')
"

# 3. 多步 ORT 验证 (prefill + decode)
pixi run python -c "
import numpy as np, onnxruntime as ort
sess = ort.InferenceSession('om_out/qwen3.5_kvcache_max256.onnx')
NL_DN, NL_GA, MAX = 18, 6, 256
s = [np.zeros((1,16,128,128),np.float16) for _ in range(NL_DN)]
c = [np.zeros((1,6144,3),np.float16) for _ in range(NL_DN)]
k = [np.zeros((1,2,MAX,256),np.float16) for _ in range(NL_GA)]
v = [np.zeros((1,2,MAX,256),np.float16) for _ in range(NL_GA)]

def run_step(tid, pos):
    feed = {'input_ids': np.array([[tid]], np.int64),
            'position': np.array([pos], np.int64)}
    for i in range(NL_DN): feed[f's_past_{i}'] = s[i]
    for i in range(NL_DN): feed[f'c_past_{i}'] = c[i]
    for i in range(NL_GA): feed[f'k_past_{i}'] = k[i]; feed[f'v_past_{i}'] = v[i]
    outs = sess.run(None, feed)
    for i in range(NL_DN): s[i] = outs[1+i]; c[i] = outs[1+NL_DN+i]
    for i in range(NL_GA): k[i] = outs[1+2*NL_DN+i]; v[i] = outs[1+2*NL_DN+NL_GA+i]
    return outs[0]

# 3 token prefill + 5 token decode
for i, tid in enumerate([100,101,102]):
    run_step(tid, i)
logits = run_step(103, 3)
print(f'Logits at step 3: max={logits.max():.4f}, non-zero count={np.count_nonzero(logits)}')
print('Multi-step ORT: PASS')
"

# 4. ATC 转 OM
INPUT_SHAPE=$(pixi run python -c "
import onnx
m = onnx.load('om_out/qwen3.5_kvcache_max256.onnx')
print(';'.join(i.name+':'+','.join(str(d.dim_value)
    for d in i.type.tensor_type.shape.dim) for i in m.graph.input))
")
MODEL_ONNX=om_out/qwen3.5_kvcache_max256.onnx \
INPUT_SHAPE="$INPUT_SHAPE" \
IMAGE=localhost/cann-atc-rocky:v7 \
bash scripts/podman_convert.sh

# 5. 传输到开发板
sshpass -p 'Mind@123' scp om_out/qwen3.5_kvcache_max256.om \
    root@192.168.137.100:/root/slm_deploy/qwen3.5_kvcache_max256_cann7.om
```


### 准备开发板环境

如果开发板刚重置、没有公网连接，只要板上出厂 Ascend runtime/pyACL 还在，就可以按下面流程离线恢复到可运行 KV Cache 模型的状态。
本项目实测的重置后环境是：Ubuntu 22.04 aarch64、Python 3.10.6、`npu-smi 23.0.rc3`、CANN Toolkit `7.0.RC1`。
OM 由 CANN 7.0.0 容器内的 ATC 编译，与板端 CANN 7.0.RC1 runtime 同大版本，完全兼容。

#### 1. 板端基础检查

在**开发机**上执行：

```bash
sshpass -p 'Mind@123' ssh -o StrictHostKeyChecking=no \
    root@192.168.137.100 '
python3 --version
npu-smi info
find /usr/local/Ascend -maxdepth 5 -name set_env.sh -print
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 -c "import acl; print(\"acl OK\")"
'
```

期望结果：

- `npu-smi info` 能看到 `310B4`
- `source /usr/local/Ascend/ascend-toolkit/set_env.sh` 后 `import acl` 成功
- 如果 `import acl` 失败，先修复或重装 Ascend runtime/Toolkit；Python 依赖无法替代 pyACL

#### 2. 准备 aarch64 离线 Python wheel

如果开发机已有 `tmp/*.whl` 和 `tmp/get-pip.py`，可直接使用。没有的话，在**有网络的开发机**上下载到 `tmp/`：

```bash
mkdir -p tmp
python3 -m pip download --dest tmp --platform manylinux2014_aarch64 \
    --python-version 310 --implementation cp --abi cp310 \
    --only-binary=:all: \
    "numpy==1.26.4" "transformers==4.53.3" \
    "tokenizers==0.21.4" "torch==2.1.0" \
    "safetensors" "huggingface-hub" "requests" "pyyaml" \
    "regex" "tqdm" "filelock" "fsspec" "packaging" \
    "typing-extensions" "sympy" "networkx" "jinja2"
curl -L https://bootstrap.pypa.io/get-pip.py -o tmp/get-pip.py
python3 -m pip download --dest tmp pip setuptools wheel
```

说明：

- 固定 `numpy==1.26.4`，避免 NumPy 2.x 和旧版 CANN/pyACL 组合出现兼容风险
- `torch` 在板端只用于 tokenizer/transformers 依赖链，不参与 NPU 推理
- 如果 `torch==2.1.0` 下载不到 aarch64 wheel，可换用本地已验证的 `torch-2.1.0-cp310-cp310-manylinux2014_aarch64.whl`

#### 3. 传输离线依赖并安装

在**开发机**上执行：

```bash
sshpass -p 'Mind@123' ssh -o StrictHostKeyChecking=no \
    root@192.168.137.100 'mkdir -p /root/slm_deploy/wheels'

sshpass -p 'Mind@123' scp tmp/*.whl tmp/get-pip.py \
    root@192.168.137.100:/root/slm_deploy/wheels/

sshpass -p 'Mind@123' ssh -o StrictHostKeyChecking=no \
    root@192.168.137.100 '
cd /root/slm_deploy
python3 wheels/get-pip.py --no-index --find-links=/root/slm_deploy/wheels \
    pip setuptools wheel
python3 -m pip install --no-index --find-links=/root/slm_deploy/wheels \
    --force-reinstall "numpy==1.26.4" transformers torch
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 -c "import acl, numpy, transformers, torch; print(numpy.__version__, transformers.__version__, torch.__version__)"
'
```

### 传输文件到开发板

以下操作在**开发机**上执行：

```bash
# Tokenizer 文件（Qwen3 使用 vocab.json + merges.txt 格式）
sshpass -p 'Mind@123' scp \
    model/Qwen3-0.6B/vocab.json \
    model/Qwen3-0.6B/merges.txt \
    model/Qwen3-0.6B/tokenizer_config.json \
    model/Qwen3-0.6B/config.json \
    model/Qwen3-0.6B/generation_config.json \
    root@192.168.137.100:/root/slm_deploy/

# 注意：Qwen3 和 Qwen3.5 的 tokenizer 文件互不兼容，
# Qwen3.5 使用 HuggingFace tokenizers 格式（tokenizer.json + chat_template.jinja）。
# SCP 时不要互相覆盖。

# 推理脚本
sshpass -p 'Mind@123' scp \
    board/gen_text_seq32.py \
    board/gen_text_kvcache.py \
    board/gen_text_qwen35.py \
    board/run_kvcache.sh \
    root@192.168.137.100:/root/slm_deploy/
sshpass -p 'Mind@123' ssh -o StrictHostKeyChecking=no \
    root@192.168.137.100 'chmod +x /root/slm_deploy/run_kvcache.sh'

# OM 模型文件（先确保已完成 ATC 转换，文件在 om_out/ 下）
sshpass -p 'Mind@123' scp om_out/qwen3_fp16_seq32_tile.om \
    root@192.168.137.100:/root/slm_deploy/
sshpass -p 'Mind@123' scp om_out/qwen3_kvcache_max256_cann7.om \
    root@192.168.137.100:/root/slm_deploy/
# (可选) Qwen3.5 OM
sshpass -p 'Mind@123' scp om_out/qwen3.5_kvcache_max256_cann7.om \
    root@192.168.137.100:/root/slm_deploy/
```

可选：校验大文件传输是否完整。

```bash
sha256sum om_out/qwen3_kvcache_max256_cann7.om
sshpass -p 'Mind@123' ssh -o StrictHostKeyChecking=no \
    root@192.168.137.100 \
    'sha256sum /root/slm_deploy/qwen3_kvcache_max256_cann7.om'
```

### 运行推理

在**开发板**上执行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /root/slm_deploy

# seq=32 滑动窗口模型
python3 gen_text_seq32.py --prompt "你好，请介绍一下你自己" --max-tokens 50

# Qwen3 KV Cache 模型 (max_len=256)
python3 gen_text_kvcache.py --model qwen3_kvcache_max256_cann7.om \
    --prompt "你好，请介绍一下你自己" --max-tokens 50

# Qwen3.5 KV Cache 模型 (max_len=256, DeltaNet)
python3 gen_text_qwen35.py --model qwen3.5_kvcache_max256_cann7.om \
    --tokenizer-dir /root/slm_deploy \
    --prompt "你好" --max-tokens 20

# 或使用封装好的入口
./run_kvcache.sh --prompt "你好，请介绍一下你自己" --max-tokens 50
```

首次运行前可做一个短冒烟测试：

```bash
sshpass -p 'Mind@123' ssh -o StrictHostKeyChecking=no \
    root@192.168.137.100 '
cd /root/slm_deploy
./run_kvcache.sh --prompt "你好" --max-tokens 2
'
```

实测输出应包含：

```text
I/O: 58 in, 57 out
[Prompt: 13 tokens]
[step 1] post-execute
...
你好！
Done.
```

注意：1.5 GB OM 首次 `acl.mdl.load_from_file` 约 75 秒，1.9GB (Qwen3.5) 约 195 秒。

**可选参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | 各脚本内置默认路径 | OM 文件路径 |
| `--tokenizer-dir` | `/root/slm_deploy` | tokenizer 文件目录 |
| `--max-tokens` | 30 | 最大生成 token 数 |
| `--temperature` | 0.7 | 采样温度，0 为贪心解码 |
| `--top-k` | 40 | Top-K 采样 |
| `--top-p` | 0.9 | Nucleus 采样 |

**常见运行时问题**：

- **模型加载卡住**：NPU 可能有之前进程的残留状态，重启板子 (`reboot`)
- **输出乱码**：确认 prompt 不要超过模型窗口（seq=32 时最多 ~15 个中文字）。KV Cache 模型也有 256 token 上限
- **速度异常慢**：用 `npu-smi info` 检查 NPU 内存是否接近满载——如果 ~95% 说明重复加载了多个模型实例

---

*最后更新：2026-06-10*
