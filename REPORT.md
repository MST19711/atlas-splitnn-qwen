# Qwen3 / Qwen3.5 在 Atlas 200I DK A2 上的部署实验报告

本文档记录了将通义千问小语言模型部署到华为昇腾 Atlas 200I DK A2 开发板的完整实验过程，包括每个阶段的设计思路、遇到的工程问题及解决方案。

---

## 目录

1. [背景知识](#背景知识)
2. [阶段一：静态窗口模型](#阶段一静态窗口模型)
3. [阶段二：KV Cache 模型](#阶段二kv-cache-模型)
4. [阶段三：Qwen3.5-0.8B DeltaNet 模型](#阶段三qwen35-08b-deltanet-模型)
5. [阶段四：SplitNN 原型设计](#阶段四splitnn-原型设计)
6. [阶段五：SplitNN 控制器与 OpenAI 接口](#阶段五splitnn-控制器与-openai-接口)
7. [解决的关键工程问题](#解决的关键工程问题)
8. [阶段六：SplitNN 通用化](#阶段六splitnn-通用化)

---

## 背景知识

### 自回归文本生成

语言模型逐个 token 地生成文本。每一步输入当前序列，输出下一个 token 的概率分布，采样后拼接到序列末尾。

### Left-padding

对于固定长度的输入（比如 32 token），如果实际序列更短，需要在左边填 0（padding），右边放真实 token。配合 `attention_mask`（0 表示忽略），causal attention 保证 padding 不影响实际 token 的计算。

### KV Cache

Transformer 每层的注意力需要计算 Key、Query、Value。没有缓存时，每一步都要重新计算所有历史 token 的 K 和 V（重复劳动）。KV Cache 把它们存下来，下一步只需算新 token 的 K/V，注意力复杂度从 O(N²) 降到 O(N)。

---

## 阶段一：静态窗口模型

> **注意**：此方案在 CANN 7.1.0.3.220 + Ascend310B4 下经验证不可用（ATC 编译产物输出与 ONNX 不匹配），相关代码（`scripts/export_qwen3_static.py`、`scripts/patch_qwen3_static_onnx.py`、`board/gen_text_qwen3_static.py`）已从仓库移除。本节仅作历史参考。

### 思路

不用 KV Cache。每次把完整序列 left-pad 到固定长度（比如 32），一次性送入模型。尽管有重复计算，但实现简单，适合快速验证全链路。

**序列长度为什么选 32？** 这是反复试探的结果。1 太小（无上下文），128 太大（NPU 内存可能不够，且每步太慢）。32 是折中——够容纳一句日常对话，也能在 ~280ms 内完成一步。

### 导出 ONNX

`scripts/export_qwen3_static.py` 把 Qwen3 封装为固定 seq_len 的静态图：

```python
dummy_input_ids = torch.ones((1, 32), dtype=torch.int64)
torch.onnx.export(wrapper, (dummy_input_ids, ...), "output.onnx",
                  opset_version=15, do_constant_folding=True)
```

关键设置：`use_cache=False`（不产生 KV Cache 输出），`dynamo=False`（用传统的 TorchScript 追踪器，兼容性更好）。

导出后用 ONNX Runtime 验证——检查 left-padding 下末尾位置 logits 与全 1 mask 一致。

### 修补 ONNX：处理 GQA 的 Expand 节点

导出后的 ONNX 图里有 56 个 `Expand` 节点——每层注意力要把 K/V 从 8 头重复到 16 头（GQA），重复因子为 `[1, 8, 2, 32, 128]` 中的那个 `2`。问题在于这个 shape 是通过 `Where + Equal + ConstantOfShape` **动态计算**的，ATC 编译器无法静态推断。

`scripts/patch_qwen3_static_onnx.py` 遍历所有 `self_attn` 中的 Expand，替换为 `Tile` 算子配合静态常量 `[1, 1, 2, 1, 1]`。`Tile` 和 `Expand` 在这个场景下语义完全相同——都是把张量沿某一维复制指定次数。

修补后用 ORT 重新验证——确保数值输出不变。

### ATC 编译

```bash
MODEL_ONNX=om_out/qwen3_fp16_seq32_tile.onnx \
INPUT_SHAPE="input_ids:1,32;attention_mask:1,32" \
bash scripts/podman_convert.sh
```

ATC（Ascend Tensor Compiler）把 ONNX 图编译为板载 NPU 可执行的 OM 文件。过程约 3 分钟，产物 1.5 GB。

### 板载推理

`board/gen_text_qwen3_static.py` 实现了 left-padding 滑动窗口的完整生成循环：

1. Tokenize 用户输入
2. 取最后 32 token，左侧填 0，构建 attention_mask
3. ACL 执行 → 取 logits 的最后一个有效位置 → 采样
4. 新 token 加入窗口，最旧的 token 滑出

每 token 约 280ms，解码速度约 3.6 tok/s，输出是连贯的中文。

---

## 阶段二：KV Cache 模型

### 思路

静态窗口方案能跑通，但扩大窗口到 256 时，每步就需要计算 256×256 的注意力矩阵，O(N²) 的增长意味着每步要等好几秒。KV Cache 解决的就是这个问题——**存储而不重复计算**。

由于这台 NPU 实际上不支持动态计算图，我们的做法是**修改 Qwen3 内部的注意力实现**（monkey-patch），让它把原本动态增长的 K/V 替换为预分配的固定大小缓冲区。

#### 为什么必须 monkey-patch？

不是因为方便——而是因为 ONNX 和 CANN 的组合约束使得其他路径走不通：

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

### Monkey-patch：把 `cat` 换成 `Where`

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

#### 为什么用 CANN 7 而不是 CANN 8？

开发板出厂预装 CANN 7.0.RC1 runtime，OM 模型文件的二进制格式需要与 runtime 版本兼容。我们最初尝试了 CANN 8.0.RC3 编译的 OM——Qwen3-0.6B 的 OM 在板端可以运行（只用了 MatMul/Where 等基础算子），但 Qwen3.5-0.8B 的 OM 加载时出现 `drv_soft_fault (err_type=0xa)`，NPU 驱动直接拒绝执行。这是因为 Qwen3.5 的 DeltaNet 算子（矩阵乘法 + l2norm + sigmoid 等的组合）在 CANN 8 编译后的二进制格式不被 CANN 7 runtime 识别。与其升级板端固件（风险高），不如将编译工具链降级到与板端匹配的 CANN 7 版本——这就是 `Containerfile.v2-cann7` 的来由。

#### 为什么用 Rocky Linux 9（RPM 系）而不是 Ubuntu？

CANN 7.0.0 for x86_64 的内核包只提供了 `.rpm` 格式的下载。用 RPM 系发行版可以直接 `rpm -ivh` 安装，避免了 `alien` 转包的兼容性风险。实际选用的容器基镜像是 Rocky Linux 9（Python 3.9，符合 CANN 7 的要求 `>= 3.7`）。

#### 关键发现

CANN 的 ATC 对不同芯片需要安装对应的"内核包"。310P 包只覆盖 P1/P3 型号，而开发板是 310B4 芯片，**必须用 310B 内核包**。用错包的症状是 ATC 编译能过，但开发板加载 OM 时返回 `ret=500002`。

另一个重要的修复：CANN 的 TBE Python 代码里硬编码了编译器路径 `/usr/local/Ascend/CANN-1.84/`。我们在容器里创建一个符号链接指向实际安装位置即可解决。

此外，CANN 7 的 TBE 编译器需要 pip 安装 `attrs cloudpickle psutil synr tornado`，并且需要 `numpy<2`（与 numpy 2.x 不兼容），同时需要 `gcc-c++` 提供 C++ 标准库头文件（CCE 编译器依赖 `<type_traits>`）。

### 板载推理

与 seq=32 方案不同，KV Cache 需要在 NPU 上维护一套持续的 K/V 缓冲区：

1. `acl.rt.malloc` 预分配 56 个 K/V tensor（每个 512KB，共约 28MB）
2. Prefill 阶段：逐 token 输入，K/V 逐步填充
3. Decode 阶段：每步从输出取回 logits 采样，同时把更新后的 K/V 复制回输入缓冲区

### 性能优化历程

开发板推理经历了三轮优化：

#### 初始版本（每步 alloc + memcpy，ACL 框架开销严重）

```
每步 execute():
  acl.rt.malloc × 115      ← 为 58 个输入 + 57 个输出分配 buffer
  acl.rt.memcpy × 112      ← K/V 在 host↔device 间全量搬运（28MB）
  acl.mdl.execute()         ← 174ms NPU 计算
  acl.rt.free × 115         ← 释放

每步约 420ms
```

Profiler 数据（`msprof --ascendcl=on --task-time=on --ai-core=on`）：

| 类别 | 耗时 | 占比 |
|------|------|------|
| BatchMatMulV2 (NPU 计算) | 174ms | 41% |
| Host 侧 alloc + memcpy | ~200ms | 48% |
| Python / ACL 框架 | ~46ms | 11% |

#### 第一轮优化：预分配 Device 缓冲区 + K/V 指针轮转（双缓冲）

消除每步的 `acl.rt.malloc`/`acl.rt.free` 和 112 次 K/V memcpy：

```
每步只剩: H2D(16B) + execute + D2H(303KB)
K/V 留在 device 内存，指针身份互换
```

#### 第二轮优化：预创建 AB/BA 两组 Dataset

消除每步的 `acl.mdl.create_dataset`/`destroy_dataset`/`create_data_buffer`/`add_dataset_buffer`。

**为什么需要两组 dataset？** 每步 ACL 执行后，输出 K/V 指针指向缓冲区 B，下一步需要把它作为输入——如果只有一组 dataset，就需要用 `acl.update_data_buffer` 来换绑（实验中发现这个 API 在 CANN 7.0.RC1 上有稳定性问题）。两组 dataset 各绑定不同的 K/V 指针组，轮流使用——这一步的输出指针恰好是下一步的输入指针，完全避免了 buffer 的动态换绑。

```
每步只剩: H2D(16B) + execute + D2H(303KB)
（全部数据结构在模型生命周期内一次性创建）
```

最终每步约 400ms（Profiler 显示 NPU 纯计算 174ms，剩余 ~226ms 为 ACL 框架调用和 Python 采样开销，受限于 CANN 7.0.RC1 runtime 与 Python 调用开销）。

> 注意：1.5 GB OM 在 CANN 7.0.RC1 runtime 上首次加载需要约 75 秒，期间无输出不是卡死。

---

## 阶段三：Qwen3.5-0.8B DeltaNet 模型

### 架构对比

Qwen3.5-0.8B 是 Qwen3 的新一代模型，参数量稍大（~800M），使用了全新的混合架构：

| 组件 | Qwen3-0.6B | Qwen3.5-0.8B |
|------|-----------|-------------|
| 总层数 | 28 | 24 |
| DeltaNet 层 | 0 | 18 |
| Gated Attention 层 | 28 | 6 |
| 词表大小 | 151936 | 248320 |
| 参数量 | ~600M | ~800M |
| Tokenizer | vocab.json + merges.txt | tokenizer.json |
| 权重大小 | 1.5 GB | 1.9 GB |
| OM 大小 | 1.5 GB | 1.9 GB |

Qwen3.5 的核心创新是**混合架构**：18 层使用 DeltaNet（一种线性注意力变体，O(1) 内存复杂度）处理长距离依赖，6 层使用标准 Gated Attention 捕捉局部模式。

### 设计挑战：四类 ONNX/CANN 不兼容操作

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

### Monkey-patch：四个轻量修复

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
    out, S_new = torch_recurrent_gated_delta_rule(
        q, k, v, g, beta, S, True, use_qk_l2norm_in_kernel=True)
    return self.out_proj(out), S_new, new_conv

# Patch 3: Attention — cat→Where + 手动 Where+Equal causal mask
# Patch 4: RMSNorm — type_as → to(dtype)
```

### 需要维护的 Cache

Qwen3.5 每步需要传入并更新以下状态：

| 状态 | 数量 | Shape | 用途 |
|------|------|-------|------|
| `S` (DeltaNet 状态矩阵) | 18 | `(1, 16, 128, 128)` | DeltaNet 线性注意力状态 |
| `conv` (Conv1D 隐藏状态) | 18 | `(1, 6144, 3)` | CausalConv1D 的滑动窗口 |
| `K` (Gated Attention Key) | 6 | `(1, 2, 256, 256)` | Gated Attention KV Cache |
| `V` (Gated Attention Value) | 6 | `(1, 2, 256, 256)` | Gated Attention KV Cache |

总计 50 个输入 / 49 个输出（logits + 18 S + 18 conv + 6 K + 6 V）。

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
# 注意：INPUT_SHAPE 值包含 50 个分号，必须先 export 再运行脚本，不能内联展开
INPUT_SHAPE=$(pixi run python scripts/gen_input_shape.py om_out/qwen3.5_kvcache_max256.onnx)
export INPUT_SHAPE MODEL_ONNX="om_out/qwen3.5_kvcache_max256.onnx"
bash scripts/podman_convert.sh
```

产物 `qwen3.5_kvcache_max256.om` 约 1.9 GB。有一个 warning（`rotary_emb/Expand` 不在高优先级算子库），不影响功能。

### 板端推理

`board/gen_text_qwen35_kvcache.py` 实现了完整的 50→49 路 I/O 管理：

1. 预分配双缓冲：`S_bytes = 16×128×128×2 = 524KB`, `C_bytes = 6144×3×2 = 36KB`, `KV_bytes = 2×256×256×2 = 256KB`
2. S/conv/K/V 总量：`18×524K + 18×36K + 12×256K ≈ 13.2 MB`（双缓冲 × 2 = 26.4 MB）
3. AB/BA 双数据集预创建，避免每步 `create_dataset` 开销
4. 每步只需 `H2D(16B) + execute + D2H(485KB)`（logits=248320×2=485KB）

测试结果（CANN 7 编译的 OM）：

```text
Prompt: "你好" → "您好！很高兴能与您聊天。我是 Qwen"
[prefill 52.3s, decode 5 tok in 1.4s, 3.7 tok/s, 274 ms/tok]
```

Qwen3.5 的纯解码速度与 Qwen3 接近（~3.7 tok/s），但 prefill 耗时远大于 Qwen3（52.3s vs 5.6s）——这是因为 DeltaNet 层每步需要更新 18 个 16×128×128 的稠密矩阵状态，而 Qwen3 的 Attention 只需要写一行稀疏 K/V。

### 上下文窗口测试

Qwen3.5 使用混合架构，18 个 DeltaNet 层的状态与上下文长度无关（O(1)），只有 6 个 GA 注意力层受 O(L²) 影响。这意味着在静态图下**增加上下文长度几乎不增加 NPU 内存开销**：

| max_len | 加载 | NPU 占用 | decode | prefill (13 tok) | 状态 |
|---------|------|---------|--------|-----------------|------|
| 256 | 185s | 95% | 4.1 tok/s | 48s | ✓ 可用 |
| 1024 | 191s | 97% | 3.6 tok/s | 67s | ✓ RAG 推荐 |
| 2048 | 214s | 96% | 3.3 tok/s | 65s | ✓ |
| 4096 | 199s | 96% | 2.7 tok/s | 63s | ✓ |
| 8192 | 210s | 96% | 2.0 tok/s | 70s | ✓ |
| 16384 | 212s | 96% | 0.8 tok/s | 117s | ✓ 勉强可用 |
| 32768 | 216s | 96% | — | 超时 | 加载成功，极慢 |

**关键发现**：NPU 内存始终在 95-97%，上下文长度对内存影响可忽略（K/V 缓存从 256 的 6MB 增长到 32768 的 200MB，相比模型本体 2.7GB 微不足道）。性能退化全部来自 GA 层的 O(L²) 注意力计算。

对于轻量 RAG 场景，推荐 **1024**——检索文档 + 用户提问 + 回复绰绰有余，且 decode 速度保持 3.6 tok/s。

---

## 阶段四：SplitNN 原型设计

在 KV Cache 版本的 Qwen3.5 已经可以单机运行之后，下一步自然的问题是：能不能把模型切成几段，让开发板只负责少量前后层，而把最重的中间计算交给主机完成？这就是 SplitNN 原型的起点。

### 设计目标

SplitNN 原型阶段的目标并不是一开始就做 OpenAI 服务，而是先回答三个更基础的问题：

1. **模型能否沿层边界被正确切开？**
2. **切开后 cache 语义是否仍然正确？**
3. **两端通过网络传 hidden state，是否还能稳定完成自回归生成？**

只有这三个问题都成立，控制器服务化才有意义。

### 为什么选 4 / 16 / 4

Qwen3.5-0.8B 一共 24 层，层类型按 4 层为一个周期反复出现：

```text
3 x linear_attention + 1 x full_attention
```

因此 `4 / 16 / 4` 有两个直接好处：

1. **切分点天然落在周期边界**
   - 不会把一组 cache 语义“切半”
   - 前段、中段、后段都保持完整层组

2. **工程实现最干净**
   - 前段：`layers[0:4]`
   - 中段：`layers[4:20]`
   - 后段：`layers[20:24]`

这种切法既兼顾结构对齐，也兼顾板端负载下降。

### 三段职责

#### 1. 前段（prefix）

负责：
- `embed_tokens`
- `layers[0:4]`
- 输出 `hidden_state_l4`

输入是：
- `token_id`
- `position`
- 前段自己的 cache

输出是固定 shape：

```text
(1, 1, 1024), fp16
```

#### 2. 中段（middle）

负责：
- `layers[4:20]`
- 输出 `hidden_state_l20`

输入是：
- `hidden_state_l4`
- `position`
- 中段 cache

中段被设计为独立的远端服务 `server/qwen35_split_service.py`，按 `session_id` 保存其 16 层内部 cache。

#### 3. 后段（suffix）

负责：
- `layers[20:24]`
- `final norm`
- `lm_head`
- 输出 `logits`

输入是：
- `hidden_state_l20`
- `position`
- 后段自己的 cache

### 为什么网络只传 hidden state

理论上也可以把 cache 在网络上传来传去，但工程上几乎不可接受：

1. **体积太大**  
   cache 大小远大于单步 hidden state，网络代价会迅速膨胀。

2. **切分点强绑定**  
   cache 的 layout 与具体层类型绑定很深，不适合作为跨进程/跨设备通用接口。

3. **接口会变脆弱**  
   一旦模型切分变化，整个网络协议都要变。

因此原型阶段明确采用：

- 网络上传输：`hidden_state_l4` / `hidden_state_l20`
- 本地保留：prefix/suffix cache
- 远端保留：middle cache

这样每 token 只需传输一个 `(1,1,1024)` 的 FP16 hidden state，往返大约 4 KB。

### 原型验证路径

SplitNN 原型不是一步到位跑在开发板上的，而是按三层验证逐步收敛：

#### 第一层：纯 PyTorch reference

先在开发机上验证：

```text
prefix -> middle -> suffix
```

与完整 `Qwen35KVCacheWrapper` 的逐 token logits 是否一致。

这是为了证明：
- 切分边界没有问题
- cache 更新逻辑没有问题

#### 第二层：prefix/suffix ONNX 多步验证

再把前后段导出成 ONNX，用 ORT 验证：
- prefix hidden 是否与 PyTorch 对齐
- suffix logits 是否与 PyTorch 对齐
- 多步 prefill + decode 时 cache 是否正常累积

这是为了证明：
- 前后段真的可以脱离完整模型独立执行

#### 第三层：本地模拟联调

最后再做：

```text
ONNX prefix/suffix + middle server
```

通过真实 HTTP 协议串起来，输入 prompt 后实际生成并解码文本。

这一步最重要，因为它回答了最关键的系统问题：

> 这不只是几段子图能跑，而是整个 SplitNN 链路能真实完成自回归文本生成。

### 原型结论

到这个阶段，已经可以确认：

1. `4 / 16 / 4` 切分在数值和 cache 语义上成立
2. ONNX 前后段可以独立工作
3. middle server 的会话协议可以支撑逐 token 生成
4. SplitNN 已经从“概念想法”变成“可运行原型”

也正因为这个原型已经成立，后续才有必要继续建设统一控制器和 OpenAI 兼容接口。

---

## 阶段五：SplitNN 控制器与 OpenAI 接口

在完成 `4 / 16 / 4` SplitNN 原型之后，系统仍有一个明显缺口：主机仿真和开发板部署分别依赖不同的入口脚本，OpenAI 风格的上层调用方式也还没有统一。因此引入了一个新的“控制器中间层”。

### 设计目标

控制器层的目标有三个：

1. **对外统一接口**  
   提供 OpenAI 兼容的 `/v1/chat/completions`，让主机仿真和开发板部署都使用同一套调用方式。

2. **对内统一编排**  
   把 tokenizer、chat template、采样、生成循环、session 生命周期统一放到控制器中，而不是散落在板端脚本和仿真脚本里。

3. **统一前后段引擎抽象**  
   前后段推理通过统一接口切换：
   - 开发机：`OnnxSplitEngine`
   - 开发板：`OmSplitEngine`

### 架构划分

控制器被拆成四层：

#### 1. API 层：`controller/openai_split_controller.py`

负责：
- `GET /healthz`
- `GET /v1/models`
- `POST /v1/chat/completions`

首版支持：
- 非流式响应
- `stream=true` 的 SSE 流式响应

#### 2. 编排层：`controller/orchestrator.py`

负责：
- 读取 `messages`
- `apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)`
- 完整 prompt 重新 prefill
- decode 循环
- `temperature` / `top_k` 采样
- `stop` / `eos` / `max_tokens` / `max_len` 终止逻辑

这层是整个系统的“控制器核心”。

#### 3. 引擎层：`controller/engine/`

定义统一接口：
- `load()`
- `close()`
- `start_session()`
- `end_session()`
- `run_prefix(token_id, position)`
- `run_suffix(hidden_state, position)`

这意味着控制器本身不再关心 ACL、ORT、双缓冲、dataset 或 feed cache 的细节，只依赖统一的前后段执行语义。

#### 4. 远端中段层：`controller/remote_middle.py`

负责与现有的 middle server 协议对接：
- `health`
- `open`
- `step`
- `close`

控制器不管理中段 cache；中段 cache 仍由 server 端按 `session_id` 维护。

### Cache 归属

这是控制器设计中最重要的边界之一：

- **前段 cache / 后段 cache**：由本地引擎实例内部管理  
  - `OnnxSplitEngine` 持有 ORT feed cache
  - `OmSplitEngine` 持有 ACL/OM 双缓冲 cache

- **中段 cache**：由远端 middle server 管理

因此控制器只“间接管理” cache 生命周期，但并不直接操作张量细节。

### 为什么首版采用“无状态多轮”

控制器首版故意没有做“跨请求 cache 复用”，而是采用无状态多轮：

- 每次 `/v1/chat/completions` 请求都基于完整 `messages` 重新 prefill
- 不在请求之间保留 prefix/suffix/middle cache

这样做的原因：

1. 与 OpenAI API 的无状态使用方式一致  
2. 控制器不需要额外管理跨请求 session 存活、并发冲突和 cache 泄漏  
3. 先把“正确生成”与“统一接口”打通，再考虑低时延增量会话

### ONNX 后端验证

控制器的第一条完整验证链路是：

```text
OpenAI Client
  -> OpenAI Controller
  -> OnnxSplitEngine (prefix/suffix)
  -> Remote Middle Server
  -> 返回 OpenAI 响应
```

已完成的验证包括：

1. `GET /healthz`
2. `GET /v1/models`
3. `POST /v1/chat/completions` 非流式
4. `POST /v1/chat/completions` 流式 SSE
5. 多轮 `messages` 输入

### 实际验证结果

开发机本地联调中，控制器已经可以返回正常中文文本。例如提示：

```text
你好，请用一句话介绍一下你自己。
```

非流式返回示例：

```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "我是 Qwen3.5，由通义实验室自主研发的超大规模语言模型，具备强大的逻辑推理、代码"
      }
    }
  ]
}
```

流式模式下，控制器按 OpenAI 风格逐 chunk 返回：

```text
data: {"choices":[{"delta":{"role":"assistant","content":""}}]}
data: {"choices":[{"delta":{"content":"我是"}}]}
data: {"choices":[{"delta":{"content":" Q"}}]}
...
data: [DONE]
```

这说明：

- SplitNN 前后段 ONNX 推理已经可以嵌入统一控制器
- middle server 协议可以稳定支撑 OpenAI 风格生成
- 系统已经从“验证脚本集合”进化为“可服务化调用的推理系统”

### CUDA 主机兼容性修复

在 SplitNN 中段切到 CUDA 主机之后，实际又遇到一个新的工程问题：虽然机器上有 NVIDIA GPU，`nvidia-smi` 也正常，但最初通过 `pixi` 安装的 PyTorch 版本并不能正确识别这张较新的显卡，导致 `torch.cuda.is_available()` 为假，middle server 启动时报 `CUDA unavailable`。

根因是环境里的 CUDA/PyTorch 组合对新架构显卡支持不完整。最终做法是：

1. 移除原先 conda 侧的 `pytorch` 依赖
2. 在 `pixi.toml` 中改为使用 PyPI 官方 `cu128` wheel
3. 同步补回 `accelerate` 等依赖

修复后，主机端已经可以稳定运行：

```bash
pixi run python server/qwen35_split_service.py \
    --host 0.0.0.0 --port 18080 \
    --model-path model/Qwen3.5-0.8B \
    --device cuda:0 --max-len 16384
```

### 16K 长上下文扩展

在确认 CUDA 中段可用、并且 SplitNN 已经能够降低板端内存占用之后，下一步就是把 SplitNN 的上下文从最初的 `256` 扩展到 `16384`。

这一步主要涉及三处：

1. **中段 server 扩容**
   - `server/qwen35_split_service.py` 支持 `--max-len 16384`
   - `scripts/qwen35_split_common.py` 的默认 `MAX_LEN` 扩展到 `16384`

2. **前后段 ONNX 重新导出**
   - `qwen3.5_split_prefix_max16384.onnx`
   - `qwen3.5_split_suffix_max16384.onnx`

3. **板端 OM 重新编译**
   - `qwen3.5_split_prefix_max16384.om`
   - `qwen3.5_split_suffix_max16384.om`

在开发机上，`max16384` 的 prefix/suffix ONNX 已经通过 ORT 校验：

- prefix PyTorch vs ORT：`max_diff = 0.000488`
- suffix PyTorch vs ORT：`max_diff = 0.000000`

说明长上下文版本的前后段导出没有引入新的数值问题。

### CUDA 中段 16K 实测速度

为避免把“长上下文可加载”误当作“长上下文可用”，还单独对 middle server 做了 16K 配置下的实际测速。

单 token 中段前向结果：

- `avg_rtt_ms = 111.115`
- `avg_server_ms = 108.952`
- `tok_per_s_rtt = 9.000`
- `tok_per_s_server = 9.178`

对照 `max_len=256` 的同类测试，吞吐几乎没有明显下降。原因是 middle server 的单步执行本来就是增量 cache 递推；上下文上限主要体现在可容纳的 cache 尺寸，而不是每步都重新跑全长序列。

### 开发板 16K OM 实机联调

更关键的一步，是验证“真正的板端 SplitNN”是否也能吃下长上下文，而不仅仅是开发机 ORT 仿真。

实际联调链路如下：

```text
Board OM prefix/suffix
  -> OpenAI Split Controller (OM backend)
  -> HTTP reverse tunnel
  -> CUDA middle server
  -> 返回 logits 并在板端解码
```

完成了以下实测：

1. 启动 `max_len=16384` 的 CUDA middle server
2. 在开发机到板端之间建立反向 SSH 隧道
3. 在板端启动 `run_openai_split_controller_om_16k.sh`
4. 调用 `/healthz`，确认板端控制器和主机 middle server 都处于可用状态
5. 构造一个**明显超过 256 token** 的长 prompt（约 700 个 `hello` 组成），再追加中文指令“请只回复：收到。”

最终板端实际返回：

```text
收到。
```

这一步的意义很直接：

- 不只是 `max16384.om` 能加载
- 不只是 host middle server 能处理更大的 cache
- 而是**真实的板端 OM 前后段 + 远端 CUDA 中段**已经能够在超过 256 token 的 prompt 上完成完整 prefill、decode 和文本解码

### 当前状态总结

到这里，SplitNN 相关工作已经形成了一个完整闭环：

1. `4 / 16 / 4` 切分在 reference、ORT 和本地模拟层面成立
2. OpenAI 控制器已经把 ONNX / OM 两种前后段引擎统一起来
3. CUDA middle server 已能稳定工作，并支持 `16384` 上下文
4. 开发板上的 OM SplitNN 也已经完成真实 16K 长 prompt 联调

剩余的主要问题不再是“能不能跑通”，而是：
- 板端 SSE 流式收尾还需继续排查
- 后续若要支持低时延多轮，需要设计跨请求 cache 复用协议

### 当前局限

这版控制器仍有几个明确边界：

1. **只支持文本**
2. **只支持 `Qwen3.5 split 4/16/4`**
3. **只支持无状态多轮**
4. **开发板 OM 后端的非流式路径已经实机验证通过，但 SSE 流式收尾仍待继续排查**

不过从工程阶段来看，至此 SplitNN 已经进入“部署联调”而不再只是“算子验证”阶段。

---

## 阶段六：SplitNN 通用化

SplitNN 体系在 `4 / 16 / 4` 原型验证成功之后，下一步自然是将硬编码的参数化，使其能适配不同模型尺寸、不同上下文长度和不同切分方案。

### 设计目标

1. **支持任意 Qwen3.5 模型尺寸**（0.8B / 2B / 4B / 9B / 27B）
2. **支持自定义切分方案**（`--split prefix_end,suffix_start`）
3. **支持自定义上下文长度**（`--max-len`）
4. **支持 thinking 开关**（`enable_thinking`）

### 核心抽象

#### ModelSpec

从 `config.json` 动态读取所有架构参数，替代硬编码常量：

```python
@dataclass
class ModelSpec:
    hidden_size: int          # 1024 / 2048 / 2560 / 4096 / 5120
    vocab_size: int           # 248320
    num_hidden_layers: int    # 24 / 24 / 32 / 32 / 64
    num_key_value_heads: int  # 2 / 2 / 4 / 4 / 4
    head_dim: int             # 256 (所有尺寸)
    linear_num_key_heads: int # 16 (所有尺寸)
    linear_num_value_heads: int  # 16 / 16 / 32 / 32 / 48
    linear_key_head_dim: int  # 128
    linear_value_head_dim: int # 128
    layer_types: list[str]    # 从 full_attention_interval=4 自动生成
```

关键派生量：
- `conv_dim`: `K_H × K_DIM × 2 + V_H × V_DIM`（随模型尺寸变化：6144 / 6144 / 8192 / 8192 / 10240）
- `compute_segment(start, end) → (nl_dn, nl_ga)`: 统计区间内 DN/GA 层数

#### SplitConfig

```python
@dataclass
class SplitConfig:
    prefix_end: int     # prefix 层范围 [0, prefix_end)
    suffix_start: int   # suffix 层范围 [suffix_start, total)
```

从切分点自动计算各段 nl_dn/nl_ga。

#### 零依赖设计

为使板端脚本能导入 `ModelSpec`/`SplitConfig`（板端无 PyTorch），将这些数据结构拆分到独立的 `scripts/qwen35_model_spec.py`，零外部依赖。x86 侧的 `scripts/qwen35_split_common.py` 导入它们并补充 torch 相关逻辑。

### 4B 模型的适配修复

Qwen3.5-4B（32 层，hidden_size=2560）在 SplitNN 导出的首个测试中暴露了两个与 0.8B 不同的架构特征：

#### 1. DeltaNet K/V head 不匹配

Qwen3.5-4B 的 DeltaNet 层中：
- `linear_num_key_heads` = 16（K heads）
- `linear_num_value_heads` = 32（V heads）

而 0.8B 中两者均为 16。原生 `Qwen3_5GatedDeltaNet.forward` 通过 `repeat_interleave` 将 q/k 的 head 数从 16 扩展到 32，以匹配 v/g/beta：

```python
if self.num_v_heads // self.num_k_heads > 1:
    query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
    key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
```

我们的 monkey-patch `_patched_dn_fwd` 漏掉了这一步，导致 `torch_recurrent_gated_delta_rule` 内部 K head 数（16）与 V/g head 数（32）不匹配。修复：在 patch 中追加 `repeat_interleave`。

#### 2. S-state 维度错误

S-state（DeltaNet recurrent state）的正确 shape 为 `(1, num_v_heads, head_k_dim, head_v_dim)`。导出脚本和缓存构造代码错误地使用了 `num_k_heads` 作为第一维。对于 0.8B 这无意中正确（K=V=16），但对 4B（K=16, V=32）导致越界。修复：统一使用 `linear_num_value_heads` 创建 S-state。

### 测试验证

#### 0.8B 向后兼容（4/16/4）

- ONNX 导出：PASS（checker + ORT 校验通过）
- 参考链路：SplitNN v.s. 完整 KVCache，max_diff=0.000000
- 控制端到端：非流式 / 流式 / 多轮 / thinking 均通过

#### 4B 长上下文（1/30/1, 16K）

| 测试项 | 结果 |
|--------|------|
| ONNX 导出 | PASS（prefix 1427.9 MB, suffix 1417.9 MB） |
| ORT 校验 | prefix max_diff=0.000244, suffix max_diff=0.000000 |
| 短问题 | "你好，我是 Qwen3.5，阿里巴巴最新推出的通义千问" |
| 长上下文 | 正确从重复 prompt 中提取核心信息 |
| 流式 SSE | "1 + 1 等于 **2**" |
| 多轮对话 | 正确记住用户名 |
| Thinking | 输出思考过程 |

#### 切分灵活性验证

`--split` 参数可以任意指定切分边界：

```bash
# 0.8B: 4/16/4（默认）
--split 4,20

# 4B: 1/30/1（板端内存最优）
--split 1,31

# 0.8B: 8/8/8（均衡示例）
--split 8,16
```

### 端边协同实测

**第一轮 — 0.8B 模型初步验证 (4/16/4, 256)：**

| 模式 | Prefill | Decode | 输出 |
|------|---------|--------|------|
| 普通 | 3.8s (13 tok, 295 ms/tok) | 1.8 tok/s | "你好！很高兴见到你..." |
| Thinking | 5.1s (20 tok, 257 ms/tok) | 2.4 tok/s | "Thinking Process: 1. Analyze..." |

**第二轮 — 4B (1/30/1, 16K) — 板端 1 GA 层瓶颈：** 0.3 tok/s, 319 tok 长上下文通过

**第三轮 — 4B 全服务器 (0/32/0, 16K) — GA 全 offload：**

| 指标 | 1/30/1 | 0/32/0 | 变化 |
|------|--------|--------|------|
| Decode | 0.3 tok/s | **1.2 tok/s** | 4x |
| Prefill (13 tok) | 63.4s | 5.1s | 12.4x |
| 长上下文 | 收到。 | 收到。 | ✓ |

### msprof 精确性能剖析

使用 CANN `msprof`（`--ascendcl=on --task-time=on`）对板端 OM 执行进行算子级时序采集：

**0.8B Suffix（4 层: 3DN+1GA+norm+lm_head, 256 ctx）：**

| OP Type | Core | 次数 | 总计 | 最大单次 |
|---------|------|-----|------|---------|
| BatchMatMulV2（全部 matmul） | AI_CORE | 34 | **60.9 ms** | 45.4 ms |
| Mul/Cast/Conv2D 等 | — | — | <3 ms | — |

**4B Head（仅 norm + lm_head, 16K ctx）：**

| OP Type | Core | 次数 | 总计 |
|---------|------|-----|------|
| **BatchMatMulV2 (lm_head)** | **AI_CORE** | **1** | **556.9 ms** |
| rmsnorm ×5 | AI_VECTOR | 5 | <0.1 ms |

**根因分析：**

- 4B lm_head（2560×248320, 635M 参数）单次 MatMul **557 ms**，占 Head OM 执行时间的 99.9%
- 0.8B lm_head（1024×248320, 254M 参数）仅 45 ms——尺寸差 2.5 倍，耗时差 **12 倍**
- msprof 报告标注 `Low data memory handling efficiency` for lm_head — Ascend310B4 DDR 带宽在 2560 维度下 tiling 参数恶化
- Embed OM（仅查表）<10μs，可忽略

**0/32/0 方案的 833 ms/tok 分布：**

```
Head OM lm_head MatMul:  557 ms (67%) ← 唯一瓶颈
Host middle 32层 + 网络:  250 ms (30%)
采样 + Python:             30 ms ( 3%)
```

### 硬件配置

- 开发板：Atlas 200I DK A2 (Ascend310B4, 4GB NPU)
- 主机：x86_64 + RTX 5070 Ti (16GB) — CUDA PyTorch
- 协议：每步 (1,1,hidden_size) FP16 over SSH reverse tunnel

### 板端文件布局（重要）

为使板端 Python 包正确相互引用，`/root/slm_deploy/` 必须包含 `__init__.py` 并按如下结构组织：

```
/root/slm_deploy/
├── gen_text_qwen35_splitnn.py     # 入口脚本
├── scripts/                        # Python 包
│   ├── __init__.py                 # 必须（可为空文件）
│   └── qwen35_model_spec.py       # ModelSpec/SplitConfig/load_metadata
├── controller/
│   ├── __init__.py                 # 必须
│   └── engine/
│       ├── __init__.py             # 必须
│       ├── base.py                 # SplitEngine ABC
│       └── om_engine.py            # OmSplitEngine (ACL 管理)
├── *.om                            # OM 模型文件
├── *.metadata.json                 # 配套元数据（与 OM 同名前缀）
├── tokenizer.json
├── tokenizer_config.json
└── chat_template.jinja
```

**为何需要 `__init__.py`：** `controller/engine/base.py` 使用 `from scripts.qwen35_model_spec import ModelSpec` 包导入方式。若 `scripts/` 目录无 `__init__.py`，Python 不将其作为包看待，导入失败。

**metadata.json 命名规则：** 与 `.om` 文件同名前缀。如 `qwen3.5_split_prefix_max256.om` 对应 `.metadata.json`。脚本通过 `Path(om_path).with_suffix(".metadata.json")` 自动查找。

### 架构变更总结

| 变更 | 影响文件 |
|------|---------|
| `ModelSpec` / `SplitConfig` | `scripts/qwen35_model_spec.py`（新增） |
| `SegmentRunner` 参数化 | `scripts/qwen35_split_common.py` |
| DeltaNet K/V head 修复 | `scripts/qwen35_split_common.py` |
| 导出脚本支持 `--split` | `export_qwen35_split_prefix.py`, `suffix.py` |
| 服务端支持 `--split` | `server/qwen35_split_service.py` |
| 控制器支持 `--split --model-path` | `controller/openai_split_controller.py` |
| 引擎参数化 | `controller/engine/onnx_engine.py`, `om_engine.py` |
| 协议动态 hidden_size | `controller/remote_middle.py` |
| Thinking 开关 | `controller/schemas.py`, `orchestrator.py` |
| 板端复用 OmSplitEngine | `board/gen_text_qwen35_splitnn.py` |
| OM 引擎条件 position | `controller/engine/om_engine.py` (`nl_ga > 0`) |
| metadata.json 机制 | `scripts/qwen35_model_spec.py` |

### 当前局限

1. **板端 decode 速度受 GA 注意力 O(L) 限制**：4B 16K 仅 0.3 tok/s，瓶颈在 suffix 的 1 层 GA 每步需计算 16384 个位置的注意力（218M MACs + 67MB DMA）。0.8B 16K 达 1.3 tok/s 因其后缀 4 层中仅 1 层 GA，3 层 DN(O(1)) 分摊了 GA 开销
2. **板端 SSE 流式未修复**：板端 OM 后端 SSE 流式收尾问题仍然存在
3. **全模型 KVCache 未更新**：`export_qwen35_kvcache.py` 仍使用硬编码的 0.8B 常量，暂不适用于 4B
4. **OM 模型不含 position 输入时的适配**：已通过 `nl_ga > 0` 条件判断修复（仅 GQA 层需要 position）

## 解决的关键工程问题

### CANN 版本匹配

开发板出厂 CANN 7.0.RC1，而最新发布的是 CANN 8.0。直接用 CANN 8 编译的 OM 在板端可能出现 `soft_fault`（Qwen3.5 实测）。原因是不同 CANN 大版本的 ATC 编译器可能生成不被旧版 runtime 识别的算子二进制。**解决方案**：让容器内的 CANN 版本与板端匹配（都是 7.0 系列），从根本上消除兼容性问题。代价是需要处理 CANN 7 特有的依赖问题（numpy<2、RPM 内核包、gcc-c++ 等）。

### ONNX opset 选择

ONNX opset 版本决定了哪些算子在导出图中可用。opset 过高可能导致 ATC 不支持某些新算子；opset 过低则可能缺少 `Where` 等关键算子。我们选择了 opset 15：`Where` 在 opset 9 就已支持，而 opset 15 是 ATC 编译器最成熟的版本（实验证明 opset 17/18 产出的图在 ATC 编译中遇到更多 shape 推断错误）。

### TorchScript vs Dynamo

PyTorch 提供了两种 ONNX 导出路径：传统的 `torch.onnx.export`（基于 TorchScript 追踪）和新的 `torch.export` + `torch.onnx.dynamo_export`（基于 Dynamo）。我们选择了传统路径（`dynamo=False`），原因是：
- 传统 TorchScript 对 monkey-patch 的兼容性更好——它直接执行一次模型前向传播并记录所有 torch 操作，不关心调用栈的来源
- Dynamo 会尝试对 Python 代码进行图捕获，在遇到我们修改过的 `forward` 方法时，其符号追踪器可能无法正确推断某些动态分支（如 layer_type 判断）
- 在实际测试中，Dynamo 导出在 DeltaNet 的循环推理上因 Python 控制流失败

### 大模型 ONNX 外部数据

PyTorch 2.11 对超过 ~2GB 的模型会自动使用外部数据格式——权重 tensor 作为独立文件存储，ONNX 文件中只保留图结构和引用路径。ATC 不能直接处理分散的外部数据文件。需要用 `onnx.save(..., save_as_external_data=True, all_tensors_to_one_file=True)` 将所有权重整合为单个 `.data` 文件，ATC 才能加载。

### TBE 编译器路径硬编码

`tbe/tvm/contrib/ccec.py` 里写死了 `/usr/local/Ascend/CANN-1.84/`。容器里 symlink 解决。

### Thinking 模板

Qwen3 的 chat template 默认开启 `<think>` 推理链，对 0.6B 小模型反而浪费时间。`enable_thinking=False` 关闭。

### NPU 进程残留

ACL 进程被 kill -9 后（尤其 D 状态），NPU 内存无法自动回收。解决方案：重启板子。遇到脚本卡死时不要反复跑——先确认 NPU 是否干净。

### ACL API 返回值

`acl.mdl.add_dataset_buffer()` 返回 `(ptr, ret)` 而非单个 ret code，需要 `_, ret = ...` 解包。
