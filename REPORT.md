# Qwen3 / Qwen3.5 在 Atlas 200I DK A2 上的部署实验报告

本文档记录了将通义千问小语言模型部署到华为昇腾 Atlas 200I DK A2 开发板的完整实验过程，包括每个阶段的设计思路、遇到的工程问题及解决方案。

---

## 目录

1. [背景知识](#背景知识)
2. [阶段一：静态窗口模型](#阶段一静态窗口模型)
3. [阶段二：KV Cache 模型](#阶段二kv-cache-模型)
4. [阶段三：Qwen3.5-0.8B DeltaNet 模型](#阶段三qwen35-08b-deltanet-模型)
5. [解决的关键工程问题](#解决的关键工程问题)

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

```
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
