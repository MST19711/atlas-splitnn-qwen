# 设计决策

从开发历程中提取的关键设计决策和取舍。

---

## Monkey-patch：为什么必须修改模型代码

ONNX 和 CANN 的组合约束使得直接导出原生 Qwen3/Qwen3.5 模型不可行：

1. **`cat` 动态 shape → ONNX 静态图不可表达**。原生的 `torch.cat([past_k, new_k])` 每次输出维度 +1，ONNX 图无法表达动态 shape，ATC 编译直接失败。
2. **`scatter_nd` 不稳定**。尝试过 `scatter_nd` 来原地写入，但它在 CANN 310B4 内核编译中反复出现 shape 推断错误。
3. **`Where` 是唯一稳定通过的方案**。`Where(mask, new_kv, cache)` 是 ONNX 最基础的逻辑算子（opset 9），CANN 任何版本都稳定支持。

代价是需要预分配整块 K/V 缓冲区并把 mask 广播到完整尺寸——多占用一些内存，但换来了绝对稳定的编译成功率。

---

## KV Cache 缓冲区：双缓冲 AB/BA

**为什么需要两组 dataset？** 每步 ACL 执行后，输出 K/V 指针指向缓冲区 B，下一步需要把它作为输入。如果只有一组 dataset，就需要用 `acl.update_data_buffer` 来换绑（CANN 7.0.RC1 上存在稳定性问题）。两组 dataset 各绑定不同的 K/V 指针组，轮流使用——这一步的输出指针恰好是下一步的输入指针，完全避免了 buffer 的动态换绑。

```
偶数步: dataset_A (输入) → execute → dataset_B (输出)
奇数步: dataset_B (输入) → execute → dataset_A (输出)
```

### 性能优化历程

1. **初始版本**：每步 `acl.rt.malloc×115` + `acl.rt.memcpy×112` + `acl.rt.free×115` → ~420ms
2. **第一轮**：预分配 Device 缓冲区 + K/V 指针轮转 → 消除 alloc/free 和 K/V memcpy
3. **第二轮**：预创建 AB/BA 两组 Dataset → 消除每步的 create_dataset/destroy_dataset

最终每步约 400ms（Profiler 显示 NPU 纯计算 174ms，剩余 ~226ms 为 ACL 框架和 Python 开销）。

---

## CANN 版本选择：7.0 vs 8.0

开发板出厂预装 CANN 7.0.RC1 runtime。我们尝试了 CANN 8.0.RC3 编译的 OM：
- Qwen3-0.6B 可以运行（只用 MatMul/Where 等基础算子）
- Qwen3.5-0.8B 加载时出现 `drv_soft_fault (err_type=0xa)`，NPU 驱动拒绝执行

Qwen3.5 的 DeltaNet 算子在 CANN 8 编译后的二进制格式不被 CANN 7 runtime 识别。与其升级板端固件，不如将编译工具链降级到与板端匹配的 CANN 7 版本。

---

## SplitNN 切分策略

### 4/16/4 切分（0.8B）

```text
layers[0:4]  → Prefix (板端 OM, 含 Embedding + 3层DeltaNet + 1层GQA)
layers[4:20] → Middle (CUDA 主机, 14层DeltaNet + 2层GQA)
layers[20:24] → Suffix (板端 OM, 1层DeltaNet + 3层GQA + LM Head)
```

选择 4/16/4 的理由：
- Prefix 需要至少 1 层 GQA（提供初始 attention cache）
- Suffix 需要能够计算完整的 logits（至少 1 层 + LM Head）
- Middle 承担最大的计算负载（16 层）

### 0/24/0 切分（2B 参数绑定）

板端仅执行 embedding lookup + tied lm_head matmul，主机承担全部 Transformer 层。利用 `tie_word_embeddings=True` 特性，只需在板端保存一份 `tied_weight.bin`。

### 模型与引擎模式的映射说明

当前文档中 0.8B ↔ om_split、2B ↔ bound_embed_head、4B ↔ om_split 的分配关系源于**实验路线图的时间顺序**（阶段四 0.8B SplitNN → 阶段七 2B 参数绑定 → 阶段十 4B SplitNN），而非技术上的硬约束：

- 三种引擎模式（`om_split`、`bound_embed_head`、`onnx`）与模型大小、切分方案在语法层面是**正交可组合**的。例如 0.8B 同样可以走 `splitnn_bound_embed_head` + `4/20` 切分（需额外导出纯注意力段），2B 也可以走 `splitnn_om` + 非零 attention 层切分。
- 所有模型均满足 `tie_word_embeddings=True`，理论上均能使用参数绑定模式。
- 当前的分配只是各自实验阶段中最先验证通过的组合，并非排他性设计。



## Controller 设计：无状态多轮

控制器首版故意采用无状态多轮：每次请求都基于完整 `messages` 重新 prefill，不在请求之间保留 cache。理由：

1. 与 OpenAI API 的无状态使用方式一致
2. 不需要额外管理跨请求 session 存活、并发冲突和 cache 泄漏
3. 先把"正确生成"与"统一接口"打通

---

## 三层架构：API / Controller / Engine

```
API 层    → 解析请求，路由到对应端点，返回 OpenAI 格式响应
Controller → chat template、prefill/decode 循环、采样、stop 判断
Engine    → 执行 NPU 推理（ACL），管理设备内存与 cache
```

这种分层使得：
- 控制器逻辑在 ONNX 仿真和 NPU 部署中完全共享（换引擎即可）
- 引擎实现可以独立优化（ACL 双缓冲、OM 拆分等）
- 上层 OpenAI API 兼容性不影响下层推理引擎

---

## thinking 模式

`enable_thinking=True` 时使用不同的采样策略：

| 参数 | thinking=True | thinking=False |
|------|:---:|:---:|
| temperature | 1.0 | 0.7 |
| top_k | 20 | 40 |
| top_p | 0.95 | 1.0 |
| presence_penalty | 1.5 | 0.0 |

通过检测 `</think>` token 自动切换阶段——thinking 内容在 `</think>` 之前，之后的为可见输出。

---

## 上下文窗口限制

- `max_len=256`: 对话历史足够容纳日常对话，NPU 内存开销可控
- `max_len=4096`: 支持长文档分析，OM 文件增大至 2.0GB
- `max_len=16384`: 仅 SplitNN 方案支持（CUDA 主机管理大部分 cache）

每次增大 max_len，K/V buffer 线性增长（每层 K 和 V 各一个 max_len×dim 的 buffer），需要在内存和上下文之间权衡。
