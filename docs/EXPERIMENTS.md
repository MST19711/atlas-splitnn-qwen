# 实验记录

Qwen3/Qwen3.5 在 Atlas 200I DK A2 上部署的十个阶段实验记录概要。

---

## 阶段一：静态窗口模型 (已废弃)

用固定 seq_len=32 的滑动窗口替代 KV Cache，实现简单但性能差。在 CANN 7.1.0.3.220 + Ascend310B4 下 ATC 编译产物输出错误，相关代码已移除。

- 关键教训：GQA 的 Expand 节点需要静态 shape，动态计算的 Expand 在 ATC 中不可用 → 改为 Tile
- 每 token 约 280ms，解码速度 ~3.6 tok/s

---

## 阶段二：KV Cache 模型

用 monkey-patch 将原生 `torch.cat` 替换为 `torch.where` 实现静态 shape KV Cache。

- **核心发现**：`Where` 是唯一在 ONNX→ATC 链路上稳定通过的 cache 写入方案
- **三层 patch**：Qwen3Attention → Qwen3DecoderLayer → KVCacheWrapper
- **双缓冲优化**：从每步 420ms 优化到 ~400ms（消除 malloc/free 和 K/V memcpy）
- OM 大小 1.5GB，上下文 256 tok

---

## 阶段三：Qwen3.5-0.8B DeltaNet 模型

Qwen3.5 引入 DeltaNet（线性注意力）+ Full Attention 混合架构。

- **四个轻量 patch**：cat→Where（K/V 缓存）、Trilu→Where+Equal（causal mask）、RMSNorm type_as→to、conv copy_→返回新 state
- DeltaNet 的 recurrent 计算完全不需要 patch
- OM 大小 1.9GB，上下文 256/1024 tok

---

## 阶段四：SplitNN 原型设计

将模型切分为三段（4/16/4），前后段在板端，中段在 CUDA 主机。

- 四层验证：单段 ONNX 导出 → 三段联调 (ORT) → 本地模拟联调 (HTTP) → 板端 OM
- 设计约束：中间段不能产生 logits（避免 FP16/FP32 精度转换矛盾，改用 hidden state 传递）

---

## 阶段五：SplitNN 控制器与 OpenAI 接口

引入统一的控制器中间层，提供 OpenAI 兼容 API。

- 四层架构：API 层 / 编排层 / 引擎层 / 远端中段层
- 首版无状态多轮（每次请求重新 prefill）
- 流式 SSE + 非流式两种响应模式
- CUDA 兼容性修复（切换到 cu128 PyTorch wheel）

---

## 阶段六：SplitNN 通用化

引入 ModelSpec/SplitConfig 零依赖数据类，通过 `--split` 参数化切分方案。

- `qwen35_model_spec.py`：无 torch 依赖的模型架构描述
- `SplitConfig`：prefix_end / suffix_start 参数化
- 导出脚本统一支持 `--split` 参数（不再硬编码切分点）

---

## 阶段七：板端参数绑定与 OpenAI 控制器落地

利用 tied weights 特性实现 0/24/0 切分：板端仅 Embedding + LM Head。

- tied_weight.memmap 做查表（避免加载完整模型）
- ACL single-op MatMul 执行 LM Head（14KB OM）
- 板端 tied_weight 970MB（FP16，2048×248320）
- 支持 2B 模型，8K 上下文

---

## 阶段八：断连回收与服务器内存异常修复

修复中段服务器内存泄漏和 session 超时问题。

- GC 后台线程：每 30 秒清理过期 session（默认超时 60 秒）
- Session state 空闲内存清零（不用时释放 KV cache 显存）
- HTTP 连接超时处理

---

## 阶段九：纯板端 KV Cache OpenAI API 控制器落地

将 KV Cache 方案接入统一的 OpenAI 控制器框架。

- 新增 `qwen35_kvcache_om` 后端
- 无需 CUDA 主机，板端独立运行完整 API
- 模型加载约 210 秒（1.9GB OM）

---

## 阶段十：SplitNN RMSNorm 修复、Thinking 解禁与 4B 部署

- 修复 SplitNN 中 RMSNorm 的 `output.type_as(x)` ONNX 兼容性（`to(x.dtype)` 替换 `type_as`）
- Thinking 模式解禁（`enable_thinking` 参数正式支持）
- Qwen3.5-4B SplitNN 1/30/1 切分验证通过（Prefix+Suffix ~2.8GB OM, 16K 上下文, ~1.1 tok/s）
