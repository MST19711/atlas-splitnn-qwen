# 架构设计

本文档描述推理系统的整体架构、三种引擎模式的工作原理及组件关系。

---

## 总体架构

```
┌─────────────────────────────────────────────────────┐
│  OpenAI 兼容 API (FastAPI + uvicorn :8000)             │
│  GET /healthz  GET /v1/models  POST /v1/chat/completions │
├─────────────────────────────────────────────────────┤
│  控制器中间层 (controller/)                            │
│  ├── openai_controller.py    —— API 入口，参数解析      │
│  ├── schemas.py              —— Pydantic 请求/响应模型  │
│  ├── generation/runner.py    —— 统一 prefill/decode 循环│
│  ├── generation/config.py    —— SamplingParams 采样参数 │
│  ├── generation/strategies.py —— 采样策略               │
│  ├── modeling/factory.py     —— 后端模型工厂            │
│  └── tokenization/qwen35.py  —— Tokenizer 适配器        │
├─────────────────────────────────────────────────────┤
│  板端引擎 (modeling/ + engine/)                        │
│  ├── Qwen35KvCacheModel      —— 纯板端 KV Cache        │
│  ├── SplitNNQwen35Model      —— SplitNN 前后段          │
│  │   ├── OmSplitEngine       —— ACL OM 执行            │
│  │   └── RemoteMiddleClient  —— HTTP 远端中段协议       │
│  └── _ACLSessionRuntime      —— ACL 底层封装            │
└─────────────────────────────────────────────────────┘
```

三层职责分明：

- **API 层**：解析请求，路由到对应端点，返回 OpenAI 格式响应
- **控制器层**：chat template 格式化、prefill/decode 循环、采样、stop 判断——与引擎无关
- **引擎层**：执行 NPU 推理（ACL），管理设备内存与 cache——与控制逻辑无关

### 模型工厂

`controller/modeling/factory.py` 通过 `BackendConfig` 数据类描述部署参数，`create_model()` 工厂函数根据 `--backend` 参数选择对应引擎。

统一启动格式：

```bash
python3 controller/openai_controller.py \
  --host 0.0.0.0 --port 8000 \
  --backend <mode> \
  --model-name <name> \
  --tokenizer-dir <path> \
  --max-len <n> \
  [mode-specific args...]
```

### 三后端对照

| 后端 | 模型类 | 引擎 | 说明 |
|------|--------|------|------|
| `qwen35_kvcache_om` | `Qwen35KvCacheModel` | 无（自包含） | 完整 OM 在板端运行 |
| `splitnn_om` | `SplitNNQwen35Model` | `OmSplitEngine` (mode="om_split") | 板端 prefix+suffix OM |
| `splitnn_bound_embed_head` | `SplitNNQwen35Model` | `OmSplitEngine` (mode="bound_embed_head") | 板端 embed+head |

所有后端共享同一套 OpenAI API 端点：

| 端点 | 方法 | 说明 |
|------|------|------|
| `/healthz` | GET | 健康检查，含模型加载状态、后端类型、max_len |
| `/v1/models` | GET | 模型列表（OpenAI 兼容格式） |
| `/v1/chat/completions` | POST | 对话补全（支持 `stream: true/false`、`enable_thinking`） |

---

## 引擎模式一：纯板端 KV Cache (`qwen35_kvcache_om`)

**最简部署模式**——开发板独立运行全部推理，无需外部 CUDA 主机。

```
┌── 开发板 ──────────────────────────────┐
│  OpenAI 控制器                            │
│    ↓                                     │
│  Qwen35KvCacheModel                      │
│    ├── Embedding 层（OM 内置）             │
│    ├── 18 层 DeltaNet（线性注意力 + conv）  │
│    ├── 6 层 Full Attention（KV Cache）     │
│    ├── RMSNorm + LM Head（OM 内置）        │
│    └── _ACLSessionRuntime               │
│         ├── acl.mdl.load_from_file(.om)  │
│         ├── 双缓冲 AB/BA dataset          │
│         └── H2D/D2H memcpy + execute     │
└──────────────────────────────────────────┘
```

模型编译为一个完整的 OM 文件（50 输入 / 49 输出），`_ACLSessionRuntime` 使用交替双缓冲区管理 DeltaNet 的 S/Cache（18 组）和 Full Attention 的 K/V Cache（6 组），避免每步复制整个 cache 状态。

- 模型加载约 210 秒（1.9GB OM）
- decode 阶段约 200ms/tok（~4.7 tok/s）
- 支持 256 / 4096 上下文窗口

---

## 引擎模式二：SplitNN OM (`splitnn_om`)

将模型切分为三段，前后段在板端 NPU，中段在 CUDA 主机。

```
┌── 开发板 ──────────────────┐    ┌── CUDA 主机 ────────────────┐
│  OpenAI 控制器               │    │  Middle Server (:18080)      │
│    ↓                        │    │    ↓                         │
│  SplitNNQwen35Model         │    │  MiddleWrapper               │
│    ├── OmSplitEngine        │    │    ├── layers[4:20]          │
│    │   ├── Prefix OM(0-3层) │    │    ├── S/Cache (16组)        │
│    │   └── Suffix OM(20-23层)│   │    └── K/V Cache (2组)        │
│    └── RemoteMiddleClient   │    │    SessionState (per-session) │
│         └── HTTP /v1/session/step │                              │
│  ────────────SSH隧道:28080─────────────────────────────        │
└─────────────────────────────┘    └──────────────────────────────┘
```

每步推理三个阶段：

1. **Prefix**（板端 NPU）：Embedding → layers[0:4] → hidden state
2. **Middle**（CUDA 主机）：HTTP 传输 hidden state → layers[4:20] → hidden state
3. **Suffix**（板端 NPU）：layers[20:24] → LM Head → logits

切分格式 `split-4-16-4` 表示前段 4 层 / 中段 16 层 / 后段 4 层。`--split 4,20` 表示 `prefix_end=4, suffix_start=20`。

---

## 引擎模式三：SplitNN 参数绑定 (`splitnn_bound_embed_head`)

板端利用 `tie_word_embeddings=True` 共享 `tied_weight.bin` 处理 Embedding + LM Head（避免 OM 中重复存储）。支持任意 split（如 `0/24/0` 或 `4/20`）：

- **split=0/24/0**: 板端仅 Embedding + LM Head，全部 Transformer 在主机
- **split=4/20**: 板端 Embedding + 前4层注意力 + 后4层注意力 + LM Head，中段在主机

```
┌── 开发板 ──────────────────┐    ┌── CUDA 主机 ────────────────┐
│  Prefix: tied_weight 查表    │    │  layers[prefix_end:         │
│    token_id → hidden_state  │    │           suffix_start]      │
│         ↓ (可选)            │    │    → 返回 hidden_state       │
│  Prefix OM(attention only)  │    │                              │
│    hidden → ... → hidden    │    │                              │
│         ↓                   │    │                              │
│  ───── HTTP step ────────→ │    │                              │
│         ↓                   │    │                              │
│  (可选) Suffix OM            │    │                              │
│    hidden → ... → hidden    │    │                              │
│         ↓                   │    │                              │
│  Suffix: RMSNorm + tied_weight^T                               │
│    → logits (ACL MatMul)    │    │                              │
└─────────────────────────────┘    └──────────────────────────────┘
```

prefix/suffix OM 为纯注意力段（`HiddenSegmentWrapper`），不含 embedding 也不含 head，板端通过 `np.memmap` 读取 `tied_weight.bin` 做查表，通过 ACL single-op `MatMul` 做 head 计算。

---

## 生成流水线

```
OpenAIChatAdapter.run_non_stream() / run_stream()
  → resolve_sampling_params(request)         # 请求 → SamplingParams
  → build_prompt_ids(request)                # messages → 模板 → tokenize
  → TokenGenerationRunner.generate(prompt_ids, params)
      → model.create_session()  → Qwen35Session
      → session.prefill(input_ids)            # 初始前向传播
      → for _ in range(max_new_tokens):
          → _select_token(logits, params, counts)   # 采样
          → 处理 thinking_phase 转换
          → 检测 eos_token / stop strings
          → session.decode_next(token_id)     # 自回归步
          → yield GenerationStep(delta_text, finish_reason)
      → session.close()
```

双缓冲机制：交替使用两组 cache 缓冲区（A/B），偶数步读 A 写 B，奇数步读 B 写 A，避免每步复制整个 cache。

---

## 抽象层设计

### Qwen35Model (modeling/base.py)

```
Qwen35Model (ABC)
├── load() → None
├── close() → None
├── create_session() → Qwen35Session
├── is_loaded() → bool
└── info: ModelInfo
```

### Qwen35Session (modeling/base.py)

```
Qwen35Session (ABC)
├── prefill(input_ids: list[int]) → np.ndarray  # logits
├── decode_next(token_id: int) → np.ndarray
└── close() → None
```

### SplitEngine (engine/base.py)

```
SplitEngine (ABC)
├── load() → None
├── close() → None
├── start_session() → None
├── end_session() → None
├── run_prefix(token_id, position) → np.ndarray
└── run_suffix(hidden, position) → np.ndarray
```

---

## 模型对照

| 模型 | 参数量 | hidden_size | 层数 | 上下文 | 方案 |
|------|--------|-------------|------|--------|------|
| Qwen3-0.6B | ~600M | 1024 | 28 (GQA) | 256 tok | KV Cache |
| Qwen3.5-0.8B | ~800M | 1024 | 24 (18DN+6GA) | 256/4096 tok | KV Cache / SplitNN |
| Qwen3.5-2B | ~2B | 2048 | 24 (18DN+6GA) | 8K tok | SplitNN 参数绑定 |
| Qwen3.5-4B | ~4B | 2560 | 32 (24DN+8GA) | 16K tok | SplitNN OM |

---

## 关键设计决策

### DeltaNet 混合架构

Qwen3.5 使用 DeltaNet（线性注意力）+ Full Attention 混合架构。每 `full_attention_interval=4` 层插入一层全局注意力：
- 前 18 层为 DeltaNet（线性注意力，S/Cache 为状态矩阵）
- 后 6 层为 Full Attention（标准 KV Cache）

双缓冲机制同时管理两种 cache 格式。

### tied weights

Qwen3.5 设置 `tie_word_embeddings=True`，embedding 和 lm_head 共享权重矩阵。参数绑定模式利用此特性，板端只需保存一份 `tied_weight.bin`。

### thinking 模式

`enable_thinking=True` 时使用不同采样参数（temperature=1.0, top_k=20, presence_penalty=1.5），检测 `</think>` token 自动切换阶段。
