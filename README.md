# Qwen3 / Qwen3.5 在华为 Atlas 200I DK A2 上的部署

[English](./README_EN.md) | [实验报告](./REPORT.md)

将通义千问小语言模型部署到华为昇腾 Atlas 200I DK A2 边缘计算板，实现 NPU 加速的端侧中文对话。

---

## 成果概览

| 方案 | 模型 | 上下文 | OM 大小 |
|------|------|--------|---------|
| Qwen3 KV Cache | Qwen3-0.6B | 256 tok | 1.5 GB |
| Qwen3.5 KV Cache | Qwen3.5-0.8B | 256 tok | 1.9 GB |
| Qwen3.5 KV Cache | Qwen3.5-0.8B | 4096 tok | 2.0 GB |
| **Qwen3.5 KV Cache + OpenAI API** | **Qwen3.5-0.8B** | **256 tok** | **1.9 GB，纯板端独立运行** |
| **Qwen3.5 SplitNN** | **Qwen3.5-4B (1/30/1)** | **16K tok** | **Prefix+Suffix ~2.8 GB** |
| **Qwen3.5 SplitNN 参数绑定** | **Qwen3.5-2B (0/24/0)** | **8K tok** | **板端共享 tied weight 970MB + 单算子 head OM 14KB** |

> SplitNN 方案在开发机 ONNX 后端已联调通过，4B 模型 16K 上下文可稳定推理。板端 OM 部署待 ATC 编译后实测。
>
> 目前仓库已新增一条可实际部署到开发板的 `Qwen3.5-2B split 0/24/0` 参数绑定链路：板端提供 OpenAI 兼容控制器，主机承担全部 Transformer 主干层，板端仅执行 `embedding + tied lm_head`。

---

## 硬件与工具链

| 组件 | 说明 |
|------|------|
| 开发板 | Atlas 200I DK A2 (Ascend310B4, 4GB NPU) |
| 模型 | Qwen3-0.6B / Qwen3.5-0.8B / Qwen3.5-4B (FP16) |
| CANN | 7.0.0 (ATC 编译容器) / 7.0.RC1 (板端 runtime) |
| ONNX | opset 15, TorchScript 导出 |
| 容器 | Podman + Rocky Linux 9, 镜像 `cann-atc-rocky:v7` |
| Python | pixi 管理 (x86), pip (板端 aarch64) |

---

## 项目结构

```
├── model/                    # 模型权重 + tokenizer
├── scripts/                  # ONNX 导出 & ATC 转换 (x86 dev)
│   ├── qwen35_model_spec.py       # ModelSpec/SplitConfig (无 torch 依赖)
│   ├── qwen35_split_common.py     # SplitNN 共享代码（Wrappers + Patches）
│   ├── export_qwen35_split_prefix.py   # Prefix ONNX 导出（支持 --split）
│   ├── export_qwen35_split_suffix.py   # Suffix ONNX 导出（支持 --split）
│   ├── export_qwen35_bound_embed_head.py  # 导出参数绑定资产（tied weight + metadata）
│   ├── export_qwen3_kvcache.py         # Qwen3 KV Cache 导出
│   ├── export_qwen35_kvcache.py          # Qwen3.5 DeltaNet KV Cache 导出
│   ├── gen_input_shape.py        # ONNX → ATC INPUT_SHAPE 辅助
│   ├── podman_convert.sh         # 容器化 ATC 转换
├── board/                    # 板端推理 (aarch64)
│   ├── gen_text_qwen3_kvcache.py       # Qwen3 KV Cache 推理
│   ├── gen_text_qwen35_kvcache.py        # Qwen3.5 DeltaNet KV Cache 推理
│   ├── gen_text_qwen35_splitnn.py        # Qwen3.5 SplitNN 推理（复用 OmSplitEngine）
│   ├── run_openai_kvcache_controller.sh      # 纯板端 OpenAI 控制器启动脚本（推荐入门）
│   ├── run_openai_split_controller_bound_2b.sh  # 板端 OpenAI 控制器启动脚本（2B 参数绑定）
│   └── run_qwen3_kvcache.sh
├── controller/               # OpenAI API 控制器（FastAPI + 可插拔后端引擎）
│   ├── openai_controller.py         # 主入口，支持 --backend qwen35_kvcache_om
│   ├── schemas.py                   # Pydantic 数据模型 (含 enable_thinking)
│   ├── orchestrator.py              # 消息编排、采样、生成循环
│   ├── remote_middle.py             # 与远端中段服务的 HTTP 协议
│   ├── modeling/
│   │   ├── base.py                  # Qwen35Model/Qwen35Session 抽象
│   │   ├── factory.py               # 模型工厂（kvcache_om / splitnn / bound_embed_head）
│   │   ├── kvcache_qwen35.py       # 纯板端 KV Cache 模型（ACL 后端）
│   │   └── splitnn_qwen35.py       # SplitNN 模型（引擎 + 远端中段）
│   ├── engine/
│   │   ├── base.py                  # 引擎抽象基类
│   │   ├── onnx_engine.py           # ONNX Runtime 引擎 (x86 仿真)
│   │   └── om_engine.py             # OM (NPU) 引擎
│   ├── generation/                  # 生成循环与采样策略
│   └── tokenization/                # Qwen3.5 tokenizer 适配器
├── docker/                   # ATC 容器构建
│   └── Containerfile.v2-cann7
├── server/                   # CUDA 主机服务
│   └── qwen35_split_service.py      # 中段 HTTP 服务（支持 --split）
├── om_out/                   # ATC 编译产物 (*.om, *.onnx)
└── logs/                     # 编译日志
```

---

## 板端推理架构

以下描述开发板上完整的推理系统架构，包括**控制器中间层**与三种**板端引擎模式**。

### 总体架构

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

### 控制器-引擎中间层

`controller/modeling/factory.py` 是连接控制器与引擎的关键。通过 `BackendConfig` 数据类描述部署参数，`create_model()` 工厂函数根据 `--backend` 参数选择对应的引擎。

启动命令通用格式：

```bash
python3 controller/openai_controller.py \
  --host 0.0.0.0 --port 8000 \
  --backend <mode> \
  --model-name <name> \
  --tokenizer-dir <path> \
  --max-len <n> \
  [mode-specific args...]
```

统一的 OpenAI API 端点：

| 端点 | 方法 | 说明 |
|------|------|------|
| `/healthz` | GET | 健康检查，含模型加载状态、后端类型、max_len |
| `/v1/models` | GET | 模型列表（OpenAI 兼容格式） |
| `/v1/chat/completions` | POST | 对话补全（支持 `stream: true/false`） |

统一的可调参数（所有模式通用）：

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `model` | str | — | 模型名，需与 `--model-name` 一致 |
| `messages` | list | — | 对话历史 `[{"role":"user","content":"..."}]` |
| `max_tokens` | int | 64 | 最大生成 token 数（上限 1024） |
| `temperature` | float | 0.7 | 温度，0 = 贪心 |
| `top_k` | int | 40 | Top-K 采样 |
| `top_p` | float | 1.0 | Top-P 采样 |
| `repetition_penalty` | float | 1.0 | 重复惩罚 |
| `presence_penalty` | float | 0.0 | 存在惩罚 |
| `stop` | str / list | — | 停止词 |
| `stream` | bool | false | 是否 SSE 流式 |
| `enable_thinking` | bool | false | 是否启用 `<think>` 推理链 |

---

### 引擎模式一：纯板端 KV Cache (`qwen35_kvcache_om`)

**最简部署模式**——开发板独立运行全部推理，无需外部 CUDA 主机或网络。

#### 工作原理

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

模型编译为一个完整的 OM 文件（50 输入 / 49 输出），包含全部 24 层 Transformer。`_ACLSessionRuntime` 使用交替双缓冲区管理 DeltaNet 的 S/Cache（18 组）和 Full Attention 的 K/V Cache（6 组）。

#### 启动

```bash
cd /root/slm_deploy
bash run_openai_kvcache_controller.sh
```

等价于：

```bash
python3 controller/openai_controller.py \
  --host 0.0.0.0 --port 8000 \
  --backend qwen35_kvcache_om \
  --model-name qwen3.5-0.8B-kvcache-om \
  --model-path /root/slm_deploy \
  --model-om /root/slm_deploy/qwen3.5_kvcache_max256.om \
  --tokenizer-dir /root/slm_deploy \
  --max-len 256
```

#### 板端必需文件

| 文件 | 用途 |
|------|------|
| `controller/`（完整目录） | 控制器 + 模型 + 引擎 + tokenizer |
| `qwen35_model_spec.py` | ModelSpec 结构定义 |
| `qwen3.5_kvcache_max256.om` | KV Cache OM 模型（1.9GB） |
| `config.json` | 模型架构参数 |
| `tokenizer.json`, `vocab.json`, `merges.txt`, `tokenizer_config.json`, `chat_template.jinja` | Tokenizer |
| `run_openai_kvcache_controller.sh` | 一键启动脚本 |

#### 可用 OM 模型

| 文件 | 上下文 | 大小 |
|------|--------|------|
| `qwen3.5_kvcache_max256.om` | 256 tok | 1.9 GB |
| `qwen3.5_kvcache_max4096.om` | 4096 tok | 1.9 GB |

切换模型只需修改 `--model-om` 和 `--max-len`。

#### 注意事项

- 模型加载约 **210 秒**（OM 文件从磁盘载入 NPU 内存）
- prefill 阶段按 prompt 长度逐 token 执行，长 prompt 需数十秒
- decode 阶段约 200ms/tok（~4.7 tok/s）
- 异常退出后 NPU 变 Alarm → 必须 `reboot`

---

### 引擎模式二：SplitNN OM (`splitnn_om`)

**将模型切分为三段**，前段和后段在板端 NPU 执行，中段在 CUDA 主机执行。适合需要更大模型或更长上下文的场景。

#### 工作原理

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

每步推理经过三个阶段：

1. **Prefix**（板端 NPU）：Embedding → layers[0:4]，产出 hidden state
2. **Middle**（CUDA 主机）：HTTP 传输 hidden state → layers[4:20] → 返回 hidden state
3. **Suffix**（板端 NPU）：layers[20:24] → LM Head → logits

#### 启动

**主机侧**（需要 CUDA GPU）：

```bash
pixi run python server/qwen35_split_service.py \
  --host 0.0.0.0 --port 18080 \
  --model-path model/Qwen3.5-0.8B \
  --device cuda:0 --max-len 16384 --split 4,20
```

**建立 SSH 反向隧道**（在开发机上执行）：

```bash
sshpass -p 'Mind@123' ssh -o StrictHostKeyChecking=no \
  -o ExitOnForwardFailure=yes \
  -N -R 28080:127.0.0.1:18080 \
  root@192.168.137.100
```

**板端**启动控制器：

```bash
cd /root/slm_deploy
bash run_openai_split_controller_om.sh
```

等价于：

```bash
python3 controller/openai_controller.py \
  --host 0.0.0.0 --port 8000 \
  --backend splitnn_om \
  --model-name qwen3.5-split-4-16-4-om \
  --remote-model-name Qwen3.5-0.8B-split-4-16-4 \
  --tokenizer-dir /root/slm_deploy \
  --server-url http://127.0.0.1:28080 \
  --max-len 16384 \
  --prefix-om /root/slm_deploy/qwen3.5_split_prefix_max16384.om \
  --suffix-om /root/slm_deploy/qwen3.5_split_suffix_max16384.om \
  --checksum
```

#### 板端必需文件

| 文件 | 用途 |
|------|------|
| `controller/` | 控制器 + SplitNN 模型 + OM 引擎 |
| `qwen35_model_spec.py` | ModelSpec / SplitConfig |
| `qwen3.5_split_prefix_max16384.om` (+ `.metadata.json`) | 前段 OM（layers 0-3） |
| `qwen3.5_split_suffix_max16384.om` (+ `.metadata.json`) | 后段 OM（layers 20-23） |
| Tokenizer 文件 | 同上 |

#### 切分方案说明

模型名 `split-4-16-4` 表示：前段 4 层 / 中段 16 层 / 后段 4 层（共 24 层）。`--split 4,20` 表示 `prefix_end=4, suffix_start=20`（即 layers[0:4] + layers[4:20] + layers[20:24]）。

#### 扩展性

- 支持 `--max-len 16384`（16K 上下文，已验证）
- 同样模型架构可用于不同尺寸（2B / 4B，替换 OM + 调整 split 参数）
- `--checksum` 开启 HTTP 传输 CRC32 校验，防止 SSH 隧道误码

---

### 引擎模式三：SplitNN 参数绑定 (`splitnn_bound_embed_head`)

**最极端的切分模式**——板端只承担 Embedding 查表和 LM Head 矩阵乘法（tied weights），所有 24 层 Transformer 在 CUDA 主机执行。

#### 工作原理

```
┌── 开发板 ──────────────────┐    ┌── CUDA 主机 ────────────────┐
│  Prefix: 查表 tied_weight    │    │  All 24 Transformer layers   │
│    token_id → hidden_state  │    │    → 返回 post-norm hidden   │
│         ↓                   │    │                              │
│  ───── HTTP step ────────→ │    │                              │
│         ↓                   │    │                              │
│  Suffix: hidden @ tied_weight^T │                             │
│    → logits (ACL MatMul)    │    │                              │
└─────────────────────────────┘    └──────────────────────────────┘
```

板端利用 Qwen3.5 模型 **Embedding 和 LM Head 共享权重**（tied weights）的特性，只需在 NPU 上保存一份 `tied_weight.bin`，通过 ACL single-op `MatMul` 执行 LM Head 计算。

#### 启动

与 SplitNN OM 有共同的启动流程，但控制器配置不同：

```bash
cd /root/slm_deploy
bash run_openai_split_controller_bound_2b.sh
```

等价于：

```bash
python3 controller/openai_controller.py \
  --host 0.0.0.0 --port 8000 \
  --backend splitnn_bound_embed_head \
  --model-name qwen3.5-2b-split-0-24-0-om \
  --remote-model-name Qwen3.5-2B-split-0-24-0 \
  --tokenizer-dir /root/slm_deploy/model_2b \
  --server-url http://127.0.0.1:28080 \
  --max-len 8192 \
  --split 0,24 \
  --bound-asset-dir /root/slm_deploy/qwen3.5_2b_bound_embed_head \
  --checksum
```

关键参数 `--split 0,24`：前段 0 层 / 中段 24 层 / 后段 0 层，板端只负责首尾的 embedding + tied head。

#### 板端必需文件

| 文件 | 用途 |
|------|------|
| `qwen3.5_2b_bound_embed_head/` | 参数绑定资产目录 |
| `  ├── tied_weight.bin` | 共享权重（vocab_size × hidden_size，FP16） |
| `  ├── final_norm_weight.bin` | 最终 RMSNorm 权重 |
| `  ├── bound_embed_head.metadata.json` | 元数据 |
| `  └── op_models/` | ACL single-op MatMul 编译产物 |

---

### 三种引擎模式对比

| 维度 | KV Cache 纯板端 | SplitNN OM | SplitNN 参数绑定 |
|------|:---:|:---:|:---:|
| 外部依赖 | 无 | CUDA 主机 + SSH 隧道 | CUDA 主机 + SSH 隧道 |
| 板端 NPU 执行 | 全部 24 层 | 前段 4 层 + 后段 4 层 | Embedding + LM Head |
| 支持模型 | 0.8B | 0.8B / 4B / 可扩展 | 2B / 可扩展 |
| 上下文 | 256 / 4096 | 256 / 16384 | 8192 |
| 板端 OM 大小 | 1.9 GB | ~2.8 GB（前后段合计） | tied_weight 970 MB |
| 部署复杂度 | 低 | 高 | 中 |
| 适用场景 | 入门、独立部署 | 大模型、长上下文 | 极致降低板端负载 |

---

## 快速开始

### 纯板端部署：Qwen3.5-0.8B KV Cache + OpenAI API（推荐入门）

这是最简部署路径——**无需 CUDA 主机、无需远程中段服务**，开发板独立运行完整的 OpenAI 兼容 API。

**前置条件：** 板端需预装 `fastapi`, `uvicorn`, `pydantic`, `transformers`, `tokenizers`, `jinja2`, `markupsafe`, `numpy`。详见下方 [环境准备](#1-环境准备)。

#### 1. 准备板端文件

```bash
# 上传控制器代码
sshpass -p 'Mind@123' scp -r -o StrictHostKeyChecking=no \
  controller/ \
  root@192.168.137.100:/root/slm_deploy/

# 上传 ModelSpec（无 torch 依赖的结构定义）
sshpass -p 'Mind@123' scp -o StrictHostKeyChecking=no \
  scripts/qwen35_model_spec.py \
  root@192.168.137.100:/root/slm_deploy/scripts/

# 上传 KV Cache OM 模型（约 1.9GB）
sshpass -p 'Mind@123' scp -o StrictHostKeyChecking=no \
  om_out/qwen3.5_kvcache_max256.om \
  root@192.168.137.100:/root/slm_deploy/

# 上传 Qwen3.5 模型配置（ModelSpec 读取 hidden_size/vocab_size 等用）
sshpass -p 'Mind@123' scp -o StrictHostKeyChecking=no \
  model/Qwen3.5-0.8B/config.json \
  root@192.168.137.100:/root/slm_deploy/

# 上传 tokenizer 文件（Qwen3.5 专用，勿与 Qwen3 混用）
sshpass -p 'Mind@123' scp -o StrictHostKeyChecking=no \
  model/Qwen3.5-0.8B/tokenizer.json \
  model/Qwen3.5-0.8B/tokenizer_config.json \
  model/Qwen3.5-0.8B/vocab.json \
  model/Qwen3.5-0.8B/merges.txt \
  model/Qwen3.5-0.8B/chat_template.jinja \
  root@192.168.137.100:/root/slm_deploy/

# 上传启动脚本（如首次部署）
sshpass -p 'Mind@123' scp -o StrictHostKeyChecking=no \
  board/run_openai_kvcache_controller.sh \
  root@192.168.137.100:/root/slm_deploy/
sshpass -p 'Mind@123' ssh -o StrictHostKeyChecking=no root@192.168.137.100 \
  'chmod +x /root/slm_deploy/run_openai_kvcache_controller.sh'
```

板端最终文件结构：

```text
/root/slm_deploy/
├── controller/                  # 控制器代码（完整目录）
│   ├── openai_controller.py
│   ├── modeling/kvcache_qwen35.py
│   ├── engine/om_engine.py
│   └── ...
├── scripts/qwen35_model_spec.py
├── qwen3.5_kvcache_max256.om    # KV Cache OM 模型（1.9GB）
├── config.json                  # Qwen3.5 模型配置
├── tokenizer.json, vocab.json, merges.txt, tokenizer_config.json, chat_template.jinja
└── run_openai_kvcache_controller.sh
```

#### 2. 启动控制器

```bash
# 登录开发板后执行
cd /root/slm_deploy

# 前台运行（Ctrl+C 停止）
bash run_openai_kvcache_controller.sh

# 或后台运行
nohup bash run_openai_kvcache_controller.sh > controller_kvcache.log 2>&1 &
tail -f controller_kvcache.log
```

> 模型加载约需 **210 秒**（1.9GB OM 从磁盘加载到 NPU），就绪后日志显示 `Uvicorn running on http://0.0.0.0:8000`。

#### 3. 测试 API

```bash
# 健康检查
curl http://127.0.0.1:8000/healthz

# 模型列表
curl http://127.0.0.1:8000/v1/models

# 非流式对话
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-0.8B-kvcache-om",
    "messages": [{"role": "user", "content": "你好，介绍一下你自己"}],
    "max_tokens": 64,
    "temperature": 0.7,
    "stream": false
  }'

# 流式对话
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-0.8B-kvcache-om",
    "messages": [{"role": "user", "content": "请用中文写一首关于春天的四句诗"}],
    "max_tokens": 128,
    "temperature": 0.4,
    "top_p": 0.85,
    "top_k": 20,
    "repetition_penalty": 1.08,
    "stream": true
  }'
```

#### 4. 支持的解码参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `temperature` | float | 0.7 | 温度，0 = 贪心解码 |
| `top_k` | int | 40 | Top-K 采样 |
| `top_p` | float | 1.0 | Top-P (nucleus) 采样 |
| `repetition_penalty` | float | 1.0 | 重复惩罚（1.0 = 无惩罚） |
| `presence_penalty` | float | 0.0 | 存在惩罚 |
| `max_tokens` | int | 64 | 最大生成 token 数（上限 1024） |
| `stream` | bool | false | 是否流式 SSE 返回 |
| `stop` | str/list | — | 停止词 |
| `enable_thinking` | bool | false | 是否启用 `<think>` 推理链 |

#### 5. 注意事项

- **启动前确认 NPU 状态**：`npu-smi info` 显示 `Health: OK`，否则需 `reboot`
- **异常退出后 NPU 变 Alarm**：ACL 进程被 kill 后驱动不自动清理，必须重启开发板
- **预填充耗时**：首次请求的 prefill 阶段按 prompt 长度需要数十秒（NPU 逐 token 执行），之后 decode 阶段约 200ms/tok
- **上下文窗口**：当前 OM 模型为 `max_len=256`。板端另有 `qwen3.5_kvcache_max4096.om`（2.0GB）可支持 4096 窗口
- **与其他方案的区别**：本路径**不依赖**任何外部 CUDA 服务器或 SSH 隧道，所有推理在 NPU 上完成

### 新增：Qwen3.5-2B SplitNN 参数绑定链路

这条链路的目标是以尽可能小的板端改动支持参数绑定：

- 主机：
  - 运行 `Qwen3.5-2B` 的全部 24 层 Transformer 主干
  - 提供中段 HTTP 服务
- 开发板：
  - 运行 OpenAI 兼容控制器
  - `run_prefix()` 执行 embedding lookup
  - `run_suffix()` 执行 tied `lm_head`

控制器外部接口不变，仍然使用：

```text
POST /v1/chat/completions
GET  /v1/models
GET  /healthz
```

### 1. 准备 2B 参数绑定资产

在主机上导出板端需要的共享权重与元数据：

```bash
cd /home/CX_Li/EF_clean

pixi run python scripts/export_qwen35_bound_embed_head.py \
  --model-path model_dl/Qwen3.5-2B \
  --output-dir qwen3.5_2b_bound_embed_head \
  --split 0,24
```

导出目录至少包含：

```text
qwen3.5_2b_bound_embed_head/
├── tied_weight.bin
├── final_norm_weight.bin
└── bound_embed_head.metadata.json
```

### 2. 编译板端 `lm_head` 单算子模型

参数绑定模式下，板端 `run_suffix()` 默认可退回 CPU `numpy`，但为了可用速度，实际部署时应为真实 head shape 编译 ACL single-op `MatMul`：

```bash
mkdir -p tmp_singleop_matmul_qwen35_2b_head/run/out/test_data/config
cat > tmp_singleop_matmul_qwen35_2b_head/run/out/test_data/config/acl_op.json <<'EOF'
[
  {
    "op": "MatMul",
    "input_desc": [
      {"format": "ND", "type": "float16", "shape": [1, 2048]},
      {"format": "ND", "type": "float16", "shape": [248320, 2048]}
    ],
    "output_desc": [
      {"format": "ND", "type": "float16", "shape": [1, 248320]}
    ],
    "attr": [
      {"name": "transpose_x1", "type": "bool", "value": false},
      {"name": "transpose_x2", "type": "bool", "value": true}
    ]
  }
]
EOF
```

使用现有 CANN 7 容器编译：

```bash
podman run --rm --network=host --http-proxy=false \
  -e http_proxy= -e https_proxy= -e HTTP_PROXY= -e HTTPS_PROXY= \
  -v /home/CX_Li/EF_clean:/workspace:Z \
  -w /workspace/tmp_singleop_matmul_qwen35_2b_head/run/out \
  localhost/cann-atc-rocky:v7 \
  bash -lc 'atc --singleop=test_data/config/acl_op.json \
    --soc_version=Ascend310B4 \
    --output=op_models'
```

编译结果示例：

```text
op_models/0_MatMul_1_2_1_2048_1_2_248320_2048_1_2_1_248320.om
```

### 3. 同步到开发板

```bash
sshpass -p 'Mind@123' ssh -F /dev/null -o StrictHostKeyChecking=no \
  root@192.168.137.100 'mkdir -p /root/slm_deploy/qwen3.5_2b_bound_embed_head/op_models'

sshpass -p 'Mind@123' scp -F /dev/null -o StrictHostKeyChecking=no \
  qwen3.5_2b_bound_embed_head/* \
  root@192.168.137.100:/root/slm_deploy/qwen3.5_2b_bound_embed_head/

sshpass -p 'Mind@123' scp -F /dev/null -o StrictHostKeyChecking=no \
  tmp_singleop_matmul_qwen35_2b_head/run/out/op_models/* \
  root@192.168.137.100:/root/slm_deploy/qwen3.5_2b_bound_embed_head/op_models/

sshpass -p 'Mind@123' scp -F /dev/null -o StrictHostKeyChecking=no \
  controller/openai_split_controller.py \
  controller/orchestrator.py \
  controller/remote_middle.py \
  controller/schemas.py \
  controller/engine/om_engine.py \
  scripts/qwen35_model_spec.py \
  board/run_openai_split_controller_bound_2b.sh \
  root@192.168.137.100:/root/slm_deploy/
```

如果板端目录结构已经和仓库同步过，只需要额外更新：

- `controller/engine/om_engine.py`
- `controller/openai_split_controller.py`
- `controller/orchestrator.py`
- `scripts/qwen35_model_spec.py`
- `board/run_openai_split_controller_bound_2b.sh`
- `qwen3.5_2b_bound_embed_head/`

### 4. 启动主机侧中段服务

```bash
cd /home/CX_Li/EF_clean

pixi run python server/qwen35_split_service.py \
  --host 127.0.0.1 \
  --port 18080 \
  --model-path model_dl/Qwen3.5-2B \
  --split 0,24 \
  --device cuda:0 \
  --max-len 8192 \
  --session-timeout-sec 60 \
  --max-sessions 1
```

推荐保留 `--max-sessions 1`。当前这条 `0/24/0` 参数绑定链路默认按“单开发板控制器 + 单主机中段服务”部署，额外并发只会放大 session cache 与 HTTP 请求处理开销。

再建立反向 SSH 映射，使开发板通过自身 `127.0.0.1:28080` 访问主机 `18080`：

```bash
sshpass -p 'Mind@123' ssh -F /dev/null \
  -o StrictHostKeyChecking=no \
  -o ExitOnForwardFailure=yes \
  -N -R 28080:127.0.0.1:18080 \
  root@192.168.137.100
```

### 5. 启动开发板 OpenAI 控制器

登录开发板后执行：

```bash
cd /root/slm_deploy
./run_openai_split_controller_bound_2b.sh
```

后台运行：

```bash
cd /root/slm_deploy
nohup ./run_openai_split_controller_bound_2b.sh >/tmp/bound_controller.log 2>&1 &
tail -f /tmp/bound_controller.log
```

### 6. 手动测试 OpenAI 接口

健康检查：

```bash
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/v1/models
```

非流式：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-2b-split-0-24-0-om",
    "messages": [
      {"role": "user", "content": "请用三句话介绍一下你自己"}
    ],
    "temperature": 0.0,
    "max_tokens": 64,
    "stream": false
  }'
```

流式：

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-2b-split-0-24-0-om",
    "messages": [
      {"role": "user", "content": "你好，请简单和我打个招呼"}
    ],
    "temperature": 0.0,
    "max_tokens": 32,
    "stream": true
  }'
```

如果想观察纯 SSE 数据，可以再接一段：

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-2b-split-0-24-0-om",
    "messages": [
      {"role": "user", "content": "你好，请简单和我打个招呼"}
    ],
    "temperature": 0.0,
    "max_tokens": 32,
    "stream": true
  }' | sed -n 's/^data: //p'
```

开启 thinking：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-2b-split-0-24-0-om",
    "messages": [
      {"role": "user", "content": "请先认真思考，再简短回答：1+1为什么等于2？"}
    ],
    "temperature": 0.0,
    "max_tokens": 128,
    "stream": false,
    "enable_thinking": true
  }'
```

`Qwen3.5-2B` 当前部署默认不建议开启 thinking。实际联调中发现 2B thinking 模式容易出现长时间思维链循环、乱码和收尾失败；控制器当前会直接拒绝这一路径，建议显式使用 `enable_thinking=false`。

### 6.1 断连与清理行为

当前控制器已补充“请求方断连后自动取消”的处理：

- 流式请求如果前端程序崩溃、浏览器标签页关闭、或命令行 `Ctrl+C` 中断，控制器会在当前 token step 结束后停止生成
- 非流式请求如果客户端提前断开，也会设置取消标志并尽快退出生成循环
- 退出时会自动执行远端 `session/close`，并调用本地 `engine.end_session()`

需要注意的是，取消粒度仍是“当前 step 完成后”而不是立即抢占。也就是说，如果控制器正阻塞在一次 `remote_middle.step()` 中，仍需等这一小步返回后才能进入清理逻辑。

### 7. 当前状态

目前这条 `Qwen3.5-2B split 0/24/0` 参数绑定链路已经完成以下验证：

- 开发板控制器可正常启动，不再卡在 `Waiting for application startup`
- 非流式 OpenAI 兼容请求可正常返回文本
- 流式 OpenAI 兼容请求可完整返回 SSE chunk 与 `[DONE]`
- 客户端中途断开后，控制器不会再无限继续生成；会在当前 step 完成后自动关闭远端 session
- 板端实际生成路径为：
  - `run_prefix()`：共享 tied weight 的 embedding lookup
  - `remote_middle.step()`：主机侧全部 24 层 Transformer 主干
  - `run_suffix()`：开发板 ACL single-op `MatMul` 执行 tied `lm_head`
- 主机侧中段服务已补充以下保护：
  - session 超时回收与显式 cache 释放
  - `CUDA_OOM` 时主动回收故障 session
  - `max_sessions` 并发限制
  - 单线程 `HTTPServer`，避免 `/v1/session/step` 高频请求持续创建线程
- 主机侧注意力缓存更新已改为原地写回，不再每个 token 重建整块 K/V cache
- GQA 计算已移除“物理复制 KV 头”的实现，改为按组直接计算，降低系统内存与显存抖动

### 7.1 最近修复摘要

这次新增的修复主要集中在两条线上：

1. 控制器稳定性
   - 支持客户端断连后的自动取消与 session 清理
   - `Qwen3.5-2B` thinking 模式显式禁用，避免已知的长思维链循环问题

2. 主机侧中段服务内存稳定性
   - session 生命周期改为显式释放 cache，而不是仅依赖字典删除
   - `Qwen3_5Attention` 的 K/V 更新改为原地 `copy_()` 写回
   - GQA 不再把 2 个 KV heads 扩成 8 个 query heads 的物理副本
   - HTTP 服务从 `ThreadingHTTPServer` 改为 `HTTPServer`，避免每 token step 产生线程级系统内存膨胀

### 1. 环境准备

#### 开发机 (x86_64)

```bash
cd EF && pixi install

# 下载模型权重
hf download Qwen/Qwen3-0.6B --local-dir model/Qwen3-0.6B
hf download Qwen/Qwen3.5-0.8B --local-dir model/Qwen3.5-0.8B

# 下载 CANN 7.0.0 安装包到 docker/ 目录
wget -O docker/Ascend-cann-toolkit_7.0.0_linux-x86_64.run \   # ATC 编译器 (~1.6GB)
  "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%207.0.0/Ascend-cann-toolkit_7.0.0_linux-x86_64.run"
wget -O docker/Ascend-cann-kernels-310b-7.0.0-linux.noarch.rpm \  # 310B 内核 (~351MB, 必须用 310B 而非 310P)
  "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%207.0.0/Ascend-cann-kernels-310b-7.0.0-linux.noarch.rpm"

# 构建 ATC 容器（需 --network=host 确保容器内 dnf/pip 可联网）
podman build --network=host -t localhost/cann-atc-rocky:v7 \
    -f docker/Containerfile.v2-cann7 docker/
```

#### 下载板端 Python wheel（在有网络的开发机上执行）

板端推理脚本实际只依赖 `numpy` + `transformers`。**必须用 `--no-deps` 分别下载**，原因见下方说明。

```bash
mkdir -p tmp

# 主要依赖
python3 -m pip download --dest tmp --platform manylinux2014_aarch64 \
    --python-version 310 --implementation cp --abi cp310 --only-binary=:all: --no-deps \
    "numpy==1.26.4" "transformers==4.57.6" "tokenizers==0.22.2" \
    "huggingface_hub==0.34.6" "safetensors" "requests" "pyyaml" "regex" "tqdm" \
    "filelock" "fsspec" "packaging" "typing-extensions" \
    "certifi" "charset-normalizer" "idna" "urllib3"

# jinja2 + markupsafe：apply_chat_template(enable_thinking=False) 需要
python3 -m pip download --dest tmp --platform manylinux2014_aarch64 \
    --python-version 310 --implementation cp --abi cp310 --only-binary=:all: --no-deps \
    "markupsafe"
python3 -m pip download --dest tmp --platform any \
    --python-version 310 --only-binary=:all: --no-deps \
    "jinja2"

# SplitNN 控制器额外依赖（板端 OpenAI API）
mkdir -p tmp/board_controller_wheels
python3 -m pip download --dest tmp/board_controller_wheels --platform manylinux2014_aarch64 \
    --python-version 310 --implementation cp --abi cp310 --only-binary=:all: --no-deps \
    "fastapi" "uvicorn" "pydantic" "pydantic-core" "starlette" \
    "annotated-types" "typing-inspection" "anyio" "h11" "click" \
    "sniffio" "exceptiongroup"

# pip 自举：板端出厂无 pip，且板端 pip.conf 指向不可达的豆瓣镜像
python3 -m pip download --dest tmp --platform manylinux2014_aarch64 \
    --python-version 310 --implementation cp --abi cp310 --only-binary=:all: --no-deps \
    "pip" "setuptools" "wheel"

curl -L https://bootstrap.pypa.io/get-pip.py -o tmp/get-pip.py
```

> **为什么 `--no-deps`？** `huggingface-hub>=0.34` 在 aarch64 平台要求 `hf-xet>=1.1.3`，但 PyPI 上 `hf-xet` 的 aarch64 wheel 最高只有 `0.1.x`。`hf-xet` 仅用于 HuggingFace Hub 并行下载加速，板端只通过 `AutoTokenizer.from_pretrained(local_dir)` 读取本地 tokenizer 文件，以 `--no-deps` 安装 huggingface-hub 安全可行。

### 2. 导出 ONNX → 编译 OM

```bash
# 导出 ONNX（可自定义 --max-len）
pixi run python scripts/export_qwen3_kvcache.py --max-len 256 \
    --output om_out/qwen3_kvcache_max256.onnx

pixi run python scripts/export_qwen35_kvcache.py --max-len 256 \
    --output om_out/qwen3.5_kvcache_max256.onnx

# ATC 编译（统一命令，自动读取 ONNX 的 input shape）
# 注意：INPUT_SHAPE 值包含分号，必须先 export 再运行，不能内联展开
INPUT_SHAPE=$(pixi run python scripts/gen_input_shape.py om_out/qwen3.5_kvcache_max256.onnx)
export INPUT_SHAPE MODEL_ONNX="om_out/qwen3.5_kvcache_max256.onnx" OUTPUT_PREFIX="om_out/qwen3.5_kvcache_max256"
bash scripts/podman_convert.sh
```

可选参数：

| 参数 | 说明 |
|------|------|
| `--max-len` | KV Cache 上下文长度（默认 256，可选 128/512/1024 等） |
| `--output` | ONNX 输出路径 |
| `IMAGE=` | ATC 容器镜像（默认 `cann-atc-rocky:v7`） |
| `OUTPUT_PREFIX=` | OM 输出路径前缀 |

### 3. 板端环境

开发板为 Ubuntu 22.04 aarch64，出厂预装 CANN 7.0.RC1。板端 `/root/.pip/pip.conf` 配置了豆瓣镜像（无外网不可达），需用 pip wheel 自举安装。

#### 传输文件

```bash
sshpass -p 'Mind@123' ssh root@192.168.137.100 'mkdir -p /root/slm_deploy/wheels'
sshpass -p 'Mind@123' scp tmp/*.whl tmp/get-pip.py root@192.168.137.100:/root/slm_deploy/wheels/
sshpass -p 'Mind@123' ssh root@192.168.137.100 'mkdir -p /root/slm_deploy/board_controller_wheels'
sshpass -p 'Mind@123' scp tmp/board_controller_wheels/*.whl \
    root@192.168.137.100:/root/slm_deploy/board_controller_wheels/
```

#### 在开发板上安装

```bash
cd /root/slm_deploy/wheels

# 1. 用 pip wheel 自举安装 pip（get-pip.py 因镜像不可达会失败）
python3 -c "
import zipfile, sys, os
whl = [f for f in os.listdir('.') if f.startswith('pip-')][0]
zf = zipfile.ZipFile(whl)
zf.extractall('/tmp/_pip')
sys.path.insert(0, '/tmp/_pip')
import pip._internal
pip._internal.main(['install', '--no-deps', '--no-index', '--force-reinstall', whl])
"

# 2. 基础库（无上层依赖）
python3 -m pip install --no-deps --no-index --find-links=. \
    numpy-*.whl typing_extensions-*.whl packaging-*.whl filelock-*.whl \
    fsspec-*.whl tqdm-*.whl regex-*.whl safetensors-*.whl

# 3. huggingface-hub（--no-deps 跳过 hf-xet）
python3 -m pip install --no-deps --no-index --find-links=. \
    huggingface_hub-0.34.6-*.whl

# 4. requests → tokenizers → transformers
python3 -m pip install --no-deps --no-index --find-links=. \
    requests-*.whl charset_normalizer-*.whl tokenizers-*.whl transformers-*.whl

# 5. jinja2 + markupsafe（chat template 模板渲染需要）
python3 -m pip install --no-deps --no-index --find-links=. \
    markupsafe-*.whl jinja2-*.whl

# 6. 放宽 tokenizers 版本上限
sed -i 's/tokenizers>=0.22.0,<=0.23.0/tokenizers>=0.22.0,<=0.23.1/' \
  /usr/local/lib/python3.10/dist-packages/transformers/dependency_versions_table.py

# 7. 验证
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 -c "import acl, numpy, transformers, tokenizers, huggingface_hub, safetensors, jinja2; print('OK')"

# 8. SplitNN 控制器额外依赖
cd /root/slm_deploy/board_controller_wheels
python3 -m pip install --no-deps --no-index --find-links=. \
    annotated_types-*.whl exceptiongroup-*.whl sniffio-*.whl anyio-*.whl \
    click-*.whl h11-*.whl typing_inspection-*.whl pydantic_core-*.whl \
    pydantic-*.whl starlette-*.whl uvicorn-*.whl fastapi-*.whl

# 9. 控制器依赖验证
python3 -c "import fastapi, uvicorn, pydantic; print('controller deps OK')"
```

> **关于 `hf-xet`**：`huggingface-hub>=0.34` 在 aarch64 上依赖 `hf-xet>=1.1.3`，但 PyPI 上 `hf-xet` 仅 `0.1.x` 提供 aarch64 wheel。`hf-xet` 仅用于 HuggingFace Hub 并行下载，板端只通过 `from_pretrained(local_dir)` 读本地 tokenizer 文件，不会触发下载路径。

可选：关闭桌面服务释放 ~120MB 内存：

```bash
systemctl stop sddm && systemctl disable sddm
pkill -f xfce4-power-manager
pkill -f xfce4-notifyd
pkill -f tumblerd
```

### 4. 部署并运行

从开发机传输文件：

```bash
# Qwen3 KV Cache
scp om_out/qwen3_kvcache_max256_cann7.om root@192.168.137.100:/root/slm_deploy/
scp board/gen_text_qwen3_kvcache.py          root@192.168.137.100:/root/slm_deploy/

# Qwen3.5 KV Cache（tokenizer 与 Qwen3 不兼容，需单独传输）
scp om_out/qwen3.5_kvcache_max256.om         root@192.168.137.100:/root/slm_deploy/
scp board/gen_text_qwen35_kvcache.py          root@192.168.137.100:/root/slm_deploy/
scp model/Qwen3.5-0.8B/tokenizer.json model/Qwen3.5-0.8B/tokenizer_config.json \
    model/Qwen3.5-0.8B/chat_template.jinja     root@192.168.137.100:/root/slm_deploy/
```

> **tokenizer 兼容性**：Qwen3 使用 `vocab.json` + `merges.txt`，Qwen3.5 使用 `tokenizer.json`。两者互不兼容，部署时注意不要互相覆盖。

在开发板上运行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /root/slm_deploy

# Qwen3 KV Cache
python3 gen_text_qwen3_kvcache.py \
    --model qwen3_kvcache_max256_cann7.om --prompt "你好" --max-tokens 50

# Qwen3.5 KV Cache
python3 gen_text_qwen35_kvcache.py \
    --model qwen3.5_kvcache_max256.om --tokenizer-dir /root/slm_deploy \
    --prompt "你好" --max-tokens 50
```

## SplitNN 通用化架构

SplitNN 体系现已从硬编码的 `4/16/4` 切分演进为**参数化架构**，支持：

- **任意 Qwen3.5 模型尺寸**（0.8B / 2B / 4B / 9B / 27B）
- **自定义上下文长度**（256 / 16K / 任意）
- **自定义切分方案**（通过 `--split prefix_end,suffix_start` 指定）
- **Thinking 开关**（`enable_thinking` 选项透传至 chat template）

### 核心组件

| 组件 | 路径 | 作用 |
|------|------|------|
| `ModelSpec` | `scripts/qwen35_model_spec.py` | 从 `config.json` 动态读取所有架构参数 |
| `SplitConfig` | `scripts/qwen35_model_spec.py` | 切分点参数化，自动计算各段 DN/GA 层数 |
| `metadata.json` | 导出时附带 | 板端无 PyTorch，通过 JSON 获取模型参数 |

### 使用 `--split`

```bash
# 0.8B 经典 4/16/4 切分（默认）
pixi run python scripts/export_qwen35_split_prefix.py --split 4,20 --max-len 256

# 4B 模型，前后各 1 层（板端内存最优）
pixi run python scripts/export_qwen35_split_prefix.py \
  --model-path model_dl/Qwen3.5-4B --split 1,31 --max-len 16384
```

## SplitNN 原型（板端 + CUDA 主机）

首版 SplitNN 使用 `4 / 16 / 4` 切分（Qwen3.5-0.8B），现扩展为通用参数化体系：
- 开发板运行前段和后段（两个 OM 或 ONNX）
- CUDA 主机运行中间段（PyTorch）
- 板端与主机通过 `HTTP/1.1 + application/octet-stream` 传输 `(1,1,hidden_size)` 的 `fp16 hidden state`

### 设计动机

引入 SplitNN 的原因不是单纯为了"分布式"，而是为了把不同硬件各自擅长的部分拆开：

- 开发板负责前后段，尽量贴近最终端侧部署形态
- 主机负责中间段的大部分计算量，减轻板端算力和内存压力
- 两端之间只传输单步 hidden state，而不传完整 cache，从而把网络负担控制在每 token 约 `hidden_size × 2` 字节往返

对 Qwen3.5 而言，`full_attention_interval=4` 的周期结构使得切分可以灵活选择边界——`4 / 16 / 4`（0.8B 经典）、`1 / 30 / 1`（4B 内存最优）等多种方案均可。

### 系统职责划分

SplitNN 中一共分成三部分：

1. **前段（prefix）**
   - 输入：`token_id + position + 前段 cache`
   - 输出：`hidden_state`
   - 运行位置：开发板 OM 或开发机 ONNX

2. **中段（middle）**
   - 输入：`hidden_state + position + 中段 cache`
   - 输出：`hidden_state`
   - 运行位置：主机 `server/qwen35_split_service.py`

3. **后段（suffix）**
   - 输入：`hidden_state + position + 后段 cache`
   - 输出：`logits`
   - 运行位置：开发板 OM 或开发机 ONNX

其中：
- **前后段 cache** 留在本地执行端
- **中段 cache** 留在远端 server
- 网络上传输的只有 `(1, 1, hidden_size)` 的 `fp16 hidden state`

### 原型验证状态

在引入控制器之前，原始 SplitNN 原型已经完成了三层验证：

1. **纯 PyTorch reference 对齐**
   - `prefix -> middle -> suffix`
   - 与完整 `Qwen3.5 KV Cache` 模型逐 token 对齐

2. **前后段 ONNX 多步 ORT 校验**
   - 验证 prefix hidden 和 suffix logits 的数值正确性

3. **本地模拟联调**
   - `ONNX prefix/suffix + middle server`
   - 已经可以按真实 prompt 生成并解码出正常中文文本

因此，控制器并不是从零开始设计，而是建立在“SplitNN 原型本身已经可工作”的基础上。

### 1. 导出前段 / 后段 ONNX

```bash
# 0.8B 经典 4/16/4 切分（默认）
pixi run python scripts/export_qwen35_split_prefix.py \
    --max-len 256 --output om_out/qwen3.5_split_prefix_max256.onnx

pixi run python scripts/export_qwen35_split_suffix.py \
    --max-len 256 --output om_out/qwen3.5_split_suffix_max256.onnx

# 4B 模型 1/30/1 切分 + 16K 上下文
pixi run python scripts/export_qwen35_split_prefix.py \
    --model-path model_dl/Qwen3.5-4B --split 1,31 --max-len 16384 \
    --output om_out/qwen3.5_4b_split_prefix_max16384.onnx

pixi run python scripts/export_qwen35_split_suffix.py \
    --model-path model_dl/Qwen3.5-4B --split 1,31 --max-len 16384 \
    --output om_out/qwen3.5_4b_split_suffix_max16384.onnx
```

本地参考链路校验：

```bash
pixi run python scripts/validate_qwen35_split_reference.py
```

导出后 ORT 多步校验：

```bash
pixi run python scripts/validate_qwen35_split_ort.py \
    --prefix-onnx om_out/qwen3.5_split_prefix_max256.onnx \
    --suffix-onnx om_out/qwen3.5_split_suffix_max256.onnx
```

然后分别用现有 `scripts/gen_input_shape.py` + `scripts/podman_convert.sh` 编译成 `.om`。

### 2. 启动 CUDA 主机服务

```bash
# 0.8B 经典 4/16/4
pixi run python server/qwen35_split_service.py \
    --host 0.0.0.0 --port 18080 \
    --model-path model/Qwen3.5-0.8B \
    --device cuda:0 --max-len 16384

# 4B 1/30/1 切分 + 16K 上下文
pixi run python server/qwen35_split_service.py \
    --host 0.0.0.0 --port 18080 \
    --model-path model_dl/Qwen3.5-4B \
    --split 1,31 --device cuda:0 --max-len 16384
```

健康检查：

```bash
curl http://<server-ip>:18080/v1/health
```

### 3. 开发板运行 SplitNN

将以下文件传到板端 `/root/slm_deploy/`：
- `om_out/qwen3.5_split_prefix_max256.om`
- `om_out/qwen3.5_split_suffix_max256.om`
- `om_out/qwen3.5_split_prefix_max16384.om`
- `om_out/qwen3.5_split_suffix_max16384.om`
- `board/gen_text_qwen35_splitnn.py`
- `board/run_qwen35_splitnn.sh`
- `board/run_qwen35_splitnn_16k.sh`
- `model/Qwen3.5-0.8B/tokenizer.json`
- `model/Qwen3.5-0.8B/tokenizer_config.json`
- `model/Qwen3.5-0.8B/chat_template.jinja`

短上下文（256）运行：

```bash
python3 -u /root/slm_deploy/gen_text_qwen35_splitnn.py \
    --server-url http://<server-ip>:18080 \
    --prefix-model /root/slm_deploy/qwen3.5_split_prefix_max256.om \
    --suffix-model /root/slm_deploy/qwen3.5_split_suffix_max256.om
```

长上下文（16K）运行：

```bash
cd /root/slm_deploy
./run_qwen35_splitnn_16k.sh
```

> `run_qwen35_splitnn_16k.sh` 默认使用 `qwen3.5_split_prefix_max16384.om`、`qwen3.5_split_suffix_max16384.om` 和 `--max-len 16384`。

## SplitNN 控制器（OpenAI API）

在原始 SplitNN 原型之上，仓库额外提供了一个“控制器中间层”：
- 对外提供 OpenAI 兼容的 `/v1/chat/completions`
- 对内管理 tokenizer、chat template、采样、生成循环
- 前后段通过统一引擎接口切换：
  - `OnnxSplitEngine`：开发机仿真
  - `OmSplitEngine`：开发板部署
- 中间 16 层继续通过 `server/qwen35_split_service.py` 远端执行

### 控制器架构

```
OpenAI Client
    |
    v
controller/openai_split_controller.py
    |
    +-- orchestrator.py      # messages -> prompt -> prefill/decode -> OpenAI 响应
    +-- remote_middle.py     # 与 middle server 的 open/step/close 协议
    +-- engine/onnx_engine.py
    `-- engine/om_engine.py
```

### 控制器特点

- **无状态多轮**：每次请求都基于完整 `messages` 重新 prefill，不跨请求复用 cache
- **前后段 cache**：由本地引擎实例内部管理
- **中段 cache**：由远端 middle server 按 `session_id` 管理
- **流式输出**：支持 `stream=true` 的 SSE 响应
- **Thinking 开关**：支持 `enable_thinking` 参数控制思考模式
- **灵活切分**：通过 `--split` / `--model-path` 适配不同模型尺寸和切分方案

### 开发机启动（ONNX 后端）

先启动 middle server：

```bash
# 4B 1/30/1 长上下文
pixi run python server/qwen35_split_service.py \
    --host 127.0.0.1 --port 18080 \
    --model-path model_dl/Qwen3.5-4B \
    --split 1,31 --device cuda:0 --max-len 16384
```

再启动控制器：

```bash
# 4B 1/30/1 + 16K ONNX 后端
pixi run python controller/openai_split_controller.py \
    --host 127.0.0.1 --port 8000 \
    --engine onnx \
    --model-path model_dl/Qwen3.5-4B \
    --model-name qwen3.5-4b-split-1-30-1-onnx \
    --remote-model-name "Qwen3.5-2560B-split-1-30-1" \
    --split 1,31 --max-len 16384 \
    --server-url http://127.0.0.1:18080 \
    --prefix-onnx om_out/qwen3.5_4b_split_prefix_max16384.onnx \
    --suffix-onnx om_out/qwen3.5_4b_split_suffix_max16384.onnx
```

### 开发板启动（OM 后端）

#### 板端文件布局

板端 `/root/slm_deploy/` 需按以下结构组织，确保模块间相互引用正确：

```
/root/slm_deploy/
├── gen_text_qwen35_splitnn.py     # 板端推理入口（独立脚本）
├── scripts/                        # Python 包（需 __init__.py）
│   ├── __init__.py
│   └── qwen35_model_spec.py       # ModelSpec/SplitConfig/load_metadata
├── controller/
│   ├── __init__.py
│   └── engine/
│       ├── __init__.py
│       ├── base.py                 # SplitEngine 抽象基类
│       └── om_engine.py            # OmSplitEngine（ACL NPU 推理）
├── qwen3.5_split_prefix_max256.om         # OM 模型文件
├── qwen3.5_split_prefix_max256.metadata.json   # 配套元数据
├── qwen3.5_split_suffix_max256.om
├── qwen3.5_split_suffix_max256.metadata.json
├── tokenizer.json                 # Qwen3.5 tokenizer（与 Qwen3 不兼容）
├── tokenizer_config.json
└── chat_template.jinja
```

**关键引用链：**
- `gen_text_qwen35_splitnn.py` → `qwen35_model_spec.load_metadata()` → 从 `.metadata.json` 读取模型参数（无需 PyTorch）
- `gen_text_qwen35_splitnn.py` → `controller.engine.om_engine.OmSplitEngine` → ACL 管理 .om 文件
- `controller/engine/base.py` → `scripts.qwen35_model_spec.ModelSpec`（需 `scripts/__init__.py`）

**注意：**
- `scripts/`、`controller/`、`controller/engine/` 都必须有 `__init__.py`（可为空文件），否则 Python 包导入失败
- `.metadata.json` 文件名需与对应 `.om` 前缀一致（如 `qwen3.5_split_prefix_max256.om` → `...metadata.json`）
- 首次运行前需 `source /usr/local/Ascend/ascend-toolkit/set_env.sh` 加载 ACL 环境

#### 传输文件

```bash
# 核心代码
sshpass -p 'Mind@123' scp board/gen_text_qwen35_splitnn.py \
    root@192.168.137.100:/root/slm_deploy/
sshpass -p 'Mind@123' scp scripts/qwen35_model_spec.py \
    root@192.168.137.100:/root/slm_deploy/scripts/
sshpass -p 'Mind@123' scp controller/__init__.py \
    root@192.168.137.100:/root/slm_deploy/controller/
sshpass -p 'Mind@123' scp controller/engine/{__init__.py,base.py,om_engine.py} \
    root@192.168.137.100:/root/slm_deploy/controller/engine/

# OM 模型 + 元数据
sshpass -p 'Mind@123' scp om_out/qwen3.5_split_prefix_max256.om \
    root@192.168.137.100:/root/slm_deploy/
sshpass -p 'Mind@123' scp om_out/qwen3.5_split_prefix_max256.metadata.json \
    root@192.168.137.100:/root/slm_deploy/
sshpass -p 'Mind@123' scp om_out/qwen3.5_split_suffix_max256.om \
    root@192.168.137.100:/root/slm_deploy/
sshpass -p 'Mind@123' scp om_out/qwen3.5_split_suffix_max256.metadata.json \
    root@192.168.137.100:/root/slm_deploy/

# Tokenizer
sshpass -p 'Mind@123' scp model/Qwen3.5-0.8B/{tokenizer.json,tokenizer_config.json,chat_template.jinja} \
    root@192.168.137.100:/root/slm_deploy/
```

#### 建立网络隧道并运行

如果开发板无法直接访问主机 `18080` 端口，先在开发机建立反向隧道：

```bash
# 开发机执行（将板端 28080 转发到本机 18080）
sshpass -p 'Mind@123' ssh -o StrictHostKeyChecking=no -N \
  -R 28080:127.0.0.1:18080 root@192.168.137.100 &
```

板端运行（通过隧道 `127.0.0.1:28080` 访问中间服务）：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /root/slm_deploy
python3 gen_text_qwen35_splitnn.py \
    --server-url http://127.0.0.1:28080 \
    --prefix-model /root/slm_deploy/qwen3.5_split_prefix_max256.om \
    --suffix-model /root/slm_deploy/qwen3.5_split_suffix_max256.om \
    --tokenizer-dir /root/slm_deploy \
    --prompt "你好" --max-tokens 50
```

### 当前验证状态

- `4 / 16 / 4` SplitNN 原型本身已通过 reference / ORT / 本地模拟三层验证
- **SplitNN 通用化完成**：通过 `ModelSpec` + `SplitConfig` 支持任意模型尺寸和切分方案
- **4B 模型 1/30/1 切分 16K 上下文**本地 ONNX 联调通过（非流式 + 流式 + 多轮 + thinking）
- **端边协同实测数据**：

| 模型 | 切分 | 上下文 | Prefill | Decode | 说明 |
|------|------|--------|---------|--------|------|
| Qwen3.5-0.8B | 4/16/4 | 256 | 13 tok / 3.8s | 1.8 tok/s | 短问答 + thinking |
| Qwen3.5-4B | 1/30/1 | 16K | 13 tok / 63s | 0.3 tok/s | 板端 1GA层 是瓶颈 |
| Qwen3.5-4B | 1/30/1 | 16K | 319 tok | — | 长上下文 ✓ |
| **Qwen3.5-4B** | **0/32/0** | **16K** | **13 tok / 5.1s** | **1.2 tok/s** | **GA offload → 4x 提速** |
| **Qwen3.5-4B** | **0/32/0** | **16K** | **419 tok** | — | 长上下文 ✓ |

- **0/32/0 切分**：板端仅 embedding + lm_head（无任何注意力层），所有 32 层 GA/DN 在主机 GPU 执行，decode 从 0.3 提升到 1.2 tok/s
- 瓶颈已从板端 NPU GA 注意力转移到服务器 GPU 全模型推理 + 网络延迟
- **板端脚本**已重构为复用 `OmSplitEngine`，通过 `metadata.json` 获取模型参数
- `max_len=16384` 的 CUDA middle server 已完成实际测速，单 token 中段吞吐约 `9 tok/s`
- 开发板 OM 后端的 SSE 流式收尾仍待继续排查，当前建议优先使用非流式接口做板端联调

两种推理脚本：

| 脚本 | 对应模型 | 关键参数 |
|------|---------|---------|
| `gen_text_qwen3_kvcache.py` | Qwen3 KV Cache | `--model X.om --prompt` |
| `gen_text_qwen35_kvcache.py` | Qwen3.5 KV Cache | `--model X.om --tokenizer-dir /root/slm_deploy` |

---

## 相关文档

- [REPORT.md](./REPORT.md) — 详细实验报告（设计思路、踩坑记录、性能优化）
- [AGENTS.md](./AGENTS.md) — AI 辅助开发速查
