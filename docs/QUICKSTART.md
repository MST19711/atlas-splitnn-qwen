# 快速开始

本文档提供从零开始完成 Qwen3.5-0.8B 纯板端部署的最简路径。

---

## 前置条件

### x86 开发机

- Python 3.10 (pixi 管理)
- Podman (ATC 编译容器)
- CUDA GPU (SplitNN 方案需要)

### 模型下载

项目默认不包含模型权重。若要运行模型，先用 pixi 环境里的 `hfcli` 下载到约定目录：

```bash
# 0.8B
pixi run hf download Qwen/Qwen3.5-0.8B \
  --local-dir model/Qwen3.5-0.8B

# 2B
pixi run hf download Qwen/Qwen3.5-2B \
  --local-dir model_dl/Qwen3.5-2B

# 4B
pixi run hf download Qwen/Qwen3.5-4B \
  --local-dir model_dl/Qwen3.5-4B
```

下载后确认目录存在 `config.json`：

```bash
ls model/Qwen3.5-0.8B/config.json
ls model_dl/Qwen3.5-4B/config.json
```

### 板端 (Atlas 200I DK A2)

- SSH 可访问 (`root@192.168.137.100`, 密码 `Mind@123`)
- 预装: `fastapi`, `uvicorn`, `pydantic`, `transformers`, `tokenizers`, `jinja2`, `markupsafe`, `numpy`
- NPU 状态正常 (`npu-smi info` → `Health: OK`)

---

## 纯板端 KV Cache 部署 (推荐入门)

无需 CUDA 主机，开发板独立运行完整 OpenAI API。

### 1. 导出 ONNX

```bash
pixi run python scripts/export_qwen35_kvcache.py \
  --max-len 256 \
  --output om_out/qwen3.5_kvcache_max256.onnx
```

### 2. ATC 编译

```bash
export INPUT_SHAPE="$(pixi run python scripts/gen_input_shape.py om_out/qwen3.5_kvcache_max256.onnx)"
MODEL_ONNX=om_out/qwen3.5_kvcache_max256.onnx \
  bash scripts/podman_convert.sh
```

### 3. 上传文件到板端

```bash
# 控制器代码
sshpass -p 'Mind@123' scp -r -o StrictHostKeyChecking=no \
  controller/ root@192.168.137.100:/root/slm_deploy/

# ModelSpec
sshpass -p 'Mind@123' scp -o StrictHostKeyChecking=no \
  scripts/qwen35_model_spec.py root@192.168.137.100:/root/slm_deploy/scripts/

# OM 模型
sshpass -p 'Mind@123' scp -o StrictHostKeyChecking=no \
  om_out/qwen3.5_kvcache_max256.om root@192.168.137.100:/root/slm_deploy/

# 模型配置 + Tokenizer
sshpass -p 'Mind@123' scp -o StrictHostKeyChecking=no \
  model/Qwen3.5-0.8B/config.json \
  model/Qwen3.5-0.8B/tokenizer.json \
  model/Qwen3.5-0.8B/tokenizer_config.json \
  model/Qwen3.5-0.8B/vocab.json \
  model/Qwen3.5-0.8B/merges.txt \
  model/Qwen3.5-0.8B/chat_template.jinja \
  root@192.168.137.100:/root/slm_deploy/

# 启动脚本
sshpass -p 'Mind@123' scp -o StrictHostKeyChecking=no \
  board/run_openai_kvcache_controller.sh \
  root@192.168.137.100:/root/slm_deploy/
sshpass -p 'Mind@123' ssh -o StrictHostKeyChecking=no root@192.168.137.100 \
  'chmod +x /root/slm_deploy/run_openai_kvcache_controller.sh'
```

### 4. 板端启动

```bash
# 登录开发板
cd /root/slm_deploy
bash run_openai_kvcache_controller.sh
```

模型加载约 210 秒，就绪后监听 `http://0.0.0.0:8000`。

### 5. 测试

```bash
# 健康检查
curl http://127.0.0.1:8000/healthz

# 非流式对话
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-0.8B-kvcache-om",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 64,
    "stream": false
  }'
```

---

## OpenAI 请求参数与示例

所有后端统一暴露 OpenAI 兼容接口：

- `GET /healthz`
- `GET /v1/models`
- `POST /v1/chat/completions`

### 1. 先查询可用模型名

请求中的 `model` 字段必须与服务当前注册的模型名完全一致。最稳妥的做法是先查：

```bash
curl http://127.0.0.1:8000/v1/models
```

返回示例：

```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen3.5-0.8B-kvcache-om",
      "object": "model",
      "owned_by": "local"
    }
  ]
}
```

常见默认模型名（以仓库内启动脚本为准）：

- `run_openai_kvcache_controller.sh` → `qwen3.5-0.8B-kvcache-om`
- `run_openai_split_controller_om.sh` → `qwen3.5-split-4-16-4-om`
- `run_openai_split_controller_om_16k.sh` → `qwen3.5-split-4-16-4-om-16k`
- `run_openai_split_controller_bound_2b.sh` → `qwen3.5-2b-split-0-24-0-om`
- `run_openai_split_controller_bound_4b.sh` → `qwen3.5-4b-split-0-32-0-bound`

若你手动启动控制器并修改了 `--model-name`，请求里也必须同步改成对应值。

### 2. 请求体字段

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `model` | `string` | 是 | — | 从 `/v1/models` 返回的 `id` 中选择 |
| `messages` | `array` | 是 | — | OpenAI 风格消息列表，常用 `system` / `user` / `assistant` |
| `stream` | `bool` | 否 | `false` | 是否启用流式输出 |
| `max_tokens` | `int` | 否 | `64` | 最多生成 token 数，当前接口限制 `1~4096` |
| `temperature` | `float` | 否 | `thinking=false` 时为 `0.7`，`thinking=true` 时为 `1.0` | 采样温度 |
| `top_k` | `int` | 否 | `thinking=false` 时为 `40`，`thinking=true` 时为 `20` | Top-K 采样 |
| `top_p` | `float` | 否 | `thinking=false` 时为 `1.0`，`thinking=true` 时为 `0.95` | Top-P 采样 |
| `presence_penalty` | `float` | 否 | `thinking=false` 时为 `0.0`，`thinking=true` 时为 `1.5` | 出现惩罚 |
| `repetition_penalty` | `float` | 否 | `1.0` | 重复惩罚 |
| `stop` | `string` 或 `string[]` | 否 | `null` | 停止字符串 |
| `enable_thinking` | `bool` | 否 | `false` | 是否启用 thinking 模式 |

常用消息格式：

```json
[
  {"role": "system", "content": "你是一个简洁的助手。"},
  {"role": "user", "content": "请介绍一下你自己。"}
]
```

### 3. 非流式请求示例

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-0.8B-kvcache-om",
    "messages": [
      {"role": "system", "content": "你是一个简洁的中文助手。"},
      {"role": "user", "content": "请用两句话介绍 Atlas 200I DK A2。"}
    ],
    "max_tokens": 128,
    "stream": false
  }'
```

### 4. 流式请求示例

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -N \
  -d '{
    "model": "qwen3.5-0.8B-kvcache-om",
    "messages": [
      {"role": "user", "content": "请逐步解释什么是 SplitNN。"}
    ],
    "max_tokens": 128,
    "stream": true
  }'
```

流式返回为标准 SSE，结束时会收到：

```text
data: [DONE]
```

### 5. 常用采样参数示例

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-0.8B-kvcache-om",
    "messages": [
      {"role": "user", "content": "写一段关于边缘部署的短文。"}
    ],
    "max_tokens": 256,
    "temperature": 0.8,
    "top_k": 32,
    "top_p": 0.9,
    "presence_penalty": 0.2,
    "repetition_penalty": 1.05
  }'
```

### 6. Thinking 模式示例

`enable_thinking=true` 时，服务会自动切换到另一组默认采样参数，更适合推理型任务。

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-0.8B-kvcache-om",
    "messages": [
      {"role": "user", "content": "比较 SplitNN 与纯板端 KV Cache 的优缺点。"}
    ],
    "max_tokens": 256,
    "enable_thinking": true
  }'
```

### 7. 常见注意事项

- 如果返回 `unsupported model`，先调用 `/v1/models`，不要手猜模型名
- `max_tokens` 是生成长度，不是总上下文长度；总长度还受服务启动时的 `--max-len` 限制
- 中段服务（SplitNN）健康检查是 `GET /v1/health`，OpenAI 控制器健康检查是 `GET /healthz`
- 某些模型不建议打开 thinking，例如 Qwen3-0.6B

---

## SplitNN OM 部署

板端执行前/后段，CUDA 主机执行中段。

### 1. 导出 + 编译

```bash
# 前缀
pixi run python scripts/export_qwen35_split_prefix.py \
  --model-path model/Qwen3.5-0.8B --max-len 16384 --split 4,20 \
  --output om_out/qwen3.5_split_prefix_max16384.onnx

# 后缀
pixi run python scripts/export_qwen35_split_suffix.py \
  --model-path model/Qwen3.5-0.8B --max-len 16384 --split 4,20 \
  --output om_out/qwen3.5_split_suffix_max16384.onnx

# ATC 编译前缀
export INPUT_SHAPE="$(pixi run python scripts/gen_input_shape.py om_out/qwen3.5_split_prefix_max16384.onnx)"
MODEL_ONNX=om_out/qwen3.5_split_prefix_max16384.onnx \
  bash scripts/podman_convert.sh

# ATC 编译后缀
export INPUT_SHAPE="$(pixi run python scripts/gen_input_shape.py om_out/qwen3.5_split_suffix_max16384.onnx)"
MODEL_ONNX=om_out/qwen3.5_split_suffix_max16384.onnx \
  bash scripts/podman_convert.sh
```

### 2. 上传文件到板端

```bash
# 控制器代码
sshpass -p 'Mind@123' scp -r -o StrictHostKeyChecking=no \
  controller/ root@192.168.137.100:/root/slm_deploy/

# ModelSpec
sshpass -p 'Mind@123' scp -o StrictHostKeyChecking=no \
  scripts/qwen35_model_spec.py root@192.168.137.100:/root/slm_deploy/scripts/

# Prefix / Suffix OM 与 metadata
sshpass -p 'Mind@123' scp -o StrictHostKeyChecking=no \
  om_out/qwen3.5_split_prefix_max16384.om \
  om_out/qwen3.5_split_prefix_max16384.metadata.json \
  om_out/qwen3.5_split_suffix_max16384.om \
  om_out/qwen3.5_split_suffix_max16384.metadata.json \
  root@192.168.137.100:/root/slm_deploy/

# tokenizer 与配置
sshpass -p 'Mind@123' scp -o StrictHostKeyChecking=no \
  model/Qwen3.5-0.8B/config.json \
  model/Qwen3.5-0.8B/tokenizer.json \
  model/Qwen3.5-0.8B/tokenizer_config.json \
  model/Qwen3.5-0.8B/vocab.json \
  model/Qwen3.5-0.8B/merges.txt \
  model/Qwen3.5-0.8B/chat_template.jinja \
  root@192.168.137.100:/root/slm_deploy/

# 启动脚本
sshpass -p 'Mind@123' scp -o StrictHostKeyChecking=no \
  board/run_openai_split_controller_om_16k.sh \
  root@192.168.137.100:/root/slm_deploy/
sshpass -p 'Mind@123' ssh -o StrictHostKeyChecking=no root@192.168.137.100 \
  'chmod +x /root/slm_deploy/run_openai_split_controller_om_16k.sh'
```

### 3. 主机启动中段服务

```bash
pixi run python server/qwen35_split_service.py \
  --host 0.0.0.0 --port 18080 \
  --model-path model/Qwen3.5-0.8B \
  --device cuda:0 --max-len 16384 --split 4,20
```

### 4. SSH 反向隧道

```bash
sshpass -p 'Mind@123' ssh -o StrictHostKeyChecking=no \
  -o ExitOnForwardFailure=yes \
  -N -R 28080:127.0.0.1:18080 root@192.168.137.100
```

### 5. 板端启动

```bash
cd /root/slm_deploy
bash run_openai_split_controller_om_16k.sh
```

---

## SplitNN 参数绑定部署 (2B)

板端负责 Embedding + LM Head（通过 tied_weight.bin 共享），支持任意 split。

### 1. 导出参数绑定资产

```bash
pixi run python scripts/export_qwen35_bound_embed_head.py \
  --model-path model_dl/Qwen3.5-2B \
  --output-dir om_out/qwen3.5_2b_bound_embed_head \
  --split 0,24 --compile-op
```

### 2. (可选) 导出并编译纯注意力 segment ONNX

若板端需承担注意力层（如 split=4/20），需额外导出：

```bash
# Prefix attention segment
pixi run python scripts/export_qwen35_middle.py \
  --model-path model_dl/Qwen3.5-2B --max-len 8192 --split 4,20 \
  --segment prefix --output om_out/qwen3.5_2b_prefix_dn_8k.onnx

# Suffix attention segment
pixi run python scripts/export_qwen35_middle.py \
  --model-path model_dl/Qwen3.5-2B --max-len 8192 --split 4,20 \
  --segment suffix --output om_out/qwen3.5_2b_suffix_ga_8k.onnx

# ATC 编译 prefix attention OM
export INPUT_SHAPE="$(pixi run python scripts/gen_input_shape.py om_out/qwen3.5_2b_prefix_dn_8k.onnx)"
MODEL_ONNX=om_out/qwen3.5_2b_prefix_dn_8k.onnx \
  bash scripts/podman_convert.sh

# ATC 编译 suffix attention OM
export INPUT_SHAPE="$(pixi run python scripts/gen_input_shape.py om_out/qwen3.5_2b_suffix_ga_8k.onnx)"
MODEL_ONNX=om_out/qwen3.5_2b_suffix_ga_8k.onnx \
  bash scripts/podman_convert.sh
```

### 3. 上传文件到板端

```bash
# 控制器代码
sshpass -p 'Mind@123' scp -r -o StrictHostKeyChecking=no \
  controller/ root@192.168.137.100:/root/slm_deploy/

# ModelSpec
sshpass -p 'Mind@123' scp -o StrictHostKeyChecking=no \
  scripts/qwen35_model_spec.py root@192.168.137.100:/root/slm_deploy/scripts/

# bound 资产
sshpass -p 'Mind@123' scp -r -o StrictHostKeyChecking=no \
  om_out/qwen3.5_2b_bound_embed_head root@192.168.137.100:/root/slm_deploy/

# tokenizer 与配置
sshpass -p 'Mind@123' scp -r -o StrictHostKeyChecking=no \
  model_dl/Qwen3.5-2B root@192.168.137.100:/root/slm_deploy/model_2b

# 若 split != 0/N/0，再上传 attention OM
sshpass -p 'Mind@123' scp -o StrictHostKeyChecking=no \
  om_out/qwen3.5_2b_prefix_dn_8k.om \
  om_out/qwen3.5_2b_suffix_ga_8k.om \
  root@192.168.137.100:/root/slm_deploy/
```

### 4. 板端启动（非 0/N/0 需传 OM 参数）

```bash
cd /root/slm_deploy
python3 controller/openai_controller.py \
  --backend splitnn_bound_embed_head \
  --model-name qwen3.5-2b-bound-4-20 \
  --remote-model-name Qwen3.5-2B-split-4-20 \
  --tokenizer-dir /root/slm_deploy/model_2b \
  --server-url http://127.0.0.1:28080 \
  --max-len 8192 --split 4,20 \
  --bound-asset-dir /root/slm_deploy/qwen3.5_2b_bound_embed_head \
  --prefix-om /root/slm_deploy/qwen3.5_2b_prefix_dn_8k.om \
  --suffix-om /root/slm_deploy/qwen3.5_2b_suffix_ga_8k.om \
  --checksum
```

## Qwen3.5-4B 参数绑定部署（推荐 0/32/0）

Qwen3.5-4B 在 `0/32/0` 切分下，板端只负责 embedding 与 lm_head，中段 32 层全部在 CUDA 主机执行。

当前推荐配置：
- embedding: CPU（直接从 `tied_weight.bin` 取词向量）
- lm_head: NPU ACL `MatMul`
- split: `0,32`

### 1. 导出并编译 bound 资产

```bash
pixi run python scripts/export_qwen35_bound_embed_head.py \
  --model-path model_dl/Qwen3.5-4B \
  --output-dir om_out/qwen3.5_4b_bound_embed_head \
  --split 0,32 --compile-op
```

`0/32/0` 下板端不承担 attention 层，因此**不需要**额外导出或编译 prefix/suffix attention OM。

### 2. 上传文件到板端

```bash
# 控制器代码
sshpass -p 'Mind@123' scp -r -o StrictHostKeyChecking=no \
  controller/ root@192.168.137.100:/root/slm_deploy/

# ModelSpec
sshpass -p 'Mind@123' scp -o StrictHostKeyChecking=no \
  scripts/qwen35_model_spec.py root@192.168.137.100:/root/slm_deploy/scripts/

# bound 资产
sshpass -p 'Mind@123' scp -r -o StrictHostKeyChecking=no \
  om_out/qwen3.5_4b_bound_embed_head root@192.168.137.100:/root/slm_deploy/

# tokenizer 与配置
sshpass -p 'Mind@123' scp -o StrictHostKeyChecking=no \
  model_dl/Qwen3.5-4B/config.json \
  model_dl/Qwen3.5-4B/tokenizer.json \
  model_dl/Qwen3.5-4B/tokenizer_config.json \
  model_dl/Qwen3.5-4B/vocab.json \
  model_dl/Qwen3.5-4B/merges.txt \
  model_dl/Qwen3.5-4B/chat_template.jinja \
  root@192.168.137.100:/root/slm_deploy/model_4b/

# 启动脚本
sshpass -p 'Mind@123' scp -o StrictHostKeyChecking=no \
  board/run_openai_split_controller_bound_4b.sh \
  root@192.168.137.100:/root/slm_deploy/
sshpass -p 'Mind@123' ssh -o StrictHostKeyChecking=no root@192.168.137.100 \
  'chmod +x /root/slm_deploy/run_openai_split_controller_bound_4b.sh'
```

### 3. 主机侧中段服务

```bash
pixi run python server/qwen35_split_service.py \
  --host 0.0.0.0 --port 18080 \
  --model-path model_dl/Qwen3.5-4B \
  --device cuda:0 --max-len 16384 --split 0,32
```

### 4. SSH 反向隧道

```bash
sshpass -p 'Mind@123' ssh -o StrictHostKeyChecking=no \
  -o ExitOnForwardFailure=yes \
  -N -R 28080:127.0.0.1:18080 root@192.168.137.100
```

### 5. 板端启动

```bash
cd /root/slm_deploy
bash run_openai_split_controller_bound_4b.sh
```

说明：
- 中段服务健康检查路径是 `GET /v1/health`，不是 `/healthz`
- 4B 上 `GatherV2` 词嵌入单算子可能失败，当前默认已改为 CPU embedding
- 实测 `CPU embedding + NPU lm_head` 约 `4.5~4.7 tok/s`
- `600 token` 长输出测试未观察到异常发散

---

## 下一步

- [架构设计](./ARCHITECTURE.md) — 理解三种引擎模式的工作原理
- [部署详解](./DEPLOYMENT.md) — 板端文件布局、SSH 隧道、脚本说明
- [开发指南](./DEVELOPMENT.md) — 环境配置、工具链、导出流程
- [踩坑速查](./GOTCHAS.md) — 常见问题与解决方案
