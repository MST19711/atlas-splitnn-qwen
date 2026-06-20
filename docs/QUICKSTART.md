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
  --local-dir model_dl/Qwen3.5-0.8B

# 2B
pixi run hf download Qwen/Qwen3.5-2B \
  --local-dir model_dl/Qwen3.5-2B

# 4B
pixi run hf download Qwen/Qwen3.5-4B \
  --local-dir model_dl/Qwen3.5-4B
```

下载后确认目录存在 `config.json`：

```bash
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
MODEL_ONNX=om_out/qwen3.5_kvcache_max256.onnx \
INPUT_SHAPE="$(pixi run python scripts/gen_input_shape.py om_out/qwen3.5_kvcache_max256.onnx)" \
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

# ATC 编译 (前缀 + 后缀各一次)
```

### 2. 主机启动中段服务

```bash
pixi run python server/qwen35_split_service.py \
  --host 0.0.0.0 --port 18080 \
  --model-path model/Qwen3.5-0.8B \
  --device cuda:0 --max-len 16384 --split 4,20
```

### 3. SSH 反向隧道

```bash
sshpass -p 'Mind@123' ssh -o StrictHostKeyChecking=no \
  -o ExitOnForwardFailure=yes \
  -N -R 28080:127.0.0.1:18080 root@192.168.137.100
```

### 4. 板端启动

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
  --output-dir qwen3.5_2b_bound_embed_head \
  --split 0,24 --compile-op
```

### 2. (可选) 导出纯注意力 segment ONNX

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

# ATC 编译
```

### 3. 板端启动（非 0/N/0 需传 OM 参数）

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

板端启动：

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
