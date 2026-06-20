# Qwen3 / Qwen3.5 on Huawei Atlas 200I DK A2

[中文](./README.md)

Deploy Qwen small language models on the Huawei Ascend Atlas 200I DK A2 edge computing board for NPU-accelerated on-device Chinese conversation.

---

## Three Engine Modes

| Mode | Model | Context | Board Execution | External Deps | Use Case |
|------|-------|---------|-----------------|---------------|----------|
| **KV Cache Standalone** | 0.8B | 256/4096 tok | All 24 layers | None | Quick start, standalone |
| **SplitNN OM** | 0.8B/4B | 16K tok | First 4 + last 4 layers | CUDA host + SSH | Large models, long context |
| **SplitNN Param Binding** | 2B | 8K tok | Embedding + LM Head | CUDA host + SSH | Minimal board load |

---

## Model Overview

| Model | Params | hidden_size | Layers | Speed | Details |
|-------|--------|-------------|--------|-------|---------|
| Qwen3-0.6B | ~600M | 1024 | 28 GQA | 3.6 tok/s | [Model Details](./docs/MODELS.md) |
| Qwen3.5-0.8B | ~800M | 1024 | 24 (18DN+6GA) | ~4.7 tok/s | [Model Details](./docs/MODELS.md) |
| Qwen3.5-2B | ~2B | 2048 | 24 (18DN+6GA) | ~5.3 tok/s | [Model Details](./docs/MODELS.md) |
| Qwen3.5-4B | ~4B | 2560 | 32 (24DN+8GA) | ~1.1 tok/s | [Model Details](./docs/MODELS.md) |

---

## Documentation

| Document | Description |
|----------|-------------|
| [Quick Start](./docs/QUICKSTART.md) | Three steps to standalone deployment |
| [Architecture](./docs/ARCHITECTURE.md) | Three-layer design, engine modes, generation pipeline |
| [Deployment](./docs/DEPLOYMENT.md) | Board file layout, SSH tunnels, startup scripts |
| [Development](./docs/DEVELOPMENT.md) | Environment, toolchain, export workflow, validation |
| [Design Decisions](./docs/DESIGN.md) | Monkey-patch rationale, double buffering, split strategy |
| [Experiment Log](./docs/EXPERIMENTS.md) | Ten-stage development history |
| [Gotchas](./docs/GOTCHAS.md) | Common pitfalls and solutions |
| [Models](./docs/MODELS.md) | Detailed parameters for all supported models |

---

## Hardware & Toolchain

| Component | Description |
|-----------|-------------|
| Board | Atlas 200I DK A2 (Ascend310B4, 4GB NPU) |
| CANN | 7.0.0 (ATC compiler) / 7.0.RC1 (board runtime) |
| ONNX | opset 15, TorchScript export |
| Container | Podman + Rocky Linux 9, image `cann-atc-rocky:v7` |
| Python | pixi-managed (x86), pip (board aarch64) |

---

## Quick Start

```bash
# 1. Export ONNX
pixi run python scripts/export_qwen35_kvcache.py \
  --max-len 256 --output om_out/qwen3.5_kvcache_max256.onnx

# 2. ATC Compile
export INPUT_SHAPE=$(pixi run python scripts/gen_input_shape.py \
  om_out/qwen3.5_kvcache_max256.onnx)
MODEL_ONNX=om_out/qwen3.5_kvcache_max256.onnx \
  bash scripts/podman_convert.sh

# 3. Upload & Start (board)
cd /root/slm_deploy && bash run_openai_kvcache_controller.sh

# 4. Test
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5-0.8B-kvcache-om","messages":[{"role":"user","content":"Hi"}],"max_tokens":64}'
```

See [Quick Start](./docs/QUICKSTART.md) for details.
