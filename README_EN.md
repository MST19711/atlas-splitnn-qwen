# Qwen3 / Qwen3.5 on Huawei Atlas 200I DK A2

[中文](./README.md) | [Experiment Report](./REPORT.md)

Deploy Qwen3-0.6B and Qwen3.5-0.8B small language models on the Huawei Ascend Atlas 200I DK A2 edge computing board for NPU-accelerated on-device Chinese conversation.

---

## Results

| Model | Context | Prefill | Decode | OM Size |
|-------|---------|---------|--------|---------|
| Qwen3 static window | 32 tok | — | 3.6 tok/s | 1.5 GB |
| Qwen3 KV Cache | 256 tok | ~5.6s | 3.6 tok/s | 1.5 GB |
| Qwen3.5 KV Cache | 256 tok | ~52s | 3.7 tok/s | 1.9 GB |
| Qwen3.5 KV Cache | 1024 tok | ~67s | 3.6 tok/s | 1.9 GB |

---

## Hardware & Toolchain

| Component | Description |
|-----------|-------------|
| Board | Atlas 200I DK A2 (Ascend310B4, 4GB NPU) |
| Models | Qwen3-0.6B / Qwen3.5-0.8B (FP16) |
| CANN | 7.0.0 (ATC container) / 7.0.RC1 (board runtime) |
| ONNX | opset 15, TorchScript export |
| Container | Podman + Rocky Linux 9, image `cann-atc-rocky:v7` |
| Python | pixi-managed (x86), pip (board aarch64) |

---

## Project Structure

```
├── model/                    # Model weights + tokenizer
├── scripts/                  # ONNX export & ATC conversion (x86)
│   ├── export_qwen3_static.py        # Qwen3 static window export
│   ├── export_qwen3_kvcache.py     # Qwen3 KV Cache export
│   ├── export_qwen35_kvcache.py      # Qwen3.5 DeltaNet KV Cache export
│   ├── patch_qwen3_static_onnx.py         # GQA Expand→Tile (static window only)
│   ├── gen_input_shape.py    # ONNX → ATC INPUT_SHAPE helper
│   ├── podman_convert.sh     # Containerized ATC conversion
│   └── download_qwen3.py     # HF model download
├── board/                    # On-board inference (aarch64)
│   ├── gen_text_qwen3_static.py     # Static window inference
│   ├── gen_text_qwen3_kvcache.py   # Qwen3 KV Cache inference
│   ├── gen_text_qwen35_kvcache.py    # Qwen3.5 DeltaNet inference
│   └── run_qwen3_kvcache.sh
├── docker/                   # ATC container build
│   └── Containerfile.v2-cann7
├── om_out/                   # ATC output (*.om)
└── logs/                     # Build logs
```

---

## Quick Start

### 1. Setup

#### Dev Machine (x86_64)

```bash
cd Embedded_FinalHW && pixi install

# Download model weights
hf download Qwen/Qwen3-0.6B --local-dir model/Qwen3-0.6B
hf download Qwen/Qwen3.5-0.8B --local-dir model/Qwen3.5-0.8B

# Download CANN 7.0.0 packages into docker/
wget -O docker/Ascend-cann-toolkit_7.0.0_linux-x86_64.run \   # ATC compiler (~1.6GB)
  "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%207.0.0/Ascend-cann-toolkit_7.0.0_linux-x86_64.run"
wget -O docker/Ascend-cann-kernels-310b-7.0.0-linux.noarch.rpm \  # 310B kernel (~351MB, must use 310B not 310P)
  "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%207.0.0/Ascend-cann-kernels-310b-7.0.0-linux.noarch.rpm"

# Build ATC container
podman build -t localhost/cann-atc-rocky:v7 \
    -f docker/Containerfile.v2-cann7 docker/
```

#### Download Board Python Wheels (run on dev machine with internet)

```bash
mkdir -p tmp
python3 -m pip download --dest tmp --platform manylinux2014_aarch64 \
    --python-version 310 --implementation cp --abi cp310 --only-binary=:all: \
    "numpy==1.26.4" "transformers==4.53.3" "tokenizers==0.21.4" \
    "torch==2.1.0" "safetensors" "huggingface-hub" "requests" \
    "pyyaml" "regex" "tqdm" "filelock" "fsspec" "packaging" \
    "typing-extensions" "sympy" "networkx" "jinja2"
curl -L https://bootstrap.pypa.io/get-pip.py -o tmp/get-pip.py
```

### 2. Export ONNX → ATC → OM

```bash
# Export ONNX (pick one; customize --max-len as needed)
pixi run python scripts/export_qwen3_static.py \
    --output om_out/qwen3_seq32.onnx

pixi run python scripts/export_qwen3_kvcache.py --max-len 256 \
    --output om_out/qwen3_kvcache_max256.onnx

pixi run python scripts/export_qwen35_kvcache.py --max-len 256 \
    --output om_out/qwen3.5_kvcache_max256.onnx

# ATC compile (auto-detects input shape from ONNX)
INPUT_SHAPE=$(pixi run python scripts/gen_input_shape.py om_out/qwen3.5_kvcache_max256.onnx) \
MODEL_ONNX=om_out/qwen3.5_kvcache_max256.onnx \
bash scripts/podman_convert.sh
```

> Static window model requires `patch_qwen3_static_onnx.py` before ATC:
> ```bash
> pixi run python scripts/patch_qwen3_static_onnx.py om_out/qwen3_seq32.onnx
> ```

### 3. Board Setup

The board ships with Ubuntu 22.04 aarch64 and CANN 7.0.RC1. Install Python deps using wheels downloaded in step 1.

Transfer wheels from dev machine:

```bash
scp tmp/*.whl tmp/get-pip.py root@192.168.137.100:/root/slm_deploy/wheels/
```

On the board:

```bash
cd /root/slm_deploy
python3 wheels/get-pip.py
python3 -m pip install --no-index --find-links=wheels \
    "numpy==1.26.4" transformers torch
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 -c "import acl, numpy, transformers; print('OK')"
```

Optional: free ~120MB RAM:

```bash
systemctl stop sddm && systemctl disable sddm
pkill -f xfce4-power-manager
pkill -f xfce4-notifyd
pkill -f tumblerd
```

### 4. Deploy & Infer

From dev machine:

```bash
scp om_out/qwen3.5_kvcache_max256.om root@192.168.137.100:/root/slm_deploy/
scp board/gen_text_qwen35_kvcache.py      root@192.168.137.100:/root/slm_deploy/
```

On the board:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /root/slm_deploy
python3 gen_text_qwen35_kvcache.py \
    --model qwen3.5_kvcache_max256.om \
    --prompt "你好" --max-tokens 50
```

Inference scripts:

| Script | Model | Key Parameters |
|--------|-------|----------------|
| `gen_text_qwen3_static.py` | Qwen3 static window | `--prompt`, `--max-tokens` |
| `gen_text_qwen3_kvcache.py` | Qwen3 KV Cache | `--model X.om --prompt` |
| `gen_text_qwen35_kvcache.py` | Qwen3.5 KV Cache | `--model X.om --tokenizer-dir /root/slm_deploy` |

---

## Related Docs

- [REPORT.md](./REPORT.md) — Detailed experiment report (design decisions, issues, optimization)
- [AGENTS.md](./AGENTS.md) — AI-assisted development quick reference
