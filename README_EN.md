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
cd EF && pixi install

# Download model weights
hf download Qwen/Qwen3-0.6B --local-dir model/Qwen3-0.6B
hf download Qwen/Qwen3.5-0.8B --local-dir model/Qwen3.5-0.8B

# Download CANN 7.0.0 packages into docker/
wget -O docker/Ascend-cann-toolkit_7.0.0_linux-x86_64.run \   # ATC compiler (~1.6GB)
  "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%207.0.0/Ascend-cann-toolkit_7.0.0_linux-x86_64.run"
wget -O docker/Ascend-cann-kernels-310b-7.0.0-linux.noarch.rpm \  # 310B kernel (~351MB, must use 310B not 310P)
  "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%207.0.0/Ascend-cann-kernels-310b-7.0.0-linux.noarch.rpm"

# Build ATC container (--network=host required for dnf/pip inside container)
podman build --network=host -t localhost/cann-atc-rocky:v7 \
    -f docker/Containerfile.v2-cann7 docker/
```

#### Download Board Python Wheels (run on dev machine with internet)

The board inference scripts only depend on `numpy` + `transformers`. **Must use `--no-deps` for each download** (see explanation below).

```bash
mkdir -p tmp

# Core dependencies
python3 -m pip download --dest tmp --platform manylinux2014_aarch64 \
    --python-version 310 --implementation cp --abi cp310 --only-binary=:all: --no-deps \
    "numpy==1.26.4" "transformers==4.57.6" "tokenizers==0.22.2" \
    "huggingface_hub==0.34.6" "safetensors" "requests" "pyyaml" "regex" "tqdm" \
    "filelock" "fsspec" "packaging" "typing-extensions" \
    "certifi" "charset-normalizer" "idna" "urllib3"

# jinja2 + markupsafe: required by apply_chat_template(enable_thinking=False)
python3 -m pip download --dest tmp --platform manylinux2014_aarch64 \
    --python-version 310 --implementation cp --abi cp310 --only-binary=:all: --no-deps \
    "markupsafe"
python3 -m pip download --dest tmp --platform any \
    --python-version 310 --only-binary=:all: --no-deps \
    "jinja2"

# pip bootstrap: board has no pip, and pip.conf points to unreachable mirror
python3 -m pip download --dest tmp --platform manylinux2014_aarch64 \
    --python-version 310 --implementation cp --abi cp310 --only-binary=:all: --no-deps \
    "pip" "setuptools" "wheel"

curl -L https://bootstrap.pypa.io/get-pip.py -o tmp/get-pip.py
```

> **Why `--no-deps`?** `huggingface-hub>=0.34` requires `hf-xet>=1.1.3` on aarch64, but the latest `hf-xet` aarch64 wheel on PyPI is only `0.1.x`. `hf-xet` is only used for parallel HF Hub downloads; the board only reads local tokenizer files via `from_pretrained(local_dir)`, so installing huggingface-hub with `--no-deps` is safe.

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
# NOTE: INPUT_SHAPE contains semicolons — must export before calling the script
INPUT_SHAPE=$(pixi run python scripts/gen_input_shape.py om_out/qwen3.5_kvcache_max256.onnx)
export INPUT_SHAPE MODEL_ONNX="om_out/qwen3.5_kvcache_max256.onnx" OUTPUT_PREFIX="om_out/qwen3.5_kvcache_max256"
bash scripts/podman_convert.sh
```

> Static window model requires `patch_qwen3_static_onnx.py` before ATC:
> ```bash
> pixi run python scripts/patch_qwen3_static_onnx.py om_out/qwen3_seq32.onnx
> ```

### 3. Board Setup

The board ships with Ubuntu 22.04 aarch64 and CANN 7.0.RC1. The board's `/root/.pip/pip.conf` points to an unreachable mirror (doubanio), so pip must be bootstrapped from a wheel.

#### Transfer files

```bash
sshpass -p 'Mind@123' ssh root@192.168.137.100 'mkdir -p /root/slm_deploy/wheels'
sshpass -p 'Mind@123' scp tmp/*.whl tmp/get-pip.py root@192.168.137.100:/root/slm_deploy/wheels/
```

#### Install on the board

```bash
cd /root/slm_deploy/wheels

# 1. Bootstrap pip from wheel (get-pip.py fails due to unreachable mirror)
python3 -c "
import zipfile, sys, os
whl = [f for f in os.listdir('.') if f.startswith('pip-')][0]
zf = zipfile.ZipFile(whl)
zf.extractall('/tmp/_pip')
sys.path.insert(0, '/tmp/_pip')
import pip._internal
pip._internal.main(['install', '--no-deps', '--no-index', '--force-reinstall', whl])
"

# 2. Base packages (no upstream deps)
python3 -m pip install --no-deps --no-index --find-links=. \
    numpy-*.whl typing_extensions-*.whl packaging-*.whl filelock-*.whl \
    fsspec-*.whl tqdm-*.whl regex-*.whl safetensors-*.whl

# 3. huggingface-hub (--no-deps skips hf-xet)
python3 -m pip install --no-deps --no-index --find-links=. \
    huggingface_hub-0.34.6-*.whl

# 4. requests → tokenizers → transformers
python3 -m pip install --no-deps --no-index --find-links=. \
    requests-*.whl charset_normalizer-*.whl tokenizers-*.whl transformers-*.whl

# 5. jinja2 + markupsafe (chat template rendering)
python3 -m pip install --no-deps --no-index --find-links=. \
    markupsafe-*.whl jinja2-*.whl

# 6. Relax tokenizers version upper bound
sed -i 's/tokenizers>=0.22.0,<=0.23.0/tokenizers>=0.22.0,<=0.23.1/' \
  /usr/local/lib/python3.10/dist-packages/transformers/dependency_versions_table.py

# 7. Verify
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 -c "import acl, numpy, transformers, tokenizers, huggingface_hub, safetensors, jinja2; print('OK')"
```

> **About `hf-xet`**: `huggingface-hub>=0.34` depends on `hf-xet>=1.1.3` on aarch64, but only `0.1.x` has aarch64 wheels on PyPI. `hf-xet` is only used for parallel HF Hub downloads; the board only reads local tokenizer files via `from_pretrained(local_dir)`, so installing huggingface-hub with `--no-deps` is safe.

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
# Qwen3 KV Cache
scp om_out/qwen3_kvcache_max256_cann7.om root@192.168.137.100:/root/slm_deploy/
scp board/gen_text_qwen3_kvcache.py          root@192.168.137.100:/root/slm_deploy/

# Qwen3.5 KV Cache (tokenizer is incompatible with Qwen3 — separate files required)
scp om_out/qwen3.5_kvcache_max256.om         root@192.168.137.100:/root/slm_deploy/
scp board/gen_text_qwen35_kvcache.py          root@192.168.137.100:/root/slm_deploy/
scp model/Qwen3.5-0.8B/tokenizer.json model/Qwen3.5-0.8B/tokenizer_config.json \
    model/Qwen3.5-0.8B/chat_template.jinja     root@192.168.137.100:/root/slm_deploy/
```

> **Tokenizer compatibility**: Qwen3 uses `vocab.json` + `merges.txt`, Qwen3.5 uses `tokenizer.json`. They are mutually incompatible — do not overwrite one with the other.

On the board:

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
