# Qwen3 / Qwen3.5 on Huawei Atlas 200I DK A2

[中文](./README.md) | [Experiment Report](./REPORT.md)

Deploy Qwen3-0.6B and Qwen3.5-0.8B small language models on the Huawei Ascend Atlas 200I DK A2 edge computing board for NPU-accelerated on-device Chinese conversation.

---

## Results

| Model | Context | Prefill | Decode | OM Size |
|-------|---------|---------|--------|---------|
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
│   ├── export_qwen3_kvcache.py     # Qwen3 KV Cache export
│   ├── export_qwen35_kvcache.py      # Qwen3.5 DeltaNet KV Cache export
│   ├── gen_input_shape.py    # ONNX → ATC INPUT_SHAPE helper
│   ├── podman_convert.sh     # Containerized ATC conversion
├── board/                    # On-board inference (aarch64)
│   ├── gen_text_qwen3_kvcache.py   # Qwen3 KV Cache inference
│   ├── gen_text_qwen35_kvcache.py    # Qwen3.5 DeltaNet inference
│   └── run_qwen3_kvcache.sh
├── controller/               # SplitNN controller (OpenAI API + pluggable front/back engines)
│   ├── openai_split_controller.py
│   ├── orchestrator.py
│   ├── remote_middle.py
│   └── engine/
├── docker/                   # ATC container build
│   └── Containerfile.v2-cann7
├── server/                   # Remote middle-segment service
│   └── qwen35_split_service.py
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

# Extra dependencies for the SplitNN controller on the board
mkdir -p tmp/board_controller_wheels
python3 -m pip download --dest tmp/board_controller_wheels --platform manylinux2014_aarch64 \
    --python-version 310 --implementation cp --abi cp310 --only-binary=:all: --no-deps \
    "fastapi" "uvicorn" "pydantic" "pydantic-core" "starlette" \
    "annotated-types" "typing-inspection" "anyio" "h11" "click" \
    "sniffio" "exceptiongroup"

# pip bootstrap: board has no pip, and pip.conf points to unreachable mirror
python3 -m pip download --dest tmp --platform manylinux2014_aarch64 \
    --python-version 310 --implementation cp --abi cp310 --only-binary=:all: --no-deps \
    "pip" "setuptools" "wheel"

curl -L https://bootstrap.pypa.io/get-pip.py -o tmp/get-pip.py
```

> **Why `--no-deps`?** `huggingface-hub>=0.34` requires `hf-xet>=1.1.3` on aarch64, but the latest `hf-xet` aarch64 wheel on PyPI is only `0.1.x`. `hf-xet` is only used for parallel HF Hub downloads; the board only reads local tokenizer files via `from_pretrained(local_dir)`, so installing huggingface-hub with `--no-deps` is safe.

### 2. Export ONNX → ATC → OM

```bash
# Export ONNX (customize --max-len as needed)
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

### 3. Board Setup

The board ships with Ubuntu 22.04 aarch64 and CANN 7.0.RC1. The board's `/root/.pip/pip.conf` points to an unreachable mirror (doubanio), so pip must be bootstrapped from a wheel.

#### Transfer files

```bash
sshpass -p 'Mind@123' ssh root@192.168.137.100 'mkdir -p /root/slm_deploy/wheels'
sshpass -p 'Mind@123' scp tmp/*.whl tmp/get-pip.py root@192.168.137.100:/root/slm_deploy/wheels/
sshpass -p 'Mind@123' ssh root@192.168.137.100 'mkdir -p /root/slm_deploy/board_controller_wheels'
sshpass -p 'Mind@123' scp tmp/board_controller_wheels/*.whl \
    root@192.168.137.100:/root/slm_deploy/board_controller_wheels/
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

# 8. Extra packages for the SplitNN controller
cd /root/slm_deploy/board_controller_wheels
python3 -m pip install --no-deps --no-index --find-links=. \
    annotated_types-*.whl exceptiongroup-*.whl sniffio-*.whl anyio-*.whl \
    click-*.whl h11-*.whl typing_inspection-*.whl pydantic_core-*.whl \
    pydantic-*.whl starlette-*.whl uvicorn-*.whl fastapi-*.whl

# 9. Verify controller deps
python3 -c "import fastapi, uvicorn, pydantic; print('controller deps OK')"
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
| `gen_text_qwen3_kvcache.py` | Qwen3 KV Cache | `--model X.om --prompt` |
| `gen_text_qwen35_kvcache.py` | Qwen3.5 KV Cache | `--model X.om --tokenizer-dir /root/slm_deploy` |

---

## Related Docs

- [REPORT.md](./REPORT.md) — Detailed experiment report (design decisions, issues, optimization)
- [AGENTS.md](./AGENTS.md) — AI-assisted development quick reference

---

## SplitNN Prototype (Board + CUDA Host)

The first SplitNN prototype uses a `4 / 16 / 4` split:
- board executes the first 4 layers and last 4 layers
- host executes the middle 16 layers
- the two sides exchange only `(1,1,1024)` FP16 hidden states over HTTP

### Why this split

The goal is not “distributed inference” for its own sake, but a practical division of labor:

- keep the front/back segments close to the final edge deployment shape
- move the main compute-heavy middle segment to the host
- avoid sending full caches over the network

For `Qwen3.5-0.8B`, `4 / 16 / 4` is a clean cut because:

1. the split points stay on layer boundaries, preserving residual and cache semantics
2. Qwen3.5 has a repeating 4-layer pattern (`3 linear_attention + 1 full_attention`), so `4 / 16 / 4` keeps each segment structurally aligned

### Responsibilities

The raw SplitNN prototype has three parts:

1. **Prefix**
   - input: `token_id + position + prefix cache`
   - output: `hidden_state_l4`
   - runtime: board OM or dev-machine ONNX

2. **Middle**
   - input: `hidden_state_l4 + position + middle cache`
   - output: `hidden_state_l20`
   - runtime: host `server/qwen35_split_service.py`

3. **Suffix**
   - input: `hidden_state_l20 + position + suffix cache`
   - output: `logits`
   - runtime: board OM or dev-machine ONNX

Ownership is intentionally split:
- prefix/suffix cache stays on the local execution side
- middle cache stays on the remote service
- only hidden states cross the network

### Validation before the controller

Before adding the controller layer, the raw SplitNN prototype had already passed three levels of validation:

1. **pure PyTorch reference alignment**
2. **multi-step ORT validation for prefix/suffix**
3. **local simulation with `ONNX prefix/suffix + middle server`**, including prompt-based decoded text generation

The controller therefore builds on top of a working SplitNN prototype rather than introducing the split design from scratch.

### Export prefix / suffix ONNX

```bash
pixi run python scripts/export_qwen35_split_prefix.py \
    --max-len 256 --output om_out/qwen3.5_split_prefix_max256.onnx

pixi run python scripts/export_qwen35_split_suffix.py \
    --max-len 256 --output om_out/qwen3.5_split_suffix_max256.onnx

# 16K long-context variant
pixi run python scripts/export_qwen35_split_prefix.py \
    --max-len 16384 --output om_out/qwen3.5_split_prefix_max16384.onnx

pixi run python scripts/export_qwen35_split_suffix.py \
    --max-len 16384 --output om_out/qwen3.5_split_suffix_max16384.onnx
```

Reference-chain validation:

```bash
pixi run python scripts/validate_qwen35_split_reference.py
```

Multi-step ORT validation:

```bash
pixi run python scripts/validate_qwen35_split_ort.py \
    --prefix-onnx om_out/qwen3.5_split_prefix_max256.onnx \
    --suffix-onnx om_out/qwen3.5_split_suffix_max256.onnx
```

Then compile the two ONNX models into `.om` files with `scripts/gen_input_shape.py` + `scripts/podman_convert.sh`.

### Start the CUDA middle service

```bash
pixi run python server/qwen35_split_service.py \
    --host 0.0.0.0 --port 18080 \
    --model-path model/Qwen3.5-0.8B \
    --device cuda:0 --max-len 16384
```

Health check:

```bash
curl http://<server-ip>:18080/v1/health
```

### Run raw SplitNN on the board

Copy the following files to `/root/slm_deploy/`:
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

Short context (`256`) example:

```bash
python3 -u /root/slm_deploy/gen_text_qwen35_splitnn.py \
    --server-url http://<server-ip>:18080 \
    --prefix-model /root/slm_deploy/qwen3.5_split_prefix_max256.om \
    --suffix-model /root/slm_deploy/qwen3.5_split_suffix_max256.om
```

Long context (`16K`) example:

```bash
cd /root/slm_deploy
./run_qwen35_splitnn_16k.sh
```

`run_qwen35_splitnn_16k.sh` defaults to `qwen3.5_split_prefix_max16384.om`, `qwen3.5_split_suffix_max16384.om`, and `--max-len 16384`.

---

## SplitNN Controller (OpenAI API)

On top of the raw SplitNN prototype, the repo now includes a controller layer that:
- exposes OpenAI-compatible `/v1/chat/completions`
- manages tokenizer, chat template, sampling, and the autoregressive loop
- switches front/back execution via a unified engine interface:
  - `OnnxSplitEngine` for dev-machine simulation
  - `OmSplitEngine` for board deployment
- keeps the middle 16 layers on the remote service `server/qwen35_split_service.py`

### Controller Layout

```
OpenAI Client
    |
    v
controller/openai_split_controller.py
    |
    +-- orchestrator.py
    +-- remote_middle.py
    +-- engine/onnx_engine.py
    `-- engine/om_engine.py
```

### Design Summary

- **Stateless multi-turn:** each request re-prefills from the full `messages` history
- **Prefix/suffix cache:** owned by the local engine instance
- **Middle cache:** owned by the remote middle server, keyed by `session_id`
- **Streaming:** supports `stream=true` via SSE

### Start on Dev Machine (ONNX backend)

Start the middle service first:

```bash
pixi run python server/qwen35_split_service.py \
    --host 127.0.0.1 --port 18080 \
    --model-path model/Qwen3.5-0.8B \
    --device cuda:0 --max-len 16384
```

Then start the controller:

```bash
pixi run python controller/openai_split_controller.py \
    --host 127.0.0.1 --port 8000 \
    --engine onnx \
    --model-name qwen3.5-split-4-16-4-onnx \
    --remote-model-name Qwen3.5-0.8B-split-4-16-4 \
    --tokenizer-dir model/Qwen3.5-0.8B \
    --server-url http://127.0.0.1:18080 \
    --max-len 256 \
    --prefix-onnx om_out/qwen3.5_split_prefix_max256.onnx \
    --suffix-onnx om_out/qwen3.5_split_suffix_max256.onnx
```

### Start on the Board (OM backend)

Copy the following files to `/root/slm_deploy/` on the board:
- `controller/`
- `board/run_openai_split_controller_om.sh`
- `board/run_openai_split_controller_om_16k.sh`
- `om_out/qwen3.5_split_prefix_max256.om`
- `om_out/qwen3.5_split_suffix_max256.om`
- `om_out/qwen3.5_split_prefix_max16384.om`
- `om_out/qwen3.5_split_suffix_max16384.om`
- `model/Qwen3.5-0.8B/tokenizer.json`
- `model/Qwen3.5-0.8B/tokenizer_config.json`
- `model/Qwen3.5-0.8B/chat_template.jinja`

If the board cannot directly reach the host's `18080` port, create a reverse tunnel from the dev machine first:

```bash
sshpass -p 'Mind@123' ssh -o StrictHostKeyChecking=no -N \
  -R 28080:127.0.0.1:18080 root@192.168.137.100
```

Then start the controller on the board:

```bash
cd /root/slm_deploy
./run_openai_split_controller_om.sh
```

16K controller variant:

```bash
cd /root/slm_deploy
./run_openai_split_controller_om_16k.sh
```

Health check:

```bash
curl http://127.0.0.1:8000/healthz
```

Non-streaming example:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.5-split-4-16-4-onnx",
    "messages": [{"role":"user","content":"Hello, introduce yourself in one sentence."}],
    "stream": false,
    "max_tokens": 24,
    "temperature": 0
  }'
```

Streaming example:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.5-split-4-16-4-onnx",
    "messages": [{"role":"user","content":"Hello, introduce yourself in one sentence."}],
    "stream": true,
    "max_tokens": 12,
    "temperature": 0
  }'
```

### Validation Status

- the raw `4 / 16 / 4` SplitNN prototype already passed reference / ORT / local simulation validation
- multi-step ORT validation for prefix/suffix passed
- local integration of `ONNX prefix/suffix + middle server + OpenAI controller` passed
- both non-streaming and streaming OpenAI requests returned valid Chinese text
- `OmSplitEngine` has now been validated on the real Atlas board for non-streaming requests, including consecutive requests with cache reset between sessions
- the CUDA middle service has been extended to `max_len=16384` and benchmarked at about `9 tok/s` for the middle segment
- `max16384` prefix/suffix ONNX export passed ORT validation, and the corresponding `.om` models were compiled successfully
- the real board-side 16K OM controller completed a long-prompt test well beyond `256` tokens and returned decoded text successfully (example response: `收到。`)
- SSE streaming on the board-side OM backend still needs follow-up debugging, so non-streaming is the recommended path for current on-board testing
