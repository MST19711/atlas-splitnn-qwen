#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(dirname "$0")"
source "$SCRIPT_DIR/setup_ascend_env.sh"

[ -d /root/slm_deploy ] || { echo "ERROR: /root/slm_deploy not found" >&2; exit 1; }
cd /root/slm_deploy
exec python3 -u controller/openai_controller.py \
  --host 0.0.0.0 \
  --port 8000 \
  --backend splitnn_om \
  --model-name qwen3.5-split-4-16-4-om-16k \
  --remote-model-name Qwen3.5-0.8B-split-4-16-4 \
  --tokenizer-dir /root/slm_deploy \
  --server-url http://127.0.0.1:28080 \
  --max-len 16384 \
  --prefix-om /root/slm_deploy/qwen3.5_split_prefix_max16384.om \
  --suffix-om /root/slm_deploy/qwen3.5_split_suffix_max16384.om \
  --checksum \
  "$@"
