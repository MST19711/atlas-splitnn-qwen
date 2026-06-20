#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(dirname "$0")"
source "$SCRIPT_DIR/setup_ascend_env.sh"

[ -d /root/slm_deploy ] || { echo "ERROR: /root/slm_deploy not found" >&2; exit 1; }
cd /root/slm_deploy
exec python3 -u controller/openai_controller.py \
  --host 0.0.0.0 \
  --port 8000 \
  --backend qwen35_kvcache_om \
  --model-name qwen3.5-0.8B-kvcache-om \
  --model-path /root/slm_deploy \
  --model-om /root/slm_deploy/qwen3.5_kvcache_max256.om \
  --tokenizer-dir /root/slm_deploy \
  --max-len 256 \
  "$@"
