#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(dirname "$0")"
source "$SCRIPT_DIR/setup_ascend_env.sh"
exec python3 -u controller/openai_controller.py \
  --host 0.0.0.0 --port 8000 \
  --backend qwen35_kvcache_om \
  --model-name qwen3.5-0.8B-kvcache-om-4096 \
  --model-path /root/slm_deploy/model \
  --model-om /root/slm_deploy/qwen3.5_kvcache_max4096.om \
  --tokenizer-dir /root/slm_deploy/model \
  --max-len 4096 \
  --cache-max-bytes 104857600 \
  "$@"
