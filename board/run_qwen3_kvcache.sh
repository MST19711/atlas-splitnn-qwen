#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(dirname "$0")"
source "$SCRIPT_DIR/setup_ascend_env.sh"

[ -d /root/slm_deploy ] || { echo "ERROR: /root/slm_deploy not found" >&2; exit 1; }
cd /root/slm_deploy
exec python3 -u gen_text_qwen3_kvcache.py \
  --model /root/slm_deploy/qwen3_kvcache_max256_cann7.om \
  --tokenizer-dir /root/slm_deploy \
  "$@"
