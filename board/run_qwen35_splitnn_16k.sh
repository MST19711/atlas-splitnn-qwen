#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(dirname "$0")"
source "$SCRIPT_DIR/setup_ascend_env.sh"

[ -d /root/slm_deploy ] || { echo "ERROR: /root/slm_deploy not found" >&2; exit 1; }
cd /root/slm_deploy
exec python3 -u /root/slm_deploy/gen_text_qwen35_splitnn.py \
  --server-url http://127.0.0.1:28080 \
  --prefix-model /root/slm_deploy/qwen3.5_split_prefix_max16384.om \
  --suffix-model /root/slm_deploy/qwen3.5_split_suffix_max16384.om \
  --max-len 16384 \
  "$@"
