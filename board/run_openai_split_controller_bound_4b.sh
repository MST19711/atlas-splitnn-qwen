#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(dirname "$0")"
source "$SCRIPT_DIR/setup_ascend_env.sh"

[ -d /root/slm_deploy ] || { echo "ERROR: /root/slm_deploy not found" >&2; exit 1; }
cd /root/slm_deploy

MODEL_NAME="${MODEL_NAME:-qwen3.5-4b-split-0-32-0-bound}"
REMOTE_MODEL_NAME="${REMOTE_MODEL_NAME:-Qwen3.5-4B-split-0-32-0}"
TOKENIZER_DIR="${TOKENIZER_DIR:-/root/slm_deploy/model_4b}"
SERVER_URL="${SERVER_URL:-http://127.0.0.1:28080}"
MAX_LEN="${MAX_LEN:-16384}"
SPLIT="${SPLIT:-0,32}"
BOUND_ASSET_DIR="${BOUND_ASSET_DIR:-/root/slm_deploy/qwen3.5_4b_bound_embed_head}"
PREFIX_OM="${PREFIX_OM:-}"
SUFFIX_OM="${SUFFIX_OM:-}"

ARGS=(
  --host 0.0.0.0
  --port 8000
  --backend splitnn_bound_embed_head
  --model-name "$MODEL_NAME"
  --remote-model-name "$REMOTE_MODEL_NAME"
  --tokenizer-dir "$TOKENIZER_DIR"
  --server-url "$SERVER_URL"
  --max-len "$MAX_LEN"
  --split "$SPLIT"
  --bound-asset-dir "$BOUND_ASSET_DIR"
  --checksum
)

if [ -n "$PREFIX_OM" ]; then
  ARGS+=(--prefix-om "$PREFIX_OM")
fi

if [ -n "$SUFFIX_OM" ]; then
  ARGS+=(--suffix-om "$SUFFIX_OM")
fi

exec python3 -u controller/openai_controller.py "${ARGS[@]}" "$@"
