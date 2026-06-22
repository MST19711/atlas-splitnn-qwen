#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR"

MODEL_PATH="${MODEL_PATH:-model/Qwen3.5-4B}"
MAX_LEN="${MAX_LEN:-16384}"
SPLIT="${SPLIT:-0,32}"
PORT="${PORT:-28080}"
HOST="${HOST:-0.0.0.0}"

export KMP_DUPLICATE_LIB_OK=TRUE

exec pixi run python server/qwen35_split_service.py \
  --host "$HOST" \
  --port "$PORT" \
  --model-path "$MODEL_PATH" \
  --max-len "$MAX_LEN" \
  --split "$SPLIT" \
  --device mps \
  --session-timeout-sec 300 \
  --max-sessions 8 \
  "$@"
