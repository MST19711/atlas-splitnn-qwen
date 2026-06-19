#!/usr/bin/env bash
# Compile ACL single-op MatMul OM for bound_embed_head mode.
# Usage: bash scripts/compile_head_matmul.sh <bound_asset_dir>
# Example: bash scripts/compile_head_matmul.sh om_out/qwen3.5_4b_bound_embed_head
set -eo pipefail

BOUND_DIR="${1:?usage: $0 <bound_asset_dir>}"
CONFIG_FILE="${BOUND_DIR}/op_models_config/acl_op.json"
OUTPUT_DIR="${BOUND_DIR}/op_models"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: acl_op.json not found at $CONFIG_FILE" >&2
    echo "Run export_qwen35_bound_embed_head.py first to generate it." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Compiling ACL single-op MatMul for $BOUND_DIR..."

podman run --rm --network=host --http-proxy=false \
    -e http_proxy= -e https_proxy= -e HTTP_PROXY= -e HTTPS_PROXY= \
    -v "${PROJECT_DIR}:/workspace:Z" \
    -w "/workspace/$(realpath --relative-to="$PROJECT_DIR" "$BOUND_DIR")/op_models_config" \
    localhost/cann-atc-rocky:v7 \
    bash -lc 'atc --singleop=acl_op.json \
        --soc_version=Ascend310B4 \
        --output=../op_models'

echo "Done. Output in $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"/*.om 2>/dev/null
