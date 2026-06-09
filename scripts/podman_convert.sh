#!/usr/bin/env bash
set -euo pipefail
# Usage: MODEL_ONNX=model.onnx INPUT_SHAPE="a:1,2;b:3,4" [OUTPUT_PREFIX=om_out/model] bash scripts/podman_convert.sh

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
IMAGE=${IMAGE:-localhost/cann-atc-ubuntu22:v4}

: "${MODEL_ONNX:?Need MODEL_ONNX}"
: "${INPUT_SHAPE:?Need INPUT_SHAPE}"
OUTPUT_PREFIX=${OUTPUT_PREFIX:-${PROJECT_DIR}/om_out/$(basename "${MODEL_ONNX}" .onnx)}
LOG_DIR=${LOG_DIR:-${PROJECT_DIR}/logs}
LOG_FILE=${LOG_FILE:-${LOG_DIR}/podman_convert_$(date +%Y%m%d_%H%M%S).log}

mkdir -p "${LOG_DIR}" "$(dirname "${OUTPUT_PREFIX}")"

MODEL_REL=$(realpath --relative-to="${PROJECT_DIR}" "${MODEL_ONNX}")
OUTPUT_REL=$(realpath --relative-to="${PROJECT_DIR}" "${OUTPUT_PREFIX}")

echo "=== Podman ATC ==="
echo "Model : ${MODEL_ONNX}"
echo "Output: ${OUTPUT_PREFIX}.om"

podman run --rm --network=host --http-proxy=false \
  -e http_proxy= -e https_proxy= -e HTTP_PROXY= -e HTTPS_PROXY= \
  -v "${PROJECT_DIR}:/workspace:Z" \
  -w /workspace \
  -e _ATC_MODEL="/workspace/${MODEL_REL}" \
  -e _ATC_OUTPUT="/workspace/${OUTPUT_REL}" \
  -e _ATC_SHAPE="${INPUT_SHAPE}" \
  "${IMAGE}" bash -lc '
set -eo pipefail
rm -rf /workspace/kernel_meta

echo "--- atc start ---"
atc --model=${_ATC_MODEL} \
    --framework=5 \
    --output=${_ATC_OUTPUT} \
    --input_format=ND \
    --input_shape="${_ATC_SHAPE}" \
    --soc_version=Ascend310B4 \
    --precision_mode=allow_fp32_to_fp16 \
    --log=info
echo "--- atc done ---"
ls -lh ${_ATC_OUTPUT}.om
' 2>&1 | tee "${LOG_FILE}"

echo "Log : ${LOG_FILE}"
echo "Done."
