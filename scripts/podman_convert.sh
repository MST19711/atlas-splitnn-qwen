#!/usr/bin/env bash
set -euo pipefail
# Usage: MODEL_ONNX=model.onnx INPUT_SHAPE="a:1,2;b:3,4" OUTPUT_PREFIX=om_out/model bash scripts/podman_convert.sh

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
IMAGE=${IMAGE:-localhost/cann-atc-ubuntu22}

: "${MODEL_ONNX:?Need MODEL_ONNX}"
: "${INPUT_SHAPE:?Need INPUT_SHAPE (e.g. 'input_ids:1,32;attention_mask:1,32')}"
OUTPUT_PREFIX=${OUTPUT_PREFIX:-${PROJECT_DIR}/om_out/$(basename "${MODEL_ONNX}" .onnx)}
LOG_DIR=${LOG_DIR:-${PROJECT_DIR}/logs}
LOG_FILE=${LOG_FILE:-${LOG_DIR}/podman_convert_$(date +%Y%m%d_%H%M%S).log}

mkdir -p "${LOG_DIR}" "$(dirname "${OUTPUT_PREFIX}")"

echo "=== Podman ATC ==="
echo "Model : ${MODEL_ONNX}"
echo "Shape : ${INPUT_SHAPE}"
echo "Output: ${OUTPUT_PREFIX}.om"

podman run --rm --network=host --http-proxy=false \
  -e http_proxy= -e https_proxy= -e HTTP_PROXY= -e HTTPS_PROXY= \
  -v "${PROJECT_DIR}:/workspace:Z" \
  -w /workspace \
  "${IMAGE}" bash -lc "
set -eo pipefail
rm -rf /workspace/kernel_meta
mkdir -p /usr/local/Ascend
ln -sfn /workspace/cann8_install/ascend-toolkit/8.0.RC3 /usr/local/Ascend/CANN-1.84

export ASCEND_TOOLKIT_HOME=/workspace/cann8_install/ascend-toolkit/latest
export ASCEND_OPP_PATH=/workspace/cann8_install/ascend-toolkit/latest/opp
export ASCEND_AICPU_PATH=/workspace/cann8_install/ascend-toolkit/latest
export ASCEND_HOME_PATH=/workspace/cann8_install/ascend-toolkit/latest
export TOOLCHAIN_HOME=/workspace/cann8_install/ascend-toolkit/latest/toolkit
export PATH=/workspace/cann8_install/ascend-toolkit/latest/bin:/workspace/cann8_install/ascend-toolkit/latest/compiler/bin:/workspace/cann8_install/ascend-toolkit/latest/compiler/ccec_compiler/bin:/workspace/cann8_install/ascend-toolkit/latest/tools/ccec_compiler/bin:\$PATH
export LD_LIBRARY_PATH=/workspace/cann8_install/ascend-toolkit/latest/lib64:/workspace/cann8_install/ascend-toolkit/latest/tools/aml/lib64:/workspace/cann8_install/ascend-toolkit/latest/tools/aml/lib64/plugin:/workspace/cann8_install/ascend-toolkit/latest/x86_64-linux/devlib:/workspace/cann8_install/ascend-toolkit/latest/x86_64-linux/lib64:/workspace/cann8_install/ascend-toolkit/latest/compiler/lib64:\${LD_LIBRARY_PATH:-}
export PYTHONPATH=/workspace/cann8_install/ascend-toolkit/latest/python/site-packages:\${PYTHONPATH:-}

atc --model=/workspace/\$(basename \"${MODEL_ONNX}\") \
    --framework=5 \
    --output=/workspace/om_out/\$(basename \"${OUTPUT_PREFIX}\") \
    --input_format=ND \
    --input_shape='${INPUT_SHAPE}' \
    --soc_version=Ascend310B4 \
    --precision_mode=allow_fp32_to_fp16 \
    --log=info
echo '--- atc done ---'
ls -lh /workspace/om_out/\$(basename \"${OUTPUT_PREFIX}\").om
" 2>&1 | tee "${LOG_FILE}"

echo "Log : ${LOG_FILE}"
echo "Done."
