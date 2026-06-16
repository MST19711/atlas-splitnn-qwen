#!/usr/bin/env bash
set -eo pipefail

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  set +u
  . /usr/local/Ascend/ascend-toolkit/set_env.sh
  set -u
elif [ -f /usr/local/Ascend/ascend-toolkit/latest/set_env.sh ]; then
  set +u
  . /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
  set -u
else
  echo "Ascend toolkit set_env.sh not found" >&2
  exit 1
fi

cd /root/slm_deploy
exec python3 -u controller/openai_split_controller.py \
  --host 0.0.0.0 \
  --port 8000 \
  --engine om \
  --model-name qwen3.5-split-4-16-4-om-16k \
  --remote-model-name Qwen3.5-0.8B-split-4-16-4 \
  --tokenizer-dir /root/slm_deploy \
  --server-url http://127.0.0.1:28080 \
  --max-len 16384 \
  --prefix-om /root/slm_deploy/qwen3.5_split_prefix_max16384.om \
  --suffix-om /root/slm_deploy/qwen3.5_split_suffix_max16384.om \
  --checksum \
  "$@"
