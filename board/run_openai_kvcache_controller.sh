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
