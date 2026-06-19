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
  --backend splitnn_bound_embed_head \
  --model-name qwen3.5-2b-split-0-24-0-om \
  --remote-model-name Qwen3.5-2B-split-0-24-0 \
  --tokenizer-dir /root/slm_deploy/model_2b \
  --server-url http://127.0.0.1:28080 \
  --max-len 8192 \
  --split 0,24 \
  --bound-asset-dir /root/slm_deploy/qwen3.5_2b_bound_embed_head \
  --checksum \
  "$@"
