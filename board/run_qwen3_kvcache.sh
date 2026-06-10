#!/usr/bin/env bash
set -euo pipefail

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  . /usr/local/Ascend/ascend-toolkit/set_env.sh
elif [ -f /usr/local/Ascend/ascend-toolkit/latest/set_env.sh ]; then
  . /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
else
  echo "Ascend toolkit set_env.sh not found" >&2
  exit 1
fi

cd /root/slm_deploy
exec python3 -u gen_text_qwen3_kvcache.py \
  --model /root/slm_deploy/qwen3_kvcache_max256_cann7.om \
  --tokenizer-dir /root/slm_deploy \
  "$@"
