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
exec python3 -u /root/slm_deploy/gen_text_qwen35_splitnn.py \
  --server-url http://127.0.0.1:28080 \
  --prefix-model /root/slm_deploy/qwen3.5_split_prefix_max16384.om \
  --suffix-model /root/slm_deploy/qwen3.5_split_suffix_max16384.om \
  --max-len 16384 \
  "$@"
