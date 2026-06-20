#!/usr/bin/env bash
# Shared Ascend toolkit environment setup for board scripts.
# Source this at the top of any board script that needs ACL/CANN runtime.
#
# Usage: source "$(dirname "$0")/setup_ascend_env.sh"

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  . /usr/local/Ascend/ascend-toolkit/set_env.sh
elif [ -f /usr/local/Ascend/ascend-toolkit/latest/set_env.sh ]; then
  . /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
else
  echo "ERROR: Ascend toolkit set_env.sh not found" >&2
  exit 1
fi
