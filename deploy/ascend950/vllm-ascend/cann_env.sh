#!/usr/bin/env bash
# Source inside the pinned official image.
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann-9.1.0/share/info/ascendnpu-ir/bin/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi="${ATB_CXX_ABI:-1}"
export PYTHONPATH="/usr/local/python3.12.13/lib/python3.12/site-packages:${PYTHONPATH:-}"
filter_workspace_paths() {
  local item result='' items=()
  IFS=: read -r -a items <<< "$1"
  for item in "${items[@]}"; do
    case "$item" in ''|*/vllm-workspace*) continue ;; esac
    result="${result:+$result:}$item"
  done
  printf '%s' "$result"
}
export PYTHONPATH="$(filter_workspace_paths "${PYTHONPATH:-}")"
export LD_LIBRARY_PATH="$(filter_workspace_paths "${LD_LIBRARY_PATH:-}")"
set -u
