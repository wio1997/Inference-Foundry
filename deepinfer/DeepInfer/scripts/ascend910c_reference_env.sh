#!/usr/bin/env bash
# Source this file inside a dedicated container. No host/global changes.
export DEEPINFER_ACCELERATOR=ascend
export DEEPINFER_ASCEND_910C_REFERENCE=1
_deepinfer_hcc="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.0}/tools/hcc"
if [[ ! -x "$_deepinfer_hcc/bin/aarch64-target-linux-gnu-g++" ]]; then
    echo 'CANN HCC toolchain unavailable; provide a GCC >= 13 CXX/CC and matching libstdc++.' >&2
    return 1 2>/dev/null || exit 1
fi
export CC="$_deepinfer_hcc/bin/aarch64-target-linux-gnu-gcc"
export CXX="$_deepinfer_hcc/bin/aarch64-target-linux-gnu-g++"
export LD_LIBRARY_PATH="$_deepinfer_hcc/aarch64-target-linux-gnu/lib64:${LD_LIBRARY_PATH:-}"
export MAX_JOBS="${MAX_JOBS:-2}"
unset _deepinfer_hcc
