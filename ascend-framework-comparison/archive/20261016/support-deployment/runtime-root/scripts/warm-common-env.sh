#!/usr/bin/env bash
# Verified baseline communication settings; private MemFabric LOCAL-symbol overlay.
set -u
export PYTHONPATH="/data/wio/sglang-glm53-33-41-20261015/warm93-memcache-aisbench-20261015/sdk-mf-local:${PYTHONPATH:-}"

NOMC_ENV_MARK=warm93-common-env
ROOT=${REMOTE_ROOT:-/data/wio/sglang-glm53-33-41-20261015/warm93-memcache-aisbench-20261015}
T=$ROOT/logs
model=/model

nomc_env_fail() { echo "FATAL[$NOMC_ENV_MARK]: $*" >&2; return 1; }

# ---------------------------------------------------------------- CANN / ATB runtime
# Discovery order: the image's own env (which the container inherits) first, then the fixed
# install locations used by the 9.1.0 layout, then a version glob. Missing = abort.
CANN_ASCEND_HOME=${ASCEND_HOME_PATH:-}
CANN_TOOLKIT_SETENV=""
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  CANN_TOOLKIT_SETENV=/usr/local/Ascend/ascend-toolkit/set_env.sh
elif [ -n "$CANN_ASCEND_HOME" ] && [ -f "$CANN_ASCEND_HOME/set_env.sh" ]; then
  CANN_TOOLKIT_SETENV=$CANN_ASCEND_HOME/set_env.sh
else
  for candidate in /usr/local/Ascend/ascend-toolkit/*/set_env.sh /usr/local/Ascend/cann-*/set_env.sh; do
    [ -f "$candidate" ] && CANN_TOOLKIT_SETENV=$candidate && break
  done
fi
[ -n "$CANN_TOOLKIT_SETENV" ] || { nomc_env_fail "no CANN set_env.sh found (ASCEND_HOME_PATH='$CANN_ASCEND_HOME')"; exit 3; }

CANN_IR_SETENV=""
for candidate in "${CANN_ASCEND_HOME}/share/info/ascendnpu-ir/bin/set_env.sh" \
                 /usr/local/Ascend/cann-*/share/info/ascendnpu-ir/bin/set_env.sh; do
  [ -f "$candidate" ] && CANN_IR_SETENV=$candidate && break
done
ATB_SETENV=/usr/local/Ascend/nnal/atb/set_env.sh
[ -f "$ATB_SETENV" ] || { nomc_env_fail "ATB set_env.sh missing at $ATB_SETENV"; exit 3; }
export ATB_CXX_ABI=${ATB_CXX_ABI:-1}

set +u
# shellcheck disable=SC1090
source "$CANN_TOOLKIT_SETENV" || { set -u; nomc_env_fail "sourcing $CANN_TOOLKIT_SETENV failed"; exit 3; }
if [ -n "$CANN_IR_SETENV" ]; then
  # shellcheck disable=SC1090
  source "$CANN_IR_SETENV" || { set -u; nomc_env_fail "sourcing $CANN_IR_SETENV failed"; exit 3; }
fi
# shellcheck disable=SC1090
source "$ATB_SETENV" --cxx_abi="$ATB_CXX_ABI" || { set -u; nomc_env_fail "sourcing $ATB_SETENV failed"; exit 3; }
set -u

# MemFabric1.2.1 bundles libhcom.so and its wheel dependencies outside the
# default loader directories. Preserve the image's complete CANN/ATB paths.
MF_LIBRARY_DIRS=$(python3 - <<'PY'
import importlib.util
from pathlib import Path
spec=importlib.util.find_spec('memfabric_hybrid')
assert spec and spec.origin, 'MemFabric transport package missing'
p=Path(spec.origin).parent
paths=[p/'lib',p.parent/'memfabric_hybrid.libs']
print(':'.join(str(v) for v in paths if v.is_dir()))
PY
) || exit 3
export LD_LIBRARY_PATH="$MF_LIBRARY_DIRS:${LD_LIBRARY_PATH:-}"

# ---------------------------------------------------------------- proxies / offline
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ASCEND_LAUNCH_BLOCKING 2>/dev/null || true
export NO_PROXY='*' no_proxy='*' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRANSFORMERS_VERBOSITY=error

# ---------------------------------------------------------------- HCCL / comm (baseline-verified)
export HCCL_SOCKET_IFNAME=bond4.12 GLOO_SOCKET_IFNAME=bond4.12
export HCCL_OP_EXPANSION_MODE=AIV
export STREAMS_PER_DEVICE=32
export SGLANG_SET_CPU_AFFINITY=1 SGLANG_NPU_USE_MULTI_STREAM=1 SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

# ---------------------------------------------------------------- DeepEP (common part)
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1

# ---------------------------------------------------------------- MemFabric (native Ascend PD)
# transfer_engine.py reads ASCEND_MF_STORE_URL and ASCEND_MF_TRANSFER_PROTOCOL from the env.
export ASCEND_MF_STORE_URL=tcp://172.16.12.36:24946
export MF_CONFIG_STORE_URL="$ASCEND_MF_STORE_URL"
export ASCEND_MF_TRANSFER_PROTOCOL=sdma
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=3600
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=3600
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# ---------------------------------------------------------------- private caches
# multiprocessing.Manager uses AF_UNIX sockets below TMPDIR (108-byte limit).
# Keep persistent caches under ROOT; temporary IPC stays in the owned container.
export HF_HOME="$ROOT/cache/hf" XDG_CACHE_HOME="$ROOT/cache" TMPDIR=/tmp/sgl-warm93
export TRITON_CACHE_DIR="$ROOT/cache/triton" TORCHINDUCTOR_CACHE_DIR="$ROOT/cache/torchinductor" \
       TORCH_EXTENSIONS_DIR="$ROOT/cache/torch_extensions"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$T/logs" 2>/dev/null || true

echo "[$NOMC_ENV_MARK] node=${NOMC_NODE:-?} role=${NOMC_ROLE:-?} cannp=$CANN_TOOLKIT_SETENV ir=${CANN_IR_SETENV:-none} atb_cxx_abi=$ATB_CXX_ABI"
echo "[$NOMC_ENV_MARK] mf_store=$ASCEND_MF_STORE_URL protocol=$ASCEND_MF_TRANSFER_PROTOCOL hicache=ROLE_DEPENDENT"

echo "[$NOMC_ENV_MARK] bootstrap_timeout=$SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT waiting_timeout=$SGLANG_DISAGGREGATION_WAITING_TIMEOUT"

export PYTHONPATH=/data/wio/sglang-glm53-33-41-20261015/warm93-memcache-aisbench-20261015/sdk-official-13-a085-0649-v2/probe/overlay:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/data/wio/sglang-glm53-33-41-20261015/warm93-memcache-aisbench-20261015/sdk-official-13-a085-0649-v2/probe/overlay/memfabric_hybrid/lib:/data/wio/sglang-glm53-33-41-20261015/warm93-memcache-aisbench-20261015/sdk-official-13-a085-0649-v2/probe/overlay/memcache_hybrid/lib:${LD_LIBRARY_PATH:-}
