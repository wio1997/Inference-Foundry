#!/usr/bin/env bash
set -u
ROOT=/data/wio/sglang-glm53-33-41-20261015/warm93-memcache-aisbench-20261015
set +u
source /usr/local/Ascend/cann-9.1.0/set_env.sh || exit 3
source /usr/local/Ascend/cann-9.1.0/share/info/ascendnpu-ir/bin/set_env.sh || exit 3
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1 || exit 3
set -u
SDK_LIBS=$(python3 - <<'PY'
import importlib.util
from pathlib import Path
ps=[]
for name in ['memfabric_hybrid','memcache_hybrid']:
 p=Path(importlib.util.find_spec(name).origin).parent
 ps += [q for q in [p/'lib',p.parent/(name+'.libs')] if q.is_dir()]
print(':'.join(map(str,ps)))
PY
) || exit 3
export LD_LIBRARY_PATH="$SDK_LIBS:${LD_LIBRARY_PATH:-}"
export HCCL_SOCKET_IFNAME=bond4.12 GLOO_SOCKET_IFNAME=bond4.12
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_PROXY='*' no_proxy='*'
export TMPDIR=/tmp/sgl-warm93
mkdir -p "$TMPDIR"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
