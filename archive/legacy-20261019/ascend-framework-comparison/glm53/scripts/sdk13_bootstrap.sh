#!/bin/bash
# Read-only project SDK overlay; preserve native CANN/driver paths.
set -eu
glm53_sdk_root=/opt/glm53/sdk13
test -f "$glm53_sdk_root/_glm53_overlay_manifest.json"
export PYTHONPATH="$glm53_sdk_root:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$glm53_sdk_root/memfabric_hybrid/lib:$glm53_sdk_root/memcache_hybrid/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1
exec python3 -c "$1" "$2" "$3" "$4"
