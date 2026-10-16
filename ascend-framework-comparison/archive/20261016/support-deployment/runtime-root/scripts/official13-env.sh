#!/bin/bash
set -e
source /data/wio/sglang-glm53-33-41-20261015/warm93-memcache-aisbench-20261015/scripts/warm-common-env.sh
unset LD_PRELOAD
export PYTHONPATH=/data/wio/sglang-glm53-33-41-20261015/warm93-memcache-aisbench-20261015/sdk-official-13-a085-0649-v2/probe/overlay:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/data/wio/sglang-glm53-33-41-20261015/warm93-memcache-aisbench-20261015/sdk-official-13-a085-0649-v2/probe/overlay/memfabric_hybrid/lib:/data/wio/sglang-glm53-33-41-20261015/warm93-memcache-aisbench-20261015/sdk-official-13-a085-0649-v2/probe/overlay/memcache_hybrid/lib:${LD_LIBRARY_PATH:-}
exec "$@"
