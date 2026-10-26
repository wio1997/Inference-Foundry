#!/usr/bin/env bash
# warm93-memcache-aisbench-20261015 :: MemFabric store on node 36, private port 24946.
#
# Native Ascend PD needs the MemFabric store as TRANSPORT for P->D KV transfer; it is not a KV
# cache tier. It must be up BEFORE any P or D rank starts. Runs INSIDE the owned D container on
# node 36 (sglang-nightly-nomc-d-36-20261015).
set -u
ROOT=${REMOTE_ROOT:-/data/wio/sglang-glm53-33-41-20261015/warm93-memcache-aisbench-20261015}
L=$ROOT/logs/mf-store-24946.log
E=$ROOT/logs/mf-store-24946.exit
P=$ROOT/logs/mf-store-24946.pid
mkdir -p "$ROOT/logs"
source "$ROOT/scripts/warm-common-env.sh" || exit 3
echo $$ > "$P"; rm -f "$E"
export ASCEND_MF_STORE_URL=tcp://172.16.12.36:24946
echo "### MF_STORE start $(date -Is) url=$ASCEND_MF_STORE_URL host=$(hostname)" >> "$L"
python3 -m memfabric_hybrid.launch_ascend_mf_store >> "$L" 2>&1
echo $? > "$E"
echo "### MF_STORE done rc=$(cat "$E") $(date -Is)" >> "$L"
