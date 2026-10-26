#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/wio/Inference_Foundry
OUT=$ROOT/evidence/20260926_loop074_refill/run343
CONTAINER=vllm-ascend26-dsv4f-w4a8
CORE=/data/wio/vllm_ascend_26/framework/vllm/vllm/v1/engine/core.py
BASE_SHA=3ae1381a6af841e21058c825702382dc66faae45c950ac5acb8495d2d3d05aad
mkdir -p "$OUT/runtime"

# Do not displace an active workload.
for _ in $(seq 1 6); do
    snapshot=$(npu-smi info)
    max_used=$(printf '%s\n' "$snapshot" | grep -oE '[0-9]+ */ *65536' |
        awk -F/ '{gsub(/ /,"",$1); print $1}' | sort -rn | head -1)
    if printf '%s\n' "$snapshot" | grep -Eiq 'VLLM|python' \
       || [ "${max_used:-65536}" -gt 10240 ]; then
        echo 'NPUs occupied; refusing to start Run343' >&2
        exit 75
    fi
    sleep 10
done

cleanup() {
    status=$?
    trap - EXIT
    bash /data/wio/vllm_ascend_26/scripts/99_stop_service.sh >"$OUT/stop.log" 2>&1 || status=1
    if [ -f "$OUT/patch_install.json" ]; then
        python3 "$ROOT/scripts/loop074_run343_fence_patch.py" restore \
            --record "$OUT/patch_restore.json" >"$OUT/restore.log" 2>&1 || status=1
    fi
    actual=$(sha256sum "$CORE" | awk '{print $1}')
    [ "$actual" = "$BASE_SHA" ] || status=1
    exit "$status"
}
trap cleanup EXIT

python3 "$ROOT/scripts/loop074_run343_fence_patch.py" install \
    --record "$OUT/patch_install.json" >"$OUT/install.log"
docker exec -e EXTREME_RUN343_FENCE=1 "$CONTAINER" bash -lc \
    "cd '$ROOT' && ./scripts/run_loop036_static_e2e.sh '$OUT'" \
    >"$OUT/driver.log" 2>&1
