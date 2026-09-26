#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/wio/Inference_Foundry
OUT=$ROOT/evidence/20260926_loop074_refill/run338
DATASET=/data/wio/vllm_ascend_26/datasets/GSM8K-in32768-num48-DeepSeek-V4-Flash-0731-w4a8-repeatRate0.9.jsonl
CONTAINER=vllm-ascend26-dsv4f-w4a8
PATCH=$ROOT/scripts/loop074_prefill_moe_graph_patch.py
mkdir -p "$OUT/runtime" "$OUT/fixture"
cleanup() {
    status=$?
    trap - EXIT
    bash /data/wio/vllm_ascend_26/scripts/99_stop_service.sh >"$OUT/stop.log" 2>&1 || status=1
    if test -f "$OUT/patch_install.json"; then
        python3 "$PATCH" restore --record "$OUT/patch_restore.json" >"$OUT/restore.log" 2>&1 || status=1
    fi
    python3 - "$PATCH" <<'PY' || status=1
import hashlib
from pathlib import Path
import sys
p=Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/worker/model_runner_v1.py')
want='004dbd0d5b1a5c3f9533fa1d12327bd0ae421fe24bb2674544b2c4a25d74aaba'
got=hashlib.sha256(p.read_bytes()).hexdigest()
if got!=want: raise SystemExit(f'source restore failed: {got}')
PY
    exit "$status"
}
trap cleanup EXIT
python3 "$PATCH" install --record "$OUT/patch_install.json" >"$OUT/install.log"
docker exec \
    -e MAX_MODEL_LEN=1048576 \
    -e RUN_TS=LOOP074-RUN338 \
    -e EXTREME_RUNTIME_RUN_DIR="$OUT/runtime" \
    -e EXTREME_RUNTIME_SERVE=1 \
    -e EXTREME_RUNTIME_RESERVE_TOKENS=1088 \
    -e EXTREME_NATIVE_TARGET_METADATA=1 \
    -e EXTREME_RUNTIME_TARGET_GRAPH=1 \
    -e EXTREME_DSPARK_SLOT_REFRESH=1 \
    -e EXTREME_TARGET_METADATA_STATIC_KV_MAX=1 \
    -e EXTREME_RUN338_DIR="$OUT/fixture" \
    -e OUT="$OUT" \
    -e DATASET="$DATASET" \
    "$CONTAINER" bash -lc '
        set -euo pipefail
        cd /data/wio/Inference_Foundry
        bash scripts/serve.sh >"$OUT/launcher.log" 2>&1
        for i in $(seq 1 180); do
            if curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1; then break; fi
            sleep 10
        done
        curl -fsS http://127.0.0.1:8080/health >/dev/null
        python3 scripts/bench.py --dataset "$DATASET" --out "$OUT/warmup48.json" --limit 48 --concurrency 12 --max-tokens 1024 >"$OUT/warmup48.log" 2>&1
        touch "$OUT/fixture/enable"
        python3 scripts/bench.py --dataset "$DATASET" --out "$OUT/measured12.json" --limit 12 --concurrency 12 --max-tokens 1024 >"$OUT/measured12.log" 2>&1
    '
