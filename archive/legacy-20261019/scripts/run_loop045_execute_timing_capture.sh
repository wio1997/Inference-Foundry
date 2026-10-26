#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/wio/Inference_Foundry
OUT=$ROOT/evidence/20260925_loop045_boundary/run158
DATASET=/data/wio/vllm_ascend_26/datasets/GSM8K-in32768-num48-DeepSeek-V4-Flash-0731-w4a8-repeatRate0.9.jsonl
CONTAINER=vllm-ascend26-dsv4f-w4a8
mkdir -p "$OUT/runtime" "$OUT/boundary" "$OUT/exec_timing"
cleanup() {
    python3 "$ROOT/scripts/loop045_execute_timing_patch.py" restore --record "$OUT/timing_patch_restore.json" >"$OUT/timing_restore.log" 2>&1 || true
    python3 "$ROOT/scripts/loop045_boundary_patch.py" restore --record "$OUT/boundary_patch_restore.json" >"$OUT/boundary_restore.log" 2>&1 || true
    bash /data/wio/vllm_ascend_26/scripts/99_stop_service.sh >"$OUT/stop.log" 2>&1 || true
}
trap cleanup EXIT
python3 "$ROOT/scripts/loop045_boundary_patch.py" install --record "$OUT/boundary_patch_install.json" >"$OUT/boundary_install.log"
python3 "$ROOT/scripts/loop045_execute_timing_patch.py" install --record "$OUT/timing_patch_install.json" >"$OUT/timing_install.log"
docker exec \
    -e MAX_MODEL_LEN=1048576 \
    -e RUN_TS=LOOP045-RUN158 \
    -e EXTREME_RUNTIME_RUN_DIR="$OUT/runtime" \
    -e EXTREME_RUNTIME_SERVE=1 \
    -e EXTREME_RUNTIME_RESERVE_TOKENS=1088 \
    -e EXTREME_NATIVE_TARGET_METADATA=1 \
    -e EXTREME_RUNTIME_TARGET_GRAPH=1 \
    -e EXTREME_DSPARK_SLOT_REFRESH=1 \
    -e EXTREME_TARGET_METADATA_STATIC_KV_MAX=1 \
    -e EXTREME_BOUNDARY_DIR="$OUT/boundary" \
    -e EXTREME_EXEC_TIMING_DIR="$OUT/exec_timing" \
    -e OUT="$OUT" -e DATASET="$DATASET" \
    "$CONTAINER" bash -lc '
        set -e
        cd /data/wio/Inference_Foundry
        bash scripts/serve.sh >"$OUT/launcher.log" 2>&1
        for i in $(seq 1 180); do
            if curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1; then break; fi
            sleep 10
        done
        curl -fsS http://127.0.0.1:8080/health >/dev/null
        python3 scripts/bench.py --dataset "$DATASET" --out "$OUT/warmup48.json" --limit 48 --concurrency 12 --max-tokens 1024 >"$OUT/warmup48.log" 2>&1
        python3 scripts/bench.py --dataset "$DATASET" --out "$OUT/measured12.json" --limit 12 --concurrency 12 --max-tokens 1024 >"$OUT/measured12.log" 2>&1
    '
