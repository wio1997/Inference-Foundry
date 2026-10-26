#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/wio/Inference_Foundry
OUT=$ROOT/evidence/20260925_loop047_tail/run179
DATASET=/data/wio/vllm_ascend_26/datasets/GSM8K-in32768-num48-DeepSeek-V4-Flash-0731-w4a8-repeatRate0.9.jsonl
CONTAINER=vllm-ascend26-dsv4f-w4a8
mkdir -p "$OUT/slope"
cleanup() {
    rm -f "$OUT/arm.txt"
    python3 "$ROOT/scripts/loop047_target_drift_patch.py" restore --record "$OUT/patch_restore.json" >"$OUT/restore.log" 2>&1 || true
    bash /data/wio/vllm_ascend_26/scripts/99_stop_service.sh >"$OUT/stop.log" 2>&1 || true
}
trap cleanup EXIT
python3 "$ROOT/scripts/loop047_target_drift_patch.py" install --record "$OUT/patch_install.json" >"$OUT/install.log"
docker exec \
    -e MAX_MODEL_LEN=1048576 \
    -e RUN_TS=LOOP047-RUN179 \
    -e EXTREME_RUN179_SLOPE_DIR="$OUT/slope" \
    -e EXTREME_RUN179_ARM_FILE="$OUT/arm.txt" \
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
        for epoch in 12a 8a 6a 4a 1a 12b; do
            n="${epoch%?}"
            tag="${epoch: -1}"
            python3 scripts/bench.py --dataset "$DATASET" --out "$OUT/warmup${epoch}.json" --limit "$n" --concurrency "$n" --max-tokens 1024 >"$OUT/warmup${epoch}.log" 2>&1
            echo "$((n*8)):$tag" >"$OUT/arm.txt"
            python3 scripts/bench.py --dataset "$DATASET" --out "$OUT/measured${epoch}.json" --limit "$n" --concurrency "$n" --max-tokens 1024 >"$OUT/measured${epoch}.log" 2>&1
            rm -f "$OUT/arm.txt"
        done
    '
