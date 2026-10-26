#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/wio/Inference_Foundry
OUT=$ROOT/evidence/20260925_loop048_prefill/run188
DATASET=/data/wio/vllm_ascend_26/datasets/GSM8K-in32768-num48-DeepSeek-V4-Flash-0731-w4a8-repeatRate0.9.jsonl
CONTAINER=vllm-ascend26-dsv4f-w4a8
mkdir -p "$OUT/forward" "$OUT/runtime"
cleanup() {
  rm -f "$OUT/tag.txt" "$OUT/wait.txt"
  python3 "$ROOT/scripts/loop048_admission_run188_patch.py" restore --record "$OUT/patch_restore.json" >"$OUT/restore.log" 2>&1 || true
  bash /data/wio/vllm_ascend_26/scripts/99_stop_service.sh >"$OUT/stop.log" 2>&1 || true
}
trap cleanup EXIT
python3 "$ROOT/scripts/loop048_admission_run188_patch.py" install --record "$OUT/patch_install.json" >"$OUT/install.log"
docker exec \
  -e MAX_MODEL_LEN=1048576 \
  -e RUN_TS=LOOP048-RUN188 \
  -e EXTREME_RUNTIME_RUN_DIR="$OUT/runtime" \
  -e EXTREME_RUNTIME_SERVE=1 \
  -e EXTREME_RUNTIME_RESERVE_TOKENS=1088 \
  -e EXTREME_NATIVE_TARGET_METADATA=1 \
  -e EXTREME_RUNTIME_TARGET_GRAPH=1 \
  -e EXTREME_DSPARK_SLOT_REFRESH=1 \
  -e EXTREME_TARGET_METADATA_STATIC_KV_MAX=1 \
  -e EXTREME_RUN188_TAG_FILE="$OUT/tag.txt" \
  -e EXTREME_RUN188_FORWARD_DIR="$OUT/forward" \
  -e EXTREME_RUN188_WAIT_FILE="$OUT/wait.txt" \
  -e EXTREME_RUN188_WAIT_TRACE="$OUT/wait.jsonl" \
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
    echo A >"$OUT/tag.txt"
    python3 scripts/bench.py --dataset "$DATASET" --out "$OUT/A.json" --limit 12 --concurrency 12 --max-tokens 1024 >"$OUT/A.log" 2>&1
    echo B >"$OUT/tag.txt"
    touch "$OUT/wait.txt"
    python3 scripts/bench.py --dataset "$DATASET" --out "$OUT/B.json" --limit 12 --concurrency 12 --max-tokens 1024 >"$OUT/B.log" 2>&1
    rm -f "$OUT/wait.txt"
    echo A2 >"$OUT/tag.txt"
    python3 scripts/bench.py --dataset "$DATASET" --out "$OUT/A2.json" --limit 12 --concurrency 12 --max-tokens 1024 >"$OUT/A2.log" 2>&1
    rm -f "$OUT/tag.txt"
  '
