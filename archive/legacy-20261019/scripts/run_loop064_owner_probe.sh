#!/usr/bin/env bash
set -euo pipefail
RUN_ID=${1:?run id required}
ROOT=/data/wio/Inference_Foundry
OUT="$ROOT/evidence/20260926_loop064_cp/$RUN_ID"
DATASET=/data/wio/vllm_ascend_26/datasets/GSM8K-in32768-num48-DeepSeek-V4-Flash-0731-w4a8-repeatRate0.9.jsonl
mkdir -p "$OUT/runtime" "$OUT/ownership"
SERVICE_STARTED=0
cleanup() {
  if [ "$SERVICE_STARTED" = 1 ]; then
    bash /data/wio/vllm_ascend_26/scripts/99_stop_service.sh >"$OUT/stop.log" 2>&1 || true
  fi
}
trap cleanup EXIT
for _ in $(seq 1 6); do
  snapshot=$(npu-smi info)
  max_used=$(printf '%s\n' "$snapshot" | grep -oE '[0-9]+ */ *65536' | awk -F/ '{gsub(/ /,"",$1); print $1}' | sort -rn | head -1)
  if printf '%s\n' "$snapshot" | grep -Eiq 'VLLM|python' || [ "${max_used:-65536}" -gt 10240 ]; then
    echo 'NPUs occupied; refusing ownership probe' >&2
    exit 75
  fi
  sleep 10
done
SERVICE_STARTED=1
docker exec -e MAX_MODEL_LEN=1048576 -e RUN_TS="LOOP064OWNER-${RUN_ID}" \
  -e EXTREME_RUNTIME_RUN_DIR="$OUT/runtime" -e EXTREME_RUNTIME_SERVE=1 \
  -e EXTREME_RUNTIME_RESERVE_TOKENS=1088 -e EXTREME_NATIVE_TARGET_METADATA=1 \
  -e EXTREME_RUNTIME_TARGET_GRAPH=1 -e EXTREME_DSPARK_SLOT_REFRESH=1 \
  -e EXTREME_TARGET_METADATA_STATIC_KV_MAX=1 \
  -e EXTREME_SCHEDULE_NEXT_TARGET_METADATA=off \
  -e EXTREME_TARGET_PAGE_AUDIT_DIR="$OUT/ownership" \
  -e EXTREME_TARGET_PAGE_AUDIT_CYCLES=0 -e EXTREME_CP_OWNERSHIP_PROBE=1 \
  -e OUT="$OUT" -e DATASET="$DATASET" vllm-ascend26-dsv4f-w4a8 bash -lc '
    set -euo pipefail
    cd /data/wio/Inference_Foundry
    bash scripts/serve.sh >"$OUT/launcher.log" 2>&1
    for i in $(seq 1 180); do
      if curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1; then break; fi
      sleep 10
    done
    curl -fsS http://127.0.0.1:8080/health >/dev/null
    python3 scripts/bench.py --dataset "$DATASET" --out "$OUT/warmup48.json" --limit 48 --concurrency 12 --max-tokens 1024 >"$OUT/warmup48.log" 2>&1
    python3 scripts/bench.py --dataset "$DATASET" --out "$OUT/bench12.json" --limit 12 --concurrency 12 --max-tokens 1024 >"$OUT/bench12.log" 2>&1
  '
