#!/usr/bin/env bash
set -euo pipefail
MODE=${1:?off, serial or overlap required}
RUN_ID=${2:?run id required}
case "$MODE" in off|serial|overlap) ;; *) exit 64 ;; esac
ROOT=/data/wio/Inference_Foundry
OUT="$ROOT/evidence/20260926_loop063_schedule/$RUN_ID"
DATASET=/data/wio/vllm_ascend_26/datasets/GSM8K-in32768-num48-DeepSeek-V4-Flash-0731-w4a8-repeatRate0.9.jsonl
mkdir -p "$OUT/runtime"
SERVICE_STARTED=0
cleanup() { if [ "$SERVICE_STARTED" = 1 ]; then bash /data/wio/vllm_ascend_26/scripts/99_stop_service.sh >"$OUT/stop.log" 2>&1 || true; fi; }
trap cleanup EXIT
ACTUAL=$(sha256sum /data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/models/deepseek_v4.py | awk '{print $1}')
[ "$ACTUAL" = 11dd3e983dc1d628bf42a95581f9617861471b1f658739db44c20df886147247 ]
if npu-smi info | grep -Eq 'VLLMWorker_TP|VLLM'; then echo 'NPUs already occupied' >&2; exit 75; fi
SERVICE_STARTED=1
docker exec -e MAX_MODEL_LEN=1048576 -e RUN_TS="LOOP063-${RUN_ID}" \
  -e EXTREME_RUNTIME_RUN_DIR="$OUT/runtime" -e EXTREME_RUNTIME_SERVE=1 \
  -e EXTREME_RUNTIME_RESERVE_TOKENS=1088 -e EXTREME_NATIVE_TARGET_METADATA=1 \
  -e EXTREME_RUNTIME_TARGET_GRAPH=1 -e EXTREME_DSPARK_SLOT_REFRESH=1 \
  -e EXTREME_TARGET_METADATA_STATIC_KV_MAX=1 \
  -e EXTREME_SCHEDULE_NEXT_TARGET_METADATA="$MODE" -e EXTREME_SCHEDULE_VERIFY=0 \
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
    python3 scripts/bench.py --dataset "$DATASET" --out "$OUT/bench48.json" --limit 48 --concurrency 12 --max-tokens 1024 >"$OUT/bench48.log" 2>&1
  '
