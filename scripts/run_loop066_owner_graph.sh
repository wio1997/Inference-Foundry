#!/usr/bin/env bash
set -euo pipefail
RUN_ID=${1:?run id required}
ROOT=/data/wio/Inference_Foundry
OUT="$ROOT/evidence/20260926_loop066_owner/$RUN_ID"
DATASET=/data/wio/vllm_ascend_26/datasets/GSM8K-in32768-num48-DeepSeek-V4-Flash-0731-w4a8-repeatRate0.9.jsonl
PATCH="$ROOT/scripts/loop066_owner_graph_patch.py"
mkdir -p "$OUT/runtime" "$OUT/fixture"
ACTIVE="$OUT/active"
: > "$ACTIVE"
SERVICE_STARTED=0
cleanup() {
  rm -f "$ACTIVE"
  if [ "$SERVICE_STARTED" = 1 ]; then
    bash /data/wio/vllm_ascend_26/scripts/99_stop_service.sh >"$OUT/stop.log" 2>&1 || true
  fi
  if [ -f "$OUT/patch.json" ]; then
    python3 "$PATCH" restore --record "$OUT/patch.json" >"$OUT/restore.log" 2>&1 || true
  fi
}
trap cleanup EXIT
for _ in $(seq 1 6); do
  snapshot=$(npu-smi info)
  max_used=$(printf '%s\n' "$snapshot" | grep -oE '[0-9]+ */ *65536' | awk -F/ '{gsub(/ /,"",$1); print $1}' | sort -rn | head -1)
  if printf '%s\n' "$snapshot" | grep -Eiq 'VLLM|python' || [ "${max_used:-65536}" -gt 10240 ]; then
    echo 'NPUs occupied; refusing owner graph patch' >&2
    exit 75
  fi
  sleep 10
done
python3 "$PATCH" install --record "$OUT/patch.json" >"$OUT/install.log"
SERVICE_STARTED=1
docker exec -e MAX_MODEL_LEN=1048576 -e RUN_TS="LOOP066GRAPH-${RUN_ID}" \
  -e EXTREME_RUNTIME_RUN_DIR="$OUT/runtime" -e EXTREME_RUNTIME_SERVE=1 \
  -e EXTREME_RUNTIME_RESERVE_TOKENS=1088 -e EXTREME_NATIVE_TARGET_METADATA=1 \
  -e EXTREME_RUNTIME_TARGET_GRAPH=0 -e EXTREME_DSPARK_SLOT_REFRESH=1 \
  -e EXTREME_TARGET_METADATA_STATIC_KV_MAX=1 \
  -e EXTREME_SCHEDULE_NEXT_TARGET_METADATA=off \
  -e EXTREME_OWNER_CONSUMER_FIXTURE_DIR="$OUT/fixture" \
  -e EXTREME_OWNER_GRAPH_DIR="$OUT/graph" \
  -e OUT="$OUT" -e DATASET="$DATASET" vllm-ascend26-dsv4f-w4a8 bash -lc '
    set -euo pipefail
    cd /data/wio/Inference_Foundry
    bash scripts/serve.sh >"$OUT/launcher.log" 2>&1
    _owns_healthy_service() {
      for _pid in $(pgrep -x vllm || true); do
        if [ -r "/proc/$_pid/environ" ] &&
           grep -Fzxq "RUN_TS=$RUN_TS" "/proc/$_pid/environ"; then
          return 0
        fi
      done
      return 1
    }
    _ready=0
    for i in $(seq 1 180); do
      if [ ! -f "$OUT/active" ]; then exit 125; fi
      if curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1; then
        _owns_healthy_service || exit 126
        _ready=1
        break
      fi
      sleep 10
    done
    test "$_ready" = 1
    test -f "$OUT/active"
    _owns_healthy_service
    python3 scripts/bench.py --dataset "$DATASET" --out "$OUT/bench12.json" --limit 12 --concurrency 12 --max-tokens 1024 >"$OUT/bench12.log" 2>&1
    python3 scripts/check_loop066_owner_consumer.py --fixture-dir "$OUT/fixture" >"$OUT/fixture_check.json"
    python3 scripts/check_loop066_owner_graph.py --graph-dir "$OUT/graph" >"$OUT/graph_check.json"
  '
