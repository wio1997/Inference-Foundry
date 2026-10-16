#!/usr/bin/env bash
set -euo pipefail
if [[ ${RUN_BENCH:-0} != 1 ]]; then echo "Set RUN_BENCH=1 after owner approval" >&2; exit 64; fi
[[ -n ${RUN_NAME:-} ]] || { echo "Set RUN_NAME" >&2; exit 64; }
[[ ${PHASE:-} == warmup || ${PHASE:-} == full ]] || { echo "Set PHASE=warmup|full" >&2; exit 64; }
AIS_BENCH=${AIS_BENCH:-/usr/local/python3.12.13/bin/ais_bench}
PYTHON=${PYTHON:-/usr/local/python3.12.13/bin/python}
ROOT=${ROOT:-/data/tiankuan/aisbench/warm_w2/codex_4p2d_20261008}
METRICS_SNAPSHOT_CMD=${METRICS_SNAPSHOT_CMD:-"$PYTHON $ROOT/scripts/capture_metrics.py"}
export PATH="/usr/local/python3.12.13/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export no_proxy='*' NO_PROXY='*'
[[ -x "$AIS_BENCH" && -x "$PYTHON" ]] || { echo "Missing AISBench CLI/Python" >&2; exit 2; }
[[ -s "$ROOT/manifest.json" && -s "$ROOT/runs/$RUN_NAME/data/gsm8k/test.jsonl" ]] || { echo "Prepared run or manifest missing" >&2; exit 2; }
[[ -f "$ROOT/scripts/capture_metrics.py" && -f "$ROOT/scripts/analyze_run.py" && -f "$ROOT/scripts/analyze_metrics.py" ]] || { echo "Missing metrics/analysis tools" >&2; exit 2; }
RUN_DIR="$ROOT/runs/$RUN_NAME"
mkdir -p "$ROOT/locks"
LOCK="$ROOT/locks/benchmark.lock"
mkdir "$LOCK" 2>/dev/null || { echo "Phase lock exists: $LOCK" >&2; exit 2; }
cleanup_lock() { rmdir "$LOCK" 2>/dev/null || true; }
trap cleanup_lock EXIT

abort_if_aisbench_active() {
  local f cmd
  for f in /proc/[0-9]*/cmdline; do
    cmd=$(tr '\0' ' ' < "$f" 2>/dev/null || true)
    case "$cmd" in *"aisbench_test.py"*|*"/ais_bench "*|*"/bin/ais_bench"*) echo "ABORT active AISBench: $f $cmd" >&2; return 1 ;; esac
  done
}
snapshot_metrics() {
  local label=$1 out="$RUN_DIR/metrics/${1}.metrics"
  mkdir -p "$RUN_DIR/metrics"
  SNAPSHOT_OUT="$out" RUN_ROOT="$ROOT" RUN_NAME="$RUN_NAME" bash -c "$METRICS_SNAPSHOT_CMD" > "$out.capture.log" 2>&1
}

if [[ $PHASE == warmup ]]; then
  [[ ! -e "$RUN_DIR/WARMUP_STARTED" && ! -e "$RUN_DIR/WARMUP_FINISHED" ]] || { echo "Duplicate warmup refused" >&2; exit 2; }
  cleanup_warmup() { local rc=$?; printf '%s\n' "$rc" > "$RUN_DIR/WARMUP_EXIT_CODE"; cleanup_lock; }
  trap cleanup_warmup EXIT
  abort_if_aisbench_active
  date -u +%FT%TZ > "$RUN_DIR/WARMUP_STARTED"
  snapshot_metrics before_warmup
  WARM_OUT="$RUN_DIR/outputs/prefix_warmup"
  mkdir -p "$WARM_OUT"
  "$AIS_BENCH" --models prefix_warmup --datasets prefix_warmup_dataset \
    --config-dir "$ROOT/shared/prefix_warmup/configs" --work-dir "$WARM_OUT" \
    --mode perf --summarizer default_perf --num-warmups 0 \
    > "$WARM_OUT/ais_bench.stdout.log" 2>&1
  DETAILS=$(find "$WARM_OUT" -name gsm8k_details.jsonl -type f -print)
  [[ $(printf '%s\n' "$DETAILS" | wc -l | tr -d ' ') == 1 ]] || { echo "Expected one warmup detail JSONL" >&2; exit 2; }
  "$PYTHON" - "$DETAILS" <<'PY'
import json,sys
r=[json.loads(x) for x in open(sys.argv[1]) if x.strip()]
assert len(r)==16 and all(x.get('success') for x in r)
assert all(x.get('output_tokens')==1 for x in r)
print('warmup_successes=16 output_tokens=1')
PY
  snapshot_metrics after_warmup
  date -u +%FT%TZ > "$RUN_DIR/WARMUP_FINISHED"
  echo "Warmup done. Confirm P4/D16 counter coverage before PHASE=full."
  exit 0
fi

[[ -e "$RUN_DIR/WARMUP_FINISHED" ]] || { echo "Run PHASE=warmup first" >&2; exit 2; }
[[ ${WARMUP_APPROVED:-0} == 1 ]] || { echo "Set WARMUP_APPROVED=1 after reviewing 16-row success and P4/D16 coverage" >&2; exit 64; }
[[ ! -e "$RUN_DIR/FULL_STARTED" && ! -e "$RUN_DIR/FULL_FINISHED" ]] || { echo "Duplicate full run refused" >&2; exit 2; }
case "$RUN_NAME" in
  sat_c16_pilot) C=16; N=384;; sat_c32_pilot) C=32; N=384;;
  sat_c64_pilot) C=64; N=384;; sat_c96_pilot) C=96; N=384;;
  rate093_c16_pilot) C=16; N=48;; rate093_c32_pilot) C=32; N=48;;
  *) echo "Unknown RUN_NAME: $RUN_NAME" >&2; exit 2;;
esac

SAMPLER_PID=
AFTER_SNAPSHOT_DONE=0
stop_sampler() {
  if [[ -n ${SAMPLER_PID:-} ]]; then touch "$RUN_DIR/metrics/STOP_SAMPLES"; wait "$SAMPLER_PID" || true; SAMPLER_PID=; fi
}
finish_full() {
  local rc=$?
  trap - EXIT
  stop_sampler
  if [[ -e "$RUN_DIR/FULL_STARTED" && $AFTER_SNAPSHOT_DONE != 1 ]]; then
    snapshot_metrics after_full || true
    AFTER_SNAPSHOT_DONE=1
  fi
  if [[ -e "$RUN_DIR/FULL_STARTED" ]]; then
    printf '%s\n' "$rc" > "$RUN_DIR/FULL_EXIT_CODE"
    if [[ $rc == 0 ]]; then date -u +%FT%TZ > "$RUN_DIR/FULL_FINISHED"; fi
  fi
  cleanup_lock
  exit "$rc"
}
trap finish_full EXIT
abort_if_aisbench_active
date -u +%FT%TZ > "$RUN_DIR/FULL_STARTED"
snapshot_metrics before_full
FULL_OUT="$RUN_DIR/outputs/full"
mkdir -p "$FULL_OUT" "$RUN_DIR/metrics"
rm -f "$RUN_DIR/metrics/STOP_SAMPLES"
SNAPSHOT_OUT="$RUN_DIR/metrics/samples.jsonl" METRICS_STOP_FILE="$RUN_DIR/metrics/STOP_SAMPLES" \
  RUN_ROOT="$ROOT" RUN_NAME="$RUN_NAME" "$PYTHON" "$ROOT/scripts/capture_metrics.py" \
  > "$RUN_DIR/metrics/sampler.stdout.log" 2>&1 &
SAMPLER_PID=$!
for _ in $(seq 1 15); do
  [[ -s "$RUN_DIR/metrics/samples.jsonl" ]] && break
  kill -0 "$SAMPLER_PID" 2>/dev/null || { echo "Metrics sampler died before first sample" >&2; exit 2; }
  sleep 1
done
[[ -s "$RUN_DIR/metrics/samples.jsonl" ]] || { echo "No initial metrics sample" >&2; exit 2; }
"$AIS_BENCH" --models "$RUN_NAME" --datasets "${RUN_NAME}_dataset" \
  --config-dir "$RUN_DIR/configs" --work-dir "$FULL_OUT" --mode perf \
  --summarizer default_perf --num-warmups 0 \
  > "$FULL_OUT/ais_bench.stdout.log" 2>&1
stop_sampler
snapshot_metrics after_full
AFTER_SNAPSHOT_DONE=1
"$PYTHON" "$ROOT/scripts/analyze_metrics.py" "$RUN_DIR" \
  > "$RUN_DIR/metrics/metrics_analysis.stdout.json" 2>&1
"$PYTHON" "$ROOT/scripts/analyze_run.py" "$RUN_DIR" --concurrency "$C" --expected-requests "$N" \
  > "$FULL_OUT/offline_analysis.stdout.json" 2>&1
"$PYTHON" - "$RUN_DIR/analysis.json" <<'PY'
import json,sys
assert json.load(open(sys.argv[1])).get('integrity_pass') is True, 'analysis integrity_pass=false'
PY
echo "Full phase finished. after_full snapshot and exit status are written by exit trap."
