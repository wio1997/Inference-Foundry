#!/usr/bin/env bash
# Start the E33 DeepInfer DP8 diagnostic service in the task-owned model container.
# The benchmark shape is unchanged from E27: 8192 tokens/rank, max_num_seqs=16,
# ctx 98304, DSpark5, ZOS and native graphs.  E33 only fixes the profile-control
# environment and adds a selectable container-side launcher.
#
# A detached `docker exec` can only prove that the process was started.  Capture
# success must be checked with tools/e33_profile_control.py against
# /profile/status; this script never claims that a profile was collected.
set -euo pipefail

NAME=${NAME:-performance-20261021-deepinfer-runtime}
PORT=${PORT:-18121}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG=${LOG:-/work/performance-20261021/logs/e33-di-$STAMP.log}

# Default stays the existing E27 container launcher.  The main controller can
# point this at a diagnostic launcher that emits the complete event set, without
# changing the host-side environment contract.
CONTAINER_LAUNCHER=${CONTAINER_LAUNCHER:-/work/performance-20261021/scripts/remote/run_adapter_e27_cache.sh}

# E33 profile-control environment.  PROFILE_ARM_AFTER and PROFILE_ARM_LEAD are
# consumed by run_adapter_e27_cache.sh today; the remaining values are forwarded
# for a diagnostic launcher that accepts the same ProfilePlan fields.
PROFILE_DIR=${PROFILE_DIR:-}
PROFILE_RUN_ID=${PROFILE_RUN_ID:-}
PROFILE_CASE_ID=${PROFILE_CASE_ID:-}
PROFILE_BASELINE_RUN_ID=${PROFILE_BASELINE_RUN_ID:-}
PROFILE_REQUEST_IDS=${PROFILE_REQUEST_IDS:-}
PROFILE_ARM_AFTER=${PROFILE_ARM_AFTER:-0}
PROFILE_ARM_LEAD=${PROFILE_ARM_LEAD:-2}
PROFILE_MODE=${PROFILE_MODE:-prompt}
PROFILE_STEPS=${PROFILE_STEPS:-4}
PROFILE_PROMPT_WINDOWS=${PROFILE_PROMPT_WINDOWS:-1}
PROFILE_DECODE_WINDOWS=${PROFILE_DECODE_WINDOWS:-1}
PROFILE_DECODE_TRIGGER=${PROFILE_DECODE_TRIGGER:-4}
PROFILE_PROMPT_INTERVAL=${PROFILE_PROMPT_INTERVAL:-100}
PROFILE_PROMPT_TRIGGER_MODULI=${PROFILE_PROMPT_TRIGGER_MODULI:-100}
PROFILE_PROMPT_TRIGGER_K_FRACS=${PROFILE_PROMPT_TRIGGER_K_FRACS:-}
PROFILE_PROMPT_TRIGGER_REQUIRES_LOCAL=${PROFILE_PROMPT_TRIGGER_REQUIRES_LOCAL:-0}
PROFILE_LEVEL=${PROFILE_LEVEL:-Level1}
PROFILE_INITIAL_PROMPT=${PROFILE_INITIAL_PROMPT:-0}
PROFILE_IMMEDIATE_PROMPT=${PROFILE_IMMEDIATE_PROMPT:-0}
PROFILE_RECORD_SHAPES=${PROFILE_RECORD_SHAPES:-0}
PROFILE_WITH_STACK=${PROFILE_WITH_STACK:-0}

die() {
  printf 'E33_LAUNCHER_ERROR: %s\n' "$1" >&2
  exit "${2:-2}"
}

case "$PROFILE_ARM_AFTER" in
  ''|*[!0-9]*) die "PROFILE_ARM_AFTER must be a non-negative integer, got '$PROFILE_ARM_AFTER'" 2 ;;
esac
case "$PROFILE_ARM_LEAD" in
  ''|*[!0-9]*) die "PROFILE_ARM_LEAD must be a non-negative integer, got '$PROFILE_ARM_LEAD'" 2 ;;
esac
case "$PROFILE_MODE" in
  prompt|decode|both) ;;
  *) die "PROFILE_MODE must be prompt, decode or both, got '$PROFILE_MODE'" 2 ;;
esac
case "$PROFILE_STEPS" in
  ''|*[!0-9]*) die "PROFILE_STEPS must be a positive integer, got '$PROFILE_STEPS'" 2 ;;
esac
[ "$PROFILE_STEPS" -ge 1 ] || die "PROFILE_STEPS must be a positive integer, got '$PROFILE_STEPS'" 2
case "$PROFILE_PROMPT_INTERVAL" in
  ''|*[!0-9]*) die "PROFILE_PROMPT_INTERVAL must be a positive integer, got '$PROFILE_PROMPT_INTERVAL'" 2 ;;
esac
[ "$PROFILE_PROMPT_INTERVAL" -ge 1 ] || die "PROFILE_PROMPT_INTERVAL must be a positive integer, got '$PROFILE_PROMPT_INTERVAL'" 2
require_nonneg() {
  local name="$1" value="$2"
  case "$value" in
    ''|*[!0-9]*) die "$name must be a non-negative integer, got '$value'" 2 ;;
  esac
}
require_nonneg PROFILE_PROMPT_WINDOWS "$PROFILE_PROMPT_WINDOWS"
require_nonneg PROFILE_DECODE_WINDOWS "$PROFILE_DECODE_WINDOWS"
require_nonneg PROFILE_DECODE_TRIGGER "$PROFILE_DECODE_TRIGGER"

if [ -n "$PROFILE_DIR" ]; then
  [ -n "$PROFILE_RUN_ID" ] || die "PROFILE_DIR is set but PROFILE_RUN_ID is empty" 2
  # This is the E32 failure mode: profiling was configured, but no rank could
  # ever arm because PROFILE_ARM_AFTER stayed 0 and PROFILE_REQUEST_IDS was
  # empty.  Refuse to start instead of producing another invalid Run.
  if [ "$PROFILE_ARM_AFTER" -eq 0 ] && [ -z "$PROFILE_REQUEST_IDS" ] \
      && [ "${PROFILE_ALLOW_UNARMED:-0}" != "1" ]; then
    die "profile is configured but has no arming trigger: set PROFILE_ARM_AFTER>0 or PROFILE_REQUEST_IDS; set PROFILE_ALLOW_UNARMED=1 only for an intentional unarmed diagnostic" 3
  fi
  if [ "$PROFILE_PROMPT_WINDOWS" -eq 0 ] && [ "$PROFILE_DECODE_WINDOWS" -eq 0 ] \
      && [ "${PROFILE_ALLOW_UNARMED:-0}" != "1" ]; then
    die "profile is configured but both prompt and decode window counts are zero; set a window count or PROFILE_ALLOW_UNARMED=1 for an intentional no-window diagnostic" 3
  fi
  if [ "$PROFILE_MODE" = "prompt" ] && [ "$PROFILE_DECODE_WINDOWS" -gt 0 ]; then
    printf 'E33_LAUNCHER_WARNING: PROFILE_MODE=prompt ignores PROFILE_DECODE_WINDOWS=%s\n' "$PROFILE_DECODE_WINDOWS" >&2
  elif [ "$PROFILE_MODE" = "decode" ] && [ "$PROFILE_PROMPT_WINDOWS" -gt 0 ]; then
    printf 'E33_LAUNCHER_WARNING: PROFILE_MODE=decode ignores PROFILE_PROMPT_WINDOWS=%s\n' "$PROFILE_PROMPT_WINDOWS" >&2
  fi
elif [ "$PROFILE_ARM_AFTER" -gt 0 ] || [ -n "$PROFILE_REQUEST_IDS" ]; then
  printf 'E33_LAUNCHER_WARNING: profile trigger is set but PROFILE_DIR is empty; profiling stays disabled\n' >&2
fi

[ -n "$CONTAINER_LAUNCHER" ] || die "CONTAINER_LAUNCHER is empty" 2

docker exec -d \
  -e PORT="$PORT" -e LOG="$LOG" \
  -e ALL="${ALL:-1}" -e ANY="${ANY:-1}" \
  -e MAX_SEQS="${MAX_SEQS:-16}" -e BUDGET="${BUDGET:-8192}" \
  -e MAX_LEN="${MAX_LEN:-98304}" -e TOPK="${TOPK:-4096}" \
  -e NEXTN="${NEXTN:-5}" -e MTP_VARLEN="${MTP_VARLEN:-1}" \
  -e ZOS="${ZOS:-1}" -e GRAPH="${GRAPH:-1}" \
  -e LABEL="${LABEL:-e27-dspark}" \
  -e E27_PREFIX_PARENT_LIMIT="${E27_PREFIX_PARENT_LIMIT:-32}" \
  -e DEEPINFER_NUM_STATE_SLOTS_RATIO="${DEEPINFER_NUM_STATE_SLOTS_RATIO:-4}" \
  -e YOCO_MODE="${YOCO_MODE:-off}" -e YOCO_KEEP="${YOCO_KEEP:-140}" \
  -e DIAG_REQUESTS_PATH="${DIAG_REQUESTS_PATH:-}" \
  -e GEMM_WARM="${GEMM_WARM:-0}" \
  -e GEMM_WARM_REPORT="${GEMM_WARM_REPORT:-/work/runtime-20261021/e27-gemm-readiness.json}" \
  -e STARTUP_WARM="${STARTUP_WARM:-0}" \
  -e STARTUP_WARM_PROMPTS="${STARTUP_WARM_PROMPTS:-adapter/e15_warm_prompts.json}" \
  -e STARTUP_WARM_REPORT="${STARTUP_WARM_REPORT:-/work/runtime-20261021/e27-startup-warm.json}" \
  -e STARTUP_WARM_TIMEOUT="${STARTUP_WARM_TIMEOUT:-600}" \
  -e STARTUP_WARM_MAX_TOKENS="${STARTUP_WARM_MAX_TOKENS:-600}" \
  -e SPARSE_MQA_WARM="${SPARSE_MQA_WARM:-0}" \
  -e SPARSE_MQA_WARM_MANIFEST="${SPARSE_MQA_WARM_MANIFEST:-adapter/e15_sparse_mqa_families.json}" \
  -e SPARSE_MQA_WARM_REPORT="${SPARSE_MQA_WARM_REPORT:-/work/runtime-20261021/e27-sparse-mqa-readiness.json}" \
  -e SPARSE_MQA_WARM_MAX_Q="${SPARSE_MQA_WARM_MAX_Q:-8192}" \
  -e SPARSE_MQA_WARM_WORKERS="${SPARSE_MQA_WARM_WORKERS:-8}" \
  -e SPARSE_MQA_WARM_TIMEOUT="${SPARSE_MQA_WARM_TIMEOUT:-600}" \
  -e PROFILE_DIR="$PROFILE_DIR" -e PROFILE_RUN_ID="$PROFILE_RUN_ID" \
  -e PROFILE_CASE_ID="$PROFILE_CASE_ID" \
  -e PROFILE_BASELINE_RUN_ID="$PROFILE_BASELINE_RUN_ID" \
  -e PROFILE_REQUEST_IDS="$PROFILE_REQUEST_IDS" \
  -e PROFILE_ARM_AFTER="$PROFILE_ARM_AFTER" -e PROFILE_ARM_LEAD="$PROFILE_ARM_LEAD" \
  -e PROFILE_MODE="$PROFILE_MODE" -e PROFILE_STEPS="$PROFILE_STEPS" \
  -e PROFILE_PROMPT_WINDOWS="$PROFILE_PROMPT_WINDOWS" -e PROFILE_DECODE_WINDOWS="$PROFILE_DECODE_WINDOWS" \
  -e PROFILE_DECODE_TRIGGER="$PROFILE_DECODE_TRIGGER" \
  -e PROFILE_PROMPT_INTERVAL="$PROFILE_PROMPT_INTERVAL" \
  -e PROFILE_PROMPT_TRIGGER_MODULI="$PROFILE_PROMPT_TRIGGER_MODULI" \
  -e PROFILE_PROMPT_TRIGGER_K_FRACS="$PROFILE_PROMPT_TRIGGER_K_FRACS" \
  -e PROFILE_PROMPT_TRIGGER_REQUIRES_LOCAL="$PROFILE_PROMPT_TRIGGER_REQUIRES_LOCAL" \
  -e PROFILE_LEVEL="$PROFILE_LEVEL" \
  -e PROFILE_INITIAL_PROMPT="$PROFILE_INITIAL_PROMPT" -e PROFILE_IMMEDIATE_PROMPT="$PROFILE_IMMEDIATE_PROMPT" \
  -e PROFILE_RECORD_SHAPES="$PROFILE_RECORD_SHAPES" -e PROFILE_WITH_STACK="$PROFILE_WITH_STACK" \
  -e CONTAINER_LAUNCHER="$CONTAINER_LAUNCHER" \
  "$NAME" bash -lc 'exec "$CONTAINER_LAUNCHER"'

if [ -n "$PROFILE_DIR" ]; then
  echo "started E33 DI port=$PORT budget=${BUDGET:-8192} maxseq=${MAX_SEQS:-16} maxlen=${MAX_LEN:-98304} parents=${E27_PREFIX_PARENT_LIMIT:-32} state_ratio=${DEEPINFER_NUM_STATE_SLOTS_RATIO:-4} profile=on run=${PROFILE_RUN_ID} case=${PROFILE_CASE_ID:-none} mode=$PROFILE_MODE arm_after=$PROFILE_ARM_AFTER arm_lead=$PROFILE_ARM_LEAD steps=$PROFILE_STEPS prompt_windows=$PROFILE_PROMPT_WINDOWS decode_windows=$PROFILE_DECODE_WINDOWS dir=$PROFILE_DIR launcher=$CONTAINER_LAUNCHER log=$LOG"
else
  echo "started E33 DI port=$PORT budget=${BUDGET:-8192} maxseq=${MAX_SEQS:-16} maxlen=${MAX_LEN:-98304} parents=${E27_PREFIX_PARENT_LIMIT:-32} state_ratio=${DEEPINFER_NUM_STATE_SLOTS_RATIO:-4} profile=off launcher=$CONTAINER_LAUNCHER log=$LOG"
fi
echo "note: detached start only; verify arming/export with tools/e33_profile_control.py"
