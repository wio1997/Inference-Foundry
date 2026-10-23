#!/usr/bin/env bash
# Start the E27 DeepInfer DP8 service in the task-owned model container.
# This script only launches; it never stops or replaces other services.
set -euo pipefail
NAME=${NAME:-performance-20261021-deepinfer-runtime}
PORT=${PORT:-18121}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG=${LOG:-/work/performance-20261021/logs/e27-di-$STAMP.log}
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
  -e PROFILE_DIR="${PROFILE_DIR:-}" -e PROFILE_RUN_ID="${PROFILE_RUN_ID:-}" \
  -e PROFILE_CASE_ID="${PROFILE_CASE_ID:-}" -e PROFILE_REQUEST_IDS="${PROFILE_REQUEST_IDS:-}" \
  -e PROFILE_MODE="${PROFILE_MODE:-prompt}" -e PROFILE_STEPS="${PROFILE_STEPS:-4}" \
  -e PROFILE_PROMPT_WINDOWS="${PROFILE_PROMPT_WINDOWS:-1}" -e PROFILE_DECODE_WINDOWS="${PROFILE_DECODE_WINDOWS:-1}" \
  -e PROFILE_DECODE_TRIGGER="${PROFILE_DECODE_TRIGGER:-4}" -e PROFILE_LEVEL="${PROFILE_LEVEL:-Level1}" \
  -e PROFILE_INITIAL_PROMPT="${PROFILE_INITIAL_PROMPT:-0}" -e PROFILE_IMMEDIATE_PROMPT="${PROFILE_IMMEDIATE_PROMPT:-0}" \
  -e PROFILE_RECORD_SHAPES="${PROFILE_RECORD_SHAPES:-0}" -e PROFILE_WITH_STACK="${PROFILE_WITH_STACK:-0}" \
  "$NAME" bash -lc 'exec /work/performance-20261021/scripts/remote/run_adapter_e27_cache.sh'
echo "started E27 DI port=$PORT budget=${BUDGET:-8192} maxseq=${MAX_SEQS:-16} maxlen=${MAX_LEN:-98304} parents=${E27_PREFIX_PARENT_LIMIT:-32} state_ratio=${DEEPINFER_NUM_STATE_SLOTS_RATIO:-4} log=$LOG"
