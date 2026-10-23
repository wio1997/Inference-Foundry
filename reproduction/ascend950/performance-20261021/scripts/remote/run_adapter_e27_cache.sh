#!/usr/bin/env bash
# Container-side launcher for the E27 DeepInfer DP8 cache-comparison service.
# Explicit E27 shape: 8192 effective tokens/rank, max_num_seqs=16, ctx 98304.
# Historical E14/E15 512-scale exhaustive warmers are OFF by default.  Use the
# main controller's one full-shape HTTP load for new-shape warm/JIT checks.
set -euo pipefail
LOG="${LOG:-/work/performance-20261021/logs/e27-di-$(date -u +%Y%m%dT%H%M%SZ).log}"
exec >> "$LOG" 2>&1
ulimit -s 65536 2>/dev/null || true
source /usr/local/Ascend/cann-9.2.0/set_env.sh
export LD_LIBRARY_PATH=/work/engine-venv/lib/python3.12/site-packages/torch/lib:/work/engine-venv/lib/python3.12/site-packages/torch_npu/lib:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:${LD_LIBRARY_PATH:-}
export TILELANG_CACHE_DIR=${TILELANG_CACHE_DIR:-/work/runtime-20261021/tilelang-cache}
export DG_JIT_CACHE_DIR=${DG_JIT_CACHE_DIR:-/work/runtime-20261021/dg-jit}
export EP_JIT_CACHE_DIR=${EP_JIT_CACHE_DIR:-/work/runtime-20261021/ep-jit}
# 65 state slots at ratio 4: 32 parents + 16 live children + 1 dummy + 16 margin.
export DEEPINFER_NUM_STATE_SLOTS_RATIO="${DEEPINFER_NUM_STATE_SLOTS_RATIO:-4}"
export E27_PREFIX_PARENT_LIMIT="${E27_PREFIX_PARENT_LIMIT:-32}"
cd /work/performance-20261021
if [ "${YOCO_MODE:-off}" != "off" ]; then
  export DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE="${YOCO_MODE}"
  export DEEPINFER_YOCO_PREFILL_KEEP_TOKENS="${YOCO_KEEP:-140}"
fi

if [ "${BUDGET:-8192}" != "8192" ] || [ "${MAX_SEQS:-16}" != "16" ] || [ "${MAX_LEN:-98304}" != "98304" ]; then
  echo "WARNING: E27 shape overridden: budget=${BUDGET:-8192} maxseq=${MAX_SEQS:-16} maxlen=${MAX_LEN:-98304} (expected 8192/16/98304)" >&2
fi

args=(--serve --port "${PORT:-18121}"
      --max-num-seqs "${MAX_SEQS:-16}"
      --max-num-batched-tokens "${BUDGET:-8192}"
      --max-model-len "${MAX_LEN:-98304}"
      --force-topk "${TOPK:-4096}"
      --prefill-attempt-interval "${ALL:-1}"
      --prefill-any-attempt-interval "${ANY:-1}"
      --nextn "${NEXTN:-5}"
      --label "${LABEL:-e27-dspark}"
      --no-trace-steps)
if [ "${MTP_VARLEN:-1}" = "1" ]; then args+=(--mtp-varlen); fi
if [ "${ZOS:-1}" = "1" ]; then args+=(--zos); fi
if [ "${GRAPH:-1}" = "1" ]; then args+=(--no-enforce-eager); fi
if [ -n "${DIAG_REQUESTS_PATH:-}" ]; then args+=(--diag-requests-path "$DIAG_REQUESTS_PATH"); fi
if [ "${GEMM_WARM:-0}" = "1" ]; then
  echo "WARNING: E14 GEMM manifest is historical 512-scale coverage, not evidence for the E27 8192 shape" >&2
  args+=(--gemm-warm-manifest "${GEMM_WARM_MANIFEST:-adapter/e14_gemm_families.json}"
         --gemm-warm-report "${GEMM_WARM_REPORT:-/work/runtime-20261021/e27-gemm-readiness.json}"
         --gemm-warm-workers "${GEMM_WARM_WORKERS:-10}"
         --gemm-warm-max-m "${GEMM_WARM_MAX_M:-8192}"
         --gemm-warm-timeout "${GEMM_WARM_TIMEOUT:-900}")
fi
if [ "${STARTUP_WARM:-0}" = "1" ]; then
  echo "WARNING: E15 startup prompts are historical 512-scale coverage; verify the E27 new-shape load separately" >&2
  args+=(--startup-warm
         --startup-warm-prompts "${STARTUP_WARM_PROMPTS:-adapter/e15_warm_prompts.json}"
         --startup-warm-report "${STARTUP_WARM_REPORT:-/work/runtime-20261021/e27-startup-warm.json}"
         --startup-warm-timeout "${STARTUP_WARM_TIMEOUT:-600}"
         --startup-warm-max-tokens "${STARTUP_WARM_MAX_TOKENS:-600}")
else
  args+=(--no-startup-warm)
fi
if [ "${SPARSE_MQA_WARM:-0}" = "1" ]; then
  echo "WARNING: E15 sparse-MQA manifest is historical 512-scale coverage, not 8192 evidence" >&2
  args+=(--sparse-mqa-warm-manifest "${SPARSE_MQA_WARM_MANIFEST:-adapter/e15_sparse_mqa_families.json}"
         --sparse-mqa-warm-report "${SPARSE_MQA_WARM_REPORT:-/work/runtime-20261021/e27-sparse-mqa-readiness.json}"
         --sparse-mqa-warm-max-q "${SPARSE_MQA_WARM_MAX_Q:-8192}"
         --sparse-mqa-warm-workers "${SPARSE_MQA_WARM_WORKERS:-8}"
         --sparse-mqa-warm-timeout "${SPARSE_MQA_WARM_TIMEOUT:-600}")
fi
if [ -n "${PROFILE_DIR:-}" ]; then
  args+=(--profile-dir "$PROFILE_DIR"
         --profile-run-id "${PROFILE_RUN_ID:?profile-run-id required}"
         --profile-case-id "${PROFILE_CASE_ID:-}"
         --profile-request-ids "${PROFILE_REQUEST_IDS:-}"
         --profile-arm-after-dispatched "${PROFILE_ARM_AFTER:-0}"
         --profile-mode "${PROFILE_MODE:-prompt}"
         --profile-steps "${PROFILE_STEPS:-4}"
         --profile-prompt-windows "${PROFILE_PROMPT_WINDOWS:-1}"
         --profile-decode-windows "${PROFILE_DECODE_WINDOWS:-1}"
         --profile-decode-trigger "${PROFILE_DECODE_TRIGGER:-4}"
         --profile-level "${PROFILE_LEVEL:-Level1}"
         --profile-arm-lead "${PROFILE_ARM_LEAD:-2}")
  if [ "${PROFILE_INITIAL_PROMPT:-0}" = "1" ]; then args+=(--profile-initial-prompt-window); fi
  if [ "${PROFILE_IMMEDIATE_PROMPT:-0}" = "1" ]; then args+=(--profile-immediate-prompt-window); fi
  if [ "${PROFILE_RECORD_SHAPES:-0}" = "1" ]; then args+=(--profile-record-shapes); fi
  if [ "${PROFILE_WITH_STACK:-0}" = "1" ]; then args+=(--profile-with-stack); fi
fi
exec /work/engine-venv/bin/python adapter/deepinfer_dp8_adapter_e27_cache.py "${args[@]}"
