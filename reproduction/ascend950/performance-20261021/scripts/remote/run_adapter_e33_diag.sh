#!/usr/bin/env bash
# Container-side launcher for the E33 DeepInfer DP8 full-cohort diagnostic service.
#
# Shape is unchanged from E27: 8192 effective tokens/rank, max_num_seqs=16,
# context 98304, DSpark5, ZOS, native graphs and the E27 prefix cache.
#
# E33 changes only the profiler control surface:
#   * the legacy PROFILE_DIR / PROFILE_* step-window arguments are NOT passed;
#     PROFILE_* is reported and ignored, so no 4-step window can auto-stop;
#   * full recording is started/stopped only through the adapter's loopback
#     POST /e33/profile/start and POST /e33/profile/stop endpoints;
#   * this launcher keeps the E27 in-container environment and benchmark args.
set -euo pipefail
LOG="${LOG:-/work/performance-20261021/logs/e33-diag-$(date -u +%Y%m%dT%H%M%SZ).log}"
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
# Only /e33 out_dir values below this task root are accepted.
export E33_TASK_ROOT="${E33_TASK_ROOT:-/work/performance-20261021}"
cd /work/performance-20261021
if [ "${YOCO_MODE:-off}" != "off" ]; then
  export DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE="${YOCO_MODE}"
  export DEEPINFER_YOCO_PREFILL_KEEP_TOKENS="${YOCO_KEEP:-140}"
fi

if [ -n "${PROFILE_DIR:-}" ] || [ -n "${PROFILE_RUN_ID:-}" ]; then
  echo "E33_LAUNCHER_NOTICE: PROFILE_DIR/PROFILE_RUN_ID are ignored; the legacy step-window controller is disabled. Use POST /e33/profile/start|stop." >&2
fi
unset PROFILE_DIR PROFILE_RUN_ID PROFILE_CASE_ID PROFILE_BASELINE_RUN_ID \
      PROFILE_REQUEST_IDS PROFILE_ARM_AFTER PROFILE_ARM_LEAD PROFILE_MODE \
      PROFILE_STEPS PROFILE_PROMPT_WINDOWS PROFILE_DECODE_WINDOWS \
      PROFILE_DECODE_TRIGGER PROFILE_PROMPT_INTERVAL \
      PROFILE_PROMPT_TRIGGER_MODULI PROFILE_PROMPT_TRIGGER_K_FRACS \
      PROFILE_PROMPT_TRIGGER_REQUIRES_LOCAL PROFILE_LEVEL \
      PROFILE_INITIAL_PROMPT PROFILE_IMMEDIATE_PROMPT \
      PROFILE_RECORD_SHAPES PROFILE_WITH_STACK 2>/dev/null || true

if [ "${BUDGET:-8192}" != "8192" ] || [ "${MAX_SEQS:-16}" != "16" ] || [ "${MAX_LEN:-98304}" != "98304" ]; then
  echo "WARNING: E33 shape overridden: budget=${BUDGET:-8192} maxseq=${MAX_SEQS:-16} maxlen=${MAX_LEN:-98304} (expected 8192/16/98304)" >&2
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
      --no-trace-steps
      --e33-task-root "${E33_TASK_ROOT}")
if [ "${MTP_VARLEN:-1}" = "1" ]; then args+=(--mtp-varlen); fi
if [ "${ZOS:-1}" = "1" ]; then args+=(--zos); fi
if [ "${GRAPH:-1}" = "1" ]; then args+=(--no-enforce-eager); fi
if [ -n "${DIAG_REQUESTS_PATH:-}" ]; then args+=(--diag-requests-path "$DIAG_REQUESTS_PATH"); fi
if [ "${GEMM_WARM:-0}" = "1" ]; then
  echo "WARNING: E14 GEMM manifest is historical 512-scale coverage, not evidence for the E33 8192 shape" >&2
  args+=(--gemm-warm-manifest "${GEMM_WARM_MANIFEST:-adapter/e14_gemm_families.json}"
         --gemm-warm-report "${GEMM_WARM_REPORT:-/work/runtime-20261021/e27-gemm-readiness.json}"
         --gemm-warm-workers "${GEMM_WARM_WORKERS:-10}"
         --gemm-warm-max-m "${GEMM_WARM_MAX_M:-8192}"
         --gemm-warm-timeout "${GEMM_WARM_TIMEOUT:-900}")
fi
if [ "${STARTUP_WARM:-0}" = "1" ]; then
  echo "WARNING: E15 startup prompts are historical 512-scale coverage; verify the E33 new-shape load separately" >&2
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
exec /work/engine-venv/bin/python adapter/deepinfer_dp8_adapter_e33.py "${args[@]}"
