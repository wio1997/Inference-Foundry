#!/bin/bash
# 2026-10-06: GLM-5.3 defaults; historical Run validation belongs to GLM-5.2.
# This template has no GLM-5.3 fit/correctness/performance acceptance; follow PLAN.md.
# ============================================================================
# GLM-5.3 PD — P (prefill / kv_producer) on node 166, inside glm52-single
# Layout: DP1 PP1 TP16 EP16 DCP16 PCP1   |  port 9081  |  kv_port 36000
# MTP draft tokens: 3; native draft eager, target FULL_DECODE_ONLY
# Resident P validated Run12; not formal KEEP
# NOTE: NEW task-local file. Original start_single_tp16.sh / start_glm52_single.sh
#       are NOT modified.
# ============================================================================
set -o pipefail

export PD_P_ADDITIONAL_CONFIG='{"multistream_overlap_shared_expert": true, "recompute_scheduler_enable": true}'
export PD_ROLE="P"
export PD_LOCAL_IP="172.16.10.166"
export PD_NIC="business"
export PD_PORT="${PD_PORT:-9081}"
export PD_KV_PORT="${PD_KV_PORT:-36000}"
export PD_MODEL="${PD_MODEL:-/data/tiankuan/wio/GLM-5.3-w8a8}"
export PD_MTP_TOKENS="${PD_MTP_TOKENS:-3}"
export PD_MAX_SEQS="${PD_MAX_SEQS:-8}"
export PD_GMU="${PD_GMU:-0.87}"
export PD_MAX_BATCHED_TOKENS="${PD_MAX_BATCHED_TOKENS:-4096}"
export PD_LOG_DIR="${PD_LOG_DIR:-/data/tiankuan/wio/glm52-pd/deploy/logs}"

# ---------------------------------------------------------------------------
# P-side additional_config.
# NOTE (2026-10-01): `enable_dsa_cp` was REMOVED after a reproducible engine-init
# failure.  With enable_dsa_cp the engine logs "DSA-CP is enabled. Auto-enabling
# SP." + "Sequence-parallel MoE is enabled.", and the first warm-up forward then
# hangs in the MoE expert-parallel all-to-all: the runtime dumps kernel
# `MoeDistributeDispatchV2` and reports
#   aclrtSynchronizeEvent error 507034 / "vector core timeout"
# after ~9 minutes, and the HCCL watchdog terminates every rank.
# The same model/parallel layout without enable_dsa_cp (the D node, and P below)
# initialises fine, so DSA-CP's sequence-parallel-MoE path is the trigger.
# Set PD_P_ENABLE_DSA_CP=1 to reproduce.
# ---------------------------------------------------------------------------
export PD_P_ADDITIONAL_CONFIG='{"multistream_overlap_shared_expert": true, "recompute_scheduler_enable": true}'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=pd_common_env.sh
source "/data/tiankuan/wio/glm52-pd/deploy/scripts/pd_common_env.sh"

mkdir -p "${PD_LOG_DIR}"
LOG_PATH="${PD_LOG_DIR}/P_166.log"
: > "${LOG_PATH}"

{
  echo "============================================================"
  echo "GLM-5.3 PD  ROLE=P (prefill/kv_producer)  NODE=166   $(date -Is)"
  echo "model        : ${PD_MODEL}"
  echo "port         : ${PD_PORT}     kv_port: ${PD_KV_PORT}"
  echo "layout       : DP1 PP1 TP${PD_TP} EP + DCP${PD_DCP} PCP1"
  echo "max_model_len: ${PD_MODEL_LEN}   max_num_seqs: ${PD_MAX_SEQS}"
  echo "max_num_batched_tokens: ${PD_MAX_BATCHED_TOKENS}"
  echo "gpu_mem_util : ${PD_GMU}"
  echo "MTP          : n=${PD_MTP_TOKENS}"
  echo "graph        : FULL_DECODE_ONLY"
  echo "additional   : ${PD_P_ADDITIONAL_CONFIG}"
  echo "quantization : ascend (W8A8, no c8 switches)"
  echo "connector    : MooncakeConnectorV1 producer"
  echo "============================================================"
} | tee -a "${LOG_PATH}"

vllm serve "${PD_MODEL}" \
  --host 0.0.0.0 \
  --port "${PD_PORT}" \
  --served-model-name "${PD_SERVED_NAME}" \
  --seed 1024 \
  --safetensors-load-strategy prefetch \
  --max-model-len "${PD_MODEL_LEN}" \
  --max-num-seqs "${PD_MAX_SEQS}" \
  --max-num-batched-tokens "${PD_MAX_BATCHED_TOKENS}" \
  --gpu-memory-utilization "${PD_GMU}" \
  --data-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --tensor-parallel-size "${PD_TP}" \
  --prefill-context-parallel-size 1 \
  --decode-context-parallel-size "${PD_DCP}" \
  --cp-kv-cache-interleave-size 128 \
  --enable-expert-parallel \
  --quantization ascend \
  --trust-remote-code \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --additional-config "${PD_P_ADDITIONAL_CONFIG}" \
  --speculative-config "{\"num_speculative_tokens\": ${PD_MTP_TOKENS}, \"method\": \"deepseek_mtp\", \"enforce_eager\": true}" \
  --kv-transfer-config "{\"kv_connector\": \"MooncakeConnectorV1\", \"kv_role\": \"kv_producer\", \"kv_port\": \"${PD_KV_PORT}\", \"engine_id\": \"0\", \"kv_connector_extra_config\": {\"use_ascend_direct\": true, \"prefill\": {\"dp_size\": 1, \"tp_size\": ${PD_TP}}, \"decode\": {\"dp_size\": 1, \"tp_size\": ${PD_TP}}}}" \
  2>&1 | tee -a "${LOG_PATH}"
