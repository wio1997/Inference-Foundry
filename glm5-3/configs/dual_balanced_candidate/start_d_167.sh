#!/bin/bash
# ============================================================================
# GLM-5.2 PD — D (decode / kv_consumer) on node 167, inside glm52-single
# Layout: DP1 PP1 TP16 EP16 DCP16 PCP1   |  port 9900  |  kv_port 36200
# MTP draft tokens: 3, graph mode FULL_DECODE_ONLY
# NOTE: NEW task-local file. Original start scripts are NOT modified.
# ============================================================================
set -o pipefail

export PD_ROLE="D"
export PD_LOCAL_IP="${PD_LOCAL_IP:-172.16.10.167}"
export PD_NIC="${PD_NIC:-business}"
export PD_PORT="${PD_PORT:-9900}"
export PD_KV_PORT="${PD_KV_PORT:-36200}"
export PD_MODEL="${PD_MODEL:-/data/tiankuan/wio/GLM-5.2-w8a8}"
export PD_MTP_TOKENS="${PD_MTP_TOKENS:-3}"
export PD_MAX_SEQS="${PD_MAX_SEQS:-8}"
export PD_GMU="${PD_GMU:-0.87}"
export PD_MAX_BATCHED_TOKENS="${PD_MAX_BATCHED_TOKENS:-4096}"
export PD_LOG_DIR="${PD_LOG_DIR:-${PD_ROOT:-/data/tiankuan/wio/glm52-pd/deploy}/logs}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=pd_common_env.sh
source "/data/tiankuan/wio/glm52-pd/deploy/scripts/pd_common_env.sh"

mkdir -p "${PD_LOG_DIR}"
LOG_PATH="${PD_LOG_DIR}/D_167.log"
: > "${LOG_PATH}"

{
  echo "============================================================"
  echo "GLM-5.2 PD  ROLE=D (decode/kv_consumer)  NODE=167   $(date -Is)"
  echo "model        : ${PD_MODEL}"
  echo "port         : ${PD_PORT}     kv_port: ${PD_KV_PORT}"
  echo "layout       : DP1 PP1 TP${PD_TP} EP + DCP${PD_DCP} PCP1"
  echo "max_model_len: ${PD_MODEL_LEN}   max_num_seqs: ${PD_MAX_SEQS}"
  echo "max_num_batched_tokens: ${PD_MAX_BATCHED_TOKENS}"
  echo "gpu_mem_util : ${PD_GMU}"
  echo "MTP          : n=${PD_MTP_TOKENS}"
  echo "graph        : FULL_DECODE_ONLY"
  echo "additional   : multistream_overlap_shared_expert + recompute_scheduler_enable"
  echo "quantization : ascend (W8A8, no c8 switches)"
  echo "connector    : MooncakeConnectorV1 consumer"
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
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --additional-config '{"multistream_overlap_shared_expert": true, "recompute_scheduler_enable": true}' \
  --speculative-config "{\"num_speculative_tokens\": ${PD_MTP_TOKENS}, \"method\": \"deepseek_mtp\", \"enforce_eager\": true}" \
  --kv-transfer-config "{\"kv_connector\": \"MooncakeConnectorV1\", \"kv_role\": \"kv_consumer\", \"kv_port\": \"${PD_KV_PORT}\", \"engine_id\": \"1\", \"kv_connector_extra_config\": {\"use_ascend_direct\": true, \"prefill\": {\"dp_size\": 1, \"tp_size\": ${PD_TP}}, \"decode\": {\"dp_size\": 1, \"tp_size\": ${PD_TP}}}}" \
  2>&1 | tee -a "${LOG_PATH}"
