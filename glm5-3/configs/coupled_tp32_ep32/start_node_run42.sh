#!/bin/bash
# 2026-10-06: GLM-5.3 defaults; historical Run validation belongs to GLM-5.2.
# This template has no GLM-5.3 fit/correctness/performance acceptance; follow PLAN.md.
# Two physical hosts, one native DP1/TP32/DCP1/EP32 scheduler. Node1 headless.
set -o pipefail
: "${GLM_NODE_RANK:?}"; : "${GLM_LOCAL_IP:?}"; : "${GLM_API_PORT:?}"; : "${GLM_SUPERPOD_ID:?}"
export PD_LOCAL_IP="${GLM_LOCAL_IP}" PD_NIC=business
source /data/tiankuan/wio/glm52-pd/deploy/scripts/pd_common_env.sh
export PYTHONPATH="/data/tiankuan/wio/glm52-pd/deploy/scripts${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_HOST_IP="${GLM_LOCAL_IP}"
export HCCL_LOGIC_SUPERPOD_ID="${GLM_SUPERPOD_ID}"
LOG="/data/tiankuan/wio/glm52-pd/deploy/logs/TP_run42_${GLM_NODE_RANK}.log"
: > "${LOG}"
EXTRA=()
if [ "${GLM_NODE_RANK}" = 1 ]; then EXTRA+=(--headless); fi
vllm serve /data/tiankuan/wio/GLM-5.3-w8a8 \
 --host 0.0.0.0 --port "${GLM_API_PORT}" --served-model-name glm-53 \
 --seed 1024 --safetensors-load-strategy prefetch --trust-remote-code \
 --max-model-len 144384 --max-num-seqs 8 --max-num-batched-tokens 16384 --gpu-memory-utilization 0.8 \
 --data-parallel-size 1 --data-parallel-size-local 1 \
 --data-parallel-address 172.16.10.166 --data-parallel-rpc-port 32620 --data-parallel-backend mp \
 --nnodes 2 --node-rank "${GLM_NODE_RANK}" --master-addr 172.16.10.166 --master-port 32630 \
 --worker-cls atomic_mq_worker.AtomicMQWorker --distributed-executor-backend mp --pipeline-parallel-size 1 --tensor-parallel-size 32 \
 --prefill-context-parallel-size 1 --decode-context-parallel-size 1 --cp-kv-cache-interleave-size 128 \
 --enable-expert-parallel --quantization ascend --enable-prefix-caching --enable-chunked-prefill \
 --enable-auto-tool-choice --tool-parser-plugin /data/tiankuan/wio/glm52-pd/deploy/scripts/glm_tool_contract_run42.py \
 --tool-call-parser glm47_contract --reasoning-parser glm45 \
 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
 --additional-config '{"multistream_overlap_shared_expert":true,"enable_dsa_cp":false,"enable_fused_mc2":0,"mc2_comm_alg":"hierarchy"}' \
 --speculative-config '{"num_speculative_tokens":5,"method":"deepseek_mtp","enforce_eager":true}' \
 "${EXTRA[@]}" \
 2>&1 | tee -a "${LOG}"
