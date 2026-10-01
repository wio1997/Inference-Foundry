#!/bin/bash
# Two-host coupled native DP2/TP16/DCP16/EP32; one DP rank per physical node.
set -o pipefail
: "${GLM_DP_RANK:?}"; : "${GLM_LOCAL_IP:?}"; : "${GLM_API_PORT:?}"; : "${GLM_SUPERPOD_ID:?}"
export PD_LOCAL_IP="${GLM_LOCAL_IP}" PD_NIC=business
source /data/tiankuan/wio/glm52-pd/deploy/scripts/pd_common_env.sh
export PYTHONPATH="/data/tiankuan/wio/glm52-pd/deploy/plugins/coupled_dp_run32:${PYTHONPATH:-}"
export VLLM_HOST_IP="${GLM_LOCAL_IP}"
export HCCL_LOGIC_SUPERPOD_ID="${GLM_SUPERPOD_ID}"
LOG="/data/tiankuan/wio/glm52-pd/deploy/logs/DP_run32_${GLM_DP_RANK}.log"
: > "${LOG}"
vllm serve /data/tiankuan/wio/GLM-5.2-w8a8 \
 --host 0.0.0.0 --port "${GLM_API_PORT}" --served-model-name glm-52 \
 --seed 1024 --safetensors-load-strategy prefetch --trust-remote-code \
 --max-model-len 144384 --max-num-seqs 8 --max-num-batched-tokens 16384 --gpu-memory-utilization 0.87 \
 --data-parallel-size 2 --data-parallel-size-local 1 --data-parallel-rank "${GLM_DP_RANK}" --data-parallel-external-lb \
 --data-parallel-address 172.16.10.166 --data-parallel-rpc-port 32620 --data-parallel-backend mp \
 --nnodes 2 --node-rank "${GLM_DP_RANK}" --master-addr 172.16.10.166 --master-port 32630 \
 --worker-cls coupled_dp_worker_v2.CoupledMetadataWorker --distributed-executor-backend mp --pipeline-parallel-size 1 --tensor-parallel-size 16 \
 --prefill-context-parallel-size 1 --decode-context-parallel-size 16 --cp-kv-cache-interleave-size 128 \
 --enable-expert-parallel --quantization ascend --enable-prefix-caching --enable-chunked-prefill \
 --enable-auto-tool-choice --tool-parser-plugin /data/tiankuan/wio/glm52-pd/deploy/scripts/glm_tool_contract_run32.py \
 --tool-call-parser glm47_contract --reasoning-parser glm45 \
 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
 --additional-config '{"multistream_overlap_shared_expert":true,"enable_dsa_cp":false,"enable_fused_mc2":0,"mc2_comm_alg":"hierarchy"}' \
 --speculative-config '{"num_speculative_tokens":5,"method":"deepseek_mtp","enforce_eager":true}' \
 2>&1 | tee -a "${LOG}"
