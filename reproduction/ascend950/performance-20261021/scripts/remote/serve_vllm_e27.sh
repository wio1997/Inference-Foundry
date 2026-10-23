#!/usr/bin/env bash
# E27 vLLM DP8 launcher: E20 accepted native-fusion environment, E27 shape.
# Effective budget is 8256 - 4*16 = 8192 (DSpark5 reserves 4 slots/seq).
# This script only launches; it never stops or replaces other services.
set -euo pipefail
NAME=${NAME:-performance-20261021-vllm-official-ds41}
PORT=${PORT:-18122}
LOG=${LOG:-/work/performance-20261021/logs/e27-vllm-8256-16-$(date -u +%Y%m%dT%H%M%SZ).log}
MAX_LEN=${MAX_LEN:-98304}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8256}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}
CAPTURE_SIZES=${CAPTURE_SIZES:-[6,12,24,48,96]}
VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}
E20_TRACE_FUSION=${E20_TRACE_FUSION:-0}
HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-1024}
E20_PROVIDER=/work/performance-20261021/e20-install/opp/vendors/customize
if [ -z "$CAPTURE_SIZES" ]; then
  echo "CAPTURE_SIZES must not be empty" >&2
  exit 2
fi
if [ "$MAX_NUM_SEQS" -lt 1 ] || [ "$MAX_NUM_BATCHED_TOKENS" -lt 1 ] || [ "$MAX_LEN" -lt 1 ]; then
  echo "MAX_NUM_SEQS, MAX_NUM_BATCHED_TOKENS and MAX_LEN must be positive" >&2
  exit 2
fi

docker exec -d "$NAME" bash -c "set -eo pipefail
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GLOO_SOCKET_IFNAME=lo
export HCCL_SOCKET_IFNAME=enp161s0f0
export HCCL_IF_BASE_PORT=49381
export HCCL_BUFFSIZE=$HCCL_BUFFSIZE
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export VLLM_ASCEND_ENABLE_PRIVATE_CIRCLE_POOL=1
export LOCAL_WORLD_SIZE=8
export TASK_QUEUE_ENABLE=1
export VLLM_LOGGING_LEVEL=$VLLM_LOGGING_LEVEL
export E19_STATIC_MANIFEST=${E19_STATIC_MANIFEST:-0}
export E20_TRACE_FUSION=$E20_TRACE_FUSION
unset E03_DISABLE_VISION_GATING_FUSION || true
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann-9.1.0/share/info/ascendnpu-ir/bin/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=\${ATB_CXX_ABI:-1}
export ASCEND_CUSTOM_OPP_PATH=$E20_PROVIDER:/usr/local/Ascend/cann-9.2.0/opp/vendors/customize:/usr/local/Ascend/cann-9.2.0/opp/vendors/custom_transformer
export PYTHONPATH=/usr/local/python3.12.13/lib/python3.12/site-packages:\${PYTHONPATH:-}
export LD_LIBRARY_PATH=$E20_PROVIDER/op_api/lib:/usr/local/Ascend/cann-9.2.0/opp/vendors/customize/op_api/lib:/usr/local/Ascend/cann-9.2.0/opp/vendors/custom_transformer/op_api/lib:\${LD_LIBRARY_PATH:-}
export LD_LIBRARY_PATH=\$(printf '%s' \"\${LD_LIBRARY_PATH:-}\" | tr ':' '\n' | grep -v '/vllm-workspace' | paste -sd: -)
export PYTHONPATH=\$(printf '%s' \"\$PYTHONPATH\" | tr ':' '\n' | grep -v '/vllm-workspace' | paste -sd: -)
set -u
cd /work/performance-20261021
exec vllm serve /data/wio/DeepSeek-V4.1-Flash-1017 \
  --host 0.0.0.0 --port $PORT \
  --data-parallel-size 8 --data-parallel-size-local 8 \
  --tensor-parallel-size 1 --enable-expert-parallel \
  --served-model-name DeepSeek-V4.1-Flash-1017 \
  --max-model-len $MAX_LEN \
  --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
  --max-num-seqs $MAX_NUM_SEQS \
  --async-scheduling \
  --gpu-memory-utilization 0.85 \
  --block-size 128 \
  --trust-remote-code \
  --safetensors-load-strategy lazy \
  --quantization deepseek_v4_fp8 \
  --enable-prefix-caching \
  --compilation-config '{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":$CAPTURE_SIZES}' \
  --speculative-config '{\"num_speculative_tokens\":5,\"method\":\"dspark\"}' \
  --additional-config '{\"enable_engram\":true,\"engram_storage\":\"fp8\",\"enable_engram_offload\":true,\"engram_tp_size\":8,\"weight_nz_mode\":2,\"multistream_overlap_shared_expert\":true,\"enable_cpu_binding\":true,\"ascend_compilation_config\":{\"enable_npugraph_ex\":true,\"enable_static_kernel\":true}}' \
  >> $LOG 2>&1"
echo "started E27 V on $PORT; log=$LOG maxlen=$MAX_LEN nominal_budget=$MAX_NUM_BATCHED_TOKENS maxseq=$MAX_NUM_SEQS capture=$CAPTURE_SIZES; expected effective=8256-4*16=8192; verify all8 ranks"
