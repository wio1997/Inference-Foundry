#!/usr/bin/env bash
set -euo pipefail
cd "${WIO_ROOT:?Export WIO_ROOT}/projects/910c-deepseek-v4.1-flash"
source config/serve.env
export HOME="$PWD/home"
export TMPDIR="$PWD/tmp"
export XDG_CACHE_HOME="$PWD/cache/xdg"
export HF_HOME="$PWD/cache/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_CACHE_ROOT="$PWD/cache/vllm"
export TORCHINDUCTOR_CACHE_DIR="$PWD/cache/torchinductor"
export TRITON_CACHE_DIR="$PWD/cache/triton"
export ASCEND_CACHE_PATH="$PWD/cache/ascend"
export ASCEND_WORK_PATH="$PWD/cache/ascend-work"
export ASCEND_PROCESS_LOG_PATH="$PWD/logs/ascend"
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export PYTHONDONTWRITEBYTECODE=1
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_ENGINE_READY_TIMEOUT_S=36000
export HCCL_BUFFSIZE=1024 HCCL_OP_EXPANSION_MODE=AIV
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
if [[ -f /usr/lib/aarch64-linux-gnu/libjemalloc.so.2 ]]; then
  export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2
fi
# First bring-up: eager, short context, no speculative decoding.
# Sixteen logical devices: DP2 x TP8 / EP16 with Engram CPU offload only.
exec vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 --port "$SERVICE_PORT" --served-model-name dsv41 \
  --data-parallel-size 2 --tensor-parallel-size 8 --enable-expert-parallel --api-server-count 1 \
  --max-model-len 8192 --max-num-batched-tokens 2048 --max-num-seqs 4 \
  --gpu-memory-utilization 0.9 --block-size 128 --quantization ascend \
  --tokenizer-mode deepseek_v41 --reasoning-parser deepseek_v41 \
  --tool-call-parser deepseek_v41 --enable-auto-tool-choice \
  --safetensors-load-strategy lazy \
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":32}' \
  --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
  --enforce-eager --async-scheduling \
  --additional-config '{"enable_cpu_binding":true,"enable_fused_mc2":1,"enable_dsa_cp":true,"enable_flashcomm1":true,"enable_shared_expert_dp":true}'
