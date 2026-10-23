#!/usr/bin/env bash
# Run inside the deployment container; foreground service.
set -euo pipefail
source /workspace/deploy/runtime_env.sh
cd /workspace/deploy
exec vllm serve /models/DeepSeek-V4.1-Flash \
  --host 0.0.0.0 --port "${SERVICE_PORT:-8000}" \
  --served-model-name DeepSeek-V4.1-Flash-1017 \
  --data-parallel-size 8 --data-parallel-size-local 8 \
  --tensor-parallel-size 1 --enable-expert-parallel \
  --max-model-len 98304 --max-num-batched-tokens 8256 --max-num-seqs 16 \
  --async-scheduling --gpu-memory-utilization 0.85 --block-size 128 \
  --trust-remote-code --safetensors-load-strategy lazy \
  --quantization deepseek_v4_fp8 --enable-prefix-caching \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[6,12,24,48,96]}' \
  --speculative-config '{"num_speculative_tokens":5,"method":"dspark"}' \
  --additional-config '{"enable_engram":true,"engram_storage":"fp8","enable_engram_offload":true,"engram_tp_size":8,"weight_nz_mode":2,"multistream_overlap_shared_expert":true,"enable_cpu_binding":true,"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":true}}'
