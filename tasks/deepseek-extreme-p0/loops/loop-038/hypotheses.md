# 假设记录

## H001 - ACTIVE

Starting and stopping torch_npu.profiler on a Runtime cycle schedule can capture a few steady 96-token target cycles on all eight TP ranks without unstable HTTP RPC timing; bounded traces can expose target compute, HCCL overlap and gaps.
