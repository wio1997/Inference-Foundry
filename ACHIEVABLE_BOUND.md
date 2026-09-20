# Achievable Bound V0

Updated: 2026-09-20 UTC. This is a partial lower-bound model, not a claim of attainable end-to-end throughput.

## Weight traffic input

Safetensors headers: 169,725,460,216 bytes total; routed experts 152,228,069,376 bytes; other weights 17,497,390,840 bytes. With 6 of 256 routed experts per token, uniformly sized experts, the selected-weight estimate is `17,497,390,840 + 152,228,069,376 × 6/256 = 21,065,236,216` bytes model-wide, or 2.633 GB/card with ideal EP/TP8 sharding.

Using the historical 1.3 TB/s single-kernel HBM reference gives an **optimistic weight-read floor** of about 2.0 ms for one selected-expert model path on each card. The real floor may rise from cache misses, multiple draft/target passes, quant metadata, uneven expert choice, non-overlapped TP8 communication, KV traffic, compute, and launch dependencies. It may fall per output token through batch reuse or multiple accepted speculative tokens. Therefore this number is not yet comparable to TPOT.

## Model

`T_step >= max(T_compute, T_required_HBM, T_unoverlapped_HCCL, T_critical_path) + irreducible launch/sync`, with explicit overlap accounting. Unknown terms remain unknown until profiling and microbenchmarks. Prefill bound also needs GEMM effective throughput, attention/KV, routing and communication. For end-to-end mixed load, compare latency and output throughput on the same fixed workload.

## Next measurement

Obtain reliable current DP1/TP8 end-to-end baseline; separate prefill and decode device spans, TP8 communication and host stalls. Recalculate bound from actual active bytes and measured attainable bandwidth for relevant kernels. `weight_inventory.json` provides a reproducible starting input.
