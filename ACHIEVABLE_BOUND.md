# Achievable Bound V0

Updated: 2026-09-20 15:57 UTC. Current measured runtime is available; full achievable bound remains **UNKNOWN** until current TP8 device/communication/host timing is collected.

## Current runtime

Warm-cache, 48×32K→1024, concurrency 12, three corrected passes: median output throughput **543.65 tok/s**, median request-mean TPOT **19.73 ms**, median request-mean TTFT **1.33 s**. Across runs output TPS ranges 523.15–545.85 (4.3% span) and TTFT ranges 1.12–1.37 s. Any candidate gain near this noise band needs additional A/B samples.

## Necessary weight traffic, partial floor

Safetensors headers give **169,725,460,216 B** total, of which routed experts are **152,228,069,376 B**. Uniform 6/256 selection gives `17,497,390,840 + 152,228,069,376 × 6/256 = 21,065,236,216 B` selected weights model-wide for one token's model path. Ideal eight-way distribution is **2.633 GB/card**. With an older 1.3 TB/s RMS-kernel HBM reference, the optimistic weight-read floor is **2.03 ms per path**. Reference: `/data/wio/vllm_ascend_26/results/r16_prefill_device_forward_candidate/evidence/RUN.md:81`.

This 2.03 ms is **not a TPOT bound**: an output token may involve multiple draft and target passes; one pass processes a batch and can reuse weights; expert selection is nonuniform; TP8 communication, KV, quant metadata, compute and serial dependencies remain unmeasured. Accordingly `19.73 / 2.03` is not a valid headroom ratio. Do not present it as achievable speedup.

## Bound model and next measurements

For each prefill chunk and decode iteration, estimate `T_compute`, `T_required_HBM`, `T_KV`, `T_unoverlapped_HCCL`, `T_launch`, `T_sync`, and dependency critical path. Combine only after measuring overlap; neither naïve summation nor `max()` alone is justified. Measure current rank-local HBM throughput, expert traffic, HCCL, target/draft spans and host stalls. Then report `measured / credible lower_bound`, bound assumptions and confidence separately for cold prefill and warm decode.

Cold-prefill current E2E TTFT: 2.663 and 2.627 s (four distinct 32K prompts each, c1). Its per-chunk/compute/communication lower bound is not yet quantified, so cold-prefill headroom is UNKNOWN. The second group had zero prefix hits.
