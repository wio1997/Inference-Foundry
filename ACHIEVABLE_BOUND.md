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

## Update 2026-09-20 16:30 UTC — rank-0 activity envelope

The improved safetensors inventory separates target and MTP selected weights: approximately 16.741 GB and 4.324 GB model-wide under uniform 6/256 routed-expert selection, combined 21.065 GB. Under ideal eight-way balance and the older 1.3 TB/s RMS-kernel HBM reference, optimistic target-only and MTP-only read floors are 1.61 and 0.42 ms. These are per-path traffic estimates, not output-token TPOT bounds: batch reuse, draft repetitions, nonuniform experts, KV and communications remain unmeasured.

A profiled exact-repeat warm 4×128 c4 set took 3.633 s E2E for 512 output tokens. TP0 event span was 3.401 s, of which 1.688 s compute/copy union, 1.165 s HCCL union, 2.799 s all-op union and 0.602 s without a device op. Measured overlap between compute/copy and HCCL is ~0.054 s. A deliberately optimistic fixed-work activity floor for this same short workload is `max(1.688, 1.165) = 1.688 s` if all communication overlaps and idle is removed, or ~303 output tok/s versus observed 141. A no-overlap, no-idle hypothetical is 2.853 s (~179 tok/s). Neither is a credible achievable target yet: graph dependencies, other ranks, profiling effects and required idle are not bounded. The official 48×1024 c12 achievable bound remains UNKNOWN.

A profiled cold 4×32K→128 c1 set took 18.915 s E2E; TP0 event span 18.583 s, compute/copy union 10.829 s, HCCL union 6.386 s and all-op union 16.510 s. Four distinct prompts ran sequentially; aggregate activity does not give a single-prefill bound. HCCL task durations may include waiting, so the next A/B must test whether FlashComm1 costs are actually removable in DP1/TP8. Evidence paths and hashes are in `evidence/20260920_diagnostic/app_profile_index.json`.

## Update 2026-09-20 16:39 UTC — Loop 004 bound status

No runtime bound changed: FlashComm1-off alone failed before execution because DSA CP requires SP. The measured 1.165 s warm-window TP0 HCCL activity cannot be assumed removable by this invalid configuration. The pending coupled-path A/B will test whether an alternative valid execution trades communication for compute and improves E2E.

## Update 2026-09-20 16:55 UTC — Loop 005 bound status

The coupled-path service loaded, but the benchmark was gated by invalid exact-output matching before performance measurement. Runtime and bound estimates remain unchanged. A functional check will permit exploratory E2E comparison; no candidate can be marked KEEP until correctness is established with a stable gate.
