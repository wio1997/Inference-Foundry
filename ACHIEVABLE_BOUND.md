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

## Update 2026-09-20 17:06 UTC — tested bound on coupled communication path

The valid SP/DSA-CP-off execution delivered no measurable E2E improvement (median output TPS -1.61%, inside baseline spread). Thus the prior 1.165 s TP0 HCCL union is not an independently removable end-to-end gap by disabling this coupled path. It may be required, replaced by other communication or exchanged for compute; cross-rank and candidate device traces were not taken. Keep the official workload achievable bound UNKNOWN. The next isolated DSA CP A/B will separate part of the tradeoff.

## Update 2026-09-20 17:27 UTC — DSA CP isolated result

DSA CP off with FlashComm1 on regressed E2E median TPS 4.70%, TTFT 15.1% and TPOT 3.8%. It is not a route to remove the measured HCCL activity. The joint-off path showed no gain, and FlashComm1-only is invalid, so the communication activity envelope still cannot be converted into an achievable E2E bound. The next bound update requires target/draft host/device scope timing and cross-rank overlap.

## Update 2026-09-20 17:59 UTC — profiled envelope, bound still unknown

In the short warm profiler window, TP0 was device active 4.094/5.473 s and inactive 1.379 s by interval union; cold active 9.481/11.178 s and inactive 1.697 s. These inactive spans are only an optimistic diagnostic envelope. They cannot all be eliminated: interrequest boundaries, NPU dependencies, other ranks and the profiler itself contribute. HCCL duration may include waiting and cannot be subtracted as a speedup. The torch profile increased short warm E2E from a separate 3.633 s unprofiled observation to 5.473 s, so profile-derived host time is not a credible throughput bound.

The official 48×32K→1024 c12 achievable throughput bound remains **UNKNOWN**. Next measure a specific synchronization removal under the original unprofiled serving protocol; any bound update must use verified same-condition before/after device, host and E2E evidence.

## Update 2026-09-20 18:17 UTC — no bound change

The stack-enabled profiler crashed at stop and lost host operator events. Its short window is invalid for host/device critical-path attribution or performance bounds. The official warm mixed workload bound remains UNKNOWN. Loop 008 device activity is a diagnostic envelope only; targeted instrumentation is required before estimating removable synchronization.

## Update 2026-09-20 18:35 UTC — candidate synchronization envelope

A no-profiler tracer found a repeated NPU scalar synchronization at DSA CP QLI metadata. Approximate TP0 cumulative deltas are 0.210 s in a 4.419 s warm measured set and 1.268 s in a subsequent 9.199 s two-request cold set. The warm fraction is ~4.8% of that diagnostic wall, but the call can wait for preceding required NPU work, and other ranks have much shorter measured durations. This is an upper envelope for a candidate experiment, not a credible bound or predicted speedup. The official 48×1024 c12 achievable bound remains UNKNOWN until same-condition candidate E2E and cross-rank timing. Correctness requires proving CPU maxima equal NPU maxima for async decode and cold prefill.

## Live update 2026-09-20 18:58 UTC — candidate has no mixed bound improvement

QLI CPU maxima eliminate the explicit NPU scalar reads while preserving measured metadata values on warm/cold samples. Three full warmed mixed passes gave median540.40 output tok/s, inside and slightly below the frozen543.65 baseline; therefore the item-call duration cannot be counted as removable mixed E2E time. Cold candidate TTFT appears lower on different prompts but has no paired baseline yet. Official mixed and cold achievable bounds remain UNKNOWN.

## Update 2026-09-20 19:20 UTC — measured cold reducible time

A same-prompt, same-warmup comparison showed QLI CPU maxima reduce cold 32K→128 c1 mean TTFT from2633.90 to2349.06 ms across eight prompts, all eight improving205–343 ms. This establishes ~285 ms of reducible wall time for that intervention and workload, conditional on the full all-step variant. It does not imply an achievable lower bound of2349 ms: further bottlenecks remain and the candidate has mixed TTFT uncertainty. Warm mixed output throughput changed only +0.52% versus paired original. The official whole-workload hardware achievable bound remains UNKNOWN; next isolate the cold branch and assess whether that gain survives without decode effects.


## Update 2026-09-20 19:48 UTC — verified cold reduction, mixed bound unknown

The prefill-only QLI intervention reduced same-prompt 16-request cold mean TTFT by 263.14 ms (10.02%), from 2626.03 to 2362.89 ms, after matched warmup with zero prefix-cache hits. This is measured reducible wall time for this workload and implementation. It is not an upper hardware performance bound or a claim that 2362.89 ms is the floor. The full 48×32K→1024 c12 output TPS median was 547.55 vs paired original 537.60 (+1.85%) and frozen 543.65 (+0.72%), both within observed baseline variation. Therefore the achievable bound for the official mixed workload remains **UNKNOWN**. Next estimate requires a profiled or instrumented warm decode critical-path hypothesis confirmed by same-condition E2E change.

## Live update 2026-09-20 19:56 UTC — no new bound

Loop013 dynamic msprof attach failed before profiling; no data changed the mixed workload achievable bound, which remains UNKNOWN. A short c12 torch-NPU profile is pending; its overhead and partial workload will keep it diagnostic rather than a hardware floor.


## Update 2026-09-20 20:24 UTC — c12 diagnostic envelope, no bound promotion

A profiled 12×512 c12 window gave TP0 device active13.038/16.459 s, compute/copy7.736 s and HCCL5.510 s. Other ranks had similar compute but variable HCCL wait and idle. The 3.421 s TP0 inactivity and host draft scope9.036 s are diagnostic envelopes; request boundaries, dependencies, other ranks and profiler overhead prevent subtracting them as speedup. The short profiled TPS387.7 differs in duration, length and instrumentation from the official 48×1024 c12 workload, so it cannot update that workload bound. The official achievable throughput bound remains UNKNOWN. A valid future bound needs a specific supported intervention with correctness and paired unprofiled E2E change.

## Live update 2026-09-20 20:30 UTC — no host speedup inferred

The 9.57 ms median prepare-input host self time and 52.89 ms median draft host total are inclusive diagnostic durations, not proven critical-path savings. Loop014 will measure substage spans and require a paired unprofiled E2E improvement before promoting any reducible-time or bound claim. Official mixed bound remains UNKNOWN.

## Live checkpoint 2026-09-20 20:33 UTC — Loop014 stage tracer

Loop014 tracer host spans are diagnostic; a large stage duration alone does not set a removable E2E bound. Official mixed achievable bound remains UNKNOWN until a correct paired unprofiled intervention.

## Update 2026-09-20 20:47 UTC — Loop014 first stage trace

Loop014 measured a 10.80-14.17ms median per-rank host span for compression-position and attention metadata construction in pure decode, but the span contains required work, possible async waits and cross-rank overlap. It is a candidate envelope only. No achievable-bound update; official mixed bound remains UNKNOWN until a specific change passes numerical correctness and paired unprofiled 48x1024 benchmark.
