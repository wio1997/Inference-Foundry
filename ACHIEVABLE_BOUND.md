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

## Live checkpoint 2026-09-20 20:51 UTC — builder trace loading

Per-builder timing is pending and cannot change the mixed achievable bound. The 10.80-14.17ms metadata span remains a candidate envelope only.

## Update 2026-09-20 21:08 UTC — Loop014 PIVOT

Loop014 found no safe way to convert the 15.165ms TP0 metadata host span into an E2E speedup: eight builders are group-specific, shared state is cached and the first builder may wait for prior device work. No change to official mixed achievable bound UNKNOWN. The remaining cold prefill wall (~2.36s) also lacks a credible lower bound; Loop015 should measure stage critical path before estimating.

## Live checkpoint 2026-09-20 21:13 UTC — Loop015 bound remains open

Cold-profile kernel totals cannot be subtracted from 2.36s kept-source TTFT: calls include post-first-token decode, rank overlap and dependency waits. The pending slot-index probe tests whether a specialized scatter path is legal. No new achievable bound; official mixed bound UNKNOWN.

## Update 2026-09-20 21:29 UTC — no bound promotion from Loop015

Captured index uniqueness on one call per rank makes a direct scatter specialization plausible only if the construction invariant and fallback are established. V2 was 69% slower than SK in an isolated kernel screen, so it cannot tighten the cold TTFT bound. Existing scatter task totals remain only a potential envelope with prefill/decode overlap unresolved. Cold and mixed achievable bounds remain UNKNOWN; Loop016 must establish a correct candidate and paired E2E effect.

## Live checkpoint 2026-09-20 21:43 UTC — Loop016 candidate bound not yet promoted

Isolated SWA scatter saved about0.83ms per representative call with page-interleaved simulated cache and index flatten, but old full-request profile call counts and multistream overlap make summed savings an invalid E2E bound. The exact cold c1 TTFT bound and mixed throughput bound remain UNKNOWN until service A/B confirms this candidate on the actual cache layout and workload.

## Live checkpoint 2026-09-20 21:47 UTC — no bound update

Loop016 service A/B has started but produced no E2E measurement. Achievable bounds remain UNKNOWN.

## Update 2026-09-20 22:08 UTC — Loop016 bound still unknown

The first service A/B stopped before candidate timing because of a dataset mismatch in the parity harness and a candidate guard that did not activate. Saved clean no-flag cold TTFT is2367.81ms over eight exact prompts. This is a comparison baseline, not an achievable lower bound. Revised candidate run pending; bounds remain UNKNOWN.

## Live checkpoint 2026-09-20 22:12 UTC — revised A/B loading

No new bound until candidate path traces and same-prompt E2E run complete.

## Update 2026-09-20 22:25 UTC — bound unchanged

A/B2 was invalid because the edited dsa_v1 path was not active under DSA-CP. The active DSA-CP callsite is now patched for A/B3. No candidate E2E measurement yet, so achievable bounds remain UNKNOWN.

## Live checkpoint 2026-09-20 22:29 UTC — bound unchanged

Active DSA-CP candidate is loading; no achieved E2E delta or bound update yet.


## Update 2026-09-20 22:39 UTC — Loop016 bound unchanged

The tested direct SWA scatter specialization made cold TTFT 2.9642% worse on eight exact-prompt requests, with zero prefix hits and eight-rank activation. Its isolated ~0.9ms/operator saving is excluded from achievable E2E bound because full-service work can overlap and incurs unmeasured cost. Cold TTFT and mixed TPS achievable bounds remain UNKNOWN; no new lower bound is claimed. Compressor ratio4 old-profile task totals are a screening signal only, not additive TTFT savings.


## Live checkpoint 2026-09-20 22:46 UTC — Compressor screening envelope

Ratio4 Compressor [8096,4096;1024,4096] occupies ~126.4ms of summed device task time per profiled cold request across84 calls. This is only a gross per-request operator-time envelope, not a TTFT achievable saving or hardware bound; overlap and AIV/CUBE attribution still need measurement. Achievable cold TTFT and mixed throughput bounds remain UNKNOWN.


## Live checkpoint 2026-09-20 22:50 UTC — local dependency verified

The ~126.4ms/request Compressor task sum belongs to84 serial per-layer Compressor→scatter→attention chains on the profiled device stream. It is a local path envelope: perfect removal of this operator on that stream could save no more than its occupied time before considering cross-stream overlap, launch overhead and changed scheduling. This is not a measured achievable lower bound; cold and mixed bounds remain UNKNOWN.


## Live checkpoint 2026-09-20 23:59 UTC — no E2E bound promotion

Shape-matched isolated Compressor timing1.630ms/call corroborates the old full-service1.503ms/call order of magnitude, but does not imply 84×isolated latency is removable from TTFT. The existing ~126.4ms/request main-shape task sum is at most 5.3% of the ~2.368s cold TTFT before overlap and redesign costs. A 25% isolated operator reduction would be only ~31.6ms gross, ~1.3% of cold TTFT, prior to overlap; this is a screening estimate, not an achievable bound. Cold and mixed attainable E2E bounds remain UNKNOWN.


## Live checkpoint 2026-09-21 00:06 UTC — no candidate bound

An isolated mBase256 tiling package is compiling. The stock same-seed reference device median is1.514ms. No candidate timing or end-to-end measurement exists, so no achievable-bound change.


## Live checkpoint 2026-09-21 00:22 UTC

No bound change while isolated mBase256 build is pending. Candidate must demonstrate correct isolated output and a material operator gain before any E2E extrapolation or service run.


## Update 2026-09-21 01:10 UTC — bounds unchanged

Loop017 mBase256 produced no valid candidate latency, numerical equivalence or E2E benchmark, so it supplies no achievable improvement bound. The 126.4ms/request main Compressor task sum remains only a gross cold screening envelope; the ~5.3% cold TTFT ratio assumes complete removal before overlap and is not attainable evidence. c12 communication.json time is entirely classified Idle, so its elapsed totals cannot be subtracted from mixed runtime. Cold TTFT and mixed throughput achievable hardware bounds remain UNKNOWN.


## Update 2026-09-21 01:15 UTC — no communication bound

Loop018 demonstrates a ~579us rank6 device timestamp offset under a common-end calibration hypothesis and residual ~206us median end spread. The profiler does not independently synchronize clocks or separate network transit from Idle Time. Neither raw nor corrected rank start spread is an achievable communication saving. Cold and mixed hardware-attainable bounds remain UNKNOWN.


## Live checkpoint 2026-09-21 01:59 UTC

First DSA-CP builder timing (up to6.50ms median/rank) is an inclusive host span, not an achievable saving; device operation enqueue/synchronization and cross-rank overlap remain unresolved. No cold or mixed E2E bound update.


## Live checkpoint 2026-09-21 02:13 UTC

QLI first-builder median host span1.735–4.306ms/rank may include synchronization and scheduling; it is not a removable bound or measured E2E opportunity. No cold or mixed attainable hardware-bound update.


## Update 2026-09-21 03:11 UTC — QLI gap not promoted

First QLI scalar host wait reaches3.716ms median/rank, but Loop011 removed the reads without a robust mixed TPS gain. Therefore no sum of QLI wait times is admitted as an attainable E2E bound. Metadata op host call ~0.36–0.41ms is also not a proven saving. Mixed hardware bound remains UNKNOWN; next quantify DSpark work and acceptance.


## Live checkpoint 2026-09-21 03:20 UTC

The 52.894ms median draft_token host scope coincides with41.860ms median device busy union on TP0, but async ownership is unknown. Their difference is not an achievable saving. No bound promoted.


## Live checkpoint 2026-09-21 03:36 UTC

DSpark proposer host run_draft28.43–32.69ms and attention metadata5.67–6.41ms per step are gross inclusive spans, not achievable reductions. The draft acceptance mean3.54–3.64 for7 speculative tokens informs token efficiency but alone yields no hardware lower bound. Mixed achievable bound remains UNKNOWN.


## Bound refinement — Loop020 causal draft attribution (2026-09-21 04:09 UTC)

For TP0 warm c12, median `draft_token` host time is 52.894 ms and device tasks causally launched inside that same-thread scope occupy 51.661 ms of clipped wall-time union. Therefore the directly observed host-only residual is at most 1.233 ms at the median under this trace accounting; it is not a guaranteed saving because trace scopes and device queues overlap. The earlier 5.67–6.41 ms host attention-metadata span cannot be treated as an independent E2E bound. The remaining achievable bound is still UNKNOWN until device task dependencies and required work per accepted token are separated.


## Bound correction — EVENT_WAIT is not active compute (2026-09-21 05:10 UTC)

Loop020's 51.661 ms causal device-task union cannot bound necessary compute because it includes long `EVENT_WAIT` tasks. Loop021 attributes 51.169 ms median clipped union to outer draft waits and 26.595 ms to waits under `vllm::moe_forward_shared`; overlapping waits across streams may double-count wall time. The Index and Pad candidates provide at most sub-millisecond observed unions when present. Therefore the host-only residual estimate of 1.233 ms is withdrawn. Achievable decode bound remains UNKNOWN until event producer/consumer streams are resolved and wait overlap with useful compute is measured.


## Bound refinement — event waits are overlapped (2026-09-21 06:10 UTC)

The 53–54 ms outer event waits are on two copy streams and gate asynchronous memcpy; the 25.734 ms MoE wait is on the shared-expert stream between layer invocations. Neither is an additive main-path saving. Measured intra-call MoE dependencies are sub-0.2 ms median and the main-stream final wait is effectively zero. Remove event-wait occupancy from compute-gap estimates. The dominant bound uncertainty is now useful accepted tokens per proposer/verification work; a k5/k7 service comparison is required.


## Constraint update — speculative length (2026-09-21 07:07 UTC)

k5 cannot establish a lower-work bound because it is invalid under TP8 sequence-parallel graph divisibility. k7 is the minimum runnable length with draft block>=5. A concrete framework capacity gap remains:96 reserved draft slots reduce the configured8192 batch budget to8096 scheduled tokens. Raising the physical budget to8288 should restore8192 scheduled tokens; its achievable E2E value remains unmeasured until Loop024.


## Bound update — scheduler reserve (2026-09-21 08:14 UTC)

Restoring scheduled capacity8096→8192 produced no screen gain beyond observed variability, so the96-slot reservation is excluded from the verified achievable gap. The next bound is architectural: compare total target-only step cost against k7 proposer+verification cost per accepted token. Do not infer this from inclusive scopes; measure matched service E2E.

## Bound update — DSpark versus target-only (2026-09-21 09:09 UTC)

Target-only54.333ms/token versus k7 DSpark17.554ms/token proves verification amortization is necessary and rules out exiting speculative decoding. Using3.64 advanced tokens and17.554ms TPOT gives an approximate63.9ms speculative cycle. Balanced attribution of the three draft layers suggests an8.62ms one-layer opportunity and a rough3.15 advanced-token break-even after bypass. These are cross-run/inclusive estimates and are not added to the verified achievable gap; Loop026 E2E and acceptance measurements must validate them.


## Bound update — draft-layer ablation (2026-09-21 10:09 UTC)

One layer accounts for a measured7.22ms of proposer model time, but it is not removable: bypass reduces advanced tokens from3.54–3.64 to1.363 and destroys E2E throughput. Exclude this7.22ms from achievable savings. The exact three-layer model still spends roughly31.1ms in proposer model execution; any future bound must preserve its logits/acceptance and target graph/runtime overhead rather than remove trained computation.

## Bound update — specialized runtime boundary (2026-09-21 11:18 UTC)

The failed v2 startup provides no numeric speedup bound because graph capture was never reached. It does establish that stock generic KV grouping is not a usable route for this checkpoint without integration work. Framework abstractions are no longer assumed necessary. The roughly31.1ms exact three-layer proposer execution remains required work until a parity-preserving replay measures otherwise; metadata, launch, dispatch and synchronization portions must be isolated by the standalone fixed-cycle contract before entering the achievable gap. No startup or inclusive host duration is added as removable time.

## Loop028 bound update — replay feasibility without a numeric savings claim

The c12 proposer inputs are structurally close to a fixed-address execution contract: every observed shape/stride is fixed and ten of twelve traced tensor fields already keep one address per rank. Two changing input allocations (`target_token_ids`, `target_positions`) are concrete state-layout work for a dedicated runtime. Exact 8-rank proposer replay shows that repeating the materialized operator/HCCL/KV segment is semantically stable at its draft-token output boundary.

No time is added to the achievable gap. The synchronized 37.021 ms first-call and 31.102 ms replay medians do not isolate launch, metadata or compute savings, and the run did not compare accepted tokens or complete state mutation. A numeric bound requires a standalone or segmented replay with explicit state inputs, target verification/acceptance parity and matched timing against the oracle.

## 2026-09-24 product-bound checkpoint

The official mixed-workload achievable hardware throughput bound is still
UNKNOWN. Run107’s synchronized, profiled target window reports median
compute union39.932 ms, communication union11.547 ms, overlap1.292 ms and
86 grouped-matmul kernel durations summing9.966 ms/cycle. These values are
instrumented activity and summed task cost, not independently removable wall
or a formal E2E lower bound. Run99’s valid571.681 tok/s establishes only the
current implemented point relative to Stock543.655. A credible bound must
separate required W4A8 target/DSpark work, per-rank HBM and HCCL traffic,
critical-path waits and serving overhead under the frozen contract. Local
Loop thresholds, including 15% over Stock, do not define the achievable
limit or terminate optimization.

## Loop047 tail exposure correction (2026-09-25)

Run175 observed18.525% parked slot-cycles in one legal warmed cohort (44 cycles with<=6 active slots). Multiplying this by the separate Run98 target event median46.560ms gives2.553s/cohort only under ideal linear target scaling and zero switching overhead. GMM shared W4 weight reads, HCCL and the fixed c12 graph may be insensitive to active slots; no c6/c12 matched target latency is measured. Do not add2.553s to the achievable gap. Run171-173 admission hold experiments establish no verified net speedup. Hardware/product achievable throughput bound remains UNKNOWN.

## Loop047 Run180 marginal target-size screen (2026-09-25)

Run180 repeats c12 after smaller graph sizes in one service and finds+1.536ms c12 endpoint drift. Using early/late c12 baselines and Run155 measured active counts gives0.471–0.602s/cohort of *zero-cost Stock target event* screen. It cannot be promoted to an Extreme achievable bound: graph/metadata bank selection, live-row packing, KV remapping and canonical-slot scatter are unmeasured, and the sampled `_model_forward` scope differs from Extreme target.execute. This small screened benefit justifies deprioritizing compaction; the frozen-product achievable throughput bound remains UNKNOWN.

## Loop048 Run184 prefill Host exposure (2026-09-25)

In one legal warmed Extreme cohort, five prefill `_model_forward` calls have per-call maximum-rank wall summing1.901s. DSA+MoE inclusive CPU scopes occupy87.7–88.0% of individual forward wall, but include required exact-state operator dispatch and synchronization. Neither the1.901s nor its scope fraction is a removable saving; comparison with Run165's profiler-perturbed 401ms device-free span cannot establish one. Official achievable bound remains UNKNOWN.

## Loop048 Run186 Host CPU bound caution (2026-09-25)

Detailed legal Extreme prefill forward wall/thread CPU per-call rank-max sums3.424/3.416s, with per-rank median ratios99.70–99.98%. The small wall-minus-thread-CPU difference rejects a large blocked-thread opportunity, but active CPU time includes required operator launch and collective/event dependency work. It is not an attainable savings bound. Different control/detailed prefill batch shapes preclude an intervention effect. The frozen-product hardware throughput bound remains UNKNOWN.

## Loop048 Run188 whole-cohort admission test (2026-09-25)

Full-cohort consolidation eliminated6–7 eager prefill calls but improved one diagnostic client envelope only0.163s versus two-control mean, while decode cycles differed32. This rules out promoting the gross2.36–2.59s prefill wall reduction to an E2E achievable saving. Single sequential A/B/A-prime does not establish a positive formal gain. Official hardware-attainable throughput bound remains UNKNOWN.

## Loop057 bound status (2026-09-25)

Official frozen-product achievable throughput bound remains UNKNOWN. Run234 target-name census does not convert kernel task sums into critical-path savings. Community shared graph pool could change the Run227 prefill MoE bank memory result, but first88 occurs only once per observed 12-request cohort and a linear extrapolation of Run225 four-layer diagnostic is ~179ms against a ~20.3s diagnostic cohort, without all43 correctness or formal E2E. Run235 fails the FP32 numerical gate for the only registered RMS+cast surrogate at local12-row shape; Run23396-row speed result does not bound the unavailable official op. No >=1ms/cycle target saving is currently source-backed.

## Extreme performance model V0 (2026-09-26, Loop058)

The executable conditional replay is `scripts/extreme_bound_model.py`; method and evidence matrix are in `evidence/20260926_loop058_bound/run236/method_evidence.md`. It reproduces all three formal Run99 samples exactly (612.962/567.573/571.681 tok/s, median 571.681) using actual 1217/1212/1206 cycles. At fixed cycles and useful tokens, illustrative exposed-saving scenarios yield median Engineering 581–607 tok/s and Aggressive 616–682 tok/s. These are conditional projections, **not** a proved achievable hardware bound. Current is 94–98% of the illustrative Engineering range and 84–93% of Aggressive; distance to the true hardware limit remains UNKNOWN. A current valid 612.962 TPS sample exceeds the median-trajectory Engineering endpoint, so workload trajectory and serving variation must be conditioned.

Run146 routed packed GMM weight estimate is ~8.462 GB/rank/target cycle; Runs148/150 one-card memory counters read ~1.055/1.075× active packed W1/W2 at ~1.118/0.995 TB/s. This is a conditional GMM screen, not an eight-rank floor. Run107 device/communication unions cannot be summed or subtracted; Run145 first collective includes arrival wait. New Run237 TP8 BF16/FP32 forty-pair chain measured 0.53549 ms/pair latest-rank median without interleaved compute. It measures a different eager API path and does not calibrate product graph HCCL capacity (Run240 correction). Run238 formal wave audit shows client-minus-decode windows 2.545–3.148 s in the fastest pass versus 3.848–4.521 s in slower passes, locating much of E2E variation outside runtime-owned decode wall. Attribution among prefill, admission, queueing and publication is open. Full KV bytes, non-GMM shape capacity and resource-DAG dependencies remain unmeasured.

## Loop059 Run239 legal boundary pass (2026-09-26)

The reused Loop045 reversible patch was applied to exact prior source SHA and removed after the run. Warmup and diagnostic pass each completed 48/48 exact 1024 output; measured pass 592.618 tok/s is instrumented and not a new formal product point. Across measured cohorts 5–8, same-host all-rank boundaries give client-to-first-execute 0.207–0.234s, first-execute-to-handoff 2.356/3.747/4.115/3.318s, runtime serve 16.800/17.504/16.241/17.268s, and publication-to-client-end 0.173–0.178s. Handoff/build and serve/publication edges are <=0.007s. All eight rank runtime rows passed. The full phase records and client requests are at `evidence/20260926_loop059_boundary/run239/`; raw service log is retained by SHA index. Exit 0, service stopped, source hashes restored, eight NPUs idle.

This calibrates the V0 residual: prefill-to-handoff and decode work both vary substantially across cohorts; admission and output publication are small, stable in this one pass. It does not identify removable prefill time or a hardware upper bound. Next update the executable model with measured phase distributions and compare counterfactual savings against full-cohort dependency and Run188 admission tradeoff; only then rank prefill architecture against target DAG.

## Loop059 Run241 phase calibration and trajectory coupling (2026-09-26)

`scripts/extreme_bound_phase_calibrate.py` reads Run239 same-host all-rank boundaries and Run188 admission A/B/A-prime. The 82.940s instrumented pass contains 13.536s first-execute-to-handoff prefill (16.32%), 67.813s runtime serve (81.76%), and 1.600s in the measured client/admission/build/publication edges (1.93%). Across 1195 decode cycles, latest-rank runtime wall is 56.782ms/cycle on this trajectory. This is phase accounting, not necessary-work or removable-time attribution.

At fixed cycles and unchanged semantics, a hypothetical 0.5s/cohort exposed prefill saving would project 607.261 TPS from diagnostic 592.618. Adding 8 decode cycles/cohort reduces that projection to 593.928 TPS. Run188 actual admission hold removed 2.476s gross prefill forward wall but added 32 decode cycles and 1.711s runtime wall; observed client improvement was only 0.163s versus controls. The simple gross prefill minus runtime arithmetic predicts 0.764s improvement and misses the observed outcome by ~0.602s, due to overlap, changing shapes/acceptance and other boundary effects. Therefore fixed-trajectory Engineering/Aggressive scenarios remain conditional and cannot be promoted to achievable product bounds. Evidence: `evidence/20260926_loop059_boundary/run241/calibration.json`.

## Loop060 Runs242–243 resource inventory (2026-09-26)

`extreme_resource_inventory.py` records source hashes, 43 target layers (21 c4, 20 c128, 2 uncompressed), Run115 product W4A8 shapes and Run146 real c12 routes. The target GMM has 8.462GB active packed expert weights and about 155.676GFLOP logical matmul/rank/cycle; applying Runs148/150 one-card counter bandwidth gives a conditional 7.880ms packed read estimate versus Run107 profiled 9.966ms GMM task sum. Different cohorts and concurrency prevent turning the 2.086ms difference into exposed gain. Cache ABI records c4/c128 state dimensions, BF16 SWA and block-size-32 mappings, but actual KV HBM bytes and attention reads remain UNKNOWN. Four Run239 warmed prefill cohorts have 7/11/12/10 scheduled-token calls, not a single repeatable prefill shape. Evidence: `evidence/20260926_loop060_resource/run242/inventory.json`.

`extreme_kv_row_census.py` reuses Run84 eight-rank 256-cycle legal page audit without service. Over cycles64–255, c4 compressor and indexer each emit exactly24 valid rows/rank/cycle (504 layer-rows across21 layers), while c128 compressor emits median1, mean0.698 valid rows/rank/cycle (20 layer-rows median across20 layers); first c128 write is cycle8 on all ranks. Page candidate counts and row cardinalities are not HBM transactions. SWA, MTP, cache read reuse and physical page padding must be accounted before converting these to bytes. Evidence: `evidence/20260926_loop060_resource/run243/kv_rows.json`. The largest unresolved target resource term remains non-GMM DSA/quant/attention execution and actual KV read traffic; next measurement must preserve real shapes, dependencies and graph path.

## Loop060 Run247 eight-rank graph memory counters

Run246 legal instrumented capture was exported and checked in Run247: all 80 target rank-cycle windows passed kernel-count and positive-counter gates, including 16 latest windows across all eight ranks. Median AIC+AIV reads are 9.244GB GMM, 3.465GB quant matmul, 1.530GB compressor, 1.062GB sparse attention and 3.618GB other per rank-cycle; 18.965GB read and 2.380GB written overall. GMM read divided by the Run146 active packed weight estimate is 1.092 across different route/cycle samples; the numerator also includes non-weight reads, so this does not quantify same-cycle weight amplification. AIC-only GMM read is 8.990GB; omitting vector counters would understate traffic and falsely report zero scatter writes. HCCL link bytes are not present. These task counters may include repeat reads, and the synchronized profiler changes timing. The median 54.796ms target scope is not a formal E2E bound. No unique compulsory-byte floor, cross-rank attainable bandwidth or full critical path has yet been established. Details: `evidence/20260926_loop060_resource/run247/findings.md`.

## Loop060 Run248 traffic attribution

Across 16 latest valid rank-cycle windows, the 3.618GB `other` read group decomposes into plain matmul1.149GB, transpose batch matmul0.639GB, inplace copy0.618GB, AivKernel/Hc* names0.756GB, indexer0.097GB and remaining0.357GB per rank-cycle medians. Gross reported read+write is 21.344GB. At hypothetical 1.0/1.3/1.6TB/s, simple byte/rate screens give 21.344/16.419/13.340ms, **not** lower bounds: these are task counters rather than proved unique compulsory traffic, and attainable concurrent graph bandwidth is unknown. `communication.json` has zero transit size for inspected graph HCCL tasks. No valid subtraction from the profiled target time or formal E2E is available. The GMM reported-read/active-packed cross-sample ratio is 1.092, which is not a same-cycle traffic amplification factor; next work investigates same-shape non-GMM traffic and dependencies without presuming priority. Official bound remains UNKNOWN. Evidence: `evidence/20260926_loop060_resource/run248/findings.md`.

## Loop060 Run249 shape-rate discrimination

The largest real graph quant matmul shape reads1.533GB in1.469ms summed profiler task time across43 calls, a 1.043TB/s counter rate. Three main Compressor shapes read0.679/0.505/0.347GB in1.283/0.951/1.145ms; transpose matmul reads0.639GB in1.484ms. These ratios under a synchronized profiler do not represent removable time or a graph-wide bandwidth ceiling. Loop044 showed the 62 Compressor calls produce distinct required outputs and Run197 showed that direct cache writes require a custom op ABI/kernel. No valid implementation candidate or hardware-attainable bound follows from this shape screen. Evidence: `evidence/20260926_loop060_resource/run249/findings.md`.

## Loop061 hardware and model-status correction (2026-09-26)

The bound hierarchy is now explicit:

| Layer | Current evidence | Status |
| --- | --- | --- |
| Current Formal E2E | Run99, 571.681 tok/s median; all 48/48×1024 and rank gates | measured |
| Engineering Bound | V0 581–607 tok/s fixed-trajectory sensitivity, not a constructed executable schedule | UNKNOWN |
| Aggressive Bound | V0 616–682 tok/s fixed-trajectory sensitivity, not proven graph/fusion/persistent capacity | UNKNOWN |
| Hardware/Algorithmic Ceiling | Frozen compulsory FLOPs, unique HBM/KV bytes, HCCS wire bytes, full dependency DAG and useful-token efficiency are incomplete | UNKNOWN |

Astra High's independent Run255 review confirms the V0 arithmetic exactly replays formal samples, but its savings parameters are assumed rather than measured. Neither interval has an attainability confidence level. The Run247 GMM 9.244GB / Run146 active-packed 8.462GB = 1.092 ratio joins different route/cycle samples and includes non-weight reads; it is not a same-cycle amplification factor. A measured one-card or HCCL Test rate is an attainable *example*, not an absolute capacity upper bound suitable for a strict resource lower bound.

Run250 extracted the exact 8-rank graph trace collective signature for 80/80 windows: 265 HCCL AivKernel events and 25,651,200B of reported Size(Byte) per rank-cycle. This is operation payload, not HCCS wire traffic. Exported link/transport types are INVALID_TYPE and communication.json transit sizes are zero. Run253 CANN 9.1.0 official HCCL Test (8×910B3, matching data_size) measured normal 96KiB BF16 AllGather/ReduceScatter/AllToAll at 159.32/114.36/211.07us; paired Run254 with HCCL_BUFFSIZE=256MB measured -t1 device-only 40.03/39.74/62.17us. These standalone numbers do not reproduce captured Graph interleaving, arrival or overlap and cannot be multiplied by 265. They inform a future same-path communication calibration but do not produce an E2E bound. Both runs passed tool correctness; eight NPUs are idle. Full method and limits: evidence/20260926_loop061_bound/findings.md.

The next model step must derive per-node unique tensor footprints and necessary arithmetic from the actual frozen model, then identify Graph-path resource sharing and dependency edges. For each node, Current cost, optimistic physical lower bound, measured executable engineering cost and confidence should be separate fields. Unknown values remain null; do not fill them by summing profiler durations. The first candidate screen is source-backed non-GMM dataflow with same-state target Graph intervention; the priority may move to communication, scheduling, acceptance or broader execution architecture if exposed savings evidence warrants it.

## Loop062 executable bound node: unused MTP stash in DSpark (2026-09-26)

The frozen product uses DSpark7. In the borrowed DeepSeek V4 target, a pre-hc-head MTP residual branch still performs one AllGather and two 8192×16384 BF16 copies per FULL Graph cycle, although the DirectTarget/DirectDSpark runtime never consumes that buffer and the borrowed runner only rebinds it for `method == "mtp"`. Run263 records the source-consumer check. This branch has **zero compulsory work under the frozen DSpark algorithm**; the original MTP method remains unchanged.

| Graph node | Current measured implementation | Executable Engineering candidate | Critical-path evidence / confidence |
| --- | --- | --- | --- |
| DSpark-unused MTP hidden stash | Run257: 2×256 MiB BF16 TensorMove, 0.5375 GB reported read and 0.5374 GB write/rank-cycle, 0.799 ms summed tasks; Run250: one 393,216 B reported AllGather payload | Run260–262 opt-in DSpark guard: both copies absent in 16/16 latest windows; AllGather payload absent in 15/16 exact HCCL windows, one window has adjacent asynchronous events | Removal of work is high confidence. Exposed wall saving is **unknown**: synchronized target-scope medians 54.796→54.418 ms across separate runs, but matched rank-cycle deltas split 8 faster/8 slower. Formal E2E pending. |

The candidate reduces median `other` profiler read 3.618→3.066 GB and write 1.355→0.812 GB/rank-cycle while preserving the counted target math kernel families. These AIC+AIV task counters are not unique compulsory HBM bytes. Run260's instrumented 48-request warmup and 12-request sample passed all 8-rank runtime checks; profiler throughput is invalid for product comparison. This local node is only one edge of the larger dependency DAG. The model's Engineering/Aggressive/Hardware TPS intervals remain UNKNOWN until formal intervention and broader compute/HBM/HCCL/host/algorithmic calibration. Evidence: `evidence/20260926_loop062_nongmm/run257/`, `run261/`, `run262/`, `run263/`.

## Dual-bound V1: resource and scheduling limits (2026-09-26, Run265)

The model now keeps four distinct objects:

| Layer | Quantity and proof requirement | Current state |
| --- | --- | --- |
| Current | Measured complete execution DAG and repeated frozen Product E2E | Run99 baseline 571.681 tok/s median; Run259 candidate 594.133 pending contemporary control Run264 |
| Hardware/Resource Bound | Necessary FLOPs, unique compulsory HBM/activation/KV/metadata bytes and collective wire work divided by **attainable concurrent** 8-rank FULL Graph compute/HBM/HCCL capacity, with resource sharing | Numeric time lower bound UNKNOWN |
| Scheduling-aware Bound | Shortest dependency and resource-constrained makespan under legal state lifetimes, rank rendezvous and overlap; distinguish this lower bound from an actually implemented schedule's measured cost | Numeric time lower bound UNKNOWN |
| Product E2E Bound | Prefill, useful tokens/cycle and acceptance, finite 48-request trajectory/parking, Host and publication joined to decode DAG | Numeric throughput ceiling UNKNOWN |

`scripts/extreme_dual_bound_v1.py` emits machine-validated current and candidate dependency graphs at `evidence/20260926_loop062_nongmm/run265/model.json`. Edges are typed as semantic RAW, KV-state RAW, storage WAR and current implementation order. It computes a critical-path relaxation only when all same-path node costs are supplied, and labels that result optimistic because resource contention and communication arrival are not yet modeled. It emits **no TPS bound** while those inputs are absent. The old V0 581–607 / 616–682 tok/s intervals remain conditional sensitivity scenarios, not renamed to these bounds.

Astra High independently identified a scheduling opportunity: after acceptance and state advance, next-cycle target positions, lengths, block slots and RoPE/SAS/QLI metadata have their mathematical inputs, while current DSpark still runs. They do not need the next draft token IDs. Current DSpark still reads the old target positions/ids/slots, and metadata reuses a global RoPE buffer, so early in-place overwrite is invalid. An executable candidate needs private scratch output, a safe commit into fixed Graph addresses after old-state readers finish, and invalidation when serving parks a slot after the step. The causal control is current A0 versus serial scratch+commit A versus side-stream scratch overlapping DSpark B. Old Run98 ~0.667ms metadata is a screening figure only; simultaneous execution may increase AICPU/AIV/HBM contention. Correctness and repeated formal E2E decide KEEP. Details and independent confidence assessment: `evidence/20260926_loop062_nongmm/run265/astra_review.md`.

DSA's main compressor/indexer/query branches and sparse-attention join remain a larger possible fan-out/fan-in scheduling space. Existing DSA CV multistream and delayed Host count copy are already counted as Current, not future savings. Same-token cross-layer hidden recurrence is a semantic edge; current operator, collective, Target/DSpark and cycle boundaries are implementation choices unless source/data-dependency proof shows otherwise.

Run266 adds a same-protocol **observed schedule map**, not an ideal schedule: 16 latest baseline rank-cycle windows have median device span54.161ms, compute/copy union39.903ms, HCCL union11.200ms, overlap1.078ms and gap4.649ms. The DSpark-stash candidate has 53.745/38.905/9.376/1.113/4.794ms respectively in a different profiler run. HCCL AivKernel rows were excluded from compute to avoid double-counting communication; interval union is used rather than summed kernel time. Cross-run differences, especially HCCL waiting and asynchronous scope spillover, are not an intervention's exposed savings. The 4.649ms gap is not automatically eliminable: host submission, stream event waits and rank arrival need attribution. This map is the starting point for semantic-versus-implementation DAG reconstruction. Evidence: `evidence/20260926_loop062_nongmm/run266/`.

### Run267 correction: first collective duration includes cross-rank arrival (2026-09-26)

The first 98,304 B `reduce_scatterAivKernel` in each latest rank target scope finishes almost together despite skewed entry. In Run246 baseline cycles 0/1, the eight ranks start that collective 22.752/8.413 ms apart, while completion spreads are only 0.008/0.012 ms. The candidate Run260 cycles have 6.449/9.346 ms collective-start spread and 0.015/0.024 ms completion spread; one ranks second candidate scope includes three prior asynchronous events, so matching by collective identity is required. The large first HCCL duration is mostly rank-arrival wait in these synchronized profiler captures. Run266s 11.200 ms median HCCL interval union must **not** be read as collective transport time, compulsory communication or an E2E removable term. Rank skew may itself arise from profiling synchronization; unprofiled cross-rank cycle timestamps and event readiness are required to determine whether any of that wait persists in product execution. Evidence and exact per-rank durations: `evidence/20260926_loop062_nongmm/run267/`.

### Loop062 formal calibration and verdict (Run259/264/268)

The opt-in DSpark-unused MTP stash elimination is a real resource reduction, but no exposed Product E2E saving has been proved. Three candidate formal TPS samples are 583.892/603.802/594.133 (median594.133); the same-host unpatched control samples are 582.852/574.855/588.301 (median582.852). Both have exact48/48 outputs per pass, FULL Graph and all128 rank/cohort records passing. The median difference is +11.282tok/s (+1.936%), yet latest-rank runtime normalized by actual cycles differs −0.093/+0.671/−0.150ms per cycle across the three passes and client-minus-runtime residuals differ −2.073/−6.057/−1.501s. This sequential comparison establishes no stable exposed-cycle saving, even though graph traffic removal is causal. Do not promote either median to a Hardware/Resource or Scheduling-aware bound. Accepted Current Formal remains Run99 571.681tok/s pending a reproducible KEEP. The guarded intervention remains available for a future stronger same-state causal test. Details: `evidence/20260926_loop062_nongmm/run268/compare.json`.

## Loop063 partial resource inventory and executable scheduling experiment

Run271 updates the dual-bound executable DAG: `state_advance → DSpark old-state preparation` is a semantic RAW edge because the proposer consumes the advanced acceptance/count state; it is not merely current implementation order. For a fixed c12 target cycle, Run242's sampled real route implies 155.675787264 GFLOP of canonical routed GMM matmul per rank, and Run256's 62 fixed-shape Compressor calls imply 58.38471168 GFLOP of canonical two-projection matmul per rank. Their sum is a **partial accounting of current algorithmic arithmetic**, not a proof that all those FLOPs are an irreducible mathematical complexity lower bound or that target work is complete. Run242's 8.462GB active packed GMM weights and Run256's 0.60817408GB Compressor weights are distinct parameter tensor footprint screens, not compulsory HBM reads. Full target/DSpark/prefill FLOPs, KV traffic and eight-rank concurrent capacity remain unknown, so Resource/Hardware latency and TPS bounds remain null in `evidence/20260926_loop063_schedule/run271/model.json`.

Run269 fixes the first Scheduling-aware test: acceptance determines next target geometry and metadata before DSpark draft IDs exist. A serial private scratch/commit control and a side-stream scratch/commit candidate test whether this semantic independence reduces the complete critical path. Scratch must not alias the old target, DSpark or KV storage; serving parking invalidates it. An eight-rank parity diagnostic, actual candidate-consumed correctness and verification-free A0/A/B timing are separate gates. Astra High Run273 found the initial verification implementation rewrites active buffers with its serial reference before Target, so Run270 can only establish stable metadata parity. This has to be fixed before claiming candidate-consumed correctness or E2E benefit. The small Run98 metadata time is not a bound on broader scheduling opportunities; DSA query/KV/indexer/compressor fan-out remains open.

Run271 recalculates Run247 totals per **whole rank-cycle window**, rather than summing independent family medians. Across the latest 16 windows, whole-window median reported read is 18.9497552GB and write 2.380326528GB. The earlier 18.965061632GB read and 2.380235712GB write are sums of per-family medians; both aggregations are valid descriptive summaries, but the latter is not the median of a real window. This 0.08% read distinction has no bound impact, yet the executable model now labels both so resource accounting stays auditable.

### Run276: Frozen DSA CP dependency correction and next scheduling opportunity

Astra High's independent source review found that the frozen `enable_dsa_cp` Target executes `AscendDSACPImpl._forward` in `context_parallel/dsa_cp.py`, **not** the CV multistream `dsa_v1.py` implementation. Therefore the latter's three-block overlap cannot be credited to Current. In the actual c4 CP path, indexer cache update precedes indexer query/QLI, while main compressor and its KV scatter depend on gathered hidden and their own cache/metadata, not on QLI indices. Current code serializes indexer query/QLI before main compressor; sparse attention is the genuine join. A private-stream fork of main compressor/scatter after indexer-cache update, concurrent with indexer query/QLI and joined before sparse attention, is a concrete Scheduling-aware candidate. It keeps TP collective order. The semantic independence is high confidence; storage/workspace safety, Graph capture, resource contention and exposed Product E2E gain are unknown. A one-layer same-state causal gate, all-rank complete Target/cycle measurement and repeated formal E2E are required before promotion. Run276 review: `evidence/20260926_loop063_schedule/run276/astra_cp_review.md`.

The executable V1 model now includes a nested c4 CP Target DAG with typed semantic and implementation edges (`scripts/extreme_dual_bound_v1.py`, Run271 model JSON). It represents the original `indexer QLI → main compressor` order and a legal fork after indexer cache, with their true fan-in at sparse attention. A synthetic one-unit node-cost fixture changes the local no-contention critical path from 8 to 7 units, validating DAG calculation only. Real node costs and shared-resource contention are absent, so both reported c4 critical-path times remain null and no TPS number follows.

Run275 fixed the verifier so that after comparing the candidate's stable metadata headers with a serial same-state rebuild, it restores the **entire candidate scratch** before Target. The eight-rank overlap diagnostic exited 0: warmup48 and bench12 each produced exact1024 tokens; all 40/40 rank-cohort rows passed FULL Graph/runtime gates; 40/40 storage-interval audits had no overlap with protected Target, DSpark or KV storage; all eight ranks reported parity at candidate cycles1/64/128/256 across five cohorts. This establishes candidate-consumed continuous correctness within the checked scope, including parking and cohort exit. The verification-on 461.847 tok/s bench12 is **not** a performance sample. Run277 verification-free A0/B screening is the next gate.

Run271 model now distinguishes a **physical resource relaxation** (compulsory work divided by a defensible capacity *upper* bound) from an **attainable engineering scenario** (a concrete allocation using measured concurrent FULL Graph capacity). Observing a bandwidth or FLOPS level proves that level can be reached under its test conditions; it does not prove the hardware cannot go faster and therefore cannot by itself create a strict latency lower bound. Scheduling-aware analysis similarly separates a dependency/resource critical-path relaxation from an actually executable resource-constrained schedule with storage, rank arrival, events and contention. All numeric fields remain null until those inputs are measured or bounded.

The frozen DSpark7/Target8/c12 cardinality gives a loose **algorithmic** screen: at most 8 useful outputs per active request per target cycle, or 96 across 12 slots. Exactly 49,152 requested outputs therefore require at least 512 target-cycle equivalents, whereas Run99 recorded 1217/1212/1206 complete cycles. This 512 is only a count lower bound; the actual acceptance distribution, parking, prefill and resource-dependent cycle duration prevent converting it to TPS or declaring the difference removable. It is recorded explicitly in Run271 model JSON so Scheduling/Execution and Product accounting cannot silently assume unlimited useful tokens per cycle.

Run277/278 verification-free scheduling screen rejects promotion of the current next-target-metadata side-stream implementation. B exited0, while contemporary A0 emitted complete benchmark/rank artifacts but launcher exited127 after the active shell script was edited; A0 is diagnostic-only. B versus A0 had +1.472ms/cycle latest-rank runtime, +1.803s total runtime and +1 cycle, with all four cohort per-cycle deltas slower. The B client TPS605.801 versus A0 574.437 arose alongside a −6.233s client-minus-runtime residual difference, so no exposed Target/cycle benefit is established. Formal Current remains571.681. The specific metadata implementation is PIVOT, not a Scheduling-aware ceiling. Run276's real c4 CP fork/join remains the next architecture candidate. Full evidence: `evidence/20260926_loop063_schedule/findings.md`.


### Loop064 semantic scheduling refinement (2026-09-26)

The 512-cycle cardinality floor applies to the actual frozen Run99 Runtime path because the borrowed runner passes `initial_output_counts=[0]*12` for each cohort. All 96 measured rank/cohort reports for the three formal repeats record 1024 outputs per slot after handoff. In general, use `sum_cohort max_slot ceil((1024 - initial_output_count)/8)`; the Run99 value is 4 cohorts × 128 = 512 per formal repeat. This is a loose count bound only.

Astra High's independent source review found that the earlier coarse c4 DSA CP DAG bundled `qr` production with completed main Q, and bundled indexer query preparation with the final QLI cache read. Those bundles impose implementation order that is not required by the finer semantic dependencies. `scripts/extreme_dual_bound_v1.py` now records both the coarse restricted current/fork schedules and a separate fine semantic DAG. The fine graph can support a Scheduling-aware relaxation only after same-path node costs, resource contention, native workspace lifetimes and 8-rank rendezvous are measured. The numeric Hardware/Resource, Scheduling-aware and Product E2E bounds remain unknown. Independent review: `evidence/20260926_loop064_cp/astra_high_review.md`.

Run280 selected the intended c4 layer on all eight ranks during capture but stopped on an audit design error: the storage utility flagged overlap between two tensors within the main branch. It generated no performance evidence. The service was stopped and the borrowed source restored; Run281 retries with a cross-branch-only alias audit. See `evidence/20260926_loop064_cp/run280/findings.md`.


The executable model now also records the three existing Run99 Product wall envelopes without rerunning them. The three client walls are 80.188/86.600/85.978 s; sums of each cohort's slowest-rank Runtime wall are 69.061/69.477/69.049 s, leaving descriptive cross-clock residuals 11.126/17.124/16.929 s. These residuals combine Prefill/admission/output and clock-boundary effects; they are neither a measured removable gap nor a Product Bound. Their variation while Runtime wall stays near 69 s makes the serving boundary a required term in the future critical-path model.


Run281's one-layer `immediate` fork captured all 16 FULL Decode Graph sizes and passed visible cross-branch KV checks on all eight ranks. It also produced two exact 12-request Runtime cohorts (16/16 rank/cohort reports pass). But warmup48+bench12 should have yielded five cohorts, and only two entered FixedCohortServing; the other client outputs used the surrounding path. Thus the 311.397/277.869 tok/s client diagnostics are mixed-path observations, not comparable Extreme E2E evidence. The numerical/resource/scheduling bounds remain open. Astra High's independent review rejects an unmeasured assumption that max-num-seqs16 caused the coverage drift; a bounded handoff predicate/shape probe is prepared to identify the actual gate behavior. Evidence: `evidence/20260926_loop064_cp/run281/findings.md`, `evidence/20260926_loop064_cp/astra_handoff_capacity_review.md`.

Run282 captured the one-c4-layer CP fork on a second NPU stream in all eight FULL Graph traces. The moved Compressor overlaps indexer query preparation for about 65 μs, while QLI itself follows the Compressor. Astra High found that the concurrent indexer Rotary median is about 64.19 μs versus 5.67 μs in historical baseline traces. Different profiling configurations prevent a causal subtraction, but the inflation makes the overlap rectangle unsuitable as a scheduling saving. Run284 uses the same profiler configuration with an immediate wait as the direct control. Until its full fork-to-attention and Target/cycle results exist, Scheduling-aware numeric savings remain unknown. See `evidence/20260926_loop064_cp/run282/astra_timeline_review.md`.

Run283 original CP path, with a reversible borrowed-runner handoff predicate probe, produced five exact 12×8 handoffs mapping to all four warmup cohorts and the bench cohort. All 40/40 rank/cohort Runtime reports passed FULL Graph gates. Its five ready events were each preceded by prefill or partial-admission shapes; Run281 candidate coverage of 2/5 remains an unexplained cross-run effect. The 564.524/548.293 tok/s probe client rates are diagnostics, not formal E2E. The probe was restored to the original source SHA. See `evidence/20260926_loop064_cp/run283/`.

Astra High identified a larger conditional Resource/Hardware question. Fixed query_start_loc `[0,8,...,96]` maps the 12 local query rows on each rank to exactly two logical requests, 16 request-rank memberships total, while current c4 CP gathers all 96 hidden rows and updates all 12 request replicas on every rank. An owner of a split request still needs the full historical KV and all eight new rows for that request, so the conditional screen is 16 full request rows per rank, not merely 12 local query rows. Native QLI and c4 attention address local-query request rows of local caches, but Target/DSpark storage alias, recurrent state, physical page sharing, prefix validity and future cohort reuse remain unclosed. The 96 current versus 16 conditional memberships are **not** a sixfold byte or time gain and do not alter the bound until consumer/lifetime evidence and a same-state intervention support deletion. The next read-only probe should capture ownership, physical read/write ranges and prefix frontier before a one-layer indexer update experiment. See `evidence/20260926_loop064_cp/astra_resource_ownership_review.md` and `astra_ownership_lifetime_review.md`.

Community asset check on 2026-09-26: vLLM-Ascend ModelRunner V2 [DSpark FullGraph](https://github.com/vllm-project/vllm-ascend/pull/12017), [DeepSeek V4 DSpark](https://github.com/vllm-project/vllm-ascend/pull/12968), [unified main/draft stream fix](https://github.com/vllm-project/vllm-ascend/pull/13600), and [DSA Decode ACLGraph](https://github.com/vllm-project/vllm-ascend/pull/14762) are merged. The current borrowed framework HEAD contains the DSpark FullGraph merge commit; Extreme still uses the frozen MRV1 product path. These implementations are candidate assets and correctness patterns, not a comparable 910B3 W4A8 TP8 DSpark7 performance ceiling.

A historical Run75 eight-rank cache manifest contains 67 cache views per rank, including three `mtp.*` draft cache views. Recomputing conservative storage intervals from storage base, offset, shape, stride and dtype found zero Target-versus-MTP cache-view interval overlaps on all eight ranks (`evidence/20260926_loop064_cp/ownership_manifest_reaudit.json`). This narrows one alias concern for that bootstrap only; non-cache state, current allocations, shared request pages and prefix lifecycle remain unproved. No non-owner update is deleted on this basis.

Run284 completed the matched one-layer immediate-join profiler control. Across 16 rank-cycle samples, delayed join reached the next AllToAll completion 39.875 µs earlier at the paired median (16/16 faster). This is a local cross-run screen. Immediate itself changes the original A0 order, and different acceptance trajectories prevent a full-cycle subtraction; no formal gain is established. Astra High requires an eager same-prestate A0/overlap/A0 numerical and post-cache gate before Graph/all-layer promotion. `scripts/extreme_dual_bound_v1.py` now exposes distinct Current, Hardware/Resource, Scheduling-aware and Product E2E fields, keeps the latter three numeric values null, and records the Run284 local screen without converting it into a bound. Evidence: `evidence/20260926_loop064_cp/run284/`.

Run285 eager Target five-pass `off/overlap/off/immediate/off` from one captured prestate completed on all eight ranks; 12/12 diagnostic requests returned exactly 1024 tokens. The first/third off matched integer decisions, while first/fifth off after candidate passes differed. The same-state numerical gate is inconclusive because an uncaptured candidate side effect and ordinary replay variation are not separated. Whole-cache alias-view float deltas include mixed typed regions and invalid slots, so they are not layer2-only semantic errors. Astra High recommends a fixed fork-entry typed first-divergence gate if this CP fork is revisited. No all21 or formal promotion follows. Run286 instead tests the larger conditional owner/replica-update Resource gap on the original path with actual request geometry and page frontiers. Even if ownership remains stable, actual scatter, recursive state reads, DSpark/prefix consumers and all-rank lifetime must be proved before any traffic is called removable. Evidence: `evidence/20260926_loop064_cp/run285/` and `astra_ownership_probe_checklist.md`.

## Loop064 Run287 observed ownership geometry (2026-09-26)

Original-path Target FULL Graph read-only census covered 8 ranks x 5 cohorts = 11,968 rank-cycles. Every cycle kept global qsl 0:8:96, local 12 query rows per rank, two request-owner slots, and 16 full input update rows for those owners. 60/60 client outputs were exact 1024; 40/40 Runtime rank-cohort gates passed. Conservative layer2 owner compressed-history page envelopes did not overlap derived nonowner new compressed pages, but actual native scatter, state history, DSpark, cross-cache storage aliases and later prefix consumers are not closed. The observed 80 logical nonowner rows are therefore no HBM or TPS saving claim. Probe-induced Host sync makes its client TPS noncomparable. Resource/Hardware, Scheduling-aware and Product E2E numeric bounds remain unknown. Evidence: evidence/20260926_loop064_cp/run287/findings.md and dual_bound_v2.json.

## Loop065 H003: hidden AllGather / local Q scheduling screen

The frozen CP Target layer2 receives local BF16 `[12,4096]` and gathers `[96,4096]`. Source dependence permits its local Q chain to start from the local hidden while the full hidden AllGather runs; WKV/cache update still waits for the gathered hidden. The executable model records this as an alternative Scheduling-aware graph edge set, with unchanged necessary arithmetic, traffic and TP group. This does not assign a numeric Hardware or Product bound.

Run292 delayed-wait B passed the isolated 48+12×1024 client contract and 40/40 eight-rank FULL Graph cohort gates. Latest-per-rank profiling across two cycles gives 16/16 device intervals where hidden AllGather overlaps actual local Q kernels. Median HCCL task 14.88 µs, device overlap 13.75 µs; all gathers finish before the first WKV consumer, with near-zero main-stream join wait. Astra High independently mapped the layer2 device tasks and verified this mechanism. Instrumented single-run TPS is not a performance intervention result. Run293 same-profiler immediate-wait control is underway; comparison must use the common predecessor RmsNorm end, first WKV consumer and next AllToAll completion on each rank. Same-prestate typed parity, original A0 and repeated formal E2E remain open. See `evidence/20260926_loop065_gather/run292/`.

### Run293 matched scheduling calibration and next resource question

Astra High independently matched Run292 delayed B and Run293 immediate A1 by device task IDs at the same fixed layer2 shape. Their paired B−A1 median is −17.0µs from common RmsNorm end to first WKV input and −10.125µs to the following AllToAll completion; the latter is earlier in16/16 sampled rank-cycles. The local Q chain did not materially expand, while that AllToAll duration grew3.93µs paired median. The complete Target tail was faster in only8/16 samples, the first-to-next Target interval in4/8 ranks; different acceptance/route/state and rank arrival prevent net Product attribution. H003 changes schedule only, so no Resource compulsory-work estimate or TPS bound changes. It remains a candidate, not a KEEP. Evidence: `evidence/20260926_loop065_gather/run293/astra_matched_review.md`.

Run287 ownership/progress reanalysis exposes a separate architectural question. On its instrumented five cohorts, 2,989/17,952 slot-cycles were parked, while 96-row Target execution continued; active requests staged4.139 tokens/cycle on average. The 16.65% slot fraction does not equal unused arithmetic, bytes or wall time and does not transfer unmeasured to formal Run99. The first Loop066 experiment tests whether each rank's two owner requests can update c4 indexer state from16 complete rows instead of96, initially on private alias-preserving storage at one real layer entry. Full native state/scatter/QLI/attention/lifetime closure and measured 8-rank critical-path effect are required before classing any current traffic as removable or constructing a Hardware/Resource bound. See `evidence/20260926_loop066_owner/design.md`.

### Run294 owner16 output parity does not close the Resource bound

All eight real Target layer2 private owner16 c4 Compressor outputs match full96 by native physical slot; A/A2/A3 controls are stable. But A/B full owner-state-page bytes disagree on seven ranks. Thousands of historical pages and likely shared prefix mappings are in the compared set, so neither a semantic failure nor a safe write elimination is established without first locating changed bytes and physical aliases. No estimate of compulsory state traffic, exposed target-cycle saving or Product TPS is promoted. Run295 will separate B actual prestate-relative changes from A-only writes and identify the first differing physical byte. The pinned historical Performance Knowledge entries are method and risk priors, not capacity measurements.

### Run295 prestate-relative owner state attribution

The eight-rank private c4 layer2 A/A2/B/A3 fixture again has exact owner16 Compressor output slots. Every byte changed by B relative to its same-call prestate has the same value in A on all ranks; all A/B differences over whole owner block-table pages are A-only changes. The first differing byte on each affected rank maps to a nonowner request current native write block that also appears as an older owner table entry. Source c4 state has an 8-token sliding window, so whole historical block tables are too broad a live-state comparison. This is strong diagnostic support for removing Run294 whole-page inequality as direct owner-computation failure, but only the first differing byte is address-attributed and unchanged-value writes are not observed by the difference mask. Run294 and Run295 use different prestates. The exact owner current write/live8 read domains, typed key/scale scatter, QLI/Sparse, persistent lifetime, FULL Graph and exposed E2E remain unclosed. No compulsory traffic, latency floor or TPS bound changes. `dual_bound_v2.json` now records this as a private local relationship with numeric bounds still null. Independent Astra review prefers a same-prestate typed scatter→QLI gate over isolated native timing.

### Run296 typed consumer closure and next scheduling test

Run296 passed on all eight ranks in the real eager Target layer2 private same-prestate fixture: owner16 Compressor output slots, complete owner current-write and conservative live8-read state bytes before/after original typed scatter, INT8 key/FP16 scale owner slots, and native fixed-real-query QLI topk `[12,1,512]` are bit exact to full96. A/A2/A3 self controls, 12×1024 carrier correctness and eight Runtime reports passed; borrowed sources restored. This closes one immediate indexer consumer, not Sparse, SWA/main Compressor, cross-cycle state, Graph capture or a measured compulsory traffic reduction. Numeric Resource, Scheduling-aware and Product bounds stay unknown. Run297 screens same-prestate private A96/B16/A96 update and QLI endpoint event spans; eager Host dispatch may appear in these spans, so even a stable local delta is not a Product TPS projection.

### Run297 eager chain cost screen

Run297 repeated the 8-rank real layer2 private owner16 semantic gate and 21 topk checks/rank, then timed five A96–B16–A2 full-chain triplets per rank after two warmups. B−full rank-median event span is +0.00303 ms at update-end and +0.00016 ms through QLI; B beats both controls in only 9/40 pairs. A/A2 drift is materially larger. This rejects promotion of eager owner16 as a measured speedup. Eager event spans may include Host submission gaps; one private Graph replay screen is the bounded next disambiguation. Hardware/Resource, Scheduling-aware and Product numeric bounds remain null.

### Run298 private Graph and whole producer pivot

Run298 captured the same real layer2 private A96/B16/A2 indexer chain through QLI on all eight ranks. Topk, typed slots and owner state bytes match eager full96. Across 80 paired Graph replays, pooled B−full median is −0.005645 ms, while A/A2 drift median is 0.008240 ms and B wins both controls in 40/80 pairs. This small unstable signal cannot become a complete Target or Product bound. Run287 alias census over 40/40 rank-cohorts shows three layer2-related unique backing stores totaling 2.376 GB/rank, two shared with neighboring layers. The next architecture test is read-only typed view and byte-liveness census of all DSA producer branches and adjacent consumers. Resource/Scheduling/Product numeric bounds remain unknown.

### Run299 typed alias and bounded-read census (2026-09-26)

The original FULL Graph c12 diagnostic passed 12/12×1024 clients and eight Runtime ranks. It registered 196 typed KV leaves/rank; three Draft leaves did not overlap the three layer2-related backing stores in this allocation. Ten selected samples/rank, including first parking 192→193, found no source-derived page intersection between layer2 nonowner current writes and same-layer owner bounded reads. An all-history SWA envelope created 35 page overlaps, while the source 128-token window refined to `(page, slot)` had zero. These are sampled address classifications, not native value/read-from observations or a compulsory traffic deduction. Neighboring Target layers share two stores but their live read/write metadata were not captured. Resource/Hardware, Scheduling-aware and Product numeric bounds remain unknown; the next test is cross-layer typed byte/lifetime closure. See `evidence/20260926_loop067_liveness/run299/`.

## Loop067 cross-layer source-bound result (2026-09-26)

Run301 live runner exited1 only because the post-check expected the wrong layer0/1 Compressor ratio; client 12/12×1024 and all8 FULL Graph Runtime captures completed. Run302 corrected offline validation of unchanged captures passed. Run303 compared all 3094 later-layer Compressor state full-prefix page hits with the SHA-pinned actual container packaged source's c4/c128 conservative history windows; none falls inside (minimum ages 222/1126/1326 tokens versus windows 8/128/128). This reduces a sampled address-alias uncertainty, conditional on packaged source matching the loaded native binary. It neither removes compulsory traffic nor proves persistent owner semantics. The next gate is a single-layer full WKV/SWA, main Compressor, indexer and Sparse same-prestate private A/A/B/A value comparison, then one private Graph critical-path screen if it passes. Historical R21/R28 warn that Compressor overlap can lose to resource competition; see PK-008. Current Formal remains 571.681 tok/s; Resource/Hardware, Scheduling-aware and Product E2E numeric ceilings remain unknown. Evidence: `evidence/20260926_loop067_liveness/run301/`.

## Loop068 full-producer owner private Graph (2026-09-26)

Run305 exact same-prestate all8 layer2 WKV/SWA, main Compressor, indexer/QLI and Sparse owner16 versus full96 gate. Run307/308 private Graph A/B/A screens kept all8 Sparse/QLI and (Run308) every-replay persistent owner cache write domains exact. Paired B faster than both A controls 77/80 then 80/80; diagnostic max-rank paired medians −31.65 and −24.46 µs. This is reproducible *local* producer→Sparse work reduction. It excludes AllGather, local Q, rank rendezvous, full Target/cycle and serving. Astra High supports a single-layer live Target test with communication unchanged. Current Formal 571.681 tok/s and numeric Resource/Hardware, Scheduling-aware and Product bounds remain unknown. See PK-009 and `evidence/20260926_loop068_producer/run307/findings.md`.


## Loop069–070 live owner and dynamic metadata boundary (2026-09-26)

Run310 original / Run311 one-layer live owner16 / Run312 original each completed a 12×1024 diagnostic carrier with 8/8 FULL Graph Runtime in their own services. All 8 ranks of Run311 captured and replayed the owner producer with the expected fixed shape/key, but original A/A across services also changed all 12 reasoning hashes and 2 content hashes. Run311 cannot establish candidate correctness or Product savings; latest-rank ms/cycle was 56.810 versus original 56.545/56.643. Run314 private same-prestate native metadata-in-Graph at the real entry (`delta=0`) passed eager A versus Graph A/B/A2 and fresh metadata, owner typed state, QLI and finite Sparse on all eight ranks. Five ranks failed only in synthetic start shifts when even eager/full Graph A produced NaN Sparse. Those artificial states are not real next cycles. No formal E2E or bound number changes. Evidence: `evidence/20260926_loop069_live_owner/findings.md`, `evidence/20260926_loop070_dynamic_metadata/findings.md`, PK-010/011.

Astra High independently recommends a real adjacent-cycle A→A/B→B persistent-state gate, with complete mutable byte-domain restoration between branches, A/A proof, separate Graph dispatch keys and all-rank first-difference checks. `RuntimeAssets.snapshot_pages(strict=True)` needs backing-byte coverage audit; typed views may omit adjacent scale bytes. Owner16 is not the Resource floor because alternative request ownership/halo schedules might trade compute and communication. Current Formal remains 571.681 tok/s; numerical Hardware/Resource, Scheduling-aware and Product E2E bounds remain unknown. Design: `evidence/20260926_loop071_two_cycle/design.md`.

## Loop071 two-cycle restoration feasibility screen (2026-09-26)

Run315–317 audited the actual Runtime state dependencies before any persistent owner intervention. `RuntimeAssets.snapshot_pages(strict=True)` covers caller-selected Target cache typed rows plus only two registered auxiliary buffers; it does not own Target metadata/RoPE, Draft KV, asynchronous Host count copies/mirrors, FixedDecodeState or serving parking. On the Run299 three selected layer2-related backings, 72/72 sampled backing pages have complete typed-view byte coverage only when both indexer key and scale select the same page. Reusing Run301 original FULL Graph real adjacent-cycle address envelopes, the 14 selected layer0–5 alias sources require a conservative 7.181–7.705MB/rank two-cycle byte-interval union across 40 rank-pairs. This sizes a possible shadow restoration set; it is neither native write traffic nor a compulsory-memory bound. The incomplete mutation registry blocks a valid A→A/B→B result for now. The next experiment must close mutable-state recovery and dispatch, or pivot to a larger measured architecture gap. No change to Current Formal 571.681 tok/s or numeric Resource/Hardware, Scheduling-aware and Product E2E bounds. Evidence: `evidence/20260926_loop071_two_cycle/run315/`, `run316/`, `run317/`, PK-012.

## Loop071→072 architecture priority pivot (2026-09-27)

Astra High independently compared the one-c4-layer owner16 private Graph saving (24–32µs against ~56ms/cycle) with the unclosed two-cycle Target/Draft/Host/metadata restoration transaction. Even a 21-layer linear extrapolation is only a conditional ~0.5–0.7ms/cycle screen; it is not a saving bound or reason to reject broader owner/halo architectures. Loop071 preserves the owner evidence and pivots effort to a larger original-path scheduling question: Run318 validated all-eight identical masks over 1,496 Run287 cycles, with 2,989/17,952 parked slot-cycles and 678 cycles below 12 active. These are slot counts, not removable FLOPs or traffic. The next private full-MoE layer4 A/A/B/A at a real parked prestate tests whether inactive rows can be removed while preserving active output and reducing the complete all-rank block endpoint. Source gate Run319 requires a compact forward context and separate Graph identity; no result yet. Historical R14/R35 are conditional priors in PK-013. Numerical Resource/Hardware, Scheduling-aware and Product E2E bounds stay unknown; Current Formal 571.681 tok/s.

## Loop072 Dual-Bound correction — post-gather routed work (2026-09-27)

Run322 proves the real Target layer4 MoE geometry is global96/local12 under TP8 FlashComm1, with ALLGATHER prepare and unchanged reduce-scatter finalize. Run323 shows 579/678 parked sampled cycles retain a full12 local rank. A fixed-ownership max-row linear **screening proxy** permits only 2.875% aggregate local-row-stage reduction on these masks; it is not a hardware, scheduling or Product ceiling and does not apply to post-gather global active48 routed compaction. The latter may remove routed expert work after an already-required gather without a new collective, but shared experts, communication and other Target work remain. No candidate B or measured critical-path saving exists. Current Formal remains571.681 tok/s; Hardware/Resource, Scheduling-aware and Product numeric bounds remain UNKNOWN. Next measure same-prestate active output and all-rank complete MoE endpoint before any bound update.
