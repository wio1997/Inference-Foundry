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

Run246 legal instrumented capture was exported and checked in Run247: all 80 target rank-cycle windows passed kernel-count and positive-counter gates, including 16 latest windows across all eight ranks. Median AIC+AIV reads are 9.244GB GMM, 3.465GB quant matmul, 1.530GB compressor, 1.062GB sparse attention and 3.618GB other per rank-cycle; 18.965GB read and 2.380GB written overall. GMM read is 1.092× the Run146 active packed weight estimate, so gross weight traffic amplification is modest in the real graph. AIC-only GMM read is 8.990GB; omitting vector counters would understate traffic and falsely report zero scatter writes. HCCL link bytes are not present. These task counters may include repeat reads, and the synchronized profiler changes timing. The median 54.796ms target scope is not a formal E2E bound. No unique compulsory-byte floor, cross-rank attainable bandwidth or full critical path has yet been established. Details: `evidence/20260926_loop060_resource/run247/findings.md`.
