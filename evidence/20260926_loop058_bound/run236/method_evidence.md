# Extreme Performance Model V0 — 2026-09-26

Contract: DeepSeek V4 Flash W4A8; eight 910B3, DP1×TP8, DSpark7; warmed 48×32K→1024 at c12. Correctness and formal repeated E2E remain the gate. This document reports a **conditional capacity model**, not a proved hardware upper bound.

## Executable hierarchy

`scripts/extreme_bound_model.py` consumes Run99's three formal client durations and rank0 per-cohort actual cycles, verifies 48/48×1024, then replays exposed counterfactual savings. `run236/model_replay.json` is the machine-readable output. The cohort mapping is warmup 1–4, measured passes 5–8, 9–12, 13–16; confirm against logs if later artifacts change. Current is exact by construction: 80.188, 86.600, 85.978 s = 612.962, 567.573, 571.681 tok/s. Measured cohort cycles total 1217, 1212, 1206. Rank0 decode-wall sums 69.050, 69.457, 69.040 s; E2E minus those sums is 11.137, 17.143, 16.938 s. **This residual is not all prefill**: it includes admission, nonoverlapped prefill, output drain, other ranks, and mismatched wall windows. It varies more than decode work.

Next version should represent each target/MTP layer and HCCL collective as a node with rank, stream, dependencies and HBM/link resources. For each node, use `max(bytes/attainable HBM bandwidth, FLOPs/attainable compute, link bytes/attainable link service, launch floor)`. Calculate a resource-constrained DAG makespan; collective start waits for all ranks. Per-cycle useful tokens are clipped to each request's 1024 remaining tokens. Four c12 cohorts and their prefill/admission/streaming edges produce E2E. This is a specification for V1, not functionality claimed by the V0 replay script.

## V0 conditional scenarios

All deltas below are **assumed E2E-exposed savings**, holding Run99's actual cycle/acceptance trajectory and 49,152 tokens fixed. They are neither independent opportunities nor validated optimizations. Target and other decode deltas apply per actual cycle; prefill delta applies per cohort. Serving residual remains intact.

| Scenario | Target ms/cycle | Other decode ms/cycle | Prefill s/cohort | Median projected TPS | Formal-trajectory projected range |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current | 0 | 0 | 0 | 571.681 | 567.573–612.962 |
| Engineering conservative | 1 | 0 | 0.05 | 581.186 | see JSON |
| Engineering optimistic | 3 | 0.5 | 0.20 | 607.137 | see JSON |
| Aggressive conservative | 4 | 0.5 | 0.20 | 616.319 | see JSON |
| Aggressive optimistic | 8 | 1.5 | 0.60 | 681.522 | see JSON |

Thus the **illustrative Engineering range is 581–607 tok/s on the median trajectory, Aggressive 616–682 tok/s**. They are scenario envelopes, not an actual upper bound. The current 571.681 is 94–98% of the Engineering scenario and 84–93% of Aggressive. This ratio must not be described as proximity to the true hardware limit. A current valid 612.962 TPS replicate exceeds the engineering median-trajectory endpoint, reflecting workload/serving variation and why one narrow global ceiling is invalid.

## Evidence matrix and comparability

| Dimension | Evidence | Admissible use | Gap / confidence |
| --- | --- | --- | --- |
| Formal Current | Run99 three 48/48 exact-length warm passes; Run93 prior reliability | E2E calibration and final judge | High for measured samples; 3 passes give weak distribution estimate |
| Cohort/cycle/useful tokens | Run99 128 rank/cohort files; Run155 legal boundary cohort | Preserve cycles, clipped tokens, rank wall; Run155 is a separate diagnostic cohort | High for Run99 counts; medium for attribution |
| Low-overhead decode stages | Run98 12-request diagnostic, target 46.560 ms, proposer 6.391 ms, stage-median sum 54.177 ms | Mechanism ranking only; do not add medians to formal E2E | Medium, different cohort |
| Target DAG activity | Run107/143 synchronized profiled 15 rank-cycle windows: device union 50.281, compute 39.932, comm 11.547, overlap 1.292, GMM task sum 9.966 ms | Kernel family and overlap map; not unprofiled stage floor | Medium structure, low capacity |
| HBM and weights | Run146 8.462 GB active packed expert weights/rank/target cycle; Runs148/150 one-card GMM read/active ratios 1.055/1.075 and effective read 1.118/0.995 TB/s | Conditional GMM capacity screen: Run230 7.880–8.373 ms vs profiled 9.966; no 8-rank floor | Medium local, low transferability |
| KV/cache traffic | Run197 c4+c128 compressor scatter task sum 1.123 ms and all scatter 2.344 ms | Required cache writes, not removable time | Byte inventory and bandwidth unknown |
| TP8 communication | Run145 first reduce-scatter mostly peer-arrival wait; remaining 259 task sum ~5.205 ms; Run152 33–41 adjacent BF16/FP32 gather pairs sum ~0.838 ms/window | Do not call 11.547 ms union a network floor or available saving | Low direct service-time confidence; TP8 chain calibration needed |
| Prefill | Run204 20 distinct token/request signatures across 53 calls; Run227 separate-pool 43-layer capture OOM at layer17; Run234 shared-pool source audit | Dynamic metadata/addresses constrain replay; OOM limited to separate pools | Low achievable savings confidence |
| Fusion | Run233 synthetic96 surrogate replay ~0.45 μs/call; Run235 local12 FP32 gate fails | Reject surrogate as product-saving evidence | High for tested surrogate only |
| MRV2 | Pinned community SHA 2bb3f44716f3505d2723a4e5badb10211a6c5589, Run227/234 audit | Design reference; no performance bound | Medium source, no valid product measurement |

FLOPs, full KV bytes, scale/activation HBM bytes, representative per-layer dependency edges, and an eight-rank no-wait collective service curve are still UNKNOWN. Peak FLOPS or quoted HBM bandwidth cannot fill these fields. Original `ACHIEVABLE_BOUND.md` 2.633 GB/card/token selected-weight estimate is not a target-cycle floor: real Run146 route union is ~8.462 GB/rank/cycle. Run230's 1.593–2.086 ms GMM gross spread is conditional across different runs, not exposed E2E savings.

## Largest uncertainty and next discriminator

Run99 decode-wall sums stay near 69 s while formal E2E varies 80.2–86.6 s, so prefill/admission/serving residual needs direct attribution before declaring target the largest E2E gap. At the layer level, non-GMM target work dominates the profiled compute union after GMM; no shape-matched resource floor exists. Minimal next calibration: benchmark the actual TP8 BF16 49,152-element plus FP32 3,072-element all-gather chain without per-call barriers, report all-rank latest completion and variation, then compare to Run152's profiled pair. Do not treat this standalone chain as product savings. After that, a legal low-overhead cohort boundary capture should attribute non-decode residual on the formal schedule if it still changes architecture priority. Next architecture choice should be based on a whole-cohort predicted seconds saved under preserved routes and correctness.

Independent reviewer request: `gpt-6-astra` / `high`; actual backend variant was not independently exposed. The reviewer performed read-only SSH inspection; no hardware run. Reviewer confirmed the hierarchy and challenged the Run99 612.962 replicate, route-byte floor, collective arrival wait, and interpreting Run116's non-GMM remainder as a necessary-work lower bound.

## Calibration results after V0 freeze

Run237 executed `torchrun` on all eight 910B3 devices inside the existing dsv4ab container (exit 0). Forty back-to-back BF16[49152] + FP32[3072] all-gather pairs, thirty timed repeats after eight warmups, no per-call barrier, yielded latest-rank chain median 21.420 ms, range 20.913–23.647 ms, or 0.53549 ms/pair. Eight rank medians span 21.294–21.407 ms. Run152 0.83810 ms is the total of 33–41 adjacent product graph pairs per window, ~0.02288 ms/pair by Run240 audit. The eager Run237 API path is ~23.4× slower per pair and its payload interpretation may differ. It cannot calibrate product communication capacity or savings. The microbenchmark also excludes reduce-scatter, full product sequencing and model-compute contention.

Run238 reuses Run99 formal client wave timestamps and saved rank0 cohort wall without new service. Per-wave client window minus rank0 decode wall is 2.545–3.148 s in the 612.962 TPS pass, 3.848–4.450 s in the 567.573 pass, and 3.996–4.521 s in the 571.681 pass. Maximum eight-rank decode-wall spread per cohort is <=0.019 s. Much formal pass variation lies outside runtime-owned decode wall; the difference does not isolate prefill, admission, output publication or overlap. Next calibration must timestamp those boundaries within one legal formal-schedule cohort or pass at low overhead.

## Loop059 Run240 communication comparability correction (2026-09-26)

Run152's 0.83810ms is the **sum across 33–41 adjacent BF16/FP32 all-gather pairs in a target window**, median ~0.02288ms per product HCCL task pair. Run237 measured 0.53549ms per pair through eager `torch.distributed.all_gather_into_tensor`, about 23.4× slower. Graph replay, host dispatch, surrounding compute and even the profiler count-to-buffer interpretation differ. Run237 remains a valid measurement of that standalone eager API path, but is **invalid as a calibration of product TP8 communication capacity or removable E2E time**. The previous per-pair reading of Run152 was corrected in the V0 evidence matrix, ACHIEVABLE_BOUND and PERFORMANCE_MAP. Evidence: `evidence/20260926_loop059_boundary/run240/analysis.json`.
