# Performance Map V0

Updated: 2026-09-20 15:57 UTC. Current evidence: 3 warm-cache DP1/TP8 end-to-end runs; no current kernel/HCCL timeline yet.

| Component | Observed current runtime | Evidence and confidence | Remaining Gap / next discrimination |
|---|---:|---|---|
| E2E warm mixed workload, 48×32K→1024, c12 | output TPS 543.65 median (range 523.15–545.85); TTFT mean 1.33 s median; TPOT mean 19.73 ms median | `evidence/20260920_baseline/bench48_[123].json`, high confidence in order of magnitude, ~4.3% TPS range | Separate cold and warm phases; quantify repeat noise before accepting small gains |
| Prefix reuse | 99.75% hit on measured passes | before/after Prometheus counters; high | This measured workload is decode-heavy after warmup; first-pass prefill must be measured separately |
| DSpark speculation | accepted 2.94 tokens/draft, 42.0% acceptance (runs 1–2); 41.2% run 3 | `analysis.json`, Prometheus; high for counters | Measure draft and target spans/bytes; compare speculation against a correctness-preserving control only if predicted gain is large |
| Device occupancy | AICore median 77%, p90 83%; HBM occupancy ~60.3 GB/card | `npu_samples.txt`, 5 s snapshots during runs 1–2; medium | Identify idle/serial gaps, memory or communication stalls; utilization alone is not root cause |
| Cold 32K prefill, c1, 128 output | TTFT mean 2.663 / 2.627 s for two disjoint four-prompt groups; second had 0/131404 cache hits | `evidence/20260920_diagnostic/cold4*.json` and metrics; high confidence for E2E TTFT | Chunk/kernels, KV/indexer copies and critical path still unknown |
| Decode kernels, graph, host | not yet measured on current topology | historical R08 is DP2/TP4 c1 only | Draft/target split, launch and D2H/sync migration |
| TP8 HCCL | not yet measured | historical R39 shows contention in DP2/TP4, not transferable | Actual TP8 communication duration, bytes, overlap |
| W4A8 weight traffic | ~21.1 GB selected weights per token model-wide before batch reuse | `weight_inventory.json`; approximate | Actual rank-local selected experts, cache reuse, draft pass count |

Do not infer Compute Gap versus Framework Gap from AICore utilization alone. The next diagnostic must explain exposed time at the same workload. Historical R23 Super Kernel regression and R39 R29 regression make those paths lower-priority unless current trace contradicts prior mechanisms.

System msprof on device 0 during four further cold prompts succeeded; exported HBM/HCCS/AICore CSV under `evidence/20260920_diagnostic/system_summary/`. It does not contain a per-op timeline and its 30 s window includes idle time, so do not infer the bottleneck from average HBM/link numbers. The 172 MB raw trace remains at the indexed `raw_profile/` path.

## Update 2026-09-20 16:30 UTC — current TP0 application traces

The earlier rows marked unmeasured are superseded for TP0 short diagnostic windows by this table. The official 48×1024 c12 workload remains unprofiled.

| Window | E2E diagnostic | TP0 event span | Compute/copy union | HCCL union | All-op union | Key remaining question |
|---|---:|---:|---:|---:|---:|---|
| Four cold distinct 32K→128 c1 | 18.915 s, 4/4 | 18.583 s | 10.829 s | 6.386 s | 16.510 s | Which prefill chunks, target/draft and other ranks are critical? |
| Four exact-repeat warm 128-token c4 | 3.633 s, 4/4 | 3.401 s | 1.688 s | 1.165 s | 2.799 s | Can FlashComm1 reduce-scatter/all-gather be improved without regressing correctness or E2E? |

Warm HCCL types: reduce-scatter 0.577 s, all-gather 0.356 s, all-to-all 0.231 s. Cold types: 3.588 / 1.934 / 0.864 s. Warm compute and communication union overlap by only ~0.054 s. Warm 0.602 s of the TP0 window has no recorded device op, spread across many small gaps; only five gaps exceed 5 ms. This is an opportunity map, not a proof that all these times are removable. One rank and profiler overhead limit causal attribution.

`op_summary.csv` includes duplicate descriptive HCCL rows with null Task ID; `scripts/analyze_profile.py` excludes them. Evidence: `evidence/20260920_diagnostic/app_profile_cold2/summary.json`, `app_profile_warm4/summary.json`, `app_profile_index.json`, and `gaps.json` in each profile directory. Source links in the bound vllm-ascend tree identify FlashComm1 sequence-parallel reduction and MTP all-gather at `models/deepseek_v4.py:1144`.

## Update 2026-09-20 16:39 UTC — configuration constraint

FlashComm1-off alone is not runnable with the frozen `enable_dsa_cp=true`: worker initialization raises `ValueError: DSA CP requires SP`. This is a framework dependency, not a measured speed gap. The next comparison changes the coupled SP/DSA-CP path as one coherent intervention; attribute any outcome to the pair, not to FlashComm1 in isolation. No Performance Map timing changed in Loop 004.

## Update 2026-09-20 16:55 UTC — correctness gate limitation

The coupled SP/DSA-CP-off candidate is runnable, but no performance data yet. Exact 128-token reasoning text is nondeterministic even on the same candidate for 4/4 long prompts. The earlier proposed golden hash equality must not be used to label this candidate incorrect. Functional output checks and later numerical equivalence are separate from E2E performance measurement.

## Update 2026-09-20 17:06 UTC — coupled path A/B

SP/DSA-CP both off is a runnable path, but three full warmed passes yield median 534.88 output tok/s versus baseline 543.65 (-1.61%, within 4.3% baseline spread). TTFT worsens ~7.8%, TPOT ~0.7%. The short TP0 trace showed ~1.165 s HCCL union, but eliminating/changing that path together with DSA CP did not produce E2E gain. This does not isolate FlashComm1 from DSA CP. Next highest-value discrimination is DSA CP off alone while SP remains on; after that, measure draft/target/host boundaries rather than pursuing HCCL activity totals in isolation.

## Update 2026-09-20 17:27 UTC — isolated DSA CP A/B

With FlashComm1 on and only DSA CP off, median full warmed output TPS is 518.10 vs 543.65 baseline (-4.70%); TTFT +15.1%, TPOT +3.8%. Thus disabling DSA CP is a negative E2E move on this topology. Coupled SP/DSA-CP off was only -1.61% and within noise, so a FlashComm1-only effect cannot be assigned because that configuration is invalid. Current largest unresolved opportunities are rank-0 short warm device idle ~17.7% (distributed sub-ms gaps), communication/compute critical path and cold-prefill ScatterNdUpdateSk/Compressor/HCCL spans. Profile target/draft/host scopes and rank synchronization before choosing a patch.

A provisional cold-trace split by HTTP TTFT, aligned to the device trace with an inferred ~0.13 s offset, puts 1.152 s ScatterNdUpdateSk, 0.848 s Compressor and 4.143 s HCCL summed task time before first tokens across four requests. `evidence/20260920_diagnostic/app_profile_cold2/cold_phase_approx.json` records the method. Boundaries and overlap are approximate; these are candidate selectors, not critical-path contributions.

## Update 2026-09-20 17:59 UTC — TP0 CPU and device phase map

The baseline-flag torch-NPU scope profile contains separate HTTP warm and cold windows. `scripts/analyze_scope_device.py` clips device intervals to those windows and excludes duplicate null-task HCCL rows.

| Window | Wall | TP0 device busy union | HCCL union | Compute/copy union | CPU scope sums |
|---|---:|---:|---:|---:|---|
| Warm exact-repeat 4×128 c4 | 5.473 s | 4.094 s | 2.269 s | 1.876 s | prepare 1.320 s; draft 2.237 s; forward 0.969 s |
| Cold distinct 2×32K→128 c1 | 11.178 s | 9.481 s | 4.640 s | 5.203 s | prepare 2.384 s; draft 3.405 s; forward 4.293 s |

Warm `prepare input` contained 1364 `aten::item` calls in 47 scopes, 0.436 s nested total. Most scopes contain 29 calls. The representative median prepare scope had ~9.0 ms of `item` work within 26.7 ms total. These spans can include device synchronization; source stack and exposed time are unproven. Warm rank-local device idle by interval difference is 1.379 s, but includes request boundaries and profiler overhead. HCCL 2.269 s includes waiting; the earlier configuration toggles showed no benefit. Source: `evidence/20260920_scope_profile/run1/` and indexed raw TP0 trace. The official 48×1024 c12 workload remains unprofiled.

## Update 2026-09-20 18:17 UTC — stack collection invalid

Stack-enabled torch-NPU profiling did not produce usable host call stacks: `/stop_profile` caused worker segfaults and offline export omitted FRAMEWORK operator events. The Loop 008 timing map remains the latest valid diagnostic; no new timing or performance claim is made. The next discriminator is narrowly scoped source instrumentation around `prepare input` to identify CPU versus NPU `.item()` and actual exposed delay.

## Update 2026-09-20 18:35 UTC — localized host synchronization

Loop 010 used no torch profiler. A temporary wrapper around Python `Tensor.item` only during `prepare input` found the dominant call at DSA CP QLI metadata `dsa_cp.py:1008`, taking an NPU scalar from `seq_lens_q.max()`. Each warm step calls it once per rank. The same function has already calculated CPU `max_local_query_len` and `max_local_seq_lens`; the CPU and NPU local arrays use the same formula, but actual runtime equality still needs checking across async decode/prefill. Snapshot differences are approximate phase attribution: TP0 line 1008 210 ms/54 calls in measured warm set; TP1 209 ms, ranks 2–7 5–37 ms. Cold subsequent set: TP0 1268 ms/70 calls, TP1 1171 ms, ranks 2–7 648–783 ms. Rank skew suggests synchronization and preceding device work, so these durations cannot be simply subtracted from wall time. Line 1009 follows the sync and is ~0.06 ms/call. Evidence: `evidence/20260920_item_trace/run1/item_summary.json` and raw tracer lines.

## Live update 2026-09-20 18:58 UTC — QLI candidate pending cold pairing

Replacing QLI NPU scalar maxima with CPU local maxima passed >=192 runtime parity checks per rank over warm/cold requests and the functional gate. Full mixed 48×32K→1024 c12 warmed median output TPS is 540.40 vs baseline543.65 (-0.60%, no gain). Two candidate cold 4×32K→128 c1 groups at new offsets24 and28 have mean TTFT 2.338/2.360 s. The previous baseline groups used different prompts and had 2.663/2.627 s. Same-offset baseline is required before attributing this apparent cold difference. Do not convert it into a component Gap or bound yet.

## Update 2026-09-20 19:20 UTC — paired QLI result

The QLI CPU-max change was verified numerically on warm/cold metadata (>=192 checks per rank), then benchmarked without verification overhead. Full warmed mixed output TPS median540.40 vs paired original537.60 (+0.52%, below noise); candidate mean TTFT median1413.44 vs original1262.74 ms, so the all-step path lacks a safe mixed benefit. Exact same cold prompts24-31 gave candidate TTFT mean2349.06 vs original2633.90 ms (-10.81%); all eight improved205–343 ms and original prefix hits were0/262808 queried tokens. This localizes a useful cold-prefill opportunity. Next keep original decode path and apply CPU maxima only when DSA CP builder reports prefill; then retest mixed and cold.


## Update 2026-09-20 19:48 UTC — Loop 012 prefill-only QLI KEEP

| Case | Original source | Prefill-only candidate | Interpretation |
|---|---:|---:|---|
| Cold 16 distinct 32851-token prompts, c1→128, mean TTFT | 2626.03 ms | 2362.89 ms | -263.14 ms (-10.02%); all 16 faster, no prefix hits |
| Warmed mixed 48×32K→1024 c12, median output TPS, 3 passes | 537.60 | 547.55 | +1.85%, inside the 4.3% baseline spread |
| Same mixed passes, median request-mean TTFT | 1262.74 ms | 1143.58 ms | No observed TTFT regression |
| Same mixed passes, median request-mean TPOT | 19.503 ms | 19.252 ms | Within noise |

The exact same 16 cold prompts were paired; candidate per-prompt deltas range -355.75 to -202.61 ms. This change removes a prefill QLI NPU scalar read using CPU maxima already computed in the same builder, with prior runtime equality checks on all eight ranks. Pure decode is unchanged. The main unsolved performance gap is warm mixed output throughput; existing HCCL and device profiles show a large communication envelope but no proven removable component. Next loop must localize an exposed decode critical path before changing code.

## Live update 2026-09-20 19:56 UTC — warm decode diagnostic pending

The existing 4×128 c4 short torch profile placed 2.237 s of 5.473 s wall in TP0 draft_token host scope and 2.269 s in HCCL union, with overlap and profiler overhead. These are not exposed c12 critical-path costs. Loop013 is collecting a warm 12×512 c12 trace on the kept prefill-only QLI source to distinguish device activity and gaps. Dynamic msprof attach failed twice with no valid pid values; these attempts produced no timing claim.


## Update 2026-09-20 20:24 UTC — c12 decode window across eight ranks

After 48×128 full prefix warmup, a 12×512 c12 sample succeeded 12/12 under torch-NPU profiler. The request window was16.459 s, while the bench itself reported15.847 s and387.7 output tok/s; profiler overhead and changed request length make this TPS non-comparable with the official 48×1024 baseline.

| Rank/window metric | TP0 | Eight-rank range |
|---|---:|---:|
| Device busy union | 13.038 s | 9.672–13.317 s |
| Compute/copy union | 7.736 s | 7.631–7.745 s |
| HCCL union | 5.510 s | 2.236–5.843 s |
| Device inactive within16.459 s window | 3.421 s | 3.142–6.787 s |

TP0 HCCL duration by type: reduce-scatter3.117 s, all-gather1.677 s, all-to-all0.715 s, with task waiting and overlap; these sums are not independent E2E costs. TP0 top compute/copy task types include GroupedMatmulSwigluQuantV2 1.034 s, QuantBatchMatmulV3 0.882 s, GroupedMatmul0.584 s, Compressor0.568 s. A TP0 host trace scanned5.79M events: prepare-input171 scopes sum3.729 s (median21.42 ms); draft170 scopes sum9.036 s (median52.89 ms); forward170 scopes sum2.244 s but median1.90 ms and occasional prefill outliers. Typical draft nested scopes: three moe_forward_shared total15.77 ms and three dsa_forward total10.09 ms. Across the full trace, 682 MoE calls have host self-time1.397 s and 682 DSA calls0.964 s. These are candidate pools, not removable times. Next resolve which host self operations or cross-rank waits are on the unprofiled decode critical path.

## Live update 2026-09-20 20:30 UTC — decode host attribution still open

Loop014 source audit shows prepare-input host self median9.57 ms among170 c12 steps, while a typical nested NPU item call was0.70 ms. The broad prepare scope includes _update_states, _prepare_inputs, Mamba preprocessing, attention metadata and _preprocess. The current trace cannot assign that self time to a single removable function; a low-overhead stage timer is the next discriminator. No change to measured E2E performance.

## Live checkpoint 2026-09-20 20:33 UTC — Loop014 stage tracer

Loop014 env-gated no-profiler prepare-input stage timestamps pending. They will split the prior 21.42 ms median c12 prepare scope into state update, input assembly, Mamba/dispatch, attention metadata and preprocess. No new measured gap yet.

## Update 2026-09-20 20:47 UTC — Loop014 first stage trace

Loop014 no-profiler tracer, 157 pure decode steps per rank in successful c12 sample: TP0 median prepare19.55ms; state update0.78, input assembly4.81, dispatch/Mamba0.19, compression plus attention metadata12.99, preprocess0.31ms. Eight-rank dominant substage median10.80-14.17ms. This localizes an investigation target. It does not establish removable time or official 48x1024 throughput gain. Short diagnostic output TPS471.14 cannot be compared with frozen baseline because request count/output length differ and tracing is active.

## Live checkpoint 2026-09-20 20:51 UTC — builder trace loading

Loop014 second diagnostic will split the 10.80-14.17ms rank-median compression/attention stage into DCP setup, per-attention-builder calls and residual work. Prior first-stage trace remains the current measured map until this succeeds.

## Update 2026-09-20 21:08 UTC — Loop014 PIVOT

Loop014 builder trace on 143 pure-decode steps/rank: eight DSA-CP metadata builders per step. TP0 median metadata total15.165ms, builder calls14.025ms, DCP0.002ms, residual1.117ms. First builder7.860ms, seven others0.725-1.096ms. The builder uses common_ratio_to_sas_metadata and cached local metadata across groups. Inclusive host duration is not a directly removable gap; no mixed TPS optimization was kept. Next prioritizes cold prefill device critical path.

## Live checkpoint 2026-09-20 21:13 UTC — Loop015 cold kernel candidates

Prior original-source cold msprof shape audit across four 32K->128 c1 requests found dominant ScatterNdUpdateSk [8096,2] index shape: 736 calls, 760.8ms summed task time, median1017us. Compressor ratio4 shape [8096,4096]: 336 calls, 505.7ms summed task time, median1503us. The totals span prefill and decode and overlap other ranks/operations. An on-request slot uniqueness probe is running; no removable time or throughput improvement established. See evidence/20260920_loop015_cold_kernels/shape_summary.json.

## Update 2026-09-20 21:29 UTC — Loop015 scatter screen

Eight rank captures of one real long-prefill SK call each: 8096/8096 valid and unique slot pairs, 0 duplicates. The representative TP0 index stream is block-local ordered but globally not sorted. In a single-NPU 25-call isolated screen using captured indices and matching cache/update shapes, SK device median1.427ms and V2 device median2.415ms, ratio1.692; output bit-equal. V2 is rejected. The old-profile SK shape median task1.017ms was from loaded full-service execution; different measurement settings mean it should not be directly compared to the isolated1.427ms. No E2E improvement or established critical-path saving.

## Live checkpoint 2026-09-20 21:43 UTC — Loop016 SWA index_copy screen

The 8096-row shape is SWA prefill scatter. On captured unique indices with matching cache/update shapes, isolated contiguous SK1.428ms versus precomputed-flat index_copy0.510ms device median (25 calls); built-in npu_scatter_nd_update_1.402ms (25 calls). Simulated page-interleaved stride32768 with flatten computed each call: SK1.528ms versus index_copy0.694ms (20 calls), bit-exact logical cache. This suggests a sizeable candidate on cold c1 prefill but excludes model overlap and true cache stride. Candidate flag-gated service A/B and exact output parity pending.

## Live checkpoint 2026-09-20 21:47 UTC — Loop016 A/B pending

Full-service candidate parity and paired cold TTFT are now queued on API PID809483. No new measured E2E result yet; isolated screen remains the map.

## Update 2026-09-20 22:08 UTC — Loop016 cold no-flag baseline

Same current source with candidate flag absent, fresh service, no-repeat dataset offsets8–15 c1, all eight 32851→128 succeeded: mean TTFT2367.81ms, individual 2334.5/2326.9/2371.8/2404.3/2332.2/2375.0/2396.1/2401.6ms. This is the exact prompt set for revised candidate run. The first attempted candidate did not activate (0/8 rank traces), so no E2E delta yet.

## Live checkpoint 2026-09-20 22:12 UTC — revised A/B loading

API PID815939 and runner PID2226077 are loading. No candidate E2E result yet.

## Update 2026-09-20 22:25 UTC — DSA-CP dispatch identified

A/B2 did not invoke dsa_v1 fast path under the enabled DSA-CP backend. Actual [8096,2] SWA scatter caller is dsa_cp.py:1522. Previous isolated kernel screens remain valid for the operator shape; no new full-service candidate timing. Third patch targets the active caller, pending A/B3.

## Live checkpoint 2026-09-20 22:29 UTC — DSA-CP A/B3 loading

Third service API PID822276 and runner PID2242486 loading; no candidate E2E result yet.


## Update 2026-09-20 22:39 UTC — Loop016 E2E rejection

Cold c1 32851→128, same prompts offsets8–15, eight requests, no prefix hits: no-flag baseline mean TTFT2367.81ms; active DSA-CP index_copy candidate2438.00ms; delta +70.19ms / +2.9642% (worse), 0/8 improved. All eight ranks recorded fast-path activation, cache stride=(16384,512,512,1). Candidate not kept. Isolated operator index_copy0.510ms versus SK1.428ms was not predictive of full-service TTFT; overlap, launch and host effects remain unmeasured. Next map target: Compressor ratio4 cold prefill shape[8096,4096], old msprof 336 calls/four full requests, median task1.503ms; attribute which calls are prefill and on critical path before claiming opportunity.


## Live checkpoint 2026-09-20 22:46 UTC — Loop017 Compressor clustering

Original-source cold profile four-request time clusters: Compressor [8096,4096;1024,4096] 84 calls/request, median1.503ms/task, 126.3–126.6ms summed device task/request; [8096,4096;512,4096] 80 calls/request, ~40.0ms sum; [8096,4096;256,4096] 84 calls/request, ~29.8ms sum. Source CSV and exact per-cluster stats: evidence/20260920_loop017_compressor/profile_clusters.json. Ratios and totals cannot be added directly to TTFT because of overlapped streams and downstream dependencies. E2E cold baseline remains Loop016 no-flag2367.81ms on offsets8–15; kept QLI Loop012 vs original cold comparison remains authoritative.


## Live checkpoint 2026-09-20 22:50 UTC — serial Compressor chain

All336 main-shape ratio4 Compressor tasks in frozen four-cold-request profile are followed on the same device stream by ScatterNdUpdateSk then SparseAttnSharedkv (336/336 each). Median gap to next task2.87us; median Compressor-start to SparseAttn-start1.84475ms. This is a per-layer serial chain, materially stronger than aggregate op time alone. It does not quantify cross-stream overlap or guaranteed TTFT saving. See evidence/20260920_loop017_compressor/stream_order.json.


## Live checkpoint 2026-09-20 23:59 UTC — shape-matched Compressor screen

Single-NPU synthetic shape-matched ratio4/coff2 Compressor baseline: device median1.630ms across25 event-timed calls, min1.499ms; old full-service cold profile median task1.503ms. Synthetic values and isolated scheduling differ, so only compare future isolated candidates to this screen. The operator produced finite [2025,512] output on scaled inputs; exact model numerical parity untested. Source arch32 tiling uses mBaseSize128 for coff2, nSize2, 20 AIC blocks; large8096-token case retains base M tile and runs repeated cross-core sync.


## Live checkpoint 2026-09-21 00:06 UTC — candidate pending

Stock Compressor one-NPU fixed-seed reference screen has device median1.514ms on25 calls, finite output, exact shape/config from cold profile. Earlier independent synthetic screen median1.630ms shows isolated-run variation. Do not compare candidate to cross-run values without the saved fixed-seed reference. mBase256 tiling candidate is compiling; no measured delta.


## Live checkpoint 2026-09-21 00:22 UTC

No new measured performance. Isolated mBase256 package build ongoing; watchdog will run numerical and single-NPU comparison on completion. Preserve stock reference1.514ms median as only current same-seed micro baseline.


## Update 2026-09-21 01:10 UTC — Loop017 rejected; c12 communication audit

Isolated stock Compressor fixed-seed reference 1.51436ms/device-call median. The mBase256 candidate has no completed screen, no numerical comparison and no E2E result; it is excluded from performance comparisons. Eight-minute stall is an operational failure of this isolated implementation, with cause unidentified. Cold Compressor gross ~126.4ms/request envelope (~5.3% of 2367.81ms TTFT) remains a screening estimate, not a realized gain.

Saved c12 8-rank communication.json has identical per-rank counts: allGather25160, alltoall7820, reduceScatter15980. Apparent HCCL elapsed differs by rank, but each entry has Elapse Time=Idle Time and Transit/Wait=0. Thus these fields cannot establish wire time, critical path, or removable HCCL duration. Next gap is service-level decode rank arrival/collective overlap attribution; no patch chosen yet. Evidence: evidence/20260921_decode_comm_audit/collective_time_components.json.


## Update 2026-09-21 01:15 UTC — collective timestamps confounded

Loop018 aligned48,960 identical collective types across8 ranks. Raw start/end rank spread medians0.6105/0.583ms; rank6 earliest in48,829 starts. Median-end offset sensitivity estimates rank6 -579.172us, with corrected start/end spread medians0.2175/0.206ms. Since corrected end spread is similar to corrected start spread and independent device clock synchronization is absent, rank arrival skew cannot be promoted to a critical-path savings claim. HCCL report is entirely Idle Time. Next candidate search returns to host metadata, previously measured at15.165ms of19.55ms pure decode prepare on TP0, eight DSA-CP builders14.025ms; these are inclusive spans and need active-callsite, no-profiler A/B verification. Evidence evidence/20260921_decode_comm_audit/collective_arrival_skew.json and collective_clock_sensitivity.json.


## Live checkpoint 2026-09-21 01:59 UTC — first DSA-CP builder dominates

No-profiler sample 12×32K→512 c12, 174 pure-decode steps/rank: first active DSA-CP builder median total3.61–6.50ms/rank; build_req_metadata2.74–5.74ms, shared setup0.57–0.66ms, slot format~0.18–0.21ms. Other seven builder median total0.67–0.85ms. This is host inclusive timing with rank variation and trace I/O; no E2E saving claim. The first builder's request-metadata path is next attribution target. Evidence evidence/20260921_loop019_builder_stage/stage_summary.json.


## Live checkpoint 2026-09-21 02:13 UTC — first-builder QLI stage

Loop019 no-profiler first DSA-CP build_req_metadata subphase audit (169–170 pure decode steps/rank): QLI median1.735–4.306ms/rank, device-local0.477–0.548ms, CPU-local0.255–0.292ms, SAS0.422–0.485ms. Rank variation is substantial; QLI may include .item() device synchronization and metadata op cost. These are inclusive host scopes, not additive to 19.55ms prepare or E2E savings. Prior Loop011 all-step CPU maxima had no robust mixed TPS win; next isolate QLI suboperations and graph/host overlap before choosing a patch. Evidence evidence/20260921_loop019_builder_req/request_subphases_summary.json.


## Update 2026-09-21 03:11 UTC — QLI host sync isolated, not priority

Loop019 QLI subphase (155–156 uncached pure-decode calls/rank): first q.max().item median0.402–3.716ms/rank, second k.max().item0.110–0.135ms, metadata op+clones0.356–0.411ms. Rank variance suggests synchronization waits on prior device work; direct replacement with CPU maxima already failed mixed E2E gate in Loop011 (+0.52% TPS within noise, TTFT worse). Exclude these host spans from available savings. Re-rank next warm gap toward DSpark proposer/target work: old c12 TP0 host draft scopes9.036s in16.459s profile window, but overlap and device attribution unresolved.


## Live checkpoint 2026-09-21 03:20 UTC — DSpark co-occurrence

Saved profiled c12 TP0: draft_token host median52.894ms across170 scopes; temporally coincident device union median41.860ms, compute37.641ms, HCCL5.272ms. Thus treating all draft host time as framework waste is invalid. Async target work may overlap draft scope; next no-profiler proposer stage trace will separate metadata preparation from model run on host. Evidence evidence/20260921_dspark_audit/host_device_scope_overlap_tp0.json.


## Live checkpoint 2026-09-21 03:36 UTC — DSpark stage map

No-profiler warm c12×512 sample,160 pure decode proposer calls/rank: _propose median36.11–41.38ms; eager run_draft28.43–32.69ms (~78–80% of proposer wall), step0 all-group attention metadata5.67–6.41ms (~15–16%), set_inputs1.47–1.66ms. Saved profiled 10-second acceptance buckets: mean3.54–3.64 advanced tokens per7 drafted, per-position acceptance declines to~8% at seventh. Scope/device temporal co-occurrence is not ownership; next isolate model-owned device work and evaluate graph feasibility or metadata repetition.


## Loop020 causal DSpark map — 2026-09-21 04:09 UTC

- TP0 saved trace: 170 `draft_token` scopes, host median 52.894 ms.
- Trace-flow integrity: 183,679 async finishes map to same-stream device X starts with 0.0 us median/p99 delta; 121,038 launches originate within same-thread draft scopes.
- Causal device union clipped to draft scope: median 51.661 ms (p90 52.713 ms); including tasks extending after scope: median 63.259 ms. This is device-active time, not an additive saving.
- Highest launch counts in draft include EVENT_WAIT 18,795, IndexCheck 5,566, Index 5,095, MEMCPY_ASYNC 4,352, Cast 3,894 and Pad/MemSet 3,718 each. Counts alone do not prove critical-path savings.
- Current largest verifiable gap: eager draft device fragmentation and low acceptance efficiency. ACLGraph is explicitly disabled/unsupported on this DSpark path, so Loop021 must isolate a bounded source-level chain before implementation.


## Loop021 event-aware draft map — 2026-09-21 05:10 UTC

- All 121,038 draft-owned async flows exact-match device tasks; no unmatched flow or task.
- Repeated leaf chains are individually small: Index clipped union median 0.545 ms when present; Pad 0.337 ms; InplaceCopy 0.112 ms. They do not justify a standalone service patch against 4.3% baseline TPS spread.
- The prior 51.661 ms causal task union is dominated by device `EVENT_WAIT`. Outer draft wait union is 51.169 ms median across 170 scopes; MoE semantic-scope wait union is 26.595 ms. These are stream dependency spans, not AICore occupancy.
- Highest-value unknown moves to MoE event topology: whether producer lag or stream ordering stalls the critical compute stream, versus an intended wait on an auxiliary stream hidden by useful work.


## Loop022 event dependency resolution — 2026-09-21 06:10 UTC

- Outer draft `EVENT_WAIT`: streams42/43, one each per step, 53.104/54.095 ms median, always followed immediately by `MEMCPY_ASYNC`; auxiliary copy dependency, not main-stream serialization.
- MoE shared stream36 has four waits per shared-expert call. The long boundary wait (qmatmul → next layer dynamic quant) is 25.734 ms median and represents the shared stream idling until the next layer input is produced.
- Within a shared-expert call, wait medians are 0.191 ms before activation, 0.051 ms before down projection and 0.011 ms before gate projection. Main stream47 final wait is about 0.00002 ms. No material removable MoE synchronization gap is established.
- Next highest verifiable system lever: speculative length. k7 advances only 3.54–3.64 tokens, so Loop023 benchmarks whether k5 reduces rejected-token work.


## Loop023 speculative-length constraint — 2026-09-21 07:07 UTC

- k5 is unrunnable at TP8 in the current graph/sequence-parallel path: `(k+1)=6` conflicts with TP divisor8 during worker initialization. No benchmark data exists.
- Model draft block requires k>=5, so k7 is the minimum valid value satisfying `(k+1) % 8 == 0`.
- vLLM reserves `max_num_new_slots_for_drafting * max_num_seqs = 6*16 = 96`; current max batched8192 therefore limits scheduled tokens to8096 and warns of suboptimal performance.
- Loop024 changes only max batched tokens to8288 to restore scheduled capacity8192 and measures the end-to-end effect.


## Loop024 scheduler capacity screen — 2026-09-21 08:14 UTC

- Raising max batched8192→8288 removes the8096 scheduled-token warning and passes correctness.
- Bounded c12 result:472.871 tok/s, TTFT mean2007.49ms, TPOT mean17.300ms.
- Relative to six prior k7 diagnostic screens: +2.77% versus median, -3.83% versus closest Loop020 screen; within4.3% noise. This capacity gap is not a verified E2E lever.
- Next decisive system question: net k7 DSpark value versus target-only decode under matched settings.

## Loop025 DSpark net value — 2026-09-21 09:09 UTC

- Target-only matched screen:208.047 tok/s, TPOT54.333ms, TTFT1644.40ms.
- k7 DSpark matched screen:491.698 tok/s, TPOT17.554ms, TTFT1981.90ms.
- DSpark provides2.363x output TPS and67.69% lower TPOT, while short-screen TTFT rises20.52%. The speculative path is decisively valuable for decode.
- Draft model source contains three sequential DSpark layers; prior profiled nested MoE+DSA host scopes total about25.86ms across them. Loop026 tests whether removing only the middle layer improves cost/advanced token.


## Loop026 trained-layer necessity — 2026-09-21 10:09 UTC

- Skip-middle proposer model time:31.106→23.887ms rank median (-23.21%); total proposer39.545→32.298ms.
- Acceptance collapses to0.363 accepted drafts and1.363 advanced tokens/cycle, below3.15 estimated break-even. Positions0–3 acceptance becomes30.54%,4.82%,0.80%,0.16%; positions4–6 zero.
- E2E:491.698→217.092 tok/s (-55.85%); TPOT17.554→47.972ms (+173.29%).
- Conclusion: full three-layer learned computation is necessary. Remaining proposer opportunity must preserve exact semantics; fixed-shape eager execution is now the primary framework-level candidate.

## Loop027 graph-path and runtime-boundary update — 2026-09-21 11:18 UTC

- Legacy Ascend DSpark explicitly forces eager execution. The available v2 DSpark implementation contains an Ascend graph manager, but the exact DeepSeek V4 checkpoint resolves its draft architecture as `DeepSeekV4MTPModel` and v2 KV discovery produces no draft attention groups.
- The v2 probe fails during `initialize_kv_cache`, before graph capture or correctness: `AssertionError: No draft attention groups found.` This is a framework integration gap, not measured evidence that the model or Ascend graph mechanism is infeasible.
- All failed worker processes were removed and all eight NPUs returned to idle.
- The optimization boundary is now the fixed product runtime. Highest-value next unknown: the minimum explicit device-state and mutation contract for one correct proposer→verification→acceptance cycle, independent of generic scheduler and request abstractions.

## Loop028 — fixed c12 proposer replay boundary (2026-09-21)

- Scope: legacy DSpark proposer entry during warm pure decode, DP1×TP8, DSpark7, c12.
- Observations: 172 calls per rank across all eight TP ranks; `num_tokens=84`, one metadata step, fixed tensor shapes and strides.
- Stable process-local addresses on every rank: block table, draft-token buffer, hidden-state buffer, input-id buffer, position buffer, query start locations, sample indices, sequence lengths, slot mapping and target hidden states.
- Dynamic addresses: target token IDs and target positions. A specialized runtime therefore needs explicit stable owner buffers or copies for these two inputs before graph capture.
- Exact replay: one immediate repeat of the already-materialized proposer closure matched the `[12,7]` draft token tensor exactly on 8/8 ranks. First execution median was 37.021 ms and replay median 31.102 ms, but these one-shot synchronized diagnostics are not benchmark values or removable-time estimates.
- Proven graph/replay candidate: the full three-layer DSpark proposer after its inputs and metadata have been materialized.
- Still outside the proven boundary: target verification, rejection sampling/acceptance, sequence advance, target/draft KV state comparison, output publication and next-cycle input mutation.
- The traced decode sample is invalid for E2E comparison because tracing and a duplicate proposer call were enabled. No change to the accepted 543.65 tok/s baseline.
## 2026-09-21 Loop029 architecture boundary

The first product-owned fixed execution region is now semantically closed across all TP ranks: greedy acceptance → accepted-count calculation → sequence/position advance → next target-input ABI construction. This region is a candidate for one device-resident fused transition/SuperKernel because its shapes and temperature-0 semantics are frozen. No speedup is claimed yet. The immediate critical path is outside this region: direct target forward, target KV/recurrent/GDN mutation, DSpark proposal, and their TP collective boundaries must be moved under runtime ownership before graph/replay or cross-operator fusion can be measured honestly.

## 2026-09-23 Loop034 formal serving result

- Decode hot path: one admitted c12 cohort remains inside Extreme Runtime until
  its 1024-token product limit; Scheduler/ModelRunner do not execute per cycle.
- Output drain: fixed device history plus one terminal D2H; exact per-slot trim
  is proven on CPU for uniform and varying acceptance counts.
- KV ownership: bootstrap reserves 1024 lookahead tokens per request before
  handoff, making the long-run block table a real allocation contract.
- Serving residue: generic prefill/admission and final HTTP publication remain
  between cohorts, but the decode cycles themselves stay runtime-owned.
- Formal warm-cache output TPS: `217.342 / 218.884 / 215.889`, median
  `217.342 tok/s`, versus Stock `543.655 tok/s` (`-60.022%`). All runs passed
  48/48 requests at exactly 1024 output tokens.
- Median-of-runs TTFT p50 is `1942.120 ms`; TPOT p50 is `53.298 ms`. Rank-0
  cohort wall median is `53.373 s` for about 1025 cycles.
- Interpretation: the serving shell is correct, but it did not close the
  sustained decode gap. The next evidence target is a full-chain runtime-only
  profile, not more control-plane compatibility work.
## Update 2026-09-23 06:36 UTC — Loop035 Extreme decode DAG and state

| Extreme short c12 stage (eight real-weight cycles) | Median NPU event time across rank/cycle |
|---|---:|
| Target | 45.07 ms |
| DSpark proposer | 5.93 ms |
| Acceptance | 0.38 ms |
| Prepare target | 0.27 ms |
| State advance | 0.02 ms |

This is a short diagnostic, not the full 1024-cycle critical path. The dominant immediate E2E gap remains acceptance/progress, not these small transition stages. The first confirmed stale target fields are DSA-CP start_pos and local_seq_lens from cycle 1; refreshing only them did not recover acceptance. Stock's warm 12-request long cohort achieved 2.908 accepted drafts/iteration versus Extreme near one output/slot/cycle later in its short diagnostic. SAS/QLI derived metadata and same-state target/proposer comparison are the next discriminators. Sources: evidence/20260923_loop035_diagnostic/.

Loop035 oracle metadata control: per-cycle DSA builder refresh passed local state gates at 209 outputs/8×12 slots; adding GDN builder yielded 215, with unordered prompts. Direct eager target and builder overhead make these diagnostic outputs unsuitable for throughput comparison. The remaining semantic gap is first draft/target mismatch; next compare Extreme and Stock proposer on identical target hidden states, accepted tokens, and restored draft caches.

## Loop035 sustained Extreme-owned DAG, 1024 cycles (2026-09-23)

Run18 completed 1024 c12 cycles on all eight 910B3 ranks with target FULL graph,
exact fixed-state advance and host mirrors. Across ranks and cycles 16–1023,
NPU event medians (p90) were target 45.510 (45.984) ms, DSpark proposer
5.943 (6.028) ms, acceptance 0.305 (0.340) ms, prepare target
0.210 (0.254) ms, state advance 0.020 (0.020) ms. Within DSpark,
the borrowed model took 5.865 ms median; hidden packing 0.073 ms and
input preparation 0.004 ms. The profile is a standalone one-cohort diagnostic,
not a formal 48-request A/B.

The limiting semantic trend is progressive loss of speculation: rank-identical
outputs/slot/cycle fell from 1.625 in cycles 0–7 to 1.099 in 64–127,
1.012 in 512–575, and exactly 1.000 in 960–1023. Overall 1.0419
outputs/slot/cycle. Rank-0 emitted 12,797 tokens in 53.571 s, 238.99 tok/s
diagnostic; client TPS after intentional standalone abort is invalid.
Stock's first 8 cycles in its 64-cycle trace averaged 2.135, with later
8-cycle windows up to 4.094. The existing Stock long-run counter gives
about 3.908 outputs/slot/iteration. The first short-cycle deficit alone is
not a sufficient divergence locator; long-running derived target and draft
state need a paired oracle.

Loop035 correction (2026-09-23): the DSA/GDN metadata-builder conclusions attributed to Run12/13/19/21 are invalid because DirectTargetHandoff.forward never invoked the assigned diagnostic callback. Their raw acceptance measurements still describe eager direct-target execution. Run22 confirmed the unexercised callback by an empty audit; Run23 will test the connected hook.

## Loop035 current acceptance context (2026-09-24)

The frozen Loop034 formal Extreme cohorts had median 1025 cycles and median
1.278 staged tokens/slot/cycle across 16 rank0 cohort records. Current
flag-on c12 diagnostics finish near 289-303 cycles, but they use different
service timing and instrumentation. Run84 adds heavy per-cycle page snapshots
and its 128.83 tok/s is explicitly not performance evidence. The established
Extreme-owned DAG target/proposer critical path still guides optimization
after continuous semantic and acceptance gates pass.

## Loop035 post-fix serving critical path (2026-09-24)

The full-serving DSpark gid2 slot correction raises formal Extreme median from
217.342 to525.417 tok/s and reduces rank0 median cohort cycles from1025 to
300.5. A real-weight eight-rank event profile after the correction measured
steady per-cycle serial stages: target47.702 ms, target metadata8.673 ms,
DSpark proposer6.359 ms (model5.939 ms), acceptance0.337 ms, target
preparation0.207 ms, state advance0.024 ms. These event intervals are
diagnostic and must not be added to official client latency. The target
graph is already FULL; the next removable-gap analysis should focus on
target execution and the metadata update, while preserving c12 KV/DSpark
semantics. Formal evidence:
evidence/20260924_loop035_formal/run93/summary.json;
stage evidence: evidence/20260924_loop035_diagnostic/run87/summary.json.

## Loop036 Run98 static target metadata profile

On the same 12×1024 c12 DP1×TP8 diagnostic contract, removing the per-cycle
`local_seq_lens.max().item()` fallback scalar sync reduced the derived target
metadata event median from Run87 8.6730 to 0.6674 ms/cycle (-8.0056 ms,
-92.3%). Target remains 46.560 ms and DSpark proposer 6.391 ms median.
The static path passed 300 continuous cycles on all 8 ranks; formal 48-request
E2E is still required before attributing product throughput gain.

## Loop036 Run99 formal product effect

The 8.0056 ms/cycle metadata-stage reduction translated into frozen warm-cache
48×32K→1024 c12 output TPS median 571.681 (three samples 612.962,
567.573, 571.681), versus Run93 Extreme 525.417 (+8.806%) and Stock
543.655 (+5.155%). All 128 rank/cohort rows passed. Target graph replay at
46.560 ms/cycle is now the largest measured serial stage; DSpark proposer
6.391 ms and metadata0.667 ms follow. Above-Stock margin is real under this
protocol but still modest, so next optimization should diagnose target device
work/communication rather than revisit metadata sync.

## Loop038 Run106 bounded target trace (2026-09-24)

A legal 8-rank 12×1024 c12 profile captured two steady cycles per rank
without HTTP control latency. Profiler target-attributed device total median
is 51.144 ms (range46.255–56.502) under instrumentation, consistent in scale
with unprofiled Run98 target event median46.560 ms. CPU target scope median
6.103 ms only measures graph enqueue; direct timestamp clipping cannot map
asynchronous graph kernels. Target-nested wait_event median50.134 ms exposes
completion of preceding device work, whereas actual nested HCCL all-gather
sum is0.247 ms. Whole-trace grouped matmul kernels sum20.592 ms across two
cycles, including target and proposer; it is a candidate family rather than
a measured target-only removable bound. Compute/communication overlap requires
correlation-aware attribution. Source:
evidence/20260924_loop038_cycle/run106/attribution.json.

## Loop038 Run107 synchronized target graph attribution

Under a two-cycle opt-in profiler/synchronization window, 15 canonical target
windows across eight TP ranks contain2,836 kernels/cycle. The target device
interval union median is50.281 ms, comprising compute union39.932 ms and
communication union11.547 ms with1.292 ms overlap. Communication union varies
5.085–25.923 ms across rank/cycle; compute union is stable39.480–40.232 ms.
The W4A8 grouped-matmul family executes86 kernels/cycle and sums9.966 ms
(median), about25% of the compute union; this is a high-value operator
candidate but its summed kernel time is only an upper limit on removable
critical-path time. One rank1 cycle65 window had143 extra kernels and is
excluded. Profiler/sync perturbs latency; unprofiled Run98 target event median
46.560 ms remains the comparable stage baseline. Evidence:
evidence/20260924_loop038_cycle/run107/target_window.json.

## Loop045-047 warmed cohort prefill and target tail (2026-09-25)

Legal warmed cohort diagnostics localize about3.4s pre-handoff execution to repeated `_model_forward` calls; one profiler-perturbed 83-token call spent90.4% of its device span with no recorded kernel, mostly before the next HostToDevice flow. This is a Host submission hypothesis, not removable-time proof. Exact-state prefill graph lacks within-cohort shape reuse, and tested250/500ms Core admission holds produced no material net diagnostic gain (Run171/172/173 client envelopes19.368/19.278/20.494s); leave them disabled. Inactive-slot tail exposure in Run155 measured cohort is18.525% slot-cycles, with44 of296 cycles at<=6 active slots. Its2.553s ideal-linear target arithmetic is not an achievable saving: target GMM weight traffic and fixed c12 metadata/graph/state may remain constant. Measure real smaller-batch target graph marginal latest-rank latency before investing in c6/c3 runtime variants. Run99 formal571.681tok/s remains the accepted product point.

## Loop047 Run180 target-tail sizing and priority (2026-09-25)

Same-service Stock FULL graph c12 early/late event medians53.461/54.996ms bound observed order drift to1.536ms at the endpoints. c8/c6/c4/c1 medians50.787/49.020/47.379/40.678ms. Applying those diagnostic size deltas to Run155's measured 296-cycle active-slot histogram yields0.471–0.602s/cohort target-only savings if graph switching and c12 ABI compaction were free. This is not Extreme target-stage or E2E savings. The broad state/metadata/DSpark rewrite is deprioritized. Run145-150 separately constrain GMM and exposed communication candidates: the first reduce-scatter's early-rank wait ends at nearly aligned timestamps; real-shape GMM weight reads are close to packed bytes at ~1TB/s. The largest plausible tractable gap to test next is warmed prefill Host submission, not an established numeric saving.

## Loop048 Run184 low-overhead Extreme prefill phase (2026-09-25)

A legal warmed Extreme 12-request cohort has five rank-matched prefill forward shapes88/264/368/152/328 tokens. Summing each call's eight-rank maximum wall gives1.901s, while latest-rank decode Runtime wall is16.307s. DSA+MoE custom-op CPU wall covers87.7–88.0% of forward wall and executes43 layers each per call. This lowers the priority of an unassigned generic Host-gap optimization: the Run165 profiled device-free time largely coexists with explicit layer/operator dispatch, and no part is yet demonstrated avoidable. It does not prove these Host paths are at bound. Target graph execution remains the largest serial stage in steady decode; Run99 formal record unchanged.

## Loop048 Run186 prefill Host activity (2026-09-25)

A legal same-service Extreme detailed cohort had nine eager prefill calls on all8 ranks. Per-call max-rank forward wall summed3.424s and thread CPU3.416s; per-rank median CPU/wall was99.70–99.98%. The Run165 device-free spans are therefore consistent with active Host submission, not a large off-CPU sleep, in this newer cohort. Control had six eager prefill calls plus three FULL forwards, detailed had nine eager calls; different shapes and sequential order prevent a causal perturbation or admission gain claim. To recover0.5s from nine calls requires roughly55.6ms less critical-path wall per call, about15% of the observed forward wall, with exact state preserved. No such edit is yet validated.

## Loop048 Run188 admission closure (2026-09-25)

A same-service A/B/A-prime legal Extreme diagnostic gathered all12 Core requests after1.089s wait and reduced prefill forwards7/8→1, saving gross max-rank prefill wall2.359–2.592s. Complete client envelope changed only20.822/20.769/21.042s; B decode was331 cycles versus controls299 and its latest-rank Runtime wall rose18.685s versus16.930/17.019s. Local prefill elimination is not a proxy for throughput gain. Keep the admission hold disabled. Next target-stage screen tests active expert weight-load imbalance against same-cycle GMM/HCCL timing; do not treat HCCL peer wait or GMM kernel sums as independently removable.

## Loop049 Run189 active expert balance screen (2026-09-25)

Run121 cycles64/65 rank aggregate active expert-layer counts span645–697 and655–712. A physically impossible per-layer perfect balance would remove161/175 packed expert reads, roughly2.0–2.2ms at a representative1TB/s; this is not a removable latency bound. Separate-service Run107 GMM rank-sum spread is only0.39–0.44ms. Expert placement may redistribute layer maxima but no same-state timing correlation or cheap supported remap is established. Audit existing placement support before implementing anything.

## Loop049 Run190-191 static placement screen (2026-09-25)

The available `expert_map_path` changes Ascend execution routing without changing upstream checkpoint physical weight placement, so arbitrary static remap is unsafe without loader integration. Fixed per-layer 32-expert/rank maps tuned on one Run121 cycle saved57/78 active reads in sample but transferred only+3/-1 reads to the other captured cycle; those read counts correspond to roughly+0.038/-0.013ms at the illustrative1TB/s before routing and implementation costs. Two cycles are insufficient to establish a general placement bound, but do not justify an invasive correctness-sensitive remap. Pivot to same-state target communication/compute attribution; formal Run99 remains571.681tok/s.

## Loop050 Run192-195 target arrival correction (2026-09-25)

Run107 profiled first reduce-scatter durations reflect rank arrival: start skew8.65–20.53ms, end skew<=0.014ms; following259 HCCL tasks sum median5.205ms. The prior proposer appears to propagate target-entry skew *inside that profile*, but Run107 proposer CPU scope median55.644ms is8.62× Run98 low-overhead event median6.454ms on separate services. In Run98's260 steady cycles, proposer eight-rank duration spread median0.063ms and target0.090ms. Do not count profiled arrival skew as removable product time. Target graph remains the largest measured serial stage, with its GMM/other compute and required communication still unbounded for achievable throughput.

## Loop051 Run197 compressor cache-write screen (2026-09-25)

A precise source/trace match bounds c128 compressor-following scatter to20 kernels and0.337ms profiled summed device time/cycle; c4 adds42 kernels and0.793ms. The full compressor-scatter family1.123ms is less than half Run143's all-scatter2.344ms. Direct destination writes require a custom compressor kernel/interface change, not merely removing a Python call, and cannot save required cache writes. No achievable product gain is established; this path has lower priority than the unlocalized prefill active-submission cost.

## Loop052 Run200-202 prefill stream/event closure (2026-09-25)

Prefill-only auxiliary-stream removal activated on8 ranks in legal same-service A/B/A-prime; shape-matched B8-call max-rank forward sum3.075s versus control A-prime3.057s, with no material client improvement. A one-card same-stream record+wait Host microbench gives15.65µs/pair net; two pairs×43 layers would be1.35ms/forward gross. Existing stream/event plumbing does not explain the3.4s/cohort prefill Host exposure. Exact candidate output parity was not established because control response hashes were unstable across cohorts; no patch kept or formal E2E run.

## 2026-09-25 Loop057 target mechanism checkpoint

Run107/143 synchronized target-window median device union50.313ms: GMM1+2 summed9.966ms, quant matmul4.822ms, Compressor3.366ms, HC-pre2.962ms, scatter2.344ms, Indexer1.746ms, SparseAttn1.621ms, routing init1.056ms. HCCL reduce-scatter8.771ms includes rank arrival wait. These profiled task sums overlap and cannot be subtracted from cycle wall. Run98 unprofiled target event median46.560ms remains the stage reference. Run233 Graph surrogate at synthetic96-row shape saved only0.45us/call; Run235 local12-row FP32 numerical gate failed before timing. No target candidate was promoted. See evidence/20260925_loop057_target_bound/run234/ and run235/.

## Loop058 V0 hierarchy and calibration (2026-09-26)

Formal Run99 has 1217/1212/1206 cycles and 80.188/86.600/85.978s E2E across three passes, while rank0 decode wall sums 69.050/69.457/69.040s. Run238 per-wave client-minus-decode windows are 2.545–4.521s; cross-rank decode-wall spread is <=0.019s. Major formal variation is outside runtime-owned decode wall, not evidence for another target kernel intervention. Run237 TP8 no-service forty-pair BF16[49152]+FP32[3072] all-gather chain took 21.420ms latest-rank median (0.53549ms/pair). Run152 0.83810ms is the SUM of 33–41 product graph pairs per target window (~0.02288ms/pair), not a per-pair time. Run237 eager API is ~23.4× slower and cannot calibrate graph HCCL capacity or product savings; see Run240. See `evidence/20260926_loop058_bound/run236/method_evidence.md`. Next discriminator is low-overhead boundary capture of prefill/admission/publication on a legal formal schedule.

## Loop059 Run239 legal boundary pass (2026-09-26)

The reused Loop045 reversible patch was applied to exact prior source SHA and removed after the run. Warmup and diagnostic pass each completed 48/48 exact 1024 output; measured pass 592.618 tok/s is instrumented and not a new formal product point. Across measured cohorts 5–8, same-host all-rank boundaries give client-to-first-execute 0.207–0.234s, first-execute-to-handoff 2.356/3.747/4.115/3.318s, runtime serve 16.800/17.504/16.241/17.268s, and publication-to-client-end 0.173–0.178s. Handoff/build and serve/publication edges are <=0.007s. All eight rank runtime rows passed. The full phase records and client requests are at `evidence/20260926_loop059_boundary/run239/`; raw service log is retained by SHA index. Exit 0, service stopped, source hashes restored, eight NPUs idle.

This calibrates the V0 residual: prefill-to-handoff and decode work both vary substantially across cohorts; admission and output publication are small, stable in this one pass. It does not identify removable prefill time or a hardware upper bound. Next update the executable model with measured phase distributions and compare counterfactual savings against full-cohort dependency and Run188 admission tradeoff; only then rank prefill architecture against target DAG.

## Loop059 Run241 phase calibration and trajectory coupling (2026-09-26)

`scripts/extreme_bound_phase_calibrate.py` reads Run239 same-host all-rank boundaries and Run188 admission A/B/A-prime. The 82.940s instrumented pass contains 13.536s first-execute-to-handoff prefill (16.32%), 67.813s runtime serve (81.76%), and 1.600s in the measured client/admission/build/publication edges (1.93%). Across 1195 decode cycles, latest-rank runtime wall is 56.782ms/cycle on this trajectory. This is phase accounting, not necessary-work or removable-time attribution.

At fixed cycles and unchanged semantics, a hypothetical 0.5s/cohort exposed prefill saving would project 607.261 TPS from diagnostic 592.618. Adding 8 decode cycles/cohort reduces that projection to 593.928 TPS. Run188 actual admission hold removed 2.476s gross prefill forward wall but added 32 decode cycles and 1.711s runtime wall; observed client improvement was only 0.163s versus controls. The simple gross prefill minus runtime arithmetic predicts 0.764s improvement and misses the observed outcome by ~0.602s, due to overlap, changing shapes/acceptance and other boundary effects. Therefore fixed-trajectory Engineering/Aggressive scenarios remain conditional and cannot be promoted to achievable product bounds. Evidence: `evidence/20260926_loop059_boundary/run241/calibration.json`.

## Loop060 Runs242–243 resource inventory (2026-09-26)

`extreme_resource_inventory.py` records source hashes, 43 target layers (21 c4, 20 c128, 2 uncompressed), Run115 product W4A8 shapes and Run146 real c12 routes. The target GMM has 8.462GB active packed expert weights and about 155.676GFLOP logical matmul/rank/cycle; applying Runs148/150 one-card counter bandwidth gives a conditional 7.880ms packed read estimate versus Run107 profiled 9.966ms GMM task sum. Different cohorts and concurrency prevent turning the 2.086ms difference into exposed gain. Cache ABI records c4/c128 state dimensions, BF16 SWA and block-size-32 mappings, but actual KV HBM bytes and attention reads remain UNKNOWN. Four Run239 warmed prefill cohorts have 7/11/12/10 scheduled-token calls, not a single repeatable prefill shape. Evidence: `evidence/20260926_loop060_resource/run242/inventory.json`.

`extreme_kv_row_census.py` reuses Run84 eight-rank 256-cycle legal page audit without service. Over cycles64–255, c4 compressor and indexer each emit exactly24 valid rows/rank/cycle (504 layer-rows across21 layers), while c128 compressor emits median1, mean0.698 valid rows/rank/cycle (20 layer-rows median across20 layers); first c128 write is cycle8 on all ranks. Page candidate counts and row cardinalities are not HBM transactions. SWA, MTP, cache read reuse and physical page padding must be accounted before converting these to bytes. Evidence: `evidence/20260926_loop060_resource/run243/kv_rows.json`. The largest unresolved target resource term remains non-GMM DSA/quant/attention execution and actual KV read traffic; next measurement must preserve real shapes, dependencies and graph path.

## Loop060 Run247 product-graph memory counters

All 80 exported eight-rank target windows passed exact family counts; 16 latest-cohort rank-cycle windows are valid. Latest medians per rank-cycle: GMM 9.244GB read (1.092 cross-sample ratio to active packed estimate, not same-cycle amplification), quant matmul 3.465GB, compressor 1.530GB, sparse attention 1.062GB, other 3.618GB; total AIC+AIV reads 18.965GB and writes 2.380GB. HCCL 265 tasks/window has no link-byte counter. The substantial non-GMM and unclassified traffic moves next analysis to DSA/quant/attention dataflow and dependency mapping. The 54.796ms profiled target-scope median and kernel-duration sums cannot be treated as formal E2E or exposed savings. See `evidence/20260926_loop060_resource/run247/findings.md`.

## Loop060 Run248 next bottleneck screen

Reconciled latest 16 rank-cycle graph windows attribute the Run247 `other` 3.618GB read mainly to plain/transpose matmuls1.788GB combined, inplace copy0.618GB and AivKernel/Hc* names0.756GB. Quant matmul3.465GB, compressor1.530GB and sparse attention1.062GB are separately measured. GMM's 9.244GB reported read / 8.462GB active packed estimate is a cross-sample 1.092 ratio with non-weight traffic in the numerator; it weakens a multi-fold reread hypothesis but does not measure removable weight traffic. HCCL link bytes are still unavailable. Byte/rate sensitivity of total 21.344GB traffic is only illustrative and does not establish a full-graph bound. Choose a non-GMM same-shape intervention and judge it by correctness and formal E2E. See `evidence/20260926_loop060_resource/run248/findings.md`.

## Loop060 Run249 exact non-GMM shapes

Latest 16-window counter-rate audit: leading quant matmul shape 1.043TB/s across43 calls, Compressor main shapes0.529/0.531/0.303TB/s, transpose matmul0.431TB/s. These are reported read divided by summed profiled task time. Required arithmetic and distinct output products prevent treating lower ratios as waste. Loop044/Run197 found no safe source-level Compressor bypass; a custom numerically equivalent intervention is needed before formal E2E testing. Evidence: `evidence/20260926_loop060_resource/run249/findings.md`.

## Loop061 HCCL payload and independent capacity (2026-09-26)

Run250 validates 80/80 rank-cycle target windows: each has 265 Graph HCCL events with the same reported size signature, summing 25,651,200B per rank-cycle. Link/transport fields are INVALID_TYPE and transit-size counters remain zero. This establishes the current collective operation inventory only, not necessary TP8 wire bytes. CANN 9.1.0 HCCL Test in the same container on 8×910B3 gives independent 96KiB BF16 -t1 device-only AllGather/ReduceScatter/AllToAll 40.03/39.74/62.17us (Run254), with normal tool time 167.40/102.23/205.65us under the same 256MB HCCL buffer. Standalone tool and captured Graph service times differ, so neither is a product critical-path saving. Run252's -i0 nonterminating test is INVALID and excluded. Full evidence: evidence/20260926_loop061_bound/findings.md.

Astra High independently reviewed V0 and Run247–249 (Run255): 581–607 and 616–682 tok/s remain assumed exposed-saving scenarios, not Engineering/Aggressive achievable bounds; the true hardware/algorithmic ceiling is UNKNOWN. Current biggest identifiable uncertainty is non-GMM compulsory versus extra data movement *and* whether any change reaches the latest-rank target boundary. Compressor BF16 two-matrix unique weight footprint is 0.6082GB/cycle across 62 calls versus 1.5309GB reported read; the difference includes necessary X/state/workspace and possible repeat reads, so no waste claim follows. Next source-level read census, then a minimal same-state Graph-path intervention with full correctness and formal E2E if promising. Do not restrict search to Compressor when another measured exposed Gap is larger.
