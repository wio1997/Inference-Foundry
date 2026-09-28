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

## Loop062 next scheduling path (2026-09-26)

Run265 adds Current, Hardware/Resource, Scheduling-aware and Product E2E as separate objects. Source dependence permits next target geometry and RoPE/SAS/QLI metadata to be derived after acceptance while current DSpark runs, but old target buffers and shared RoPE storage cannot be overwritten until DSpark consumers finish. Serving parking can change next positions after `step()`, requiring invalidation. A0 current, A serial private scratch and commit, B private scratch side-stream overlap are the next causal variants. Expected exposed gain is UNKNOWN; Run98 metadata time is only a screen. Run267 found the first Graph reduce-scatter duration dominated by profiler rank-arrival skew, so Run266 HCCL union is not a communication bound. Exact source and formal comparison: `evidence/20260926_loop062_nongmm/findings.md`.

Run276 corrects the frozen DSA path: `enable_dsa_cp` invokes `AscendDSACPImpl._forward` in `context_parallel/dsa_cp.py`; the CV `dsa_v1.py` multistream overlap is not part of Current. c4 indexer query/QLI and main compressor/scatter have independent inputs after indexer cache update and join at sparse attention. This is a larger Scheduling-aware candidate than next-cycle metadata, but benefit remains unmeasured. See `evidence/20260926_loop063_schedule/run276/astra_cp_review.md`.

Loop063 B next-target scratch overlap passed candidate-consumed 8-rank correctness/storage gates but one-pass no-verifier scheduling screen showed B +1.472ms/cycle slower than contemporary A0 on latest-rank runtime, despite +31.364 client tok/s from −6.233s client-minus-runtime residual. A0 launcher exit127 followed a live-script edit after complete valid artifacts; this comparison is diagnostic, not formal. PIVOT the specific metadata candidate, keep Run99 Current571.681, and test the frozen c4 CP indexer/main-compressor fork/join identified by Astra High. See `evidence/20260926_loop063_schedule/findings.md`.

## Loop064 dual bound checkpoint (2026-09-26)

Resource/Hardware and Scheduling-aware limits are separate unknowns. Current Formal is Run99 571.681 tok/s median; neither Stock 543.655 nor V0 Engineering/Aggressive sensitivity scenarios are ceilings. Run284's one-layer delayed CP Compressor join reached the next AllToAll 39.875 µs earlier at paired rank-cycle median than an immediate join, but the immediate control already changes A0 order. There is no attributed full-cycle or Product gain. Run285 all8 eager same-prestate gate was inconclusive because later off self-replay differed after candidate passes; 12/12 exact client length is only a contract check. Candidate stays unpromoted. Run286 read-only original-path ownership/page-frontier probe is the next Resource uncertainty reduction, with actual scatter, state history, DSpark and prefix lifetime still open.

## Loop064 Run287: CP request ownership

All 8 ranks x 5 original-path FULL Graph cohorts confirm 12 local query rows and two request owners per rank, requiring 16 full new-input rows for the two owners. The other 80 per-rank rows are only a conditional removal candidate. Conservative layer2 compressed page-envelope intersection is empty in 11,968 rank-cycles; native scatter/state/DSpark/prefix lifetime closure is still missing. This separates a plausible resource redesign from the independent H003 hidden AllGather/local-Q scheduling candidate. Neither is yet a formal E2E gain; Current remains 571.681 tok/s. See evidence/20260926_loop064_cp/run287/findings.md.

## Loop065 matched scheduling / Loop066 architecture pivot

Run292/293 establish a real one-layer Scheduling-aware relaxation in the frozen FULL Graph: Target layer2 hidden AllGather overlaps local Q in all 16 delayed rank-cycle trace samples. Versus same-profiler async-immediate control, delayed wait advances the first WKV consumer by paired median17.0µs and the same-layer next AllToAll completion by10.125µs (16/16). The subsequent AllToAll itself lengthened by3.93µs paired median; full Target tail was faster only8/16 and first-to-next Target span only4/8. This is a local device mechanism, not a full-cycle or E2E gain. Same-entry typed parity and A0 remain open. See evidence/20260926_loop065_gather/run293/astra_matched_review.md.

Astra High also reconstructed Run287 request progress: 2,989/17,952 slot-cycles (16.65%) had parked slots while the fixed96-row target continued; active requests produced4.139 staged tokens/cycle on this instrumented trajectory. Row/capacity fractions are not removable FLOPs, HBM, wall time or Run99 formal acceptance. Together with Run287 owner geometry, this motivates Loop066 private layer2 c4 owner16-versus-full96 fixture. Resource/HW, Scheduling-aware and Product E2E numeric bounds remain UNKNOWN; Current Formal stays571.681tok/s. See `evidence/20260926_loop065_gather/astra_dual_bound_next_review.md` and `evidence/20260926_loop066_owner/design.md`.

## Loop066 Run294 and Performance Knowledge integration (2026-09-26)

Run294 private same-prestate owner16 native c4 indexer Compressor at real Target layer2 produced four owner output slots bit-exact against full96 on all eight ranks; full A/A2/A3 self-controls were stable. The broader whole-owner-page FP32 state byte gate failed on ranks0–6 and passed on rank7. This cannot classify 80 nonowner logical rows as removable HBM or compute. The next measurement localizes A/B differences to physical bytes changed by B versus prestate and block-table aliases, then tests typed key/scale, QLI/Sparse and lifetime only if safe. Eager diagnostic 12×1024 passes Runtime but its 83.299 tok/s is instrumentation cost, not Product performance.

The pinned historical experience source and query tool are now `performance_knowledge/sources.json` and `scripts/performance_knowledge.py`. Six curated on-demand entries connect historical R20/R21/R28/R18/R33 to current H003 and owner16 evidence. Before new architecture candidates, retrieve old conditions and failure modes; never import an old KEEP/REVERT into the Current→Bound verdict. Current Formal remains571.681 tok/s; Hardware/Resource, Scheduling-aware and Product numeric bounds remain unknown.

## Loop066 Run295 state attribution

The Run294 7/8 whole-owner-page inequality is not direct evidence that owner16 produces wrong live state. In a fresh Run295 private fixture, all eight owner outputs remain exact; `B_changed_bytes_different_in_A=0` on every rank, and all A/B differing bytes are A-only prestate changes. First differences on ranks0–6 map to nonowner current writes at pages also listed at old owner block-table positions. The c4 state sliding window is8 tokens. Full per-byte live write/read closure and typed scatter/QLI remain pending, so this evidence is a Resource hypothesis refinement, not an accepted deletion or current gain. The executable Dual-Bound model retains null Hardware/Resource, Scheduling-aware and Product ceilings. See `evidence/20260926_loop066_owner/run295/findings.md` and `performance_knowledge/entries.jsonl` PK-005.

## Loop066 Run296 typed consumer gate and Run297 cost screen

Run296 passed all8 real layer2 private state/scatter/native QLI gates: owner16's four valid output slots, source-derived current write/live8 read state bytes before and after scatter, typed key/scale owner slots, and fixed-query QLI topk `[12,1,512]` match full96 bit exactly. A/A2/A3 controls, 12×1024 eager carrier and all8 Runtime reports passed. This is one-call consumer closure, not Graph, lifetime or E2E KEEP. Run297 will use paired full96–owner16–full96 private event spans at update-end and QLI-end to test whether native execution cost actually falls. Existing historical R21/R28 resource-competition failures remain priors only. Formal Current 571.681 tok/s and numeric Hardware/Resource, Scheduling-aware and Product bounds remain unchanged.

## Loop066 Run297 paired eager cost screen

All8 same-prestate owner16 parity and repeated topk remain exact, but paired A96–B16–A2 eager event spans do not show a stable update or through-QLI advantage (9/40 both-control wins at either endpoint; rank-median B−full +0.00303/+0.00016 ms). No formal performance gain is claimed. A private captured Graph replay is the sole bounded Host-gap disambiguation before pivoting from the current native owner16 small-shape path. PK-005 records the conditions and failure mode; numeric bounds stay UNKNOWN.

## Loop066 Run298 private Graph and full producer dependency

Run298 all8 private Graph capture/replay passes state, key/scale and native QLI parity. B16 outputs 6 compressed rows versus A96 36, yet the paired replay signal is only about 4–6 μs and below 8.24 μs median full-control drift; no Product speed claim. Astra High supports pivoting from current isolated native small-shape integration toward entire DSA producer owner/halo dependency closure. Run287 alias census: 3 related backings, 2.376 GB/rank, same layout in 40/40 rank-cohort samples, with two cross-layer shared backings. PK-007 records why byte-level last-writer and consumer liveness must precede a multi-GB private clone or halo experiment.

## Loop067 Run299 — storage reuse versus semantic dependence

Run299 original FULL Graph 8-rank typed registry records 15 views over three layer2-related backings per rank and all three DSpark draft cache leaves. Draft views have no physical overlap with those stores. For ten selected cycles/rank, layer2 nonowner current-write versus owner bounded-read source-derived page domains are disjoint; the SWA all-prefix page false positive disappears at the real 128-token `(page,slot)` window. The two large Target backings are shared with layers0/1/3/4/5, so their read/write lifetime remains an architectural scheduling question. No traffic amount or E2E saving is inferred. Next gather those 14 alias source metadata domains compactly, then use actual value gates only for concrete unresolved overlaps.

## Loop067 full-producer execution gate (2026-09-26)

Run301 live runner exited1 only because the post-check expected the wrong layer0/1 Compressor ratio; client 12/12×1024 and all8 FULL Graph Runtime captures completed. Run302 corrected offline validation of unchanged captures passed. Run303 compared all 3094 later-layer Compressor state full-prefix page hits with the SHA-pinned actual container packaged source's c4/c128 conservative history windows; none falls inside (minimum ages 222/1126/1326 tokens versus windows 8/128/128). This reduces a sampled address-alias uncertainty, conditional on packaged source matching the loaded native binary. It neither removes compulsory traffic nor proves persistent owner semantics. The next gate is a single-layer full WKV/SWA, main Compressor, indexer and Sparse same-prestate private A/A/B/A value comparison, then one private Graph critical-path screen if it passes. Historical R21/R28 warn that Compressor overlap can lose to resource competition; see PK-008. Current Formal remains 571.681 tok/s; Resource/Hardware, Scheduling-aware and Product E2E numeric ceilings remain unknown. Evidence: `evidence/20260926_loop067_liveness/run301/`.

## Loop068 local complete producer gap (2026-09-26)

Run305 exact same-prestate all8 layer2 WKV/SWA, main Compressor, indexer/QLI and Sparse owner16 versus full96 gate. Run307/308 private Graph A/B/A screens kept all8 Sparse/QLI and (Run308) every-replay persistent owner cache write domains exact. Paired B faster than both A controls 77/80 then 80/80; diagnostic max-rank paired medians −31.65 and −24.46 µs. This is reproducible *local* producer→Sparse work reduction. It excludes AllGather, local Q, rank rendezvous, full Target/cycle and serving. Astra High supports a single-layer live Target test with communication unchanged. Current Formal 571.681 tok/s and numeric Resource/Hardware, Scheduling-aware and Product bounds remain unknown. See PK-009 and `evidence/20260926_loop068_producer/run307/findings.md`.

## Loop072 real MoE row geometry and post-gather candidate

Run322 all8 real layer4 MoE census: Target outer96/pad0, local12 rows/rank, FlashComm1, ALLGATHER MoE prepare and EP reduce-scatter finalize, `is_sequence_parallel=False` inside MoE. Six active requests yielded `[8,0,4,12,4,12,8,0]` local active rows. Run323's original-mask remap found 579/678 parked cycles still max-local12, so local-only row deletion seldom moves a max-rank row-proportional endpoint. Its 2.875% aggregate max-row proxy is conditional and cannot be promoted to a numeric Scheduling Bound. The existing gather offers a smaller branch: compact global96→active48 for routed expert apply, scatter routed result back to96, preserve shared path/collectives. Correctness, routed HBM savings, shared/communication overlap, full-MoE endpoint and Product E2E are unmeasured. See `evidence/20260926_loop072_parked_moe/run322/`, `run323/`, PK-014.

## Loop074 Run339–340 priority (2026-09-27)

Official installed CANN9.1 TP8 HCCL Test gives isolated actual-payload service medians 4.95–32.10µs across five collective cases (three independent checked processes each); no mixed-chain or Product bound. Run337 no-profiler Host call arrival skew is separate. Run225/227 prior already covers first88 four-layer Graph and independent-bank memory failure, so the next larger Scheduling gap is conditional b2–4 early refill, not a repeat single-layer Graph. Run340 source audit identifies bulk output/slot ownership as the transaction boundary; Run341 original-path natural microbatch timing is next. Current Formal571.681 tok/s; Hardware/Resource, Scheduling-aware and Product numeric bounds UNKNOWN.

## Loop074 Run341 original-path microbatch screen

Natural warmed admission occurs in1–3-request groups with83–251 residual prompt tokens per `_model_forward`; 26 such calls across four measured cohorts cost353–384ms max-rank Host wall and354–411ms max-rank current-stream event. Eight-rank entry/return spreads grow materially across the sequence. Original path correctness/Runtime passed, but no incremental refill, complete seed-ready DAG, or formal E2E candidate exists. Pursue all-rank complete preparation/first-Target dependency and state ownership before a live b3/b4 refill; do not turn the Run340 screening budget into a bound.

## Loop074 Run346 Bound V3 priority correction

Run346 makes the main uncertainty explicit: Current Formal571.681tok/s is established, but no defensible finite overall TPS ceiling exists yet. Run287 fixed-duration1011/1015 cycles are capacity relaxations; zero-cost FIFO1118/1122 are constructed conditional schedules; Run99's512 is a cardinality floor for the current four zero-output handoff cohorts. None includes the complete legal product DAG. The cross-run b2/b3/b4 break-even screen requires hiding about46–56% of Run332's6.935s later-wave preparation if that same work is needed. This is useful for ranking measurements, not a refill performance forecast. Prioritize a correlated formal48 request work/dependency ledger joining real arrival, prefix/residual prefill, seed-ready, useful acceptance, Target/DSpark resource work, collective joins, state/KV and final publication. Run345 segmented publication remains a narrow legal-arrival gate, pending this Bound audit; its partial code has no correctness or E2E result. See `evidence/20260926_loop074_refill/run346/findings.md` and `bound_calibration_v3.json`.
# Run347 corrected — saved Run239 acceptance and scheduling calibration

This is a **read-only reanalysis** of Run239's measured 48-request diagnostic pass, not a new service benchmark. All eight ranks have identical accepted-count matrices for each of the four measured cohorts. Replaying and clipping the 12-slot counts at 1024 verifies **49,152 Runtime-retained tokens**; the client benchmark separately reports 48×1024 external outputs. The saved boundary matrices do not contain each request's actual pre-handoff published token count, so those two 49,152 values must not be equated token by token. The cohorts ran 298/310/285/302 Target cycles, **1195 total**, or 41.131 Runtime-retained tokens per Target cycle. This is observed algorithmic efficiency of the current Runtime suffix, not an algorithmic ceiling.

The 48 observed per-request Runtime completion durations total **12,534 slot-cycles**. Their slot-capacity terms are `sum_cohort ceil(sum_slot duration / 12) = 1046` for four fixed cohorts and `ceil(12534 / 12) = 1045` with arbitrary refill. **These terms alone are not complete lower bounds**, because each request must traverse its own serial Target cycles. The fixed-cohort request-chain term is `297+309+284+301 = 1191`, dominating 1046. Thus the fixed-cohort conditional lower relaxation is **1191 cycles**; the observed 1195 includes one post-retained-completion cycle per cohort. Arbitrary refill under *unchanged per-request durations* has only the weaker `max(max_i duration = 309, slot-capacity term = 1045) = 1045` relaxation. It does not include legal request arrival, new prefill/seed, KV/state, shared eight-card resource contention or acceptance changes and is not an attainable schedule or Product bound. Run287's 1011/1015 capacity terms and FIFO1118/1122 use a different diagnostic trajectory; they cannot be combined with Run239 as one bound.

The independent **512** cardinality term applies to four current zero-output handoff cohorts at eight possible tokens per request per Target cycle. It does not prove that acceptance eight is attainable or that the pre-handoff/client output ledger is zero. The final Runtime-retained token was staged **one Target cycle before** the serving loop stopped in each cohort, consistent with the Host progress mirror lag. Four trailing cycles are a Current control observation, not an automatically removable Product gain. Raw staged counts exceeded Runtime-retained counts by 67/72/78/87 per cohort, including overshoot and trailing work, without resource criticality attribution.

Run239's maximum rank serving wall per cohort was 16.810/17.514/16.251/17.279 s. These instrumented host boundaries do not supply per-cycle device costs or a formal Run99 latency. The V3 model carries this same-trace acceptance calibration while leaving finite Hardware/Resource, Scheduling and Product bounds unknown. Next obtain actual per-request pre-handoff published `p_i`, close compulsory Target+DSpark/prefill work and same-path capacity, and capture actual arrival→seed-ready if refill is revisited.

Evidence: `run347/acceptance_reanalysis.json`, corrected `run347/bound_calibration_v3_1.json`, Run239 saved eight-rank boundary matrices and measured client JSON. Script: `scripts/extreme_bound_run239_acceptance.py`. Corrected after independent Astra High review; the prior wording treated the capacity term1046 as a fixed-cohort bound and conflated Runtime-retained with externally new tokens.

## Loop075 Run348–365: measured GMM bank sensitivity with no Product-bound promotion (2026-09-27)

The eight-card GMM Graph A/B/A2 screen was corrected from a mixed-ordinal route to the same real Run121 ordinal64 across all ranks. Under isolated synthetic zero-weight/input 16-call Graph replay, bank8 versus bank1 controls slowed GMM2 by8.037% median, strict on8/8 ranks; GMM1 changed0.212% median, strict4/8. All-rank Host windows overlap, but per-replay device overlap and original Target/DSpark/HCCL contention are not established. A separate nonzero eight-bank Graph fixture exactly matched eager outputs across the120 active rows for all16 captured calls.

One-card rank4 Level1 MemoryAccess exports show two complete native16-task replay groups under bank1 and bank8, after partial profiler starts. GMM2 AIC read94,728KB/call and GM→L1 93,824KB/call were unchanged; AIV read changed slightly, while profiled duration rose83.592→91.092µs. The exports logged ACL→NPU flow-event correlation errors; timing is not substituted for unprofiled attained service. These counters do not identify physical HBM bytes or cache hit behavior, and the isolated bank conditions do not bracket original FULL Graph node cost.

The executable V3.3 Bound model records this as a **conditional Resource/Hardware service sensitivity** and a Scheduling node-cost condition. It still leaves compulsory Target+DSpark/prefill work, true mixed resource capacity, collective/data dependency DAG and a finite Product E2E upper endpoint unresolved. Run99 Current Formal remains571.681tok/s. The next Bound gate is a same-cycle original FULL Graph correlation of Target/DSpark GMM route, access/cache state and downstream all-rank join; existing Run246–249 and Run121 assets narrow instrumentation. See `evidence/20260927_loop075_bound/run365/findings.md`, `run360/analysis.json`, `run363/bound_calibration_v3_3.json`, PK-031.

## Loop076 Resource/Scheduling evidence (2026-09-27)

Matched original FULL Graph cycle64/65 Target route→GMM task counters closes a current-work attribution, not compulsory HBM: 141.029–170.020 GFLOP/rank-cycle GMM standard arithmetic, 8.946–9.576GB active packed weights, median1.0711 task read/active-footprint ratio. Fixed Target dense170.204 and separately sourced Compressor58.385 GFLOP/rank-cycle remain partial. Corrected Draft dense is28.8053 GFLOP/rank-cycle; Host scope excluded some queued device tail tasks. Instrumented Target timing is perturbed by hot-path D2H/sync, so no Scheduling-aware or Product numeric ceiling is promoted. Priority is full compulsory work and unperturbed resource-constrained DAG, then Current→Bound Gap ranking. Evidence Run368/370/372/373, PK-032.

## Loop077 Run391 HCCL ABI closure

The ordered current Target chain now has a source/ABI-consistent tensor ledger: RS87, AG135, A2A43; API input115.10784MB and output145.342464MB per rank-cycle. It closes the previous `count` convention uncertainty for the current schedule, while compulsory logical bytes and physical link service remain open. Direct actual DSA CP path, embedding zero mask and derived router outputs prevent treating this as an architecture-independent communication floor. Run391 findings and PK-033 contain the exact 265-task map.

## Loop077 Run393 measurement validity correction

A terminated outer `docker exec` left Run386s inner health-wait/bench shell alive. When Run389 service became healthy it sent another60 requests. Run389 logged120 POST, and saved Run386/389 client windows overlap almost exactly with combined inflight24. Run389/390 event calculations are numerically valid for the contaminated load but **INVALID for frozen c12 Current or Scheduling Bound**; the initial TaskCtl pass statuses were corrected, and V3.7 Run392 was rejected before promotion. Run394 repeats after a no-orphan/health000/source-SHA preflight and enforces exactly60 POST. Run393 is the forensic evidence.

## Loop077 Run397 conditional retained-row work

Current padded Target routed GMM is1,246.614GFLOP/TP8-cycle; 12.985565GFLOP per candidate row under the conventional six-expert algorithm. Run397 converts ten actual route histograms and Runtime-clipped useful counts into broad retained-row expert-union packed-footprint relaxations, with strict per-layer count and Run380 footprint checks. This quantifies a missing Algorithm/Resource numerator, not attainable HBM or wall-time reduction. Next capture per-token route identity, retained mask and Draft live rows in one original cycle before estimating a new architecture or finite Hardware/Scheduling Bound.

## Loop077 Run394–396 clean Current envelope

Clean exact60-POST, five-cohort original FULL Graph event capture replaces invalid Run389/390 timing. Across 40 same-rank cycle64→65 pairs, current-stream median 56.548 ms; across 80 selected rank-cycles, Target-label median 49.465 ms and proposer-label median 6.397 ms. This narrows the observed Current scheduling path under sparse instrumentation. It does not close side-stream or cross-rank critical path, compulsory work/traffic, mixed resource capacity or attainable Product ceiling. V3.7 retains null finite upper endpoints and Formal Current571.681 tok/s. See Run394–396 evidence.

Run99 also includes a valid one-run 612.962tok/s observation (80.188s), now explicit in V3.7. It constrains claims about a hard Product ceiling but does not replace the formal571.681tok/s median or establish repeatability.

## Loop078 dependency structure and unresolved storage order

Run400 independently reviewed source inventory records a normal cycle c count-copy side stream that can overlap next Target and normally gates c+1 Draft Host mirrors, plus same-cycle parking's conditional join. It also records an unproved num_sampled D2H-read versus next overwrite constraint. Run401 is a measurement design only; no ordering margin has been observed. Run402 V3.8 therefore improves the structural Scheduling model without a new numerical floor or Product ceiling.

## Loop078 Run403–406 route numerator correction (2026-09-27)

Clean Run403 48+12 exact1024 c12 diagnostic produced 60 POSTs, five cohorts × eight ranks, 40 Target/Draft route captures and 40 passing FULL Graph reports; six borrowed sources were restored and service stopped. Astra independently accepts route/count/ownership arithmetic but finds cross-layer row identity incomplete under active FlashComm1/DSA CP/chunk paths. Target current standard routed GMM is1246.614GFLOP and its selected active packed weight set63.141–70.414GB/TP8-cycle. The retained-prefix 42.203–58.246GB selected set is conditional on unproven all-layer slot-major mapping and clairvoyant rejection knowledge; it is neither compulsory HBM nor an executable saving. Draft84-query partial routed estimate is format-conditional and excludes much of Draft. Run406 V3.9 therefore leaves all finite Algorithm/Hardware/Scheduling/Product ceilings UNKNOWN. Formal Current stays Run99 median571.681tok/s; Run403 diagnostic TPS is excluded. Next close actual row labels and Run401 copy/overwrite dependency before deriving a numeric interval. See `evidence/20260927_loop078_bound/run405/findings.md` and PK-038.

## Loop078 Run407 row-identity closure decision

Independent Astra source audit finds ordered FlashComm1 gather/pad/chunk and DSA CP query/head all-to-all maps that could, with actual branch/group/Graph certificates and trusted native row-order contracts, prove all43 Target router row identities without a redundant per-layer device label collective. Current Run403 lacks those dynamic branch and compiled/native dispatch records, so the retained expert union remains conditional. The next Resource numerator measurement should capture the actual target_logits_indices, embedding/sequence-parallel/FlashComm/DSA CP/prepare paths and group rank order once per graph entry, bind them to selected replays, then add shadow row labels only at unresolved transforms. This is a Bound proof task, not a local speed candidate. See `evidence/20260927_loop078_bound/run407/design.md`.

## Loop078 Run414–415 communication cut qualification

Astra independently classified the265 ordered Target HCCL calls in Run391. Current per-rank API tensor inventory is115.108MB input/145.342MB output; neither is physical or compulsory wire traffic. Under an explicitly restrictive fixed-dense, opaque-payload, no-reuse/recompute/compression/placement-change scenario, a logical 4|4 rank cut carries306.659328MB/Target-cycle both directions combined. This is only a conditional transport census. Embedding zeros, derivable router logits, an unused MTP stash and alternative output/ownership representations disprove promotion of all265 current materializations to a model-semantic minimum. Run246 all8 transport exports are empty; zeros in transit fields mean unavailable measurement. The saved HCCS adjacency does not establish cut capacity. V3.10 stores this scenario with no compulsory communication bytes, latency floor or Product ceiling. The next minimal communication measurement must first prove native peer/bytes/path export on one actual DSA A2A, then derive physical capacity and legal information cuts. See `evidence/20260927_loop078_bound/run414/astra_comm_resource_review.md` and PK-039.

## Loop078 Run412 local scheduling closure and next Bound gate

B-arm 40/40 same-device read-complete before next overwrite-start, median direct margin53.747 ms, with first Host count/sequence-mirror consumers recorded. Draft/DSA downstream and parking/terminal paths remain open. A1 Run411 startup pinned Host OOM; Run417 passed after container reset, which invalidates A/A overhead comparison. No Current→Bound wall-time saving or finite ceiling follows. Resource priority: prove all43 router row maps and external output ledger; Hardware priority: distinguish physical HBM/link traffic and valid capacity upper caps; Scheduling priority: all8 typed Graph/HCCL and actual Draft joins. Evidence Run412/416/418. Formal Current571.681tok/s.

## Loop079 Run421–425 terminal output ledger

The frozen diagnostic has 584 Scheduler-generated tokens before terminal bulk processing, so its61,440 Runtime-retained IDs are not all externally newly admitted:60,856 were admitted and584 length-clipped. The handoff-time already-generated and API-published counts remain distinct unknowns. Resource ceiling requires a matched necessary W-minus / maximum C-plus certificate, not a measured kernel peak; Run423 found the smallest candidate but no certified numeric pair. Scheduling saving and full router identity remain unproved. Next: request-correlated handoff+publication ledger, row/Graph/native certificate, and 910B3 SKU capacity binding. Evidence Run421–425/PK-042.

## Loop079 Run426–429 reviewed Host cutoffs (2026-09-27)

Run427 clean but Host-instrumented exact60 48+12 c12 diagnostic and Run428 independent review close a scoped Host chain: 376 ordered Scheduler→OutputProcessor receive→queue→Chat consumption segments match in length/SHA256 and causal order. At five cohort H_probe envelopes, aggregate Scheduler committed G=[795,796] and Chat raw consumed A=[674,796]; generated-output yield Y=[213,268]. At least 50 requests completed at least 213 nonempty reasoning-delta yields before their cohort's earliest H_probe. H_probe precedes Runtime.run and is not device-ready. These counts are neither device-completed D(H), literal raw-token publication, ASGI send nor client receipt; Run427 counts cannot transfer to Run403/Run99. V3.13 retains all finite Algorithm/Hardware/Scheduling/Product endpoints null and formal Current571.681tok/s. Next close all43 router row identity and a genuine 910B3 W-minus/C-plus resource certificate. See `evidence/20260927_loop079_identity/run429/findings.md` and `evidence/20260927_loop079_identity/run429/bound_calibration_v3_13.json`.

## Loop079 V3.21 Current→Bound map

Current Formal is571.681tok/s. Run487/488 establishes one **instrumented selected** Target path: existing pre-replay sync and handoff → FULL96 replay-stream envelope (median45.986019ms) → four BF16 hidden/aux native TP8 AllGathers → sampled hidden. It measures current operator order, not mathematically compulsory boundaries. After graph-update is bound to a Python callable and configured stream102, but effective backend/device work and private-stream join are unproved. This is the largest measured local interval in its selected chain, not necessarily the largest removable Gap. No additive comparison with Run394 Target label or Run477 terminal interval is valid.

Resource/Hardware remains blocked on matching in-window W-minus and authoritative aggregate 910B3 C-plus; FULL Graph current traffic and isolated attained rates are not compulsory/capacity maxima. Scheduling remains blocked on actual graph output producers/child joins, all-rank arrival and feasible overlap/resource contention. Product E2E remains blocked on necessary-path translation, legal schedule and repeated formal correctness/performance. Run492/493 proves installed graph JSON can expose event joins, enabling a focused same-process production dependency certificate. V3.21 has all finite endpoints null. See Run487/488/490 and `ACHIEVABLE_BOUND.md`.

## Loop079 Run495–497 Current→Bound update

Run497 narrows the *proof target* for one conditional Resource subset via the source identity sampled first token = first Target argmax. It is not a measured resource cost or a formal retained-token witness. W-minus freshness/external lineage and matching C-plus maximum remain unclosed. Run496 blocks Run494 production graph capture until source hash, task schema and actual selected update-backend certificates are repaired. Thus the Run487 selected FULL replay median45.986019ms remains an instrumented local Current interval, not a removable gap or Scheduling floor. Largest decision-relevant unknowns remain graph producer/join and full DAG/resource overlap, plus compulsory-work/capacity pairing. Formal Current571.681tok/s; all finite endpoints unknown.

## Loop079 V3.22 Current→Bound update

Current Formal:571.681tok/s. Run502/503 proves a source-bound selected FULL96 Target graph and actual MLA after-update callable in a clean 48+12 diagnostic, plus a conditional internal event DAG. The selected update's **effective iteration count**, same-generation ExternalEvent/handle, four hidden/aux typed last writers and model-terminal→caller R1 continuation remain unknown. Run487's 45.986019ms instrumented replay envelope belongs to a separate acquisition and is not promoted to a Run502 cost, necessary scheduling floor or removable Product gap.

Resource/Hardware: Run497 first-position lemma narrows one candidate W-minus, but formal-window external retention/fresh work and authoritative aggregate C-plus remain missing. Graph task counts do not measure compulsory traffic or capacity. Scheduling: source permits same-replay private MLA update and records no proof that it executed; typed producer/event/terminal join and all-rank mixed-service measurements are next. Product: no legal attained schedule or repeated formal E2E improvement has been shown by this diagnostic. V3.22/Run505 keep all finite endpoints null. The next experiment is chosen to close the largest structural Scheduling uncertainty while Resource certification continues in parallel.

## Loop079 Run507–515: MLA update count, interval capacity and terminal source (2026-09-27)

Run507 is INVALID: its workload completed, but an absolute-path validator import failed and final admission returned exit1. The fresh Run508 controller passed exact48 warmup +12 diagnostic at c12/1024, with 60 HTTP200 and exact-length client outputs, 40 all8 capture/Runtime records, eight Graph dumps and complete stop/restore/SHA/final gates. Astra Run511 independently checked 172 inputs and accepted a narrow selected-call fact: for each of the 40 selected FULL96 Target calls, the actual `AscendMLAImpl.update_graph_params` has 170 attention keys but empty `attn_params`, handle and ExternalEvent lists. The four-way zip therefore has zero iterations and no source-inferred per-key task-update/FIA/event-record loop. The method still enters stream102 context. This does not prove all private work absent, eliminate any graph wait, supply timing savings or change formal correctness/TPS. Run507 raw counts are not used.

Run509 independently rechecked the strict Resource certificate: no positive formal-window W-minus is paired with a matching exact-board C-plus. The one first-position layer42 BF16 `wo_a` group is only a conventional-dense-class candidate until request/generation/publication/freshness and permitted evaluation semantics are proved. A capacity rate must bound cumulative service over the formal interval; if it applies only with boundary allowance B, the floor is `max(0, W_minus-B)/C_plus` and positive only when `W_minus>B`. Current/rated clock or attained GEMM/HCCL service cannot substitute for the exact 910B3 Bin6/firmware upper guarantee. Resource and Product numerical endpoints remain null.

Run510 reviewed official CANN9.1 release runtime source against installed B243 files. The general model-completion mechanism records an endGraphNotify on the model stream and waits on the execution stream after asynchronous submission. Exporter placeholder labels consume no argument bytes, placing Run502's ReduceMean aux address at raw byte24; exporter `ts/dur` are synthetic. The exact installed backend/build branch, native notify object/generation, MIX typed output role and exhaustive four-output last writers remain unproved. This narrows the Scheduling identity search but does not bind Run502/508 terminal notification to caller R1. No local instrumented interval has been transferred to Product.

V3.23 Run512 and Astra Run513 encode the zero-loop result and Resource interval-service/B gate; V3.24 Run514 adds the general terminal/ABI source finding at its distinct scope. All 19 proof nodes and finite Algorithm/Resource, Hardware/Resource, Scheduling/Execution and Product E2E endpoints remain unpromoted. Formal Current remains **571.681 tok/s**; the numerical distance to a credible overall limit is still not certified. Next Bound work is the exact loaded B243/end-notify identity and compiled MIX launch ABI plus semantic four-output producer/overlap join, while independently obtaining formal48 fresh retained W-minus and exact-board interval C-plus/B. Only after necessary DAG edges and mixed all8 resource service are closed should these become a Scheduling-aware/Product interval. See Run507/508 findings, Run509/510/511 reviews, Run512/514 model JSON and PK-058/059.

## Run516–525 Bound-first update (2026-09-27)

Accepted formal Run99 saved artifacts do not contain a complete fresh retained Target-to-client witness; formal Current stays571.681 tok/s. Static all43×8 BF16 `wo_a` work is2.885681152G conventional operations per fresh required Target row in the stated dense class; routed MoE adds12.985565184G separate W4A8 GEMM-equivalent units. Neither observed cycles nor49,152 outputs supply the unknown formal fresh multiplier F. Checkpoint storage and FULL Graph counters are not compulsory HBM. V3.26 leaves all numerical Bound endpoints null. Independent Scheduling review ranks a correlated full48 product/work ledger ahead of another broad graph dump; only its selected frontier should then receive all8 mixed-service and typed dependency measurement.

## Run526–531 scope (2026-09-27)

Warmup already generated the same48 prompts to1024, so measured-pass launches alone cannot establish all-architecture compulsory work. V3.27 tracks unrestricted, online-inference and ordinary dense BF16 classes separately, with fresh `F[layer,group]` still null. The next same-run full48+48 ledger needs request/generation, raw output, Scheduler G, Runtime q/R and cache/prefill/seed/publication boundaries. Client SSE acquisition passed independent source/fake-transport checks; server and all8 admission remain pending. Current571.681 tok/s and all finite Bound endpoints are unchanged.

## Bound-first checkpoint Run542–548 (2026-09-27)

Formal Current remains Run99 median **571.681 tok/s**. Run542 completed frozen warmup48+measured48 c12 Host acquisition but its original controller failed on a validator enum-string bug; Run543 corrected offline replay and independent Run544 admitted the retained raw ledger as **posthoc diagnostic** without relabeling the failed controller or diagnostic603.648 tok/s as formal performance. Stop, all8 idle and five borrowed source restorations passed.

Run545/546 independently close the measured same-run output accounting: Runtime sampled q49,462, retained R49,152, Scheduler ordinary before terminal bulk G401, terminal Runtime bulk admitted48,751, and all48 first Runtime tokens retained. G at rank0 handoff is400; one ordinary append happened later. OutputProcessor reports all48 cached32,768 prompt tokens and residual computed83–85, total3,987; these Host fields do not prove device-ready or necessary physical work. Rank0 cohort cycles289/299/322/308 total1,218. Conditional same-trajectory cycle relaxations508/510/512/1035/1206 omit release, preparation and resource contention; none is a Product time floor. Instrumented Host spans overlap and are not removable wall-time sums.

V3.29 Run547 and independent Run548 retain **all finite Algorithm/Resource, Hardware/Resource, Scheduling/Execution and Product E2E endpoints null**, including fresh Target F and Current→credible-limit distance. The next high-value Bound acquisition is all8 cached residual preparation→seed/KV device-ready→first useful Target, followed by compatible mixed Target/DSpark resource service and legal c12 release. Strict compulsory work/traffic and matching certified exact-board cumulative capacity C⁺/B continue in parallel. See `evidence/20260927_loop079_identity/run546/findings.md`, `evidence/20260927_loop080_bound/run545/astra_bound_review.md`, `run547/findings.md` and `run548/astra_v3_29_review.md`.

## Run549–553: next Bound frontier and client/source preflight

Astra Run550 accepted the next acquisition direction with live preflight gates: measured cohort1→2 c12 release, prior-slot retained-output completion and safe state/slot lifetime, cached residual83–85 prompt-token preparation, ordinary sample/DSpark seed/KV readiness and first useful Target on all8. The instrumented acquisition needs matched no-mark A0/A1 controls before its durations can represent unmarked Current. A missing side-stream completion leaves that edge null without discarding valid Host lineage. Conditional mixed-service B can calibrate Engineering/Scheduling only after the legal release window and shapes are known; it is not a hardware maximum. Historical R20/R28/R31 are scoped priors (PK-065/066), not current verdicts.

Run551/552 source-only client component now records semaphore attempt/acquired, SSE DONE, stream end and post-release brackets while preserving `Semaphore(12)` and the frozen request body. Fake48 request c12 and17 transport regression cases pass; Astra independently verified actual Semaphore calls lie within markers. It yields conservative eligible release sets, **not unique permit-parent edges**. No live service or NPU run has used it. Run553 source audit finds async scheduling can keep draft seeds device-resident and conditionally skip D2H copy; the successor seed collector must dynamically bind the actual GPU scatter/CPU fallback, side-stream event and first Target input generation. All finite strict Bound endpoints and the numerical Current→limit gap remain unknown; Formal Current remains571.681 tok/s.

## Run558/559 fixed-work Current→Bound map

Run558's admitted selected cohort5 slot5 crosses a conservative full Runtime1024 history count at cycle188/299. All8 rank-local sparse copy markers bracket cycles184–192; the 111 following current cycles are **not** removable time. Actual successor Host path begins with an 83-token residual upload and later async scatter branch. The first Runtime Target event intervals do not cover the full preceding residual/seed producer path or all8 makespan. Client c12 report has eligible release sets but zero unique permit-parent certificates. V3.30 keeps finite Resource/Hardware, Scheduling/Execution and Product endpoints and Current→credible-limit distance null; Current Formal is571.681tok/s. The active gap question is minimum execution time for **the same DSpark7 acceptance, output and model work**, including whether a completed slot can legally publish/release during ongoing old-cohort work. A0/A1, producer joins and all8 mixed contention are required before converting marked intervals to an attainable schedule. See Run558/559 findings.

## Run560 active gap definition

Current Formal stays571.681tok/s. The active Current→Bound gap is **execution time for fixed logical DSpark7 acceptance/output/work**, not a reduction in cycle count or a higher acceptance rate. V3.31 separates semantic `W₀` from current physical calls and permits fusion/layout/Graph/overlap changes. Run558’s measured ledger is diagnostic; the formal `W₀` identity, compulsory work/traffic, exact-board C⁺/B, complete Scheduling DAG, legal c12 release and attainable all8 mixed service remain uncertified. All finite Product and Hardware/Scheduling endpoints are null. Next certify one normal early response/release while old work and KV/state continue, then calibrate mixed resource service under fixed `W₀`. See Run560 findings.

## Run561–562 fixed-work Scheduling gate

The first dynamically completed slot is a **candidate** c12 release window, not a measured removable interval. Run562's source-only pause component now preserves the original loop's pre-parking termination decision and synchronizes eight-rank branch decisions each cycle; CPU tests and Astra review pass. Its sampled/count export can test part of `W₀`, but semantic Target/Draft keys and KV/state, real early API delivery, actual client permit/ADD, exactly-once Scheduler settlement and all8 exception handling remain absent. Current Formal571.681tok/s; numeric Current→credible-limit distance and all finite Bound endpoints remain unknown. Next measurement priority remains the matched fixed-`W₀` legal-arrival/frontier DAG, followed by compatible all8 mixed-service capacity; Resource compulsory work/traffic and exact-board C⁺/B continue in parallel.

## Run563 Scheduling preflight

The no-publication A0/B/A1 control is a necessary semantic screen before the more complex Run561 early-publication transaction. Its approved source-only patch preserves one ordinary final Scheduler/API result and restores fixed source hashes in temp-copy tests. A valid live run would test the **segmented plus per-cycle consensus bundle** against matched entry semantics, not isolate pause latency or establish a Scheduling bound. No live result yet; Resource/Hardware C⁺/B and full critical path remain separate open gates. Current571.681tok/s and numerical gap remain unknown.

## Run566 fixed-work diagnostic mapping

A0/B/A1 complete service and all8 self-consistency gates pass, but their matched cohort entries already differ at cycle0 in computed positions, Target input and Draft seed. B's descriptive longer wall and different cycles are therefore not a removable Runtime gap for frozen W₀. The current largest **uncertainty** is legal same-state Scheduling: dynamic first-completion parking→pause noninterference→normal API publication/c12 release→successor ADD and mixed resource service. Use a single natural S* with complete model-readable state/ownership checks; do not rerun cross-service A/B as a causal pause timing test. Resource compulsory work/traffic and matching exact-board C⁺/B remain independent open gaps.

## Run568/569 Current→Bound classification

Run568 eliminates one narrow same-S* pause mutation concern: the 60 enumerated tensors and captured scalars/identities were unchanged on all8 at cohort5 cycle187 slot4. It does **not** measure removable execution time or certify full KV/fixed-W₀ continuation, normal API release, successor ADD or all8 mixed service. The legal schedule and Product E2E gap remain unknown.

Run569 separates per-fresh-row BF16 `wo_a` and W4A8 routed MoE formulas, current 96-row geometry, checkpoint operand footprints, and 16 instrumented current counter windows. The 18.965GB-class current read cannot be promoted to compulsory traffic; HCCL link bytes remain unavailable. The largest Bound uncertainty is now broad formal W₀ required work plus exact-board cumulative capacity/residency and mixed resource service/critical-path timing. Formal Current571.681tok/s; numerical Current→credible-limit gap and every finite strict endpoint remain null.

## Run570 observed parked-row distinction

Run566 A0 all8 row census separates 98,496 active-slot rows from 17,280 parked-slot rows within 115,776 current physical Target inputs/rank over four diagnostic cohorts. The parked share 14.92537% is a current geometry observation. Whether variable-active-row execution reduces compute/HBM or critical-path wall requires compatible Graph/layout, same fixed W₀ and all8 mixed-resource measurement; no such gain or Product Bound is inferred. The Resource gap remains dominated by formal W₀ necessary-work/traffic and matching C⁺/B uncertainty.

## Run571 empirical service versus strict capacity

The source-pinned matrix has 18 isolated service priors and five instrumented Current rows, with per-row shape, source SHA and transfer restrictions. The GMM2 bank8 +8.037% median effect (8/8 strict) confirms working-set sensitivity only under synthetic-zero independent Graph timing. Official isolated HCCL Test t1 values and Run395/487 current intervals are different timing classes. No row supports a strict cumulative C⁺/B or removable Product wall-time claim; do not sum cross-run medians. The dominant uncertainty remains same-W₀ real-data/residency-matched all8 mixed resource service, while the independent strict Resource track still lacks complete compulsory work/traffic and matching exact-board capacity. All finite Bound endpoints remain null.

## Run572 exported task intervals, current diagnostic only

Run246/247 latest FULL Graph windows contain paired HCCL `AivKernel` execution rows and `hcom_*` pseudo envelopes (265 of each per rank Target scope). A prior category-only reading would fabricate ~10ms of apparent communication with no AI; the corrected rank-local interval reducer finds zero pseudo-envelope-only time because exported AI task intervals cover it. This does not prove link work is hidden or no communication Gap exists. The 5.250ms median no-exported-task coverage gap is not removable device idle. Cross-rank clock/makespan is unvalidated. Scheduling priority is a producer/arrival/HCCL/device-completion DAG and same-W₀ mixed service, not a sum of profiler categories. No numeric Bound update.

## Run573 Resource numerator identity gate

Run403 routing/ownership can become a semantic active/parked expert-set census only after each router row is joined to the actual Target candidate identity in the same admitted trajectory. Run573's source index and synthetic primitive map rejects wrong DSA/dispatch/layer maps but does not establish the actual 43-layer composition. A diagnostic must capture production handoff, first post-park Graph replay, rank/pad/branch maps, all43 router inputs and Target/Draft packed-weight metadata. This is a Resource identity gate, not a clairvoyant retained-prefix optimization or a Bound time. All finite endpoints stay null.

## Run574 source-level native attention row order

Reviewed SCFA/SWA TND `FLASH_DECODE=0` source writes attention output at the Query token/head offset; package objects match build objects and the container imports the audited source tree. This is a conditional local mapping only. Actual loaded B1-targeted binary on 910B3, tiling/metadata, Graph generation, DSA TP8 A2A/wo_b and all43 router rows are still open. Resource numerator and Current→credible-limit numerical gap remain unknown; next evidence should bind real loaded branch and one same-trajectory first-post-parking row/weight witness. See PK-075 and Run574.

## Run575 Graph/package branch evidence

An eight-rank Run502 Graph→Run574 package join identifies 344 selected BF16 sparse-attention tasks, 43/rank, with key distribution SWA2/CFA20/SCFA21 and static FD0/TND/PA_ND branch under the installed CANN9.1 encoder. This supports the native Query-row preservation hypothesis for that Graph, but dynamic prefix/head/group, actual loaded object bytes and all-layer router provenance still need an admitted same-trajectory witness. It is structural Bound calibration, not a latency or savings measurement. See PK-076.

## Run576 conditional Resource geometry

The live first-post-park census joins all8 Target/Draft route ownership and W4A8 entry operand geometry to one diagnostic cohort. Selected-slice arithmetic (~8.36–8.89GB Target plus~0.29–0.48GB Draft per rank) is neither actual GMM traffic nor compulsory traffic; it cannot be compared causally to Run247–249 GMM counter reads across trajectories. It reduces uncertainty about the working set a matched Engineering service test should reproduce. It does not resolve formal W₀ necessary bytes, attainable all8 mixed capacity, communication overlap or a legal Scheduling makespan. Current571.681tok/s; all strict numerical Bound intervals and Current→credible-limit distance remain null. See PK-077 and Run576.

## Run577 Engineering service calibration

The terminal eight-rank 43-layer real-weight, synthetic-activation GMM Graph provides one attained isolated service point: slowest rank median10.1311698ms/replay (20 samples/rank). It addresses production weight residency and cross-layer working-set transfer from Run571, but lacks physical traffic and concurrent Target/Draft/KV/HCCL contention. A 156.482ms common Host submission window is not calibrated device overlap. This service time cannot be summed with profiler stage intervals or subtracted from Product wall-time. Strict Resource capacity C⁺/B, Scheduling critical path and numeric Current→credible-limit distance remain unknown. See PK-078.

## Run578 partial MemoryAccess evidence

CANN9.1 offline recovery from a failed online profiler produced two exact43-pair real-weight/private-synthetic-input GMM Graph replay CSVs per rank. Eight-rank per-replay counter-reported main-memory read8.878–9.504GB, write0.197–0.227GB; cross-rank medians9.156191/0.213860GB. A0 was not persisted and A1 not run, so this is counter-only evidence, not attained bandwidth. The traffic is neither certified physical HBM nor compulsory W₀. V3.34 strict Resource/Scheduling/Product endpoints and numeric Current→limit gap stay null; Formal Current571.681tok/s. Next bound discriminator: same-work GMM×HCCL four-condition mixed service plus production dependency witness. See PK-079/Run578.

## Run579–580 fixed-work mixed-service result

The all8 independent-ready production-weight GMM Graph × full265 current-order TP8 HCCL Graph diagnostic passes frozen client/Runtime and fresh output gates. Run579 same-run unprofiled slowest-rank medians are GMM10.266ms, HCCL4.180ms, serial14.402ms, concurrent16.205ms; Run580 repeats 10.312/4.282/14.573/16.242ms. Thus whole-chain simultaneous replay is slower than explicit serial for this fixture. It must not be converted into a Product negative KEEP or a general overlap prohibition.

Run580's copied Level0 trace attributes two full GMM and HCCL Graph replays per rank and excludes profiler barriers. Concurrent first HCCL task is only0.37–0.42ms after GMM first, but HCCL task envelope rises to15.10–15.35ms from serial4.07–4.44ms. Matched first84/86 collective task-time medians rise from13.34–14.36µs to144.73–146.40µs; the post-GMM matched tail remains near13–14µs. The task envelopes coexist; Level0 cannot apportion actual AIV/HBM/link occupancy, internal wait or rank arrival. Next select the largest *legal* scheduling gap only after a same-W₀ production dependency DAG, and continue compulsory W-minus/traffic plus exact-board cumulative C⁺/B separately. Formal Current571.681tok/s; finite Resource/Scheduling/Product Bound and numeric Current→limit gap remain unknown. See V3.36, PK-080/081 and Run579/580 findings.

## Run581 source gate for a production TP dependency slice

Read-only installed-source and prior-evidence audit selected layer0 DSA `wo_b` partial result → TP collective → attention output copy → first HC-post as one narrow fixed-DSpark7 Scheduling DAG cut. Thirteen source/prior files are SHA-pinned; nine source predicates and Run378's 265-call RS87/AG135/A2A43 inventory passed. The loaded `wo_b.custom_op`, actual SequenceRowParallel/MMRS/OTP branch and native collective ordinal are **not** recorded by existing evidence. Run378 ordinal3 is a candidate, not an identity certificate. Run502 exporter `ts/dur` and address text cannot supply readiness time or typed last writer. Historical R20/R27 remain prefill/EP priors; Run292–293 already cover a different layer2 hidden-AG/Q window and are not repeated. Next admit actual selected FULL96 all8 branch and typed producer/collective/copy/HC-post storage lineage, then rank-local ready/completion timing and all8 mixed contention. This gate adds no numeric bound or Product saving. Current Formal571.681tok/s; all strict finite Resource/Hardware, Scheduling/Execution and Product endpoints, plus Current→credible-limit distance, remain null. See Run581 findings and PK-082.

## Run582–585 fixed-work Bound map

Run583 all8 FULL96 capture resolves the actual layer0 `wo_b` implementation branch and capture-time partial→RS-result→attention-destination object lineage. Its `[96,4096]` partial and `[12,4096]` result are typed geometry, not compulsory link bytes. The first replay was submitted, not device-completed, and capture rows are not joined to unique measured requests. Run582's compiled hook failure is invalid instrumentation. Native collective, actual last writer, HC-post input and timing remain the Scheduling frontier; no Product gain is implied.

Run584 independent Resource review identifies a narrower positive W⁻ acquisition: one logical, measured Target evaluation→actual BF16 `wo_a` group-row under declared cache/reuse and ordinary dense online assumptions. Run585 is source-only preflight; no W⁻ is yet promoted. Strict exact-board cumulative `C⁺/B` remains missing, so the finite Resource/Scheduling/Product Bound and numerical Current→limit gap remain unknown. Formal Current571.681 tok/s. See PK-083/084 and Run582–585 evidence.

## Run586–588 native DAG mapping method

Run586 recovers a historical all8 static MatMul→RS→copy→HC-post event-order candidate from Run502, but synthetic dump times and reused Graph-pool addresses cannot bind Run583 or quantify a scheduling window. Run587 rejects a direct Python TLS tagging route on current queue1. Run588 establishes a source-compatible queue1 C++ sidecar tagging method on an isolated Graph, including native `MEMCPY_ASYNC`. Production applicability is the Run589 question; no Current→Bound time shrinks yet. Formal Current571.681 tok/s; strict finite endpoints stay null. See PK-085/086.

## Run589 same-run static DAG and Resource measurement gate

The all8 measured cohort5 FULL96 Graph now binds source-labeled layer0 BF16 partial MatMul→event-paired RS→source-labeled attention copy to native Model45 tasks in one capture generation. It is a scoped dependency identity, not measured node timing or a proven HC-post typed consumer; copy ABI bytes are not independently exported. The independent Resource review found a possible 500 ms cached HCCS statistics path in installed device-side source, with actual booted-image/API identity still open. This blocks short-window physical-link byte inference until freshness is certified. Next bound-calibrating experiments are a perturbation-controlled production ready/completion timing slice, scoped fixed-W₀ fresh work, and exact-board HCCS measurement semantics. Formal Current571.681 tok/s; strict numerical Bound endpoints and Current→limit distance remain null. See PK-087 and Run589 evidence.

## Run590 HCCS inventory scope

The supported current CLI yields all8 seven-link static HCCS records with lane mode4 and standard224 Gb/s, but no remote peer/cut map, maximum serializer rate, loaded counter implementation or sample freshness. Run590 therefore informs future physical communication measurement design without assigning necessary bytes, overlap or a Resource ceiling. The largest unresolved Bound pieces remain fixed-`W₀` compulsory work/traffic plus matching exact-board cumulative capacity, and a timed all8 production dependency DAG. Current571.681 tok/s; strict numerical endpoints remain null. See PK-088.

## Run591 consumer dependency qualification

Pinned source establishes an alias-only attention output→HC-post typed-input read edge, and Run589 all8 native argument words corroborate it. Compiled ACL descriptor, full last-writer census and timing remain open. The immediate Scheduling measurement target is the narrower partial→RS→copy cut with a validated Level0 time source and same-generation task correlation; causal savings still require unmarked/fixed-`W₀` controls and formal E2E. No finite Bound or numerical Current→limit gap yet. See PK-089.

## Run592 historical native timing calibration

Existing Run246 Level1 trace supplies a rank-local Model45 partial→RS→SDMA copy historical instrumented interval:31.7695µs median start-to-copy-end over16 all8 occurrences. The copy→HC-post start gap is0.020–0.0405µs in that trace. This eliminates a speculative large idle-gap hypothesis for one profiled adjacency, but missing direct cohort/request join, cross-run semantic identity and profiler perturbation prevent unmarked Current or Bound promotion. A fresh narrow profile is deferred until it changes a decision; broader fixed-work and full critical-path certificates lead. Current571.681 tok/s; strict numeric endpoints remain null. See PK-090.

## Run593–596 Bound uncertainty update

Run593 broad historical profiler coverage reveals non-Graph Runtime/DSpark work and phase-boundary tasks still requiring launch/consumer ownership. Run594/596 verify a compact fixed-algorithm Target lineage basis on diagnostic A0 only. Run595 narrows formal Run99 **conditional current Target-8 active rows** to 79,968–116,608 / 78,360–114,672 / 80,856–115,144 for the three repeats; prior 49,672 lower bounds were weaker relaxations. This contracts one workload-cardinality unknown but not compulsory fresh work, physical traffic, mixed-service capacity or all8 critical path. Formal Current571.681 tok/s; strict numerical Bound endpoints and Current→credible-limit gap stay null. Next acquire one complete new diagnostic W₀ lineage and explicit parking/ordinary joins, then prioritize the missing Resource and Scheduling certificates by interval sensitivity. See PK-091–094.

## Run597–598 fixed-work ledger and Bound priority

One new guarded measured W₀ has a validated all-cycle accepted/draft/Host-park basis and sparse actual Target/DSpark inputs, with full client→Runtime joins and all8 restoration. It records1200 cycles, 99,976 active-class /115,200 submitted Target-8 rows and source-static Q7 conditional Draft geometry. This closes a new-trajectory cardinality/identity uncertainty only; current rows cannot become compulsory fresh work or a time Bound. The next Bound gate is semantic necessity plus actual mixed resource capacity and a typed, timed all8 Target/Draft/KV/Host dependency cut. Historical R21 and Run579/580 warn that source-independent overlap can lose to contention. Formal Current571.681 tok/s; strict numerical Resource/Scheduling/Product endpoints remain null. See PK-095/096.

## Run599 dependency frontier and Product boundary priority

Source-typed Target aux→Draft context KV can precede acceptance-dependent Draft query preparation, but no ready/completion or legal mixed-overlap time is measured. The49,523 Run597 active greedy-prefix support positions, and formal Run99 staged totals49,470/49,508/49,471, are semantic support/accounting columns only; they are not compulsory fresh Target work. Existing Run153/238/239/241/341 already account for the large Product boundary remainder and residual prompt microbatches. Next close one same-W₀ request→prefill/seed/KV/state→first Target-ready dependency with all8 timestamps and output lineage before ranking a narrow tail overlap against it. Current571.681 tok/s; finite Bound endpoints null. See PK-097.

## Run600 existing-W₀ Product preparation split

A 42-input all8 read-only Run341 reconciliation finds 2.879–3.514s cohort Host first-execute→Runtime-built envelopes and 0.594–1.253s rank-local time outside observed `_model_forward` Host spans. The latter includes nonforward Host operations and async device effects, not an idle or removable Gap. Scheduler handoff counts include possible placeholders and cannot establish delivered p_i. The next acquisition needs same-new-W₀ prefill/initial DSpark KV/device-ready and output-delivery joins; narrow Target-tail overlap remains secondary until the full Product dependency is ranked. Resource compulsory traffic and exact-board mixed C⁺/B stay open. Strict Bound endpoints remain null. See PK-098.

## Run601–603 Bound uncertainty update

One new guarded fixed-DSpark7 diagnostic W₀ closes a Product accounting edge: Scheduler actual prebulk1,017 + accepted Runtime bulk48,135 =49,152 API/client IDs, with768 async placeholders excluded. One prefix ID is parser-suppressed from SSE yield; 898 payload-associated IDs had arrived at the client before latest all8 Host Runtime-build mark. Run603 finds71 ordinary Target/proposal pairs per rank and7,283 scheduled/7,496 Target padded rows in preparation. This narrows output ownership and current preparation issuance. It does **not** establish initial seed/KV/state writer readiness, actual ordinary Graph/attention dispatch, compulsory fresh work/traffic, a removable Host remainder, legal mixed overlap or finite Bound. Run602 timing is perturbed. Source-classify ordinary dispatch and then target the full preparation producer→consumer frontier; maintain exact-board mixed Resource capacity and Runtime critical path as parallel uncertainties. Formal Current571.681tok/s; strict numeric endpoints null. See PK-099/100.

## Run604–606 Dynamic ordinary dispatch constraint

Run604 source review predicted eager ordinary DSpark and conditional Target Graph/CP paths; Run605's full dispatch failed because observer ownership followed a proposer.runner shim left by warmup, so zero Draft hooks were an instrumentation failure. Run606 corrected the owner and independently admitted one new all8 diagnostic W₀:40 ordinary pairs/rank;24 Target NONE/fallthrough with4 prefill-only/20 mixed metadata;16 pure-decode FULL replay, each with existing synchronize. Eager anchor Q7 issues5675 context and1813 query rows/rank. The Host sync sum2.712–3.195ms/rank and context→Runtime entry marker ages are observed Current only. Actual per-layer KV writes, side-stream ready/consumer generation, strict compulsory traffic and mixed all8 attainable C⁺/B are not established. Run606 observer perturbs timing and is not Run99 same-state. Formal Current571.681tok/s; strict Resource, Scheduling and Product endpoints and numerical gap remain null. See Run605/606 findings and independent Astra reviews.

## Run607–609 strict-versus-conditional Bound map

Run607 independent dual-Bound review separates three proof gaps: fixed-DSpark7 necessary fresh work/traffic `W⁻`, exact-board cumulative compute/HBM/HCCS `C⁺,B`, and an all8 legal storage-generation/stream/resource DAG. Attained microbench service is an Engineering prior, not a hardware no-faster-than cap. Run608 V3.37 rechecks all pinned Run606 current topology and keeps every strict finite endpoint and numeric Current→credible-limit gap null. Run609 selects same prior-W₀ cohort5 cycles64/65 for the next packet; current numeric context/query slot-label overlap49 per group is a layout hazard hypothesis, not a mandatory edge or saving. Include Host count-copy lag and Target66 first use; actual native identity and perturbation-controlled service remain required. Historical R09/R21 and Run579/580 inform contention risk but transfer no verdict. Formal Current571.681tok/s; acceptance/cycle trajectory stays frozen. See Run607–609 evidence and PK-103/104.

## Run610–611 profile admission and Current-versus-Bound separation

Run610 Level0 controller is INVALID because its original validator crossed CANN RAW and Python MONOTONIC clock domains. Clean stop/restore and all8 client/Basis/Product/dispatch identity allow a separate raw-only Run611 admission. Run611 retains604 SHA-verified raw files and eight offline parses; its 24 Host scope records are observer-perturbed Current. The two client anchors and profiler end anchors support only an explicitly assumed1ms mapping consistency, not a strict device-to-Host timing error. Target Host widths2.469–25.934ms and proposer50.776–59.495ms cannot be summed into a DAG floor or interpreted as removable time.

This evidence reduces acquisition/clock-method uncertainty and establishes an all8 native parsing substrate. It does **not** close exact KV generation aliasing, device-ready/consumer timing, legal overlap under contention, compulsory traffic or C⁺/B. V3.38 keeps strict Resource/Scheduling/Product endpoints and numeric Current→limit gap null; Current Formal571.681tok/s. Highest-information next step is typed same-W₀ native correlation on these copies, while Resource certification proceeds independently.

## Run612 native replay ownership frontier

Run611 parsed all8 Model45 traces now partition into three exact5,412-key native occurrences/rank using task occurrence and stream1/task0 anchors, independently reviewed. Their exported event envelopes45.102–72.209ms outlast the Target CPU scope by42.9605–70.071ms on the instrumented trace. This eliminates treating a Target Host scope as a device-completion boundary, but the last event may be zero-duration NOTIFY_RECORD and cannot quantify removable time. Each replay has260 hcom-named tasks; message bytes, mandatory collectives and actual all8 overlap are still unproved. Rank7 eager async flow count differs. V3.39 strict endpoints stay null; first type Draft KV writer/read generation and audit flow coverage, with Resource W⁻ and certified upper cumulative C⁺/B parallel.

## Run613 exported eager ownership, missing typed KV cut

All8 same-W₀ trace has exact216 CPU→native `torch_to_npu` flow joins:144 ScatterNdUpdateSk and72 SparseAttnSharedkv, each on stream47. Three initial scatters/proposer lack layer tags but match context-precompute source order; later scatter+attention pairs have actual layer43/44/45 Host scopes. This narrows current execution DAG ownership and proves Host-cycle bins are invalid on rank5. It does not prove consumed cache row generation, compulsory HBM traffic, observer-off critical-path time or legal alternative overlap. A bounded typed cache/slot/first-reader packet is the highest-information Scheduling acquisition; include one fresh semantic context projection witness to begin W⁻. Certified upper cumulative Resource C⁺/B remains parallel. V3.40 strict endpoints/gap null, Formal Current571.681tok/s.

## Run614 next uncertainty-reducing packet

The exact216 native flow joins leave actual cache allocation/content/row writer generation, consumed row domain and predecessor/successor lineage open. Resource review adds entry-result credit and fresh semantic context input as a second admission on the same bounded packet; any accepted projection is only conditional fixed-expression `W⁻`, not compulsory HBM bytes or an unrestricted algorithmic theorem. Exact-board certified upper cumulative `C⁺,B` is still unavailable; attained rates do not supply it. Run614 design is scoped PASS but not live-ready. Implement without new sync and keep the selected-row copy <=128KiB/rank, then validate same-W₀ correctness. Strict numerical Bound/gap null.

## Latest Loop081 preparation coverage — Run633–635 / V3.46 (2026-09-28)

Fixed DSpark7 acceptance/cycles/output/model work remains frozen. Formal Current Run99 is **571.681 tok/s**. Run633 uses Run606's single observer-perturbed W0 to join all8 ordinary preparation calls to exact request owners and union Host intervals; Astra independently verified32 unions/640 calls and eight negative cases. Each rank has37 own Target/proposal pairs plus3 prior-cohort carryovers. The unclassified first-own→built Host-envelope complement is0.22–0.56s/rank/cohort; zero overlap between Target/proposer Python call intervals does not prove device no-overlap.

Run634 raw frontier reconstruction identifies per rank24 own NONE,13 own FULL and3 prior FULL Target calls. Own NONE cumulative Host interval8.334–8.984s/rank, own FULL0.450–0.551s/rank. These are current diagnostic Host windows, not required device work, exposed time, an available saving or formal Run99 transfer. Astra's independent review passed8 negative challenges and ranks physical NONE readiness/HCCL dependency measurement above another selected KV row **among preparation candidates**. Runtime still dominates the formal paired arithmetic scope, so preparation is not declared the largest whole-Product gap.

V3.46 pins these facts; Astra independent review passed92 endpoint/evidence and19 certificate challenges. **Strict Resource/Hardware floor, Scheduling/Execution floor, Product E2E ceiling, conditional Engineering interval and numeric Current→credible-limit gap remain null.** Next: preflight a bounded low-overhead same-W0 all8/client packet for actual ordinary Target NONE input-ready, separate logits/aux producer-complete, related HCCL side streams/existing waits, first real consumer and Product publication across prefill-only/mixed shapes; retain OFF/ON/OFF timing transfer and full correctness ledger. Typed KV is supporting. Runs633–635 used no live service/NPU/formal E2E; tagged service remains stopped. Recovery anchors: Run633/634 findings/reviews, Run635 V3.46/review, PK-119.

## Loop081 Runtime-first physical Bound packet gate — Run636 (2026-09-28)

Astra High reviewed the seven-source SHA-pinned Run636 measurement plan and gave **SCOPED PASS / NOT LIVE-READY**. Whole-Product priority defaults to a bounded adjacent Runtime-cycle actual ready→first-consumer packet, with ordinary Target NONE preparation as fallback if Runtime closure cannot be implemented/verified with acceptable observer cost. The same Run606 W0 built→allrank existing-sync Host envelopes total roughly69.28s over four cohorts, while own Target NONE Host call sums are8.334–8.984s/rank; neither is disjoint, compulsory or removable wall. Formal Run99 Runtime spans about69s in a different W0/observer and cannot be subtracted against Run606.

The packet must separate Host submission from device readiness, main hidden→logits→acceptance from aux→DSpark, count-copy/schedule side streams, acceptance/state/Draft/commit→next Target and final staging/parking/publication. Do not add a measured-path sync. Existing current-stream events and profiler/native task endpoints need actual tensor/stream generation and consumer ownership before a Scheduling critical path claim. Run636 did not install an observer or run NPU; all numeric Bound endpoints and the Current→credible-limit gap remain null, Current Formal571.681tok/s. Next implement the reversible bounded observer and seek independent live preflight. See Run636 JSON/review and PK-120.

## Loop081 bounded observer candidate — Run637 (2026-09-28)

Run637 created a dormant `runtime/bound_observer.py` and a source-hash-pinned dry-run generator for five candidate edits; **no edit was installed into the live ModelRunner/Runtime**. Candidate ASTs compile, added textual sync/wait counts are zero, and CPU10 negative checks pass. Astra High independently ran the candidate step AST with fake operators across cycles62–65, verified the immutable cycle ordinal and 48 selected markers, and gave **source-only SCOPED PASS / NOT LIVE-READY**. The early63–65 stratum cannot park under the frozen 0→1024 limit and ≤8 tokens/cycle, but this does not cover tail/park work.

OFF/ON share an armed endpoint ledger of effective/staged counts, canonical accepted prefixes and Runtime bulk output hash, computed from existing CPU copies after the Runtime timer; raw padded hash is diagnostic only. This does not certify parked-slot raw acceptance or whole-W0 model work. Installed torch_npu2.10.0.post4 Event `elapsed_time` invokes synchronization APIs during post-Runtime extraction, so ON Product publication timing can be perturbed and must be controlled. The candidate records current/copy stream progress, not full native/HCCL completion or a numerical Bound. Next build guarded OFF/ON/OFF controller with actual backend/config, RUN_TS/arm/output identity, all8 Product/client ledger join, NPU/Graph qualification and stop/restore before any live acquisition. Formal Current571.681tok/s; all strict Resource/Scheduling/Product endpoints and numeric gap null. See Run637 findings/review and PK-121.

## Framework/Scheduling-Only V0 — Runs638–641 (2026-09-28)

The fixed DSpark7 algorithm and existing primitive implementations/costs are the separate Framework-only contract; no strict Hardware C⁺ or compulsory-traffic proof is a prerequisite. Formal Current Run99 stays **571.681 tok/s**. Run638 guarded OFF/ON/OFF produced three individually valid 48+48/c12 arms and clean source/service restoration, but 0/48 client texts matched OFF_A→ON and cycle totals were1175/1227/1179. Cross-arm fixed W₀ and observer timing transfer failed. Packet `product_output_sha256` is Runtime bulk IDs, not final Scheduler/API token IDs. The ON arm's 96 sampled rank-cycles have current-stream medians: cycle55.538ms, Target46.901ms, DSpark6.346ms; *paired* Target+DSpark53.331ms and outside residual1.707ms. This is a restricted early-cycle diagnostic, not a Product bound.

Run639 gives a 16-row historical local MatMul→RS→copy fixed-observed-cost paired residual median2.160µs, only a local conditional screen. Run640 independently replayed all8 Run611 rank windows: physical and wait intervals overlap140.946–198.821ms; wait-only6.173–7.103ms, not removable idle. Run641's same-W₀ 24 Target Graph fixed-stream relaxation has observed span45.102–72.209ms, longest-stream physical sum36.905–38.083ms and conditional gap median15.819ms. All HCCL-named stream0 task intervals lie inside stream1 wait intervals and have **zero** overlap with stream1 physical tasks. HCCL task duration can include peer wait; GMM/HCCL fixture contention rejects ideal overlap. The conditional gap is neither legal saving nor whole-Product TPS. The largest unresolved schedule term is Target internals/HCCL wait ownership and resource interference. Full Framework-only Product TPS interval remains **unidentified**, not zero and not an asserted ceiling. See `FRAMEWORK_SCHEDULING_BOUND.md` and Run638–641 evidence.

Next: type producer→collective→consumer edges and resource modes for one real Graph layer/cut using existing Run611 trace/source; distinguish HCCL active service from peer wait, then choose the smallest same-W₀ measurement that closes the dominant unknown. Keep correctness/repeated formal E2E as final performance gate.

## Framework-only first-collective arrival and precursor — Run643–644

In the same Run611 Level0 diagnostic W₀, all8×3 first Model45 ReduceScatter events match high-level count49,152/BFP16/MESH-RING-NHR. Arrival spread is26.733/9.739/9.783ms, while finish spread is12–21µs; rank7 arrives last with a34–35µs native duration. The first collective begins49–55µs after each Graph entry. Early-rank HCCL task duration is thus strongly consistent with **peer wait**, not an intrinsic frozen communication cost. The first occurrence may include profiler startup disturbance. Cross-rank clock alignment is conditionally supported, but not independently proven to microsecond precision.

Run644 joins the two adjacent Graph transitions: prior exported Graph last-event spread52/56µs, next Graph entry spread9.741/9.785ms, rank7 latest both times. The skew reappears between observed Graph boundaries. Previous proposer Host-scope end and next Target Host start spread much more, but those Host scopes overlap native execution; rank5 proposer Host end precedes the prior exported Graph last event by15.567ms. No disjoint Host cost or removable time is inferred. **Framework-only full Product TPS remains unidentified**; neither early-rank waiting nor the Run641 conditional gap can be multiplied by cycle count. Next resolve rank7 pre-submit necessary predecessor/wait lineage from existing API/native trace, then only the missing bounded all8 device-ready/enqueue measurement. See Run643/644 findings and independent Astra reviews.

## Fixed-primitive Host issue coverage — Run645–647

Run645 replays two adjacent Run611 Graph transition windows all8. Rank7 spans24.591/22.214ms with only8.103/8.080ms physical kernel/copy union and16.267/13.941ms **cumulative** time without any exported native task. This is spread over258/253 short gaps (largest0.863/0.380ms), not one long Host stall or certified device idle. Rank7 stream47 physical work stays near7.1ms, comparable to other ranks; other ranks' longer stream38 communication-named tasks can contain peer waiting.

Run646 exactly joins stream47 native tasks to `torch_to_npu` flow and CPU-op begin: rank7 adjacent-task gaps total16.353/14.012ms, of which the next CPU operation had not begun for10.047/8.323ms. Run647 also joins the corresponding Host async task queue Enqueue/Dequeue: next Enqueue had not begun for10.116/8.387ms. This rejects a pure “already enqueued, only device queued” explanation for those conditional fragments, but says nothing yet about legal advance. The *next op* scopes are DSpark layer45, derived Target metadata and other DSpark model work; scope labels do not identify the preceding cause. Dequeue starts invert native start by up to6.491µs in11 rank6 records, so no Dequeue timing bound is used. Astra independently reviewed Runs645–647.

The Framework-only DAG now explicitly includes Host CPU issue, async queue Enqueue, native start, per-rank collective arrival and all8 completion as separate nodes. The next highest-value offline action is the rank7 preceding CPU op/input-ready lineage for late Enqueues, retaining mandatory state/DSpark/metadata edges and resource overlap. The10/8ms conditional issue fragments cannot be subtracted from Product wall or extrapolated across1,206 formal cycles. The full Product Framework-only TPS interval is still **unidentified**; Formal Current remains571.681tok/s. Large Run646 raw per-task data remains on the measurement host with SHA in its committed summary.
