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
