# Results

## 2026-09-20 — Loop 002: corrected DP1/TP8 baseline (Loop 001 pivoted after parser failure)

**Outcome: accepted warm-cache baseline, 48/48 on each of three corrected passes.** No performance optimization is claimed yet.

| Pass | Output TPS | Mean TTFT | Mean TPOT | Success |
|---|---:|---:|---:|---:|
| warm 1 | 543.65 tok/s | 1120.88 ms | 19.45 ms | 48/48 |
| warm 2 | 523.15 tok/s | 1372.41 ms | 20.08 ms | 48/48 |
| warm 3 | 545.85 tok/s | 1331.80 ms | 19.73 ms | 48/48 |
| median | 543.65 tok/s | 1331.80 ms | 19.73 ms | — |

Each pass generated 49,152 tokens. Run 1–2 prefix hit rate was 99.75%; DSpark acceptance was 42.01% / 2.94 accepted per draft. Third acceptance was 41.21%. NPU AICore median/p90 were 77%/83% in 5-second snapshots during passes 1–2. The 20.5 tok/s spread between runs 1 and 2 is a practical noise warning; future KEEP needs a clear margin or paired repeats.

Correctness: two deterministic short prompts returned `42` identically. Four 32K prompts with 128 output tokens each were frozen in `evidence/20260920_baseline/golden4.json` for candidate output equality checks. The original first full run had a client parser bug (`reasoning` deltas ignored), so its TTFT/TPOT are INVALID; raw file and corrected 2/2 parser proof are retained. No framework source was modified.

Freeze: container image ID, source commits, model config/index hashes, dataset hash, versions and device preflight in `evidence/20260920_baseline/`. Launch command in `scripts/serve.sh`; measured client in `scripts/bench.py`; runner and analysis in `scripts/run_baseline.sh` and `scripts/analyze_baseline.py`. `logs/` has raw service log on this host. The corrected passes followed a full-dataset warmup from the invalid client pass; for a restarted candidate, run one full pass before official warm-cache measurement.

Next: current-topology profiling and a separate cold-prefill measurement. Rank DSpark, TP8 communication, kernel/HBM and host exposed time from actual traces before implementation.

## 2026-09-20 — Loop 003 diagnostic in progress

Two disjoint uncached 32K→128 c1 groups each passed 4/4; TTFT mean 2662.7 and 2627.2 ms. The second group had 0 cache hits over 131,404 queried tokens. System msprof on device 0 succeeded during a third group (4/4); output is indexed under `evidence/20260920_diagnostic/`, but no per-op kernel timeline was captured. Baseline service was stopped by terminating its own API process; all 8 NPU process lists cleared, container remained running. The same serving configuration has been relaunched with `PROFILING_MODE=dynamic` for current-topology application profiling. No performance KEEP/REJECT yet.

## 2026-09-20 — Loop 003 pivoted after diagnostic diagnostic: TP0 application profile

The service was restarted with frozen DP1/TP8 serving parameters plus `PROFILING_MODE=dynamic`, ready at 16:17:54 UTC. Dynamic `msprof` attached to TP0 worker. First profile attempt used a wrong dataset path and issued no request; its small failure log is retained. The corrected cold profile used four new distinct 32K→128 c1 prompts: 4/4, mean TTFT 2719.5 ms under profiler. A separate exact-repeat warm profile used four 128-token requests at c4: 4/4, 140.94 output tok/s under profiler. Neither is directly comparable to the official 48×1024 c12 baseline.

| TP0 trace | Event span | Compute/copy union | HCCL union | All-op union | Largest HCCL types |
|---|---:|---:|---:|---:|---|
| cold 4×32K→128 c1 | 18.583 s | 10.829 s | 6.386 s | 16.510 s | reduce-scatter 3.588 s, all-gather 1.934 s, all-to-all 0.864 s |
| warm 4×128 c4 | 3.401 s | 1.688 s | 1.165 s | 2.799 s | reduce-scatter 0.577 s, all-gather 0.356 s, all-to-all 0.231 s |

Union times can overlap and do not prove removability. The warm trace has only ~0.054 s compute/HCCL overlap and 0.602 s with no recorded device op. FlashComm1 sequence-parallel reductions occur in layers and an all-gather feeds MTP (`models/deepseek_v4.py:1144` in the bound source). This justifies a single-variable FlashComm1-off experiment, not a performance claim. Raw paths and op-summary hashes are in `evidence/20260920_diagnostic/app_profile_index.json`; summarize again with `scripts/analyze_profile.py`. No performance KEEP has been earned.

## 2026-09-20 — Loop 004 pivoted: invalid FlashComm1-only configuration

The service command confirmed DP1/TP8 and `enable_flashcomm1=false` with all other options frozen. All workers failed before model load with `ValueError: DSA CP requires SP to be enabled`. No correctness or performance request was issued; the waiting benchmark was stopped. This is an invalid intervention, not a REJECT on performance. Exact failure excerpt and full log SHA256 are under `evidence/20260920_flashcomm_off/`. Next loop tests the coupled FlashComm1-off/DSA-CP-off path explicitly.

## 2026-09-20 — Loop 005 pivoted: exact golden is nondeterministic

The coupled FlashComm1-off/DSA-CP-off service loaded and became healthy at 16:52:57 UTC. Its four 32K prompt outputs did not match the frozen baseline hashes. Repeating those same prompts on the unchanged candidate also yielded 4/4 different hashes, with all prompts equal and all completions at 128 tokens. The strict reasoning-text gate is invalid for this setup; no official throughput benchmark ran. This is not evidence of a semantic or numerical regression. Raw responses and gate results are in `evidence/20260920_sp_dsa_off/`. Next: stable functional checks, exploratory E2E benchmark, then stronger correctness validation if the candidate is performant.

## 2026-09-20 — Loop 006 REJECT: coupled SP/DSA CP off

The valid coupled path passed functional checks (long 4/4 at 128 tokens; exact short answers 42, OK, 4). A full-dataset warmup ran before three official, same-protocol 48×32K→1024 c12 passes. Each pass succeeded 48/48.

| Pass | Output TPS | Mean TTFT | Mean TPOT |
|---|---:|---:|---:|
| 1 | 524.41 tok/s | 1475.53 ms | 20.04 ms |
| 2 | 541.05 tok/s | 1435.47 ms | 19.87 ms |
| 3 | 534.88 tok/s | 1400.30 ms | 19.72 ms |
| median | 534.88 tok/s | 1435.47 ms | 19.87 ms |

Versus frozen baseline medians 543.65 tok/s, 1331.80 ms, 19.73 ms: throughput -1.61%, TTFT +7.8%, TPOT +0.7%. The throughput change is within the baseline three-run 4.3% spread. **REJECT** as a performance optimization; no numerical equivalence claim. DSpark counters across the candidate warmup plus three passes show ~41.8% acceptance and 2.927 accepted tokens/draft, close to baseline, but that counter window includes warmup. Full evidence: `evidence/20260920_sp_dsa_off_perf/analysis.json`, `functional_check.json`, individual passes and metrics snapshots.

## 2026-09-20 — Loop 007 REJECT: DSA CP off alone

With FlashComm1 restored to true and only DSA CP false, functional checks passed and three full warmed 48×32K→1024 c12 passes completed 48/48 each.

| Pass | Output TPS | Mean TTFT | Mean TPOT |
|---|---:|---:|---:|
| 1 | 535.16 tok/s | 1532.98 ms | 19.88 ms |
| 2 | 518.10 tok/s | 1605.03 ms | 20.48 ms |
| 3 | 514.23 tok/s | 1435.82 ms | 20.48 ms |
| median | 518.10 tok/s | 1532.98 ms | 20.48 ms |

Versus baseline medians 543.65 tok/s, 1331.80 ms, 19.73 ms: TPS -4.70%, TTFT +15.1%, TPOT +3.8%. **REJECT**. Full warmup, exact configuration command, functional check, individual responses and metrics snapshots: `evidence/20260920_dsa_cp_off/`. No numerical equivalence claim is needed for a rejected performance path.

## 2026-09-20 17:59 UTC — Loop 008 accepted diagnostic

Restored baseline FlashComm1 and DSA CP, started torch-NPU profiling on the same DP1/TP8 W4A8 service, and completed warm 4×128 c4 (4/4) and cold distinct 2×32K→128 c1 (2/2). `phase_times.json`, `scope_summary.json`, `scope_children.json`, `prepare_item.json`, `device_phase_summary.json`, and `raw_index.json` preserve the reproducible analysis and raw trace hashes. Torch-NPU offline analysis was needed after background parsing failed.

Warm wall 5.473 s; TP0 device union 4.094 s, HCCL 2.269 s (reduce-scatter 1.146 s, all-gather 0.756 s, all-to-all 0.372 s), compute/copy 1.876 s. CPU prepare 1.320 s, draft 2.237 s; nested `aten::item` 0.436 s across 1364 calls. Cold wall 11.178 s; TP0 device union 9.481 s, HCCL 4.640 s, compute/copy 5.203 s. CPU forward 4.293 s, prepare 2.384 s, draft 3.405 s. Overlaps mean these sums are not additive. This is a diagnosis, not a performance optimization. Next identify the repeated `item` callsite and quantify exposed synchronization without profiler, then test a minimal safe change.

## 2026-09-20 18:17 UTC — Loop 009 PIVOT: stack profiler crash

Baseline flags and DP1/TP8 source stayed unchanged. A short 4×128 warm set completed 4/4 under torch-NPU `with_modules=true`; `/stop_profile` returned HTTP 500 after worker segfaults. Offline TP0 export contained device kernels but no `operator_details.csv` or `aten::item` trace events. Its parser warned of lost data. The service exited; 8 NPUs are idle. The initial runner path error was corrected before the successful request/profile attempt. Evidence: `evidence/20260920_stack_profile/run1/`. No correctness or throughput conclusion on the model. PIVOT to a local, reversible item-call tracer that records callsite, tensor device, and elapsed time only during input preparation.

## 2026-09-20 18:35 UTC — Loop 010 accepted diagnostic, no KEEP

With baseline flags and no torch profiler, the reversible `Tensor.item` wrapper completed warmup 4/4, warm measured 4/4, and cold distinct 2/2. Source line `dsa_cp.py:1008` takes `seq_lens_q.max().item()` on NPU in QLI metadata, while the builder has already calculated corresponding CPU local maxima. TP0 cumulative tracer snapshots gave 210 ms/54 line-1008 calls over an approximately aligned warm phase and 1268 ms/70 calls over a later cold phase; TP1 had 209/1171 ms and ranks 2–7 had 5–37/648–783 ms. Cross-rank skew and device dependencies mean these numbers are not E2E savings. `dsa_cp.py:1009` is the next NPU scalar and may become the synchronization point if only line 1008 changes. The tracer patch is saved as `patches/loop010_item_trace.patch`; no optimized code or numerical equivalence claim yet. Next candidate passes existing CPU maxima to QLI metadata, verifies parity on live warm/cold paths, and runs the frozen full benchmark.

## 2026-09-20 18:58 UTC — Loop 011 candidate provisional data

Runtime parity assertions passed at least 192 comparisons on each of 8 ranks across warm and cold requests. Functional check passed. After full-dataset warmup, three 48/48 mixed passes produced output TPS 556.93,524.73,540.40 (median540.40 vs baseline543.65, -0.60%); TTFT median1413.44 vs1331.80 ms and TPOT median19.765 vs19.73 ms. Thus no mixed workload gain. After parity flag removal, fresh cold offsets24-27 and28-31 each passed4/4 and mean TTFT2338.29/2359.84 ms. Old baseline cold offsets differ; a paired original-source restart and exact offset24-31 comparison is pending. No KEEP/REJECT until this test completes.

## 2026-09-20 19:20 UTC — Loop 011 PIVOT after paired cold and mixed comparison

Original-source baseline was restarted with the same DP1/TP8 serving flags and full workload warmup. It passed function checks and three 48/48 mixed passes: output TPS537.60,524.63,543.67 (median537.60), TTFT mean median1262.74 ms, TPOT mean median19.50 ms. The all-step QLI CPU-max candidate had paired median540.40 TPS (+0.52%, noise), TTFT1413.44 ms (+11.9%), TPOT19.76 ms. Cold 4-request groups at offsets24 and28, same exact prompts and 32851 input tokens each, passed8/8 in both configurations. Candidate mean TTFT2349.06 vs original2633.90 ms; all 8 differences negative, range -343.24 to -204.60 ms, mean -284.83 ms (-10.81%). Original prefix hits0 over262808 cold queried tokens. CPU/NPU QLI max parity had passed >=192 comparisons per rank. These data support a cold opportunity but not an all-step KEEP because mixed TTFT worsened and throughput did not improve. PIVOT to prefill-only intervention. Full per-prompt and mixed comparison: `evidence/20260920_qli_pair_baseline/comparison.json`.


## 2026-09-20 19:48 UTC — Loop 012 KEEP: prefill-only QLI CPU maxima

Source commit `36589852a1eb8f5e842ad920f7c80ebdf1376ee9`; patch `patches/loop012_qli_prefill_only.patch`. The prefill branch reuses CPU local query/key maxima that had already been computed; pure decode uses the original NPU scalar maxima. Loop011 parity assertions checked at least 192 CPU/NPU maxima per rank on all eight ranks with no mismatch. Loop012 functional outputs passed the short exact and complete128-token checks.

After full-dataset warmup and three mixed 48/48 passes, candidate output TPS was 547.5468/538.5285/553.3549, median 547.5468. Paired original was 537.6045/524.6297/543.6665, median 537.6045. The +1.85% median change is within baseline spread; no mixed throughput gain is claimed. Candidate median request-mean TTFT was 1143.58 vs original 1262.74 ms and TPOT 19.252 vs 19.503 ms.

Four cold groups (offsets24,28,40,44) compared the exact same 16 prompts, each 32851 input tokens, c1→128 output, with no cache hits and all requests successful. Original mean TTFT2626.0327 ms, candidate2362.8907 ms; -263.1420 ms (-10.0205%), all16 faster, range -355.75 to -202.61 ms. This supports KEEP specifically for cold prefill. Full paired data: `evidence/20260920_qli_prefill_only/comparison.json`; service log checksum and functional/perf/cold results in the same evidence directory. TaskCtl Loop012 verdict accepted. Next investigate warm mixed decode throughput.

## 2026-09-20 19:56 UTC — Loop013 diagnostic in progress

Two msprof dynamic attach attempts to the healthy TP0 worker exited255 with Argument --pid: no valid pid values and were recorded invalid. The previous service was stopped cleanly, all eight NPUs returned idle, and a new same-source DP1/TP8 torch-NPU no-stack service PID780818 is loading. scripts/profile_decode_c12.py runner PID2105542 will collect full warmup, warm12×128 c12, and profiled 12×512 c12. No performance verdict yet.


## 2026-09-20 20:24 UTC — Loop013 PIVOT: warm c12 decode trace

Two msprof dynamic-attach attempts exited255 before sampling; one torch-run warmup used one output token per request and was invalid only under bench.py's success definition because TPOT needs at least two tokens. Corrected 128-token warmup, reused the healthy profiler-enabled DP1/TP8 service with the kept QLI source, and completed 48/48 warmup, 12/12 warm c12, 12/12 profiled 512-token c12. Profile API start and stop succeeded.

Derived data: 16.459 s window, TP0 device busy13.038 s, communication union5.510 s, compute/copy7.736 s; across eight ranks compute/copy7.631–7.745 s and HCCL2.236–5.843 s. TP0 170 draft host scopes sum9.036 s, median52.894 ms; typical draft has three MoE shared scopes15.774 ms total and three DSA scopes10.093 ms total. TP0 MoE self-time1.397 s across682 calls and DSA self-time0.964 s across682 calls. The profiled short run gave387.7 TPS, not comparable to the official unprofiled baseline. Evidence: run4/decode_c12.json, phase_times.json, device_c12_summary.json, host_scope_c12_summary.json, host_children_c12_summary.json, host_operator_self.json and raw_index.json under evidence/20260920_decode_c12_profile/. Raw traces remain on disk, excluded from Git. No KEEP/REJECT optimization claim; Loop013 PIVOT to source-level host/launch and rank-wait investigation.

## 2026-09-20 20:30 UTC — Loop014 active source audit

Read model_runner_v1.py prepare-input path and Loop013 host trace. The typical prepare scope was21.42 ms, with nested item0.70 ms and self median9.57 ms; broad source stages remain unattributed. Recorded TaskCtl design-check pass with no candidate or performance verdict. Next collect gated no-profiler per-stage timing before editing inference semantics.

## Live checkpoint 2026-09-20 20:33 UTC — Loop014 stage tracer

Loop014 applied 17-line diagnostic patch, syntax and diff checks passed. New no-profiler DP1/TP8 tracer service PID789878 loading. Runner PID2135816 pending functional, full warmup and c12 diagnostic. No performance verdict.

## Update 2026-09-20 20:47 UTC — Loop014 first stage trace

Loop014 first stage tracer run passed functional gate (long complete 128-token responses and short exact 42/OK/4), 48/48 warmup and 12/12 c12x512 sample. All eight rank traces complete; 157 pure-decode steps per rank. TP0 median prepare19.55ms, dominant compression plus attention metadata12.99ms, input assembly4.81ms, all other measured stages under0.8ms each. Rank medians for dominant stage10.80-14.17ms. Evidence: evidence/20260920_loop014_prepare_trace/run1/{stage_summary.json,raw_index.json,functional_check.json,decode_c12.json} and raw rank CSVs. No optimization verdict. Next split attention metadata builders to find a single correct candidate.

## Live checkpoint 2026-09-20 20:51 UTC — builder trace loading

Loop014 builder-level trace patch compiled and passed diff check. Same DP1/TP8 no-profiler service PID796352 loading with both tracer env dirs; runner PID2153103 pending. No result or verdict yet.

## Update 2026-09-20 21:08 UTC — Loop014 PIVOT

Loop014 PIVOT after two successful no-profiler diagnostic runs. Second run passed functional and 12/12 c12x512; 143 pure-decode builder steps per rank, eight groups per step. TP0 median attention metadata15.165ms, builder calls14.025ms, DCP0.002ms, residual1.117ms; first builder7.860ms, later each0.725-1.096ms. Source already caches common state across groups; prior all-step decode QLI scalar replacement did not yield mixed TPS gain. No semantics patch or throughput KEEP. Service stopped, 8 NPUs idle, diagnostic patch reverted; kept framework commit36589852 is clean. Evidence under evidence/20260920_loop014_prepare_trace/run2/ and patches/loop014_builder_trace.patch.

## Live checkpoint 2026-09-20 21:13 UTC — Loop015 diagnostic in flight

Shape audit script and compact JSON frozen. One-shot slot probe patch compiled and runs only after a flag is placed before the cold request; probe runner PID2173051 waits for DP1/TP8 API PID802820 to become healthy, then performs functional gate and one cold request. This run is diagnostic because the probe synchronizes once. TaskCtl Loop015 scatter-slot-uniqueness Run ongoing. No KEEP/REJECT yet.

## 2026-09-20 21:29 UTC — Loop015 PIVOT: scatter evidence and rejected V2

Functional gate passed; diagnostic cold offset24 32851-token input to 128-token output succeeded with 2471.76ms TTFT, not comparable because probe CPU sync. Every one-shot rank capture had 8096 valid unique pairs and no duplicates, matching [34091,32,1,512] cache and [8096,1,512] updates. The service was stopped, 8 NPUs idle, diagnostic patch removed; framework clean at kept commit36589852. In single-NPU 25-call screen with captured TP0 indices, SK and V2 caches were bit-equal; SK device median1.427ms, V2 median2.415ms (+69%), so V2 replacement REJECT. No E2E KEEP. Evidence and reproduction script under evidence/20260920_loop015_cold_kernels/ and scripts/bench_loop015_scatter_v2.py. TaskCtl Loop015 PIVOT, next inspect direct scatter feasibility and duplicate-safe fallback.

## Live checkpoint 2026-09-20 21:43 UTC — Loop016 experimental SWA prefill index_copy

Corrected captured-op attribution to SWA prefill. Captured c1 index mapping was unique on8 ranks; source provenance is scheduler slot mapping, not compressor_metadata. Single-NPU screens with captured indices and matching shape: built-in scatter ratio0.981 versus SK; contiguous index_copy0.510ms versus SK1.428ms; simulated stride32768 with per-call flatten0.694ms versus SK1.528ms, all bit-equal. Source patch is flag-gated and limited to one unpadded SWA prefill via CPU metadata; other paths retain SK. No service A/B or KEEP yet. TaskCtl Loop016 active.

## Live checkpoint 2026-09-20 21:47 UTC — Loop016 full-service test launched

API PID809483 and detached A/B runner PID2207198 started after8 NPU idle check. Candidate flag absent by default. Runner will require exact four-prompt long output parity and eight rank fast-path traces before eight alternated cold pairs. Pending, no performance verdict.

## 2026-09-20 22:08 UTC — Loop016 invalid first service gate

Short functional check passed. Initial long golden comparison used mismatched datasets (4/4 prompt hash mismatch); correct-dataset retry matched prompt hashes but differed in all output hashes, consistent with documented nondeterministic long reasoning text even under unchanged service. No fast path trace (0/8), so candidate did not run. Saved no-flag cold offsets8–15 baseline8/8, mean TTFT2367.81ms. Stopped service, eight NPUs idle; revised patch and runner frozen. A/B2 pending. No performance verdict.

## Live checkpoint 2026-09-20 22:12 UTC — revised A/B launched

After eight NPU idle check, no-profiler DP1/TP8 service PID815939 and detached runner PID2226077 started. Pending path activation and same-prompt candidate cold TTFT. No verdict.

## 2026-09-20 22:25 UTC — Loop016 A/B2 invalid, active DSA-CP caller found

Functional check and candidate probe request passed, but 0/8 rank and 0/8 metadata traces; enabled DSA-CP dispatches to dsa_cp.py:1522, bypassing edited dsa_v1.py. No candidate result. API stopped, eight NPUs idle. Third patch moves guard to DSA-CP caller and restores dsa_v1; syntax/diff checks pass. A/B3 queued with same cold prompt baseline.

## Live checkpoint 2026-09-20 22:29 UTC — A/B3 launched

Active DSA-CP patch service PID822276 and detached runner PID2242486 started after eight NPU idle check. Pending functional, 8-rank fast trace and exact-prompt cold A/B. No verdict.


## 2026-09-20 22:39 UTC — Loop016 REJECT

Candidate: flag-gated SWA prefill index_copy in active DSA-CP callsite, patch patches/loop016_swa_prefill_index_copy_v3.patch. Functional request passed and all eight TP ranks traced fast path; eight cold 32851→128 candidate requests succeeded. Same prompts/no-repeat offsets8–15 versus saved no-flag baseline, prefix hits0 before and after: baseline TTFT mean2367.8107ms, candidate2437.9979ms, delta+70.1872ms (+2.9642%, slower), pairs improved0/8. Per-prompt pairs and metrics: evidence/20260920_loop016_direct_scatter/service_ab3/. Reject candidate; isolated screen was faster but E2E effect falsified. Experimental API stopped, eight NPUs idle, framework clean at kept commit36589852. TaskCtl Loop016 verdict rejected. Next inspect Compressor ratio4 prefill critical path.


## Live checkpoint 2026-09-20 22:46 UTC — Loop017 profile reuse

Started Loop017 and distilled frozen original-source cold msprof CSV with scripts/analyze_loop017_compressor.py. Four cold request clusters each contain84 main ratio4 Compressor tasks, ~126.4ms summed task time/request, median task1.503ms. Secondary [512,4096] shape80 calls/request/~40ms sum and [256,4096] shape84 calls/request/~29.8ms sum. Evidence evidence/20260920_loop017_compressor/profile_clusters.json; TaskCtl profile run passed. No implementation or E2E claim. Next attribute critical path and isolate one safe optimization.


## Live checkpoint 2026-09-20 22:50 UTC — Loop017 stream order

scripts/analyze_loop017_stream.py scanned frozen original-source cold msprof CSV. For all336 main-shape ratio4 Compressor calls, next same-stream task is ScatterNdUpdateSk and second is SparseAttnSharedkv. Median end-to-next gap0.00287ms; median Compressor-start to attention-start1.84475ms. JSON evidence/20260920_loop017_compressor/stream_order.json, TaskCtl profile run passed. This supports local serialization; no E2E improvement claim or source change.


## Live checkpoint 2026-09-20 23:59 UTC — Loop017 operator screen

One-NPU 25-call synthetic Compressor ratio4/coff2 shape-matched screen: finite [2025,512] output, device median1.630ms/min1.499ms, wall median1.810ms. Script scripts/bench_loop017_compressor.py; evidence evidence/20260920_loop017_compressor/shape_matched_operator.json. Initial unscaled random data gave nonfinite output, preserved separately and excluded. No reference-output correctness test and no full-service gain claim. Operator source/installed package untouched. Investigating safe isolated mBase/nSize tiling sweep; global-vendor replacement is not authorized by this screen.


## Live checkpoint 2026-09-21 00:06 UTC — mBase256 build underway

Rebuilt stock synthetic reference with fixed seed: finite output, operator device median1.514ms/25 calls; output and first4096 cache blocks saved in ignored artifacts/loop017_compressor_reference_output.pt, SHA256 e5dc8cfee73cec0bb80a5ce95dbf48ed3222a4d7f628ce6e381a615b56cbb0ec. Input fingerprint in evidence/20260920_loop017_compressor/reference_screen.json. Isolated candidate patch mBaseSize128→256 in patches/loop017_compressor_mbase256.patch; build script/log/prefix recorded. Package compilation ongoing. No verdict.


## Live checkpoint 2026-09-21 00:22 UTC

Isolated mBase256 build still active after ~19 minutes, CPU compilation progressing. Watcher scripts/watch_loop017_build.sh will automatically write candidate_run.log and candidate_exit_code.txt after build termination; no candidate result yet. Source and installed operator remain at kept baseline.


## 2026-09-21 01:10 UTC — Loop017 REJECT and communication audit

Candidate: Compressor arch32 coff2 mBaseSize128→256, reproducible patch patches/loop017_compressor_mbase256.patch. Private package build succeeded; fixed-seed stock reference device median1.51436ms. Candidate loaded private libcust_opapi.so but produced no result after >8 minutes; interrupted SIGINT/exit130, all8 NPUs idle. No numerical comparison, candidate time or service benchmark exists. TaskCtl verdict rejected, comparability invalid, causal conclusion not identifiable. Raw logs and structured observation in evidence/20260920_loop017_compressor/. Installed operator and framework unchanged.

Read-only c12 communication audit of saved eight rank profiles: all ranks contain 25160 allGather, 7820 alltoall, 15980 reduceScatter entries in the selected window. Reported elapsed is wholly Idle Time for every entry; Transit and Wait are zero. The audit cannot quantify removable communication latency. Evidence evidence/20260921_decode_comm_audit/collective_time_components.json; script scripts/analyze_hccl_comm.py. Loop018 will attribute rank arrival and critical path before implementation.


## 2026-09-21 01:15 UTC — Loop018 PIVOT

Read-only eight-rank c12 communication audit: 48,960 collective types align exactly by sequence. Raw start spread median0.6105ms, end spread0.583ms. Rank6 appears earliest in48,829/48,960 calls; a shared-end timestamp calibration estimates -579.172us offset versus rank0, yielding corrected start/end medians0.2175/0.206ms. Calibration is a sensitivity test, not proof of clock sync. No causal communication savings or patch established. TaskCtl Loop018 pivoted; scripts/analyze_collective_arrivals.py and scripts/analyze_collective_clock.py reproduce evidence/20260921_decode_comm_audit/. No service or framework modification.


## Live checkpoint 2026-09-21 01:59 UTC — Loop019 stage trace

Diagnostic only, no optimization candidate yet. Flag-gated DSA-CP builder trace on DP1/TP8 passed short functional and12/12 no-profiler c12×512 requests. Pure decode174 steps/rank; first builder median total3.61–6.50ms with request-metadata2.74–5.74ms, shared setup0.57–0.66ms. All eight raw rank CSVs and summary in evidence/20260921_loop019_builder_stage/. API stopped; all NPUs idle. Next test expands request-metadata stage timing using patches/loop019_builder_req_trace.patch.


## Live checkpoint 2026-09-21 02:13 UTC — Loop019 second diagnostic

Flag-gated request-metadata trace passed short functional and12/12 c12×512 sample. First-builder QLI subphase median1.735–4.306ms/rank across169–170 pure-decode steps; device-local0.477–0.548ms, CPU-local0.255–0.292ms, SAS0.422–0.485ms. No performance candidate, no KEEP/REJECT yet. Raw eight-rank traces and summary in evidence/20260921_loop019_builder_req/. Service stopped; framework clean at36589852, all8 NPUs idle.


## 2026-09-21 03:11 UTC — Loop019 PIVOT

Three no-profiler diagnostics localized first DSA-CP builder cost. Final QLI trace passed short functional and12/12 c12×512; 155–156 uncached pure-decode calls/rank. Median first q.max().item0.402–3.716ms/rank, second k.max().item0.110–0.135ms, metadata op+clones0.356–0.411ms. Earlier Loop011 all-step CPU maxima had verified parity but only+0.52% mixed TPS within noise and worse TTFT; no new patch kept. TaskCtl Loop019 pivoted. Raw eight-rank CSV, sample, and summary: evidence/20260921_loop019_qli_subphase/. Service stopped, all NPUs idle, framework clean36589852. Next DSpark/target critical path.


## Live checkpoint 2026-09-21 03:20 UTC — Loop020 attribution

Read-only saved c12 analysis passed: 170 TP0 draft_token scopes median52.894ms; device activity coincident inside them median41.860ms union. No candidate gain claim. Flag-gated proposer stage service loading DP1/TP8; API PID856647, runner PID2439916. Raw and distilled saved-profile evidence in evidence/20260921_dspark_audit/.


## Live checkpoint 2026-09-21 03:36 UTC — Loop020 diagnostic

Flag-gated proposer stage trace passed short functional and12/12 c12×512. Across160 pure decode calls/rank, eager _propose median36.11–41.38ms, run_draft28.43–32.69ms, step0 all-group attention metadata5.67–6.41ms, set_inputs1.47–1.66ms. Eight rank CSVs and summary in evidence/20260921_loop020_draft_stage/. Saved profiled acceptance buckets in evidence/20260921_dspark_audit/acceptance_profile.json. No optimization or E2E gain claim. Service stopped; framework clean and NPUs idle.


## 2026-09-21 04:09 UTC — Loop020 PIVOT

A reproducible TP0 flow analysis maps every async finish to a same-stream device task start exactly and restricts ownership to CPU flow origins inside the same-thread `draft_token` scope. For 170 scopes, causal device union clipped to the host interval is median 51.661 ms versus 52.894 ms host duration; device launches extend beyond the scope to 63.259 ms median union. This supports a device-active eager proposer bottleneck and rejects treating inclusive host metadata timings as removable idle time. Acceptance is 3.54–3.64 advanced tokens per 7 drafted. No code candidate or E2E gain was claimed. Evidence: `evidence/20260921_dspark_audit/async_flow_attribution_tp0.json`, stage and acceptance summaries. Loop021 begins source attribution of repeated Index/Pad/Copy/event chains.


## 2026-09-21 05:10 UTC — Loop021 REJECT

A reproducible trace join assigned all 121,038 draft-owned launches to an innermost CPU leaf, semantic vLLM scope and exact device task. The hypothesized Index/Pad/Copy critical chain is below decision value: Index median clipped union 0.545 ms when present, Pad 0.337 ms, InplaceCopy 0.112 ms. The large causal union from Loop020 is instead dominated by `EVENT_WAIT`: 51.169 ms median at outer draft scope and 26.595 ms under MoE. These values are dependency occupancy and can overlap; they are not performance gains. No implementation was attempted. Evidence: `evidence/20260921_loop021_device_parent/`. Loop022 now tests whether MoE event waits serialize the critical stream.


## 2026-09-21 06:10 UTC — Loop022 REJECT

Exact same-stream neighbors and source alignment with `_forward_shared_experts` show no material main-path event stall. Streams42/43 wait 53–54 ms only before asynchronous memcpy. Shared stream36 waits 25.734 ms between layer invocations for the next hidden state; its intra-call event waits are 0.191/0.051/0.011 ms, while stream47 final waits are ~0.00002 ms. The MoE scheduling hypothesis is falsified and no patch was made. Evidence: `evidence/20260921_loop022_event_dependency/`. Loop023 begins a controlled speculative-length efficiency screen.


## 2026-09-21 07:07 UTC — Loop023 REJECT

DSpark k5 failed before readiness with the exact invariant that graph shapes must be divisible by both k+1=6 and TP8. No correctness, warmup or performance request ran, so no comparative claim is made. Evidence: `evidence/20260921_loop023_k5_screen/startup_failure.json` and service log. The next experiment keeps valid k7 and tests max batched8288 to recover the96 slots reserved for drafting.


## 2026-09-21 08:14 UTC — Loop024 REJECT

Candidate max-batched8288 passed correctness and removed the scheduler warning. Its12-request c12 screen delivered472.871 tok/s with17.300ms mean TPOT, only +2.77% versus prior-screen median and -3.83% versus closest Loop020. The result is inside4.3% noise, so it was not promoted to full frozen benchmark. Evidence: `evidence/20260921_loop024_tokens8288/screen_comparison.json`. Loop025 compares k7 DSpark with target-only decoding.

## 2026-09-21 09:09 UTC — Loop025 ACCEPT

Matched target-only decode achieved208.047 tok/s and54.333ms TPOT; k7 DSpark achieved491.698 tok/s and17.554ms TPOT under the same12-request c12 runner. This is +136.34% TPS and -67.69% TPOT, decisively beyond noise. Correctness passed for both. Evidence: `evidence/20260921_loop025_nospec/comparison.json`. Retain DSpark; Loop026 evaluates the cost/acceptance tradeoff of bypassing its middle draft layer.


## 2026-09-21 10:09 UTC — Loop026 REJECT

Bypassing only the middle DSpark layer saved23.21% proposer model time but reduced advanced tokens to1.363/cycle and output TPS to217.092 (-55.85% versus491.698 control). Correctness passed through target verification, proving this is a proposal-quality failure rather than semantic corruption. The full three-layer drafter is necessary. Evidence: `evidence/20260921_loop026_layer_bypass/comparison.json`. Loop027 evaluates exact three-layer captured/standalone execution feasibility.

## 2026-09-21 11:18 UTC — Loop027 PIVOT

The bounded v2 DSpark graph probe failed before service readiness. Target architecture resolved as `DeepseekV4ForCausalLM`, draft architecture as `DeepSeekV4MTPModel`, and generic v2 `set_attn` found no draft KV group, raising `No draft attention groups found`. Graph capture, correctness and benchmark were not reached. Evidence: `evidence/20260921_loop027_graph_feasibility/startup_failure.json`. This closes the immediate stock-v2 route and triggers the specialized-runtime path: use the working legacy implementation as semantic/operator oracle, extract a fixed decode-cycle contract, then remove scheduler, dynamic metadata and orchestration behind exact parity gates.

## 2026-09-21 — Loop028 PIVOT: fixed proposer replay established, full cycle incomplete

- Source contract maps target forward/verification, acceptance, proposer input preparation, the three-layer DSpark backbone, Markov7 drafting and state publication.
- Corrected diagnostic passed short functional checks and recorded 172 c12 pure-decode calls per rank.
- All traced shapes/strides were fixed. Ten fields had one address per rank; target token IDs and positions changed allocation address.
- The real-weight, TP8, already-materialized proposer closure replayed once without scheduler/request rebuilding and returned exactly equal `[12,7]` draft tokens on all eight ranks.
- Verdict: **PIVOTED / partially supported**. The proposer is a valid executable replay segment, while the frozen Loop028 success gate also required target verification, accepted-token parity and full mutated-state parity. Those comparisons were not implemented.
- Performance: no claim. The 12×32K→512 diagnostic produced 412.50 tok/s with tracing and a duplicate proposer invocation, so it is not comparable with the frozen 543.65 tok/s baseline.
- Cleanup: service stopped, no NPU processes, framework restored clean at `36589852a`.
- Work paused after the Loop028 commit at the user's request. Resume from the target verification/acceptance/state boundary; do not repeat pointer discovery.
## 2026-09-21 14:35 UTC — Loop029 standalone state and acceptance parity

- Added a product-owned fixed decode scaffold with fixed c12 device state, next-target ABI materialization, and temperature-0 greedy speculative acceptance. Its control loop does not construct vLLM scheduler outputs, request objects, input batches, or model runners.
- Eight-cycle CPU and Ascend NPU semantic tests pass.
- Real DeepSeek V4 Flash W4A8, 8×910B3, DP1×TP8, DSpark7 c12 oracle-shadow result: 12/12 requests completed; next-target ABI exact in 224/224 rank-cycle comparisons; independent accepted tokens and counts exact in 232/232 rank-cycle comparisons.
- Scope limitation: target/DSpark forward, KV cache, recurrent/GDN state, collectives, and weights are still executed/owned by the oracle. Therefore this is an accepted execution-contract result, not a completed standalone runtime and not a performance result.
- Diagnostic throughput 130.44 tok/s is excluded from all baseline comparisons because the hook performs synchronous equality checks and JSONL writes every cycle.
- Next implementation gate: runtime-owned fixed KV/recurrent storage plus a direct target operator adapter, followed by at least eight causally connected real cycles with token and state-fingerprint parity.

## 2026-09-21 15:23 UTC — Loop029 independent control plane

- `ExtremeDecodeRuntime` owns cycle ordering and executes without importing vLLM.
- CPU and Ascend NPU standalone runs completed eight continuous cycles; target and proposer were each invoked eight times.
- Runtime-owned cache references enforce stable address/shape/stride and support exact selected-element fingerprints for touched physical cache locations.
- Current operators/caches in this test are deterministic/synthetic. Live DeepSeek weights, TP/EP collectives and DSA cache tensors remain the next correctness gate; no performance claim is made.

## 2026-09-22 08:16 UTC — Loop029 real-weight standalone decode milestone PASS

- One-time bootstrap transferred the real DeepSeek V4 Flash W4A8 target,
  TP8/EP communication context, 67 deduplicated KV/DSA cache tensors, fixed c12
  buffers and DSpark7 into the Extreme Runtime boundary.
- The handoff occurred before the generic `ModelRunner` target forward. Evidence
  records `oracle_target_calls_after_handoff=0`, no retained `ModelRunner`, and
  `_DP1RunnerShim` rather than the generic runner in the proposer boundary.
- All eight Ascend 910B3 ranks completed eight continuous runtime-owned cycles
  with the fixed stage order. State advance was exact, acceptance counts stayed
  within the 1..8 contract, 163 tokens were accepted/emitted, and final state
  was identical across ranks.
- Transaction diagnostics restored 69/69 mutable state entries exactly. Stock
  graph self-replay changed 2/96 argmax decisions and accepted-token values;
  direct eager self-replay changed 3/96 argmax decisions while retaining the
  accepted result. Strict repeated-forward equality is therefore not used as a
  gate beyond the existing target ABI and acceptance parity evidence.
- Evidence: `evidence/20260922_loop029_real_runtime/run17/summary.json`, rank
  records in the same directory, run16 replay audit, and TaskCtl run
  `real-extreme-runtime-pre-modelrunner-handoff-20260922`.
- Performance: no claim. The surrounding API request is intentionally failed
  after evidence capture and is not comparable with the 543.65 tok/s baseline.
  Performance work starts only on the runtime-owned DAG.
## 2026-09-23 Loop035 diagnostic checkpoint

- Eight-rank Extreme short c12 DAG event medians: target45.07 ms, proposer5.93 ms, acceptance0.38 ms, prepare0.27 ms, advance0.02 ms. These are diagnostic NPU events, not removable E2E costs.
- Warm Stock single-cohort 12×32K→1024 c12: 12/12 complete, 553.76 tok/s, 2.908 accepted drafts/iteration from counter deltas. This reinforces sustained Stock acceptance; it does not replace Loop034 formal 48-request baseline.
- DSA-CP start_pos and local_seq_lens were found stale from cycle 1. Refreshing both kept them current but produced only 127 outputs across 8×12 Extreme slot cycles; the isolated candidate was rejected. All eight rank records passed the local state contract, which is weaker than token-level oracle parity.
- Rejected/invalid probes and raw traces are indexed in TaskCtl Loop035. No structural performance KEEP or formal A/B rerun yet.

- Loop035 SWA-only slot refresh: exact bootstrap parity, all eight TP ranks passed, 161 outputs over 8×12 slot cycles, no acceptance recovery. Candidate reverted; TaskCtl run run-20260923T064032Z.
- Product state carry checks across three traces: draft, last token and num_computed each 84/84 consecutive-slot transitions exact. First draft/target argmax match degrades after cycle 0, so target/proposer comparison is the next semantic gate.

- Loop035 Run11 DSA builder oracle with shared CPU mirror was INVALID (host_mirror_exact=False); 211 outputs are excluded from causal interpretation.
- Loop035 Run12 corrected DSA builder oracle passed 8/8 local gates and emitted 209 outputs/8 cycles. Run13 DSA+GDN builder oracle likewise passed and emitted 215. Both use direct eager target and unordered prompt admission, so no Stock parity or throughput claim follows.

- Loop035 same-state proposer Run14: 69/69 touched-cache and mutable entries restored exactly on all eight ranks; Product/Stock draft outputs matched 71/84, with differences only in two count=1 slots. Proposer-private state and self-replay controls remain open; no causal root or KEEP claimed.
- Loop035 Run18: eight-rank, 1024-cycle Extreme-owned diagnostic passed state
  and host-mirror gates. Rank-0 12,797 emitted tokens/53.571 s
  (238.99 tok/s diagnostic). NPU event medians after cycle 16:
  target 45.510 ms, DSpark 5.943 ms, acceptance 0.305 ms.
  Outputs/slot/cycle decayed to exactly 1.000 for cycles 960–1023.
  This does not supersede formal Loop034 E2E 217.342 versus 543.655 tok/s.
  Evidence: evidence/20260923_loop035_diagnostic/run18/summary.json.
- Loop035 Run19: per-cycle DSA+GDN builder refresh passed 256 direct eager
  target cycles on all eight ranks, but outputs/slot/cycle declined to 1.036
  in cycles 128–191. Target metadata rebuild alone did not prevent the
  long-run collapse. Different prompt ordering and eager execution preclude
  a paired quantitative comparison to Run18. No E2E throughput claim.
- Loop035 Run20 confirmed draft gid2 context slot mapping stale in 96/96 positions
  from cycle 1, but isolated refresh did not prevent acceptance collapse.
  Run21 combined it with target DSA+GDN metadata rebuild and still reached
  1.003 outputs/slot/cycle in cycles 128–191. Both were 8-rank, 256-cycle
  diagnostics with local state gates; no product KEEP or E2E claim.

- Loop035 evidence correction (2026-09-23): Run12/13/19/21 did not execute the intended DSA/GDN diagnostic refresh. Prior builder-effect claims above are withdrawn; cycle completion, state checks, and raw acceptance measurements remain valid. Run22 was explicitly marked invalid after all eight rank files had an empty target-slot audit. Commit a11a71b connected the callback; Run23 will require a nonempty audit and a cycle-0 mapping equality gate.

- Loop035 Run32 A/B/C control, 8×910B3: exact 72 old metadata tensor and 138 touched physical-entry restorations; at cycle1 target argmax A/C 92/96, A/B 75/96. Combined builder+slot refresh has a causal target effect, but two acceptance counts decreased in this cohort and correctness is unresolved. Run33 isolates builder-only before any structural KEEP or E2E rerun.

- Loop035 Run34 slot-only A/B/C control, 8×910B3: five target cache groups changed all 96 slots at cycle1, but target A/B and A/C both matched 88/96. No slot-only effect above replay noise was resolved. Run33 builder path remains the first detectable target difference; no semantic or throughput KEEP yet.

- Loop035 Run35 native metadata parity: invalid before comparison (attention
  group nesting error); no correctness result.
- Loop035 Run36 native/reference two-cycle field comparison: 8/8 ranks wrote
  evidence, 45/55 fields exact in cycle0, 42/55 in cycle1. Full SAS/QLI and
  three compressed slot fields differed; no semantic KEEP.
- Loop035 Run37 reference self-replay control: 8/8 ranks complete. SAS/QLI
  tails are unstable within reference itself, but native SAS/QLI header bytes
  differ before self-replay noise, and compressed slot differences are native
  only. The 11/12 client success count is from intentional diagnostic abort,
  not a serving benchmark. No formal throughput claim.
- Loop035 Run38 SAS operator trace, 8/8 ranks: reference/native arguments
  matched except fixed local `cu_seqlens_q` dtype int32/int64 across c1/c4/c128.
  SWA scope correction removed the three compressed slot mismatches. Runtime
  cumsum now preserves int32; correctness awaits Run39 and continuous gates.
- Loop035 Run39 int32 cumsum correction, 8/8 ranks and two cycles: all c1/c4/c128
  SAS operator inputs and first 32 outputs match reference; all 45 other fields
  match per cycle. SAS/QLI tails remain self-replay unstable. This is a metadata
  header gate only, not target/acceptance or long-token correctness.
- Loop035 Run40 DSA-only same-state target A/native B/reference C: 8/8 ranks,
  two cycles, A/B argmax 92/96 and 94/96 versus A/C 93/96 and 92/96. A/B
  acceptance counts 12/12 each cycle; exact physical/old metadata restores.
  Candidate effect stays within self-replay noise. Long semantic correctness
  and acceptance recovery remain unverified.
- Loop035 Run41: 8/8 ranks passed 256 native-DSA eager cycles with exact state
  and host mirrors, but target graph was NONE. Its 45.2 tok/s diagnostic and
  412.5/41.1ms target/proposer event medians are excluded from FULL-graph and
  formal E2E comparisons. Run42 corrects the graph mode.
- Loop035 Run42: corrected FULL-graph 256-cycle native-DSA profile passed on
  8/8 ranks. NPU medians metadata8.616ms,target46.201ms,proposer5.880ms;
  late acceptance1.594 outputs/slot/cycle. Short standalone309.16tok/s is
  not a formal same-protocol E2E result; no acceptance repair claimed.
- Loop035 Run43 combined DSA+GDN A/B/C: INVALID before target comparison on
  8/8 ranks because initial attention metadata had no exposed GDN count view.
  The next run binds the builder-owned stable tensor at bootstrap only.
- Loop035 Run44 GDN builder-count handoff: INVALID before target comparison on
  8/8 ranks because this DeepSeek V4 Flash path has no active GDN builder.
  The optional GDN Runtime binding was removed. All older "DSA+GDN" diagnostic
  descriptions are corrected: the GDN callback branch did not run for this
  model, so those measurements characterize DSA/SWA effects only.

- Loop035 Run45: 8/8 ranks passed late sampled FULL-graph native DSA target A/B/C and state/KV restore gates through 129 cycles. Native A/B differences at cycles64,128 remained within reference A/C replay noise (argmax74 vs72 and78 vs76 of96). The replay noise is too large to establish long token equivalence; low acceptance remains unresolved.

- Loop035 Run46: late cycle128 Product/Stock DSpark proposer first drafts matched 12/12 on all 8 ranks; 83/84 full drafts matched versus Stock self-replay 79/84. Inputs, prepare fields and KV restoration were exact. Product proposer difference is not a supported explanation for low acceptance at this sampled state.

- Loop035 Run47: Stock output self-control invalidated independent exact-text oracle. Two identical 12-request 1024-token cohorts completed, but 0/12 texts matched exactly; common prefix ranged 38–920 characters. Same-state KV-restored Stock/direct target control is required.

- Loop035 Run48: INVALID before target A/B/C on 8/8 ranks due missing diagnostic group common view. Gate corrected for Run49; no target or acceptance inference.

- Loop035 Run49: on a shared Stock cycle128 KV state, native-DSA direct target matched Stock target 96/96 argmax, 12/12 first target token, 96/96 accepted tokens and 12/12 counts on all 8 ranks; Stock self-replay matched 95/96 argmax. KV and metadata restores exact. Continuous trajectory remains unverified.
