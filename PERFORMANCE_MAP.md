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
