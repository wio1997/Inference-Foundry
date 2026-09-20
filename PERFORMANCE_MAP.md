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
