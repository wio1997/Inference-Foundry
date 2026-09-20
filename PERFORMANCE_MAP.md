# Performance Map V0

Updated: 2026-09-20 15:57 UTC. Current evidence: 3 warm-cache DP1/TP8 end-to-end runs; no current kernel/HCCL timeline yet.

| Component | Observed current runtime | Evidence and confidence | Remaining Gap / next discrimination |
|---|---:|---|---|
| E2E warm mixed workload, 48×32K→1024, c12 | output TPS 543.65 median (range 523.15–545.85); TTFT mean 1.33 s median; TPOT mean 19.73 ms median | `evidence/20260920_baseline/bench48_[123].json`, high confidence in order of magnitude, ~4.3% TPS range | Separate cold and warm phases; quantify repeat noise before accepting small gains |
| Prefix reuse | 99.75% hit on measured passes | before/after Prometheus counters; high | This measured workload is decode-heavy after warmup; first-pass prefill must be measured separately |
| DSpark speculation | accepted 2.94 tokens/draft, 42.0% acceptance (runs 1–2); 41.2% run 3 | `analysis.json`, Prometheus; high for counters | Measure draft and target spans/bytes; compare speculation against a correctness-preserving control only if predicted gain is large |
| Device occupancy | AICore median 77%, p90 83%; HBM occupancy ~60.3 GB/card | `npu_samples.txt`, 5 s snapshots during runs 1–2; medium | Identify idle/serial gaps, memory or communication stalls; utilization alone is not root cause |
| Prefill kernels / KV | not yet measured on current topology | historical R16 is DP2/TP4 only | Cold 32K prefill chunk timeline, KV/indexer copies, critical path |
| Decode kernels, graph, host | not yet measured on current topology | historical R08 is DP2/TP4 c1 only | Draft/target split, launch and D2H/sync migration |
| TP8 HCCL | not yet measured | historical R39 shows contention in DP2/TP4, not transferable | Actual TP8 communication duration, bytes, overlap |
| W4A8 weight traffic | ~21.1 GB selected weights per token model-wide before batch reuse | `weight_inventory.json`; approximate | Actual rank-local selected experts, cache reuse, draft pass count |

Do not infer Compute Gap versus Framework Gap from AICore utilization alone. The next diagnostic must explain exposed time at the same workload. Historical R23 Super Kernel regression and R39 R29 regression make those paths lower-priority unless current trace contradicts prior mechanisms.
