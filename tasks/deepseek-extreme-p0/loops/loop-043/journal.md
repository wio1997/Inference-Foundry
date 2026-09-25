# 执行日志

- `2026-09-25T02:35:37Z` Loop 已冻结。下一步：Run141: offline source/call-graph audit of actual AscendDSparkProposer _runnable and metadata build, map dynamic inputs, side effects and potential fixed replay boundary; use Run140/Run98 timings to bound achievable cycle gain.

- `2026-09-25T02:36:34Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run141`（design-check）。

- `2026-09-25T02:38:26Z` Run `run141` 记录为 `pass`；正确性为 `not-applicable`。Actual DSpark path uses eager llm_base_proposer, not Step3p5. Source divides _runnable into context-KV preparation, three-layer draft forward, gather/LMHead, and fixed seven-step Markov tail. Only Markov tail has a small self-contained replay boundary; raw logits are mutated in place and must be refreshed. Run98 device stage medians sum54.199ms versus Run140 latest-rank cadence mean54.741ms across separate cohorts, so Host CPU reduction cannot be claimed as throughput gain. Astra High advisory read-only, actual service model unverified. Next measure four segments with no timed-path sync.

- `2026-09-25T02:39:08Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run142`（profile）。

- `2026-09-25T02:59:00Z` Run `run142` 记录为 `pass`；正确性为 `not-applicable`。Legal 12/12 exact1024 eight-rank; 64 consecutive steady segment samples/rank. DSpark Markov tail median Host wall2.655ms/thread CPU2.655ms/device event0.695ms; three-layer model 23.930/23.901/2.795ms; total runnable 31.050/31.009/4.690ms. No timed-path NPU sync; source restored and service stopped. Tail replay cannot be credited with >0.695ms device interval removal or E2E gain; no implementation.

- `2026-09-25T02:59:00Z` 主控结论为 `PIVOTED`。Run142 legal eight-rank segment timing shows the smallest safe replay boundary (Markov tail) has only ~0.695ms device interval; its 2.655ms Host work appears largely overlapped with target. Whole DSpark graph requires dynamic KV/metadata semantics and remains unproven. No high-value exposed DSpark replay candidate currently justifies implementation versus the ~46.6ms target stage.
