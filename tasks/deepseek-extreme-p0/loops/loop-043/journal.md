# 执行日志

- `2026-09-25T02:35:37Z` Loop 已冻结。下一步：Run141: offline source/call-graph audit of actual AscendDSparkProposer _runnable and metadata build, map dynamic inputs, side effects and potential fixed replay boundary; use Run140/Run98 timings to bound achievable cycle gain.

- `2026-09-25T02:36:34Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run141`（design-check）。

- `2026-09-25T02:38:26Z` Run `run141` 记录为 `pass`；正确性为 `not-applicable`。Actual DSpark path uses eager llm_base_proposer, not Step3p5. Source divides _runnable into context-KV preparation, three-layer draft forward, gather/LMHead, and fixed seven-step Markov tail. Only Markov tail has a small self-contained replay boundary; raw logits are mutated in place and must be refreshed. Run98 device stage medians sum54.199ms versus Run140 latest-rank cadence mean54.741ms across separate cohorts, so Host CPU reduction cannot be claimed as throughput gain. Astra High advisory read-only, actual service model unverified. Next measure four segments with no timed-path sync.
