# 执行日志

- `2026-09-27T21:56:03Z` Loop 已冻结。下一步：Admit Run579; decompose serial/concurrent task ordering with short-window all8 trace, then map legal production producer/consumer DAG

- `2026-09-27T21:56:17Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run579`（profile）。

- `2026-09-27T21:56:17Z` Run `run579` 记录为 `pass`；正确性为 `pass`。Scoped all8 terminal independent-ready production-weight GMM x full265 TP8 HCCL service PASS. Same-run slowest rank medians GMM10.266 HCCL4.180 serial14.402 concurrent16.205ms; concurrent slower on every rank. Fixed DSpark7 algorithm; no formal TPS or finite Resource/Scheduling/Product endpoint.

- `2026-09-27T22:29:42Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run580`（profile）。

- `2026-09-27T22:29:42Z` Run `run580` 记录为 `pass`；正确性为 `pass`。Scoped all8 Level0 copied timeline PASS: Model33 GMM 86 tasks and Model32 HCCL 265 tasks per arm; concurrent HCCL begins0.37-0.42ms after GMM and task envelopes overlap12.12-12.61ms, but HCCL task span inflates to15.10-15.35ms vs serial4.07-4.44ms. Unprofiled concurrent16.242ms vs serial14.573ms slowest-rank medians. Exact resource cause and Product DAG unproven; strict endpoints null.
