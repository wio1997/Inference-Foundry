# 执行日志

- `2026-09-20T16:56:35Z` Loop 已冻结。下一步：Add stable functional suite, run full-dataset warmup and three same-protocol passes on current healthy service

- `2026-09-20T16:57:36Z` 为用例 `mixed_32k_1024_c12` 创建 Run `sp-dsacp-functional-perf-20260920`（benchmark）。

- `2026-09-20T17:05:15Z` Run `sp-dsacp-functional-perf-20260920` 记录为 `pass`；正确性为 `pass`。Stable functional suite 3/3 plus four long outputs passed; three full 48/48 warmed passes median 534.88 tok/s versus baseline 543.65 (-1.61%); numerical equivalence not established, no KEEP

- `2026-09-20T17:05:15Z` 已记录对比（`comparable=yes`）：Same DP1TP8 model, image, source, dataset, 48x32K-to-1024 c12 and full-dataset warmup. Candidate median TPS 534.88 vs baseline 543.65 (-1.61%, inside 4.3% baseline spread); TTFT +7.8%, TPOT +0.7%, no >5% benefit.

- `2026-09-20T17:05:15Z` 主控结论为 `REJECTED`。Coupled SP/DSA-CP-off path has no robust E2E benefit: median output TPS -1.61% vs baseline and inside observed noise; TTFT median worsened ~7.8%, TPOT ~0.7%. Functional check passed but numerical equivalence remains unproven.
