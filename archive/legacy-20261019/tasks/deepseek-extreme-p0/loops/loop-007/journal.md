# 执行日志

- `2026-09-20T17:06:14Z` Loop 已冻结。下一步：Parameterize runner config assertions, stop rejected service, launch FlashComm1 true/DSA CP false, then full A/B

- `2026-09-20T17:07:40Z` 为用例 `mixed_32k_1024_c12` 创建 Run `dsa-cp-off-20260920`（benchmark）。

- `2026-09-20T17:26:36Z` Run `dsa-cp-off-20260920` 记录为 `pass`；正确性为 `pass`。Functional suite passed; three full 48/48 warmed passes median 518.10 tok/s vs baseline 543.65 (-4.70%), TTFT 1.533s vs1.332s, TPOT20.48ms vs19.73ms; no KEEP

- `2026-09-20T17:26:36Z` 已记录对比（`comparable=yes`）：Only DSA CP was disabled versus frozen FlashComm1-on baseline; same DP1TP8 model, image, source, dataset, full warmup and 48x32K-to-1024 c12. Median TPS 518.10 vs543.65 (-4.70%); TTFT +15.1%, TPOT +3.8%.

- `2026-09-20T17:26:36Z` 主控结论为 `REJECTED`。DSA CP off alone regressed full warmed E2E performance: median output TPS -4.70%, TTFT +15.1%, TPOT +3.8%; no >5% gain. Functional gate passed; numerical equivalence not needed for rejected candidate.
