# 执行日志

- `2026-09-25T09:30:31Z` Loop 已冻结。下一步：Run192 offline audit of Run107 target trace interval causality: first collective boundary, remaining per-layer communication, overlap and rank skew; then choose a specific intervention or pivot.

- `2026-09-25T09:31:58Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run192`（review）。

- `2026-09-25T09:31:58Z` Run `run192` 记录为 `pass`；正确性为 `not-applicable`。Run107 first reduce-scatter starts skew8.65-20.53ms across ranks and ends within0.014ms; its duration median6.223ms is mainly arrival exposure. Remaining259 HCCL tasks sum median5.205ms. No removable transfer claim; no service started.

- `2026-09-25T09:33:24Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run193`（review）。

- `2026-09-25T09:33:24Z` Run `run193` 记录为 `pass`；正确性为 `not-applicable`。Run107 target-entry skew20.532/8.624ms tracks first RS start skew; first RS end aligns. Cycle65 prior proposer durations span55.036-63.213ms and proposer-end to target-entry gap is mostly7.5-7.8ms. Suggest upstream proposer timing, not wire transfer, drives arrival exposure; causal intervention not yet established.

- `2026-09-25T09:34:49Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run194`（review）。

- `2026-09-25T09:34:49Z` Run `run194` 记录为 `pass`；正确性为 `not-applicable`。Run107 proposer profiled/synchronized CPU scope median55.644ms versus separate-service Run98 low-overhead proposer event6.454ms (8.62x); Run107 rank-arrival skew cannot be promoted to product removable gap. No service started.

- `2026-09-25T09:35:42Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run195`（review）。

- `2026-09-25T09:35:42Z` Run `run195` 记录为 `pass`；正确性为 `not-applicable`。Run98 low-overhead steady cycles20-279: eight-rank proposer duration spread median0.063ms, target0.090ms, p90 target2.329ms. No persistent large rank imbalance; Run107 arrival skew is not product gap. No service started.

- `2026-09-25T09:35:42Z` 主控结论为 `PIVOTED`。Run192/193 profiled first-collective wait traces target arrival skew, but Run194 proposer scope is8.62x Run98 low-overhead event, so profile skew cannot be promoted to product savings. Run195 low-overhead8-rank proposer duration spread median0.063ms and target0.090ms across260 steady cycles; no persistent large imbalance. Remaining HCCL kernels sum only about5.2ms in profile and have no concrete removable mechanism. Pivot from arrival skew toward required target compute/traffic and independent bound review.
