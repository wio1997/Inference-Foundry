# 执行日志

- `2026-09-25T09:30:31Z` Loop 已冻结。下一步：Run192 offline audit of Run107 target trace interval causality: first collective boundary, remaining per-layer communication, overlap and rank skew; then choose a specific intervention or pivot.

- `2026-09-25T09:31:58Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run192`（review）。

- `2026-09-25T09:31:58Z` Run `run192` 记录为 `pass`；正确性为 `not-applicable`。Run107 first reduce-scatter starts skew8.65-20.53ms across ranks and ends within0.014ms; its duration median6.223ms is mainly arrival exposure. Remaining259 HCCL tasks sum median5.205ms. No removable transfer claim; no service started.
