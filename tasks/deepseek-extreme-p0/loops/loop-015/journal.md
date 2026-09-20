# 执行日志

- `2026-09-20T21:05:37Z` Loop 已冻结。下一步：Revisit retained cold TP0 trace and per-request TTFT alignment, inspect ScatterNdUpdateSk and Compressor call sites, then select one falsifiable prefill intervention.

- `2026-09-20T21:09:47Z` 为用例 `cold_32k_128_c1` 创建 Run `cold-kernel-shape-audit-20260920`（design-check）。

- `2026-09-20T21:09:47Z` Run `cold-kernel-shape-audit-20260920` 记录为 `pass`；正确性为 `not-applicable`。Original-source cold TP0 trace has ScatterNdUpdateSk shape cache[34091,32,1,512], indices[8096,2], updates[8096,1,512], 736 calls across 4 requests, median task1017us and median AIV67us; total761ms (all phases), plus Compressor ratio4 336 calls median1503us total506ms. Source ScatterNdUpdateSk enforces deterministic sort plus SyncAll because duplicates could corrupt rows. Need capture real indices and verify valid-row uniqueness before considering specialized path; current durations include decode and cannot be treated as TTFT savings.

- `2026-09-20T21:12:01Z` 为用例 `cold_32k_128_c1` 创建 Run `scatter-slot-uniqueness-20260920`（profile）。
