# 执行日志

- `2026-09-20T15:59:58Z` Loop 已冻结。下一步：Run cold four-prompt baseline, then system profile or restart with app profiler if needed

- `2026-09-20T16:00:25Z` 为用例 `cold_32k_128_c1` 创建 Run `run-20260920T160025Z`（benchmark）。

- `2026-09-20T16:01:25Z` 为用例 `cold_32k_128_c1` 创建 Run `run-20260920T160125Z`（benchmark）。

- `2026-09-20T16:04:49Z` Run `run-20260920T160025Z` 记录为 `pass`；正确性为 `pass`。Four distinct uncached 32K prompts, 4/4, TTFT mean 2662.7ms, TPOT mean 16.35ms

- `2026-09-20T16:04:49Z` Run `run-20260920T160125Z` 记录为 `pass`；正确性为 `pass`。Second four distinct uncached 32K prompts, 4/4, TTFT mean 2627.2ms; cache hits 0/131404 query tokens

- `2026-09-20T16:04:49Z` 为用例 `cold_32k_128_c1` 创建 Run `run-20260920T160449Z`（profile）。

- `2026-09-20T16:05:00Z` Run `run-20260920T160449Z` 记录为 `pass`；正确性为 `pass`。System-level msprof on device0 collected HBM/HCCS/AICore during four cold prompts; no per-op kernel timeline, so insufficient for causal selection

- `2026-09-20T16:31:52Z` 主控结论为 `PIVOTED`。TP0 msprof distinguishes cold and warm compute/HCCL; warm HCCL union 1.165s of 3.401s with ~0.054s compute overlap, justifying a falsifiable FlashComm1 A/B; no same-condition optimization comparison yet
