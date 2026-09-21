# 执行日志

- `2026-09-20T15:39:40Z` Loop 已冻结。下一步：Wait for service ready, deterministic smoke, then frozen 48-request benchmark

- `2026-09-20T15:41:06Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260920T154106Z`（benchmark）。

- `2026-09-20T15:50:37Z` Run `run-20260920T154106Z` 记录为 `invalid`；正确性为 `invalid`。48 HTTP requests completed but custom SSE parser ignored DeepSeek reasoning field; TTFT/TPOT invalid, output only 3/48 counted

- `2026-09-20T15:50:38Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260920T155037Z`（benchmark）。

- `2026-09-20T15:57:18Z` Run `run-20260920T155037Z` 记录为 `pass`；正确性为 `pass`。Two warm-cache 48/48 runs: 543.65 and 523.15 output tok/s; TTFT 1120.9/1372.4 ms; TPOT 19.45/20.08 ms. Identical 49152 generated tokens/run; deterministic smoke 2/2.

- `2026-09-20T15:57:18Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260920T155718Z`（benchmark）。

- `2026-09-20T15:58:30Z` Run `run-20260920T155718Z` 记录为 `pass`；正确性为 `pass`。Third warm-cache 48/48 repeat: output TPS 545.85, TTFT mean 1331.8ms, TPOT mean 19.73ms; prefix hit 99.75%, spec acceptance 41.21%

- `2026-09-20T15:58:30Z` 暂存用例 `mixed_32k_1024_c12` 的基线变化，来源 Run `run-20260920T155037Z`。

- `2026-09-20T15:58:30Z` 暂存知识变化 `warm_cache_baseline`：At current DP1 TP8 source, warm-cache 48x32K-to-1024 c12 median output TPS is 543.65 across three corrected 48/48 runs; TPOT mean median 19.73ms; run TPS span 4.3%.

- `2026-09-20T15:58:54Z` 主控结论为 `PIVOTED`。Initial SSE client omitted DeepSeek reasoning deltas, making first TTFT/TPOT invalid; corrected parser and preserved evidence
