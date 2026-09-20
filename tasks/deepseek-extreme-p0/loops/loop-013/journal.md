# 执行日志

- `2026-09-20T19:49:37Z` Loop 已冻结。下一步：Prewarm 12 prompts and attach msprof to the live TP0 worker during a short c12 decode sample without restarting the service.

- `2026-09-20T19:49:54Z` 为用例 `mixed_32k_1024_c12` 创建 Run `decode-c12-msprof-20260920`（profile）。

- `2026-09-20T19:50:33Z` Run `decode-c12-msprof-20260920` 记录为 `invalid`；正确性为 `not-applicable`。msprof refused PID because output path was relative inside docker exec; service remained healthy. Retry with absolute shared path.

- `2026-09-20T19:50:33Z` 为用例 `mixed_32k_1024_c12` 创建 Run `decode-c12-msprof-absolute-20260920`（profile）。

- `2026-09-20T19:53:02Z` Run `decode-c12-msprof-absolute-20260920` 记录为 `invalid`；正确性为 `not-applicable`。msprof dynamic attach refused valid live TP0 PID 774466 even with absolute output path; service remained healthy. Pivot to torch-NPU profiler configured at launch.

- `2026-09-20T19:53:48Z` 为用例 `mixed_32k_1024_c12` 创建 Run `decode-c12-torch-profile-20260920`（profile）。
