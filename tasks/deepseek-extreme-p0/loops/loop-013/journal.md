# 执行日志

- `2026-09-20T19:49:37Z` Loop 已冻结。下一步：Prewarm 12 prompts and attach msprof to the live TP0 worker during a short c12 decode sample without restarting the service.

- `2026-09-20T19:49:54Z` 为用例 `mixed_32k_1024_c12` 创建 Run `decode-c12-msprof-20260920`（profile）。

- `2026-09-20T19:50:33Z` Run `decode-c12-msprof-20260920` 记录为 `invalid`；正确性为 `not-applicable`。msprof refused PID because output path was relative inside docker exec; service remained healthy. Retry with absolute shared path.

- `2026-09-20T19:50:33Z` 为用例 `mixed_32k_1024_c12` 创建 Run `decode-c12-msprof-absolute-20260920`（profile）。

- `2026-09-20T19:53:02Z` Run `decode-c12-msprof-absolute-20260920` 记录为 `invalid`；正确性为 `not-applicable`。msprof dynamic attach refused valid live TP0 PID 774466 even with absolute output path; service remained healthy. Pivot to torch-NPU profiler configured at launch.

- `2026-09-20T19:53:48Z` 为用例 `mixed_32k_1024_c12` 创建 Run `decode-c12-torch-profile-20260920`（profile）。

- `2026-09-20T20:06:17Z` Run `decode-c12-torch-profile-20260920` 记录为 `invalid`；正确性为 `not-applicable`。All 48 warmup requests returned one token but bench harness marked them fail because TPOT is undefined for length1; profiler was never started; service healthy. Retry with 128-token warmup.

- `2026-09-20T20:06:17Z` 为用例 `mixed_32k_1024_c12` 创建 Run `decode-c12-torch-profile-retry-20260920`（profile）。

- `2026-09-20T20:26:35Z` Run `decode-c12-torch-profile-retry-20260920` 记录为 `pass`；正确性为 `pass`。12/12 c12×512 completed under torch-NPU profiler. Across 16.459 s request window TP0 device busy13.038s, HCCL5.510s, compute/copy7.736s, 170 draft host scopes total9.036s (median52.894ms). Eight-rank HCCL spans range2.236-5.843s with similar compute7.63-7.75s. Diagnostic only; 387.7 profiled TPS is not official throughput.

- `2026-09-20T20:26:35Z` 主控结论为 `PIVOTED`。Valid c12 trace isolates large draft host and TP8 HCCL/idle envelopes but does not prove either removable. The candidate DSpark draft graph path is hard-disabled in source; prior upstream graph attempt was reverted. Further source-level cause and numerical constraints are needed before an E2E optimization.
