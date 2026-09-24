# 执行日志

- `2026-09-24T14:23:03Z` Loop 已冻结。下一步：Implement opt-in cycle-scheduled profiling in ExtremeDecodeRuntime and launch one legal cohort.

- `2026-09-24T14:26:06Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run105`（profile）。

- `2026-09-24T14:27:47Z` Run `run105` 记录为 `invalid`；正确性为 `invalid`。Launch failed before service start: script lacked executable mode (bash: Permission denied, exit 126). No workload or profiler ran.

- `2026-09-24T14:28:22Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run106`（profile）。

- `2026-09-24T14:53:38Z` Run `run106` 记录为 `pass`；正确性为 `pass`。Legal 12x1024 c12 cohort completed 12/12 exact lengths and 8/8 Runtime gates at 289 cycles/rank. Cycle 64-65 profiler captured two target scopes on all eight ranks, 5.6 MB raw/rank, parsed successfully. Profiler target device-total median 51.144 ms is dominated by nested wait_event median 50.134 ms, while nested HCCL all-gather is 0.247 ms; asynchronous graph work makes CPU scope timestamp clipping invalid. Whole-trace grouped matmul kernel sum median 20.592 ms across two cycles includes target and proposer, so target-only optimization bound is not established. Diagnostic TPS 508.89 includes profiler overhead and is not comparable to formal E2E.

- `2026-09-24T14:55:22Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run107`（profile）。
