# 执行日志

- `2026-09-25T07:51:07Z` Loop 已冻结。下一步：Run181 offline correlate Run165 prefill CPU trace and HostToDevice flows to source callsites for the 83-token forward; estimate path-specific removable time and select a narrowly instrumented legal 8-rank follow-up

- `2026-09-25T07:55:59Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run181`（profile）。

- `2026-09-25T07:55:59Z` Run `run181` 记录为 `pass`；正确性为 `not-applicable`。Run165 profiled 83-token rank0 prefill has 43 DSA scopes totaling187.371ms and 43 MoE scopes totaling207.733ms, nonoverlapping; direct-child-uncovered 80.091+82.923ms includes profiler/Python overhead, no removable-time claim
