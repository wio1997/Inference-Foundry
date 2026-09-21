# 执行日志

- `2026-09-20T17:28:41Z` Loop 已冻结。下一步：Restart frozen baseline flags with torch-NPU profiler, capture short warm and cold windows, inspect record_function scopes and device overlap

- `2026-09-20T17:30:30Z` 为用例 `mixed_32k_1024_c12` 创建 Run `scope-profile-20260920`（profile）。

- `2026-09-20T17:58:13Z` Run `scope-profile-20260920` 记录为 `pass`；正确性为 `not-applicable`。TP0 torch-NPU CPU scopes and device kernels separate short warm and cold windows; warm 5.473s, device union 4.094s, HCCL 2.269s, prepare 1.320s, draft 2.237s. Host sync item scopes 0.436s nested in prepare, causality and profiler overhead unresolved.

- `2026-09-20T17:59:15Z` 主控结论为 `ACCEPTED`。Short TP0 torch-NPU profile separates warm and cold CPU scopes and device kernels, locating 0.436s nested aten::item within warm prepare input and 1.379s rank-local device inactivity; source and exposure remain unresolved, but the diagnostic design goal is satisfied.
