# 执行日志

- `2026-09-20T17:59:53Z` Loop 已冻结。下一步：Enable torch profiler with_modules stack for a short warm window; inspect item call stacks and exact source before any patch

- `2026-09-20T18:01:28Z` 为用例 `mixed_32k_1024_c12` 创建 Run `stack-profile-20260920`（profile）。

- `2026-09-20T18:17:21Z` Run `stack-profile-20260920` 记录为 `fail`；正确性为 `not-applicable`。Four warm requests completed but torch-NPU with_modules=true segfaulted worker during stop_profile (HTTP500); offline parse lost FRAMEWORK operator traces and no item call stacks exported. Source attribution unavailable; device stack profile invalid.

- `2026-09-20T18:17:50Z` 主控结论为 `PIVOTED`。with_modules stack profiler crashed all workers during stop_profile; offline parser warned of lost data and exported no FRAMEWORK operator events, so item callsites remain unidentified
