# 执行日志

- `2026-09-22T13:06:57Z` Loop 已冻结。下一步：Implement a fixed device output drain and cohort-to-limit driver, then add the minimal bootstrap/control-plane return contract and run CPU tests before the NPU A/B.

- `2026-09-22T13:20:23Z` 为用例 `correctness_short` 创建 Run `run-20260922T132023Z`（test）。

- `2026-09-22T13:20:57Z` Run `run-20260922T132023Z` 记录为 `pass`；正确性为 `pass`。Fixed cohort serving shell stages accepted tokens on device, drains once, trims exact per-slot limits, stays vLLM-free, and the bulk output marker survives worker-to-scheduler serialization.

- `2026-09-22T13:41:09Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260922T134109Z`（test）。

- `2026-09-22T13:41:10Z` Run `run-20260922T134109Z` 记录为 `invalid`；正确性为 `invalid`。External W8A8 service restarted during launch, leaving only 37.17 GiB free; own Loop034 process tree was removed and no inference ran. Launcher now requires 60 seconds of stable-free samples.
