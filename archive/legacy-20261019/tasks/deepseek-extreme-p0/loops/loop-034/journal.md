# 执行日志

- `2026-09-22T13:06:57Z` Loop 已冻结。下一步：Implement a fixed device output drain and cohort-to-limit driver, then add the minimal bootstrap/control-plane return contract and run CPU tests before the NPU A/B.

- `2026-09-22T13:20:23Z` 为用例 `correctness_short` 创建 Run `run-20260922T132023Z`（test）。

- `2026-09-22T13:20:57Z` Run `run-20260922T132023Z` 记录为 `pass`；正确性为 `pass`。Fixed cohort serving shell stages accepted tokens on device, drains once, trims exact per-slot limits, stays vLLM-free, and the bulk output marker survives worker-to-scheduler serialization.

- `2026-09-22T13:41:09Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260922T134109Z`（test）。

- `2026-09-22T13:41:10Z` Run `run-20260922T134109Z` 记录为 `invalid`；正确性为 `invalid`。External W8A8 service restarted during launch, leaving only 37.17 GiB free; own Loop034 process tree was removed and no inference ran. Launcher now requires 60 seconds of stable-free samples.

- `2026-09-23T03:03:29Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T030100Z`（benchmark）。

- `2026-09-23T03:03:29Z` Run `run-20260923T030100Z` 记录为 `pass`；正确性为 `pass`。Warmup plus three same-protocol 48x32K-to-1024 c12 runs completed 48/48 each. Median output TPS 217.341786, median-of-runs TPOT p50 53.298 ms, TTFT p50 1942.120 ms. All 128 rank/cohort rows passed, covering four cohorts per workload across warmup plus three official runs; no per-cycle ModelRunner/Scheduler re-entry.

- `2026-09-23T03:03:29Z` 已记录对比（`comparable=yes`）：Same frozen workload/protocol: Extreme median 217.342 tok/s versus Stock 543.655 tok/s, delta -60.022%. Correctness and serving-boundary hypothesis pass, but current runtime is not performance-competitive.

- `2026-09-23T03:03:30Z` 暂存知识变化 `extreme-formal-e2e-20260923`：The first full-serving Extreme Runtime completes the frozen warm-cache 48x32K-to-1024 c12 workload correctly in four runtime-owned cohorts with one terminal bulk frame per cohort and no per-cycle ModelRunner/Scheduler re-entry. Three official runs deliver median 217.342 tok/s, TTFT p50 1942.120 ms and TPOT p50 53.298 ms, 60.022% below Stock 543.655 tok/s; rank-0 cohort wall median is 53.373 s for about 1025 cycles, so the remaining gap is sustained runtime decode rather than serving bookkeeping.

- `2026-09-23T03:03:46Z` 主控结论为 `PIVOTED`。The frozen milestone is satisfied: CPU gates pass, every official run completes 48/48 requests at exactly 1024 tokens, and 128 rank/cohort records pass with no per-cycle ModelRunner/Scheduler re-entry. TaskCtl cannot mark the mixed-history loop accepted because the preserved collision attempt is invalid; that attempt is environmental, while the formal run is valid and shows Extreme at 217.342 tok/s, 60.022% below Stock.
