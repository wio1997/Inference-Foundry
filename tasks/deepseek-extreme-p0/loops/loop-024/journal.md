# 执行日志

- `2026-09-21T06:53:47Z` Loop 已冻结。下一步：Create a k7 launcher changing only max_num_batched_tokens to8288, verify resolved scheduled capacity in logs, then run correctness and the bounded c12 screen.

- `2026-09-21T06:54:45Z` 为用例 `mixed_32k_1024_c12` 创建 Run `tokens8288-screen-20260921`（benchmark）。

- `2026-09-21T08:05:27Z` Run `tokens8288-screen-20260921` 记录为 `pass`；正确性为 `pass`。max_num_batched_tokens8288 removed the scheduled-token warning and passed functional plus12/12 c12 screen. Output TPS472.871, TTFT mean2007.49ms, TPOT mean17.300ms. This is +2.77% versus median of six prior k7 diagnostic screens but -3.83% versus closest Loop020 screen, within frozen4.3% noise; no full benchmark promotion.

- `2026-09-21T08:05:27Z` 主控结论为 `REJECTED`。The capacity change is functional and removes the8096 warning, but the bounded c12 screen is within prior k7 variability:472.871 tok/s, +2.77% vs diagnostic median and -3.83% vs closest same-runner Loop020 control. It does not exceed the4.3% promotion threshold, so avoid an expensive full benchmark.
