# 执行日志

- `2026-09-22T10:15:13Z` Loop 已冻结。下一步：Inventory the already-compiled FULL_DECODE_ONLY graph runner and its mutable input/output/cache contract at the one-time handoff, then build the smallest fixed 96-token replay probe without reintroducing ModelRunner dispatch.

- `2026-09-22T10:18:23Z` 为用例 `mixed_32k_1024_c12` 创建 Run `existing-full-decode-graph-probe-20260922`（test）。

- `2026-09-22T10:30:42Z` Run `existing-full-decode-graph-probe-20260922` 记录为 `pass`；正确性为 `pass`。First Extreme-owned target graph probe passed all eight ranks and eight real-weight cycles with exact state/host-mirror parity, identical rank state, no ModelRunner retained and no oracle target calls. Dispatcher-selected mode replay reduced median cycle wall from the refined eager 433.587 ms to 70.803 ms; internal decode-window rate rose from 38.27 to 277.18 tok/s. Longer steady run required before promotion.

- `2026-09-22T10:32:04Z` 为用例 `mixed_32k_1024_c12` 创建 Run `target-graph-64cycle-20260922`（benchmark）。

- `2026-09-22T10:43:28Z` Run `target-graph-64cycle-20260922` 记录为 `pass`；正确性为 `pass`。64 continuous target-graph cycles passed on all eight ranks with exact host/device mirrors, exact state advance and identical final rank state. Median wall 3.4103 s, 53.286 ms/cycle, 1004 emitted tokens, internal decode-window 294.400 tok/s. This is a decode-window metric, not formal E2E.

- `2026-09-22T10:43:45Z` 已记录对比（`comparable=yes`）：Same real weights, DP1xTP8, fixed c12/DSpark7, 64 cycles, no profiler, post-window exact state/mirror validation. Target graph replay improves internal decode-window TPS 38.2709->294.3999 (+669.2%) and cycle wall 433.587->53.286 ms (-87.71%). This is a valid runtime-window A/B, not comparable to the Stock 48-request HTTP E2E baseline.

- `2026-09-22T10:44:07Z` 暂存知识变化 `extreme-target-graph-replay-20260922`：Extreme Runtime can reuse the dispatcher-captured fixed target graph directly from runtime-owned 96-token buffers and metadata without retaining ModelRunner. Across 64 real-weight cycles on DP1xTP8 it preserves exact state, host mirrors and rank parity while reducing median cycle wall from 433.587 to 53.286 ms and raising internal decode-window throughput from 38.271 to 294.400 tok/s.

- `2026-09-22T10:44:21Z` 主控结论为 `ACCEPTED`。Matched 64-cycle real-weight runs establish both correctness and causality. Reusing the fixed target graph from Extreme-owned buffers preserves exact state/mirror/rank invariants and cuts cycle wall 87.71%, raising internal decode-window throughput 669.2%, with no ModelRunner or Scheduler retained in the cycle.
