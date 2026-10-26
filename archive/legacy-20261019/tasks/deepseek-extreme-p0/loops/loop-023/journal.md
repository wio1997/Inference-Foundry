# 执行日志

- `2026-09-21T06:05:32Z` Loop 已冻结。下一步：Audit the frozen launch command and DSpark configuration path, then run a bounded no-profiler k5 screen against k7 with identical warmup and 12x32K-to-512 c12 requests before a full repeated benchmark.

- `2026-09-21T06:08:15Z` 为用例 `mixed_32k_1024_c12` 创建 Run `k5-screen-20260921`（benchmark）。

- `2026-09-21T06:53:47Z` Run `k5-screen-20260921` 记录为 `error`；正确性为 `invalid`。k5 service initialization failed before readiness: cudagraph shapes must be multiples of both k+1=6 and TP8 for sequence parallelism. No port listener, correctness request, warmup or benchmark occurred.

- `2026-09-21T06:53:47Z` 主控结论为 `REJECTED`。k5 is not a runnable TP8 configuration in the current vLLM 0.26 path: graph shapes must be divisible by both 6 and 8. The service failed during KV/backend initialization, so there is no correctness or performance comparison. With model dspark_block_size>=5, the next smaller valid k below7 does not exist under this invariant.
