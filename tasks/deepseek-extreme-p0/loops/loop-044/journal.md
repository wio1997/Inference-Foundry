# 执行日志

- `2026-09-25T02:59:00Z` Loop 已冻结。下一步：Run143: reconstruct ordered target compute/communication intervals in all 15 valid Run107 windows, attribute layer-family exposure and compare unprofiled Run98 stage; estimate candidate bounds without summing overlapping kernels.

- `2026-09-25T02:59:24Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run143`（profile）。

- `2026-09-25T03:00:54Z` Run `run143` 记录为 `pass`；正确性为 `not-applicable`。Offline 15 valid Run107 synced target windows: device union median50.314ms. Timeline-exclusive coverage median GMM1 6.379ms, GMM2 3.134ms, quant matmul3.253ms, compressor3.366ms, HC pre2.961ms, cache scatter2.343ms, other compute14.099ms; communication10.248ms includes peer wait. Exclusive coverage is not causal removable gain. Next split 1413 other kernels and identify semantic family.

- `2026-09-25T03:02:56Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run144`（profile）。

- `2026-09-25T03:05:43Z` Run `run144` 记录为 `pass`；正确性为 `not-applicable`。15 valid target windows contain 48 other-compute kernel names; largest exact-name exclusive coverage is VllmQuantLightningIndexer 1.7465ms, then generic MatMul 1.68925ms, SparseAttn 1.62025ms. No single >=5ms other-compute target; multiple mandatory DSA operations cannot be summed as removable benefit. Continue peer-wait versus GMM discrimination.

- `2026-09-25T03:06:12Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run145`（profile）。

- `2026-09-25T03:08:08Z` Run `run145` 记录为 `pass`；正确性为 `not-applicable`。In 15 Run107 windows, HCCL sum median 11.281ms versus GMM sum 9.966ms. First reduce-scatter accounts for almost all HCCL variability: cross-rank start skew 20.533/8.650ms in cycles64/65 but end skew 0.010/0.0135ms. Later-rank first-call duration ~0.035ms; early-rank wait is not independent savings. Other 259 HCCL calls total stable ~4.8-5.7ms. Prioritize GMM headroom audit.

- `2026-09-25T03:08:21Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run146`（design-check）。

- `2026-09-25T03:13:29Z` Run `run146` 记录为 `pass`；正确性为 `not-applicable`。Run115 product packed weights have 8MiB W1 + 4MiB W2 per expert; Run121 active expert-layer pairs median 672.5 per rank-cycle gives conditional packed footprint 7.881GiB. Against Run107 GMM median 9.966ms this implies ~849GB/s if each active weight were loaded once. Different cohorts and no memory counters mean this is diagnostic, not a hardware bound. GMM1 skips zero-M in source; GMM2 calls torch_npu grouped matmul. Next capture actual counters.

- `2026-09-25T03:13:44Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run147`（profile）。

- `2026-09-25T03:14:46Z` Run `run147` 记录为 `invalid`；正确性为 `invalid`。Shell failed before docker exec because profile.log parent directory was absent. No GMM invocation or profiler counters. Retry with mkdir in Run148.

- `2026-09-25T03:14:59Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run148`（profile）。

- `2026-09-25T03:16:39Z` Run `run148` 记录为 `pass`；正确性为 `not-applicable`。One-card GMM1 product-shape synthetic-weight MemoryAccess profile with real 60-token/15-active-expert route: 4 samples median118.743us, main memory read129681KB, write2012.5KB, effective read1118GB/s. Read is 1.055x active W1 packed bytes. This supports weight traffic dominance but does not establish device peak or a service-level bound. Eight NPUs idle after run.
