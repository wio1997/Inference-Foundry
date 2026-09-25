# 执行日志

- `2026-09-25T02:59:00Z` Loop 已冻结。下一步：Run143: reconstruct ordered target compute/communication intervals in all 15 valid Run107 windows, attribute layer-family exposure and compare unprofiled Run98 stage; estimate candidate bounds without summing overlapping kernels.

- `2026-09-25T02:59:24Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run143`（profile）。

- `2026-09-25T03:00:54Z` Run `run143` 记录为 `pass`；正确性为 `not-applicable`。Offline 15 valid Run107 synced target windows: device union median50.314ms. Timeline-exclusive coverage median GMM1 6.379ms, GMM2 3.134ms, quant matmul3.253ms, compressor3.366ms, HC pre2.961ms, cache scatter2.343ms, other compute14.099ms; communication10.248ms includes peer wait. Exclusive coverage is not causal removable gain. Next split 1413 other kernels and identify semantic family.

- `2026-09-25T03:02:56Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run144`（profile）。

- `2026-09-25T03:05:43Z` Run `run144` 记录为 `pass`；正确性为 `not-applicable`。15 valid target windows contain 48 other-compute kernel names; largest exact-name exclusive coverage is VllmQuantLightningIndexer 1.7465ms, then generic MatMul 1.68925ms, SparseAttn 1.62025ms. No single >=5ms other-compute target; multiple mandatory DSA operations cannot be summed as removable benefit. Continue peer-wait versus GMM discrimination.
