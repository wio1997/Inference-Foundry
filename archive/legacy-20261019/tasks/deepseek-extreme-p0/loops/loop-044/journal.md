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

- `2026-09-25T03:17:09Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run149`（profile）。

- `2026-09-25T03:19:47Z` Run `run149` 记录为 `invalid`；正确性为 `invalid`。GMM2 profiler completed but synthetic call omitted product W4A8 w2_scale_bias; median794us is ~10x Run107 live GMM2 ~83us/layer. Counters invalid for product attribution. Preserve full profile and retry Run150 with bias plus GMM1-produced activation/scale.

- `2026-09-25T03:20:19Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run150`（profile）。

- `2026-09-25T03:22:02Z` Run `run150` 记录为 `pass`；正确性为 `not-applicable`。Corrected one-card GMM2 with W4A8 bias and GMM1-produced input/scale: four samples median ~68us, main-memory read ~66MB, close to active packed W2 bytes and comparable order to Run107 live ~83us/layer. Together with Run148 GMM1, both are weight-read dominated at ~1TB/s effective counter rate. No demonstrated >=5ms safe GMM edit; pivot to DSA chain audit.

- `2026-09-25T03:22:33Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run151`（design-check）。

- `2026-09-25T03:25:34Z` Run `run151` 记录为 `pass`；正确性为 `not-applicable`。15 valid Run107 target windows show exact 43-layer DSA pattern: 2 c0 layers with no compressor/indexer, 21 c4 with two distinct compressors plus indexer, 20 c128 with one compressor and no indexer. Per-window compressor3.366ms, indexer1.746ms, sparse1.621ms. Post-sparse transpose occurs after alltoall in all 645 layers, so not a DSA pre-attention fusion member. No redundant compressor or semantics-safe >=5ms DSA fusion found. Initial metadata-count assertion failed and was corrected within Run151.

- `2026-09-25T03:26:09Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run152`（design-check）。

- `2026-09-25T03:28:27Z` Run `run152` 记录为 `pass`；正确性为 `not-applicable`。Run107 15 valid target windows have 33-41 strictly adjacent BF16+FP32 allGather pairs (median36), each BF16 49152 elements and FP32 3072. FP32 second calls total median0.257ms/window; mixed dtypes prevent trivial coalescing. Remaining collectives interleave compute. No >=5ms safe collective edit. Initial 43-pair assertion corrected within Run152. New formal Run99 accounting indicates larger pre/post-runtime gap; pursue Run153 offline.

- `2026-09-25T03:29:03Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run153`（profile）。

- `2026-09-25T03:39:51Z` Run `run153` 记录为 `pass`；正确性为 `not-applicable`。Run99 formal median client85.978s versus four rank0 FixedCohort walls69.040s leaves16.938s outside timer (19.7%), but cohort gap minus max TTFT median0.140s (range0.074-0.172); most belongs to first-token range. Stock cross-date cohort max TTFT median1.514s versus Run99 Extreme3.922s. Causal prefill/bootstrap split unavailable; next legal diagnostic must timestamp boundaries. No formal E2E rerun.

- `2026-09-25T03:39:51Z` 主控结论为 `PIVOTED`。Target-stage families were decomposed through Run143-152: GMM reads active W4 weights at about 1TB/s in one-card product-shape counters, peer wait dominates apparent HCCL variation, DSA compressor calls are distinct, and small mixed-dtype allGather pairs have no large direct saving. No semantics-safe >=5ms target edit is established. Run153 identifies a larger unlocalized first-token E2E range, so pivot to E2E boundary attribution.
