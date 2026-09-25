# 执行日志

- `2026-09-25T16:44:51Z` Loop 已冻结。下一步：Extract real config and source/Run146 route bytes; reconcile GMM floor, KV/cache and TP8 payload definitions, then freeze a representative layer dependency measurement

- `2026-09-25T16:47:04Z` 为用例 `mixed_32k_1024_c12` 创建 Run `source-workload-inventory`（design-check）。

- `2026-09-25T16:48:20Z` Run `source-workload-inventory` 记录为 `pass`；正确性为 `not-applicable`。Source/real-route inventory: 43 target layers, 21 c4/20 c128/2 uncompressed, 8.462GB active packed GMM weights and 155.676GFLOP logical matmul per rank/c12 target cycle; one-card conditional packed read7.880ms vs profiled GMM sum9.966ms. KV/cache traffic, graph TP8 capacity and non-GMM floors explicitly unknown; four legal prefill call counts7/11/12/10.

- `2026-09-25T16:50:29Z` 为用例 `mixed_32k_1024_c12` 创建 Run `kv-write-row-census`（design-check）。

- `2026-09-25T16:51:39Z` Run `kv-write-row-census` 记录为 `pass`；正确性为 `not-applicable`。Run84 legal eight-rank 256-cycle page audit reused: steady cycles64-255 c4 compressor/indexer valid rows each24 per rank-cycle (504 layer-rows across21 layers each), c128 valid rows median1, mean0.698 per rank-cycle (20 layer-rows median); first c128 write at cycle8 all ranks. Counts are writes, not KV HBM bytes or read traffic.
