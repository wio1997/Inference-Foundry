# 执行日志

- `2026-09-25T16:44:51Z` Loop 已冻结。下一步：Extract real config and source/Run146 route bytes; reconcile GMM floor, KV/cache and TP8 payload definitions, then freeze a representative layer dependency measurement

- `2026-09-25T16:47:04Z` 为用例 `mixed_32k_1024_c12` 创建 Run `source-workload-inventory`（design-check）。

- `2026-09-25T16:48:20Z` Run `source-workload-inventory` 记录为 `pass`；正确性为 `not-applicable`。Source/real-route inventory: 43 target layers, 21 c4/20 c128/2 uncompressed, 8.462GB active packed GMM weights and 155.676GFLOP logical matmul per rank/c12 target cycle; one-card conditional packed read7.880ms vs profiled GMM sum9.966ms. KV/cache traffic, graph TP8 capacity and non-GMM floors explicitly unknown; four legal prefill call counts7/11/12/10.

- `2026-09-25T16:50:29Z` 为用例 `mixed_32k_1024_c12` 创建 Run `kv-write-row-census`（design-check）。

- `2026-09-25T16:51:39Z` Run `kv-write-row-census` 记录为 `pass`；正确性为 `not-applicable`。Run84 legal eight-rank 256-cycle page audit reused: steady cycles64-255 c4 compressor/indexer valid rows each24 per rank-cycle (504 layer-rows across21 layers each), c128 valid rows median1, mean0.698 per rank-cycle (20 layer-rows median); first c128 write at cycle8 all ranks. Counts are writes, not KV HBM bytes or read traffic.

- `2026-09-25T16:53:55Z` 为用例 `mixed_32k_1024_c12` 创建 Run `graph-memory-counter-design`（design-check）。

- `2026-09-25T16:55:11Z` Run `graph-memory-counter-design` 记录为 `pass`；正确性为 `not-applicable`。Frozen 8-rank two-cycle FULL target graph Level1 MemoryAccess diagnostic with 48-request warmup, 12/12 correctness, per-family counter/count gates and no E2E claim; product source opt-in default unchanged and service cleanup required

- `2026-09-25T16:56:17Z` 为用例 `mixed_32k_1024_c12` 创建 Run `full-graph-memory-counters`（profile）。

- `2026-09-25T16:57:50Z` Run `full-graph-memory-counters` 记录为 `invalid`；正确性为 `invalid`。Exit1 before model load or requests: host invoked inherited Run107 script that sources container-only /usr/local/Ascend/ascend-toolkit/set_env.sh; no profile or performance data. Stop trap ran, NPU idle. Retry via docker exec boundary.

- `2026-09-25T16:58:43Z` 为用例 `mixed_32k_1024_c12` 创建 Run `container-graph-memory-counters`（profile）。

- `2026-09-25T17:20:03Z` Run `container-graph-memory-counters` 记录为 `pass`；正确性为 `pass`。Exit0 after manual host stop corrected container cleanup; legal warmup48/48 and diagnostic12/12 exact1024, 40/40 rank-cohort runtime records pass, 40 raw Level1 profile dirs captured (~473MB), no CSV exported yet; service stopped, NPU idle. Instrumented 560.174 TPS not comparable E2E.

- `2026-09-25T17:21:58Z` 为用例 `mixed_32k_1024_c12` 创建 Run `export-graph-memory-counters`（profile）。

- `2026-09-25T17:26:39Z` Run `export-graph-memory-counters` 记录为 `pass`；正确性为 `not-applicable`。Run246 profile export exit0; all80 rank-cycle target windows valid, latest16 across8 ranks; AIC+AIV total read18.965GB/write2.380GB, GMM read9.244GB or1.092x active packed. Offline parser, no new correctness or formal E2E claim.

- `2026-09-25T17:28:05Z` 为用例 `mixed_32k_1024_c12` 创建 Run `graph-traffic-attribution`（design-check）。

- `2026-09-25T17:30:30Z` Run `graph-traffic-attribution` 记录为 `pass`；正确性为 `not-applicable`。Run247 latest16 graph windows reattributed and byte-reconciled; gross read+write21.344GB/rank-cycle and non-GMM families identified. Byte/rate screens are nonbinding; hardware bound UNKNOWN.
