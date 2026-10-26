# 执行日志

- `2026-09-27T00:20:53Z` Loop 已冻结。下一步：Read-only audit Run121 and Run246-249 raw trace IDs and source instrumentation to specify minimal original-path same-cycle capture

- `2026-09-27T00:20:54Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run366`（design-check）。

- `2026-09-27T00:24:41Z` Run `run366` 记录为 `pass`；正确性为 `not-applicable`。Saved Run246 contains all8 two Target scopes and43 native GMM1/GMM2 each, but Run121 route is a separate run and cannot be joined by cycle label; design new same-trajectory capture

- `2026-09-27T00:24:50Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run367`（profile）。

- `2026-09-27T00:32:24Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run368`（design-check）。

- `2026-09-27T00:41:49Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run370`（design-check）。

- `2026-09-27T00:42:10Z` Run `run370` 记录为 `pass`；正确性为 `not-applicable`。Run246 latest all8 two Target scopes have fixed236 quant,151 plain,43 transpose GEMM tasks; shape-derived standard arithmetic totals170.204 GFLOP/rank-cycle, excluding GMM/Compressor and not a strict minimum or Product bound

- `2026-09-27T00:42:56Z` Run `run367` 记录为 `pass`；正确性为 `pass`。Combined original-path 48 warmup plus12 diagnostic outputs exact;40/40 all8 Runtime FULL Graph pass, two-cycle profile and live counts present, borrowed sources restored and service stopped; instrumented timing excluded

- `2026-09-27T00:43:05Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run369`（profile）。

- `2026-09-27T00:44:54Z` Run `run369` 记录为 `pass`；正确性为 `not-applicable`。CANN9.1 torch_npu.profiler.profiler.analyse exported all40 Level1 profile directories; initial namespace/path preflights corrected; SHA index validates CSV and trace files

- `2026-09-27T00:45:52Z` Run `run368` 记录为 `pass`；正确性为 `not-applicable`。All8 latest cohort identity gates pass, 16 same-run cycle64/65 route-counter pairs, all43 live layers cross-rank route conservation576; GMM standard arithmetic141.03-170.02GF/rank-cycle and active packed footprint8.946-9.576GB; no scheduling or Product bound

- `2026-09-27T00:47:33Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run371`（design-check）。

- `2026-09-27T00:47:33Z` Run `run371` 记录为 `pass`；正确性为 `not-applicable`。V3.4 partial standard arithmetic adds same-run GMM141-170GF plus fixed dense170.204GF and Compressor58.385GF/rank-cycle, paired GMM current read/active footprint1.064-1.079; strict compulsory, scheduling and Product numeric bounds remain null

- `2026-09-27T00:48:57Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run372`（design-check）。

- `2026-09-27T00:49:43Z` Run `run372` 记录为 `pass`；正确性为 `not-applicable`。Corrected read-only DSpark proposer census: fixed18 quant and3 transpose plus3 GMM pairs; plain MatMul varies7-16 by rank/cycle. Dense standard FLOPs partial; routed GMM, custom kernels and Product bound unresolved

- `2026-09-27T00:54:47Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run373`（review）。

- `2026-09-27T00:54:53Z` Run `run373` 记录为 `pass`；正确性为 `not-applicable`。V3.5 records corrected DSpark 28.8053 GFLOP/rank-cycle observed dense standard work, leaves all finite upper bound endpoints null

- `2026-09-27T00:55:49Z` 暂存知识变化 `PK-032`：Same-cycle original FULL Graph route joins native GMM counters, but Host proposer scope clips device tail; neither instrumented timing nor task read counts close a Product bound

- `2026-09-27T00:56:40Z` 主控结论为 `ACCEPTED`。Same-run Target route/native GMM counter attribution and corrected Draft device-tail work census are validated; timing and Product ceiling remain uncalibrated due hot-path snapshot sync and incomplete work/dependency inventory
