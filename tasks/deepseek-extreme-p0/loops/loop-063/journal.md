# 执行日志

- `2026-09-26T02:54:51Z` Loop 已冻结。下一步：Run269: source-level semantic/alias/lifetime audit and fixed-buffer private scratch design; reject or revise the overlap before running 8 NPUs if dependencies do not hold

- `2026-09-26T02:57:02Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run269`（design-check）。

- `2026-09-26T02:57:02Z` Run `run269` 记录为 `pass`；正确性为 `not-applicable`。Acceptance determines next target geometry/metadata independently of next DSpark draft; old DSpark reads current geometry/IDs; private scratch and post-proposer commit required; serving parking must invalidate. Side-stream native op reentrancy remains empirical gate.

- `2026-09-26T03:01:07Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run270`（test）。

- `2026-09-26T03:03:28Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run271`（design-check）。

- `2026-09-26T03:03:28Z` Run `run271` 记录为 `pass`；正确性为 `not-applicable`。Partial canonical target arithmetic: routed GMM 155.675787264 GFLOP conditional route and fixed Compressor 58.38471168 GFLOP per rank-cycle; known parameter tensor footprint is not compulsory HBM. Full resource bound remains null. Corrected DSpark state-advance semantic RAW edge and candidate formal status.

- `2026-09-26T03:04:44Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run272`（test）。

- `2026-09-26T03:04:44Z` Run `run272` 记录为 `pass`；正确性为 `pass`。CPU mock verifies private metadata destinations do not alias active target, then private update+commit matches serial reference for RoPE, SAS, QLI, group lengths and SWA slots; hardware/Graph correctness still pending Run270

- `2026-09-26T03:09:39Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run273`（review）。

- `2026-09-26T03:09:39Z` Run `run273` 记录为 `pass`；正确性为 `not-applicable`。Independent review: P0 verify path overwrites candidate with reference before Target; Run270 parity-only. Stream dependency/parking order appears sound; require true storage interval audit, candidate-consumed 8-rank correctness and verify-off A0/A/B timing. Larger DSA fanout gap remains open.

- `2026-09-26T03:10:53Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run274`（test）。

- `2026-09-26T03:10:53Z` Run `run274` 记录为 `pass`；正确性为 `pass`。CPU smoke confirms conservative interval audit distinguishes disjoint views and overlapping offset/stride views; live NPU scratch-vs-KV audit pending next diagnostic

- `2026-09-26T03:19:28Z` Run `run270` 记录为 `pass`；正确性为 `pass`。Serial private-scratch parity-only diagnostic exit0: warmup48/48, bench12/12 exact 1024, 40/40 rank-cohort rows FULL Graph, all eight ranks logged stable geometry/RoPE/SAS/QLI header parity through cycle256 in five cohorts; no mismatch. Initial verify implementation rewrote active metadata with reference before Target, so candidate-consumed correctness and timing are NOT established. Borrowed source SHA restored, service stopped, eight NPUs idle.

- `2026-09-26T03:21:09Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run275`（test）。

- `2026-09-26T03:33:53Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run276`（review）。

- `2026-09-26T03:34:13Z` Run `run276` 记录为 `pass`；正确性为 `not-applicable`。Astra High corrected frozen DSA CP path and identified c4 main-compressor/indexer-QLI fan-out as next scheduling candidate; benefit unmeasured

- `2026-09-26T03:40:28Z` Run `run275` 记录为 `pass`；正确性为 `pass`。Exit0: overlap candidate restored after same-state serial-header check and consumed by Target; warmup48+bench12 all exact1024, 40/40 8-rank FULL Graph rows pass, 40/40 scratch storage audits pass with zero aliases, verify markers 1/64/128/256 on all 8 ranks across 5 cohorts; verification-on TPS invalid for performance

- `2026-09-26T03:40:36Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run277`（benchmark）。

- `2026-09-26T03:47:36Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run279`（design-check）。

- `2026-09-26T03:47:51Z` Run `run279` 记录为 `pass`；正确性为 `not-applicable`。SHA-guarded CP fork patch preview compiled without installation; one-layer/immediate/overlap causal protocol documented, no hardware claim

- `2026-09-26T03:51:21Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run278`（benchmark）。

- `2026-09-26T03:59:28Z` Run `run277` 记录为 `pass`；正确性为 `pass`。Exit0: verify-off overlap B warmup48+bench48, 64/64 8-rank FULL Graph rows pass, 48/48 exact1024; screening TPS605.801, 1186 cycles, latest-rank runtime69.205s=58.351ms/cycle; all 64 lifetime records close launches=commits+invalidations; single pass not KEEP

- `2026-09-26T04:19:11Z` Run `run278` 记录为 `invalid`；正确性为 `pass`。Launcher exit127 after same-script edit during execution; complete warmup48+bench48 JSON and 64/64 8-rank FULL Graph rows pass, 48/48 exact1024. A0 diagnostic TPS574.437, 1185 cycles, latest-rank67.402s=56.879ms/cycle. Data retained for diagnostic B comparison only, not formal acceptance

- `2026-09-26T04:20:10Z` 主控结论为 `PIVOTED`。Run275 candidate-consumed continuous correctness/alias passed, but no-verifier B versus A0 latest-rank runtime was +1.472ms/cycle slower across all four cohorts; client TPS difference tracked -6.233s residual, not exposed decode saving. Run278 launcher exit127 after active-script edit makes comparison diagnostic-only. No formal baseline promotion.
