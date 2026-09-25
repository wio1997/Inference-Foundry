# 执行日志

- `2026-09-25T07:03:24Z` Loop 已冻结。下一步：Run175 offline eight-rank tail active-slot census and safe upper-bound triage from Run154/155 and Run99.

- `2026-09-25T07:05:12Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run175`（review）。

- `2026-09-25T07:05:13Z` Run `run175` 记录为 `pass`；正确性为 `not-applicable`。Measured Run155 cohort has296 cycles, median18.525% parked slot-cycles, first park cycle173, 44 cycles active<=6,27 active<=3,16 active<=1 across8 ranks. Warmup cohort fractions vary9.68-19.47%. Ideal linear scaling arithmetic exposure2.553s/cohort is not a measured removable bound; target GMM weight traffic and communication may be batch-insensitive. Next source audit for safe compaction and marginal latency.

- `2026-09-25T07:08:15Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run176`（review）。

- `2026-09-25T07:08:15Z` Run `run176` 记录为 `pass`；正确性为 `not-applicable`。Fixed12/96 ABI spans runtime state, metadata, target closure, acceptance and DSpark; stock graph capture sizes include48 but no c6 Extreme closure. Parked slots currently use reserved KV and freeze output. Exact slot gather/scatter and state parity would be major; first measure real TP8 c6/c12 marginal target graph latency, not infer from parked fraction.

- `2026-09-25T07:11:40Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run177`（profile）。

- `2026-09-25T07:28:31Z` Run `run177` 记录为 `pass`；正确性为 `pass`。Stock-only same-service legal c12,c8,c6,c4,c1 warmup+measured cohorts all pass, 24 FULL graph event samples per size/rank. Critical event medians53.944,49.672,49.647,47.155,39.903ms. Run155 measured tail bucket-mapped target screen0.570s/cohort before any switching/packing; impossible all-tail-c1 endpoint1.713s. Sequential order, sync perturbation and stock/Extreme boundary differences require repeat c12/drift check. Borrowed source restored/service stopped.

- `2026-09-25T07:30:56Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run178`（review）。

- `2026-09-25T07:30:56Z` Run `run178` 记录为 `pass`；正确性为 `not-applicable`。Review finds 0.570 s/cohort zero-cost bucket screen, c12 drift control needed; no implementation verdict yet; actual model ID unverified

- `2026-09-25T07:31:50Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run179`（profile）。

- `2026-09-25T07:32:05Z` Run `run179` 记录为 `invalid`；正确性为 `invalid`。Shell redirection failed before runner: output directory absent; no patch, service, benchmark or NPU work
