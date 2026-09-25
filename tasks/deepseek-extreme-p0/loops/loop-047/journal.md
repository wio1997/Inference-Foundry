# 执行日志

- `2026-09-25T07:03:24Z` Loop 已冻结。下一步：Run175 offline eight-rank tail active-slot census and safe upper-bound triage from Run154/155 and Run99.

- `2026-09-25T07:05:12Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run175`（review）。

- `2026-09-25T07:05:13Z` Run `run175` 记录为 `pass`；正确性为 `not-applicable`。Measured Run155 cohort has296 cycles, median18.525% parked slot-cycles, first park cycle173, 44 cycles active<=6,27 active<=3,16 active<=1 across8 ranks. Warmup cohort fractions vary9.68-19.47%. Ideal linear scaling arithmetic exposure2.553s/cohort is not a measured removable bound; target GMM weight traffic and communication may be batch-insensitive. Next source audit for safe compaction and marginal latency.

- `2026-09-25T07:08:15Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run176`（review）。

- `2026-09-25T07:08:15Z` Run `run176` 记录为 `pass`；正确性为 `not-applicable`。Fixed12/96 ABI spans runtime state, metadata, target closure, acceptance and DSpark; stock graph capture sizes include48 but no c6 Extreme closure. Parked slots currently use reserved KV and freeze output. Exact slot gather/scatter and state parity would be major; first measure real TP8 c6/c12 marginal target graph latency, not infer from parked fraction.
