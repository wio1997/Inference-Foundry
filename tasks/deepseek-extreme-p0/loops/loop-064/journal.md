# 执行日志

- `2026-09-26T04:22:23Z` Loop 已冻结。下一步：Run one c4 Target layer immediate-join Graph diagnostic with SHA-guarded source, then overlap and device stream trace

- `2026-09-26T04:22:39Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run280`（test）。

- `2026-09-26T04:36:08Z` Run `run280` 记录为 `invalid`；正确性为 `invalid`。Graph capture stopped by audit_disjoint internal private-pair check: main.state/main.kv packed-cache hull overlap; this is same-branch and does not test main-indexer alias. No health, cohort, or performance data. Service stopped and CP source restored to base SHA.

- `2026-09-26T04:36:37Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run281`（test）。

- `2026-09-26T04:59:41Z` Run `run281` 记录为 `invalid`；正确性为 `invalid`。Launcher exit0 and 60/60 clients completed; all8 branch/alias markers, Graph capture and 16/16 FULL Graph Runtime rank-cohort reports passed (two cohorts), but only 2 of expected 5 cohorts entered fixed Runtime. Client TPS 311.397 warmup/277.869 bench12 is not a comparable frozen Extreme result. Source restored to base SHA. Need admission/handoff diagnosis and overlap-device timeline.

- `2026-09-26T05:00:05Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run282`（profile）。

- `2026-09-26T05:24:36Z` Run `run282` 记录为 `pass`；正确性为 `pass`。Overlap one-layer Graph/profile diagnostic exited0: 60/60 client outputs exact1024, 40/40 rank-cohort FULL Graph/runtime/host-mirror pass, all8 capture/alias markers. Latest-cohort 2-cycle per-rank device timeline has 2 aux-stream Compressor tasks/rank, median65.982us; median65.0us simultaneous main-stream indexer-query RoPE work, QLI starts later. This proves device concurrency for the selected branch, not unprofiled target/cycle or formal E2E gain. EngineDeadError coincided with post-bench service stop.

- `2026-09-26T05:25:05Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run283`（profile）。

- `2026-09-26T05:46:52Z` Run `run283` 记录为 `pass`；正确性为 `pass`。Gate diagnostic exit0, 60/60 exact1024 client outputs, 40/40 FULL rank-cohort Runtime reports, five exact 12x8 handoffs; numerical parity and formal TPS not assessed

- `2026-09-26T05:47:09Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run284`（profile）。

- `2026-09-26T06:15:51Z` Run `run284` 记录为 `pass`；正确性为 `pass`。Matched immediate control: 40/40 FULL Graph rank-cohort reports, 60/60 exact output lengths; overlap vs immediate next AllToAll paired median -39.875 us across 16 rank-cycle samples. This is local profiler evidence only, not same-state or A0-vs-candidate E2E.

- `2026-09-26T06:15:56Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run285`（test）。

- `2026-09-26T06:38:21Z` Run `run285` 记录为 `pass`；正确性为 `invalid`。12/12 exact1024 client outputs and all8 eager same-prestate five-pass rows; A0 self-repeat itself diverged (94/96 argmax, token differences), so one-layer CP fork numerical parity is inconclusive. No E2E gain or KEEP.

- `2026-09-26T06:41:55Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run286`（review）。

- `2026-09-26T07:04:26Z` Run `run286` 记录为 `invalid`；正确性为 `invalid`。Probe launcher exited1 before first Extreme handoff: optional EXTREME_CACHE_MANIFEST_DIR requires group slot bindings not supplied by this diagnostic. No ownership samples, no performance evidence; service stopped, NPUs idle. Remove manifest env and retry as Run287.

- `2026-09-26T07:04:36Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run287`（review）。

- `2026-09-26T07:32:46Z` Run `run287` 记录为 `pass`；正确性为 `pass`。Read-only original-path census exit0; 60/60 exact1024 clients, 40/40 all8 FULL Graph Runtime rank-cohorts and 11968 rank-cycles. Two owner requests/rank require 16 full new-input rows; derived compressed owner-history and nonowner-new page envelopes do not intersect. Native scatter/state/DSpark/prefix consumers remain unknown; no traffic or E2E gain claim. Service stopped and NPUs idle.

- `2026-09-26T07:33:47Z` 主控结论为 `PIVOTED`。Run284 local CP overlap advanced matched join but Run285 numerical gate is inconclusive; Run287 original FULL Graph confirms 16 owner full update rows per rank over 11968 rank-cycles yet native/lifetime consumers remain open. Astra High independently selects H003 hidden AllGather and local Q overlap as shorter-closure next scheduling test. No E2E gain or numeric ceiling.
