# 执行日志

- `2026-09-23T03:44:19Z` Loop 已冻结。下一步：Compute per-slot and per-cohort acceptance from Loop034 rank records, compare to frozen Stock counters, then obtain bounded runtime-only stage profile.

- `2026-09-23T03:46:18Z` 为用例 `mixed_32k_1024_c12` 创建 Run `loop034-acceptance-reanalysis`（review）。

- `2026-09-23T03:46:25Z` Run `loop034-acceptance-reanalysis` 记录为 `pass`；正确性为 `not-applicable`。Read-only 16-cohort rank0 reanalysis: 192 slot trajectories, median staged tokens/cycle 1.1941, 81/192 slots <=1.05, cohort-average rates 1.163-1.416, cycles 1013-1025. Stock baseline counter 2.9409 accepted drafts/iteration (approximately 3.9409 output tokens/iteration). This is a strong proposal-quality or state-alignment signal, not proof of cause; Loop034 length-only correctness does not establish long-run token parity.

- `2026-09-23T04:21:28Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T042128Z`（profile）。

- `2026-09-23T04:35:22Z` Run `run-20260923T042128Z` 记录为 `pass`；正确性为 `not-applicable`。Eight-rank real-weight 8-cycle diagnostic: rank-identical acceptance; mean tokens/slot fall 2.833 cycle1 to 1.417 cycle2; target NPU median 45.073ms, proposer 5.935ms, acceptance 0.383ms; full token parity unresolved.

- `2026-09-23T04:36:07Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T043607Z`（test）。

- `2026-09-23T04:48:38Z` Run `run-20260923T043607Z` 记录为 `fail`；正确性为 `fail`。Refreshing all KV-group slot mappings with one generic position/block-size formula caused 8-rank NPU vector-core exceptions during short real-weight decode; candidate reverted. Group-specific slot encoding must be established before any update.

- `2026-09-23T04:50:19Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T045019Z`（test）。

- `2026-09-23T05:03:50Z` Run `run-20260923T045019Z` 记录为 `fail`；正确性为 `invalid`。Bootstrap parity check indexed a 256-wide Mamba/GDN block table with logical index 1027; Stock intentionally skips slot-mapping computation for is_mamba_group. No runtime decode executed. Candidate will exclude groups without per-token slots and bound-check non-Mamba tables.

- `2026-09-23T05:03:58Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T050358Z`（test）。

- `2026-09-23T05:18:07Z` Run `run-20260923T050358Z` 记录为 `fail`；正确性为 `invalid`。Bootstrap safety gate found non-Mamba compressed KV group 1 has table width 256 but uncompressed position block index 1027; its generic slot mapping is not a valid runtime-owned update target. No continuous decode executed. Pivot to stale DSA decode start_pos metadata.

- `2026-09-23T05:19:09Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T051909Z`（test）。

- `2026-09-23T05:31:34Z` Run `run-20260923T051909Z` 记录为 `fail`；正确性为 `invalid`。DSA start_pos bootstrap gate found zero buffers because the handoff extractor only handled dict metadata; no candidate decode executed. Expand one-time nested metadata traversal before retry.

- `2026-09-23T05:31:45Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T053145Z`（test）。

- `2026-09-23T05:32:53Z` Run `run-20260923T053145Z` 记录为 `invalid`；正确性为 `invalid`。Stopped during service initialization before requests: source audit showed DSA-CP metadata stores start_pos under req_metadata, which the run6 extractor omitted. No candidate decode executed.

- `2026-09-23T05:33:14Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T053314Z`（test）。

- `2026-09-23T05:46:21Z` Run `run-20260923T053314Z` 记录为 `pass`；正确性为 `pass`。DSA-CP start_pos bootstrap parity passed and all eight real-weight ranks completed eight cycles; 160 accepted outputs versus 172 in the earlier trace, with no recovery of cycle2+ acceptance. Different request ordering prevents strict paired performance inference; candidate not kept.

- `2026-09-23T05:48:31Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T054831Z`（profile）。

- `2026-09-23T06:07:14Z` Run `run-20260923T054831Z` 记录为 `pass`；正确性为 `not-applicable`。Stock warm single-cohort 12x32K→1024 c12 completed 12/12 at 553.76 tok/s; Prometheus deltas 9151 accepted draft tokens / 3147 draft iterations = 2.908 accepted drafts/iteration. Oracle trace cross-run token equality is not established.

- `2026-09-23T06:09:04Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T060904Z`（profile）。

- `2026-09-23T06:22:42Z` Run `run-20260923T060904Z` 记录为 `pass`；正确性为 `not-applicable`。8-rank 8-cycle diagnostic passed. Target seq_lens and positions advanced from cycle 1, but DSA-CP local_seq_lens and start_pos remained at bootstrap values in sampled metadata. Short non-serving run; benchmark request reported EngineDead after expected standalone completion.

- `2026-09-23T06:22:50Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T062250Z`（test）。

- `2026-09-23T06:35:13Z` Run `run-20260923T062250Z` 记录为 `pass`；正确性为 `not-applicable`。8-rank 8-cycle DSA-CP refresh completed; start_pos and local_seq_lens track cycle, but acceptance stayed near one output/slot/cycle from cycle 3. Isolated candidate rejected; no paired throughput conclusion.
