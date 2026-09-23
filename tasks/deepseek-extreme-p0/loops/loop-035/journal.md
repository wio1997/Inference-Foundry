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

- `2026-09-23T06:38:47Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T063847Z`（review）。

- `2026-09-23T06:38:54Z` Run `run-20260923T063847Z` 记录为 `pass`；正确性为 `pass`。Across three 8-cycle real-weight traces, draft carry, accepted last-token carry and num_computed advance each exact in 84/84 adjacent slot transitions. First-draft/target first prediction match falls to 0-4/12 in many cycles after cycle 0; this localizes low acceptance upstream of state carry, without distinguishing target from proposer.

- `2026-09-23T06:40:32Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T064032Z`（test）。

- `2026-09-23T06:53:18Z` Run `run-20260923T064032Z` 记录为 `pass`；正确性为 `not-applicable`。SWA slot refresh passed exact bootstrap gate and eight real-weight cycles on all ranks, but continued low acceptance (roughly 1.25-1.83 outputs/slot/cycle after cycle0). Isolated candidate rejected, no E2E claim.

- 2026-09-23 06:54 UTC: corrected transcription of Run10 total_outputs_8_cycles from 173 to 161 using rank0 per-cycle counts; summary.json and all eight rank records are the source. No experiment or verdict changed.

- `2026-09-23T06:56:10Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T065610Z`（test）。

- `2026-09-23T07:09:47Z` Run `run-20260923T065610Z` 记录为 `invalid`；正确性为 `fail`。Oracle DSA builder refresh executed 8 cycles, but host mirror exactness failed on all ranks because diagnostic callback wrote shared DSpark CPU seq_lens mirror. 211-output trend is invalid; retry with private builder common metadata.

- `2026-09-23T07:09:59Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T070959Z`（test）。

- `2026-09-23T07:24:22Z` Run `run-20260923T070959Z` 记录为 `pass`；正确性为 `not-applicable`。Private-mirror DSA builder oracle diagnostic passed local state gates on all eight ranks; 209 outputs over eight cycles, but Stock same-state token oracle and GDN metadata parity remain open. Eager diagnostic is not a performance candidate.

- `2026-09-23T07:25:12Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T072512Z`（test）。

- `2026-09-23T07:39:57Z` Run `run-20260923T072512Z` 记录为 `pass`；正确性为 `not-applicable`。DSA+GDN per-cycle builder oracle diagnostic passed local gates on all eight ranks; 215 outputs over eight cycles versus unordered DSA-only 209, no major recovery toward Stock and no same-state token parity claim.

- `2026-09-23T07:42:50Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T074250Z`（test）。

- `2026-09-23T07:57:40Z` Run `run-20260923T074250Z` 记录为 `pass`；正确性为 `fail`。Same-target/same-acceptance proposer comparison restored 69 captured state entries exactly on all ranks, but Product and Stock drafts matched only 71/84 tokens; two count=1 slots differed. Need input-field and Stock self-replay controls before attributing semantic mismatch.

- `2026-09-23T07:58:37Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T075837Z`（test）。

- `2026-09-23T08:14:34Z` Run `run-20260923T075837Z` 记录为 `pass`；正确性为 `invalid`。All 8 ranks matched Product/Stock proposer input and prepare fields; target-cache restore exact, but Stock self-replay differs 17/84 tokens because future proposer KV slots were not captured. Draft parity remains unproven.

- `2026-09-23T08:16:04Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T081604Z`（test）。

- `2026-09-23T08:29:08Z` Run `run-20260923T081604Z` 记录为 `pass`；正确性为 `invalid`。All 8 ranks: draft groups 2/3 and 16 cache specs extended to future slots, snapshots restore exactly, yet Stock self-replay matches only 69/84 tokens. Product/Stock 77/84; parity control still invalid.

- `2026-09-23T08:30:39Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T083039Z`（test）。

- `2026-09-23T08:43:27Z` Run `run-20260923T083039Z` 记录为 `pass`；正确性为 `invalid`。All 8 ranks: kernel-block-aware 16-cache snapshot exact; Stock self replay 75/84, Product/Stock 67/84, but all 12 first draft tokens equal in both comparisons. Deterministic full draft parity remains invalid.

- `2026-09-23T08:43:46Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T084346Z`（profile）。

- `2026-09-23T08:57:45Z` Run `run-20260923T084346Z` 记录为 `pass`；正确性为 `pass`。8/8 ranks completed 1024 Extreme-owned cycles with exact state/mirrors. NPU event medians after warmup: target 45.51ms, proposer 5.94ms, acceptance 0.305ms. Outputs/slot/cycle decay from 1.625 in cycles0-7 to exactly1.000 in cycles960-1023; diagnostic TPS is not formal E2E.

- `2026-09-23T08:59:24Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T085924Z`（test）。

- `2026-09-23T09:14:03Z` Run `run-20260923T085924Z` 记录为 `pass`；正确性为 `pass`。8/8 ranks completed 256 eager-target cycles with DSA+GDN builder refresh and exact local state/mirrors. Outputs/slot/cycle still fell to 1.176 in cycles64-127 and 1.036 in 128-191; builder refresh alone does not prevent collapse. Client TPS invalid.

- `2026-09-23T09:14:29Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T091429Z`（test）。

- `2026-09-23T09:27:52Z` Run `run-20260923T091429Z` 记录为 `pass`；正确性为 `pass`。8/8 rank bootstrap parity exact; draft gid2 context slots stale 96/96 at cycle1 while gid3 exact. Isolated refresh completed 256 cycles but outputs/slot/cycle fell to 1.013 in cycles64-127, so no acceptance recovery; reject isolated candidate.

- `2026-09-23T09:28:14Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T092814Z`（test）。
