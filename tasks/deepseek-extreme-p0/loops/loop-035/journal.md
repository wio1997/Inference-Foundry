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

- `2026-09-23T09:42:30Z` Run `run-20260923T092814Z` 记录为 `pass`；正确性为 `pass`。8/8 ranks passed 256 cycles with combined DSA+GDN builder and draft gid2 slot refresh; bootstrap slot parity exact, gid2 stale 96/96 at cycle1. Outputs/slot/cycle still 1.013 in cycles64-127 and 1.003 in 128-191. Reject metadata patch route.

- `2026-09-23T09:45:11Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T094511Z`（test）。

- `2026-09-23T10:01:23Z` Run `run-20260923T094511Z` 记录为 `invalid`；正确性为 `invalid`。Run22 completed 8x256 cycles but oracle callback was never invoked; target slot hypothesis untested; hook wiring required.

- `2026-09-23T10:05:52Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T100552Z`（test）。

- 2026-09-23 correction: DirectTargetHandoff.forward did not invoke diagnostic_metadata_refresh before a11a71b. Run12/13/19/21 metadata-builder claims were unexercised; preserve their raw cycle/state evidence but withdraw causal conclusions. Run22 is invalid for the same reason. Run23 has the callback connected.

- `2026-09-23T10:20:19Z` Run `run-20260923T100552Z` 记录为 `pass`；正确性为 `pass`。8/8 ranks completed 256 cycles; connected oracle audit confirms cycle0 group parity and cycle1 slot changes. Combined DSA+GDN builder plus target slot refresh reached 7.48 outputs/slot/cycle in cycles128-191; token-level oracle and causal isolation remain open.

- `2026-09-23T10:21:51Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T102151Z`（test）。

Run23 (connected combined oracle) passed 8/8 rank local gates for 256 eager target cycles. All six target slot groups matched at cycle 0; five groups changed 96/96 slots by cycle 1, and the audit contained 24 entries/rank. Outputs/slot/cycle reached 7.480 in cycles 128-191 and 7.728 in 192-255, with long stretches where all 12 slots accepted all eight outputs. This is a strong combined-effect signal but suspiciously exceeds Stock's roughly 3.908 long-run output rate; false acceptance or degenerate token loops remain possible. Do not claim a semantic fix or E2E improvement. Run24 isolates connected DSA+GDN builder refresh without slot recomputation, requiring 256 callback invocations; a token-level oracle remains necessary.

- `2026-09-23T10:37:10Z` Run `run-20260923T102151Z` 记录为 `pass`；正确性为 `pass`。8/8 rank 256-cycle builder-only control; 256 callback calls/rank. Outputs/slot/cycle 1.836 in cycles128-191, no Run23 near-all-accepted regime; slot-only isolation and long token oracle required.

- `2026-09-23T10:37:29Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T103729Z`（test）。

Run24 isolated the connected DSA+GDN builder callback: all 8 ranks completed 256 eager target cycles, with exactly 256 callback invocations/rank, exact state/mirror checks, and no target-slot audit entries. Outputs/slot/cycle were 1.836 in cycles 128-191 and 1.814 in 192-255. This does not show Run23's late near-all-accepted behavior. Run25 will isolate the reference all-group physical slot update while leaving existing metadata objects unchanged, to distinguish slot effect from the combination. Different prompt admission prevents a strict paired numerical delta; token-level API oracle remains mandatory.

- `2026-09-23T10:52:30Z` Run `run-20260923T103729Z` 记录为 `pass`；正确性为 `pass`。8/8 rank 256-cycle slot-only control, 256 callback calls/rank and cycle0 slot parity. Outputs/slot/cycle 1.065 in cycles128-191; no Run23 near-all-accepted regime. Combined interaction and token correctness remain open.

- `2026-09-23T10:52:50Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T105250Z`（test）。

Run25 isolated reference all-group target slot refresh with builder rebuild skipped: 8/8 ranks passed 256 eager cycles, with 256 callback calls, 24 audit entries/rank, and cycle-0 equality. At cycle 1 five groups changed 96/96 slots, while group 3 remained equal. Yet outputs/slot/cycle fell to 1.065 in cycles 128-191 and 1.079 in 192-255. Run23's near-all-accepted regime therefore requires the combination of metadata rebuild and slot refresh on these request cohorts; it is not a validated semantic fix. Run26 now collects exact 12x32K->1024 Stock token IDs through the API return_token_ids option. A matched Extreme serving capture and first-divergence comparison follow.

- `2026-09-23T11:06:03Z` Run `run-20260923T105250Z` 记录为 `pass`；正确性为 `pass`。Stock API token oracle returned 12/12 unique prompt hashes and exact 1024 output token IDs/request, all finish_reason=length; paired Extreme capture pending.

- `2026-09-23T11:06:33Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T110633Z`（test）。

Run26 collected Stock exact token oracle through Chat Completions return_token_ids: all 12 fixed dataset requests succeeded, all 12 prompt token hashes were unique, prompt lengths were 32,851 for eleven requests and 32,853 for one, and every response contained exactly 1,024 token IDs with finish_reason=length. This is a correctness reference only, not a repeated throughput baseline. Run27 is loading combined metadata+slot oracle Extreme serving for a same-request API token comparison.

- `2026-09-23T11:25:02Z` Run `run-20260923T110633Z` 记录为 `pass`；正确性为 `invalid`。Extreme combined oracle serving completed 12/12 1024-token outputs on 8 ranks, 292 cycles first cohort. Stock-vs-Extreme matched all prompt hashes but 0/12 exact; Extreme self-repeat also 0/12 exact, so cross-run token mismatch cannot assign a semantic cause. Stock self-repeat and same-state control next.

- `2026-09-23T11:25:21Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T112521Z`（test）。

Run27 combined metadata+slot oracle in Extreme serving returned 12/12 length-exact 1,024-token API outputs on all eight ranks (first cohort 292 cycles, 139.5 s). Prompt token hashes matched Stock for all 12 requests, but no complete output sequence matched; first cross-service mismatch positions were 7-106. An Extreme self-repeat in the same service also produced 0/12 exact sequences with first mismatches 9-163. Therefore cross-run token differences are confounded by observed self-replay variation and cannot by themselves prove a product semantic error or correctness of Run23's high acceptance. Stock same-service self-repeat is now running; afterward use same-state target/proposer/acceptance gates rather than unrelated service generations to locate a causal first divergence.

- `2026-09-23T11:38:43Z` Run `run-20260923T112521Z` 记录为 `pass`；正确性为 `invalid`。Stock same-service replay completed two 12x32K->1024 cohorts, both 12/12 length-exact and prompt-matched; 0/12 outputs exact between repeats, first mismatch token 7-78. Cross-service token sequence comparison is nondiagnostic; same-state paired oracle required.

Run28 Stock same-service self-repeat returned two complete 12-request, 1,024-token API cohorts with matching prompt hashes, yet 0/12 output sequences matched exactly; first mismatch positions were 7-78. Extreme self-repeat first mismatch positions were 9-163, while Stock-vs-Extreme positions were 7-106. Thus independent service/cohort token sequences are nondeterministic at the same scale as the proposed cross-runtime difference. The combined Run23 acceptance rise remains unverified for correctness, but the cross-service token oracle cannot adjudicate it. Next freeze one in-process target/proposer/KV state, include self-replay noise controls, and compare candidate mutations at that state.

- `2026-09-23T11:41:28Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T114128Z`（test）。

- `2026-09-23T11:55:01Z` Run `run-20260923T114128Z` 记录为 `pass`；正确性为 `pass`。8/8 ranks passed 8-cycle same-state target self-replay, 69 snapshot entries, no skipped rows and exact restores. Target argmax differs 1-9/96 positions/cycle despite exact state; acceptance counts equal 12/12. Establishes paired noise baseline for A-B-A metadata discriminator.

Run29 established a same-state target replay noise baseline on 8 TP ranks for 8 c12 cycles. Each target A/B pair used the same fixed inputs, with 69 physical cache/mutable snapshots, zero skipped cache entries, and exact restoration before and after replay. State advance and Host mirrors remained exact. Yet target argmax matched only 87-95 of 96 positions by cycle; 12/12 acceptance counts matched every cycle and sampled output differed in at most one padded position in three cycles. Thus cross-run token variability has an in-process target source even after KV restoration. Candidate metadata/slot effects must be compared using A-B-A within the same state, with A-C establishing noise and restoration validity.

- `2026-09-23T11:56:57Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T115657Z`（test）。

- `2026-09-23T12:10:04Z` Run `run-20260923T115657Z` 记录为 `invalid`；正确性为 `invalid`。Run30 8/8 rank physical KV snapshots restored exactly, but cycle1 A-C target argmax match only 66/96 versus A-B 70/96; B-C 87/96. Metadata builder mutates shared derived tensors not restored by old dict/slot values, so candidate delta cannot be interpreted. Trace alias and rerun A-B-C.

- `2026-09-23T12:13:30Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T121330Z`（test）。

- `2026-09-23T12:26:40Z` Run `run-20260923T121330Z` 记录为 `invalid`；正确性为 `invalid`。Run31 8-rank A/B/C local integrity passes, common positions/seq_lens and 138 physical KV entries restore exactly; cycle1 A/C target argmax still 71/96 (A/B 71/96), so mutable metadata or untracked KV aliases invalidate candidate discrimination

- `2026-09-23T12:27:13Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T122713Z`（test）。

- `2026-09-23T12:40:11Z` Run `run-20260923T122713Z` 记录为 `pass`；正确性为 `invalid`。Run32 valid 8-rank same-state A/B/C discriminator: exact physical KV, common and 72 old metadata tensor restores; cycle1 A/C 92/96 target argmax versus A/B 75/96, two acceptance-count slots changed. Combined metadata+slot refresh causally changes target, but semantic correctness and long-run acceptance remain unverified.

- `2026-09-23T12:40:51Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T124051Z`（test）。

- `2026-09-23T12:53:36Z` Run `run-20260923T124051Z` 记录为 `pass`；正确性为 `invalid`。Run33 builder path same-state A/B/C on 8 ranks: cycle1 A/C 91/96 within replay noise, A/B 64/96, two acceptance counts changed; builder also rewrote SWA group2 slots 96/96. Causal target effect, no semantic correctness or acceptance recovery claim.

- `2026-09-23T12:53:44Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T125344Z`（test）。

- `2026-09-23T13:06:17Z` Run `run-20260923T125344Z` 记录为 `pass`；正确性为 `invalid`。Run34 slot-only A/B/C 8-rank valid restore: cycle1 five group slots changed 96/96, but target argmax A/B=88/96 and A/C=88/96; no detectable target effect beyond self-replay noise. Builder path remains first causal divergence; semantic correctness open.

- `2026-09-23T13:18:58Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T131858Z`（test）。

- `2026-09-23T13:34:34Z` Run `run-20260923T131858Z` 记录为 `invalid`；正确性为 `invalid`。Run35 native target metadata parity did not execute: 8-rank bootstrap failed because attn_groups outer element is a list; fixed nested traversal for Run36. No semantic inference.

- `2026-09-23T13:34:44Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T133444Z`（test）。

- `2026-09-23T13:47:25Z` Run `run-20260923T133444Z` 记录为 `pass`；正确性为 `invalid`。Run36 native-vs-reference two-cycle tensor comparison executed on 8 ranks: 45/55 fields exact cycle0, 42/55 cycle1, host mirrors exact. SAS/QLI 1024-element outputs differ widely, plus three slot buffers at cycle1. Reference self-repeat control required before assigning semantic fault; no acceptance claim.

- `2026-09-23T13:50:10Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T135010Z`（test）。

- `2026-09-23T14:04:34Z` Run `run-20260923T135010Z` 记录为 `pass`；正确性为 `invalid`。8/8 rank two-cycle reference A/native B/reference C field control completed. SAS and QLI self-replay tails are unstable; native SAS header index4 and QLI header index12 differ before self-noise. Three compressed state-cache slot mappings differ only in native cycle1. Native semantic parity remains invalid.

- `2026-09-23T14:06:20Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T140620Z`（test）。

- `2026-09-23T14:19:38Z` Run `run-20260923T140620Z` 记录为 `pass`；正确性为 `invalid`。8/8 ranks: SAS reference/native operator inputs differ only in cu_seqlens_q dtype int32 vs int64 for ratios 1,4,128. Native SAS header and QLI header differ; SWA-only binding removes all three compressed slot mismatches. Fix cumsum dtype and revalidate.

- `2026-09-23T14:20:18Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T142018Z`（test）。

- `2026-09-23T14:34:35Z` Run `run-20260923T142018Z` 记录为 `pass`；正确性为 `invalid`。8/8 ranks two cycles: reference/native SAS inputs and output first32 match for c1/c4/c128; all 45 non-SAS/QLI fields exact. SAS first mismatch >=97, QLI >=25 in self-unstable tails. Full target/acceptance parity still open; next same-state A/B/C.

- `2026-09-23T14:36:58Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T143658Z`（test）。

- `2026-09-23T14:50:17Z` Run `run-20260923T143658Z` 记录为 `pass`；正确性为 `pass`。8/8 ranks DSA-only same-state target A=reference B=native C=restored reference: AB argmax 92/96,94/96 vs AC 93/96,92/96; AB acceptance counts 12/12 both cycles; 138 physical KV entries restored exactly, old metadata restored, host mirrors/state exact. Isolated two-cycle parity passes within replay noise; long semantics/GDN still open.

- `2026-09-23T14:50:45Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T145045Z`（profile）。

- `2026-09-23T15:06:24Z` Run `run-20260923T145045Z` 记录为 `pass`；正确性为 `pass`。Native DSA completed 256 c12 eager cycles on 8/8 ranks with exact state and host mirrors. Target graph env was omitted, so target mode NONE; 412.5ms target,41.1ms proposer and 45.2 tok/s diagnostic are incomparable to Run18 FULL graph/formal E2E. Run42 corrects target graph and removes extra diagnostic clones.

- `2026-09-23T15:06:44Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T150644Z`（profile）。

- `2026-09-23T15:19:43Z` Run `run-20260923T150644Z` 记录为 `pass`；正确性为 `pass`。8/8 ranks 256 native-DSA FULL-graph c12 cycles, exact state/host mirrors. NPU median derived metadata8.616ms,target46.201ms,proposer5.880ms. Late cycles192-255 acceptance1.594 outputs/slot/cycle, so sustained progress remains low. Short diagnostic309tok/s is not formal E2E or paired throughput. Next GDN dynamic prior-count parity and metadata synchronization cost.

- `2026-09-23T15:21:54Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T152154Z`（test）。

- `2026-09-23T15:35:31Z` Run `run-20260923T152154Z` 记录为 `invalid`；正确性为 `invalid`。All eight ranks failed bootstrap before target A/B/C: initial attention metadata exposes no speculative GDN count consumer. Bind builder-owned stable int32 num_accepted_tokens buffer once into Runtime and rerun; no semantic or acceptance conclusion.

- `2026-09-23T15:36:54Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T153654Z`（test）。

- `2026-09-23T15:50:52Z` Run `run-20260923T153654Z` 记录为 `invalid`；正确性为 `invalid`。8/8 ranks failed bootstrap with no GDN builder count source. DeepSeek V4 Flash config/source and prior metadata records show no active GDN group; generic callback GDN branch was not executed. Retract GDN hypothesis, remove optional binding, continue actual DSA/DSpark/KV diagnostics.

- `2026-09-23T15:52:45Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T155245Z`（test）。

- `2026-09-23T16:05:29Z` Run `run-20260923T155245Z` 记录为 `pass`；正确性为 `pass`。8/8 rank late native DSA A/B/C at cycles 0,1,64,128 passed state and KV restore; native difference remains within noisy reference self-replay; no long semantic or acceptance claim

- `2026-09-23T16:08:28Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T160828Z`（test）。

- `2026-09-23T16:21:50Z` Run `run-20260923T160828Z` 记录为 `pass`；正确性为 `pass`。8/8 rank cycle128 Product/Stock proposer first drafts 12/12; full draft83/84 vs Stock self79/84; inputs and KV restores exact; trajectory root remains open

- `2026-09-23T16:25:16Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T162516Z`（test）。

- `2026-09-23T16:40:35Z` Run `run-20260923T162516Z` 记录为 `invalid`；正确性为 `invalid`。Independent exact-text oracle gate failed: 0/12 Stock outputs matched Stock self-replay, earliest text divergence 38 chars; use same-state Stock/direct target control

- `2026-09-23T16:42:32Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T164232Z`（test）。

- `2026-09-23T16:56:16Z` Run `run-20260923T164232Z` 记录为 `invalid`；正确性为 `invalid`。Diagnostic group common metadata view missing on 8 ranks before target A/B/C; capture gate corrected for Run49

- `2026-09-23T16:56:29Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T165629Z`（test）。

- `2026-09-23T17:10:20Z` Run `run-20260923T165629Z` 记录为 `pass`；正确性为 `pass`。8/8 rank Stock cycle128 same-KV A/direct B/Stock C: B equals A 96/96 argmax, 12/12 first target, accepted tokens/counts exact; Stock self95/96; KV/metadata restore exact

- `2026-09-23T17:11:48Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T171148Z`（test）。

- `2026-09-23T17:25:41Z` Run `run-20260923T171148Z` 记录为 `pass`；正确性为 `pass`。8/8 rank target writes: A/B and Stock A/C both 50/69 cache rows exact, same 19 mismatch names; raw cache self-replay noise prevents candidate-specific conclusion

- `2026-09-23T17:27:07Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T172707Z`（test）。

- `2026-09-23T17:42:27Z` Run `run-20260923T172707Z` 记录为 `pass`；正确性为 `pass`。8/8 rank cycles0,1,64,128: all 45 non-sparse DSA fields and 3 SAS op inputs/output first32 match; sparse tails self-noisy; low acceptance unresolved

- `2026-09-23T17:44:24Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T174424Z`（test）。

- `2026-09-23T17:59:55Z` Run `run-20260923T174424Z` 记录为 `fail`；正确性为 `fail`。Standalone no-reservation block-table failure: 8/8 ranks physical block0 in 66/96 target positions at cycle16 and 96/96 from cycle32; formal serving has separate 1024-token reservation and remains unproven

- `2026-09-23T18:00:26Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T180026Z`（test）。

- `2026-09-23T18:17:55Z` Run `run-20260923T180026Z` 记录为 `fail`；正确性为 `fail`。12/12 clients returned 1024, but completed slots kept decoding; cycle256 slot2 targeted physical block0 and rank0 staged 2564 tokens for that slot. Reserved serving semantic failure. Single audited cohort TPS is diagnostic only.

- `2026-09-23T18:19:35Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T184500Z`（test）。

- `2026-09-23T18:33:40Z` Run `run-20260923T184500Z` 记录为 `pass`；正确性为 `pass`。Run54 fixed-slot functional gate: 12/12 exact outputs, 8/8 rank and Host mirrors, overshoot 2149->49; sampled block0 absent through cycle256. Late physical KV audit and long token oracle remain open; 360.33 TPS is diagnostic only.

- `2026-09-23T18:34:30Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T184000Z`（test）。

- `2026-09-23T18:48:31Z` Run `run-20260923T184000Z` 记录为 `pass`；正确性为 `pass`。Run55 sampled late KV safety passes: 8 ranks, cycles 0/128/256/320/360/384/400/420, no physical block0 or negative target mappings, parked positions stable, host mirrors exact; 12/12 1024-token clients. Long token-level oracle and acceptance gap remain open.

- `2026-09-23T18:49:34Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T185000Z`（test）。

- `2026-09-23T19:03:49Z` Run `run-20260923T185000Z` 记录为 `invalid`；正确性为 `invalid`。Missing EXTREME_DSA_BUILDER_ORACLE=1; all 8 ranks fail explicit reference metadata callback gate before target A/B/C; no target or acceptance conclusion.

- `2026-09-23T19:03:58Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T190500Z`（test）。

- `2026-09-23T19:17:46Z` Run `run-20260923T190500Z` 记录为 `pass`；正确性为 `pass`。Reserved 257-cycle A/reference B/native C/reference target control passes 8/8 rank restores and host gates. At cycles64/128/256 native target and accepted counts stay within self-replay noise; no sustained native DSA target split. Client sentinel TPS invalid; continuous Stock-vs-Extreme token oracle remains open.

- `2026-09-23T19:24:04Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T192500Z`（test）。

- `2026-09-23T19:37:45Z` Run `run-20260923T192500Z` 记录为 `pass`；正确性为 `pass`。Strengthened partial KV snapshot coverage gate: zero skipped/partial snapshots on 8 ranks at cycles0/1/64/128/256; all restores exact. Native DSA target remains within self-replay noise. Does not establish full mutable-state coverage or long Stock token equivalence.

- `2026-09-23T19:40:45Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T194500Z`（test）。

- `2026-09-23T19:56:03Z` Run `run-20260923T194500Z` 记录为 `invalid`；正确性为 `invalid`。Run59 observer assumed every KV group uses absolute position/block_size; group1 table bounds disproved that assumption at first cycle. EngineCore exited; client 11/12 partial, no model semantic conclusion.

- `2026-09-23T19:56:03Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T195600Z`（test）。

- `2026-09-23T20:11:03Z` Run `run-20260923T195600Z` 记录为 `invalid`；正确性为 `invalid`。Run60 Stock client 12/12 exact 1024, but shadow never reached write-at-256 and produced no rank files; no continuous-equivalence conclusion. Added periodic checkpoints and group spec types for Run61.

- `2026-09-23T20:11:04Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T201100Z`（test）。

- `2026-09-23T20:26:09Z` Run `run-20260923T201100Z` 记录为 `pass`；正确性为 `pass`。8/8 rank 128 consecutive fixed-c12 Stock target ABI exact; 5/6 KV group slot mappings exact with no negative physical blocks. Group1 256-column state/window table lacks applicable formula. Client 12/12 exact 1024. Long Product-vs-Stock token oracle open.

- `2026-09-23T20:30:39Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T203000Z`（test）。

- `2026-09-23T20:44:15Z` Run `run-20260923T203000Z` 记录为 `pass`；正确性为 `pass`。8/8 ranks recorded 32 fixed-c12 Stock cycles with exact target ABI and complete target-input/argmax/acceptance/next-draft trace. Client 12/12 exact 1024; group1 mapping geometry remains unverified.

- `2026-09-23T20:44:16Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T204300Z`（test）。

- `2026-09-23T20:59:46Z` Run `run-20260923T204300Z` 记录为 `pass`；正确性为 `pass`。Extreme 8/8 rank serving gates and 12/12 clients exact1024; first32 Product trace captured. Paired Stock Run62 initial full batch differs in 9/12 slots; 3 aligned slots show target/draft clues, not causal proof. Same-state ABA next.

- `2026-09-23T21:03:00Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260923T210000Z`（test）。

- `2026-09-23T21:16:44Z` Run `run-20260923T210000Z` 记录为 `pass`；正确性为 `pass`。8/8 rank cycle1 same-state A/B/C: Extreme B vs Stock A accepted96/96 and counts12/12 equal, while Stock self C accepted91/96; target argmax AB89/96 vs AC88/96; touched KV write equal rows50/69 for both, same mismatch names; all restores exact. No early candidate-specific target/DSA split.

- `2026-09-24T02:02:21Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260924T020000Z`（test）。

- `2026-09-24T02:16:41Z` Run `run-20260924T020000Z` 记录为 `pass`；正确性为 `not-applicable`。Run65 read-only reserved-serving KV slot audit: 8/8 ranks and 12/12 clients complete 1024 tokens; first all-rank mapping mismatch at cycle1 in groups0,2,4,5 (96/96 each), group3 exact, group1 unsupported geometry; consumption/semantics still unproven

- `2026-09-24T02:21:18Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260924T022000Z`（test）。

- `2026-09-24T02:34:22Z` Run `run-20260924T022000Z` 记录为 `pass`；正确性为 `not-applicable`。Run66 8/8 same-state reserved cycle0/1 A/B/C passed KV restore and Host gates. At cycle1 all-group slots changed 96/96 except group3; A/B argmax92/96 vs A/C91/96, accepted92/96 for both, so no candidate-specific direct target split. Large metadata snapshot exclusions remain; client TPS invalid due intentional sentinel.

- `2026-09-24T02:55:05Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260924T030000Z`（test）。

- `2026-09-24T03:10:15Z` Run `run-20260924T030000Z` 记录为 `pass`；正确性为 `not-applicable`。Run67 opt-in DSpark gid2/3 context slot refresh passed 8/8 ranks and two 12/12 length-exact cohorts; gid2 refreshed 96/96 slots from cycle1, gid3 unchanged; cycles300/325, diagnostic TPS467.9/484.5. Same-code flag-off and long token oracle still required.

- `2026-09-24T03:11:23Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260924T031500Z`（test）。

- `2026-09-24T03:25:35Z` Run `run-20260924T031500Z` 记录为 `pass`；正确性为 `not-applicable`。Same-code two-cohort DSpark slot-refresh flag-off control: 12/12x1024, 8/8 rank and Host gates; 458/423 cycles vs Run67 flag-on 300/325. Continuous acceptance benefit supported; long token oracle pending.

- `2026-09-24T03:29:02Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260924T033000Z`（test）。

- `2026-09-24T03:44:56Z` Run `run-20260924T033000Z` 记录为 `pass`；正确性为 `not-applicable`。Read-only DSpark context-slot provenance: cycle1-7 gid2 scatter input 96/96 stale on all 8 ranks, exact copy of source mapping; gid3 exact. 12/12x1024 and 8/8 rank gates pass. Long semantic oracle pending.

- `2026-09-24T03:45:36Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260924T035000Z`（test）。

- `2026-09-24T03:59:50Z` Run `run-20260924T035000Z` 记录为 `pass`；正确性为 `not-applicable`。Refreshed DSpark context slots exact 96/96 for gid2 and gid3 at cycles0-7 on all 8 ranks; 12/12x1024 and 8/8 rank gates pass; 295 cycles vs paired flag-off Run69 458. Long token oracle pending.

- `2026-09-24T04:04:07Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260924T040500Z`（test）。

- `2026-09-24T04:19:09Z` Run `run-20260924T040500Z` 记录为 `pass`；正确性为 `not-applicable`。294-cycle rank0 product trajectory passed 3528 slot-cycle arithmetic/greedy checks and 12x1024 token coverage; all 8 rank/Host gates pass. Stock oracle and KV write-value parity pending.

- `2026-09-24T04:20:28Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260924T042000Z`（test）。

- `2026-09-24T04:36:32Z` Run `run-20260924T042000Z` 记录为 `pass`；正确性为 `not-applicable`。285-cycle all-rank DSpark context write-address trajectory: gid2/3 exact every cycle, no invalid/duplicate/cross-request ownership; 4560 phase records complete, 12x1024 client gates pass. Target consumers and Stock token/KV value parity pending.
