# 执行日志

- `2026-09-26T18:44:08Z` Loop 已冻结。下一步：Validate Run287 original all8 emitted/mask and client/runtime cohort mapping, compute no-cost FIFO and incremental refill-cost sensitivity, record historical R06/R36/R37 priors with changed TP8 conditions

- `2026-09-26T18:44:25Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run329`（simulation）。

- `2026-09-26T18:44:25Z` Run `run329` 记录为 `pass`；正确性为 `not-applicable`。Run287 warmup48 four original cohorts total1203 cycles; all8 masks identical; 2266/14436 parked slot-cycles; fixed observed-duration work floor1015 cycles, free-refill FIFO1122 cycles with current one-cycle Host mirror and zero incremental prefill/seed/state cost. Conditional scheduling screen only, no Product TPS/bound.

- `2026-09-26T18:46:06Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run330`（simulation）。

- `2026-09-26T18:46:34Z` Run `run330` 记录为 `pass`；正确性为 `not-applicable`。Read-only Run287 warmup48 all8 emitted/mask and FULL Graph Runtime/client contract closed. Four waves1203 cycles; mirror FIFO zero incremental refill cost1122 (81 conditional cycles), immediate FIFO1118; fixed-duration floor1015/1011. Within-wave order 512-sample zero-cost 1081-1188; per-slot incremental delay24 gives1194, delay32 gives1218. No actual Product/HW bound or TPS claim.

- `2026-09-26T18:54:56Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run331`（simulation）。

- `2026-09-26T18:55:04Z` Run `run331` 记录为 `pass`；正确性为 `not-applicable`。Read-only all8 historical calibration: b=2/3/4 zero-cost fixed-duration schedules 1138/1141/1149 vs observed1203 cycles; Run188 warmed residual prefill 7/8 calls and 1328/1352 scheduled tokens, Run239 four original first-execute-to-handoff 2.356-4.116s. No marginal refill cost or attainable TPS measured.

- `2026-09-26T18:58:23Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run332`（profile）。

- `2026-09-26T19:19:15Z` Run `run332` 记录为 `pass`；正确性为 `not-applicable`。Original legal warmed path: 48/48 warmup and measured clients, 64/64 Runtime reports, four measured FULL Graph cohorts, all8 scheduler parity; 48/48 enter with 32768 cached prefix tokens, residual prompt tokens 83x46/84x1/85x1, client ID-to-slot mapping unique. Host runtime-built envelopes 2.102-2.433s; device seed-ready and incremental refill cost unmeasured.

- `2026-09-26T19:22:10Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run333`（profile）。

- `2026-09-26T19:50:00Z` Run `run333` 记录为 `pass`；正确性为 `not-applicable`。All8 first warmed padded88 prefill profiler exported; CANN Computing38.30-39.01ms/rank, Stage460.78-495.56ms under profiler/sync, 12/12 client and8/8 FULL Graph Runtime pass; no Product inference.

- `2026-09-26T19:50:00Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run334`（review）。

- `2026-09-26T19:50:00Z` Run `run334` 记录为 `pass`；正确性为 `not-applicable`。Aligned264 HCCL tasks all8 by order/type: latest starter rank4 214/264, start spread median1436us, end spread20.4us; profiled rendezvous wait, not link bound.

- `2026-09-26T19:50:00Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run335`（review）。

- `2026-09-26T19:50:00Z` Run `run335` 记录为 `pass`；正确性为 `not-applicable`。All264x8 Host Enqueue/correlation Dequeue/CANN tasks matched; latest Host enqueuer equals latest device starter264/264, enqueue spread median25.774ms; profiler-perturbed Host submission progress dominates apparent HCCL wait.

- `2026-09-26T19:54:06Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run336`（review）。

- `2026-09-26T19:54:06Z` Run `run336` 记录为 `pass`；正确性为 `not-applicable`。Run188 unprofiled all8 warmed prefill Host forward wall: rank4 longest0/15 calls, rank1 longest13/15, rank7 longest2/15; cannot infer absolute collective arrival.

- `2026-09-26T19:54:06Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run337`（profile）。

- `2026-09-26T20:16:47Z` Run `run337` 记录为 `pass`；正确性为 `not-applicable`。Unprofiled all8 padded88 first warm prefill: 221 captured Host collectives/rank parity, 43 AG outside wrapper, Host entry spread3.68->43.65ms, CPU/wall~1, rank5/6 late; 12/12 clients and8/8 FULL Graph pass, source restored; no Product bound.

- `2026-09-26T20:22:23Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run338`（test）。

- `2026-09-26T20:42:24Z` Run `run338` 记录为 `invalid`；正确性为 `not-applicable`。48/48 warmup,12/12 measured and8/8 FULL Graph Runtime pass, source restored; no rank fixture report, so inner MoE Python hook missed active prefill path and no A/Graph/A2 comparison exists.

- `2026-09-26T20:45:01Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run339`（benchmark）。

- `2026-09-26T21:00:31Z` Run `run339` 记录为 `pass`；正确性为 `not-applicable`。CANN9.1 TP8 exact-payload HCCL Test 5 cases x 3 independent runs all exit0/result-check success; isolated medians 4.95-32.10us, not Product or strict lower bound; Current formal unchanged

- `2026-09-26T21:05:44Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run340`（design-check）。

- `2026-09-26T21:06:29Z` Run `run340` 记录为 `pass`；正确性为 `not-applicable`。Read-only source/evidence audit pivots from repeated first88 one-layer Graph to b2-b4 refill preparation; serial replacement budget screening b4 about1.11s/episode, no bound or product gain

- `2026-09-26T21:08:27Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run341`（profile）。

- `2026-09-26T21:30:38Z` Run `run341` 记录为 `pass`；正确性为 `not-applicable`。Original 48+48 all8 FULL Graph/exact1024,35 allrank forward calls;26 natural prompt-bearing1-3 request calls Host max353-384ms/current-stream354-411ms, no incremental refill or seed-ready; source restored, exit0

- `2026-09-26T21:36:06Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run342`（design-check）。

- `2026-09-26T21:43:57Z` Run `run342` 记录为 `pass`；正确性为 `not-applicable`。Source audit confirmed Current bulk-return prevents real b3 arrival; corrected Dual-Bound model and selected segmented-publication continuation gate. No service or E2E claim.

- `2026-09-26T21:47:31Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run343`（design-check）。

- `2026-09-26T23:20:56Z` Run `run343` 记录为 `pass`；正确性为 `pass`。Fence-only original bulk path passed 48 warmup + three formal 48-request repeats; all 128 rank-cohort FULL Graph gates pass; 16/16 fence bulk pairs; median 575.709 TPS versus prior-session Current 571.681, no causal gain claim; source restored and service stopped.

- `2026-09-26T23:23:53Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run344`（design-check）。

- `2026-09-26T23:30:00Z` Run `run344` 记录为 `pass`；正确性为 `not-applicable`。Source audit and no-NPU ghost publication protocol ledger pass five groups; exact final Scheduler settlement and real arrival remain unimplemented live gates.

- `2026-09-26T23:31:27Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run345`（build）。

- `2026-09-26T23:40:17Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run346`（design-check）。

- `2026-09-26T23:42:15Z` Run `run346` 记录为 `pass`；正确性为 `not-applicable`。Pinned read-only Dual-Bound V3: Run287 1011/1015 capacity relaxation, 1118/1122 constructed FIFO, Run99 512 current-runtime cardinality; no finite Product TPS upper bound; b2-4 break-even sensitivity and prioritized correlated ledger

## Loop074 Run346 Bound V3 priority correction

Run346 makes the main uncertainty explicit: Current Formal571.681tok/s is established, but no defensible finite overall TPS ceiling exists yet. Run287 fixed-duration1011/1015 cycles are capacity relaxations; zero-cost FIFO1118/1122 are constructed conditional schedules; Run99's512 is a cardinality floor for the current four zero-output handoff cohorts. None includes the complete legal product DAG. The cross-run b2/b3/b4 break-even screen requires hiding about46–56% of Run332's6.935s later-wave preparation if that same work is needed. This is useful for ranking measurements, not a refill performance forecast. Prioritize a correlated formal48 request work/dependency ledger joining real arrival, prefix/residual prefill, seed-ready, useful acceptance, Target/DSpark resource work, collective joins, state/KV and final publication. Run345 segmented publication remains a narrow legal-arrival gate, pending this Bound audit; its partial code has no correctness or E2E result. See `evidence/20260926_loop074_refill/run346/findings.md` and `bound_calibration_v3.json`.

- `2026-09-26T23:44:37Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run347`（design-check）。

- `2026-09-26T23:46:06Z` Run `run347` 记录为 `pass`；正确性为 `not-applicable`。Saved Run239 all8 exact count reanalysis: 1195 cycles, 49152 useful tokens, fixed-duration 1046/1045 capacity relaxations, one post-completion cycle per cohort; no Product bound promotion

- `2026-09-26T23:47:52Z` Run `run345` 记录为 `invalid`；正确性为 `not-applicable`。Design preflight only: draft segmented serving source was not installed, no service or NPU run occurred. Paused after priority correction to Bound-first Run346/347; no live publication, arrival, correctness or TPS evidence.
