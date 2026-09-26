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
