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
