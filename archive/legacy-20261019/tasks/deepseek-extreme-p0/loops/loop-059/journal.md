# 执行日志

- `2026-09-25T16:20:51Z` Loop 已冻结。下一步：Reuse and verify Loop045 reversible boundary patch; run one legal warmed 48-request diagnostic with four cohort boundary captures, then audit all-rank phase wall against client timestamps

- `2026-09-25T16:21:14Z` 为用例 `mixed_32k_1024_c12` 创建 Run `legal-formal-boundary-pass`（profile）。

- `2026-09-25T16:39:35Z` Run `legal-formal-boundary-pass` 记录为 `pass`；正确性为 `pass`。Exit0; warmup and measured diagnostic each 48/48 exact1024; measured 592.618 TPS with instrumentation; four cohort prefill-to-handoff 2.356-4.115s, serve 16.241-17.504s, client-to-execute and publication-to-client-end each about0.2s; all8 ranks pass, borrowed source restored exact SHA, service stopped

- `2026-09-25T16:40:53Z` 为用例 `mixed_32k_1024_c12` 创建 Run `collective-unit-correction`（review）。

- `2026-09-25T16:42:07Z` Run `collective-unit-correction` 记录为 `pass`；正确性为 `not-applicable`。Run152 0.83810ms is total of 33-41 adjacent graph pair tasks/window, median task pair0.02288ms; Run237 eager latest-rank pair0.53549ms is 23.4x slower and path/payload semantics differ. Comparison invalid for product capacity or E2E saving; prior docs corrected.

- `2026-09-25T16:43:03Z` 为用例 `mixed_32k_1024_c12` 创建 Run `phase-replay-calibration`（simulation）。

- `2026-09-25T16:43:57Z` Run `phase-replay-calibration` 记录为 `pass`；正确性为 `not-applicable`。Run239 phase replay: 13.536s prefill, 67.813s runtime serve, 1.600s other boundaries in 82.940s diagnostic; 1195 cycles, 56.782ms measured decode wall/cycle. 0.5s/cohort hypothetical prefill saving projects 607.261 TPS fixed cycles but only 593.928 TPS with +8 cycles/cohort; Run188 observed +32 cycles largely erased gross prefill reduction.

- `2026-09-25T16:44:21Z` 暂存知识变化 `loop059-phase-coupling`：Legal warmed 48-request diagnostic attributes 13.536s to prefill-to-handoff and 67.813s to runtime serve over 82.940s; Run188 admission hold shows gross prefill reduction can be canceled by more decode cycles and overlap, so fixed-cycle scenario projections are conditional.

- `2026-09-25T16:44:21Z` 主控结论为 `PIVOTED`。Legal all-rank boundary pass localizes Run99 residual to variable prefill and decode work rather than small publication/admission edges; Run188 shows phase reductions couple to decode cycles, and no true hardware floor or E2E candidate has been established
