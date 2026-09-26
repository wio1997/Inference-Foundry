# 执行日志

- `2026-09-26T18:05:35Z` Loop 已冻结。下一步：Read exact current gate/prepare/source dependencies and design SHA-guarded all-active one-layer private A/A/B/A with both gate-ready and full endpoint; search pinned R11/R35 priors first

- `2026-09-26T18:12:14Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run327`（design-check）。

- `2026-09-26T18:12:35Z` Run `run327` 记录为 `pass`；正确性为 `not-applicable`。Pinned R11/R35 prior searched (PK-016); current source proves replicated FP32 gate and existing hidden/logits gathers; private Loop073 fixture syntax and reversible source SHA preview passed; Astra review found/fixed AllGatherCommImpl versus prepare_finalize object boundary

- `2026-09-26T18:16:04Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run328`（test）。

- `2026-09-26T18:37:05Z` Run `run328` 记录为 `pass`；正确性为 `invalid`。All8 private full-active layer4 MoE Graph A/A_repeat/B/A2 completed; gate weights and routing IDs exact, B logits max1.38e-5; output control envelope7/8 ranks; B strictly faster0/10 paired complete endpoints, max-rank device median B0.55143ms versus A0.54025/A2 0.53960ms. No live promotion, no bound update.

- `2026-09-26T18:43:26Z` 主控结论为 `PIVOTED`。Run328 all8 same-state full-active layer4 Graph proves replicated gate weights and exact routing IDs, but B gate placement shows 0/10 strict complete-MoE wins and variable paired B-minus-A time; one rank narrowly misses output control envelope. Do not integrate this B schedule; exchange law and alternate overlap remain open.
