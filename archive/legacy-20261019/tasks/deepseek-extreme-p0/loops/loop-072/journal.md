# 执行日志

- `2026-09-26T15:58:09Z` Loop 已冻结。下一步：Run319 source-only gate for layer4 MoE row independence, selected EP dispatcher/Graph shape and all8 zero-row hazards; then implement reversible private A/A/B/A fixture if viable

- `2026-09-26T15:59:11Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run319`（design-check）。

- `2026-09-26T16:00:25Z` Run `run319` 记录为 `pass`；正确性为 `not-applicable`。Layer4 full MoE is candidate boundary; compact B requires correctly bound N_active*8 forward context and separate Graph/HCCL sequence; dynamic EPLB and numerical equivalence must be gated

- `2026-09-26T16:09:00Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run320`（test）。

- `2026-09-26T16:37:52Z` Run `run320` 记录为 `invalid`；正确性为 `invalid`。All-rank preflight gated: actual layer4 MoE input (12,4096), not assumed (96,4096); 12x1024 carrier passed, no B execution. Source restored and cards idle.

- `2026-09-26T16:37:52Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run321`（test）。

- `2026-09-26T16:40:19Z` Run `run321` 记录为 `invalid`；正确性为 `invalid`。Pre-execution Astra High source review: 12 local rows are TP8 contiguous partitions of 96 global tokens; direct 12-to-6 mask selection is semantically invalid. No service run.

- `2026-09-26T16:40:19Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run322`（test）。

- `2026-09-26T17:04:51Z` Run `run322` 记录为 `pass`；正确性为 `not-applicable`。8-rank read-only parked MoE census passed: outer96/local12, FlashComm1, ALLGATHER, active local rows 8/0/4/12/4/12/8/0; no candidate output or E2E claim.

- `2026-09-26T17:04:51Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run323`（simulation）。

- `2026-09-26T17:05:03Z` Run `run323` 记录为 `pass`；正确性为 `not-applicable`。Read-only Run287 mask remap: 579/678 parked cycles retain max local12, only 99/1496 cycles below12; 2.875% is max-local row-linear screening proxy, not E2E bound.

- `2026-09-26T17:13:14Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run324`（design-check）。

- `2026-09-26T17:13:14Z` Run `run324` 记录为 `pass`；正确性为 `not-applicable`。Astra High and Sol source gate passed: existing ALLGATHER prepares global96, routed apply token axes x/router_logits/per-token scale can select active48, restore routed96 before original finalize; all-rank trigger and SHA restore syntax pass. No service or candidate parity.

- `2026-09-26T17:13:14Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run325`（test）。

- `2026-09-26T17:33:22Z` Run `run325` 记录为 `invalid`；正确性为 `invalid`。8-rank private routed96-to48-to96 B executed with finite active output; original A/A_repeat and A/A2 non-bit-exact all ranks (0.0039-0.0078 max), so checker exit1 and B semantic gate inconclusive. 12x1024 carrier passed; source restored/cards idle.

- `2026-09-26T17:41:52Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run326`（test）。

- `2026-09-26T18:02:09Z` Run `run326` 记录为 `pass`；正确性为 `invalid`。All8 private full-MoE Graph screen completed; B post-gather routed48 is 10/10 slower than independent A/A2 paired controls (median +49.585us max-rank), only 1/7 nonempty ranks within predeclared numerical control envelope; REJECT this implementation, pivot gate placement; no bound or formal E2E update

- `2026-09-26T18:05:15Z` 主控结论为 `PIVOTED`。Run326 all8 same-prestate complete MoE Graph A/A_repeat/B/A2 showed routed48 B slower than independent A controls in 10/10 paired max-rank endpoints, median +49.585us; numerical control envelope failed on 6/7 nonempty ranks. Reject this implementation, not all parked schedules or a hardware bound.
