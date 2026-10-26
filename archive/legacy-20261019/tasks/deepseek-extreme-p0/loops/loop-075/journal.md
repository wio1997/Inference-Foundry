# 执行日志

- `2026-09-26T23:49:55Z` Loop 已冻结。下一步：Build and run a private 8-card concurrent real-shape GMM service calibration from Run148/150 assets

- `2026-09-26T23:51:10Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run348`（benchmark）。

- `2026-09-26T23:52:42Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run349`（benchmark）。

- `2026-09-26T23:53:57Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run350`（benchmark）。

- `2026-09-26T23:53:57Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run351`（benchmark）。

- `2026-09-26T23:55:55Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run352`（design-check）。

- `2026-09-26T23:57:30Z` Run `run348` 记录为 `pass`；正确性为 `not-applicable`。Direct event-per-operator eight-card pilot exits0 and overlaps all ranks, but Host enqueue is inside event interval; not hardware service calibration

- `2026-09-26T23:57:30Z` Run `run349` 记录为 `pass`；正确性为 `not-applicable`。Eight-card concurrent 16-op GMM Graph event calibration A, all8 windows overlap and real-rank shapes captured

- `2026-09-26T23:57:30Z` Run `run350` 记录为 `pass`；正确性为 `not-applicable`。Same-rank one-card 16-op Graph control B for eight-card capacity screen

- `2026-09-26T23:57:30Z` Run `run351` 记录为 `pass`；正确性为 `not-applicable`。Eight-card 16-op Graph A2 repeat with all-rank overlap and route/shape parity

- `2026-09-26T23:57:30Z` Run `run352` 记录为 `pass`；正确性为 `not-applicable`。A/B/A analysis: rank0 single versus all8 GMM1 +1.52 percent and GMM2 -1.07 percent inside A drift; isolated attained service only, no Product bound

- `2026-09-27T00:03:41Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run353`（benchmark）。

- `2026-09-27T00:04:21Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run354`（benchmark）。

- `2026-09-27T00:04:21Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run355`（benchmark）。

- `2026-09-27T00:04:21Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run356`（benchmark）。

- `2026-09-27T00:07:34Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run357`（design-check）。

- `2026-09-27T00:09:52Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run358`（profile）。

- `2026-09-27T00:10:41Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run359`（profile）。

- `2026-09-27T00:12:17Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run360`（design-check）。

- `2026-09-27T00:12:57Z` Run `run353` 记录为 `pass`；正确性为 `not-applicable`。Single-rank bank8 graph setup pilot; synthetic zero data, no Product conclusion

- `2026-09-27T00:12:57Z` Run `run354` 记录为 `pass`；正确性为 `not-applicable`。Same-layer all8 repeated one-bank Graph A, isolated synthetic timing

- `2026-09-27T00:12:57Z` Run `run355` 记录为 `pass`；正确性为 `not-applicable`。Same-layer all8 rotating eight-bank Graph B, isolated synthetic timing

- `2026-09-27T00:12:57Z` Run `run356` 记录为 `pass`；正确性为 `not-applicable`。Same-layer all8 repeated one-bank Graph A2 control

- `2026-09-27T00:12:58Z` Run `run357` 记录为 `pass`；正确性为 `not-applicable`。GMM2 bank8 +8.04 percent median versus A/A2 midpoint, all8 ranks strict; GMM1 +0.21 percent, 4/8 strict; not an E2E bound

- `2026-09-27T00:12:58Z` Run `run358` 记录为 `pass`；正确性为 `not-applicable`。Onecard rank4 bank1 Level1 MemoryAccess trace; 41 native GMM2 rows including two complete 16-task replay groups

- `2026-09-27T00:12:58Z` Run `run359` 记录为 `pass`；正确性为 `not-applicable`。Onecard rank4 bank8 Level1 MemoryAccess trace; 42 native GMM2 rows including two complete 16-task replay groups

- `2026-09-27T00:12:58Z` Run `run360` 记录为 `pass`；正确性为 `not-applicable`。Two full native GMM2 Graph replays per condition; AIC read counter unchanged, median profiled time +8.97 percent; no HBM mechanism or Product bound inferred

- `2026-09-27T00:13:38Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run361`（test）。

- `2026-09-27T00:15:05Z` Run `run361` 记录为 `invalid`；正确性为 `invalid`。Nonzero bank values yielded identical all-zero outputs because raw int64 scale ones are not encoded unit quant scale; bank-binding diagnostic inconclusive

- `2026-09-27T00:15:06Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run362`（test）。

- `2026-09-27T00:16:15Z` Run `run362` 记录为 `pass`；正确性为 `pass`。Encoded unit quant scale gives eager bank0/1 2048/4096; two-bank Graph outputs match each eager bank exactly and differ

- `2026-09-27T00:16:17Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run364`（test）。

- `2026-09-27T00:17:13Z` Run `run364` 记录为 `pass`；正确性为 `invalid`。Eight distinct packed bank first active outputs match eager per-bank and Graph alternates; full tensor equality false, likely padded inactive rows, requires active-prefix gate

- `2026-09-27T00:17:14Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run365`（test）。

- `2026-09-27T00:17:26Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run363`（design-check）。

- `2026-09-27T00:17:59Z` Run `run365` 记录为 `pass`；正确性为 `pass`。Eight distinct packed banks produce eight distinct active outputs; all16 captured Graph outputs exactly match eager on all120 active rows, zero max absolute difference

- `2026-09-27T00:18:01Z` Run `run363` 记录为 `pass`；正确性为 `not-applicable`。V3.3 dual-bound artifact integrates same-layer bank sensitivity, verified native replay/active bank binding, and Level1 counters without promoting finite Product TPS ceiling

- 2026-09-27 Run352 result summary corrected: per-rank source ordinals differ, and Host overlap does not prove device replay overlap. Original TaskCtl event remains in task-events history; corrected result and evidence are authoritative.

- `2026-09-27T00:20:06Z` 暂存知识变化 `PK-031`：On same ordinal64 isolated synthetic W4A8 GMM2 Graph, eight independent weight banks slow all8 ranks against one-bank A/A2 while Level1 AIC main-memory and GM-to-L1 byte counters stay equal; mechanism and FULL Graph transfer remain unknown

- `2026-09-27T00:20:06Z` 主控结论为 `PIVOTED`。Conditional GMM2 bank sensitivity and native replay/counter/binding gates calibrated; compulsory work and same-path FULL Graph cache/resource join still missing, so no numerical Product ceiling
