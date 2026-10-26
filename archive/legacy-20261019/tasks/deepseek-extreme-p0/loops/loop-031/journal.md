# 执行日志

- `2026-09-22T08:23:32Z` Loop 已冻结。下一步：Add low-overhead stage markers around the Extreme-owned chain and capture an eight-rank runtime-only trace; attribute target, TP/EP communication, acceptance/state advance, DSpark common refresh and proposer regions before selecting an implementation change.

- `2026-09-22T08:56:40Z` 为用例 `mixed_32k_1024_c12` 创建 Run `runtime-only-profile-20260922`（profile）。

- `2026-09-22T09:26:55Z` Run `runtime-only-profile-20260922` 记录为 `pass`；正确性为 `pass`。Eight-rank runtime-only trace captured and parsed for all 8 real-weight cycles. Steady median host cycle 536.705 ms under profiler; target 466.646 ms, proposer 66.825 ms, DSpark host common refresh 12.483 ms (p90 43.033 ms), acceptance 1.220 ms, state advance 0.700 ms. Device interval union 484.286 ms and HCCL intervals heavily overlap compute; profiler timing is attribution evidence, not an E2E baseline.

- `2026-09-22T09:27:18Z` 暂存知识变化 `extreme-runtime-critical-path-20260922`：The eight-rank real-weight Extreme trace attributes steady profiled host wall primarily to target (median 466.646 ms) and proposer (66.825 ms); acceptance (1.220 ms), state advance (0.700 ms), target preparation (1.065 ms) and draft commit (0.100 ms) are secondary. Device compute and HCCL intervals overlap, so their interval unions are not additive.

- `2026-09-22T09:27:18Z` 暂存知识变化 `dspark-host-mirror-barrier-20260922`：DSpark common-state refresh is the highest-value immediately removable framework residue: three serial device-to-host mirrors create a steady median 12.483 ms host barrier (p90 43.033 ms). Preserve exact DSA semantics by retaining fixed bootstrap CPU metadata and asynchronously copying only the 12 acceptance counts, then advancing runtime-owned host mirrors while the current proposer executes.

- `2026-09-22T09:27:35Z` 主控结论为 `ACCEPTED`。All eight ranks produced parsed runtime-only traces with correctness preserved. Scope and device-union analysis identifies the target/proposer critical path and isolates DSpark CPU mirror refresh as a removable synchronization residue, satisfying the frozen attribution and candidate-selection goal.
