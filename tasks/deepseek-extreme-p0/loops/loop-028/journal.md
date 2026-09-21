# 执行日志

- `2026-09-21T12:06:00Z` Loop 已冻结。下一步：Map the exact legacy warm-decode call graph and tensor mutation boundary from proposer input through target verification and acceptance; classify constant, device-resident, derived and host-visible state before writing the replay harness.

- `2026-09-21T12:11:48Z` 为用例 `mixed_32k_1024_c12` 创建 Run `source-contract-v0-20260921`（design-check）。

- `2026-09-21T12:11:48Z` Run `source-contract-v0-20260921` 记录为 `pass`；正确性为 `not-applicable`。Mapped the minimum warm decode cycle and classified constant, device-resident, per-cycle-derived and host-visible state. This is a source contract, not replay or performance evidence.

- `2026-09-21T12:11:48Z` 为用例 `mixed_32k_1024_c12` 创建 Run `pointer-stability-20260921`（profile）。

## 2026-09-21T12:13:00Z

- Source contract V0 completed; it classifies the minimal cycle state and generic framework removal candidates.
- Applied flag-gated pointer/shape tracer to legacy proposer. It does not copy tensor values to host.
- API PID897585 loading for correctness plus warm c12 pointer-stability sample.

- `2026-09-21T12:27:51Z` Run `pointer-stability-20260921` 记录为 `invalid`；正确性为 `invalid`。Diagnostic instrumentation insertion was wrong and failed before a model result; no contract or performance inference. Workers cleaned and patch corrected atomically.

- `2026-09-21T12:27:51Z` 为用例 `mixed_32k_1024_c12` 创建 Run `fixed-cycle-replay-20260921`（test）。

## 2026-09-21T12:25:00Z

- First pointer probe invalid due diagnostic insertion point; no model result, all NPUs cleaned.
- Corrected one atomic patch at `_propose` boundary.
- New probe records pointer stability and executes one exact c12 fixed-cycle replay twice without scheduler/request preparation between calls; draft equality is mandatory on all ranks.
- API PID904009 and runner PID2746890 loading.
