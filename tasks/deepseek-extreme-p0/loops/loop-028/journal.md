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

- `2026-09-21T12:51:38Z` Run `fixed-cycle-replay-20260921` 记录为 `pass`；正确性为 `pass`。At c12 the already-materialized real DSpark proposer closure replayed exactly on all 8 TP ranks (draft tokens equal, shape 12x7); short functional checks passed. This run did not compare target verification, accepted tokens, or full-cycle state mutation and is diagnostic, not a performance benchmark.

- `2026-09-21T12:51:38Z` 暂存知识变化 `loop028-c12-pointer-contract`：At the legacy DSpark proposer boundary for warm c12 pure decode, all observed tensor shapes and strides were fixed over 172 calls per rank; ten fields kept one address per rank while target_token_ids and target_positions changed address.

- `2026-09-21T12:51:38Z` 暂存知识变化 `loop028-proposer-replay-boundary`：The already-materialized c12 real-weight DSpark proposer closure can execute twice without intervening scheduler/request preparation and produces exactly equal 12x7 draft token tensors on all eight TP ranks; this does not establish parity for target verification, acceptance, or complete cycle state mutation.

- `2026-09-21T12:51:38Z` 主控结论为 `PIVOTED`。Loop028 established a fixed-shape c12 proposer contract and an executable exact in-process replay boundary on 8/8 ranks, but the frozen success gate required target verification, accepted-token parity and complete mutated-state comparison. Those boundaries were not captured, so the full standalone fixed-cycle claim is not yet supported.
