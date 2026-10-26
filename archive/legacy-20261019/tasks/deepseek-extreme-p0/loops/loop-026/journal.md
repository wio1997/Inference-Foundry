# 执行日志

- `2026-09-21T09:07:34Z` Loop 已冻结。下一步：Freeze the cycle-time/acceptance break-even model from saved evidence, audit the three-layer forward dependencies, then implement an environment-gated skip of only the middle draft layer for a no-profiler correctness and c12 screen.

- `2026-09-21T09:10:11Z` 为用例 `mixed_32k_1024_c12` 创建 Run `skip-middle-screen-20260921`（benchmark）。

- `2026-09-21T10:07:04Z` Run `skip-middle-screen-20260921` 记录为 `pass`；正确性为 `pass`。Middle-layer bypass passed functional and12/12 but proposer model_run fell only23.21% (31.106 to23.887ms rank median), advanced tokens collapsed to1.363/cycle from3.54-3.64, and output TPS fell55.85% to217.092 with TPOT+173.29%.

- `2026-09-21T10:07:04Z` 为用例 `mixed_32k_1024_c12` 创建 Run `k7-reference-loop020`（benchmark）。

- `2026-09-21T10:07:04Z` Run `k7-reference-loop020` 记录为 `pass`；正确性为 `pass`。Saved k7 control:491.698 tok/s,17.554ms TPOT, proposer model31.106ms rank median.

- `2026-09-21T10:07:04Z` 已记录对比（`comparable=yes`）：Middle-layer bypass: proposer model -23.21%, but TPS -55.85%, TPOT +173.29%, advanced tokens collapse to1.363/cycle; correctness remains protected by target verification.

- `2026-09-21T10:07:04Z` 主控结论为 `REJECTED`。Removing one of three trained draft layers saves23.21% proposer model time but destroys proposal quality: advanced tokens fall to1.363/cycle, far below3.15 break-even; output TPS drops55.85% to217.092, close to target-only208.047. Full three-layer semantics are necessary.
