# 执行日志

- `2026-09-21T10:07:05Z` Loop 已冻结。下一步：Audit the three-layer forward for graph-unsafe state/HCCL/KV mutations and run the smallest exact-shape capture feasibility probe; do not enable unsupported full proposer graph in production.

## 2026-09-21T11:07:00Z

- Source audit: legacy DSpark proposer disables graph unconditionally; v2 runner contains a registered AscendDSparkSpeculator with an Ascend query graph manager.
- Started minimal startup/correctness probe with VLLM_USE_V2_MODEL_RUNNER=1 and enforce_eager=false; all frozen model/parallel/scheduler settings unchanged.
- API PID 891755 loading. No correctness or performance result yet.

- `2026-09-21T12:04:20Z` 为用例 `mixed_32k_1024_c12` 创建 Run `v2-graph-startup-20260921`（test）。

- `2026-09-21T12:04:20Z` Run `v2-graph-startup-20260921` 记录为 `fail`；正确性为 `invalid`。v2 DSpark path failed during KV initialization: no draft attention groups found; graph capture and correctness were not reached; cleanup returned all NPUs idle.

- `2026-09-21T12:04:20Z` 主控结论为 `PIVOTED`。Legacy DSpark is hard-disabled from graph; the available v2 DSpark graph path fails before capture because generic KV-group discovery finds no draft attention group for this DeepSeek V4 checkpoint. Stock v2 integration is therefore not an immediately runnable graph solution. Use the working legacy path as semantic/operator oracle and extract the fixed proposer-target execution contract for a specialized runtime.
