# Loop 报告：loop-049

- 结论：`PIVOTED`
- 决定时间：`2026-09-25T09:29:43Z`
- 模式：`optimization`
- 冻结目标是否满足：`no`
- 可比性：`not-applicable`
- 因果结论：`partially-supported`

## 决定依据

Run190 finds static expert_map execution/weight-loader mismatch needing substantial correctness integration. Run191 fixed-map pair-swap simulation over two captured cycles gains57/78 active reads in sample but only+3/-1 on the other cycle; no robust transferable reduction. Run189 ideal2.0-2.2ms/cycle is unattainable arithmetic, and separate-service GMM rank spread<0.45ms. Deprioritize placement; no code or formal E2E.

## 证据

- `/data/wio/Inference_Foundry/evidence/20260925_loop049_expert_balance/run190/static_placement_source_audit.md`
- `/data/wio/Inference_Foundry/evidence/20260925_loop049_expert_balance/run191/analysis.json`

## 下一步

Open Loop050 to isolate target graph exposed communication and overlap with GMM/other compute using same-state per-rank traces; choose a concrete removable mechanism before code.
