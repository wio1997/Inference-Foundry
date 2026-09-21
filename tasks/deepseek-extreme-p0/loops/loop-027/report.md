# Loop 报告：loop-027

- 结论：`PIVOTED`
- 决定时间：`2026-09-21T12:04:20Z`
- 模式：`optimization`
- 冻结目标是否满足：`no`
- 可比性：`invalid`
- 因果结论：`partially-supported`

## 决定依据

Legacy DSpark is hard-disabled from graph; the available v2 DSpark graph path fails before capture because generic KV-group discovery finds no draft attention group for this DeepSeek V4 checkpoint. Stock v2 integration is therefore not an immediately runnable graph solution. Use the working legacy path as semantic/operator oracle and extract the fixed proposer-target execution contract for a specialized runtime.

## 证据

- `/data/wio/Inference_Foundry/evidence/20260921_loop027_graph_feasibility/source_audit.json`
- `/data/wio/Inference_Foundry/evidence/20260921_loop027_graph_feasibility/startup_failure.json`

## 下一步

Freeze the specialized runtime contract and map the minimum fixed-shape decode execution chain, beginning with device buffers, KV mutations, proposer inputs/outputs, target verification, and required collectives.
