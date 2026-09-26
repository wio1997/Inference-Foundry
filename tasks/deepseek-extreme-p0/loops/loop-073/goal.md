# Loop 目标：Full-active MoE replicated-gate placement across existing hidden AllGather

- Loop ID：`loop-073`
- 模式：`design`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-26T18:05:35Z`

## 目标/假设

At real all-active DP1xTP8 layer4, moving identical per-token FP32 replicated gate after existing hidden AllGather removes the logits AllGather without changing routing and may shorten complete all-rank MoE critical path despite 176.16MFLOP/rank extra gate work

## 允许修改的路径

- `scripts`
- `evidence/20260926_loop073_gate_placement`
- `performance_knowledge`

## 目标用例

- mixed_32k_1024_c12

## 接受条件

All8 same-prestate global96 logits/routes and complete MoE output pass predeclared numerical controls; private complete endpoint improves reproducibly before live formal E2E

## 否定条件

Gate weights differ, routing/numerics violate controls, or full endpoint does not improve beyond A/A drift; no other path claimed

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
