# Loop 目标：M026/M028/M030 NPU MoE路由与必要工作合同

- Loop ID：`L019-npu-moe-routing-work-contract`
- 模式：`spec`
- 所属 Task：`sglang-pd-spec-36`
- 执行 Skill：`NONE`
- 冻结时间：`2026-10-18T21:15:23Z`

## 目标/假设

MoE physicalwork depends actualNPU runner/dispatch, groupedrouting andsharedexpert placement.8logicalroutes pertoken do not imply8experts globally orbalancedrankwork; requestconcurrency doesnotfixMTP/prefilltokenrows.

## 允许修改的路径

- `/Users/wio/work/sglang-learning-lab/mechanism-task-20261018`
- `/Users/wio/work/sglang-learning-lab/repo-export/tuning-brainstorm`

## 目标用例

- source-contract
- c2-short
- c3-long

## 接受条件

Trace fixed installed GLM-DSA target/draft MoE router through actualNPU selection andquant method; freeze minimal routing/permutation orsupport checks andrankwork/communication bounds withunknown native rates explicit.

## 否定条件

Assume genericDeepEP means activeNPUbackend; count8experts globally orusebatch2/3asactualtokenrows; claimbalancedload/peak orreduceTopK/drop expertwork/disableMTP.

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
