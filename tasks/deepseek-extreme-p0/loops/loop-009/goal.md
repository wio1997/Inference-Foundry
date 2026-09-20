# Loop 目标：Locate repeated prepare input scalar sync

- Loop ID：`loop-009`
- 模式：`design`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-20T17:59:53Z`

## 目标/假设

The 29 repeated aten::item calls per warmed decode step arise from a small number of CPU metadata construction sites, and at least one blocks device progress enough to justify a safe minimal removal

## 允许修改的路径

- `/data/wio/Inference_Foundry/scripts`
- `/data/wio/Inference_Foundry/evidence`

## 目标用例

- mixed_32k_1024_c12

## 接受条件

Call stacks identify exact callsites, synchronized time and device idle are aligned, and one bounded change with predicted >=5 percent gain is selected or falsified

## 否定条件

Item calls are already on CPU tensors or entirely profiler overhead, or no exposed >=5 percent opportunity remains

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
