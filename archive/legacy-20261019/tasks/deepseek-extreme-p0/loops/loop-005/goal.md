# Loop 目标：Test coupled SP and DSA CP disabled path

- Loop ID：`loop-005`
- 模式：`optimization`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-20T16:39:46Z`

## 目标/假设

A valid non-sequence-parallel path with FlashComm1 and dependent DSA CP both disabled reduces TP8 exposed communication enough to improve full E2E throughput despite possible extra compute

## 允许修改的路径

- `/data/wio/Inference_Foundry/scripts/serve.sh`

## 目标用例

- mixed_32k_1024_c12
- correctness_short

## 接受条件

Golden4 exact outputs and three full warmed 48x32K-to-1024 c12 passes improve median output TPS by >5 percent over frozen baseline 543.65 with no material TTFT or TPOT regression

## 否定条件

Correctness fails, valid service cannot load, or robust E2E gain is <=5 percent or latency materially regresses

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
