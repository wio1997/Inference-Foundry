# Loop 目标：Freeze current DP1 TP8 baseline

- Loop ID：`loop-001`
- 模式：`baseline`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-20T15:39:40Z`

## 目标/假设

Current exact runtime can serve correctly and produce repeatable TTFT TPOT TPS on protected workload

## 允许修改的路径

- `/data/wio/Inference_Foundry`

## 目标用例

- mixed_32k_1024_c12

## 接受条件

48/48 complete and repeated metrics within observed noise

## 否定条件

Service or correctness fails, or metrics unstable

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
