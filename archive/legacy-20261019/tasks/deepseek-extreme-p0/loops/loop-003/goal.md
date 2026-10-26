# Loop 目标：Diagnose current TP8 cold prefill and warm decode gaps

- Loop ID：`loop-003`
- 模式：`optimization`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-20T15:59:58Z`

## 目标/假设

Current topology measurements can separate prefill, draft/target, TP8 communication, memory and host exposed time enough to rank a >=5 percent candidate

## 允许修改的路径

- `/data/wio/Inference_Foundry`

## 目标用例

- cold_32k_128_c1
- mixed_32k_1024_c12

## 接受条件

Cold-prefill baseline and profiler or timing evidence identify a reproducible exposed gap with bounded measurement uncertainty

## 否定条件

No repeatable gap or profiler data cannot distinguish competing mechanisms

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
