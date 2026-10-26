# Loop 目标：Attribute the remaining target graph critical path

- Loop ID：`loop-037`
- 模式：`optimization`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-24T13:13:41Z`

## 目标/假设

After static metadata removes the 8 ms sync, the fixed 96-token target graph still consumes about 46.56 ms/cycle. A runtime-only NPU trace can separate its exposed compute, communication, overlap, and gaps and identify a removable component large enough to make above-Stock throughput robust.

## 允许修改的路径

- `/data/wio/Inference_Foundry/scripts`
- `/data/wio/Inference_Foundry/diagnostics`

## 目标用例

- mixed_32k_1024_c12

## 接受条件

A legal 12x1024 c12 TP8 DSpark7 cohort passes all 8-rank Runtime gates while a parsed target-only NPU profile reports per-cycle device union, compute/communication overlap, top operators and rank spread; select one falsifiable optimization candidate with a quantified bound.

## 否定条件

Profiler overhead or incomplete rank/cycle mapping prevents attribution, or target time is dominated by irreducible model work with no supported high-value candidate; then pivot to acceptance/overlap or stop.

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
