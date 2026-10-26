# Loop 目标：Extreme full-chain acceptance and cycle attribution

- Loop ID：`loop-035`
- 模式：`optimization`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-23T03:44:19Z`

## 目标/假设

The formal throughput gap is dominated by sustained proposal acceptance collapse and/or excess per-cycle runtime cost; existing E2E records can separate the two before any architectural rewrite.

## 允许修改的路径

- `/data/wio/Inference_Foundry`

## 目标用例

- mixed_32k_1024_c12

## 接受条件

Quantify accepted outputs per cycle and runtime stage critical path on the fixed real-weight c12 workload, identify one causal high-value structural change, and validate its correctness before performance claims.

## 否定条件

Acceptance is already Stock-equivalent or profile shows a different bottleneck; pivot to the measured critical path without preserving this hypothesis.

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
