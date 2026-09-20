# Loop 目标：Accept corrected warm-cache DP1 TP8 baseline

- Loop ID：`loop-002`
- 模式：`baseline`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-20T15:58:54Z`

## 目标/假设

Corrected stream parser yields reproducible warm-cache TTFT TPOT and TPS with 48/48 successful requests

## 允许修改的路径

- `/data/wio/Inference_Foundry`

## 目标用例

- mixed_32k_1024_c12

## 接受条件

Three corrected 48/48 runs with recorded spread and four deterministic golden outputs

## 否定条件

Any failed request, incorrect token count, or unbounded measurement variation

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
