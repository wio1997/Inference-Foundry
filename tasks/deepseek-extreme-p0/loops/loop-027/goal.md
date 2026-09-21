# Loop 目标：Exact three-layer DSpark graph feasibility

- Loop ID：`loop-027`
- 模式：`optimization`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-21T10:07:05Z`

## 目标/假设

A fixed-shape pure-decode DSpark proposer can preserve all three trained layers and exact outputs while moving a material part of its eager-only execution into a captured or standalone replay path, reducing proposer cost enough to improve E2E.

## 允许修改的路径

- `scripts`
- `evidence`
- `tasks/deepseek-extreme-p0`
- `patches`
- `PROJECT_STATE.md`
- `PERFORMANCE_MAP.md`
- `ACHIEVABLE_BOUND.md`
- `RESULTS.md`

## 目标用例

- mixed_32k_1024_c12
- correctness_short

## 接受条件

Source and minimal runtime probe identify a capturable boundary; exact logits/tokens match eager; proposer latency improves materially in a repeatable screen; only then integrate and run mixed E2E.

## 否定条件

Dynamic metadata, HCCL, custom MoE/DSA ops or mutable KV state prevent safe replay, or exact replay gain is below E2E noise.

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
