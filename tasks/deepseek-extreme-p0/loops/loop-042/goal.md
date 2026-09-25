# Loop 目标：Attribute and remove target rank phase skew

- Loop ID：`loop-042`
- 模式：`optimization`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-25T01:28:26Z`

## 目标/假设

A persistent inter-rank phase offset entering each c12 target cycle makes early TP ranks wait in the first reduce-scatter; synchronizing or pacing launch without changing token semantics can remove exposed wait from the cohort critical path, provided the late-rank arrival itself can move earlier.

## 允许修改的路径

- `runtime`
- `scripts`
- `diagnostics`

## 目标用例

- mixed_32k_1024_c12

## 接受条件

Identify a causal, product-valid scheduler or execution change that reduces unprofiled eight-rank cohort cycle wall and passes same-state target/KV/DSpark correctness, then legal formal E2E above Run99.

## 否定条件

Existing traces show phase skew is upstream, one-time, or critical-path-neutral, or no safe intervention can advance the last arrival.

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
