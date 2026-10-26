# Loop 目标：Target expert placement and critical-rank imbalance

- Loop ID：`loop-049`
- 模式：`optimization`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-25T09:20:12Z`

## 目标/假设

Frozen-model linear expert placement leaves persistent per-layer critical-rank weight-load imbalance that contributes materially to target graph completion and HCCL arrival wait

## 允许修改的路径

- `scripts`
- `runtime`
- `evidence`

## 目标用例

- mixed_32k_1024_c12

## 接受条件

Same-cycle live expert counts, per-rank GMM/HCCL timing and placement simulation identify a semantics-preserving placement with plausible >=5ms/cycle target-stage reduction, then correctness-gated eight-rank A/B/A confirms stage benefit before formal E2E

## 否定条件

Active expert load is already balanced across ranks/layers, predicted critical-rank reduction is small, or GMM/collective timing does not correlate with imbalance

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
