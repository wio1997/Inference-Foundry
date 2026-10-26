# Loop 目标：DSpark middle-layer cost versus acceptance

- Loop ID：`loop-026`
- 模式：`optimization`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-21T09:07:34Z`

## 目标/假设

The middle of three DSpark draft layers can be bypassed to remove roughly one third of proposer model work while retaining enough proposal quality to improve cost per advanced token and mixed E2E throughput.

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

A flag-gated middle-layer bypass passes exact service correctness; measured proposer time drops materially; advanced tokens remain above the precomputed break-even threshold; bounded c12 TPS improves beyond4.3%, then repeated frozen32K-to-1K confirms KEEP.

## 否定条件

Acceptance falls below break-even, proposer time does not drop materially, or bounded E2E TPS is within noise/worse.

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
