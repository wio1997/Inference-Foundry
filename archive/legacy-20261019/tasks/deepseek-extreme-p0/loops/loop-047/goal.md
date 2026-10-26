# Loop 目标：Quantify and remove inactive-slot target tail

- Loop ID：`loop-047`
- 模式：`optimization`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-25T07:03:24Z`

## 目标/假设

In the frozen c12 DeepSeek Extreme cohort, inactive slots after early completion still consume material 96-token target graph work; a product-specific active-slot execution image can shorten latest-rank cohort wall while preserving all remaining slot/KV/DSpark state.

## 允许修改的路径

- `scripts`
- `runtime`
- `bootstrap`

## 目标用例

- mixed_32k_1024_c12

## 接受条件

First quantify eight-rank active-slot trajectory and measured tail target cadence; then prove a semantically exact smaller active-slot execution path across continuous cycles and improve latest-rank wall before repeated formal E2E.

## 否定条件

Parked-slot exposure is mostly overlapped or required, shape variants/reordering destroy correctness, or smaller active-slot path does not lower latest-rank completion including switching overhead.

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
