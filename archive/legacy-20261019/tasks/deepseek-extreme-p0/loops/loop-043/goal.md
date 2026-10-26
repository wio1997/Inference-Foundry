# Loop 目标：Reduce eager DSpark Host dispatch cost

- Loop ID：`loop-043`
- 模式：`optimization`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-25T02:35:37Z`

## 目标/假设

The frozen c12 DSpark7 proposer retains repeated eager Python, attention-metadata and operator launch work (Run140 ~39.6ms Host CPU within proposer model, ~30.3ms in _runnable), and a product-specific fixed replay or execution image can shorten the actual latest-rank cycle cadence while preserving draft and KV state.

## 允许修改的路径

- `runtime`
- `bootstrap`
- `scripts`
- `patches`

## 目标用例

- mixed_32k_1024_c12

## 接受条件

Same-state eight-rank target/DSpark/KV correctness, measurable reduction in unprofiled latest-rank steady cycle cadence, and legal repeated formal E2E throughput above Run99 571.681 tok/s.

## 否定条件

Most CPU work is fully hidden under target or replay cannot preserve DSpark dynamic state/metadata, or unprofiled cycle and formal E2E do not improve.

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
