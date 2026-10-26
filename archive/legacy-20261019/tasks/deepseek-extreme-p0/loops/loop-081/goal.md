# Loop 目标：Fixed-work production dependency DAG and legal TP scheduling cut

- Loop ID：`loop-081`
- 模式：`design`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-27T22:43:58Z`

## 目标/假设

The selected FULL96 layer0 DSA output projection has a live typed producer->TP collective->consumer edge whose exposed scheduling window can be measured without changing DSpark7 acceptance, cycles or outputs

## 允许修改的路径

- `scripts`
- `evidence/20260928_loop081_bound`

## 目标用例

- mixed_32k_1024_c12

## 接受条件

All8 same-generation branch, tensor storage/producer/collective/consumer and rank-local readiness identities pass; any time claim separates current interval from necessary service

## 否定条件

Loaded branch differs from assumed TP cut, trace identity cannot be joined, fixed-work ledger changes, or marker perturbation prevents a reliable timing witness

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
