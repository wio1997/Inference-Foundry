# Loop 目标：Resource-DAG workload inventory and bound discrimination

- Loop ID：`loop-060`
- 模式：`design`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-25T16:44:51Z`

## 目标/假设

Source-verified real target and prefill workload bytes plus existing HBM/HCCL evidence can define a resource-DAG input inventory and expose the largest unresolved capacity term without assuming profiler task sums are savings

## 允许修改的路径

- `scripts/extreme_resource_inventory.py`
- `evidence/20260926_loop060_resource`
- `ACHIEVABLE_BOUND.md`
- `PERFORMANCE_MAP.md`
- `HANDOFF.md`

## 目标用例

- mixed_32k_1024_c12

## 接受条件

Machine-readable layer/traffic/dependency inventory marks exact versus conditional versus unknown fields and ranks a feasible distinguishing measurement

## 否定条件

Required target or prefill shape/byte quantities cannot be mapped to legal serving source and measured route evidence without incompatible assumptions

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
