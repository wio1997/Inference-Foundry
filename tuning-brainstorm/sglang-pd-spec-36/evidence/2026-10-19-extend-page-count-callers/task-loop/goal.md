# Loop 目标：M005/M017 alloc_extend 页计数与同步边界

- Loop ID：`L023-extend-page-count-propagation`
- 模式：`spec`
- 所属 Task：`sglang-pd-spec-36`
- 执行 Skill：`NONE`
- 冻结时间：`2026-10-18T22:32:56Z`

## 目标/假设

NPU allocator optional num_new_pages may avoid device sum/item; actual caller propagation must determine whether this is already active or an eligible remaining cost under PD+MTP.

## 允许修改的路径

- `/Users/wio/work/sglang-learning-lab/mechanism-task-20261018`
- `/Users/wio/work/sglang-learning-lab/repo-export/tuning-brainstorm`

## 目标用例

- source-contract
- c2-short
- c3-long

## 接受条件

Trace canonical real callers and count ownership/types; verify source/CPU call contract where possible, preserving native sync/performance limits and avoiding duplicate optimization.

## 否定条件

Infer an active synchronization bottleneck from allocator signature alone; substitute CPU timing for NPU; claim existing explicit page count as new optimization; bypass native small-page success gate.

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
