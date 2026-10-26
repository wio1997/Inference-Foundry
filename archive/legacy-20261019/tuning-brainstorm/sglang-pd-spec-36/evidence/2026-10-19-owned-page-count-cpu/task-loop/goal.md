# Loop 目标：M005/M017 producer 拥有者显式页计数设计

- Loop ID：`L025-owned-explicit-page-count`
- 模式：`design`
- 所属 Task：`sglang-pd-spec-36`
- 执行 Skill：`NONE`
- 冻结时间：`2026-10-18T23:07:49Z`

## 目标/假设

Count at verified ordinary/PD/EAGLE data owners and opt into the existing NPU count argument; preserve NPU allocator class/default semantics and legacy helper calls. Native synchronization/lifetime correctness and gains remain unproven.

## 允许修改的路径

- `/Users/wio/work/sglang-learning-lab/mechanism-task-20261018`

## 目标用例

- source-contract
- c2-short
- c3-long

## 接受条件

Produce isolated reviewable owner/bridge patch for all three target routes, prove original allocator class AST and legacy defaults unchanged, then freeze differential caller tests and native gates.

## 否定条件

Auto-trust CPU values in unqualified allocator default; skip PD or MTP; add count kwargs to unknown allocators; claim CPU/boundary checks prove native success or performance.

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
