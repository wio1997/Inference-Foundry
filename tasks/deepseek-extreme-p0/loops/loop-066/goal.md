# Loop 目标：c4 owner compact update semantic and resource closure

- Loop ID：`loop-066`
- 模式：`optimization`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-26T09:23:56Z`

## 目标/假设

At fixed c12 Target layer2 entry, each rank needs only the two request owners complete 16 gathered rows for indexer cache update; nonowner replica update may be removed if typed state/cache/QLI consumers are equivalent

## 允许修改的路径

- `scripts`
- `evidence/20260926_loop066_owner`
- `tasks/deepseek-extreme-p0`

## 目标用例

- mixed_32k_1024_c12

## 接受条件

A/A replay stable; private real-entry full96 versus owner16 typed owner state/key/scale, QLI and Sparse comparisons pass; measured same-shape resource and first dependent endpoint improve; allrank Graph and formal E2E gate before KEEP

## 否定条件

Owner16 changes required state/cache/QLI/Sparse under stable A/A baseline, aliases active writes, or measured resource/critical endpoint does not improve enough to justify architecture

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
