# Loop 目标：Real two-cycle persistent owner state and full critical path

- Loop ID：`loop-071`
- 模式：`design`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-26T15:41:29Z`

## 目标/假设

One-layer owner16 may remain semantically exact across real adjacent Target cycles when mutable state and shared backing writes are tracked; any local saving matters only if exposed after rank rendezvous

## 允许修改的路径

- `evidence/20260926_loop071_two_cycle`
- `scripts/loop071`
- `tasks/deepseek-extreme-p0`

## 目标用例

- mixed_32k_1024_c12

## 接受条件

Complete backing-byte write inventory and A->A restore control, then all8 exact B->B typed state/QLI/Sparse/Target/acceptance and measured exposed full-cycle advantage

## 否定条件

Unclosed mutable write domain, unstable A->A, first B persistent divergence, or no exposed eight-rank critical-path saving

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
