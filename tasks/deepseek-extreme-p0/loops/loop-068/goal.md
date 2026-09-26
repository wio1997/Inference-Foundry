# Loop 目标：Complete DSA producer owner semantics

- Loop ID：`loop-068`
- 模式：`correctness`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-26T13:00:32Z`

## 目标/假设

For one real Target layer2 c4 decode prestate, owner16 full producer chain yields the same local QLI, Sparse output and protected owner cache values as full96 while maintaining private shared storage aliases

## 允许修改的路径

- `scripts`
- `evidence/20260926_loop068_producer`
- `tasks/deepseek-extreme-p0`

## 目标用例

- mixed_32k_1024_c12

## 接受条件

All8 ranks A/A/B/A same-prestate full producer outputs, typed owner current writes, QLI indices and Sparse output exact or within frozen numerical tolerance; no live cache writes; carrier 12x1024 and runtime gates pass

## 否定条件

Any deterministic owner cache or Sparse consumer divergence, unpreserved alias, failed Runtime gate or invalid resource conditions

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
