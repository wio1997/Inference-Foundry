# Loop 状态

- Loop：`loop-023`
- 标题：DSpark speculative length efficiency sweep
- 模式：`optimization`
- 目标用例：`mixed_32k_1024_c12`, `correctness_short`
- 执行 Skill：`NONE`
- 状态：`REJECTED`
- 结论：`REJECTED`
- 下一步：Restore k7 and test the explicit scheduler warning by raising max_num_batched_tokens from8192 to8288, which restores max_num_scheduled_tokens from8096 to8192 while retaining 96 draft slots.
- 更新时间：`2026-09-21T06:53:47Z`

## Run 记录

- `runs/mixed_32k_1024_c12/k5-screen-20260921`

## 阻塞项

- 暂无

## 待归约知识

- 知识变化：`0`
- 基线变化：`0`
