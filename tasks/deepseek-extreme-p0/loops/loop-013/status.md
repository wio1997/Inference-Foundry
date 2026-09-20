# Loop 状态

- Loop：`loop-013`
- 标题：Locate warm TP8 decode critical path
- 模式：`optimization`
- 目标用例：`mixed_32k_1024_c12`
- 执行 Skill：`NONE`
- 状态：`PIVOTED`
- 结论：`PIVOTED`
- 下一步：Loop014: inspect repeated draft MoE/DSA host self time and cross-rank synchronization; identify a minimal correctness-preserving change with predicted measurable TPS benefit or pivot again.
- 更新时间：`2026-09-20T20:26:35Z`

## Run 记录

- `runs/mixed_32k_1024_c12/decode-c12-msprof-20260920`
- `runs/mixed_32k_1024_c12/decode-c12-msprof-absolute-20260920`
- `runs/mixed_32k_1024_c12/decode-c12-torch-profile-20260920`
- `runs/mixed_32k_1024_c12/decode-c12-torch-profile-retry-20260920`

## 阻塞项

- 暂无

## 待归约知识

- 知识变化：`0`
- 基线变化：`0`
