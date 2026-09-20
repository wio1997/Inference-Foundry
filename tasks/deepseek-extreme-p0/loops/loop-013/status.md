# Loop 状态

- Loop：`loop-013`
- 标题：Locate warm TP8 decode critical path
- 模式：`optimization`
- 目标用例：`mixed_32k_1024_c12`
- 执行 Skill：`NONE`
- 状态：`RUNNING`
- 结论：`PENDING`
- 下一步：执行 Run decode-c12-torch-profile-20260920：Restart baseline-flag service with torch-NPU profiler without stacks; scripts/profile_decode_c12.py full warmup then 12×512 c12 profile
- 更新时间：`2026-09-20T19:53:48Z`

## Run 记录

- `runs/mixed_32k_1024_c12/decode-c12-msprof-20260920`
- `runs/mixed_32k_1024_c12/decode-c12-msprof-absolute-20260920`
- `runs/mixed_32k_1024_c12/decode-c12-torch-profile-20260920`

## 阻塞项

- 暂无

## 待归约知识

- 知识变化：`0`
- 基线变化：`0`
