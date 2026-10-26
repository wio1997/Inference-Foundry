# Loop 状态

- Loop：`loop-011`
- 标题：Remove QLI metadata NPU scalar synchronization
- 模式：`optimization`
- 目标用例：`mixed_32k_1024_c12`, `cold_32k_128_c1`, `correctness_short`
- 执行 Skill：`NONE`
- 状态：`PIVOTED`
- 结论：`PIVOTED`
- 下一步：Loop012: apply CPU QLI maxima only when num_prefills>0; verify parity and matched mixed/cold workloads while leaving decode path original
- 更新时间：`2026-09-20T19:20:20Z`

## Run 记录

- `runs/mixed_32k_1024_c12/qli-cpu-candidate-20260920`
- `runs/cold_32k_128_c1/paired-baseline-20260920`

## 阻塞项

- 暂无

## 待归约知识

- 知识变化：`0`
- 基线变化：`0`
