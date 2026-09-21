# Loop 状态

- Loop：`loop-014`
- 标题：Find exposed decode metadata host work
- 模式：`optimization`
- 目标用例：`mixed_32k_1024_c12`, `correctness_short`
- 执行 Skill：`NONE`
- 状态：`PIVOTED`
- 结论：`PIVOTED`
- 下一步：Restore diagnostic patch and baseline-flag service, then Loop015 examine cold prefill ScatterNdUpdateSk/Compressor and HCCL critical path for a falsifiable improvement.
- 更新时间：`2026-09-20T21:04:13Z`

## Run 记录

- `runs/mixed_32k_1024_c12/decode-host-source-audit-20260920`
- `runs/mixed_32k_1024_c12/prepare-stage-trace-20260920`
- `runs/mixed_32k_1024_c12/attention-builder-trace-20260920`

## 阻塞项

- 暂无

## 待归约知识

- 知识变化：`0`
- 基线变化：`0`
