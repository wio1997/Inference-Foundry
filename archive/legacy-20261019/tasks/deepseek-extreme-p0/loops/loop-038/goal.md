# Loop 目标：Capture a bounded target graph cycle trace

- Loop ID：`loop-038`
- 模式：`optimization`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-24T14:23:03Z`

## 目标/假设

Starting and stopping torch_npu.profiler on a Runtime cycle schedule can capture a few steady 96-token target cycles on all eight TP ranks without unstable HTTP RPC timing; bounded traces can expose target compute, HCCL overlap and gaps.

## 允许修改的路径

- `runtime`
- `scripts`
- `diagnostics`

## 目标用例

- mixed_32k_1024_c12

## 接受条件

A legal 12x1024 c12 cohort passes all eight Runtime ranks; each rank has a bounded parsed trace with extreme::target scopes and kernel rows mapped to the same cycle numbers; select a quantified falsifiable optimization candidate.

## 否定条件

Cycle-scheduled profiling changes correctness, deadlocks ranks, produces oversized/unmappable traces, or reveals no removable target component.

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
