# Loop 目标：Locate warm TP8 decode critical path

- Loop ID：`loop-013`
- 模式：`optimization`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-20T19:49:37Z`

## 目标/假设

A short warmed c12 TP0 trace plus host and HCCL timing can isolate a high-value exposed decode gap; prior c4 draft_token CPU scope was about 47 ms per iteration but is not a device critical-path attribution.

## 允许修改的路径

- `scripts`
- `evidence`
- `tasks`
- `PROJECT_STATE.md`
- `PERFORMANCE_MAP.md`
- `ACHIEVABLE_BOUND.md`
- `RESULTS.md`

## 目标用例

- mixed_32k_1024_c12

## 接受条件

Capture a healthy warmed c12 trace, distinguish device compute/HCCL/idle and host launch/synchronization, and select one falsifiable code or configuration intervention.

## 否定条件

The trace cannot resolve the c12 critical path or overhead dominates; then pivot to lower-overhead targeted instrumentation without a performance claim.

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
