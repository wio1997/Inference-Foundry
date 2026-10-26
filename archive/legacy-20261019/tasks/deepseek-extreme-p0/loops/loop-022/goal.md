# Loop 目标：DSpark MoE event dependency critical path

- Loop ID：`loop-022`
- 模式：`optimization`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-21T05:07:33Z`

## 目标/假设

The large EVENT_WAIT spans attributed to vllm::moe_forward_shared reflect a main-path stream dependency or load imbalance that can be reduced by a bounded scheduling change without altering MoE numerics.

## 允许修改的路径

- `scripts`
- `evidence`
- `tasks/deepseek-extreme-p0`
- `patches`
- `PROJECT_STATE.md`
- `PERFORMANCE_MAP.md`
- `ACHIEVABLE_BOUND.md`
- `RESULTS.md`

## 目标用例

- mixed_32k_1024_c12

## 接受条件

Identify exact event producer and consumer streams/source lines, prove a non-overlapped critical-path wait above baseline noise, implement the smallest safe scheduling change, and pass correctness plus mixed E2E benchmark.

## 否定条件

EVENT_WAIT is only an asynchronous-stream dependency fully hidden by useful work, or its removable main-path component is below run noise.

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
