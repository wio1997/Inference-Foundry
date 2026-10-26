# Loop 目标：DSpark eager device fragmentation critical path

- Loop ID：`loop-021`
- 模式：`optimization`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-21T04:04:17Z`

## 目标/假设

The eager DSpark draft path launches a repeated Index/Pad/Copy/event task chain on its critical path; exact CPU-parent and source attribution can isolate a bounded fusion or reuse candidate with material mixed-workload value.

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

Identify a reproducible non-overlapped task chain, its exact source callsite and a semantics-safe candidate whose estimated service value exceeds frozen run noise, then pass correctness and mixed E2E benchmark.

## 否定条件

The frequent tasks are necessary, overlapped, target-owned, or individually too small to exceed run noise; unsupported graph capture is required for material gain.

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
