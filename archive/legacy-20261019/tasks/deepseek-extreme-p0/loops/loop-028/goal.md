# Loop 目标：Fixed decode-cycle contract extraction

- Loop ID：`loop-028`
- 模式：`design`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-21T12:06:00Z`

## 目标/假设

For the frozen DP1TP8 DSpark7 serving envelope, one warm decode cycle can be expressed as a stable set of preallocated device buffers, fixed collectives and explicit mutations without per-step scheduler/request objects or dynamic metadata construction.

## 允许修改的路径

- `scripts`
- `evidence`
- `tasks/deepseek-extreme-p0`
- `SPECIALIZED_RUNTIME.md`
- `PROJECT_STATE.md`
- `PERFORMANCE_MAP.md`
- `ACHIEVABLE_BOUND.md`
- `RESULTS.md`

## 目标用例

- mixed_32k_1024_c12
- correctness_short

## 接受条件

Produce an executable contract and one-cycle capture/replay harness using real weights/operators; oracle and replay match draft outputs, accepted tokens and mutated state at defined tolerances, with every remaining host dependency classified.

## 否定条件

A required model-semantic input or KV/collective dependency cannot be represented with fixed slots and explicit per-cycle state, or replay cannot match the working oracle after boundary correction.

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
