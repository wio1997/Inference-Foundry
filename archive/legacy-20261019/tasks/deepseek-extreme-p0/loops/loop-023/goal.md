# Loop 目标：DSpark speculative length efficiency sweep

- Loop ID：`loop-023`
- 模式：`optimization`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-21T06:05:32Z`

## 目标/假设

With only 3.54-3.64 mean tokens advanced from seven-token drafts, a smaller speculative length reduces wasted draft and verification work enough to improve frozen mixed output TPS beyond run noise while preserving correctness.

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
- correctness_short

## 接受条件

A same-software DP1 TP8 k sweep identifies a length whose repeated frozen mixed output TPS exceeds k7 baseline by more than the 4.3 percent baseline spread, with all requests successful and correctness smoke passing.

## 否定条件

No tested smaller length exceeds k7 by the frozen noise threshold, or lower acceptance coverage offsets reduced work.

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
