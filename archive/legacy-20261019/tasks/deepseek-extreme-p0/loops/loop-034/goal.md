# Loop 目标：Serve fixed 48-request cohorts through Extreme Runtime

- Loop ID：`loop-034`
- 模式：`integration`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-22T13:06:57Z`

## 目标/假设

For the frozen greedy 48x32K-to-1024 c12 workload, each 12-request cohort can remain inside the Extreme-owned decode loop until its exact per-request output limit, drain device outputs once, and return one bulk completion frame to the serving control plane; this preserves request outputs while keeping ModelRunner and Scheduler off the per-cycle hot path.

## 允许修改的路径

- `/data/wio/Inference_Foundry`

## 目标用例

- mixed_32k_1024_c12
- correctness_short

## 接受条件

CPU serving semantics pass; on 8x910B3 all 48 requests complete with exactly 1024 output tokens, no post-handoff oracle target calls, four Extreme cohorts, rank parity, and a valid same-protocol E2E output-TPS/TTFT/TPOT result.

## 否定条件

The bulk completion contract corrupts scheduler/spec accounting, output lengths/tokens, KV/DSA state, rank parity, or cannot complete the frozen client workload without per-cycle ModelRunner/Scheduler re-entry.

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
