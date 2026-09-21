# Loop 目标：Restore speculative scheduler token capacity

- Loop ID：`loop-024`
- 模式：`optimization`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-21T06:53:47Z`

## 目标/假设

At k7/max_seqs16, raising max_num_batched_tokens from8192 to8288 restores max_num_scheduled_tokens from8096 to8192 and improves the frozen mixed workload without changing model semantics.

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

Candidate passes startup and correctness, log confirms max_num_scheduled_tokens>=8192, and repeated frozen mixed output TPS improves beyond the4.3 percent baseline spread without TTFT regression.

## 否定条件

Startup/memory fails, the scheduler capacity remains8096, or comparable mixed performance is within noise or worse.

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
