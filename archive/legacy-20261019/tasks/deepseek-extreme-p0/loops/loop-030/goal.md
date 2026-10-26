# Loop 目标：Certify real-weight Extreme Runtime milestone

- Loop ID：`loop-030`
- 模式：`correctness`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-22T08:23:02Z`

## 目标/假设

The existing run17 records are sufficient to certify the first real-weight standalone continuous decode milestone without repeating validated hardware experiments.

## 允许修改的路径

- `evidence`
- `tasks/deepseek-extreme-p0`
- `PROJECT_STATE.md`
- `SPECIALIZED_RUNTIME.md`
- `RESULTS.md`

## 目标用例

- mixed_32k_1024_c12
- correctness_short

## 接受条件

Eight rank records exist and agree; every rank passed eight cycles, pre-ModelRunner handoff, zero oracle target calls, no retained ModelRunner, 67 caches, exact state advance and valid acceptance counts.

## 否定条件

Any rank record is missing, differs in final state, reports a failed invariant, or shows a generic target call after handoff.

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
