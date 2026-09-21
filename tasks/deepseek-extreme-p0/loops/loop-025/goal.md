# Loop 目标：Bound DSpark net E2E value against target-only

- Loop ID：`loop-025`
- 模式：`optimization`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-21T08:05:27Z`

## 目标/假设

Valid k7 DSpark materially improves mixed c12 output TPS over target-only decoding; a matched no-spec screen can quantify its net value and decide whether to optimize proposer/verification or exit the speculative path.

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

Target-only and k7 use identical model, TP8, warmup and workload; both pass correctness. The screen establishes a difference larger than4.3%, then the winning path advances to repeated frozen32K-to-1K benchmark.

## 否定条件

Difference is within noise, target-only is faster, or target-only is not runnable under otherwise matched settings.

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
