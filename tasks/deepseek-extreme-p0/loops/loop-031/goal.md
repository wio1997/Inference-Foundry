# Loop 目标：Profile and close the runtime-owned decode DAG

- Loop ID：`loop-031`
- 模式：`integration`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-22T08:23:32Z`

## 目标/假设

The certified Extreme-owned chain can be traced without re-entering generic ModelRunner target execution, exposing the remaining CPU metadata refresh, launch, communication and operator regions needed to choose the first device-resident, replay/persistent or fusion implementation step.

## 允许修改的路径

- `runtime`
- `bootstrap`
- `scripts`
- `patches`
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

Capture complete eight-rank stage/device attribution for the standalone eight-cycle chain; enumerate remaining host-dependent per-cycle inputs and freeze one evidence-ranked runtime-boundary migration or fusion/replay candidate without changing decode semantics.

## 否定条件

The bootstrap handoff cannot expose a complete standalone trace, or execution still depends on hidden per-cycle ModelRunner/Scheduler control absent from Extreme Runtime state.

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
