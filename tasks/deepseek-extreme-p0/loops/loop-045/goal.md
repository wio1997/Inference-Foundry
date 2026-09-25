# Loop 目标：Localize first-token and cohort-tail E2E gap

- Loop ID：`loop-045`
- 模式：`optimization`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-25T03:39:52Z`

## 目标/假设

Frozen product E2E contains avoidable per-cohort bootstrap or parked-slot execution beyond necessary prefill and decode work.

## 允许修改的路径

- `scripts`
- `runtime`
- `bootstrap`

## 目标用例

- mixed_32k_1024_c12

## 接受条件

Legal 12x1024 eight-rank diagnostic attributes >=95% of request envelope; if >=1.75s/cohort uncovered bootstrap is causally removable, implement and pass correctness then formal E2E. Tail compaction only if completed-slot cycles >=15% and state safety proven.

## 否定条件

First-token excess is necessary prefill or unavoidable handoff, or tail parked-slot fraction is small, or proposed edit fails correctness/cadence gates.

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
