# Loop 目标：Warm decode DSA-CP metadata builder active-path attribution

- Loop ID：`loop-019`
- 模式：`optimization`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-21T01:13:57Z`

## 目标/假设

A reusable or vectorizable part of eight DSA-CP metadata builders dominates the no-profiler TP0 prepare span and can reduce full-service warm decode latency.

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

Identify one active-callsite, semantics-preserving minimal patch; pass functional and numeric checks, then exceed frozen mixed c12 TPS spread in paired benchmark without TTFT regression.

## 否定条件

Builder spans are inclusive or causally overlapped, no safe reuse exists, or same-protocol E2E gain does not exceed noise.

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
