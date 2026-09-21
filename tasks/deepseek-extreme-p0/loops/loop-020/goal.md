# Loop 目标：Warm c12 DSpark draft and target verification critical-path attribution

- Loop ID：`loop-020`
- 模式：`optimization`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-21T03:16:08Z`

## 目标/假设

Low draft acceptance or proposer scheduling exposes a service-level cost larger than current measured run spread; saved host and device traces can isolate it before changing code.

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

Produce a reproducible per-step draft/target cost and accepted-token map with an independently testable, material E2E intervention, or pivot with evidence.

## 否定条件

Scopes overlap or attribution cannot separate draft from target; no safe intervention exceeds run noise.

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
