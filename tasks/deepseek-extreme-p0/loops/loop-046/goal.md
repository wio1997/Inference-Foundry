# Loop 目标：Reduce genuine prefill Host submission cost

- Loop ID：`loop-046`
- 模式：`optimization`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-25T06:02:29Z`

## 目标/假设

For frozen DeepSeek V4 W4A8 DP1xTP8 warmed first-token prefill, exact-state capture or another product-specific execution path can eliminate substantial Host submission pacing while preserving complete prefill state and first-token semantics.

## 允许修改的路径

- `scripts`
- `runtime`
- `bootstrap`

## 目标用例

- mixed_32k_1024_c12

## 接受条件

Eight-rank same-state A/B/A-prime correctness including all prefill writes and collective order, followed by lower unprofiled latest-rank prefill completion including capture refresh/amortization, then same-contract formal E2E gain.

## 否定条件

Capture is blocked by unsupported prefill control/collectives or mutable state, fails exact semantics, has sparse shape reuse, or does not improve latest-rank completion/E2E after amortization.

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
