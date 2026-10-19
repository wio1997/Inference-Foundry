# Loop 目标：M035 CPU direct native columns without tuple transposition

- Loop ID：`L006-direct-native-columns-cpu`
- 模式：`optimization`
- 所属 Task：`sglang-pd-spec-36`
- 执行 Skill：`NONE`
- 冻结时间：`2026-10-18T17:32:27Z`

## 目标/假设

For the scoped pp=1 non-hybrid/custom-pool=false sender, producing three native argument lists directly avoids redundant tuple construction and zip/list transposition while preserving exact native-call inputs and return/error behavior.

## 允许修改的路径

- `/Users/wio/work/sglang-learning-lab/mechanism-task-20261018`

## 目标用例

- descriptor-cpu-pp1

## 接受条件

Use a fresh range-sender A/A baseline and paired interleaved range/direct-column A/B. All native list values, order, calls and error returns match the protected CPU oracle. Record clean wall/thread CPU and GC. Fallback for untested PP/hybrid/custom-pool is explicit; no NPU/PD claim.

## 否定条件

Any argument/order/return/error mismatch fails. Report small-case regressions and inconclusive differences. No unconditional production rollout; original noisy evidence retained.

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
