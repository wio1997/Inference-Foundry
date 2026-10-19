# Loop 目标：M016 单步 EAGLE3 hidden relay 必要性与调用前提

- Loop ID：`L029-one-step-hidden-relay`
- 模式：`correctness`
- 所属 Task：`sglang-pd-spec-36`
- 执行 Skill：`NONE`
- 冻结时间：`2026-10-19T02:52:05Z`

## 目标/假设

For qualified one-step/topk1 EAGLE3, relayed previous-draft hidden may not affect outputs, but prepare/graph/adaptive/PP/PD consumers and shape requirements must be ruled out before omission.

## 允许修改的路径

- `/Users/wio/work/sglang-learning-lab/mechanism-task-20261018`

## 目标用例

- source-contract
- c2-short
- c3-long

## 接受条件

Canonical caller/consumer audit and exact CPU source-method hidden-vs-None output comparison with explicit model/kernel boundaries; identify eligibility and counterexamples without native acceptance.

## 否定条件

Any reachable consumer/shape requirement, adaptive transition needing prior hidden or changed tokens/topology invalidates broad one-step omission; native untested stays unaccepted.

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
