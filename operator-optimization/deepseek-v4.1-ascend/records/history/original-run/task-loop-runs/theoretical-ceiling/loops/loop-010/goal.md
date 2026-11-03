# Loop 目标：draft 长度与输出计数的可证明域

- Loop ID：`loop-010`
- 模式：`design`
- 所属 Task：`theoretical-ceiling`
- 执行 Skill：`NONE`
- 冻结时间：`2026-10-28T11:57:14Z`

## 目标/假设

固定配置足以证明所有进入 metadata 的 draft 不超过5

## 允许修改的路径

- `/Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/agents/bounds`

## 目标用例

- frozen-169

## 接受条件

逐入口证明上限和对应计数或记录例外

## 否定条件

动态长度或其他入口未被约束时为 UNKNOWN

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
