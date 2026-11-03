# Loop 目标：MC2 重放与计数器关联

- Loop ID：`loop-003`
- 模式：`design`
- 所属 Task：`layer0-diagnosis`
- 执行 Skill：`NONE`
- 冻结时间：`2026-10-28T11:10:18Z`

## 目标/假设

profiler 数据库可唯一连接本轮图重放、PMU 和通信事件，从而判明样本归属

## 允许修改的路径

- `/Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/agents/diagnosis`

## 目标用例

- frozen-169

## 接受条件

得到经验证的唯一连接关系或明确说明重复与缺失条件

## 否定条件

连接键重复且没有替代标识时保持 UNKNOWN

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
