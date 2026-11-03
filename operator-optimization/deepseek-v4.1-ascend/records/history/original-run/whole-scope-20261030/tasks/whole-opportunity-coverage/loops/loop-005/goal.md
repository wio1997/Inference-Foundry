# Loop 目标：171共享算子证据与profile入口对齐

- Loop ID：`loop-005`
- 模式：`spec`
- 所属 Task：`whole-opportunity-coverage`
- 执行 Skill：`NONE`
- 冻结时间：`2026-11-02T13:55:34Z`

## 目标/假设

复用实际内置profile字段并定点补offset/extent/epochs/values/native桥和读者，可减少无关逐对取证；未补齐证据不判独立或整体完成。

## 允许修改的路径

- `/Users/wio/work/166-vllm-analysis-continuation-20261030`

## 目标用例

- c8-input32k-output1k-prefix90
- c8-input32k-output1k-prefix90-166
- c8-input32k-output1k-prefix90-171
- bringup-one-chat-171
- provider-pad20-view-offset1-171

## 接受条件

共享证据路由可审查，明确每个字段的实际producer/作用域；171全Case采集入口和缺口有具体预算/回滚，原全部未闭合范围保留。

## 否定条件

字段语义或task/capture桥不成立，或重复logger/逐对工作无法回答结论变化；保留未知并转向有证据的入口。

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
