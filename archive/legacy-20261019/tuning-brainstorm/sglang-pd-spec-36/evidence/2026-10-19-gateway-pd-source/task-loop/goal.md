# Loop 目标：M034 cache-aware routing 与 PD 策略实际实现

- Loop ID：`L033-gateway-cache-pd-contract`
- 模式：`correctness`
- 所属 Task：`sglang-pd-spec-36`
- 执行 Skill：`NONE`
- 冻结时间：`2026-10-19T05:02:48Z`

## 目标/假设

固定版本gateway在P/D独立策略、model/cache-key与worker生命周期约束下，应只选择有效候选并保持缓存估计/负载所有权一致；路由估计不等于真实worker命中。

## 允许修改的路径

- `/Users/wio/work/sglang-learning-lab/mechanism-task-20261018`
- `/Users/wio/work/sglang-learning-lab/repo-export/tuning-brainstorm`

## 目标用例

- source-contract
- lifecycle
- c2-short
- c3-long

## 接受条件

闭合Rust策略与PD路由源码，优先运行原crate的真实CPU组件/既有测试，清楚区分部署版本、真实拓扑及AscendPD+MTP门槛；不盲调阈值。

## 否定条件

策略跨模型/角色错误选择、移除或失败worker仍被使用、负载/缓存状态不一致，或实际依赖/组合不可运行；所有失败和前提保留。

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
