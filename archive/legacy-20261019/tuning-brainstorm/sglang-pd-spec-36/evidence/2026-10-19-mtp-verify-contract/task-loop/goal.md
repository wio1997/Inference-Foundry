# Loop 目标：M031 Ascend PD与EAGLE验证/接受合同

- Loop ID：`L015-pd-mtp-verify-contract`
- 模式：`spec`
- 所属 Task：`sglang-pd-spec-36`
- 执行 Skill：`NONE`
- 冻结时间：`2026-10-18T19:54:17Z`

## 目标/假设

steps3/topk1/draft4只是配置观察；实际Ascend PD+EAGLE需要从角色分派到verify kernel、accepted含bonus及draft_extend重建形成一致合同，不能由图表或单引擎参数推断支持。

## 允许修改的路径

- `/Users/wio/work/sglang-learning-lab/mechanism-task-20261018`
- `/Users/wio/work/sglang-learning-lab/repo-export/tuning-brainstorm`

## 目标用例

- source-contract

## 接受条件

固定安装commit，收集并追踪实际P/D角色、verify/accept实现和kernel分派、含bonus长度与slot/retrieve消费者；区分源码、CPU最小oracle和原生NPU未决；给出任务PD+MTP运行的具体最小验证。

## 否定条件

把观察单引擎当任务PD；把配置steps/draft当实际接受分布；把CPU桩/抽取当native验证；关闭MTP或更换模型回避支持性。

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
