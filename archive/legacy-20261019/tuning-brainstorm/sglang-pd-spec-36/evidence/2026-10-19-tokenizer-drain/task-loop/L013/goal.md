# Loop 目标：M001 drain当前队列候选的语义与到达反例

- Loop ID：`L013-tokenizer-drain-correctness`
- 模式：`correctness`
- 所属 Task：`sglang-pd-spec-36`
- 执行 Skill：`NONE`
- 冻结时间：`2026-10-18T19:40:56Z`

## 目标/假设

L011/L012支持短封闭批次存在尾等待成本。保持cap32，仅收集当前队列已有请求的候选可能避免未知N时的等待，但会拆分稍后到达的请求；先验证完整输出、顺序、取消和异常恢复并记录批次反例。

## 允许修改的路径

- `/Users/wio/work/sglang-learning-lab/mechanism-task-20261018`
- `/Users/wio/work/sglang-learning-lab/repo-export/tuning-brainstorm`

## 目标用例

- source-contract
- tokenizer-cpu-c2
- tokenizer-cpu-c3

## 接受条件

真实安装原类与仅覆盖收集循环的候选，完整tokenizer/模板输入不变，单请求、同时到达、错开到达、异kwargs、排队取消和异常恢复均逐请求比对完整结果；保留批次组成变化和未决成本，不作性能接受。

## 否定条件

丢失/错配结果、未来取消状态异常、队列残留或异常后不恢复；修改分词器/消费者制造等价；把不同批次构成隐藏或将CPU结果写成PD/NPU收益。

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
