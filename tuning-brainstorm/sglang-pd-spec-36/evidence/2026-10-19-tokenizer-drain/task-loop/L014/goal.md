# Loop 目标：M001 drain候选的CPU配对计时与到达代价

- Loop ID：`L014-tokenizer-drain-cpu-timing`
- 模式：`optimization`
- 所属 Task：`sglang-pd-spec-36`
- 执行 Skill：`NONE`
- 冻结时间：`2026-10-18T19:46:50Z`

## 目标/假设

L013函数输出和生命周期通过且错开到达会拆批。保留cap32的drain在同时短批可减少尾等待；混合到达可能提前短请求完成但额外调用成本须实测；长批收益可能落入AA波动。

## 允许修改的路径

- `/Users/wio/work/sglang-learning-lab/mechanism-task-20261018`
- `/Users/wio/work/sglang-learning-lab/repo-export/tuning-brainstorm`

## 目标用例

- tokenizer-cpu-c2
- tokenizer-cpu-c3
- tokenizer-cpu-staggered

## 接受条件

实际原类/候选、相同tokenizer与fixture，逐case交织AAAB；clean wall/processCPU/逐请求latency和实际到达、cgroup记录；短收益超过AA且长/错开反证全部保留。仅局部CPU结论，不推广服务。

## 否定条件

输出不一致；改变到达/文本/窗口或过滤限流行制造收益；只报批完成而隐藏单请求影响；效应与AA重叠仍称提升；推成PD/NPU收益。

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
