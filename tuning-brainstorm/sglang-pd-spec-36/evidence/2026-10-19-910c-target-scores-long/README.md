# 910C：目标分数诊断与原三路32K长用例

**原3×32768输入、每条256输出的PD＋MTP用例已连续两次完成功能校验。** 原模型、两台910C各16逻辑NPU、TP16/EP16、BF16 KV、EAGLE三步/topk1/draft4及graph均保留，没有缩短输入/输出。当前合计2048个输出token通过与返回目标最大分数的一致性检查；这不等于独立模型精度金标准、KV逐字节验证或完整生命周期证明。性能基线仍未接受，整体E0、accepted_baseline=null。

## 目标分数诊断

R084读取实际EAGLE公共验证、logprob处理与调度控制源码，并确认两端ready200。确定性推理开关实际为false，sampling_backend为pytorch。源码在token选择之后调用compute_spec_logprobs，支持返回所选token及top2分数；没有据此更改模型、采样、MTP、graph或确定性开关。

R085/R086保留两路512→128及PDv2输入数组顺序，只额外返回logprob/top2，均通过原功能检查。两次共512个选中token都等于各自返回的目标最大分数，全部输出IDs、MTP计数、完整top2行也逐行重复。允许报告分数并列，并不要求所选ID出现在任意tie-break后的前两名列表里。

前序分叉位置33/40在R085这个输出版本中的top2差距为0.375/0.625个自然对数单位；不能把它们笼统称为微小舍入噪声，也没有另一版本同位置的分数，尚不能证明根因。当前只有相对于实际目标分数的MTP选择一致性证据，没有将文本相似当精度通过。

回查各Run同一P主机时间范围内的日志，发现HTTP batch2不保证实际Prefill微批形状：

| Run | 实际P微批日志 | 输出关系 |
| --- | --- | --- |
| R078 | 一批2序列/1024 token | 首次执行，第三组输出版本 |
| R079、R083 | 两批1序列/512 token | 两次完整IDs/接受直方图相同 |
| R081、R082 | 一批2序列/1024 token | 两次完整IDs/接受直方图相同，区别于上一组 |
| R085、R086 | 两批1序列/512 token | top2观察版本，复现R079/R083输出 |

这是调度形状与输出分组的相关证据，不是已证明的因果关系。后续控制批次形状或数值质量时要记录真实微批，而非仅看HTTP数组顺序。观察logprob也可能改变调度与成本，因此本轮不与无logprob请求比较速度。

## 原长输入完整执行，没有改成短用例

R087将三条原32768-token输入同时以batch3到达P/D，独占客户端PID及源文件SHA均保存。其数值客户端在R089中保持完全相同字节重复；只增加父进程监督与实际退出回执。R088/R090只读采集精确PID、结果文件、模型日志和状态，没有发送新推理或重放任务。

两轮均满足：每条返回prompt IDs与原32768输入逐个相同，P边界token与D首token相同，D完整256个合法词表ID且以length结束，MTP verify/正确草稿/proposed/接受直方图守恒。两轮共1536个输出token均与各自返回目标最大分数一致，三条输出IDs及全部MTP计数/直方图在重复间完全相同。

| 观测 | 客户端完整批窗口 | 三条verify次数 | 正确草稿数 | 生命周期观察 |
| --- | ---: | --- | --- | --- |
| R087启动 / R088读取 | 58.932 s | 92 / 91 / 100 | 163 / 164 / 156 | 完整结果、stderr空、原PID消失；非父进程无法取得实际退出码，记UNKNOWN |
| R089启动 / R090读取 | 55.094 s | 92 / 91 / 100 | 163 / 164 / 156 | 父进程wait确认客户端退出0；父/子PID均消失、stderr空 |

每轮开始和结束的两端空闲cache flush均200，结束后两端pure ready200。源码flush_cache仅在is_fully_idle通过后重置cache与请求/KV pool；这仍不替代取消时的发送drain、延迟ack、页/引用所有权测试。没有把第二轮新增退出监督算作数值或性能优化。

两个批窗口由同一910C46客户端monotonic时钟计算，包含P/D HTTP、Prefill、KV交接、排队、MTP生成和额外分数观察。没有跨主机时间戳相减。两轮没有优化A/B，58.932→55.094秒不是加速结论；当前也未报告稳态TTFT/TPOT或p95/p99。

## 三路到达与真实容量边界

max_running_requests仍为3。P/D KV容量实际为43904/43776 token。D长请求日志显示一个活跃上下文占32896/33024 token，其余预分配等待数2→1→0；npu graph为True。P按原4096 chunk推进三条完整32K输入。**通过的是三路到达及正常准入/排队，没有声称三份完整上下文同时驻留。**

R090服务报告D的target_verify/draft_decode/draft_extend图区间内存分别3.188/0.500/0.594 GB，共4.282 GB，P为原角色默认无图。普通decode graph批次列表为空，不代表EAGLE验证/草稿图未执行。该内存字段也不能直接解释成可全部消除的图节点缓冲，需要分离lazy通信分配与真实图工作区。

下一步先验证真实自然EOS、取消、重试、延迟发送/页引用回收，再冻结适合数值路径的质量判据与稳态A/A。随后按实际内存与关键路径证据选择单个配置/代码候选；不降低正确性，不关闭PD/MTP/graph，不以短输入替代原长用例。

- [主控审查、top2逐行重复、微批相关性与长用例范围](notes/L039-top2-long-review.json)
- [R085原始短请求及全部分数](runs/L039-R085-pd-functional/remote-result.json)、[R086重复](runs/L039-R086-pd-functional/remote-result.json)
- [R088长用例公开视图](runs/L039-R088-long-state/public-view.json)、[R090重复与实际退出回执](runs/L039-R090-long-state/public-view.json)
- [视图来源与原始SHA](public-view-sources.json)：只替换重复回显的prompt ID数组及重复batched body，保留所有输出、逐token分数、计数、时间、日志和错误；原始结果仍在任务本地与自有910C路径。receipt中的SHA属于原始数据，不是公开视图。
- [实际源码与运行配置](runs/L039-R084-logprob-contract/stdout.json)、[冻结原长用例](specs/L039-R087-pd-plan.json)、[退出监督版本](specs/L039-R089-pd-plan.json)
- [38条冻结命令各一次的会话元数据审计](notes/L039-group-memory-zcode-audit.json)，不公开执行器完整模型I/O。
- [此前首次真实PD及重复输出反例](../2026-10-19-910c-first-pd/README.md)
