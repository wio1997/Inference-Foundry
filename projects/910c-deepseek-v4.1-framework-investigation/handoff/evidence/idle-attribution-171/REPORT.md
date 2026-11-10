# 171设备空档：主要发生在准备下一轮输入

**结论：补采的16卡稳态闭合循环中，约96%的设备空档与worker的“prepare input”阶段重叠。** 它主要包含输入和注意力参数准备，不能笼统称为“调度器在挑请求”。草稿阶段约2%，其余很少。原服务已经恢复health200，8-token小请求成功，冻结源码SHA不变。

本轮只补一个带框架原生阶段标签的steady窗口，沿用DP各4条镜像请求、每条32768输入/1024输出。参考和profile两轮的10项配平检查全部通过，8条输出SHA也与此前参考完全相同。参考15.792秒、518.75输出token/s；采集轮17.355秒。没有优化实现或新增性能收益。

## 空档期间Host在哪个阶段

为排除采样首尾不完整迭代，按每卡第一个MODEL_EXECUTE到最后一个MODEL_EXECUTE裁出11个完整循环。每卡设备窗口平均839.32ms，Free平均322.15ms，Free占设备时间平均38.38%。先逐卡计算占比，再取16卡平均；分项互斥、合计100%。原完整采样窗口也保留，prepare input仍占约94%的Free，结论一致。

| Host阶段 | 占全部空档 | 占设备窗口 | 具体工作 |
|---|---:|---:|---|
| 准备下一轮输入 | 96.21% | 36.91% | 更新请求状态、token位置和长度、缓存映射；协调batch并构造各缓存组的注意力metadata。 |
| 生成下一轮草稿 | 1.89% | 0.72% | 准备DSpark输入、context/slot/metadata，运行草稿并交付后续候选。 |
| 目标整图启动 | 1.77% | 0.70% | 将已准备的输入交给目标整图；Host范围与实际整图设备执行范围分开计量。 |
| 其余未归类 | 0.08% | 0.03% | 没有落入上述worker阶段的空档；不直接归为EngineCore调度。 |
| 模型读出与结果整理 | 0.04% | 0.01% | 整理目标输出、选择采样位置和读出所需数据。 |
| 校验与采样 | 0.02% | 0.01% | 校验草稿并确定本轮接受的token。 |

DP0/DP1的prepare input分别占空档95.99%/96.44%。这是阶段定位，不等于这些时间全部可省；prepare input内也可能有必要等待。与设备工作重叠的CPU阶段部分没有算进Free。

## 输入准备里，重点又在哪里

代表EP0/EP8各11轮，prepare input墙钟平均45.60/45.73ms，其中与Free相交28.80/28.08ms。按实际调用边界进一步分为：

| 代表卡每轮子窗口 | EP0墙钟 / Free | EP8墙钟 / Free | where次数 |
|---|---:|---:|---:|
| 输入状态、位置/缓存映射至DP协调结束 | 12.00 / 0.37ms | 12.85 / 0.39ms | 2 |
| 协调结束至首次原生metadata算子提交前 | 31.79 / 27.72ms | 30.98 / 26.93ms | 243 |
| metadata提交及入图前收尾 | 1.81 / 0.72ms | 1.90 / 0.76ms | 0 |

中间窗口是明确的调用边界，不冒充某个Python子函数的精确耗时。源码显示_build_attention_metadata遍历缓存组和Attention组；legacy CP builder调用_build_local_token_metadata构造本卡query区间、前缀长度、suffix、掩码和local seq_lens，再做pin/copy/where等交付。CPU输入会走torch fallback。243次where的时序与这些源码相符，但主线程没有完整Python栈，尚不能逐条声明都来自某一个函数。

## 值得推进的方向

先沿已有metadata-host / BatchFrame共同生产方向，核对同代、同语义的CP区间/长度/掩码能否只算一次，复用缓冲并合并交付。图外Host构造和后端提交应与设备kernel一起计价。现有slot mapping已有fused-groups路径，不能将它重新计为新融合。每个缓存组的block table、slot、压缩状态和末读/发布合同保留，旧CPU-plan、25次H2D等历史信用不重复相加。

本次确认热点位置，候选仍为“待确认”；没有直接改优化代码。草稿Host阶段和Gloo不作为当前这笔空档的首要大项，但保留各自局部开发价值。具体净省需对替代成本、重叠和精度验证后判断。

## 原样本时序交叉验证

旧16卡steady中约97%的空档在目标整图执行区间外；按Free为分母，75.61%在下一图外计算任务提交API开始前，4.39%在调用中，14.93%在调用返回后，3.10%图内，剩余1.97%未精确关联。这张提交账和新增阶段账分开记，不相加，也不混合两次样本绝对时间。它支持优先追Host向下一步交付输入，而不是将所有空档归为设备算子内部慢。

## 数据和复算

- [结构化结论](findings.json)、[原样本时序账](submission-summary.json)、[原16卡逐调用关联](idle-attribution.json)。
- [新增16卡阶段分析](scoped-capture/scope-analysis.json)、[代表卡细分](scoped-capture/prepare-boundary-analysis.json)、[恢复验证](scoped-capture/restored-smoke.json)。
- [冻结代码索引](source-map.json)、[追加只读源码](additional-source-map.json)，正文只承认实际审阅范围，不声明全部源码已穷尽。
- 原始trace留在scoped-capture/launch.json所列171自有目录。summarize.py离线复算本结论；其余脚本和采集回执完整保留。

EngineCore调度进程本身没有直接采集，不能给它编造独立耗时，也不能把未覆盖的0.08%当作它的总CPU时间上限。所有百分比均属设备短采样，不能直接按16秒整批请求放大。profiler有开销，最终收益回到无profile、同负载验证。
