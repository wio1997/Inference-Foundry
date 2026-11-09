# 共享专家4AIC/8AIV原语准入（整体共驻仍待确认）

**当前以a3为服务实现门槛**：[服务同路径/库/实际对象打开、八项字节0及发核证据](shared-4cube-service-gate-summary.json)。SMLA调查发现服务custom库覆盖激活且与builtin对象不同，a2现仅支持其独立进程实现，[资格补充](data/native-shared-4cube-a2-qualification.json)保留原结果；不能按同shape/dtype直接外推服务实现。a3绑定服务custom优先路径，核库SHA并实际打开自带激活对象`d1e2aee…`，八项字节比较仍0，Block Num8/4/8/4、DownMix4。a3在进程专用文件打开观察下运行，事件时长不与旧a2或服务比较；完整DFC共驻仍待确认。

以下表格和558.24µs是a2原时点的独立进程记录，不能作为服务实际4/8成本。

## 介绍

冻结DFC固定20AIC/40AIV；24/48硬件参考下，完整共享专家流水可尝试使用4/8资源并发。原C8诊断cohort全shared隐藏的条件信用最高2.248%，因此先做决定去留的格式、逐位和实际发核门槛。该信用仍是原调度参考，并非生产收益。

## 到了哪里

仅在171原容器的自有RW映射子目录执行设备0、M416、seed20261109 BF16输入，读取真实第0层共享专家w1/w3/w2和scale，逐份校验SHA。Up/Down使用加载后的BF16 scale；激活的FP32 scale由该BF16转回。原权重先按w1/w3拼接并转置；参考NZ只在未限核时执行；候选ND阶段明确关闭private format。

六个原语输出分别以相同输入逐字节比较：DQ值/scale、INT32 Up、激活INT8/scale和BF16 Down。另比较NZ默认Up/Down与ND限核输出，避免上游变化污染归因。

| 项目 | 原语字节差异 | 实际profile发核记录 | 一次profile kernel时间 |
|---|---:|---|---:|
| DQ值/scale | 0/0 | Block Num8, Mix0 | 31.90µs |
| Up INT32 | 0 | Block Num4, Mix0 | 257.94µs |
| 激活值/scale | 0/0 | Block Num8, Mix0 | 65.40µs |
| Down BF16，同输入 | 0 | Block Num4, Mix4 | 166.06µs |

NZ默认到ND限核的Up和Down也均0字节差异，所有检查finite。不能把get_stream_limit返回值当成发核证据；这里有[原始kernel_details](data/native-shared-4cube-a2-evidence.json)。Block/Mix分别描述主/从加速器，profile中Accelerator Core为N/A，保留原字段，不虚构具体物理核编号或独占分配。[官方字段说明](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/devaids/Profiling/atlasprofiling_16_1149.html)。

2次warmup后7组交替NPU-event：ND默认流水中位490.98µs，ND4/8为558.24µs。一次profile kernel总和521.30µs，profile首尾739.25µs；三种口径不同，不能混减。原M416 DFC中位1191.87µs只提供窗口参考，未测同时执行，不把差额称净省。[验证与摘要](shared-4cube-gate-summary.json) · [复算脚本](../../tools/summarize_shared_4cube_gate.py)。

## 结果

**已确认可实现，仅限a3本M416服务实现原语门槛**：正确scale签名下ND/NZ转换与4/8执行均有逐位和发核记录。本次有限输入通过不代表40层/全部输入逐位证明；本轮loop-011仅接受该范围。

**待确认，完整共享支与DFC共驻**：实际同驻和DFC受扰、HBM/L2争用、所有相关M及40层权重的逐原语验证、private-format设置对其它原语的影响、图捕获/流事件和跨rank请求传播。保持此高值开发候选，按用户全链路优先要求继续其它未扫路径；不为无新信息的单独Up路线补测。

首次a1误将checkpoint FP32 scale直接用于GEMM，和原BF16签名不匹配，已invalid；[纠正](data/native-shared-4cube-a1-correction.json)与原件保留。a2为修正后的独立Run。loop-007整体INCONCLUSIVE，保留a2限定事实，不覆写a1。两次均无模型服务请求、无DFC并发，health前后200；远端新增均在`/data/tiankuan/wio`自有子目录，没有修改加载服务、源代码、权重或全局设置。
