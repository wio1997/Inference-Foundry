# EP8 L14 窗口：长 AI_CPU 条目的主机来源

在同一已保存 EP8 采集中，以 device 8 / PID 737 / stream 92 / task 3999 / 精确开始时间唯一选中既有 `reduce_scatterAicpuKernel`，其 `TASK.connectionId=98566`。表 `CONNECTION_IDS` 将它关联到 `PYTORCH_API.connectionId=175592` 的唯一 `HcclReduceScatter`，同时关联另一个 CANN `launch` 98554；两个 launch 都是该同一关联下的条目，不能据此当成独立实验或证明它们是同一物理通信事务。

按同一 host 线程内区间包含关系，该调用位于 `HcclReduceScatterBase`、`c10d::_reduce_scatter_base_`、`vllm::reduce_scatter`、`vllm::ascend_sp_reduce_scatter_impl` 内，并位于 `wio_diag:DeepseekV41DecoderLayer.forward:737-29` 内。**737-29 是调用标识，不是层号。** 同次 sidecar 恰有该 ID 的 begin/end，begin 记录 `layer_index=23`，因此可说该调用提交于已标记的第 23 层 host scope。sidecar 本项没有 main/draft 模型标识；不能由层号或 scope 独立证明模型实例、输入张量、请求、与第 14 层 MC2 的数据依赖。

`COMMUNICATION_OP` 以 **98554** 连接到 `hcom_reduceScatter__097_209_1`，记录 `group_name_5`、`rankSize=8`；另一个连接 **98566** 命中 AI_CPU TASK。本库没有以 host ID 175592 或 98566 精确匹配的 `COMMUNICATION_OP` 条目。两个记录被同一 host 关联到，不自动证明相同物理事务、通信组成员、数据或因果；group 名称本身也不能确定 rank 成员。

以第 14 层 MC2 的 DB 起点为 0：

| 导出记录 | 开始 | 结束 | 相对 MC2 结束 |
| --- | ---: | ---: | --- |
| MC2 | 0 ms | 160.365956 ms | — |
| AI_CPU TASK | 65.134919 ms | 161.236859 ms | 结束晚 0.870903 ms |
| COMMUNICATION_OP | 387.829312 ms | 388.313282 ms | 开始晚 227.463356 ms |

这是同库导出的相对坐标，跨轨时钟误差尚未独立校准；尤其不能把 COMMUNICATION_OP 时长当纯链路传输或把其与 AI_CPU 的间隔当等待时间。各轨物理事件覆盖范围及完整性也未经证明。对 95.231037 ms 的窗口交叠，只能确认长 AI_CPU TASK 与 MC2 同窗；不能因此断定 MC2 因该 reduce-scatter 变慢、被其阻塞或可以隐藏这些时间。通信大算子晚于 MC2 的坐标也不能反证其它通信阶段与其交叠。

[昇腾 910 beta1 的 DB 格式说明](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910beta1/devaids/Profiling/atlasprofiling_16_1144.html)将 `COMMUNICATION_OP.connectionId` 用于 host-device 连线、`opId` 用于连接通信小算子，并把 start/end 定义为通信大算子起止。此说明支持字段解释，但版本与当前导出生成链尚未绑定，不能视为本采集的完整语义保证。

复现 `python3 agents/diagnosis/run_ep8_l14_reduce_scatter_source.py`。只读 DB 和 sidecar，经 stage 容器身份与挂载检查，校验两个输入 SHA；查询结果 `ep8-l14-reduce-scatter-source.json` 含精确连接、包含关系、相对坐标。结果不包含原始请求或权重。仍缺同一通信域 rank 成员、记录的事件对象 ID、张量及请求身份、时钟误差和 profiler 完整性；这些缺口应由现存映射/记录进一步验证，不能用名称和时间位置填补。
