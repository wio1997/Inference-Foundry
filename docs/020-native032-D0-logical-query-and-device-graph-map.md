# 原生0.32 D0：逻辑query、实际图重放和注意力输入形状

本节只描述epoch727同一次32768输入/1024输出/C16诊断的完整D0窗口，不把TP8复制计数相加当用户tokens。原始输入、环境与913个scheduler注释见018，完整设备覆盖见019。完整跨框架地图仍待SGLang和其余D rank验收。

[实际connection映射](../evidence/logical_device_map_G867/D0_first_actual_connection_map.json)遍历8081676条TASK，将1812次aclmdlRIExecuteAsync全部关联到实际设备ModelID。906次调用各自完整落在同线程的906个非零scheduler范围内，每个非零范围恰一条；另外906次位于这些范围外。它们的ModelID分布为26:81、27:58、28:404、29:320、30:43，暂不以最近时间强行指定逻辑step。该结果纠正“一个scheduler CPU范围包含target与draft两次图执行”的简单假定；CPU注释只包execute_model，不能涵盖范围外的全部工作。

[query与设备形状检查](../evidence/logical_device_map_G867/D0_query_shape_map.json)按CANN connection对应图家族，再核对同家族真实SparseFlashMla输入的首维：

| scheduler query | 逻辑请求数 | ModelID | 实际重放次数 | SparseFlashMla query首维 |
|---:|---:|---:|---:|---:|
|6|1|37|81|6|
|12|2|38|58|12|
|18|3|39|96|18|
|24|4|40|308|24|
|30|5|41|320|36|
|36|6|41|43|36|

三种不同压缩KV形状均具有同一家族query首维；第一次检查误以为每个ModelID只有一种完整输入shape，断言失败并保留[失败原文](../evidence/logical_device_map_G867/initial_shape_assumption_failure.json)。修正后的检查要求所有形状变体的query首维一致，没有改变实际数据或query判据。

906次范围内重放的逻辑query合计21450，对应SparseFlashMla输入首维按重放加总为23370；差额1920全部来自5请求/30 query使用36首维的图，比逻辑query多8.951%。它是输入lane计数，不等于多8.951%的端到端耗时或可消除FLOPs，kernel可能mask padding，分块/算子tile也会影响收益。不能把该比例称为已经取得的优化收益。

这给出一个有直接证据的图bucket候选，但尚需固定源码解释图family、target/draft边界和捕获几何，并与SGLang匹配实际逻辑工作量。诊断中实际微批次最多6请求，客户端C16不能替代内部batch。后续优化仍需正确性和匹配端到端A/B，当前未部署新bucket，未重发NPU采集，未修改模型计算。
