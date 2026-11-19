# 原生0.32 D0：逻辑query、实际图重放和注意力输入形状

本节只描述epoch727同一次32768输入/1024输出/C16诊断的完整D0窗口，不把TP8复制计数相加当用户tokens。原始输入、环境与913个scheduler注释见018，完整设备覆盖见019。完整跨框架地图仍待SGLang和其余D rank验收。

[实际connection映射](../evidence/logical_device_map_G867/D0_first_actual_connection_map.json)遍历8081676条TASK，将1812次aclmdlRIExecuteAsync全部关联到实际设备ModelID。906次调用各自完整落在同线程的906个非零scheduler范围内，每个非零范围恰一条；另外906次位于这些范围外。它们的ModelID分布为26:81、27:58、28:404、29:320、30:43，原始连接映射保留了逐次调用顺序，后续配对见下节。该结果纠正“一个scheduler CPU范围包含target与draft两次图执行”的简单假定；CPU注释只包execute_model，不能涵盖范围外的全部工作。

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

后续[021](021-native032-allrank-coverage-and-DSPARK-source-chain.md)补齐D其余rank原始覆盖复核与实际DSPARK工厂、parallel backbone/Markov、sample_tokens/propose和图round-up源码链。原先“其余D待验收”是历史状态；范围外重放的逐步顺序与容量复核见下文，完整跨框架地图仍待同批 SGLang 采集。

## 同一 cohort 的 target 调度位置

G804 的 D 侧八 rank 使用同一份标准化 scheduler 序列。906 个非零步骤中，每请求均调度 6 个 target query。逐请求核对调度前缀与已生成位置，得到以下计数。 [原始调度注释](../evidence/profile_capture_G804/logical/decode/canonical_annotations.json) [逐批次位置账](../evidence/profile_capture_G804/derived/D0_target_position_ledger_G1447.json) [提取代码](../code/profile_capture_G804/audit_G804_decode_positions_g1447.py)

| 计数范围 | query 位置 |
|---|---:|
| 输出预算内首次覆盖 | 16,384 |
| 输出预算内重复覆盖 | 4,929 |
| 输出预算外的调度位置 | 137 |
| target 总调度位置 | 21,450 |
| D0 实际图输入额外 lane | 1,920 |

输出预算的绝对位置从 32,767 到 33,790，位置从零开始，末尾 prompt token 的 query 对应首个输出。全部 16 条请求覆盖了各自 1,024 个预算内位置。重复位置可能含不同 token，额外图 lane 描述实际输入形状。这两类计数均需结合设备事件判断时间影响。上限外调度占 0.64%，图形状额外 lane 占逻辑 query 的 8.95%。 [实际请求验收](../evidence/profile_capture_G804/parent_diagnostic_raw.json) [设备形状与映射](../evidence/logical_device_map_G867/D0_query_shape_map.json)

调度注释中的 17,875 个 proposal 槽由 17,795 个 cached 请求行槽和 80 个首行槽组成。Decode 服务计数增加 17,795 个 draft token、3,559 次 draft 和 12,864 个接受的 draft token。按已验证的请求批次和逻辑前缀长度保留未压缩因果 query–token 对数，供 SGLang target trace 归一化。当前尚未采得 SGLang 的同批 Decode 事件。 [服务计数器差值](../evidence/profile_capture_G804/actual_spec_counter_deltas.json)

## 范围外图的实际输入容量

G804 D0 有 906 次重放落在非零 target scheduler 标记之外。五类 ModelID 都有直接设备 connection 关联。每次重放各记录 3 个 SparseFlashMla 任务，每个图类的实际 query 首维分别为 5、10、20、25、35。每次重放只计一次输入容量，合计 18,570 个 query lane。范围外图标记下的命名计算任务累计耗时为 2.739 秒，范围内图为 27.494 秒，统计仍为同一物理 rank 的任务耗时之和。 [形状、计数及源摘要](../evidence/profile_capture_G804/derived/D0_outside_target_graph_lanes_G1454.json)

固定 DSPARK 源码的 anchor 路径每请求分配 5 个 draft query。先用总数约束推导出候选分桶，再用 G860 逐次连接记录独立核对。1,812 次 CANN API 调用按主机时钟严格交替，每个 target 重放后紧接一个已关联设备图的 draft 重放。按同一 scheduler 步的请求数核验，906 个 draft 图均使用能容纳对应 target 批次的最小已观测容量。若匹配后一轮 target，905 对中有 14 对的 draft 容量容不下它。源码中，`sample_tokens` 读取该轮 `execute_model` 保存的 `input_batch` 并传入 `propose`。 [逐次重放核对](../evidence/profile_capture_G804/derived/D0_ordered_target_draft_replays_G1461.json) [核对代码](../code/profile_capture_G804/audit_G804_D0_replay_pairing_g1461.py) [DSPARK 源码链](021-native032-allrank-coverage-and-DSPARK-source-chain.md)

3 请求的 96 步用容量 4，6 请求的 43 步用容量 7，两类合计多出 695 个图形状 lane，占 17,875 个逻辑 proposal 槽的 3.89%。范围外 draft 调用自身缺少 scheduler 步编号，配对依据是调用顺序、实际图容量和源码传参。API 时间差描述主机提交，掩码 lane 的设备执行成本和端到端收益仍待实测。 [原始分桶复算](../code/profile_capture_G804/audit_G804_D0_draft_lanes_g1454.py)
