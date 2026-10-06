# Performance Research Reset165 — native GLM-5.3 PD timeline diagnostic

身份：branch glm5-3-autonomous-20261001 / HEAD4120e1f84249498288eb3b85e36c6348a98bb315 / GLM-RESEARCH-RULES-v2 / rule commit93133edc6567a2ecaac7754b3c9aab1537ca7fe3 / GLM-OPT-0002。
产品与Current：GLM-5.3 W8A8标准P→KV→D/MTP动态完整服务，既有TTFT与TPOT SLA。Current=None；完整API、正式SLA、matched E2E尚无；active PERF_KEEP为空，见CURRENT_PERFORMANCE_STACK。
最大代码Gap：unknown。H3仅为条件候选：所有必要device-safe KV contributors完成后，观察/提交延迟是否错过合法D执行机会？没有已证明可节省时间或代码patch。
决定性证据：Run248 helper2334prompt、D32tokens与external hit delta2334证明一条PD；D10.72s调用不分解TTFT/TPOT。两端health200，各16workers；实际worker.profile/config/router源码已亲读、两端sha一致，当前profiler routes为空、profiler env为空。旧Run242 dynamic阻断API采集，Run243错误msprof调用，Run244虽然phase失败但已保留旧模型设备trace；均不能作为本次5.3设备因果证据。
代码假说与patch：无性能patch。一次同几何/模型/算子重启，仅增加profiler config、新engine identity与输出路径；内置CPU+NPU torch_npu Level1，stack/memory关闭，frontend开启，msmonitor daemon未启用。新机械client显式noProxy，原始SSE/错误/usage外部KV计数全保存。
最小diagnostic：warmoff8tokens、measureoff8tokens、profileon8tokens，各fresh salt相同2334prompt；全部P/D workers短窗口并显式stop。非正式SLA、非产品输出功能验收；chunk时刻不冒充逐tokenITL，跨机时钟不直接相减。trace停导出时间排除请求latency。
Hypothesis：唯一H3保持conditional，缺PD时间线，先判断能否区分必要P/transfer/target/MTP设备工作与CPU控制等待。
Distinguishing evidence：32rank非空设备trace与P/D请求raw/同client monotonic事件、实际外部KV全命中、frontend CPU关联；设备ready不足仍写unknown。
Decision table：trace覆盖全且显著等待在必要依赖后→亲读对应控制消费者，冻结一个代码候选；必要计算/传输主导→否定把该等待当控制可消除时间，park H3并更新最大Gap；覆盖/因果不足→只离线归约缺项，不以HTTP200/成功率宣称采集有效，不盲目重跑。
最小matched A/B：本次无代码A/B；profOFF/ON只审核观测开销，不计性能Gain。真正候选须correctness与同合同matched E2E后裁决。
Astra：已有ASTRA_FULL_PD_ENTRY_REVIEW已完成，当前短profile未新增高成本Runtime架构候选；无需重复Review。

本阶段没有新增代码级性能 KEEP。
