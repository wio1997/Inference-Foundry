# MC2 动态工作量、分阶段资料与 Wait 字段核对

## 结论

当前保存的 profiler 仍未提供可绑定所选 `DispatchFFNCombine` 调用的逐专家有效行数或内部 dispatch/GMM/combine 阶段时间，不能区分路由负载与内部等待。华为文档给出了同类接口和通算算子分析工具入口，但不能把另一个算子的 ABI、或测试版文档的功能支持，直接套到当前自定义 kernel 和历史 CANN 版本。

## 官方资料与版本边界

1. [CANN 9.1 aclnnMoeDistributeDispatchV3](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/aolapi/context/ops-transformer/aclnnMoeDistributeDispatchV3.md) 列 Atlas A3/A2 产品系列，提供 expert token 数输出及 `expertTokenNumsType` 选择前缀和或逐专家数量；其输出 dtype 为 INT64，参数里还有 EP 接收数和 Device workspace 大小。**它不是本任务的自定义融合算子**，不能认为其参数或值语义适用于当前的 INT32 `[24]` 输出；workspace 大小也不是专家负载。
2. 本任务冻结的 `dispatch_ffn_combine_kernel.hpp` 中，`expert_token_nums` 从累计数组输出，GMM 取相关累计数组计算 `currentM`，见 [已有源码与限制](LAYER0-MC2-DP-RECOVERY.md)。当前已存 shape 相容，但具体输出槽、历史二进制和每次调用数值未绑定；统计值仍未知。官网 A3 系列支持声明不等于已证明该自定义 kernel、所选 shape 或历史环境适用此官方 V3。
3. [CANN 商用版 8.5 通算流水图原文](https://www.hiascend.com/doc_center/source/zh/canncommercial/850/devaids/optool/atlasopdev_16_0139.html) 描述 `msprof op` 对 MC2/LCCL 的通算流水，并列出 Atlas A3 的部分 HCCL 接口；这不证明该工具能解析本任务自定义 `HcclShmem` 内核，也不证明历史 profile 已保存内部阶段。冻结源码目录中有 `HcclShmem` 和 shmem 路径，限定搜索未见 `PrintTimeStamp`；不能因此排除依赖树中的打点。
4. [CANN 社区版 9.0 PrintTimeStamp 原文](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_00002.html) 列 A3 支持及内核时间戳接口，并注明需上板、开启会影响性能，且该文档版本不支持算子入图场景打印。此为可能的后续取证接口，不证明 9.2、910C 自定义 shmem kernel 或本次图执行路径适用，也不表明可以对历史 profile 补出阶段。当前禁止新 NPU 工作，不修改目标算子。此前引用的 9.2 beta 链接是 Debugger 安装页面，检索摘要与目标页面不对应；已撤回其功能主张。此前对 9.1 PipeTimeline、instrTimeLine 及 replay 的限制同样未找到能直接核验的对应原文，因此撤回，不以搜索摘要为依据。
5. [CANN 9.1 op_summary](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910beta1/devaids/Profiling/atlasprofiling_16_0067.html) 对 `Task Wait Time(us)` 的解释是前一任务结束到当前任务开始的间隔；[kernel_details 文档](https://www.hiascend.com/doc_center/source/zh/canncommercial/81RC1/devaids/devtools/profiling/CANN%E5%95%86%E7%94%A8%E7%89%88%208.1.RC1%20%E6%80%A7%E8%83%BD%E8%B0%83%E4%BC%98%E5%B7%A5%E5%85%B7%E7%94%A8%E6%88%B7%E6%8C%87%E5%8D%97%2001.pdf) 对 `Wait Time(us)` 的说明更笼统。已核对当前 torch_npu 的 kernel_details 经 op_summary 选择/映射输出，但**历史 CANN 9.2 上游的同版字段生成契约未绑定**；不能仅凭名称将该字段视作 kernel 内同步等待耗时。

## 一条历史长任务的外部间隔对照

只读 EP9 `kernel_details.csv`，核对原文件 SHA-256 `20af7cd5f9ca46fe2aeddc66a51cd30cea5048a374c87ba572f903bd1410cdf5` 与选中 JSONL 的源文件指纹相同；以 stream 47、task 46755、开始时间及名称唯一关联 `8c18c293…` 第 0 层选定 `DispatchFFNCombine`。该任务 trace 记录 273565.15425 µs、CSV 显示 273565.154 µs；CSV 的 `Wait Time(us)` 为 **1.111 µs**。

在该文件中按同一 stream 设备开始时间排序，前一条已导出行是 task 46754 的 `MoeGatingTopKHash`；用这行结束时间和 MC2 开始时间计算的间隔为 **1.080 µs**，与字段数值相差 **0.031 µs**。这是同一数据源中两项记录**近似一致**，不是独立测量，不能证明字段算法、选中前序是历史全部活动中的唯一前序、舍入机制或漏采范围，也不能将内部总耗时、内部等待、专家负载、线上性能或可隐藏时长归因于此间隔。

仅同卡同一 CSV 时间坐标比较；未做跨卡时间差，也未绑定请求或专家数值。即使官方间隔口径在本例成立，入核前小间隔也不排除核内部等待或 profiler 扰动。

## 复算与下一步

脚本 `query_ep9_mc2_predecessor_gap.py` 在既有 stage 容器只读 CSV，输出 `ep9-mc2-predecessor-gap.json`；固定校验输入哈希、任务身份、设备记录时长以及前序行的名称/任务 ID。本机不读取请求或模型内容。现有已检查的 EP0/EP8 指定目录文件名和样本 schema 无可绑定专家计数入口，范围及限制见 `LAYER0-MC2-DP-RECOVERY.md`。不能由这次结果断言所有原始 slice 中都没有阶段数据；下一步只在找到带来源索引和该版本格式契约的现存阶段轨迹或逐调用匿名 expert 计数时做定向关联。


**后续核实（2026-10-29）**：第 4 项当时因页面原文不可达而撤回的 9.1 工具限制，现已通过 [Ascend msOpProf 固定提交指南](https://github.com/Ascend/msopprof/blob/80dae2e3701d14e191d2d461eb6be8aab714d89d/docs/en/user_guide/msopprof_user_guide.md)原文直接复核：PipeTimeline/instrTimeLine 仅 950 且不支持通算融合、PcSampling 仅 950，TimelineDetail 不支持通算融合；详见 [后续工具核查](EP9-MC2-STAGE-AND-PMU-AVAILABILITY.md)。该源码仓库提交与本任务历史 9.2 版本未绑定，旧撤回理由不再适用于上述固定文档，仍不能据此断言本机全部接口能力。
