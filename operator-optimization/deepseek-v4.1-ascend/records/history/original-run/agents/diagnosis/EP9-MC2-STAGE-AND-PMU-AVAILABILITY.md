# EP9 选定长 MC2：阶段文件入口和精确计数

## 查找范围

华为 [CANN 商用版 8.5 通算流水图说明](https://www.hiascend.com/doc_center/source/zh/canncommercial/850/devaids/optool/atlasopdev_16_0139.html) 介绍 `msprof op` 的 MC2/LCCL 通算 trace；[工具说明](https://www.hiascend.com/doc_center/source/zh/canncommercial/850/devaids/optool/atlasopdev_16_00851.html) 举例按 `OPPROF_*` 目录和各设备输出 `trace.json` / `visualize_data.bin`。这是同类工具文档，不证明 9.2 历史采集实际启用，或当前自定义 `HcclShmem` kernel 在该工具下可完整显示阶段。

对 stage 容器既有 `diagnostic-20261029/runtime/diagnostic-traces` 的 16 rank 目录做文件名盘点，递归遍历 29,824 个文件、未遇到遍历错误，未找到名称匹配 `OPPROF_*` 的目录或名为 `trace.json`、`visualize_data.bin`、`dump_workspace.bin`、`time_stamp_core_*.csv` 的文件。EP9 自身子树亦包含在内。本检查只限定根目录和列出的精确文件名，**未读取原始 slice、未穷举所有可能文件名、其他任务目录或改名产物**；不能证明历史没有内部阶段数据，也不能将其他版本工具的目录作为保存格式契约。

EP9 读到的 host/device `sample.json` 中，`aiv_profiling_mode=task-based`、`aiv_metrics=PipeUtilization`、`pcSampling=off`。这些是配置记录，不能证明该采集之外不存在额外数据，或由关闭 PC 采样推断没有内核工作。就既有已定位文件而言，尚未找到与所选任务绑定的逐专家行数或内部分阶段事件。

## 具体任务绑定与嵌套行

EP9 `8c18c293…` 第 0 层选定的 `DispatchFFNCombine`，stream 47、task 46755，trace start/end 对应数据库 `globalTaskId=4709` 的 `1793241333127617287` / `1793241333401182441` ns，时长 **273565.154 µs**。选中的 CSV 哈希 `20af7cd5f9ca46fe2aeddc66a51cd30cea5048a374c87ba572f903bd1410cdf5` 与先前 JSONL 记录相同。

同一数据库中以 stream、task、connection 过滤还命中一条 `globalTaskId=73482`；它开始比 4709 早 1.060 µs，结束晚 0.820 µs，时间包围 4709。4709 有唯一 `COMPUTE_TASK_INFO`、28 条 PMU，73482 没有这两类记录。这只是两个 **DB 记录之间的嵌套关系**；不能说它们是两个独立物理任务、同一个内核的某两个阶段或等于设备忙碌时间，也不能因为 task ID 相同盲目选行。当前分析使用精确 trace 起点对应的 4709，而非将两个区间相加。此 DB 的本次只读 SHA 为 `057b8dcb70c4732d1589d696deffc47dc1ba9fda0bcd9be35b75ff0868d85829`，不是独立证明采集时文件未改变。

## 该任务多流水字段

globalTaskId 4709 的 28 条 PMU 在此数据库恰有一个时间戳，等于该 TASK 的结束时间；相关周期与同源 CSV 一致，比例字段与 CSV 的显示数值接近：

| 同一任务、同一导出来源 | DB ratio 原精度 | CSV 显示 |
| --- | ---: | ---: |
| AIC MAC | 0.001245414 | 0.001 |
| AIV vector | 0.000409476 | 0.0 |
| AIV MTE2 | 0.133447593 | 0.133 |
| AIV scalar | 0.015120285 | 0.015 |

以 [CANN 9.1 beta op_summary 字段说明](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910beta1/devaids/Profiling/atlasprofiling_16_0067.html) 作为**版本不同的条件性参考**，各字段对应不同指令流水的周期比例；本次 9.2 生产和导出口径尚未完全绑定。该记录不能仅因 CSV 的 vector 显示 0 就断言所有引擎都没有活动；同时 MTE2 的 13.34% 不是墙钟占比、HBM 实际字节、专家数、路由负载或内部等待时间，不能推断是搬运/轮询造成的根因。AIC 与 AIV 分母不同，不能相加。数据库和 CSV 是同一 profile 的导出，不是独立采样；28 条 PMU 关联在本 DB 很强，但尚未证明 exporter 对所有 replay 的绑定规则。长样本有 profiler 开销，不能归因线上服务。

## 复算

在 stage 容器运行 `python3 agents/diagnosis/query_ep9_mc2_profile_stage_audit.py` 输出 `ep9-mc2-profile-stage-audit.json`；脚本校验 CSV 固定 SHA、精确 task/connection/start 以及嵌套关系，SQL 以 `mode=ro` 打开，限定输出任务及所查 PMU、配置字段、文件名计数。不读取请求内容、权重或设备缓冲，不调用 NPU。EP9 前一任务间隔的独立有界查询见 `MC2-OFFICIAL-PROFILING-AND-WAIT-FIELD.md`。


## msOpProf 指令流水与实际工具帮助的边界（2026-10-29）

昇腾 [CANN 9.1 msOpProf 指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/devaids/optool/docs/zh/user_guide/msopprof_user_guide.md) 对上板 PipeTimeline 和 instrTimeLine 写明：只支持 Ascend 950，且暂不支持通算融合类算子；PcSampling 写明仅支持 950；TimelineDetail 的相应能力不支持通算融合类算子。其另有针对 MC2/LCCL 及满足特定条件的通算流水图，**不能把这张图等同于逐指令 PipeTimeline**。该文件可从搜索索引读取且另有 [Ascend 官方 msopprof 源码仓库指南](https://github.com/Ascend/msopprof/blob/80dae2e3701d14e191d2d461eb6be8aab714d89d/docs/en/user_guide/msopprof_user_guide.md) 作对照；该提交未绑定本次工具版本，不能自动推广成历史 9.2 的产品支持结论。

已存 169 stage 上的 `diagnostic-20261029/msprof-op-help-9.2.txt` 是当时 `msprof op --help` 输出，列有 TimelineDetail、Source 等项目，但其列表没有 PipeTimeline、instrTimeLine、PcSampling。help 列表并不证明其他工具或隐藏接口不存在，也不证明列出的 Source、TimelineDetail 可用于该自定义 MC2。官网要求通信流水图支持的算子类型及接口需匹配；本任务 `DispatchFFNCombine` 用到的自定义 shmem 路径、旧采集和工具实现是否满足条件尚未知。此前对该目录没找到给定名称的产物，不能据此断言未保存其他格式的相关原始记录。

因此目前不能把上述 950 指令流水、PcSampling 或通算源码热点图作为 910C 现有样本的已验证取证路径，也不能根据帮助里出现 Source 就认定能定位所选 MC2 内部 PC。若未来约束变化允许新采集，先核对实际已安装工具与编译信息、支持的通算类型、使用开销及用户授权；本任务当前不新启 NPU 或更改内核。现有任务级 PMU 不能区分标志轮询、有效 payload 搬运和其他活动。
