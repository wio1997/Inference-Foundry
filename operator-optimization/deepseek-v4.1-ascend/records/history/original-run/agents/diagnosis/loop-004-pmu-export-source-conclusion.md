# Loop-004：PMU 和任务时间戳的导出语义

ZCode 在 169 的 CPU stage 容器只读检索了已安装 profiler 的两个源码根目录，共 **849 个 `.py`／`.sql` 文件**，未报告跳过的文件。检索词为 `TASK_PMU_INFO`、`TaskPmuInfo`、`task_pmu_info` 或 `globalTaskId`；每个命中文件的路径、行号和 SHA-256 保存在 `pmu-export-source-locations.json`，检索脚本是 `locate_pmu_export_source.py`。该范围不包含编译后的库，也不保证覆盖别名或其他拼写。

仅两处命中，都在 torch_npu 的 profiler 消费逻辑中：`_step_info_db_parser.py:65` 从 TASK 读取时间、connectionId 和 globalTaskId；`_trace_step_time_db_parser.py:150` 用 globalTaskId 关联计算任务信息和 TASK。**这两处都不解释 PMU 时间戳如何产生、不定义 PMU 到具体重放的唯一键，也不定义 20 ns 等差值的允许误差。**计算任务元信息的关联方式不能自动推广为逐次 PMU 的关联规则。

因此，Loop-003 中 7 个 PMU 时间戳与 7 条 TASK 结束时刻近似一一对应，是本样本强支持的候选映射；目标调用的 PMU 比率与导出 CSV 四舍五入值一致是额外佐证。它们**仍不是已由源码证明的唯一关联**。每次按时间最近匹配及其容差不应升级为普遍规则，尤其不能以 globalTaskId 单键直接做多次重放的逐行连接。

如果后续分析需要把 PMU 与单次重放严格关联，就需要取得该版本 CANN 导出器或权威格式文档对 `TASK_PMU_INFO.timestampNs`、`TASK.endNs`、重放键和时钟精度的定义。若这些信息不公开或实现位于二进制，保留未知；必要时可比较其他 rank、多个任务的现有导出，检查候选规律是否稳定，但只能增强经验支持，不能单凭重复观测证明规则。当前样本已经足以说明为何不能从融合算子的总耗时直接拆分内部计算、通信和等待，继续扩大只读搜索并不保证能得出阶段耗时；应并行推进其他模块和时间线归因。
