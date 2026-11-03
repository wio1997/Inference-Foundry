# 选定 Indexer → 索引整理 → attention：事件与张量身份核查

复用 `f08-attention-intervals.json` 中 EP3、EP8 两个已选 scope 集合：每卡两组 scope，每组层 2、8、14、20、24、28、32、36 各一条，共 **2 卡 × 2 组 × 8 层 = 32 对**。组的 F08 scatter 原逻辑行数分别为 EP3 的 4760、8180 和 EP8 的 8180、5144；这些行数并非下文 Indexer tensor 的有效行数。每对依据预存的 Indexer 和 attention 设备起点精确连接同卡 SQLite `TASK`，两端各命中一行且同 stream；在两端之间仅取同 stream、完整包含于间隔的 TASK，按 globalTaskId 查询 `COMPUTE_TASK_INFO` 和经 `STRING_IDS` 解码的 shape/type。

32/32 对各有一条 `_prepare_indexer_indices_kernel`，并按设备导出时间处于 Indexer 结束之后、attention 开始之前。同一窗口中，Indexer 首输出形状 `[T,1,512]`、prepare 首输入与输出 `[T,512]`、attention 某一输入 `[T,1,512]`，类型均为 INT32；32/32 在去掉 singleton 维后形状兼容。T 分别为 595（8 窗）、643（8 窗）、1023（16 窗），表示导出的物理 shape，不能当作有效 token 数。各设备任务之间按同一 scope、stream 和导出时间关联；相同 shape/dtype 并不证明 storage、地址或数据相同，也不能确认 attention 该输入的实参语义和对应源码调用。冻结源码中的整理/返回逻辑是独立的条件性支持，不由这组时序单独证实。

4/32 窗还包含一条 TensorMove；它们是两卡、两组的第 20 层，且按导出开始时间位于 prepare 之后。该记录名称与数据类型不能证明其是自复制、属于目标张量或可删除；待验证实际 alias 和调用对象。间隔内的任务未必穷尽部分跨越边界的任务，也未覆盖其他流和未导出设备工作。两端 CANN 同连接 API 名均为 `launch`，不提供可用的 tensor 身份。

同步事件检查范围为所有 stream 中 **startNs 位于 Indexer 结束至 attention 开始半开区间**的 `TASK`，按 taskType 解码：EP3 两窗分别有 `NOTIFY RECORD` 1 条 + `NOTIFY WAIT` 2 条，或 `EVENT WAIT` 1 条；其他 30 窗无匹配此名称的导出条目。查询不是事件对象配对，不证明这几条涉及 Indexer、prepare 或 attention；没有匹配也不证明无事件、无设备内部同步或无跨边界依赖。已查 TASK、COMPUTE_TASK_INFO、CANN_API、PYTORCH_API、CONNECTION_IDS 表字段不包含显式 tensor storage/address 或 event 对象标识，不能从这些表字段完成所需身份绑定；并不排除其他未查的采集资源或应用侧日志包含相应信息。

源 JSON SHA-256 `355cd1a831403c1129ab76e128c0803da39b05712c3304c5c285aa82e6744e6c`；EP3 DB `084b8718888e33271435ec79d7573c95ca1615f3e7484c807ea24590fad234c4`，EP8 DB `e07ee2428be27c6129afd6dd0be8b700830048f30f13411b701e97c5d88e2486`；结果 [indexer-attention-event-identity.json](indexer-attention-event-identity.json) SHA-256 `8377661b150c2024a62622b5855c36c057c119d32493a8996a8f49c409d2badc`；只读脚本 [query_indexer_attention_event_identity.py](query_indexer_attention_event_identity.py) SHA-256 `5b5599d6a1510f9d033b848163ff02da12640f837cb20ca8a12a858f1406ff51`。仅对现有数据库执行 CPU 查询，未读取完整 trace、未启动 NPU、未修改目标实现。下一步只有取得逐调用输入/输出的可比较 storage 标识及其生命周期、event 对象记录和等待的绑定，才能把相容的形状/时序提升为运行时数据或同步依赖；此处不做融合收益预测。

## Selected preparation interval in the profiled gap

Recompute with `python3 agents/diagnosis/summarize_index_preparation_gap.py`. For the same 32 pairs, the prepare task durations have min/median/max 146.060 / 182.9285 / 222.656 us; the corresponding Indexer-end to attention-start gaps have min/median/max 150.440 / 219.866 / 1352.840 us. The median of the 32 per-pair ratios is 98.06%, and 27/32 pairs exceed 90%. Medians need not refer to the same pair and cannot be divided to obtain the per-pair ratio.

These are selected task intervals from intrusive profiler samples, not unprofiled duration, device utilization, pure computation, critical-path attribution or achievable savings. Other streams can work concurrently, tensor identity remains unknown, and the gap cannot be called device idle. Input SHA-256 is checked in the script; the reproducible aggregate `index-preparation-gap-summary.json` has SHA-256 `4886b05e48d8419303f0efa0049a80b169bc2c3a2119e507994eb54463a499be`. No NPU work was launched.
