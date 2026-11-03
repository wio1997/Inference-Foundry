# 索引整理任务的采集计数字段

对 `INDEXER-ATTENTION-EVENT-IDENTITY.md` 中 EP3、EP8 各 16 个已选索引整理任务，按已保存的 `globalTaskId`、stream 和 task ID 唯一连接同卡数据库中的 TASK 与 TASK_PMU_INFO。数据库 SHA 与来源摘要中的值一致；32 个任务各有 28 条不同名称的 PMU 字段，且各自只有一个 PMU 时间戳。用同一 stream/task ID 唯一匹配同次 `kernel_details.csv`，AIV 周期数全部一致，vector、scalar、MTE2、MTE3 比例与 CSV 的显示值在 0.000501 内一致。

对原始数据库字段分别统计，EP3 的 `aiv_scalar_ratio` 最小/中位/最大约 **86.28% / 87.49% / 88.20%**，`aiv_vec_ratio` 为 **10.41% / 10.56% / 10.66%**；EP8 的对应 scalar 为 **85.54% / 87.21% / 88.11%**，vector 为 **10.34% / 10.46% / 10.63%**。这是 32 条所选任务的导出字段范围，并不是全模块或全设备利用率。每个采样点的记录时间比对应 TASK 结束时间早 120–201 ns；这不证明采样点是独立阶段的时间线，也不证明周期值的取样方式。

这些字段的分母、跨核聚合、计数覆盖与本次历史导出路径尚未完全证实，不应把比例相加、相减或等同墙钟占比。特别是 scalar 字段较高**不证明 scalar 计算是耗时根因**；这组数据只能排除“这些选中记录的 scalar 和 vector 导出字段均接近零”的表述，不证明所有流水、全卡或请求活动。不应据此断言可获得的提速。

脚本 `query_prepare_pmu.py` 在本人 stage 容器内以 SQLite `mode=ro` 对两份已有数据库执行有界查询；结果 `prepare-pmu-32-selected.json` SHA-256 为 `77ebe89f02b38a9c5dda0e749086b2a4e51f20ecb1a282e4adf8b2fd0ea5a4af`，结果同时记录数据库和 CSV 摘要。32/32 CSV 匹配只是交叉核对同份采集的两个导出视图，不是独立测试。本次没有启动新 NPU 负载或修改目标实现。
