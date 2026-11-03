# 请求身份、阶段与动态长度：现有数据覆盖

ZCode stage 085 已完成（exit code 0，执行模型 `deepseek-flash`）。对 EP8 单个已保存 profiler 数据库，`query_request_metadata_schema.py` 只读检查了所有表名和字段名中带请求、阶段、长度、metadata 或 tensor value 的名称，并在 `STRING_IDS` 中检索了 6 个固定术语；结果见 `request-metadata-schema-ep8.json`。该规则没有命中匹配的表／字段名称或术语字符串。

**当前不能从已核查数据将第 24 层三个 Indexer 调用对应到具体请求、实际输入／输出阶段，或得到每次 query 有效长度、cache 有效长度、候选块有效数量。**已有 source-validated 窗口可将它们归于某个 target 图模型内的三个采样轮次；图连接与 model ID 不是请求 ID。正式 benchmark 汇总的 32K 输入、1K 输出、8 请求与 90% 前缀命中仅描述整体负载，不能推算每个 profiler 窗口内的动态数值。

未命中结论仅适用该 EP8 数据库、上述关键词和表／字段名／短字符串；字段可能使用其他名称，其他导出或服务日志可能保存其他证据，不能因此认定所有已有数据都不存在。没有读取请求原文。算子固定属性字段的无名称数字也不能代替动态 tensor 内容。

若要区分三轮耗时差异的来源，**最小缺口**是在同一调用上可验证地关联：

1. rank、graph replay 唯一序号、层号及调用时间；
2. 对应请求或批次的匿名标识、采样阶段；
3. 每个请求／批次实际 query 行数、cache 有效长度，以及候选块有效数或等价工作量指标；
4. 一致的算子模式、输入 dtype／layout 和观测时间边界。

这些字段需要在同一采样轮次与算子关联，单独的全局汇总不足以解释差异。若未来确需新观测，应先设计最小采集并检查卡是否空闲；本次没有新设备负载，也没有推断根因或可实现收益。

## 已保存输入构建路径与目标六条记录的缺口

针对 stage179 的 EP8、层 2/24、三个 target 图内部采样窗口，重新核对**已有**源码快照和以往有来源索引的 profiler 记录，没有再次按名称扫描数据库：

1. `agents/bounds/metadata-path-snapshot/ascend_runner.py:1435–1459` 中，runner 根据已调度 token 前缀和 `num_reqs` 构造 `query_start_loc` 的 CPU 端数组，并调用 `copy_to_gpu()`；同时构造 CPU 端 `optimistic_seq_lens_cpu`。`:1603–1620` 另构造设备 `seq_lens`，条件满足时修正 CPU 的 optimistic 值；`:3610–3670` 对 async spec decode 明确令普通 `seq_lens_cpu=None`，并注释设备张量为权威，而 `_seq_lens_cpu` 和 `seq_lens_cpu_upper_bound` 仍传 optimistic 数值。因此，即使未来只找到 CPU optimistic 数组，也必须核对是否为该次、是否已修正以及后续 backend 消费者，不能直接代替 QLI 实际有效 cache 长度。
2. 冻结 `attention/dsa_v41.py:1036–1045` 对 C2 compressed 条件分支，把设备 `seq_lens` 按比例做 floor 除并计算 residual；`:1227–1278` 用 `query_start_loc`、转换后的 `cache_seq_lens`、residual 等构造并发布 `qli_metadata`。`models/deepseek_v41/indexer.py:259–280` 在非 packed 正常路径将 `query_start_loc`、`cache_seq_lens`、residual 和 `qli_metadata` 等传给 QLI。`:587–608,640–671,700–711` 还表明同一源码 forward 中 Indexer 读取 `metadata.indexer.cache`、attention 使用 `metadata.swa` / `metadata.attention`，这些是同一 metadata 包中的不同字段；尚未证明其张量值、物理存储、逐次请求及真实图参数在本次记录中相同。源码确定的是**条件性生产与消费路径**，不是六个历史调用的数值。
3. `host-gap-tail-inspection/summary.json` 对 EP0/EP8 两个 ordinal 7 的候选主线程，曾按相对于 layer0 起点的数值区间筛查 CPU API；EP8 所列末 40 条中包含 `aten::copy_`、`aclnnInplaceCopy`，CANN 尾部有 `aclmdlRIExecuteAsync`。该摘要仅保存类型、时间、线程等，不含拷贝 payload、张量地址或 `query_start_loc/cache_seq_lens` 的值；线程选择是启发式、CPU/device 时钟对应未独立校准，时间邻近也不证明 copy 属于目标重放或 metadata。因此不能从它恢复长度，也不能据其算出具体下发耗时。
4. 先前 EP8 的 `ascend_pytorch_profiler_8.db` 全 26 张表 schema 清单（见 `engram-all-metrics-fields.json`）包括 `ENUM_MEMCPY_OPERATION`，没有独立命名为 `MEMCPY_INFO` 或 `TENSOR` 的表；它不排除通用表、其他字段或其他文件保存地址或内容。stage085 的六个固定术语在此数据库 schema 和 `STRING_IDS` 中未命中，范围限制仍见上文。`profiler-export-inventory.json` 是文件清单，不是文件内容的完整检索。原始 trace 中先前只是对其他目标做过有界参数查询，不能推广为全 trace 没有相关值。

因此当前**没有发现可与 EP8 目标六条同次 replay、rank、层绑定的匿名 batch/请求身份、真实 query 行数、有效 cache/candidate 数或 metadata tensor 内容**；这不等于断言其它未检查文件不存在。若未来有带来源索引的历史快照，应同时核验：匿名 batch 或请求与 replay 的关联、各请求 `query_start_loc` 差分、设备有效 `seq_lens` 经 compression 转换后的 `cache_seq_lens` 与 residual、实际 candidate 可用数、对应层/分支和缓冲就绪及复用时刻。仅知道 `32K` 全局输入或静态 `[3,...]`、CPU optimistic 值、一般 copy 名称都不够。

快照 SHA-256：`ascend_runner.py` `568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c`；`dsa_v41.py` `3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd`；`indexer.py` `d5d6a438bbced4e63d729dac94c08869df30dab4a71ceacecdea211386b3db33`；host 摘要 `fca74bae9fa50d35c11ff0f62ccacae04cbd387d536b7ae2772237ca2ee7ee09`。上述仅复核本机既存文件，没有新查远端、没有触碰 NPU 或读请求内容。
