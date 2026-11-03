# EP3 同卡已归属任务：设备区间覆盖与局限

## 可回答的问题

此前的八卡同形状组严格排除了缺卡、不等数量和缺字段组。为避免这一筛选限定，同一个历史采集另选 EP3、摘要 `db852bff…` 的 40 个已归属 target 层 scope，把已匹配的设备记录按照原始算子名和 CSV type 汇总；无需其他卡的形状或任务数量满足条件。区间采用 trace 的设备开始时间及 duration，单个时钟域内取并集；相同 stream、task ID 和 start 三元组的 1580 条均唯一，且已关联 CSV 时长差均不大于 0.01 µs。

**所选 1580 条任务的设备区间并集为 353.614 ms。** 这是这些已选记录的时间集合总长度，不是请求耗时、完整 40 层窗口耗时、芯片忙碌时间或无 profiler 时延。采集经过 profiler；被选中的 scope 和异步任务不一定是同一个完整请求的关键路径。

| 算子类型 | 所选记录数 | 类型设备区间并集 | 占所选任务区间并集 | 与其他所选类型的重叠并集 |
| --- | ---: | ---: | ---: | ---: |
| MoE `DispatchFFNCombine` | 40 | 156.640 ms | 44.30% | 2.090 ms |
| `SparseFlashMla` | 40 | 34.437 ms | 9.74% | 3.044 ms |
| `VllmAscendQuantLightningIndexerV2` | 8 | 24.610 ms | 6.96% | 1.918 ms |
| `QuantBatchMatmulV3` | 128 | 19.257 ms | 5.45% | 3.480 ms |
| `MatMulV2` | 139 | 19.081 ms | 5.40% | 1.366 ms |

每行“重叠”是该类型的并集与其他**已选类型**并集的交集，计算为 `U(type)+U(other)−U(all)`。因此多个类型的比例可能重叠，不能相加，也不能用于量化可隐藏或可节省时间。由于只能看到已选记录，所谓相对其他类型独占也不排除未选流、没有匹配的事件或 NPU 内部子任务。

该范围内 MoE 是最值得继续调查的类型：其 40 条导出设备任务中位时长为 3.867 ms，最长为第 0 层的 9.574 ms；该条 CSV 的 AIC MAC ratio 显示 0.067、AIV vector ratio 显示 0.016。它是混合计算与通信的复合任务，两个字段分母不同，不能从低值推断等待或算力闲置，也不能将该任务完整时长当成计算或通信时间。该调查优先级只覆盖本卡所选记录，不排除其他模块在遗漏区域更重要。

## 采集覆盖的实质缺口

同一 EP3 全部 **80 个 target scope（含两个 digest）**的原始关联统计中，`DispatchFFNCombine` 80 个候选均唯一匹配 CSV，`SparseFlashMla` 同为 80，Indexer 16/16。另一方面，`EVENT WAIT` 1021/1021 和 `EVENT RECORD` 1424/1424 都无相应 CSV 行而没有进入上述 1580 条任务并集；另外有 240 个 device X 缺少名称或任务标识。这些数字的口径是 EP3 全部 80 个 scope，不能直接当作上述 40 层子集的缺失数。

因此，上表不能代表全部设备事件，也不能以与其他已选任务的重叠较少来断言没有同步等待、设备空闲、无通信重叠或关键路径暴露。raw trace 的事件可以另行核对，仍需取得事件对象、完整区间和生产消费依赖后才能解释；MC2 内部工作、CANN 导出的重复或遗漏亦未闭合。CSV 匹配是原采集派生的对应关系，不是独立测量。

## 复算

本机 `python3 agents/diagnosis/recheck_ep3_selected_device_intervals.py > agents/diagnosis/ep3-selected-device-intervals.json`，脚本先核验既有 JSONL SHA-256、EP3 和 40 层完整性、所选任务身份与时长；输出保留各类型区间并集、类型交叠、单次 MC2 时长及字段和 80 scope 原始覆盖计数。未启动 NPU、未修改推理实现。更大范围及抽取约束见 `RECAPTURE-ALL-KERNEL-RANKING.md`、`RANK-METRIC-COMPARABILITY.md`。
