# EP8 历史窗口：已导出通信和计算区间交集

本次只读固定 `kernel_details.csv`，文件 SHA-256 为 `488d3e3736677d7d041edbe8307d3c06b79553362e54aef07aba37fd74229cc7`，与 `evidence/TRACE-EXPORT-SHA256.json` 的 EP8 摘要一致。窗口取此前保存的 ordinal 7：前一目标层 39 的 MC2 结束至下一目标层 0 开始，长度 38,312,114 ns；并非完整 forward、单次请求或整个执行。行时间来自同一表的 `Start Time(us)` 和 `Duration(us)`，Decimal 转 ns；不做跨设备或主机时钟比较。此处只说明导出时间戳数学相交；尚未校准导出时间到设备真实执行区间的误差。

筛选此窗口与之相交、正时长且 Stream ID、Task ID 都非 `N/A` 的设备 8 行，通信定义为 `Type` 以 `hcom_` 开头；计算定义为 `Accelerator Core` 属于 `AI_CORE`、`AI_VECTOR_CORE`、`MIX_AIC`、`MIX_AIV` 且非 hcom。18 条通信行在 stream 39/88；1188 条计算行在 stream 35/37/47/89，即选中两类都处于不同流。裁剪后无选中行跨窗口边界。通信区间并集 290,214 ns，计算并集 6,288,067 ns，两者合并并集 6,578,281 ns，交集为 0；逐对交集也为 0。脚本 `query_ep8_comm_compute_overlap.py`，输出 `ep8_comm_compute_overlap.json`，只读原表并断言 SHA 和并集关系。

复核说明：归档 JSON 额外保存了 `query_sha256`，因此与脚本 stdout 不逐字节相同；独立复算时移除这一附加字段后内容完全一致，且该字段与脚本实际 SHA 相符。

例如第一条通信行的表内身份是 device 8、model 49、stream 88、task 497，`hcom_allGather_` 相对窗口为 91,318–107,678 ns；上述所选计算行无一与之时间相交。model/stream/task 加起点是表中局部身份，并没有再次由 trace flow 把这些通信和特定模型调用、请求、设备内部操作一一绑定；task ID 可能复用，不可推广成全局唯一标识。

这只是有限导出行的时序，不证明真实通信与计算没有重叠、不证明存在依赖、通信不可隐藏或该时间位于关键路径。`hcom_*` 行可能是通信汇总或包含嵌套；筛选排除了非数字 stream/task 的 N/A 汇总行，也排除 AI_CPU 和不在列举核类型内的所有其他行；计数不保证 trace 或设备活动完整，融合 MC2 的内部通信和计算也无法用外部区间拆分。不同流亦不意味着独立；此窗口跨草稿和目标等异步活动，未绑定相同请求、payload 或 collective。下一步如需追因，应有具体通信 task 到 source/replay、事件依赖、真实载荷及 profiler 覆盖的可验证绑定；当前结果不足以预测收益。
