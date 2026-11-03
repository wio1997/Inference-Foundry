# F08 scatter：T=8180 的已保存 profile 短长任务对照

## 样本与结果

`f08_scatter_variation.py` 只读取现有唯一匹配过的三任务和 16 rank 聚合，不启动 NPU；确认三个关键键 EP、scope、flow 唯一，以及 task ID、stream、开始时间与三任务记录一致。T=8180 共 640 个选定 scope；按照事后探索阈值，31 个任务低于 50 μs（16.68–19.76 μs）、609 个高于 200 μs（244.655–274.8545 μs）、中间无样本。它们不是 640 个独立请求的证明。31 个短任务中 29 个来自 EP3，其余各一个来自 EP0、EP8；30 个来自摘要 `db852bff...eefc8`，一个来自 `1b826883...e89d`。这是观察，不是根因。

同一 EP3、同一摘要、同 stream 35 对照：L9 scope 2128-15 / task 13226 是 16.680 μs；L30 scope 2128-36 / task 13468 是 265.060 μs，约 15.89 倍。它们导出的输入 shape 都为 `17315,128,512;8180,2;8180,512`，输出 shape 都为 `17315,128,512`，输入类型 BF16/INT32/BF16、格式 NCL/ND/ND、AI_VECTOR_CORE 和 block 48 一致，不能将差异简单归因为这几个已显示的字段不同。仍缺实际 slot 索引内容、有效元素数、地址/stride、缓存状态、并行任务和执行环境；相同摘要也不能证明两层所处理数据相同。

这两条 CSV 分别显示 AIV total cycles 为 1,146,886 与 1,268,528，AIV vector ratio 为 0.025 与 0.023。计数汇总不与 trace 时长按相同倍数变化，但既不能用该比例推断浪费时间，也不能单凭此证明 PMU 有误：这需要对同版本工具汇总口径、计数归一化及逐任务 PMU 关联进一步核实。

## 原始 trace 和 DB 对照

`f08_ep3_scatter_raw_audit.py` 是在记录任务目录的已有 stage 容器内以只读方式扫描同次 EP3 `trace_view.json` 和 `ascend_pytorch_profiler_3.db`，输出见 `f08-ep3-scatter-raw-audit.json`；原 trace SHA 可由 `target-layer-all-kernels-16rank.jsonl` EP3 `input_sha256.trace` 校验，为 `c1eb4df8708a10d2c0be9af84ef8ed2a18f29c19f6297eb1f62aa6743ecaaa85`。两事件在 trace 的同一设备 track、stream 35 各只有目标 X 与其自身时间区间相交，同 track 对应 task ID 各出现一次；同一 stream/task 的 DB TASK 行各唯一，且 startNs/endNs 差分别为 16,680 / 265,060 ns，和 CSV/trace 一致。对应 flow 终点分别精确落在各自 X 的开始时间。此检查排除在该 track 与这两个时段内可见的更长同名外层 X 或同流 ID 复用；不能排除未记录内容、其他 track、采集截断或 PMU 关联问题，也不能把三个导出文件的一致性误作独立硬件计时验证。

短样本中的 RmsNorm → rotary 和 rotary → scatter 同流 trace gap 分别约 1,290.90 μs 和 1,693.28 μs，而长样本约 0.020 和 0.020 μs。gap 是选中事件时间间隔，不能由此断言设备空闲或事件等待，也不能断言造成了 scatter 短时长。短样本和大 gap 的共现值得进一步核对 profiler 的 stream 时间线采集质量，尚不足以给线上优化排序。

复算：`python3 agents/fusion/f08_scatter_variation.py`；跨 rank 执行依赖仅用于已有 JSONL。原始 EP3 检查脚本依赖 stage 容器内的已有 trace 路径与 ijson。分析不改变模型实现、不生成 NPU 工作负载。

## 同次 EP3 PMU 点查：绑定与归一化的边界

针对上述 L9/L30 两项，`f08_counter_binding_audit.py` 只读查询原 DB 与 kernel CSV，复算结果 `f08-counter-binding-audit.json`。各自 globalTaskId 分别为 2078/2942：在该 DB 的 TASK 表各仅一行，PMU 表各仅一个时间戳、28 个名称；对应时间戳比 TASK 结束早 160/120 ns，PMU cycles 与 CSV 精确一致、vector 比率与 CSV 保留位数一致。这在本样本提供很强关联证据，但因未获得导出器生成时间戳及行关联的规则，不宣称逐次绑定语义已被源码证明，更不推广到有重放的任务。

两条 PMU 分别记录 total cycles `1,146,886 / 1,268,528` 和 total time `13274.1435185 / 14682.0370370`，两者商都是 `86.4`；各项 `*_time / total_time` 与其对应 `*_ratio` 代数相符。同一次采集的 `device_3/info.json.3` 记录 `aiv_num=48`、`aiv_frequency=1800`，恰有 `48×1800/1000=86.4`；采集配置记录 AIV task-based、PipeUtilization，PC sampling off。数值一致支持固定容量归一化的假说，但 **未找到同版导出器计算实现和这些字段单位的充分证明**，不能据此把数值换算成真实运行或等待时间，也不能用 block 数推断同时激活核心数量。特别是长任务墙钟 265.060 μs 与 PMU 总周期并不按同一倍数变化，这是需要进一步解释的现象，不足以识别成调度等待、内核轮询或采样缺失。

当前可分辨的备选解释包括：PMU 的周期或时间以部分核心的活跃窗口计量并按固定数量归一；任务时间戳包含硬件侧未被上述 PMU 包含的阶段；采样/导出存在缺失或其他归属口径。现有数据不能选定其中任何一种；还需要同版字段生产代码或权威的时间单位、时钟与多核聚合定义，以及能关联各 block 活动时段的数据。另一次 task_time.csv 只重复了相同时长，不是独立测量。
