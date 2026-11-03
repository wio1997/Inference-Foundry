# F08：EP3 两次 launch 间隔的主机记录覆盖

范围：同一个已保存 EP3 profiler 数据库，选定 L9 短写入和 L30 长写入；host 时间只与 host 时间比较，未跨 host/device 计算延迟。只读查询，未启动 NPU。两者不能外推成总体分布。前述 `F08-EP3-GAP-SYNC-AND-LAUNCH.md` 的查询只找了**完全包含**在间隔内的 API，遗漏了跨边界的 API；本分析改为取有交集的区间并裁剪，纠正其对可见记录覆盖的遗漏。

| 样本 | 同 globalTid 两个 CANN launch 的间隔 | 与 rotary 的 Dequeue 记录交集 | 与 rotary 的 CANN API 记录交集 |
| --- | ---: | ---: | ---: |
| L9 | 1687.37 μs | 1671.14 μs，99.038% | 1668.83 μs，98.901% |
| L30 | 432.16 μs | 416.44 μs，96.362% | 414.26 μs，95.858% |

同窗口还有 scatter 的 Dequeue 与 CANN API，以及两条 `aclrtGetStreamAttribute`、一条 tiling；API 存在嵌套，单项覆盖不能相加。故可以说这两个 host 间隔的大部分处在**已记录的 rotary 调用作用域内**；不能说是调用实际 CPU 忙、线程等待、算子计算、锁等待或 NPU 空闲。只凭时长无法分辨 profiling 引入的变化、库内部执行和调度行为。之前 0.67/0.68 μs tiling 自身不足以解释毫秒级间隔的限定结论仍成立。

对这四条 Dequeue，在 `CONNECTION_IDS` 表中各找到一个同共享连接值、名称分别为对应 rotary 或 scatter 的 Enqueue；Enqueue 与 Dequeue 位于两个不同 `globalTid`。共享值是导出的记录关联，具体 API 队列语义与全链路时钟仍须按同版本文档核实，不能直接认定队列等待时间。Dequeue 的 `callchainId` 均为空；Enqueue 所在线程对应时刻有 `aclnn...` 与 `_C_ascend::...` 的 PyTorch API 记录完整包含该 Enqueue。其 API 栈含 `dsa_v41.py` 的 `_apply_rotary` / `_write_swa_cache` 和 `multistream_preprocess`；时序包含支持源路径追踪，不是单独证明精确父子绑定或张量身份。

**可排除的解释仅限于记录口径：**不是所有 1.69 ms 区间都缺少同线程 host API 记录；原查询未查到中间完整包含项不能支持这一说法。内部究竟发生什么、与 scatter 的设备时长有无因果关系仍未知，下一步需同版本 profiler 对 Dequeue / CANN API 的边界语义与调用内部采集覆盖；再看既有调度或运行时栈是否能区分运行、阻塞和 profiler 开销。没有此类证据时不将高覆盖当作根因。

复算：`f08_ep3_host_cpu_overlap.py` → `f08-ep3-host-cpu-overlap.json`；`f08_ep3_host_queue_links.py` → `f08-ep3-host-queue-links.json`；`f08_ep3_queue_connection_map.py` → `f08-ep3-queue-connection-map.json`；`f08_ep3_enqueue_parent.py` → `f08-ep3-enqueue-parent.json`；`f08_ep3_enqueue_stacks.py` → `f08-ep3-enqueue-stacks.json`。脚本以 stage 容器 Python 运行且数据库使用 SQLite `mode=ro`；本地保存脚本和 JSON。连接表查询关闭 SQLite 自动索引以避免构建临时索引造成 I/O 错误，不影响查询的匹配条件。
