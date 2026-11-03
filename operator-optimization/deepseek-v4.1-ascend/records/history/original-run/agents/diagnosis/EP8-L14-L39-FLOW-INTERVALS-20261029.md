# 首次 EP8 采集：第 14 / 39 层同范围 flow 和 MC2 外部区间

这是对原有带 profiler 的一次采集的只读分析，不是无 profiler 基线、内部阶段性能分解或因果证明。没有启动服务、设备负载或新采集。目标为 EP8、PID 737、scheduler digest `1b826883…` 的第 14 层 scope `737-20` 与第 39 层 scope `737-45`；此前结果已将它们各自唯一链接到 `DispatchFFNCombine`。本次在同一个 trace 里复核同线程同层 scope 内的所有 `torch_to_npu` start，要求每个 flow 全局恰好一个 start 和 finish，再用 finish 的 `(track pid, track tid, timestamp)` 在所有 X 事件中要求唯一匹配。MC2 锚点与原有 task ID 和 start 严格匹配。

这条链仅证明**这些事件在 trace 中通过 flow 连到该层 host 提交范围**。它不证明其中任意事件与 MC2 共享 tensor、属于同一 collective、在请求关键路径上，亦不能单凭 flow 名称将标为 HCCL 的 track 等同网络物理传输或证明事件之间存在设备依赖。flow 不完整、被跳过或没有记录的活动均不在结论范围内。

| scope | 唯一关联 X | 发生于 MC2 前 | 与 MC2 相交 | MC2 后 | MC2 trace 时长 |
|---|---:|---:|---:|---:|---:|
| L14 | 102 | 95 | 1（MC2 自身） | 6 | 160365.95575 μs |
| L39 | 68 | 61 | 1（MC2 自身） | 6 | 8290.41425 μs |

表内仅包括通过条件筛选的唯一 flow→唯一 X；它不是整卡、所有流、全部 profiler 事件或所有未记录通信的覆盖。每层 102/68 个选中 flow 均有唯一匹配，0 个未匹配、0 个多重 X；全局非唯一 flow 在候选构造前排除，其数量见 JSON，不能把已选集合计数说成 trace 完整覆盖。只按半开区间算术比较同一 trace 中的 X 时间，不跨 rank 或不同 profile 的时钟。

以下三个外部 `hcom` 名称的 X 均通过上述精确 flow 关联到各自层的 host scope；时间是相对于本层 MC2 开始的 μs。它们所在 trace track `(755208,0)` 与 MC2 `(755144,47)` 不同，`hcom` 行没有 Model/Task/Physical Stream ID，不能基于相邻顺序给它们造设备任务 ID。

| 层 | trace 事件 | 相对开始 | 相对结束 |
|---:|---|---:|---:|
| 14 | hcom_allGather__097_180_1 | -7216.575 | -6833.023 |
| 14 | hcom_alltoall__097_181_1 | -2321.013 | -1666.106 |
| 14 | hcom_reduceScatter__097_182_1 | -1017.259 | -536.629 |
| 39 | hcom_allGather__097_258_1 | -4329.014 | -3957.061 |
| 39 | hcom_alltoall__097_259_1 | -2283.795 | -1685.507 |
| 39 | hcom_reduceScatter__097_260_1 | -1040.360 | -544.950 |

**可排除的窄解释：**这六个已精确关联的外部 HCCL 命名 X 显示区间与对应 MC2 显示区间没有交集，因此其导出区间本身不可能是 MC2 显示区间内直接相交的那段时间。不能据此排除更早通信对状态、调度或数据的间接影响；也不能排除 trace 未显示或发生在 `DispatchFFNCombine` 内部的通信与同步。L14 比 L39 长约 152.076 ms（单 EP8 trace），不能因此简单归因于计算或等待。

两个同范围 `EVENT WAIT` 也有与 MC2 相同的物理 stream 47：L14 task 44205 为 `[-1030.619,-523.88925]`、时长 506.72975 μs；L39 task 45351 为 `[-1053.239,-532.2895]`、时长 520.9495 μs。它们均在 MC2 起始前结束，且时长相差约 14.22 μs。flow 和同 stream 支持这是同一设备任务队列中的已记录事件；它们与附近 reduceScatter 的 X 区间相交**并不证明**那个 wait 等待的正是该 collective，或说明 MC2 内部没有等待。没有同一次消息、event record/wait ID、内部 stage 和依赖关系，不能将这些区间相加或视作可隐藏通信量。

本分析重新提出的量化结果是外部已绑定集合的时间位置，而非“找到了 MC2 长时原因”：两个 MC2 形状相同也不证明真实专家路由行数、数据、通信或内部状态相同；既有低 MAC、高 AIV MTE2 字段仅为整个任务聚合。若要辨别内部等待、有效计算和内部通信，仍需同次 task 绑定的内部阶段或相应计数与版本语义，缺失标 UNKNOWN。

## 可复算与来源

在任务根目录运行 `python3 agents/diagnosis/run_ep8_l14_l39_flow_intervals.py`。此命令只通过 SSH 在已存在的自有 stage 容器执行 nice/ionice CPU 流式读取，前后核对容器 ID、镜像和挂载，校验所保存 EP8 trace SHA，不触发 NPU。输出只含匿名 flow/track/task、事件名称和相对时间，不含 prompt 或张量内容；输出在本地分析目录。脚本和结果：

- `ep8_l14_l39_flow_intervals.py` SHA256 `8e38bc09875ccdc0818e682707e52c4e4a05ae97aa8f941c971d7057ef19f36e`。
- `run_ep8_l14_l39_flow_intervals.py` SHA256 `535fa95c2f01388497a108b6d829ab6c2078d35ba4fdc250e7a9819d9f099354`。
- `ep8-l14-l39-flow-intervals.json` SHA256 `8b84dbea06027a71905d0726d252f78fd547c5545db67b064bc72eff4575d80b`。
- 所读 `trace_view.json` SHA256 `53ec20cf6d8b8f79b67a7f2f4ddaabf823b4c0305e88e2dbf9cf62d02e7c6f23`。先前完整层范围/MC2 关联证据 `diagnostic-20261029/ep8-737-5-layer-mc2.json` SHA256 `afaeab84b04bc49ff371a00c61c07eaf4abb9ea0f7626b0f93bc48f818b2c795`。
