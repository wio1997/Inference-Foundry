# EP8 两条 MC2 任务：其他流的 TASK 记录

## 新观察

在一次带 profiler 的已有采集中，对 EP8 的第 14 层、39 层两条已绑定 `DispatchFFNCombine` 任务，分别查询**同一数据库中设备 8 的其他流 TASK 记录**。此前 `EP8-L14-L39-FLOW-INTERVALS-20261029.md` 只选相应层 host scope 绑定的 flow；六窗口通信分析排除 AI_CPU。本次不以 host 层 flow 筛选，因此能看到额外同窗记录。

| 其他流记录 | L14 慢任务 | L39 对照任务 |
| --- | ---: | ---: |
| 锚点 DB 区间时长 | 160.365956 ms | 8.290414 ms |
| 正时长 TASK 行 | 61 | 6 |
| EVENT WAIT | 2 条，并集 160.365956 ms | 2 条，并集 8.290414 ms |
| NOTIFY_WAIT_SQE | 15 条，并集 160.365956 ms | 4 条，并集 8.290414 ms |
| KERNEL_AICPU | 9 条，并集 97.497551 ms | 0 条 |
| NOTIFY WAIT | 18 条，并集 75.663589 ms | 0 条 |

以上是裁剪到各锚点时窗后的区间并集，每类内部不重复累加重叠区间；**不同类型之间可以重叠，不能相加**。L14 另有 NOTIFY RECORD 10 条、NOTIFY_RECORD_SQE 7 条；3 条零时长记录另列而未纳入并集。L39 没有零时长记录。两窗口中完整身份键重复行均为零，但这不证明不同身份行一定是不同物理工作，尤其不能排除包含、封装与不同表示的同源事件。

L14 的 9 条 `KERNEL_AICPU` 与已保存 kernel CSV 均按 device、model、stream、task、精确起点各唯一匹配，CSV 所报 `Accelerator Core` 均为 `AI_CPU`，分别是各 3 条 `allgatherAicpuKernel`、`alltoallAicpuKernel`、`reduce_scatterAicpuKernel`。其中 stream 92 / task 3999 的 `reduce_scatterAicpuKernel` 全部导出时长 96.101940 ms，落入 L14 窗口的部分为 **95.231037 ms**，从锚点开始后 65.134919 ms 一直延续至该锚点结束。另有 stream 90 / task 4001 的 `NOTIFY WAIT` 与这段裁剪区间相交 **74.985902 ms**；两条记录的 connectionId 不同，不是同一个连接的嵌套，不可仅由相交推断事件依赖。

9 条 `KERNEL_AICPU` 各有一条同 connectionId 的 CANN API 记录，名称为 `launch`，每个连接还有其他 NOTIFY 类型 TASK；这是同库字段关联，**连接、时序和名称不证明实际同一通信事务、请求、张量、事件因果或占用资源**。该发现支持继续追查外部 AI_CPU 通信任务与慢 MC2 同窗时的依赖及资源关系，但它不证明通信造成 MC2 慢、不是通信链路带宽或传输时长，更不是 NPU 闲置、AI_CPU 忙碌时间、线上服务延迟或提速承诺。L14、L39 路由与实际工作量不相同的可能性未排除；两窗口没有收录到相应记录也不证明其不存在。

## 采集与身份边界

锚点来自既有 host scope→唯一 flow→设备 X→CSV 绑定；选中 DB 同 device、PID、stream、task 和精确起点得到唯一任务。数据库单用 stream/task 各有两个嵌套候选，不能用 task ID 独立定位。trace 开始时间的十进制微秒乘 1000 与 DB startNs 精确相等；DB 与 trace 导出时长只差 +0.25 ns、−0.25 ns。两者来自同次采集的派生数据，数值相符不构成独立时钟误差校准。

仅查 `TASK` 表中同设备其他流、与锚点相交的条目，空 stream 也保留供检查；不覆盖 stream 47 上其他记录、未导出任务、MC2 内部阶段及硬件流水活动。`EVENT WAIT`、`NOTIFY_WAIT_SQE` 覆盖锚点只表示其导出区间覆盖，不意味着 MC2 等待这些记录。采样未证明完整，profiler 对时延有扰动。前述六窗口报告的 AI Core/通信筛选没有找到选中项相交，不包含本次 `AI_CPU` 类型，两者口径不同。

## 复算与下一步

在任务根目录执行 `python3 agents/diagnosis/run_ep8_mc2_allstream_overlap.py`，只读既有数据库和 CSV，运行前后检查本人 stage 容器身份、镜像及挂载，`mode=ro` 加 `query_only`，仅 CPU，不启动设备。输出 `agents/diagnosis/ep8-mc2-allstream-overlap.json` 保留锚点、各类并集、最长样本、连接关联及输入 SHA。CSV 名称关联要求五字段及精确起点唯一、duration 相差不超过 1 ns；重复键、零时长分别统计，不把 `globalTaskId` 当全局唯一运行身份。

下一步若现有同次数据允许，可核验 reduce-scatter 调用来源、通信成员/事件对象与请求/MC2 的关系，同时核查采集完整度；缺少这些时只保留同窗现象，不能归因。优先检查现有资料，不重复启动采样。
