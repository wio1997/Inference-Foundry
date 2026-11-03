# EP3 第 15/36 层 MC2：通信调度记录与整体通信范围

沿用 `MC2-DB852-L15-L36.md` 同一 profiler 采集、EP3 两条通过开始时刻等精确绑定的任务；本次另查它们 **自身完整设备区间**内，同一 SQLite 的同设备、其他 stream 的 `TASK`，并用 `globalTaskId` 连接 `COMMUNICATION_TASK_INFO`，再按 `opId` 连接 `COMMUNICATION_OP`。这不是新的运行或独立重复试验；两层有效专家工作量和 kernel 二进制身份仍未知。

| 范围 | MC2 导出时长 | 相交的其他 stream TASK 行 | 其中连接到 Notify_Wait | COMMUNICATION_OP 与 MC2 区间相交 |
| --- | ---: | ---: | ---: | ---: |
| EP3 / L15 / globalTaskId 2380 | 5021.500 μs | 0 | 0 | 0 |
| EP3 / L36 / globalTaskId 3198 | 3502.260 μs | 16 | 4 | 0 |

L36 四条 `Notify_Wait` 分别在 stream 182、142、143、144，均在 `COMMUNICATION_TASK_INFO` 通过 `globalTaskId` 匹配，且 `opId=249`；按 `opId` 连接到 `hcom_allGather__503_249_1`。四条 wait 与 MC2 导出区间相交部分的**并集**为 1545.560 μs，约为目标时长 44.13%；并集不是把四条时间相加。该 allGather 的 `COMMUNICATION_OP` 行起点在 MC2 导出结束之后 458.060 μs，故尽管 COMMUNICATION_OP 本身不与 MC2 相交，已导出的、归该 opId 的四个 Notify_Wait 仍相交。

这是在数据库中将 **通信调度记录→opId→一个通信算子记录**建立了关联，**不是将该通信算子绑定到 MC2 所需的张量、层、请求或同步依赖**。空 `COMMUNICATION_OP` 交集仅能排除符合该过滤条件的已导出行，不能排除通信相关调度；L15 的空结果也不能证明没有内部通信或整个设备空闲。四个等待开始和结束跨不同轨道，跨轨时间转换误差未核验，wait 标签本身也不说明等待的生产者、是否占用资源、实际数据传输或阻塞 MC2。L36 先结束并不证明它不依赖该 collective；亦不能归因其较高活动或较短时长。

仍需同次 MC2 的路由分布/有效专家行数、MC2 内部时间与等待对象身份、跨流/跨 rank 通信关系、同版本采集完整性；在不能新增 NPU 负载的当前阶段，这些均保持未知。此结果修正了把仅查 COMMUNICATION_OP 的负结果等价为“没有通信活动”的潜在误读。

复算 `audit_mc2_ep3_peer_communications.py` 从 stage 容器只读该 DB，选定任务唯一性、筛选条件、所有匹配行和并集在 `mc2-ep3-peer-communications.json`；不读取请求、权重或张量值。
