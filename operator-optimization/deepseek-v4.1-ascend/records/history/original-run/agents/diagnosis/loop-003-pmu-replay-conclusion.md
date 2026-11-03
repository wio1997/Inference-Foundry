# Loop-003：PMU 重复记录与通信关联

对 EP8 数据库的只读查询结果见 `mc2-pmu-replay-links-ep8.json`，查询脚本为 `query_pmu_replay_links.py`。该查询使用 `globalTaskId=18529`、`connectionId=184330`，只覆盖此 rank 的一个图任务身份。原数据库、脚本 SHA-256 均记录在 JSON。

**7 条 PMU 记录有很强的逐次对应迹象，但仍是待证实的关联规则。**数据库有 7 条 TASK 记录共享同一个 globalTaskId、connectionId、modelId=49、streamId=89 和 taskId=64，开始及结束时刻不同；7 个 PMU 时间戳各有 28 个名称。按时间顺序，两者一一对应，PMU 时间戳相对对应 TASK 结束时刻的差值分别为 **−20、−20、−1、0、−20、0、−21 ns**。这强烈支持 7 条记录对应 7 次该图任务的执行／重放，而不是一次执行内 7 个阶段；但 profiler 中时间戳的生成及四舍五入语义尚未在来源中确认，所以尚不能声称得到形式上唯一的 replay ID。

目标 TASK 结束于 `1793152958877820884 ns`，唯一最近 PMU 时间戳 `1793152958877820864 ns` 相差 20 ns；其 `aic_mac_ratio=0.0103997974`、`aic_mte2_ratio=0.0783996329` 与此前同一个调用 CSV 中的四舍五入数值 0.010、0.078 一致，进一步支持这个候选关联。这些比率是整次算子的流水数据，不是内部 dispatch、GMM、combine 或等待时长。

**关键约束：**单独用 globalTaskId 或 connectionId 不能唯一定位一次执行：二者在 7 个 TASK 行中都重复。直接以该键连接全部 7 次 TASK 和 7 组 PMU，会造成多对多重复归属。应继续核查生产这些表的 profiler 导出逻辑，并把运行时间戳及容差的依据固定后，再使用复合关联。

精确查询返回 `COMMUNICATION_TASK_INFO.globalTaskId=18529`、`COMMUNICATION_SCHEDULE_TASK_INFO.globalTaskId=18529`、`COMMUNICATION_OP.connectionId=184330` 均无行。因此**本次没有找到使用这些精确键的外部通信事件**；这并不否定 DispatchFFNCombine 融合内核内部的跨 rank 通信或等待，也不能量化其耗时。以上结论不推广到其他 15 rank、其他 task 或其他导出。

后续可不使用 NPU，查同镜像 profiler 的数据库写入源码或官方定义，明确 `TASK_PMU_INFO.timestampNs`、`TASK.endNs` 及 globalTaskId 的复用／取整规则；如来源不足，继续将对应关系标记为“时间戳支持的候选”，并优先排查有明确模型依赖与请求关联的已有 trace。只有新增数据确实必要且设备空闲时，才考虑更细分阶段采集。
