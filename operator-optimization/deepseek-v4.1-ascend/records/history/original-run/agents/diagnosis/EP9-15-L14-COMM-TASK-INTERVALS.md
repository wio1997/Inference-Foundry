# EP9–15：第 23 层 host 关联通信记录与第 14 层 MC2

以已有 `EP9-15-L14-REDUCE-SCATTER-SOURCE.md` 中 EP9–15 各卡 task 3999（AI CPU）为入口：它们各自关联到标记为第 23 层的同卡 `HcclReduceScatter` host 调用。这里选择该 host 关联的**另一条 connection** 上的 `COMMUNICATION_OP`，而非将 AI CPU 任务本身等同于通信大算子。各卡独立固定先前匹配的 device / PID / stream / task / 精确 startNs 的第 14 层 MC2 锚点；比较只使用同一个 DB 的导出坐标，不做跨卡时间相减。

| EP | 通信 op 起点距本卡 MC2 结束 ms | 子任务数 | 子任务精确关联数 | 子任务与 MC2 窗口相交数 | 最早子任务起点距 MC2 结束 ms |
|---:|---:|---:|---:|---:|---:|
| 9 | 227.462596 | 168 | 168 | 0 | 226.808049 |
| 10 | 227.412676 | 168 | 168 | 0 | 226.753109 |
| 11 | 227.470680 | 168 | 168 | 0 | 226.803080 |
| 12 | 227.412916 | 168 | 168 | 0 | 226.760409 |
| 13 | 227.460035 | 168 | 168 | 0 | 226.805188 |
| 14 | 227.450095 | 168 | 168 | 0 | 226.805168 |
| 15 | 227.461215 | 168 | 168 | 0 | 226.798589 |

各卡所选 `COMMUNICATION_OP` 的 opId 在该 DB 的 `COMMUNICATION_OP` 恰好出现一条；每卡该 opId 下 `COMMUNICATION_TASK_INFO` 的 168 个 `globalTaskId` 不重复，逐一在 `TASK` 恰好匹配一行，且每条 `timestampNs` 与匹配的 `TASK.startNs` 完全相同。这是该库范围内的字段关联，不是全局身份或历史格式契约证明。全部匹配 TASK 的 device 和 PID 均分别为本卡与本进程；均不携带所选 AI CPU TASK 的 connectionId。每卡选中 MC2 和 AI CPU TASK 的 globalTaskId 在该库 `COMMUNICATION_TASK_INFO` 中均没有匹配，不排除其他关联机制。每卡恰有一条匹配子任务起点早于其通信 op 起点，故更不能默认子任务均被 op 时间范围包含。

该 7 卡样本与此前 EP8 相同口径下的 168 条子任务不相交观察一致，详见 `EP8-L14-COMM-NOTIFY-SCOPE.md`；它不能扩展到未选中的通信操作、MC2 内部通信或未导出阶段。EP9、10、11、13 的该通信 op connection 在 `CONNECTION_IDS` 中关联两个 id，EP12、14、15 各一个；即使表内 opId 唯一，也不能将连接关系理解成唯一物理通信事务。被分析的长 AI CPU 记录与另一条关联通信 op 共同连接 host，未证明它们是同一物理阶段或同一请求。

各轨导出坐标来自同库，但时钟精度、跨轨校准和 profiler 完整性尚未独立验证；这里的时间差仅为记录差，不是实际通信链路时长、等待时长、通信占比或提速空间。无论时间不相交还是相交，都不能单独证实或排除通信/资源争用导致 MC2 耗时，不能推断第 23 层与第 14 层的张量或事件依赖。

复现：从任务根目录运行 `python3 agents/diagnosis/run_ep9_15_l14_comm_intervals.py`。脚本验证前一阶段匹配结果、目标层摘要和各卡先前首次观察的 DB SHA，检查 stage 身份/挂载，SQLite 只读并检查无 WAL，读取前后再次校验 DB。`ep9-15-l14-comm-intervals.json` 保存逐卡匹配、子任务行和摘要；仅用 CPU，不运行新 NPU 负载或修改推理实现。
