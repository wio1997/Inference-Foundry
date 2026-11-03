# EP8：输出投影与 reduce-scatter 的双向事件候选

范围是 `EP8-SP-REDUCE-SCATTER-FLOW.md` 的同一 EP8 历史采集、同一进程 737、同一个已关联的 HcclReduceScatter。只读 SQLite；不启动新负载。此前已从 host 关联 hcom 和 AI CPU，此处进一步按 **同一库的 connectionId 精确匹配**追查前后 EVENT TASK 和 CANN API。

## 可以确认的关联

同一个工作线程的 CANN API 先后包含：`aclrtRecordEvent` 88654、`aclrtStreamWaitEvent` 88655、`hcom_reduceScatter_` / `launch` 88657/88658、`aclrtRecordEvent` 88683 和 88684、`aclrtStreamWaitEvent` 88685。对应 PyTorch Dequeue 区间依次包括 record、wait、HcclReduceScatter、record、record、wait；同线程时间包含支持这些调用位于对应作用域内，不能把包含关系当作额外的独立验证。

`CONNECTION_IDS` 将原始 host `HcclReduceScatter` 174588 同时关联 CANN launch 88658 和 88670；后者的 TASK 中有先前 trace 关联的 stream 92 `reduce_scatterAicpuKernel` 3861。88658 的 TASK 包含 stream 39 上编号 26445 的通信任务。二者共用原始关联，不是两次独立实验。

| CANN connectionId | 精确匹配的 TASK | TASK 类型 | 设备导出坐标下结束点与对侧候选的比较 |
| --- | --- | --- | --- |
| 88654 | stream 47 / task 43442 | EVENT RECORD | stream 39 WAIT 结束晚 1.039 μs |
| 88655 | stream 39 / task 26442 | EVENT WAIT | 同上 |
| 88683 | stream 39 / task 26447 | EVENT RECORD | stream 47 WAIT 结束晚 1.040 μs |
| 88684 | stream 39 / task 26448 | EVENT RECORD | stream 47 WAIT 结束晚 1.000 μs |
| 88685 | stream 47 / task 43443 | EVENT WAIT | 分别比较上述两条 RECORD |

stream 47 的首个 RECORD 在先前已绑定 TensorMove 43441 后；stream 39 的 WAIT、通信任务和两个 RECORD 依次出现；stream 47 的末个 WAIT 区间则包围先前关联的 hcom 导出区间。这些顺序和相近结束时间**与两条流在通信前后进行双向事件同步相一致**，但这仍是一个候选解释：TASK/CANN_API 表没有给出上述调用的 event handle。特别是末尾两条 stream 39 RECORD 紧邻，无法确定 stream 47 WAIT 等待其中哪一条，也不能排除等待其他未识别事件。不同轨道坐标的时钟转换误差尚未独立核实，约 1 μs 的差值不能当成准确的信号传播延迟。

CANN `Notify_Record` / `Notify_Wait` 与 `aclrtRecordNotify` / `aclrtWaitAndResetNotify` 也出现在 hcom 的 CANN 作用域；它们是不同名称的通知调用，本分析没有对象标识和唯一配对，不能与上述 event 直接等同。证据仍不足以宣称 WAIT 被 hcom 阻塞、时长是可消除的通信开销、设备空闲或可达收益。

## 复现和缺口

`audit_ep8_sp_event_candidates.py` 在 stage 容器中以 `mode=ro` 查询选定主键及对应 TASK/PyTorch/CONNECTION_IDS；结果见 `ep8-sp-event-candidates.json`。线程 API 区间顺序来自同一 DB，以参数、设备轨道关联和事件对象身份为独立未知。

如未来允许有针对性的诊断采集，需对指定两流的事件记录和等待保存稳定的 event 对象标识、调用发生顺序和完整设备执行范围，并验证跨轨时钟；还应确认应用实际流与 HCCL 流之间张量的 producer/consumer。现有约束不允许启动新 NPU 负载，本轮没有进行该采集。
