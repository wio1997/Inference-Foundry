# EP3 L36：allGather 通知和 MC2 的关系

延续 `MC2-EP3-L15-L36-COMM-OVERLAP.md`，仅针对 rank 3 的一个 L36 `DispatchFFNCombine`（globalTaskId 3198）以及同库 allGather（opId 249）。本次补查昇腾官方资料、冻结源码和既有 SQLite，没有启动 NPU。

## 官网口径与版本边界

昇腾 [CANN 9.0 profiler 数据库说明](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/devaids/Profiling/atlasprofiling_16_1144.html) 将 `COMMUNICATION_TASK_INFO.globalTaskId` 描述为关联 `TASK` 的全局任务 ID、`opId` 描述为关联 `COMMUNICATION_OP` 的大算子 ID、`notifyId` 描述为 notify 的唯一 ID。[CANN 9.1 HCCL profiling 说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta3/API/hcclug/docs/zh/user_guide/perf_analysis/profiling_op_behavior.md)将 `Notify_Record` 和 `Notify_Wait` 分别解释为置位和等待后清零的同步任务。[CANN 9.0 同步说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/programug/acldevg/aclcppdevg_000531.html)还描述一次 notify 的一对一通知与等待后重置。

该采集保存的版本是 **CANN 9.2.0**，这些是 9.0/9.1 文档，不能冒充 9.2 exporter 的已核实契约。自有 stage 下只读搜索安装版 profiler、torch_npu profiler 的 `.py` 文件，没有找到可直接解释通信表事件配对的实现源码；不代表其他安装路径或二进制中不存在相应逻辑。

## 本次通知的可验证事实

先前的四个待查 `Notify_Wait` 均通过数据库字段关联到 opId 249；它们分别位于 stream 182、142、143、144，**开始于 MC2 开始之后、结束于 MC2 结束之后**。在导出坐标下，四条均比 MC2 结束晚至少 458 μs。按任务展示坐标和完成边界可信、跨轨转换误差不足以颠倒先后顺序的条件，可以排除“MC2 的完成必须等待这四个指定 wait 全部完成”这一具体解释；不能排除 wait 的某一部分、其他通知、共享资源或 MC2 内部同步影响该任务，也不能将两条流接近当作因果证明。

对 opId 249 内的通知 7290、7292、7294：分别看到 10/9/9 条 wait 和 9/8/8 条 record。同 ID 的第一条 record 均出现在 stream 182，导出完成时刻比对应第一条 wait 的结束分别早 0.500、0.520、0.500 μs；顺序与通知配对**相容**。然而同 ID 在同一个算子多次复用，且不知 9.2 导出器对 ID 复用、流时钟及隐含记录的完整契约，不能仅凭 ID 和时间宣称精确匹配或当作传播耗时。通知 7357 在 opId 249 只见一条 wait、没有同 ID record；不能因此断言没有生产者。

## 与 DSpark 辅助数据的潜在关联

`CONNECTION_IDS` 将大算子的 connectionId 102723 关联到唯一 host `HcclAllGather` 记录 431416；其保存调用栈含 `sp_all_gather` 和 `deepseek_v41/model.py(1563)`。同一条 host 线程中，目标 MC2 所关联的 `aclnnDispatchFFNCombine` 431408 的开始时间比该 host allGather 的开始早 15.794480 ms；这里只比较同线程 host 时间，不作 host→device 延迟换算。

冻结源码 [model.py](../../sources/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/model.py) 的 1556–1564 行显示：在指定辅助层，先求进入该层的 residual 平均，启用 SP 时对该辅助状态执行 `sp_all_gather`，用于 DSpark auxiliary hidden states。冻结文件 SHA-256 `101f96b38e9fc47e7be147ddfe56976aee896867c73a33a7d95c3e3c833a6e91`。若历史进程实际加载相同代码、调用栈和连接链语义保持一致，这为该 allGather 的用途提供了具体解释；但历史 Python 源码和对象身份尚未绑定，也没有该 allGather 输入的 tensor/storage ID。**不能将其直接认定为 L36 MC2 的输入、输出或等待依赖。**

## 复算

`audit_ep3_allgather_notify_ids.py` 在专属 stage 容器只读 DB，输出 `ep3-allgather-notify-ids.json`，保存有限的通信记录、通知 ID、连接、调用栈和同线程 host 时间。没有读取请求内容或权重。尚缺同版本 profiler 导出契约、完整 event/notify epoch 及真实 tensor 身份，也缺 MC2 内部阶段和专家行数；本轮不能决定异常耗时根因。
