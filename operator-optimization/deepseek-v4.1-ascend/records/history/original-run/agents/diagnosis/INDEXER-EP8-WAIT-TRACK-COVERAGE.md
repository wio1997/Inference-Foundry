# EP8 第 24 层 Indexer：低 MAC 样本的跨流同步记录

排查非 MoE 的长任务：已选 EP8 第 24 层 `VllmAscendQuantLightningIndexerV2` 三条记录分别为 32.020、498.491、498.170 μs；同一次采集的长轮 AIC MAC 比例约 0.054、AIV vector 比例约 0.03（CSV 显示值，不是整芯片利用率）。前期 `indexer-per-run-mode-findings.md` 对 384 个同类 Indexer 窗口查过 `kernel_details.csv`，第 24 层未匹配其他 stream 的 kernel 行；那项查询不含完整同步任务。

本次对同一份 EP8 历史 SQLite 按起止时间、device、stream、taskId 和 model 精确匹配三条 Indexer，并查询同设备其他 stream 与各自完整时间相交的 `TASK`。每条 Indexer 的 `globalTaskId=19466` 在该 DB 的 TASK 中出现 7 次，不能只用它识别单次重放；脚本使用精确起止约束。三次都找到整段覆盖目标时间的其他流记录：

| 内部采样轮 | 其他 stream TASK | 其中 EVENT WAIT | MODEL_WAIT_COMPLETE | NOTIFY_WAIT_SQE | 完整覆盖目标的行数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 3 | 1 | 4 | 8 |
| 2 | 11 | 6 | 1 | 4 | 11 |
| 3 | 11 | 6 | 1 | 4 | 11 |

三轮四条 NOTIFY_WAIT_SQE 均来自 stream 142、143、144、182，并各通过 `COMMUNICATION_TASK_INFO.globalTaskId` 在本库唯一匹配到 `Notify_Wait`；三轮分别连接到 opId 269、920、1063 各自的 `hcom_allGather`，这些各条通知的 globalTaskId 在 TASK 中均只有 1 次。stream 87/88 的 EVENT WAIT globalTaskId 在 TASK 中各出现 7 次，不能用 ID 取代重放时间。数据库在第 2、3 轮还各有 stream 35、37、43 上与目标相交的 EVENT WAIT，因此不是简单宣称三轮所有跨流事件完全相同。

**证据意义：**短轮和长轮均能看到相同四个 stream 上各一条关联 allGather 的通知等待，因此可排除“只有长轮出现这些 stream 上的此类等待记录”这一狭义说法；不能排除等待对象、时长、资源、路由或输入值在轮间有差异，也不能排除同步影响。此前“CSV 未找到其他流 kernel”的表述在其查询范围内成立，但不能解释为其他 stream 没有同步记录、设备空闲或没有通信。三个目标与另外的等待都位于不同流，时间先后和区间覆盖**不能证明 Indexer 等待它们或被通信阻塞**；同一次四条 wait 也不可相加认作瓶颈。MODEL_WAIT_COMPLETE 的具体 9.2 exporter 语义未在本次查明，不据名称推断该任务依赖。

## 官方资料与未解决的问题

[昇腾 CANN 9.0 profiler 表说明](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/devaids/Profiling/atlasprofiling_16_1144.html)描述 `COMMUNICATION_TASK_INFO.globalTaskId` 与 `TASK`、`opId` 与 `COMMUNICATION_OP` 的关联；[CANN 9.1 HCCL profiling 说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta3/API/hcclug/docs/zh/user_guide/perf_analysis/profiling_op_behavior.md)描述 Notify Wait/Record 的同步语义。历史采集是 CANN 版本另见 `indexer-ep8-profile-collection.json`，不能仅凭旧版文档证明本次的 9.2 导出细节、通知对象代际或跨轨时间校准。`NOTIFY_WAIT_SQE` 的任务标签与通信子任务 `Notify_Wait` 的连接是本库事实，不据此进一步假定等待的具体数据或 producer。

当前依然没有同一任务重放的有效 Q/K 长度、候选列表内容、参与核 enable、真实 binary 与当前源码的绑定；arch22 源码按有效长度切 2048 块、动态选择参与核的条件机制已在 `indexer-per-run-mode-findings.md` 分析，但不能把它升级为这三次耗时的原因。区分需同次动态 metadata 和内部阶段/资源证据及相应时钟、事件对象契约；既有证据不足，不编造根因或潜在收益。

## 复现

在 169 的现有 stage 容器运行 `audit_indexer_ep8_wait_tracks.py`，对原 DB 使用只读 SQLite，并核对保存的 SHA-256 `67d14f7f386658f8c8c5d3c9059dff9b0f1decb200eba621a3432fcf1e23de58`；结果在 `indexer-ep8-wait-tracks.json`。只读采集元数据，不访问请求、张量值或模型权重，没有启动 NPU。
