# EP8 Indexer：六条 metadata 邻近 copy 的主机与设备关联

沿用 `INDEXER-EP8-METADATA-PRODUCER-SEARCH.md` 中三轮各两条、共六条 metadata 任务候选；**候选只依据目标 Indexer 之前的时间选择，尚未证明为目标的 metadata**。本次从这六条各自关联的 host metadata API 出发，选择其同主机线程之后最近的 `aten::copy_` 内的 `Enqueue@aclnnInplaceCopy`，逐条验证 enqueue、dequeue 与设备 copy 的记录关联。仅只读已有数据库和 trace，不运行 NPU。

## 可以确认的记录关联

六条选择中，metadata API 与随后 copy enqueue 在同一个 host `globalTid`；每个 enqueue 都被同线程的 `aclnnInplaceCopy` 和 `aten::copy_` 时间范围包含。每个 enqueue 经数据库 `CONNECTION_IDS` 的共享值找到唯一 `Dequeue@aclnnInplaceCopy`；每个 dequeue 同线程内有唯一被完整包含的 CANN `aclnnInplaceCopy`，该 CANN connectionId 又精确匹配一条设备 `MEMCPY_ASYNC` TASK。

| 目标内部轮 | metadata globalTaskId | copy host enqueue ID | copy 设备 taskId / stream | 设备 copy 开始减 metadata 结束 |
| --- | ---: | ---: | --- | ---: |
| 1 | 1800 | 93962 | 14823 / 37 | 20 ns |
| 1 | 1801 | 93965 | 14826 / 37 | 40 ns |
| 2 | 13298 | 111683 | 15058 / 37 | 20 ns |
| 2 | 13299 | 111686 | 15061 / 37 | 20 ns |
| 3 | 14477 | 113016 | 15118 / 37 | 20 ns |
| 3 | 14478 | 113019 | 15121 / 37 | 40 ns |

metadata 和对应候选 copy 均在 device 8、stream 37，且导出顺序相邻。**这支持一个具体可追查的 host copy→设备 copy 关联，也与冻结源码中 metadata 后紧接 `copy_` 相容；同一线程、同一流与 20/40 ns 邻近不是内存依赖证明。**这六条 copy 是按 metadata host 调用之后的最近候选选择，不是依据 tensor 地址或 payload 确认；尚不能证明它们读取对应 metadata 输出，也不能证明其目标缓冲就是 Indexer 的输入。

## 调用栈、数据字段和重放

本采集 `profiler_info_8.json` 的 `record_shapes=false`、`profile_memory=false`、`with_stack=false`。六条已选 host `aten::copy_`、`aclnnInplaceCopy`、enqueue 的 `callchainId`、`inputShapes`、`inputDtypes` 均为空。因此没有在该次采样中找到它们的 Python 逻辑源码调用栈，不能说历史 copy 直接出自 `_qli_metadata.copy_`；冻结源码仅提供可能的路径，且其历史加载身份还未绑定。

六条对应设备 copy 的原始 `trace_view.json` X 事件以设备时间精确匹配各一条；其完整顶层键为 `args,dur,name,ph,pid,tid,ts`，完整 `args` 键为 `Batch Id,Model Id,Physic Stream Id,Subtask Id,Task Id,Task Type,connection_id`。这些选中 X 行没有显式 src/dst 地址、tensor/storage ID 或 byte-count 键；该检查不排除其他未查文件/字段有间接关联。六条 copy 没有对应的 `COMPUTE_TASK_INFO` 输入/输出描述，因此这些记录不能确认 `[1024]` copy 或内容。目标 Indexer 是另一 stream 的 graph TASK；三轮复用 globalTaskId 和 connectionId，当前无证据把这些 copy 的输出跨图生命周期关联到指定 Indexer 重放。

昇腾 [CANN 9.1 profiler 表说明](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/devaids/Profiling/atlasprofiling_16_1145.html) 将 `CONNECTION_IDS` 定义为框架侧 API 和 CANN/task 的关联，并将 callchain 视为独立字段；这符合本次用法，不是 tensor 身份契约。本次采集自述 CANN 9.2，不能将 9.1 文档视为对历史 exporter 的版本级证明。相关 metadata 构造、`copy_` 和共享 buffer 的条件路径见 `INDEXER-EP8-METADATA-BUFFER-BINDING.md`。

**结论：**已从六个 host copy 候选关联到各自设备 copy TASK，但 producer→copy 的读写身份、copy→Indexer replay 的消费身份和有效 Q/K 值仍 UNKNOWN；不能把候选链作为实际因果链或性能根因。

## 复现

`audit_indexer_ep8_metadata_copy_chain.py` 对现有 DB 使用 SQLite 只读、校验 DB 和 `profiler_info_8.json` SHA，并流式读取 trace 只输出六个 X 事件的字段名；结果 `indexer-ep8-metadata-copy-chain.json`。输入 trace SHA-256 为 `3abc8f27420c733c39810ec86e9144634a074c22c5b9fcff0e3b39929b63dc99`，与此前阶段保存值一致，但该值本身不是外部独立采集来源证明。
