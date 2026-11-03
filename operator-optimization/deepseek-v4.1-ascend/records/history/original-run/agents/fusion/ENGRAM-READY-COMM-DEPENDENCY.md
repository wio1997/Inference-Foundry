# F24 跨层联合读取：通信和发布依赖核对

本次只读已有冻结源码与两份既有诊断结果，不启动 NPU、不改变实现。结论是**源码条件约束和既存记录的身份边界**，不是宣称历史运行已绑定以下每个分支。

## 各层所需的条件依赖

在 `model.py:1199–1226` 的 `participates` 分支，所有层先共同获得 `gather_engram_hashes` 的结果，然后逐层取 view。此上游 DP hash gather 在 shared memory 或无 DP group 时可以直接返回；其他分支补齐固定 slot 后做 DP all-gather（`engram/parallel.py:58–73`）。共同 producer 不意味各层 rows 已 ready。

非 direct 分支中，每层 `embed_gathered` 将 view contiguous 后 lookup，然后 **仅在各自条件成立时**执行 DP `all_to_all_single`、TP head `all_gather`；输出截断后，调用者将本层输出 copy 到固定 buffer、清零尾部，最后才 `ready_events[layer_id].record(current_stream)`。`output_buffers is None` 时没有固定 buffer copy/尾部清零，`ready_events is None` 时也没有显式逐层 event。direct 路径需要输出 buffer 且 `dp_size == tp_size == 1`，直接 lookup 与尾部清零后 record（`model.py:1207–1226`, `engram/embedding.py:333–355`, `engram/parallel.py:76–95`）。条件通信的 group/参与者、fixed-slot padding 与完整 head 顺序需保留，不能仅因两个 view 同一 wrapper 或一个 rank 命中而让该 rank 单独跳过 collective。`participates` 为假时，仍可能清零输出并记录每层 ready；因此识别到 record 也不足以证明真实查表或通信。

mask 独立发布：`model.py:1156–1161` 在需要时复制有效 mask、清零尾部并 record 自己的 event。`model.py:1535–1537,1569–1588` 按是否 SP 等路径在 mask 消费前等待；对应 layer 在读取 lookup 及可能的 SP shard 前等待自己的 ready。普通 event 用 `current_stream().wait_event`；外部 graph event 使用 `done.wait(stream)` 然后 `done.reset(stream)`（`embedding.py:358–365`）。不能将 mask ready、layer A ready 与 layer B ready 当作同一事件，也不能无条件删除外部事件的 reset。

在异步准备路径，`model.py:1462–1485` 先让准备流等待主流，并对输入及输出 buffer 调用 `record_stream` 后提交准备；`model.py:1319–1330` 退役时主流等待准备流，异常路径还能 reset 图事件。另一种 captured producer 路径 `model.py:1274–1317` 在 `finally` 把主流 join 到 capture stream，即使消费者跳过某层或发生异常。由此证明联合读取需要保护 producer 完成、每层数据发布及复用/异常依赖；不要求新方案沿用完全相同的事件 API，但要证明等价关系。调用返回不是设备完成，禁止仅凭相邻 Python 调用判断读操作已结束。

联合 kernel 如只有完成整个 kernel 后可发布的事件，可能把本已完成层的 ready 延后；源码本身没有证明历史关键路径受影响，也不证明任一底层 API 支持 kernel 中途对外安全发布。还需评估所用图模式、0 token、dummy 分支、各层独立 table 与输出、跨 rank 约定、buffer 版本及最终消费者。

## 历史记录：可以与不可以绑定

- `agents/diagnosis/ENGRAM-PAIRED-WRAPPER-AUDIT.md` 汇总同一 observer 窗口 826 个正常返回的配对调用中 168 对：同 process/thread 的相邻调用号、同 storage wrapper token、offset 0 后 24。它未保存对应 layer、table、ready 对象、设备地址或读完成事件，且日志采集有丢失；不同 wrapper token 也不证明不同底层存储。
- 另一次 EP8 profile 的 `ENGRAM-LOOKUP-SYNC.md` 只读已记录精确任务：所选第二/三次图窗口各有两条 host-UVA gather，六个层间区间都有其他流的 EVENT WAIT。按报告 TASK 字段无 event 对象 ID；已筛选 trace args 不能证明原始 args 无更多字段。它们不与上述 observer 是同一次采集；更不能按形状/时间把这些 gather、wait 和某层 ready 连接起来。
- producer 的已查询 flow 属 `HostToDevice` 下发关联，不是 event record→wait 或 buffer 消费边。图 connection ID 复用，不能唯一标识层、请求或 replay。并非所有 trace 被穷尽，故这些结论只描述已检查范围。

要建立实际事件链，需要同一运行、同 process/rank/graph replay 的 layer/table 和 buffer 版本绑定，以及可证明 record/wait 配对的 event 身份；还要绑定 collective group 与对应 tensor、各 stream 顺序和最后读取/复用。若历史材料不含这些字段，应保留 UNKNOWN，不用时间邻近推补；当前限制下不新增设备负载。

## 冻结输入完整性

源码 `model.py` SHA-256 `101f96b38e9fc47e7be147ddfe56976aee896867c73a33a7d95c3e3c833a6e91`；`engram/embedding.py` `a7741ed16522be4fe607e062e1c988c4e2435af7d258fe7db99f672d4508c4c9`；`engram/parallel.py` `8db3d5ca64cd5dfc8c2a61ab26dc2d403687475ffce81dfc2962c3df86fabdb1`。`engram-paired-wrapper-audit.json` SHA-256 `6161a02e9892c5e01ef7d94de6d9e07f49c5e0786d1290a2b7f9a6cb9af37090`，`engram-producer-events-exact.json` `ea31f7d421d10109a6cab2a14d13c084dda169761b4085177a658741821964cb`，`ep8-selected-flow-endpoints.json` `14e07d4311d679b26cc2262eaaa79652dc201b5fd55042b0f61045232245a8b2`。以上摘要沿用并引用既存分析，没有本次重新扫描完整 trace。
