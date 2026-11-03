# EP7 两个 draft 候选：query 与 attention 记录

## 结论

一个先前未分类、没有导出 `SparseFlashMla` 的 EP7 窗口，确实记录了类似 KV 的投影、RoPE 和 cache scatter；另一个严格归属样本记录了 Q 投影候选及 `SparseFlashMla`。**当前不能判定前者的本地 query 是否为零、请求阶段是什么、它是否执行了其他形式的 attention，或者两者差异的原因。**两个样本的形状和工作量明显不同，时间差不是单变量性能对照。

原 162/288 是严格归属的选择性样本，126 条未通过该归属条件；本次只研究其中 EP7 一个有代表性但事先已选的未分类窗口，不能将此窗口结论推广到其余 125 条，也不能把未归属当作无计算。

## 选择、覆盖和证据

选择均可由已冻结资料复算：未分类样本重复 `tools/inspect_draft_missing_attention_v3.py` 对 EP7 的首个符合条件的选择，并核对第一对 HC 标记之间 3204.156 µs；严格样本取 `eager-draft-mainstream-attribution-v2/draft_call_metrics.json` 中 EP7 开始时间最早的一条，并核对四个 HC 标记。所有比较都在同一 rank、同一 device 时钟内，不依赖跨卡时钟。

| 项目 | 未分类：未导出 SparseFlashMla | 严格样本：已导出 SparseFlashMla |
|---|---|---|
| 对应 attention bracket | 3204.156 µs | 396.712 µs |
| HC 输入 shape | `[1,4,5120]` | `[2,4,5120]` |
| Q-like 路径 | 在此限定窗口所选类型中未见投影/量化/SparseFlashMla；未证明所有其他路径均不存在 | main stream 47 可见 `[1,5120] × [1280,5120]`、`[1,1280]` 动态量化、INT8 投影输出 `[1,32768]`、`SparseFlashMla` 输入 `[1,64,512]` |
| KV-like 路径 | main stream 47 可见 5 行投影、RoPE 和 cache scatter | stream 35 可见 15 行投影、RoPE 和 cache scatter |
| 其他选中行 | `ZerosLike [1,64,512]` | `ZerosLike [2,64,512]` |

“Q-like/KV-like”只是 shape 与冻结源码逻辑相符的角色候选；未捕获实际 tensor 地址或对象身份。以上差异还可能同时涉及序列数、缓存工作量、上下文和分支，不能把 `3204.156` 与 `396.712` 微秒的差值归到某一算子或调度。

29 个筛选的 CSV 算子行均与 TASK 的 model、stream、task ID 和开始/结束时间各找到一条精确匹配；每个行的 connection 均能找到一个 CANN 行，并通过 `CONNECTION_IDS` 找到一个候选 PyTorch 行。这支持这些记录的关联，不证明 tensor 消费边或 request 身份。29 个候选 PyTorch 行的 `sequenceNumber=-1`、`callchainId=None`；筛选后的 `input_shape_text` 都为 `None`，这可能是原字段缺失或未通过脚本的安全白名单，**不能断言原始字段无内容**。连接表可能是一对多，图重放还可能复用 ID，因此不以连接值单独当 replay 或 request ID。

## 与源码的关系、最小缺口

冻结 `attention/dsa_v41.py:350–355,701–713` 中，query 根据 `query_metadata.swa.num_actual_tokens` 截取，若本地 query 为空则可仅更新 cache 并跳过 attention 分支。`attention/context_parallel/dsa_v41_cp.py:189–192` 的一个 CP 实现将本地 query 和全局 cache 分开传入；`models/deepseek_v41/dspark.py:102–123` 通过 `get_v41_cp_classes()` 选择实现，而该选择取决于运行配置。这些逻辑使“query 少或零、cache 仍要更新”成为**可检验假说**，但既有窗口的对象类型、每次执行分支、query metadata 值和 tensor 身份均未绑定，所以源码不能为该窗口的零 query 作证明。

要直接分辨这两个窗口，最小且有用的关联是同一次调用上的匿名 batch/replay 标识、实际 backend 类和分支、`query_metadata.swa.num_actual_tokens` 与 `cache_metadata.swa.num_actual_tokens`、所属 target/draft 层及采样阶段，再与准确 task 范围绑定；仅靠 shape、算子名或全局 benchmark 负载无法补足。下一步优先核查已保存的服务/runner 日志或原始 profiler 标记是否确实包含上述数值和调用标识；如果没有，即保持 UNKNOWN，不以增加无目标的扫描或新 NPU 负载替代证据。若后续经授权补采，仅需这些匿名标量与标识，无需记录 token 或提示词。

### stage130：已索引日志和 EP7 profiler 的定向字段检查

- 本地保存的 `benchmark-20261028/raw-results/service-log-tail.txt`（SHA-256 `64bfa3d8523ad86e1c466fcca166cdcfe1dc4a3676573fc51b65279a69aa8aba`）固定词检查 `num_actual_tokens`、`cp_token_range`、`query_metadata`、EP7 PID `4869`、三种 `AscendDSAV41...Impl` 类名均为 0 次；该 benchmark 日志不是已确认完整的这次 profiler 诊断日志，不能推广为全部运行日志无记录。
- 查询了冻结 SHA 匹配的 EP7 `ascend_pytorch_profiler_7.db` 和 `profiler_info_7.json`，并核对 stage127 两个窗口输入 SHA。对八个固定名称 `num_actual_tokens`、`cp_token_range`、`query_metadata`、`cache_metadata`、`AscendDSAV41CPImpl`、`AscendDSAV41PCPImpl`、`AscendDSAV41Impl`、`batch_id`：数据库表/列名与 `STRING_IDS` 均未命中；`META_DATA` 共 8 行，key 名没有命中；profiler-info 的相关字段路径也没有命中（其数组最多检查前 100 项）。因此本次固定词检查没有可绑定到两个窗口的候选标签。profiler-info 是 session 信息，不是自动的逐调用记录。
- 这些检查只针对上述文件、名称及输出范围；其他名称、编码、raw trace、未检查日志或运行时对象仍可能包含信息。不能断言系统没有保存，不能将零命中解释为零 query、零 cache 或没有运行 attention。stage130 未检索请求原文，也未执行新负载。

stage130 结果 [draft-ep7-profiler-metadata-inventory.json](draft-ep7-profiler-metadata-inventory.json) SHA-256 `6e074a944ca740b63168929d1ba56a4062bf199028bf137632bea8470d06677a`，固定查询脚本 [query_ep7_metadata_inventory.py](query_ep7_metadata_inventory.py) SHA-256 `cb611af3c975385e5931409ae43640950a7ccc14bef18d01aa4e679af9ed73f2`。`operator-ceiling169-v2-130-ep7-meta` 退出码 `0`、`model_verified=true`、模型 `deepseek-flash`。

### stage131：同一未分类样本内的长等待候选

使用 stage127 唯一的 main stream 47 `ZerosLike [1,64,512]` 结束和后续 `MatMulV2 [5,1024;5120,1024]` 开始作为两端，固定同一 EP7 的半开区间为 `1793152959144368248–1793152959146620883 ns`，时长 **2252.635 µs**。这只是所选样本的窗口，并非整个 block 或请求。

区间内 DB TASK 的 stream 47 有一条标记为 `EVENT WAIT` 的记录，持续 **2199.616 µs**，占该区间约 **97.65%**。同时 CSV 在 stream 39 记录了一个 `hcom_alltoall_`，持续 **2197.576 µs**、完全落在该 wait 的时间范围内；DB 中同时间 stream 39 有 `KERNEL_AIVEC` task。stream 47 随后执行一条 `TransposeBatchMatMul`，持续 52.899 µs。

这是明确的同卡时间重叠和等待标签，但 TASK 未提供配对 event 对象身份，因此不能确定主流等的就是该通信任务，也不能把 `hcom_alltoall_` 全部持续时间当作网络传输时间。其他进程、底层实现、排队和采集语义尚未排除。只选了一个样本，不代表所有 126 条未归属样本或端到端耗时。

CSV 所有数字 stream 行的区间并集为 **2250.475 µs**，`N/A` stream 行并集为 **2197.576 µs**，所有 CSV 行并集仍为 **2250.475 µs**；该 N/A 行与 stream 39 的通信行时间相同，不能把两者加总。未被这些 CSV 行覆盖的 **2.160 µs** 是导出行的时间补集，不是证明硬件空闲。DB 共找到 8 个时间相交 task；record/wait 与其他流均只表示本次导出的记录，不证明采集完整或依赖边。

[draft-ep7-long-interval.json](draft-ep7-long-interval.json) SHA-256 `575eaa12caf97e612615e8af27069b54f4ab761d1c205a87d104aad656e8fedb`；[查询脚本](query_draft_ep7_long_interval.py) SHA-256 `6bdee8592cd9ecf2321442256ae7c2395b37a68975bc572f6698ae1bf14ba68e`。`operator-ceiling169-v2-131-draft-gap` 退出码 `0`、`model_verified=true`、模型 `deepseek-flash`。接下来若已有证据能够给出这条 wait 对象及 record 对象的身份，可继续检查它们是否配对；否则保留条件假说，而不是以相邻时序作为依赖证明。

### stage132：event 配对及通信操作关联

对该区间预选的 6 个 TASK，全部在 `trace_view.json` 找到时间、stream 和 task ID 精确匹配的记录，其中 5 条 event 记录的名称也匹配。主流 `EVENT WAIT` 与两条 stream 39 的 `EVENT RECORD` 分别以自己的 connection 命中名称分类为 `event/wait` 或 `event/record` 的 CANN API；筛选后的 trace 标识字段仅显示 Model、Task、Stream、Batch、Subtask 和 connection，未显示 event 对象 ID。脚本没有输出或检查未筛选的全部原始参数，不能推断它们必定没有 event handle。附近 flow 中有 `HostToDevice` 类，但这类下发关联不能当作 event 对象配对。**仍不能证明这条主流 wait 等的就是通信完成。**

对 `hcom_alltoall_` 候选，stream 39 的 TASK、CSV、`COMMUNICATION_OP` 在开始、结束和 connection 上吻合；`COMMUNICATION_TASK_INFO` 通过 globalTaskId 命中，同一条的 `opId=402` 与 `COMMUNICATION_OP` 一致，记录的 `rankSize=8`。这个多字段关联强烈支持它是导出的通信操作；对应 PyTorch connection-set 的 API 名称筛选结果包含 `alltoall`。globalTaskId、connection 的复用和格式语义仍需考虑，`COMMUNICATION_OP.count=4096`、task `size=8192` 只是未核实语义的字段，不能当作实际链路字节或带宽；**2197.576 µs 是该条导出通信任务的持续时间，不是已经测得的纯网络传输时间。**

在这个具体样本里，“主流长 wait”和“另一流长通信任务”是同一时窗的清楚观测，实际 event 对象配对、其先后因果和请求级暴露仍为 UNKNOWN；不能以它们时间重叠就断言已经找到全系统根因。

[draft-ep7-event-pair.json](draft-ep7-event-pair.json) SHA-256 `241fe135799eee650b46681606cca64f94ba0e510210637914ff3aaa88fc5901`，读取 EP7 trace 的首次记录 SHA-256 为 `4a3788d6e127941259ee7a7964675fb055edcbc1e8fa58e7c9fca20f6805d447`，此前只冻结了路径及文件大小，非独立的原始内容哈希比对。查询脚本 [query_draft_ep7_event_pair.py](query_draft_ep7_event_pair.py) SHA-256 `68cf23139f5ef14c557f925bf63b8951c97ad72dc95f59dbe92ad0139386cfd7`。`operator-ceiling169-v2-132-draft-pair` 退出码 `0`、`model_verified=true`、模型 `deepseek-flash`，无新 NPU 负载。

### stage135：16 rank 同规则候选的分布

复用既有的两个 target 图执行间隔、MoE shape 和四个 HC marker 规则，16 rank 共重现 **288 个候选 bracket**，与旧 rank 审计计数一致，`issues=0`。其中 162 个 bracket 恰有一条导出的 `SparseFlashMla`，126 个没有；该分类仍不等于证明 backend 分支、请求阶段或真实 query 数。每个 rank 的时间只在本卡计算，**没有比较跨卡绝对时间**。

按候选 attention bracket 内完全包含的数字 stream `hcom_alltoall_` 选取：216 个 bracket 各选出一条，72 个没有选到；未选到不代表没有通信、其他通信路径或窗口外工作。216 条都在对应卡的 DB 中各自精确匹配一条 TASK 和一条 `COMMUNICATION_OP`。其中 213 条与主流 `EVENT WAIT` 在同卡有时间重叠，另外三条没有选到重叠的主流 `EVENT WAIT`；这不证明配对或因果。

下表只统计上述 216 条已选出的 CSV 任务，`M` 指 HC 输入 shape 的首维；`count` 是通信导出原字段，不解释为真实链路字节。间隔 1 是 target 图采样轮 1→2 之间，间隔 2 是轮 2→3 之间，不是请求轮次或草稿层号。

| 图间隔 | HC M | 导出 SMLA 行数 | count / rankSize | 任务数（rank 数） | 耗时范围 / 中位数（µs） |
|---|---:|---:|---:|---:|---:|
| 1 | 1 | 0 | 4096 / 8 | 9（3） | 2055.179–2198.696 / 2077.118 |
| 1 | 1 | 1 | 4096 / 8 | 15（5） | 8.540–2063.779 / 1929.440 |
| 1 | 2 | 0 | 8192 / 8 | 33（8） | 139.457–155.697 / 150.220 |
| 1 | 2 | 1 | 8192 / 8 | 63（13） | 8.339–24.240 / 13.520 |
| 1 | 3 | 0 | 12288 / 8 | 6（2） | 147.137–154.077 / 151.147 |
| 1 | 3 | 1 | 12288 / 8 | 42（14） | 9.200–22.220 / 14.600 |
| 2 | 3 | 0 | 12288 / 8 | 6（2） | 128.678–148.297 / 141.727 |
| 2 | 3 | 1 | 12288 / 8 | 42（14） | 8.760–23.979 / 15.009 |

stage131 的 2197.576 µs 属于间隔 1、M=1、无导出 SMLA 的组；EP7 同组的三条分别为 **2064.359、2076.418、2197.576 µs**。合并同间隔、同 M、同 count/rankSize 的两个 attention 分类后，24 条中 **21 条超过 1 ms**，分布在 EP0、2、3、4、5、6、7；EP1 的三条为 8.540–9.280 µs。同一组中 21 条有主流 wait 重叠，其中 19 条覆盖整段通信任务、2 条部分覆盖；3 条没有选到重叠 wait。由此可以说这个长时长在所选采样中并非单次现象；**不能认定这些 rank 是同一次 collective、对齐同一请求，或是某条链路/对端导致延迟。**

分组之间的 shape、attention 记录和 workload 可能同时变化；各样本来自同一 profiler 采集而非独立重复实验。上表既不是纯网络时间分布，也不代表全部推理请求，更不能作为可实现收益或整机性能瓶颈比例。

[16rank-draft-alltoall-distribution.json](16rank-draft-alltoall-distribution.json) SHA-256 `633011028e6d5953f03f1493c197085bdd1a810c095cc646e4f5b0e115f61ec6`，内附 16 份输入 CSV/DB 的冻结 SHA 和逐条匹配；查询脚本 [query_16rank_draft_alltoall_distribution.py](query_16rank_draft_alltoall_distribution.py) SHA-256 `8b0ef53eb5ac5ee5a734d2c37952dd721e23097a276401bec2a6fc4b6e1b4859`。`operator-ceiling169-v2-135-draft-16rank` 退出码 `0`、`model_verified=true`、模型 `deepseek-flash`，无新 NPU 负载。

### stage137–138：24 条的通信组与局部子任务身份

对上述 M=1、图间隔 1、`count=4096`、`rankSize=8` 的 24 条，按每张卡自己的 DB 精确查询操作和关联子任务；未使用跨卡绝对时间。

- 24 条的导出 `groupName` 字符串哈希相同；导出 `opName` 有 3 个哈希，每个各 8 条，恰分别覆盖 EP0–EP7 的各一条。同时，每张卡本地 `opId` 都出现 `402/405/408`，并分别与三个名称哈希对应。24 条的操作 `dataType` 枚举为 `BFP16`，`count` 字段 4096、`rankSize` 8，算法标签哈希相同。这是**同名同组候选**，不是已证明的同一次 collective、相同请求或跨进程序号。
- 每个本地 `opId` 只关联到 **1 条 `COMMUNICATION_TASK_INFO`**，均 `isMaster=1`，其 `globalTaskId` 与所选任务一致；没有在这些 `opId` 下找到可供分解的非 master 子任务。24 条 master 的 `size` 字段均为 8192、`transportType/linkType/rdmaType/dataType` 字段均为 65535；含义和计数范围未核验，不能将其当作物理链路字节、带宽或等待阶段。`notifyId` 的导出值哈希相同也不是已证的跨 rank 或 record/wait 对象身份。
- 同镜像两个已索引 profiler 分析源码根共检查 849 个 `.py/.sql`，跳过 0 个。5 个匹配文件包括常量、通信 JSON 消费、step-time DB 消费和 C++ 数据结构引用；其中通用变量名也会匹配，摘录没有给出 `groupName/opName/opId/notifyId` 的跨 rank 唯一性或生成语义。编译 exporter、其他路径或不同拼写不在本次搜索范围；冻结 MindStudio 文档的按通信域展示说明也不能替代本版本身份契约。

要把三组 8 条提升为同次 collective，仍需要同版本 exporter 或 HCCL 的 group 身份及操作序号生成/复用契约，或已保存的可信调用标识和成员映射。要定位慢在何处，还需要该次操作的可关联子阶段/链路数据、等待定义和采集覆盖；当前这 24 条只有 master 行。**不能因 EP1 短而断定它晚到，或声称其他卡等待它。**

stage137 [communication-exporter-source-inventory.json](communication-exporter-source-inventory.json) SHA-256 `cdd7bcc669fed6a2807085a605c67604ae18c48e598b77c24c1a7ac40debd4f7`；stage138 [24-draft-alltoall-identity.json](24-draft-alltoall-identity.json) SHA-256 `bc572249e448c5470d3a9c59a08fa6f872c7e9674b78b37e33e926bad1ddf9ef`，查询脚本 [query_24_draft_alltoall_identity.py](query_24_draft_alltoall_identity.py) SHA-256 `7e37a0a0e6c73ffc6fa10a5aebd99f4982cf3f751ebf31bcdf1d5526f77216d2`。`operator-ceiling169-v2-137-hccl-exporter` 和 `operator-ceiling169-v2-138-hccl-ids` 退出码均为 `0`、`model_verified=true`、模型为 `deepseek-flash`；均为只读检查，没有新 NPU 负载。

### stage139–140：通信 JSON 后缀与零值的解释边界

8 个已索引的 `communication.json` 中，在按深度/数组上限限定的检索范围内，24 个 DB `opName` 的名称前缀各匹配一条 JSON 键，输出无截断；不能仅凭名称证明同次操作或跨 rank collective。每个键的 `@` 后缀都是同一个 19 位十进制文本，其字符串与 DB `groupName` 不同，也不字面等于该条的 EP、local rank、stream 或本地 `opId`。**不同字符串不证明不同通信组**，可能存在尚未核验的编码转换。

24 条候选 JSON 的 `Start Timestamp(us)` 数值与各自同卡 TASK 起点相差约 −221 至 +234 ns，`Elapse Time(ms)` 数值与该条 CSV 任务持续时间一致；这是强同卡关联线索。脚本通过 Python 浮点解析大数微秒时间戳，可能产生亚微秒舍入，且尚未证明导出时钟/复用契约；不能拿这点差异作为测量精度、跨卡同步或事件先后证据。

每条 JSON 候选均记录 `Wait Time(ms)=0`、`Synchronization Time(ms)=0`、已筛选的 `Transit Time(ms)=0`，而 `Idle Time(ms)` 数值等于对应候选的总持续时间。官方 [MindStudio 通信指标说明](https://www.hiascend.com/document/detail/zh/mindstudio/70RC2/msinsightug/msascendinsightug/AscendInsight_0023.html) 将 Idle 定义为通信总时长减传输和等待，界面称通信算子下发耗时；它不是芯片空闲率。当前尚未确认这份文档与本次导出器版本完全一致，也没找到证据区分零值是实际值还是子阶段缺失/未解析，因此不能断言没有网络传输、真实等待为零或已定位 CPU 下发。

EP7 保存的 `profiler_info` 显示 `Level1`、CPU 和 NPU 活动，`_sys_interconnection=false`、`_l2_cache=false`；这些是 session 配置记录，不足以单独判断这条 HCCL AIV 候选的内部传输或同步事件是否被采集。固定词检查 849 个同镜像 `.py/.sql` 文件发现通信字段的常量、消费者和部分聚合逻辑，但没有证明本次零值的缺失/降级原因；编译 exporter 及其他路径未覆盖。等待、传输和数据来源仍为 UNKNOWN。

[24-draft-hccl-json-details.json](24-draft-hccl-json-details.json) SHA-256 `2d56c4ebb8d83dc43d3cb6be57257c064e71756a3913d8a41ae8fccba895b61c`；[hccl-suffix-and-zero-semantics.json](hccl-suffix-and-zero-semantics.json) SHA-256 `8128d94ac85c43248f56f50c22a92341d2afdb63b4e605edcad84b76cb10857c`。stage139、stage140 均通过 ZCode `deepseek-flash` 只读执行，具体 exit-code 与模型验证由各 stage 的本地执行凭据记录。

### stage141：同镜像 analyzer 的组名与 Idle 定义

同镜像 CANN 源 `analyzer/communication_analyzer.py:93–98` 把来自 HCCL 数据库的 `op_task.op_name + '@' + op_task.group_name` 作为 analyzer 的键；若它就是本次 JSON 的生成路径，则 `@` 后缀表示该 analyzer 输入中的 `group_name`，并非直接的 rank、stream 或本地 opId。**尚未证明它与 `COMMUNICATION_OP.groupName` 采用相同编码，也没有确定这一标签唯一识别一次 collective。**该源码调用 `merge_op_task`；对应 `meta_parser.py:42–51` 会对相同键合并任务并取最早开始、最晚结束，所以同名和接近的起点不能排除聚合或复用。

同镜像 `msmodel/cluster_info/communication_model.py:77–98` 中，原始 HCCL 操作以 `(connection_id, iter_id)` 建立本地映射、任务按 `(op_id, iter_id)` 查找；这为**同一 HCCL 数据库内部**可检验的关联提供了具体入口，不自动使不同卡的数字或同一名字成为全局唯一 ID。

同镜像 `msparser/cluster/communication_parser.py:216–235` 将已分类的 transit 和 wait 区间合并、扣除重叠，按其区间并集计算 Idle。若本次 JSON 确实由这套路径生成，缺少或没有被该解析器归类的区间都可能使指标显示为零、Idle 等于总时间；目前尚没有证明哪一种发生，更不能把 Idle 数值直接认定为 CPU 下发耗时或芯片闲置。后续只读取其 task 分类函数和 raw HCCL 数据库的既定字段，才能判断已记录任务是否满足该分析器条件；如果原始数据或版本契约缺失则保持 UNKNOWN。

[hccl-analyzer-semantics.json](hccl-analyzer-semantics.json) SHA-256 `879c1caf8debdedd2b4be8192f45a6f226327cbf538853605750e4b0f53c66d7`，包含同镜像文件路径、SHA 和行号；仅为安装源码证据，尚不证明本次采集运行了该 analyzer 版本。stage141 退出码为 `0`、模型验证为 `deepseek-flash`，没有新 NPU 负载。

### stage142：通信时间分类的输入边界

安装源码 `communication_parser.py:180–262` 首先筛选 analyzer 输入中 `is_master=1` 的事件，按 plane 处理；传输区间来自其 SDMA、UB 判别函数或满足处理条件的 RDMA payload，等待区间来自 `Notify_Wait`，并在有窗口时裁剪及去重。源代码定义的 Idle 是操作持续时间减去这些**分类后**区间的并集。因此，若当前 JSON 实际由此版本路径生成，相关分类的有效区间为零即可得到 Idle 等于时长；单看这一结果不能区分源事件不存在、未被采到、未满足分类条件或落在窗口外，也不能证明真实网络传输或设备等待为零。

`communication_model.py:40–98` 显示该 analyzer 的输入是 HCCL 单设备数据库中的 HCCL/KFC 操作和任务，局部以 `(connection_id, iter_id)` 对应 `(op_id, iter_id)`。stage138 的单个 master 来自 `ascend_pytorch_profiler_*.db` 导出的 `COMMUNICATION_TASK_INFO`，**尚无证据说明它完整等同 analyzer 所读数据库中的原始任务**；故不能据此推算原始子任务数量或零值原因，`KERNEL_AIVEC` 标签也不足以确定通信实现及采集覆盖。各数据库、JSON 的生成链及版本绑定仍需验证。

[hccl-plane-sections.json](hccl-plane-sections.json) SHA-256 `8c2608a7833f51b9b33270526f427fc50fe0ba58f2e7c73cf3c26ebc658f6037`，含源码固定行号及哈希。stage142 退出码 0、模型验证为 `deepseek-flash`，无新 NPU 负载。已索引的 `used-raw-source-sha256.json` 是使用过的文件清单，不是原始 host 目录完整清单，未列出 `hccl.db` 不能说明它不存在；下一步仅在已知的 EP0–7 host 目录做有界只读清点，若找到再核对来源并限定 24 个候选的查询，否则记录路径与覆盖范围。

### stage143：8 个已索引 host 目录的原始数据库清点

固定的 EP0–7 的 PROF/host 目录各通过 `start_info` SHA 校验。按不跟随符号链接、目录深度至多 4 的规则，各检查 71–73 个条目，未发现大小写不敏感名称 `hccl.db`；各目录均无错误或被跳过的目录。结论仅限这 8 个 host 路径和该文件名，不能排除其他位置、其他名称、其他格式的原始通信数据，也不能证明原始事件从未采集。当前没有发现能与这 24 个候选安全绑定的 raw HCCL 数据库，因此继续凭导出 SQLite 解释子阶段没有依据，零值原因保持 UNKNOWN。

[hccl-host-db-inventory.json](hccl-host-db-inventory.json) SHA-256 `554cfb131218778a5472202ef0265614aa69b997d3d2f481d3fb6f9dc3fba860`；脚本 [inventory_hccl_host_db.py](inventory_hccl_host_db.py) SHA-256 `0ee04351c4f7afbc60af55421c867d1151364753f8f29b8ef2ce1a1f7be752b5`，输入清单 SHA-256 `cd5bac3b56c51fa2d7ce13199523ff9f88b94c4edf4c74131d4aedbdf2fc7c2a`。stage143 退出码 0、模型验证为 `deepseek-flash`，没有新 NPU 负载。除非出现新的明确来源索引，否则不再扩散目录搜索。

### 已保存数据补充：六个通信窗口与严格草稿 MC2 的同卡关联

用 stage135 的 `rank + communication globalTaskId` 找到原 attention 括号，再以**同 rank + bracket_start_ns**与严格草稿窗口、MC2 摘要分别一对一精确匹配；并核对草稿窗口结束时间及主流 ID。两来源均保留各自 rank 的纳秒整数时间戳，转微秒只除以 1000；只在每张卡本地排序，不比较跨卡时钟。六窗均满足 `窗口开始 ≤ alltoall 开始 < alltoall 结束 ≤ MC2 开始 < MC2 结束 ≤ 窗口结束`，因此所列两段在各自窗口不重叠。

| 卡 / 本地 opId | 整窗 µs | 开始→通信开始 µs | 通信 µs | 通信结束→MC2 开始 µs | MC2 µs | MC2 结束→窗结束 µs |
|---|---:|---:|---:|---:|---:|---:|
| EP0 / 402 | 4596.968 | 663.247 | 2060.259 | 538.769 | 1248.315 | 86.378 |
| EP0 / 405 | 4436.912 | 709.346 | 1925.442 | 494.590 | 1223.875 | 83.659 |
| EP0 / 408 | 4486.290 | 706.805 | 1941.302 | 518.769 | 1229.956 | 89.458 |
| EP1 / 402 | 4565.909 | 2421.452 | 8.540 | 1645.187 | 139.637 | 351.093 |
| EP1 / 405 | 4360.512 | 2289.694 | 8.860 | 1584.288 | 131.957 | 345.713 |
| EP1 / 408 | 4387.392 | 2299.294 | 9.280 | 1605.788 | 140.717 | 332.313 |

五段按本卡时间相加恰好为整窗。这里的另外三段是**时间范围，不是空闲或等待**，实际包含已导出的任务，也可能包含未导出的活动。每张卡内部通信和 MC2 先后只是时序观察，不证明它们的数据或事件依赖；两卡记录虽形状和名称相容，但未证明对应同一请求或同次 collective，无法得出跨卡因果、通信导致 MC2 慢或潜在端到端收益。MC2 周期代理不是 FLOPs、纯计算时间或内部等待；动态专家路由量、访存流量及内部阶段仍未知。特别是 EP1 通信结束到 MC2 开始仍有约 1.58–1.65 ms，不能只比较通信和 MC2 来解释整窗。

复算 `python3 agents/diagnosis/recheck_draft_alltoall_mc2.py`；输出 [draft-six-alltoall-mc2-local-join.json](draft-six-alltoall-mc2-local-join.json)，SHA-256 `1dee9a2dcebaa852c83fdfb8ec03f1d328c54cea4e59c57e21cdc42cc01c6e63`，包含四份输入和脚本哈希及逐窗原始纳秒。仅复用本地已聚合历史数据，没有读取远端或启动 NPU。

### stage144：EP0 与 EP1 六个固定同名候选的本卡前后任务

限定 EP0、EP1 的本地 opId 402、405、408，使用 stage135/138 固定样本及各自 CSV/DB 哈希，仅按每张卡自己的时间顺序查看。两卡各三条的本地名称哈希、外层 HcPre 和 MC2 输入形状相同；六窗均记录一次 `SparseFlashMla`（输出 BF16 `[1,64,512]`）、其后同形 RoPE，以及数值一致的通信 count 字段 4096 / rankSize 8。EP0 这三条通信任务分别为 **2060.259、1925.442、1941.302 µs**，同卡主流各有一条 wait 覆盖完整通信任务；EP1 三条分别为 **8.540、8.860、9.280 µs**，未见与之重叠的所选主流 wait。EP1 在通信开始之后约 39.379–59.479 µs 还有 9.880–11.980 µs 的主流 wait，其对象与原因未知。

六个窗口的主流均在通信任务结束后记录 `TransposeBatchMatMul`（`[5,1,4096] → [5,1,1024]`）及 `MatMulV2`（`[5,1024] → [5,5120]`），并在投影之后看到 `ZerosLike [3,5120]` 和 `ConcatD`。**这六个窗口没有记录 `ZerosLike [1,64,512]`**：那个 zero-padding 观察来自不同的无导出 attention 样本，不能混到这里。同镜像 `restore_tp_heads` 以 TP8 将 `[1,64,512]` 切分时每 peer 对应 4096 元素，与 count 字段一致，为待核实的候选路径；没有 tensor 地址、通信接口实例及 group/count 定义的绑定，不能认定该 alltoall 一定属于该函数，也不能仅凭 shape 和时间邻近认定通信与前后算子的生产消费依赖。空 query 源码路径是否执行 padded all-to-all 是源码条件问题，不能从此六个有 attention 导出的样本得出空 query 身份。

EP1 各自本卡从选中 HcPre 起点到通信起点的间隔约 **2289.694–2421.452 µs**，EP0 为 **663.247–709.346 µs**；计算的是每卡内部区间长度，未对不同卡的绝对时间戳做差。EP1 的这些窗口内有较大未由所选 main stream 的 kernel 行覆盖的时段，但 CSV 和事件覆盖不全、还有其他 stream/任务，不能把它当设备空闲、下发延迟或等待对端的证明，更不能因此推断跨卡先后顺序。

两份 `communication_matrix.json` 在固定名称哈希、限定深度/数组的检查内，对六条候选均无键匹配；这只说明查询没有找到可直接按该名称关联的矩阵项，不代表文件无其他汇总、不同命名的数据或真实链路通信不存在。两文件路径和大小匹配此前清单，本次首次记录内容哈希，不能声称与采集时的内容哈希一致。现有结果没有区分 2 ms 的发生原因，不宜把同名当作同一次 collective 或断言对端、网络及可节省耗时。

[ep01-fixed-alltoall-context.json](ep01-fixed-alltoall-context.json) SHA-256 `ff053235cbc854b7586018ff63111b6a9b3c944d32e6b40c8c75f5de5cc8c9a5`，包含各窗口任务、wait、输入清单和检查限制。stage144 退出码为 0、模型验证为 `deepseek-flash`，无新 NPU 负载。

### stage145：EP1 的通信前同卡主流和其他流

以 EP1 本地 opId 402、405、408 各自的 HcPre 起点和 alltoall 起点为固定半开区间，时长分别 **2421.452、2289.694、2299.294 µs**。只对同一设备时间轴计算，DB TASK 包含所选区间内零时长事件；CSV 与 DB 仅是导出记录，不构成完整设备执行证明。

| 本地 opId | 主流 TASK 正时长并集 | 其他流 TASK 正时长并集 | 主流最长正时长 TASK 缺口 | 该缺口内其他流 TASK 并集 |
|---|---:|---:|---:|---:|
| 402 | 190.077 µs | 42.677 µs | 420.352 µs | 18.820 µs |
| 405 | 200.955 µs | 38.420 µs | 590.368 µs | 0.040 µs |
| 408 | 215.737 µs | 41.420 µs | 425.492 µs | 18.000 µs |

对 TASK 时间独立裁剪并重算并集及缺口，均与输出一致。导出表存在同 taskId 的 `KERNEL_MIX_AIC` 内外重叠记录，例如 op402 的 Q 投影候选有约 42.979 和 41.119 µs 两条；它们通过区间并集去重，不能把两条时间相加。主流导出的最长单任务候选是 attention 的外层 MIX 记录约 **71.479–74.778 µs**，其内部对应 CSV 的 `SparseFlashMla` 为 **69.159–72.798 µs**；Q 投影同样存在内外记录。这样说明这三窗中的大段时间没有被这些已导出的主流正时长 TASK 覆盖，但不能推导整卡低利用率或 CPU 下发耗时。

最长缺口中的其他流只覆盖表中所示的小段，其中 op405 的 590.368 µs 缺口内部还有一个主流 **零时长 `EVENT RECORD`**；所以“缺口”精确定义为不被主流正时长 TASK 区间覆盖，并不意味着没有事件。其他流 event 与数个 AIC/AIV task 的时间重叠，不自动构成该主流的依赖；非数字 stream 的 CSV 行可能是重复或汇总，不能与数字流并集相加。profiler 遗漏、设备内部执行、主机调用与同步配对均未排除，不能称这些间隙为设备空闲、等待对端或调度开销。

下一项可证伪检查：仅选上述三个最长缺口前后边界及内部零时长 event，查同 connection 的 CANN/PyTorch API 和 trace 的固定候选 flow，验证是否有唯一的已采集关联。连接复用、flow 编码及两端时钟未确认时，结果仍不能计算主机到设备延迟或建立依赖；若没有关联，不等于没有提交。

[ep1-attention-lead-intervals.json](ep1-attention-lead-intervals.json) SHA-256 `3f98642af4e6dc3ad5c8582d30de5e6600685bb766ec1bfe1958110a0dd06326`；stage145 退出码 0、模型验证为 `deepseek-flash`，无新 NPU 负载。

## 复核

[draft-query-metadata-ep7.json](draft-query-metadata-ep7.json) SHA-256 `91bd06a12087247de6265ae5c5db915af8f86973d9ff78089b22e973b9545cf0`，内附两个原始文件和两个窗口输入 SHA；[查询脚本](query_draft_query_metadata_ep7.py) SHA-256 `edc2713b8d74748eda3c507e1211562ba554f22362fd0c5d71b45899aa202a7d`。本机 ZCode `operator-ceiling169-v2-127-draft-ep7` 退出码为 0、`model_verified=true`、模型 `deepseek-flash`；只读取 CPU staging 容器已有数据，没有启动 NPU 负载。

本机 ZCode stage `operator-ceiling169-v2-184-draft-mc2-join` 已对六窗关联脚本重新计算并与保存 JSON 逐字节比对：exit 0、`model_verified=true`（DeepSeek）；未访问设备。

### stage185：六个草稿 MC2 导出字段与等待字段语义

ZCode 只读同两张卡已冻结 `kernel_details.csv`，先验证各文件 SHA，再按 rank、stream、task ID、纳秒起止和类型匹配所选 6 条唯一记录；AIC 计数分子与分母逐条核对此前严格草稿记录。6 条的输入／输出 shape、dtype、format、Block Num=24 和 Mix Block Num=48 都相同，输入为 `"1,5120;8,5120,4608;8,2304,5120;1,3;36864;40960;1,3;1"`，但形状一致不能证明实际有效 token、各专家接收行、工作量、缓存状态、代码实例或同次调用相同。

| rank / 本地通信 opId | MC2时长 (µs) | CSV Wait Time (µs) | AIC MAC ratio | AIC MTE2 ratio | AIV vector ratio | AIV MTE2 ratio |
|---|---:|---:|---:|---:|---:|---:|
| EP0 / 402 | 1248.315 | 1.230 | 0.002 | 0.020 | 0.000 | 0.021 |
| EP0 / 405 | 1223.875 | 1.540 | 0.003 | 0.019 | 0.000 | 0.021 |
| EP0 / 408 | 1229.956 | 1.630 | 0.003 | 0.020 | 0.000 | 0.022 |
| EP1 / 402 | 139.637 | 229.381 | 0.033 | 0.270 | 0.002 | 0.036 |
| EP1 / 405 | 131.957 | 216.000 | 0.000 | 0.001 | 0.001 | 0.058 |
| EP1 / 408 | 140.717 | 221.330 | 0.032 | 0.243 | 0.002 | 0.035 |

ratio 是原始导出小数，例如 0.020 即 2%；小数精度有限，0.000 不证明无活动。这些 ratio 是各指标的管线代理，并非物理计算次数、实际流量、空闲率、内部阶段占比或可直接相加的耗时；任务与参数计数的复用、硬件和 exporter 语义仍未完全核实。特别地 EP1 的三条 `Wait Time` 均**大于各自 MC2 总时长**，所以不能把它们认作该任务内部某个等待阶段的时长。已存 [Ascend MindStudio Insight 算子详情字段说明](https://github.com/Ascend/msinsight/blob/efb28234772093b0cabc3c538e71375c957540b2/docs/zh/user_guide/system_tuning.md) 第 1782 行将同名 `Wait Time(μs)` 描述为上一任务结束与当前任务开始的间隔；当前采集 exporter 与文档版本、具体“上一个任务”的选择规则尚未绑定，不能以这一定义断言三个数字具体代表哪个先前任务，也不能计算通信等待或宣称整卡空闲。

EP1 405 的 MC2 导出 MAC 和 MTE2 ratio 显著低于 EP1 另两条，但持续时间相近；这进一步说明仅靠外层时长无法识别内部工作。两卡不是被证明的同次请求或同工作量配对实验，也不能将 EP0/EP1 的差异归因于通信或访存；真实专家计数、阶段时间和张量身份仍缺失。

结果 `draft-six-mc2-exported-fields.json` SHA-256 `0a703b6028f1ffbace5e085a48714f07e05ff914189a199d0cdde1d374bd6568`，源固定摘要以及两份 CSV SHA 均保存在结果中；查询脚本 `query_draft_six_mc2_fields.py` SHA-256 `d2a7e03a82e6c7fcbccf8273d624847b3f8358def566ad009c86f9e8cf0c8748`，wrapper `run_draft_six_mc2_fields.py` SHA-256 `654c9fd1aee8f8f167c061b77e4a203abf7fc5a1cfd3a5cb09b30786b90cb3f3`。ZCode `operator-ceiling169-v2-185-draft-six-mc2-fields` exit 0、`model_verified=true`、deepseek-flash；未启动 NPU。

### stage189：六条 MC2 的已导出同流前序记录

仅在 stage185 固定的 EP0、EP1 六条 MC2 各自开始前 1 ms，查询同一 rank、model ID `4294967295` 和主 stream 的 CSV kernel 行及 DB `TASK` 行，包括零时长事件。CSV 两份与 DB 两份先分别核对既有 SHA；六个目标以 CSV 完整字段和 DB 同 rank、model、stream、task ID 且时间相交核对。每卡只使用自己的导出设备时间，计算使用整数 ns，表中除以 1000 显示 µs。以下前序指限定范围内、**结束时间最晚且不晚于 CSV MC2 起点的记录**，不是断言框架语义上的前驱。

| rank / 本地 opId | 前序 CSV 算子 / task ID | CSV 前序终点到 MC2 起点 (µs) | 同流 DB 前序 task 到 CSV MC2 起点 (µs) | CSV `Wait Time` (µs) | 间隔减 Wait (µs) |
|---|---|---:|---:|---:|---:|
| EP0 / 402 | MoeGatingTopKHash / 48046 | 1.040 | 1.040 | 1.230 | −0.190 |
| EP0 / 405 | MoeGatingTopKHash / 48090 | 1.360 | 1.360 | 1.540 | −0.180 |
| EP0 / 408 | MoeGatingTopKHash / 48134 | 1.440 | 1.440 | 1.630 | −0.190 |
| EP1 / 402 | MoeGatingTopKHash / 48094 | 229.236 | 229.236 | 229.381 | −0.145 |
| EP1 / 405 | MoeGatingTopKHash / 48138 | 215.856 | 215.856 | 216.000 | −0.144 |
| EP1 / 408 | MoeGatingTopKHash / 48182 | 221.316 | 221.316 | 221.330 | −0.014 |

六条 DB 对应前序均为同 ID 的 `KERNEL_AIVEC`，结束时间与对应 CSV 算子结束时间相同；这六个间隔与各自 `Wait Time` **接近但不完全相等**，不能以此证明字段的计算规则或版本语义。窗口内没有找到与 `Wait Time` 精确相等的 CSV 前序候选，查询仅覆盖前 1 ms 且只选该 model/stream，不能排除其他规则或被过滤的数据。

关键边界：六个 MC2 都同时存在两个同 task ID 的 `KERNEL_MIX_AIC` DB 行，CSV 对应内层行；外层行比 CSV 起点早 **0.860–1.420 µs**，并覆盖内层行。上述“前序终点到 CSV 起点”因此包含了部分 **MC2 外层记录**，不能说整个间隔没有同流任务或设备空闲。前 1 ms 记录中，排除这种目标自身外层记录后，所查主流没有其他跨越 CSV MC2 起点的 DB 行；不代表其他流、其他设备或未导出的任务没有活动，也不证明所选 gating 的数据依赖或缺口来源。DB 内外行可能重叠，不可把持续时间相加。

结果 [draft-six-mc2-preceding-csv.json](draft-six-mc2-preceding-csv.json) SHA-256 `5de5efc398e82f21c28f1aabde8bd88141afb3343bf42d52289eb2126cdb7c42`，完整逐行记录两份 CSV/DB SHA、窗口内任务、重叠及目标自身的内外层边界；查询 [query_draft_mc2_preceding_csv.py](query_draft_mc2_preceding_csv.py) SHA-256 `2ff29c25a9fa5b7250b444f0192976a3bf9852a130b33dba39632ff826890a12`，wrapper SHA-256 `aae57d78955b0380dbae23ca86a738571872d98dd52ab35dfe604a66ebfe4999`。由本机 ZCode stage189 执行已冻结数据的 CPU 只读查询；未启动 NPU。

### stage190：EP1 草稿 gating 与 MC2 的主机连接候选

stage189 的这三个新间隔位于原 stage145–158 中 attention 最大缺口之后，边界任务及连接不同，不能套用此前队列间隔的结论。针对 EP1 三次（本地 opId 402、405、408）的 gating 和 MC2 锚点，stage190 在固定 DB 的 `TASK` 以 start/end/stream/model/task/globalTaskId/connection 全键复核，8 条锚点各有唯一匹配；MC2 各有内外两条 `globalTaskId`，共享一个 connection，不能将其视为两次独立调用。六个连接各找到一条 CANN launch 类记录；分别经 `CONNECTION_IDS` 找到一条 PyTorch 名称类别与 gating 或 dispatch/combine 一致的候选。三组的 CANN `globalTid` 相同，PyTorch `globalTid` 也相同但与 CANN 不同；六个 PyTorch 候选的 `sequenceNumber=-1`、`callchainId=null`。其他连接集合无 PyTorch 匹配只表示本次查询未找到对应行。关联 ID 复用、导出生成关系和 tensor/request 身份未证，以上是候选而非依赖证明。

若仅在各自导出表内对所选记录做端点减法，gating 对应 CANN 行结束至 MC2 对应 CANN 行开始分别为 **224.790、204.120、203.690 µs**；gating 对应 PyTorch 行结束至 MC2 对应 PyTorch 行开始分别为 **206.550、188.110、185.030 µs**。这只说明所选两条记录之间的间隔，并非排队、CPU 空闲或调度耗时；其间可有其他线程、调用、运行时活动或采集盲区。不可跨表或设备直接减法，也不能将两个间隔差或其与设备间隔差当作延迟。stage190 没有查询 queue，不能断言这六条经过或未经过此前发现的 queue 路径；版本、时钟转换与误差仍受 [EP1-GAP-HOST-RECOVERY.md](EP1-GAP-HOST-RECOVERY.md) 所述限制。

结果 [draft-mc2-gap-host-links.json](draft-mc2-gap-host-links.json) SHA-256 `2b07456e853931f2764b494ccc5e315ad6ed9af5745cfa87f0b2da19940b157e`；[查询脚本](query_draft_mc2_gap_host_links.py) SHA-256 `27acae59b8645aced25dee56345a3d7d1e8bc2958cd816941fbb2ed6b9181527`；wrapper SHA-256 `242d000e7f88c1cdab5c21b2b01c69137ab8938c73a35770ad28bc7310f87f7c`。本机 ZCode stage190 对固定历史数据执行只读查询；没有新 NPU 负载。


### stage191：EP1 三个 MC2 前间隔的全流导出覆盖

只复用 stage189 的 EP1 本地 opId 402、405、408，以各自已导出的 `MoeGatingTopKHash` 结束至 `DispatchFFNCombine` CSV 内层起点为半开区间。查询各自 rank 的历史 DB `TASK`，不按 model 或 stream 筛选，纳入跨越区间边界的正时长记录以及区间内零时长事件；CSV 另查所有 stream 的正时长相交 kernel 行。两个原始文件先与 stage189 固定 SHA 核对，仅在本卡时间轴上裁剪和计算并集。

| EP1 本地 opId | 间隔长度 (µs) | DB 所有 stream 正时长任务并集 (µs) | DB 未被正时长任务行覆盖 (µs) | 相交 CSV kernel 行数 |
|---|---:|---:|---:|---:|
| 402 | 229.236 | 0.860 | 228.376 | 0 |
| 405 | 215.856 | 1.060 | 214.796 | 0 |
| 408 | 221.316 | 0.900 | 220.416 | 0 |

三窗 DB 各有且仅有一条相交记录，分别是主 stream 47 的 `KERNEL_MIX_AIC`、task ID 48095/48139/48183；与 stage189 的目标 MC2 外层记录逐字段一致，其开头被区间裁剪后即上表覆盖长度。所查 DB **没有其他 stream 的相交 TASK 行，也没有区间内零时长事件**；所查 CSV 没有任何 stream 的正时长相交 kernel 行。这个结论只覆盖 EP1、这三个固定窗口及上述 DB/CSV 表的导出记录；不同来源数据可能遗漏设备、通信、主机或其他活动，也不能将事件缺失视为事件不存在。故未覆盖时长**不是已证实设备空闲时间**、不是 MC2 内部等待时间，也不证明下发或同步原因，更不构成不同 rank 的同请求对照。

结果 [draft-ep1-mc2-gaps-all-streams.json](draft-ep1-mc2-gaps-all-streams.json) SHA-256 `77cb2f4fc2b756a1c32af1de694e24619dcb20b18bdbc9cdd31bef390ae1b499`；[查询脚本](query_draft_ep1_mc2_gaps.py) SHA-256 `6dec3d1300e7fa7aa2ffa432dc84dd9fdf14b900e26f79238fc2a6b036ca2011`；包装脚本 SHA-256 `c101f961fe6f445342e231083998ca841d31898ff3d9c64c62a5995ec9fe9d76`。本机 ZCode stage191 执行历史数据 CPU 只读查询；独立按 stage189 核对区间端点、各行身份与裁剪算术。没有启动 NPU。

### stage192：三个主机候选间隔的同表 API 覆盖

针对 stage190 的三个固定区间，仅在各自表中按两个锚点相同的 `globalTid` 查找与区间相交的已导出 API，聚合裁剪后的时间并集；两个表的时间不互减，也不与设备时间相减。查询前核对 stage190 输入和数据库哈希，重新在相应表核对所有 12 个端点锚点唯一；两个选定 `globalTid` 范围内未发现无法解释的时间行。**这不证明 API 采集完整或线程始终执行。**

| 本地 opId | CANN 区间 / 条数 / 并集（µs） | PyTorch 区间 / 条数 / 并集（µs） | PyTorch 排除贯穿全窗的外层行后并集（µs） |
|---|---:|---:|---:|
| 402 | 224.790 / 7 / 30.370 | 206.550 / 11 / 206.550 | 61.470 |
| 405 | 204.120 / 7 / 27.190 | 188.110 / 11 / 188.110 | 55.350 |
| 408 | 203.690 / 7 / 28.830 | 185.030 / 11 / 185.030 | 54.110 |

三个 CANN 窗口内各 7 行均为导出 `type=20000`，三个 PyTorch 窗口内各 11 行均为 `type=50001`，不能由类型推断具体执行状态；六个总体并集已从输出行独立复算。三个 PyTorch 窗口各有一条名字散列相同、名称分类包含 `moe` 和 `forward` 的外层行覆盖整个窗口，持续时间分别为 1206.270、1186.080、1178.940 µs，`connectionId=null`。名称分类仅按子串，不足以确定具体函数或因果作用；外层持续范围不证明 CPU 一直在运算。去除此行后的其余记录仍可相互嵌套，排除结果也不是线程独占计算时间。CANN 未见贯穿整个窗口的行。

本查询仅覆盖一个选定 `globalTid` 在各自表中已导出的 API，未查询其他线程、采集盲区或实际 CPU 指令；没有记录覆盖的时间不等于 CPU 空闲或调度耗时。`globalTid` 编码、上游版本、时间转换和误差仍受 [EP1-GAP-HOST-RECOVERY.md](EP1-GAP-HOST-RECOVERY.md) 的条件约束，PyTorch/CANN 候选调用与设备间隔的因果未证。

[结果](draft-mc2-gap-host-intervals.json) SHA-256 `fcac062cbd5c2c48e8ef178e186b59d35a854cb563d45297ea1052e413ea3a08`；[查询脚本](query_draft_mc2_gap_host_intervals.py) SHA-256 `c52ed24dd729587148e0ce664e761e478963d2b36905112c1e3906b76da1438f`；wrapper SHA-256 `e99ceb9be2c72df1c9daf83f4add1fd0c13aa103721eaa73c40dea46d57f0175`。本机 ZCode stage192 查询固定历史记录，未启动 NPU。

### stage193：固定主机区间的接口名称复核

对 stage192 已保存的 54 条记录（EP1 的 402、405、408 三个窗口；每窗 CANN 7 条、PyTorch 11 条），由本机 ZCode 在 SHA 匹配的同一历史数据库中，以表、`globalTid`、起止时间、类型、connection 和名称 SHA 逐条重新定位；只输出完整匹配预定公开接口命名空间的名称，其他保留散列。**42 条匹配白名单，另外 12 条被保留名称，不表示这些记录没有名称。** 三个窗口在各自表内记录的名称/散列序列完全一致，但这不证明属于相同请求或工作量。

CANN 每窗依序是 `aclnnMoeGatingTopKHash`、`aclrtGetStreamAttribute`、`aclnnInnerDispatchFFNCombine`、`aclrtStreamGetId`、`aclrtGetCurrentContext`、`aclrtGetStreamAttribute`、`aclrtGetHardwareSyncAddr`。`aclnnInnerDispatchFFNCombine` 相交窗口并延伸到窗口结束之后；最后一个原按子串标记 `sync` 的名称现可精确识别为 **`aclrtGetHardwareSyncAddr`**，其记录持续分别为 0.400、0.410、0.380 µs。名称包含 Sync **不证明该调用在等待或同步，也不证明设备侧等待**；实际接口契约及本次用途尚未通过同版本文档或实现确认。

PyTorch 每窗外层是 `vllm::moe_forward_shared`；另外有三条 `aten::to`、`aten::empty_like`、`aten::empty_strided`、`aten::empty` 及四条名称未输出的记录。三窗中的 `aten::empty_strided` 均落在 `aten::empty_like` 内，三条 `aten::empty` 也分别与名称未输出的记录重叠，因此不能把调用时长相加当作额外 CPU 耗时。`aten::to` 的名称本身不证明发生了实际数据复制，`empty` 系列也不能单凭名称判定真实分配成本。外层 MoE 记录覆盖窗口，不代表线程整个区间都在执行；名称及时间包含关系均不证明 tensor 消费、调用栈、请求身份或设备因果关系。不同表及设备时间不做互减。

[结果](draft-mc2-gap-api-names.json) SHA-256 `ef21672afb40e0ff7f0d26126c38cad2f1faa7ea93d3fafeb5f317c4f2404621`；[查询脚本](query_draft_mc2_gap_api_names.py) SHA-256 `8e4ca5210b177bac95f286250bf2024f44e3f1e23aa271adb8bd2b71c2790422`；包装脚本 SHA-256 `b9dfc3ff082d25516f1b559a9915e72c16e7a67d28279f7a3dfa51c26ffeeffb`。这是对固定历史导出记录的只读识别，未新增 NPU 工作；尚不能判断此前设备记录缺口的原因。

### stage194：三个草稿 gating 和 MC2 的固定队列候选

此前 stage152 的队列属于 attention 缺口，不能代替此处结果。stage194 对 stage190 的六个固定 PyTorch/CANN 候选重新核验端点与哈希，在 PyTorch 同表查父操作内的 `type=50002` 队列记录，并以 `CONNECTION_IDS` 中共享本地 correlation ID、CANN 线程 globalTid 和父操作时间前后 2 ms 限定相关 worker 队列。三个 gating 和三个 MC2 各找到一条名称包含 enqueue 的父操作内队列，以及一条名称包含 dequeue 的关联 worker 队列；每组 CANN launch 导出时间数值均严格落在该 dequeue 的起止数值之内。它们与已保存 parser 的关联规则一致，但不是对 exporter 实际运行路径、请求级因果或采集完整性的证明。

在同一个 `PYTORCH_API` 表内，以 dequeue 开始减 enqueue 结束的端点数字（单位微秒）如下，受 [EP1-GAP-HOST-RECOVERY.md](EP1-GAP-HOST-RECOVERY.md) 所述 parser 版本、同配置及时间变换精度条件约束：

| 本地 opId | gating 端点间隔 | MC2 端点间隔 |
| --- | ---: | ---: |
| 402 | 4.04 | 3.14 |
| 405 | 3.56 | 3.26 |
| 408 | 3.65 | 3.39 |

查询的唯一性仅限固定父操作、所选字段及 2 ms 窗口；采集丢失、窗口外记录和 correlation 复用均未排除。这些数字不等于完整排队耗时、实际调度耗时或设备等待，也不说明 gating→MC2 主流缺口的原因；CANN 与 PyTorch 的数值嵌套只用于匹配条件，不能由此计算跨表延迟。

[结果](draft-mc2-fixed-queue.json) SHA-256 `1044f1a8408552f1475d0948044da754c94e87fd8c9cc36d606e96c907dafcab`；[查询脚本](query_draft_mc2_fixed_queue.py) SHA-256 `8913fa014dc54ed3c399779be3807ebe9864822318d135c02f4731ec1ee0b5a4`；wrapper SHA-256 `138f37b3af244cf8379bc160d018544995bca16a2e30030e3cd448e004df5dd4`。本机 ZCode stage194 对固定历史数据执行只读查询；没有新 NPU 负载。

### stage196：固定源码符号核对 stage193 未公开名称

对 stage193 的 12 条未输出名称记录，在本机直接计算两条预先从源码提取的完整算子符号 SHA-256，与每条记录已保存的名称散列比较（基于散列抗碰撞性识别字符串）：

| 固定符号 | 名称 SHA-256 | 命中 | 源码位置及 SHA-256 |
| --- | --- | --- | --- |
| `_C_ascend::moe_gating_top_k_hash` | `a840ca8fd2d354090b91c91c8e1bb342b9185d216e24d4efb79b553e6949c280` | 三个窗口各一条 | `agents/fusion/frozen-source/vllm_ascend/ops/fused_moe/router/fused_topk_router.py:204`；`06c47fd91e8c36f6f5465cea0b32a184280f842833c2bff27a18fc6d17fcb838` |
| `_C_ascend::dispatch_ffn_combine` | `d8d082cec399efe832279f57a70076fd0cc0add51b873888ca74bad2acdc1666` | 三个窗口各一条 | `sources/vllm-workspace/vllm-ascend/vllm_ascend/ops/fused_moe/moe_comm_method.py:591`；`7040f691355bd9acb3c668e698de3f7d3854edbc22e3f979f9b03d99800364d7` |

stage193 的旧白名单没有包含 `_C_ascend::`，所以原来这六条只输出散列。剩余六条共一个散列（`e7f4cb21074f244688a21cace454eccd89439d68046d172e1803b2477f53b9dc`）尚未识别，不猜测其名称或作用。源码文件中有对应的 Python 调用，**不证明历史运行加载了该源码对应的二进制，也不证明张量实参、请求身份、所走分支、独占 CPU 耗时或设备缺口的成因**。三个窗口中名称一致也不证明工作量相同。

本机 ZCode stage196 执行只读 CPU 复算，未访问原 DB、未启动 NPU。复算[脚本](check_fixed_moe_symbol_hashes.py) SHA-256 `d24cc3f05b2be1bf9886eec7dc3e4a3c3af8b20bde630eed66a928a4300986bc`；[结果](fixed-moe-symbol-hashes.json) SHA-256 `f53856d3a4045946ae9a884ae9f6d308e76e4e9d5553056aba4437887042ffda`。

## 16 rank 草稿物理行数序列与条件性缓存提示

复用 `16rank-draft-alltoall-distribution.json`（SHA-256 `633011028e6d5953f03f1493c197085bdd1a810c095cc646e4f5b0e115f61ec6`），以每卡各自的 bracket 开始时间排序，不比较跨卡时钟。`recheck_draft_mc2_shape_sequences.py` 校验原文件 SHA、`issues`、rank 与图间隔覆盖，逐条检查 MC2 首输入 `[M,5120]` 和第四输入 `[M,3]`：16 卡第一目标图间隔内的 M 序列均为 `[3,3,3,1,1,1,2,2,2,2,2,2,3,3,3]`，第二间隔均为 `[3,3,3]`。其中 M=1 共 48 条，15 条 bracket 内有一条已导出 SMLA，33 条无；M=2 共 96 条，63 条有、33 条无。输入形状仅说明导出的物理维度；每卡序列一致**不证明跨卡相同位置是同次 collective、每行有效或具体专家接收行数**，不等于发生对应 16 卡同步。

独立条件推论：假设某次共同 MC2 有 16 个参与 rank、各 rank 至多 1 个有效输入行、每行至多 3 个路由槽、每个槽对任一专家最多贡献 1 行，且无产生额外专家行的复制或补充，则每个专家的接收行数至多 `16×1×3=48`；各 rank 至多 2 行时至多 `16×2×3=96`。这两个算术界**不要求同一 token 的专家 ID 去重**。进一步假设历史调用确实运行对应保存源码、累计/容量裁剪不增大计数，两个界都小于等于 `L1TileShape::M=128`：`dispatch_ffn_combine.h:200` 与 `dispatch_ffn_combine_kernel.hpp:455–467,562–574` 表明代码会为这两级 GMM 的权重设置 `CACHE_MODE_DISABLE` 提示。M≤3 时朴素上限为 144，不能不加其他条件推出该提示必然命中。

上述只是可证伪的代码条件，不是实际路由分布、hint 硬件效果、L2 未命中、HBM 带宽或低活动根因的证据；源码版本／编译对象、同次 16 rank 配对、有效行数与该提示的实际行为都尚未绑定。它将后续取证优先级具体化为：同次调用的匿名 collective 标识、16 rank 有效输入行数与每专家接收计数、编译对象和缓存行为。当前限制下未新启设备负载。

## EP8 两个宽窗口中草稿候选与采样类型的时间次序（stage241）

固定 EP8 的 target forward ordinal 6、7 两个历史窗口：每窗从**前一目标第 39 层 MC2 结束**到下一目标第 0 层开始，分别为 43.067 ms、38.312 ms；前一个端点不是完整 target forward 或 logits 的结束，ordinal 6 的前一层39身份仍依赖已记录的 eager 映射假设。每个窗口内有 3 个此前按四个 HC 标记等规则严格归属的草稿候选 bracket，顺序不重叠，但三者各自 block 身份仍为 UNKNOWN；仅按这个规则没有落窗的样本不能解释为没有执行。

只从 EP8 冻结 `kernel_details.csv` 中选择四个精确 type 的正时长相交行，各窗口记录一致的 **7 条 `ArgMaxV2`、1 条 `rejection_greedy_sample_triton`、1 条 `copy_and_expand_dflash_and_dspark_inputs_kernel`、1 条 `_build_dspark_swa_indices_kernel`**。选中的每条均完整位于其窗口，model 导出为 `4294967295`、stream 为 47；限定这四类时两窗都呈现同一顺序：先两条 ArgMax 与 rejection，再 copy/expand、build-indices，随后三个严格候选 bracket，最后再出现五条 ArgMax。没有选中的其他类型或其他流不在这个时间序列内。

| 下一目标 ordinal | 最后一个 bracket 结束→五条后置 ArgMax 第一条开始 | 五条 ArgMax 首开始→末结束跨度 | 五条 ArgMax 自身时长之和 |
|---:|---:|---:|---:|
| 6 | 532.009 μs | 358.653 μs | 86.159 μs |
| 7 | 474.730 μs | 356.833 μs | 86.658 μs |

后置五条位于同一 stream 且在 CSV 中顺序不重叠，但它们的首末跨度包含其他或未导出工作，不能把跨度减去所选时长称为 CPU 开销、设备空闲或采样等待。数字 5 只描述选中窗口中的 ArgMax 行数，**不证明这五条就是 DSpark K=5 的逐步采样**；同理，按时间和名称也不能证明 rejection、输入准备与三个 bracket 对应同一请求、某个具体 target 验证、某个 token 或因果依赖。两个窗口不是端到端占比，不代表其他 rank、未归属草稿窗口或其他迭代。

复核已保存 `rank_gaps.csv` 和严格 `draft_call_metrics.json` 的 SHA，逐窗重新选 EP8 的三个完整 bracket，并逐条验证四类计数、时间关系、完整落窗和 model/stream；EP8 CSV SHA 与冻结 manifest 一致。结果 `draft-verify-timeline-ep8.json` SHA-256 `6270f9cb785b75cdfbe2442498631a5b0933941b552257c5220046ac55eaf5ce`；查询脚本 `query_draft_verify_timeline_ep8.py` SHA-256 `78fad6da80246adccc2320a4fa97dca0dba021b5cb9e6878e5b587a67820b977`，wrapper `run_draft_verify_timeline_ep8.py` SHA-256 `db3e806f23d3ad40d471ec9b8825cd7c708a353c6647406c66ea33afb5bf1d4a`。本机核验 ZCode stage241 exit 0、`model_verified=true`、模型 `deepseek-flash`；没有新 NPU 负载。

### 五条后置 ArgMax 的八个间隔：已导出任务（stage242）

在 stage241 的 EP8 ordinal 6、7 各选定窗口中，固定后置五条 ArgMax，取每条结束到下一条开始的四个半开间隔，共 **8 个间隔**。这是 stage241 记录的两个五条 ArgMax 首末跨度的**内部组成**：排除五条 ArgMax 自身时长，不是额外相加的时间。每个间隔按同卡 CSV 时间裁剪所有正时长相交行，按流分类；选中相邻 10 条 ArgMax 各自的 model、stream、task ID 与起止时间均在同一 CSV 中唯一复核。窗口依然从前一层39的 MC2 结束而非完整 forward 结束；ordinal 6 前一层身份的 eager 映射限制不变。

八个间隔各出现 **10 条**完整落在间隔内的正时长导出行，均为 stream 47、model `4294967295`，导出时间顺序一致：`Cast → ViewCopy → Slice → GatherV2 → MatMulV2 → Slice → Add → ViewCopy → Slice → Slice`。这只表示选中 CSV 的时间序列：名称、同流和相邻不证明数据依赖，也不证明它们是 DSpark 的 Markov 路径、属于同一请求或对应 K=5 的每步采样。`ViewCopy` 是导出名称，不足以证明真实物理搬运字节或可消除拷贝。

下表的类别百分比是每个间隔内相应类型裁剪时长并集除以**该间隔自身时长**；本次每窗 10 行互不重叠，未覆盖时间单列。单位 μs，百分数是局部时间比例：

| ordinal / 间隔 | 间隔 | MatMulV2 | 两条 ViewCopy | 四条 Slice | Add | GatherV2 | Cast | 未被所查 CSV 行覆盖 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 / 1 | 68.459 | 35.96% | 26.91% | 24.36% | 7.07% | 3.56% | 1.78% | 0.240 |
| 6 / 2 | 67.859 | 36.25% | 27.62% | 24.88% | 5.45% | 3.68% | 1.71% | 0.280 |
| 6 / 3 | 66.818 | 36.49% | 27.69% | 24.19% | 5.63% | 3.71% | 1.85% | 0.300 |
| 6 / 4 | 69.358 | 35.15% | 28.66% | 23.70% | 6.37% | 3.81% | 1.90% | 0.280 |
| 7 / 1 | 69.279 | 35.25% | 27.05% | 24.83% | 6.67% | 3.84% | 1.99% | 0.260 |
| 7 / 2 | 66.879 | 35.89% | 28.17% | 24.16% | 5.65% | 3.95% | 1.85% | 0.220 |
| 7 / 3 | 67.318 | 36.10% | 27.84% | 24.24% | 5.79% | 3.95% | 1.72% | 0.240 |
| 7 / 4 | 66.699 | 36.22% | 26.90% | 24.65% | 6.18% | 3.93% | 1.77% | 0.240 |

主流已导出行并集覆盖每窗约 **99.55%–99.67%**；这八窗内没有与间隔相交的其他数字 stream 或未知 stream 的**正时长 CSV 行**。这不是整卡利用率，也不排除未导出任务、零时长事件、kernel 内部等待或其他采集盲区；0.220–0.300 μs 未覆盖部分不是已证明空闲。局部 MatMul、ViewCopy 和 Slice 分别是占比较高的名称，仍需实际调用路径、tensor/权重身份及原始接口才能判断结构重组或瓶颈，不能把这些局部百分比当整请求占比或可兑现收益。

本机核对 stage241 锚点 SHA、EP8 CSV 冻结哈希、10 条邻接 anchor 和八窗各 10 个事件的类型/顺序、流、完整包含、不重叠、裁剪并集与补集，再按每窗分母重算上表；没有比较跨卡绝对时间。结果 `draft-argmax-between-ep8.json` SHA-256 `a20fd5761c9ca88b5161ea624c267c248342a21d230bb69a7f83ee589fdc0666`；只读查询 `query_draft_argmax_between_ep8.py` SHA-256 `bd0a2e3e23a40bce5fe308b0b4f2d825aaf2383940f63b28613ea6519419e500`，wrapper `run_draft_argmax_between_ep8.py` SHA-256 `b7003f738abf8e9faee5105603dd18f8dd3e99cc7f704d126cb99ea53393b862`。ZCode stage242 exit 0、`model_verified=true`、模型 `deepseek-flash`；没有 NPU 工作或实现修改。

### 八个间隔内 80 条记录的形状与配置相容性（stage243）

以 stage242 的 EP8 这 **8×10 条已锚定事件**为唯一查询范围，按同卡 model、stream、type、task ID 和精确起止时间匹配原 CSV；80/80 各唯一匹配、所需列无缺失。八窗中每个相同位置的导出 Name、shape、dtype、format 和核心标签完全相同；下表按照每窗时间顺序列出，全部张量格式报告为 ND。表里的维度是**物理形状，不是内容或有效请求数**：

| 次序 / 类型（导出 Name 片段） | 主要输入 → 输出 | dtype |
|---|---|---|
| Cast（`aclnnArgMax_CastAiCore_Cast`） | `[4] → [4]` | INT32 → INT64 |
| ViewCopy（`aclnnInplaceCopy_ViewCopyAiCore_ViewCopy`） | `[24]` 与 `[4]` 等八个输入 → `[24]` | 均 INT64 |
| Slice（`aclnnEmbedding_SliceAiCore_Slice`） | `[4,6]` 等 → `[4,1]` | INT64 |
| GatherV2（`aclnnEmbedding_GatherV2AiCore_GatherV2`） | `[129280,256]`、`[4]` 等 → `[4,256]` | BF16、INT64 等 → BF16 |
| MatMulV2 | `[4,256]`、`[129280,256]` → `[4,129280]` | BF16 |
| Slice（`aclnnAdd_SliceAiCore_Slice`） | `[4,646400]` 等 → `[4,129280]` | BF16、INT64 等 → BF16 |
| Add | `[4,129280]`、`[4,129280]` → `[4,129280]` | BF16 |
| ViewCopy（`aclnnAdd_ViewCopyAiCore_ViewCopy`） | `[2585600]`、`[4,129280]` 等八个输入 → `[2585600]` | BF16、INT64 等 → BF16 |
| Slice（`aclnnInplaceCopy_SliceAiCore_Slice`） | `[4,646400]` 等 → `[4,129280]` | BF16、INT64 等 → BF16 |
| Slice（`aclnnArgMax_SliceAiCore_Slice`） | `[4,646400]` 等 → `[4,129280]` | BF16、INT64 等 → BF16 |

`MatMulV2` 报告 AI_CORE，其余九条报告 AI_VECTOR_CORE；这是导出核心类别，不能当作整设备状态。冻结模型 `text_config` 的词表大小是 **129280**、`dspark_markov_rank=256`，启动快照请求 `K=5`；算术上 `646400=5×129280`、`2585600=4×5×129280`、`24=4×(5+1)`。这些数值、连续的 embedding 名称、`[4,256]` 到 `[4,129280]` 的形状以及随后 `[4,129280]` 相加，与已存 F30 中条件性的 Markov embedding、bias 和原位 logits 加法路径**相容**，是比仅靠 task 类型更具体的排查线索。

这仍不是 tensor 地址、权重、实际调用栈或数据消费关系：无法证明这些任务就是该 DSpark 方法、那四行均是有效请求、数值 5 表示本次实际 K，或 `ViewCopy` 必然触发物理搬运；也不能把八组形状相容的序列升级为同一请求的逐步采样证明。base logits 的来源和 tensor 生命周期、stage241 的五条 ArgMax 的具体输入与参数、实际随机/greedy 分支也仍未绑定。形状一致不能区分复用、复制或其他实现，不能据此宣称可消除比例或量化收益。

本机脚本 `recheck_draft_argmax_shapes_ep8.py` 按 stage242 的每个 `(ordinal, gap, start, end, type, taskId, model, stream)` 核对全部 80 条并检查六类/八位置的签名，以及冻结配置 SHA；输出 `draft-argmax-shape-summary-ep8.json` SHA-256 `4b79aa0221a538679901e7c1a2b63626c558ed23cfede62a329206e9f232af16`，脚本 SHA-256 `b3e548fbc8e1c38715e31033a6ec3ac4caa7400646db8181411274b4b67a3bde`。原结果 `draft-argmax-intervals-shapes-ep8.json` SHA-256 `59930fe7433cb31708f906713b8f24685dd307cb43ad1ab7a0b2d35310766dfe`；查询脚本 SHA-256 `839b983ca0fd8fbd4088c5844a8e23f480a817c1851e7d5c888bc0037d4fa03d`、wrapper SHA-256 `f4127d7c6212b5772d57f0823d6d0ce36a0af758317f1334022e821319ca6c94`。本机验证 ZCode stage243 exit 0、`model_verified=true`、模型 `deepseek-flash`，只读既有导出，无新 NPU 工作。

#### 单条 ViewCopy 的条件性逻辑大小

stage243 每个选定间隔中，`aclnnAdd_ViewCopyAiCore_ViewCopy` 一条的导出输出为 BF16 `[2585600]`。按 BF16 每元素两字节，仅从该**输出物理形状和 dtype**计算其逻辑大小是 **5,171,200 字节（4.931640625 MiB）**。数值上 `2,585,600 = 4×5×129,280`，与物理四行、冻结启动请求 K=5 和配置词表大小 129280 的组合相容；同窗 BF16 `[4,129280]` 是 **1,034,240 字节（0.986328125 MiB）**，前者恰为其 5 倍。这只是尺寸算术，不能证明它们同属一个 tensor、对应 logits，或 4 行/K=5 在该次均为真实有效运行值。

stage242 的两窗口各有四条此类已选行；四次逻辑输出大小相加为 **20,684,800 字节（19.7265625 MiB）**，只是按记录重复计算相同形状的逻辑量，不代表四个独立分配、显存峰值或实际传输字节。`ViewCopy` 的导出名称、输出形状及其 8.540–10.940 μs 持续时间均不能证明物理拷贝、源/目标内存位置、重用/别名、HBM 读写量或可以省掉该时间；若研究融合，仍须建立 tensor 身份、别名与数据消费者及精度契约。上述尺寸与前文 F30 对假设性 FP32 概率数组的大小恰好相同只是算术巧合，不能把两个对象混同或相加推显存。

计算基于上节已核对的 `draft-argmax-intervals-shapes-ep8.json` 和 `draft-argmax-between-ep8.json`，SHA 分别为 `59930fe7433cb31708f906713b8f24685dd307cb43ad1ab7a0b2d35310766dfe`、`a20fd5761c9ca88b5161ea624c267c248342a21d230bb69a7f83ee589fdc0666`。本次只在本机读取并作整数算术，没有 NPU 工作。

## EP8 已选采样间隔的 profiler 候选 API 关联（stage245）

对 stage242 的 EP8 两个窗口、八个间隔的 80 条事件及其 10 条 ArgMax 锚点，stage245 按同 model、stream、task 和精确设备起止时间查冻结 profiler DB 的 `TASK`，再查同 connection 的 `CANN_API`、`CONNECTION_IDS` 和候选 `PYTORCH_API`：**90/90 精确 TASK 各一条，每条连接有一条 CANN 行，每条经连接集合找到一条候选 PyTorch 行**。90 条 TASK、CANN 和候选 PyTorch 查询均未截断；53 条 TASK 有一个连接集合，37 条有两个，其中额外的 37 个集合没有候选 PyTorch 行。90 个候选行的 `sequenceNumber=-1`、`callchainId` 均为空。这里的唯一性仅限该查询及其连接集合，不能由此认定候选 API 是唯一调用者或 ID 全局唯一；不同事件可关联同一候选行。

八个间隔中，每窗先后相邻的 `Slice` 和 `GatherV2` 关联同一候选集合；`Slice`、`Add`、`ViewCopy` 关联另一个候选集合；`MatMulV2` 关联第三个集合。前一锚点 `ArgMaxV2` 和紧随其后的 `Cast` 也关联同一候选集合。举例 ordinal 6 的第一间隔，`Slice` task 52768 和 `GatherV2` 52769 对应的候选集合都是 128636；`Slice` 52771、`Add` 52772 和 `ViewCopy` 52773 都是 128638；前一个 `ArgMaxV2` 52765 和 `Cast` 52766 都是 128634。这是**同一导出连接集合的分组证据**，可以支持同一上层候选关联的排查，但不证明这些任务共享某个实际 tensor、同一次请求、因果关系或真实物理复制。parser 和连接集合来源及局限见 `EP1-GAP-HOST-RECOVERY.md` 的 stage149；未对这组事件独立核验 queue、具体 exporter 分支或连接 ID 的复用范围。

查询脚本对非安全白名单 API 名只保存哈希：90 个候选名称均未直接公开，归为五个哈希类别；将它们与已有 `query_draft_query_metadata_ep7.py` 的 `safe_torch` 和本次脚本 `safe` 等现存安全名称集合去重后的 18 个名称逐一计算 SHA-256 比较，**五个哈希均无命中**。不能据此给出 API 名或 DSpark 调用身份；相同哈希仅代表这里保存的名称哈希相同，不代表同一实例。当前没有调用栈、tensor/storage 与请求绑定，也没有足以将这些记录确定为 Markov 路径或逐步采样的直接证据。

结果 `draft-argmax-api-links-ep8.json` SHA-256 `8eab586ca22e79ddce8b88a5b304ba54d0f18a4aca840f0979e584b9e6e01007`；固定 EP8 DB SHA-256 `67d14f7f386658f8c8c5d3c9059dff9b0f1decb200eba621a3432fcf1e23de58`，stage242 锚点 SHA-256 `a20fd5761c9ca88b5161ea624c267c248342a21d230bb69a7f83ee589fdc0666`；查询脚本 `query_draft_argmax_api_links_ep8.py` SHA-256 `fad9ade7b3b198ea4d09da12cfd1204ae54c4a8f5d122ffd26ebe103c165494e`，wrapper `run_draft_argmax_api_links_ep8.py` SHA-256 `8879a4b8f8df1ce355db85e5cbc5efc3f4e6cdfceeb49057d7c2375f7a4be30a`。ZCode 固定执行记录 `/Users/wio/work/173-sglang-matched-172-benchmark/operator-ceiling169-v2-245-draft-api-links.exit-code` 为 0，配套 `.verified.json` 显示 `model_verified=true`、模型 `deepseek-flash`。本次未运行 NPU、未修改目标实现。

### stage245 名称哈希的补充核对：与设备导出算子前缀匹配

上述 18 个安全 PyTorch 名称未命中之后，对**已公开保存在 stage243 结果中**的 `Name`，仅取首次下划线前的 `aclnn` 前缀作为候选，再逐字节计算 SHA-256 比较，得到四个匹配：`aclnnArgMax`（26 条候选记录）、`aclnnEmbedding`（16 条）、`aclnnInplaceCopy`（16 条）、`aclnnMatmul`（8 条）；其余 24 条属于同一个仍未匹配的名称哈希。按固定 80+10 条事件逐行核算：8 个 `Slice` 与 8 个 `GatherV2` 的候选名称都是 `aclnnEmbedding`，8 个 `MatMulV2` 候选名称是 `aclnnMatmul`，8 个 `Cast`、8 个 `Slice` 以及 10 个 ArgMax 锚点候选名称是 `aclnnArgMax`；8 个 `ViewCopy` 与另外 8 个 `Slice` 候选名称是 `aclnnInplaceCopy`。剩余 8 个 `Slice`、8 个 `Add`、8 个 `ViewCopy` 的同一哈希**不等于**已保存设备名称前缀 `aclnnAdd` 的哈希，实际名称仍未知，不应自行重命名。

这个结果使设备事件与同名 `aclnn` 候选 API 的关联更具体，但它没有辨识出 DSpark 的 Python 调用栈。当前结果表名为 `PYTORCH_API`，不能据表名就把这些已匹配的候选名称称为 Python 层 `aten` 函数：现存 `_fwk_api_db_parser.py` 的 `get_api_data_for_db` 将 Python trace、队列、Torch op 和 MSTX 等多种来源写入同一表，本次 stage245 未保存 `type` 或任何调用栈。在 profiler 连接生成版本和具体事件调用链尚未闭合时，名称哈希与同一连接集合都不能证明真实张量身份、DSpark 调用、请求及跨事件的数据依赖；未匹配名称也不能排除某源码分支。

复核脚本 `inspect_draft_api_name_hashes.py` SHA-256 `062ff7389cf9228df2ceb93b804de0ca5b3659d3760046a42e4490d9292de609`，运行前校验 stage243 和 stage245 两个本地 JSON 的 SHA-256，仅使用 stage243 已公开算子名的 `aclnn` 前缀作固定字典；只读本地、未运行 NPU。候选来源和分组均限于 EP8 已选两窗，不外推全服务。

### stage246：EP8 首间隔候选记录的线程与时间字段

对于 stage245 的 EP8 ordinal 6 首间隔，固定前后 ArgMax 及中间 GatherV2、MatMulV2、Add 对应五个候选连接集合，stage246 逐条复核：5/5 各一条、名称哈希均匹配前次记录，查询完整且未截断；五条 `PYTORCH_API` 都报告 `globalTid=3122441224919`、`type=50001`，起止时间均有数字且起点小于终点，五条 `callchainId` 仍为空。这证明这五条**候选框架表记录**具备同线程包围查询所需字段，不证明它们已绑定同一请求或 DSpark。无需用这些框架表时间减设备时间，也不能把同线程看成实际数据依赖。

结果 `draft-api-candidate-fields-ep8.json` SHA-256 `e257d11a0bf04ca9e68eab6a7bbdadccd2c0e4b8ba6c4cae4d668b834e15125a`，复核固定 EP8 DB 与 stage245 结果哈希；只读查询 `query_draft_api_candidate_fields_ep8.py` SHA-256 `585ab7c0d2b1effead7dd337acf452a3b66d206a87943b75d586ebd56980f3ce`，wrapper `run_draft_api_candidate_fields_ep8.py` SHA-256 `c4861bece307df5905e4d7e604a2df2acb5297445ec42e3f1e563aa1ace95482`。`/Users/wio/work/173-sglang-matched-172-benchmark/operator-ceiling169-v2-246-draft-api-fields.exit-code` 为 0，配套 `.verified.json` 的 `model_verified=true`、模型 `deepseek-flash`。未使用 NPU、未更改目标代码。

### stage247：五个 EP8 候选的同线程包围记录

对 stage246 已固定的 ordinal 6 首间隔五条候选，在同一个 EP8 `PYTORCH_API` 表中按相同 `globalTid`、数字起止时间及**严格时间包围**查询：任务 52765、52769、52770、52772、52776 分别有 **2、1、2、1、2** 条符合条件的外层记录，共 8 条；单次查询完成、没有触发上限或输出截断。8 条均报告 `type=50001`、`callchainId` 为空；脚本限定的五个已知源码路径子串和四个函数名子串均未命中，其他名称仅保存哈希。相同名称哈希只表明名称相同，不合并为同一条记录或证明同一个调用。

结果只覆盖这五个候选所在同一线程、起止字段为数字且严格包围其时间的已导出记录；相同边界、其他线程、不同来源、未采集记录及白名单之外的源码线索不在结论范围。它**没有找到模型级调用身份**，但不能据此断言全部 Python trace 不存在，也不能排除 DSpark 调用。时间包围并不证明具体 Python 调用栈、tensor/request 关联或设备数据依赖；该近邻查询路径到此停止，不做重复扫描。

结果 `draft-api-outer-ep8.json` SHA-256 `96321da910ace7a75e38543166dccd6353981e94818ddc5b03dc742ac707cf8a`；固定 EP8 DB 和 stage246 结果已进行 SHA 核验。查询 `query_draft_api_outer_ep8.py` SHA-256 `7f8bb9b948684166d3917c86f64a4e0db9997f2169974f2f5607f4cea2e425c8`，wrapper `run_draft_api_outer_ep8.py` SHA-256 `aba1ddca75a0f3a8d3f2182eaea7e84f25e1096e720eb25c14129e4666873c1f`。`/Users/wio/work/173-sglang-matched-172-benchmark/operator-ceiling169-v2-247-draft-api-outer.exit-code` 为 0，配套 `.verified.json` 显示 `model_verified=true`、模型 `deepseek-flash`。未运行 NPU、未修改目标代码。

## EP0 长 MC2 与 EP1 短 MC2 的主机候选（stage260–261）

以 stage189 固定 EP0 本地 402、405、408 三对 gating 和 MC2 锚点，沿 stage190 完全相同的连接查询规则读取 EP0 profiler DB。每个锚点的完整 TASK 键唯一匹配，每个有 1 条同连接 CANN launch 名称候选、1 条经 connection 集合关联的 PyTorch gating 或 dispatch/combine 名称候选；同一次查询内各对的 CANN 候选线程一致、PyTorch 候选线程一致。连接集合存在复用，不能仅凭唯一候选称已证明调用的真实父子、设备依赖或请求身份。

对同一 rank、同一张表，仅计算 gating 候选结束至 MC2 候选开始的时间：EP0 CANN 分别为 **216.98、201.05、197.72 µs**，PyTorch 为 **198.09、182.42、179.17 µs**；已保存 EP1 对应 CANN 为 **224.79、204.12、203.69 µs**，PyTorch 为 **206.55、188.11、185.03 µs**。这些候选间隔相近，不能将所选主机候选端点间隔本身当作已观察 MC2 时长差的直接量度。两个表不可跨表相减，rank 间不比较时钟；其中可能有其他线程、调用及未导出活动，不是 CPU 空闲或调度开销。EP0 与 EP1 未绑定同请求、同次 collective 或等工作量，不能据此排除主机下发、等待、路由或其他具体根因，也不能解释 MC2 内部。

固定 EP0 DB SHA-256 `a6c626753b5e21cd066ede0cceffe104e8603c01a79918eb79a270dcb4ad0a09`，prior SHA `5de5efc398e82f21c28f1aabde8bd88141afb3343bf42d52289eb2126cdb7c42`，查询 `query_draft_ep0_mc2_host_links.py` SHA `6ef37ca817fa18d94aa423b7c6c91a19bdd7472a6c7d10abcabf2a040d7c5330`；ZCode stage260 exit 0、model_verified=true，结果 `draft-ep0-mc2-host-links.json` SHA `6660ac83362cb6b8dc80b7cc845e3aeac21eb93a60c4f360590faedbf2b02ad2`。复算 `recheck_draft_ep0_ep1_host_intervals.py` 绑定 EP0/EP1 输入 SHA 并限定表和线程；ZCode stage261 exit 0、model_verified=true，输出 `draft-ep0-ep1-host-intervals.json` SHA `eed00fd19cf9729d1d08329fcb6c84c16c7c50c1a828dd0f7f4db420a2bac85d`，与本机复算一致。全程仅 CPU 读取历史资料。
