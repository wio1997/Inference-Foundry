# 同一次采样 EP8–15：第 14 / 39 层其他流 AI CPU 记录

复用先前冻结的 scheduler digest `1b826883b782ba77814d5bd40fcd3e46c759e380069db2e066ff0e9cb4bce89d`、host scope → flow → device → CSV 唯一锚点，每个 EP、每层各选一条 MC2 任务。对每卡同一数据库的 `TASK`，筛选其他 stream（包括空 stream）中导出类型为 `KERNEL_AICPU` 且时间与该 MC2 窗口相交的记录；逐行按窗口裁剪，再计算去重区间并集，最后除以本卡 MC2 任务时长。两类任务在 DB 内的时钟契约与采集完整性尚未独立验证，以下是**导出坐标的交集**。

| EP | L14 MC2 时长 ms | L14 相交记录 | L14 裁剪并集 ms | L14 覆盖比例 | L39 相交记录 |
|---:|---:|---:|---:|---:|---:|
| 8 | 160.365956 | 9 | 97.497551 | 60.797% | 0 |
| 9 | 160.367275 | 6 | 123.489393 | 77.004% | 0 |
| 10 | 160.422455 | 1 | 160.422455 | 100% | 0 |
| 11 | 160.378400 | 1 | 160.378400 | 100% | 0 |
| 12 | 160.409775 | 1 | 160.409775 | 100% | 0 |
| 13 | 160.375656 | 1 | 160.375656 | 100% | 0 |
| 14 | 160.365416 | 1 | 160.365416 | 100% | 0 |
| 15 | 160.382476 | 1 | 160.382476 | 100% | 0 |

每条相交记录均按设备、model、stream、task 和精确起点与对应 CSV 唯一匹配，CSV 都标记 `AI_CPU`，记录时长差不超过 1 ns。EP10–15 的各一条均显示为 `reduce_scatterAicpuKernel`；各卡 L14 的导出窗口均被这条对应记录完整覆盖。EP8、9 的并集由多条记录构成，分别包含三类命名为 allgather、alltoall 和 reduce-scatter 的 AI CPU 记录，详见 JSON。全部 16 个窗口没有完整身份键重复记录或非正时长记录；未选中其他类型、同主流、未导出记录及 MC2 内部阶段。

EP8 原报告从全部类型 TASK 入手，核查得到 L14 AI CPU 并集 97.497551 ms、L39 零条，与本次独立运行重合。此前 16 rank 的其他流结果针对第 0 层，不能用于替代本范围。**100% 仅为一条导出记录覆盖自身 MC2 区间**，不等于链路传输全程、资源忙碌或通信被计算隐藏；相同类型、task ID、stream ID 和 scheduler digest 都不足以证明跨卡记录属于同一事务或请求。L39 未查到该类型相交，不证明无通信、无等待、无其他计算活动。L14/L39 有效工作量和专家路由未确定，这些相关性不证明慢任务根因。未做跨卡时钟对齐、时间戳相减或求和。

复现：在任务根目录执行 `python3 agents/diagnosis/run_ep8_15_l14_l39_aicpu_overlap.py`；它核对冻结锚点文件 SHA、stage 容器身份和挂载，限制 CPU/IO 优先级，使用只读 SQLite 并在读取前后校验 DB 摘要及无 WAL，核对 8 张卡 CSV 先前记录的 SHA。DB 的摘要是本次首次观察所得且前后相同，并非独立冻结的历史来源认证；结果记录 `agents/diagnosis/ep8-15-l14-l39-aicpu-overlap.json` 保留逐项证据和摘要。只读取已有数据，不启动 NPU，不改变服务或推理实现。

下一步应对 EP9–15 那些相交记录追查其 host 调用及确切通信组或事件绑定，先判断数据是否支持区分同一事务、独立请求及内部同步；在关联被证实前不能把导出重叠叫做争用或因果。

## 同次采集其他 digest 的跨层归属复核

既有 EP8 与 EP9–15 来源报告已经证明 `1b826883…` 样本中若干覆盖 L14 MC2 的 reduce-scatter AI CPU 记录来自 L23 主机提交范围；这并非本节新发现。为检查它是否仅出现于该 digest，本节在固定 SHA 的已提取 JSONL 中，按**同一 digest、同一 EP**，配对 host scope 选中的 L14 `DispatchFFNCombine` 与 L23 `reduce_scatterAicpuKernel`，每组均各唯一一条，依据记录的设备开始时间及 CSV 时长算交集。L23 是提取器通过 sidecar + host scope → flow → device → CSV 的层标签，不能当成该设备区间的排他层归属。结果不进行跨卡时钟相减：

| digest 前缀 | EP | L23 AI CPU 完整覆盖 L14 MC2 | 部分覆盖 | 无交集 |
|---|---|---:|---:|---:|
| 1b826883 | 8–15 | 6 | 2 | 0 |
| 8c18c293 | 8–15 | 8 | 0 | 0 |
| 7ddf7ae9 | 0–7 | 4 | 0 | 4 |
| db852bff | 0–7 | 4 | 0 | 4 |

例如新增的 `8c18c293…` EP8：L14 host scope `737-65` 选中 MC2 stream 47 / task 47512，CSV 时长 165416.395 µs；L23 host scope `737-74` 选中 AI CPU stream 92 / task 4255，CSV 时长 166481.293 µs，设备记录起点早于前者 373.132 µs，区间完整覆盖前者。两者的 host flow 起点分别处于各自所选主机层范围；此处未做新 DB 反查，也未将 L23 host scope 提升为张量或物理事务身份。`7ddf7ae9…` 与 `db852bff…` 在 EP4–7 为完整覆盖、EP0–3 为零交集，说明这种**已导出区间重叠在样本内并非所有卡普遍出现**；不能推断其他任务或未导出活动不存在。

这扩展了相同观测规则下的样本覆盖，但不证明两条任务同一 collective、同一请求或张量依赖，不证明真实链路持续传输、竞争、慢 MC2 的原因，也不能将两行时长相加为层排他耗时或端到端耗时。时间戳版本契约、采集完整性及未记录数据仍有限；AI CPU shape/dtype 均为 N/A。四个 digest 均来自已有 profiler 采集，而非线上基准等工作量独立对照。

复算：`python3 agents/diagnosis/check_cross_layer_aicpu_attribution.py > /tmp/cross-layer-check.json`。脚本核对输入 JSONL SHA `0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4`、每组唯一任务和 32 个配对，并使用 Decimal 对导出时间计算；结果 [cross-layer-aicpu-attribution.json](cross-layer-aicpu-attribution.json) SHA `61c8639fd775a2907ebdf31ce5103f974831df61da9ba8031216f346002b7f3f`，脚本 SHA `8e16a0d87bb14b452ee091f159d189e588993025a6fee33d947f46fbbe81db3a`。本节仅离线读取，不启动 NPU。

## 选定区间覆盖不是这些耗时的必要条件

**事后选样，意在寻找反例，并非随机样本或受控对照。**从上面的 32 对中选两组各含部分零交集和完整覆盖的 digest，对每卡再关联同 digest、同卡、host scope 对应的 L39 `DispatchFFNCombine`。每层同名任务在其 scope 中必须唯一；L14 身份、start、CSV 行、时长须与已有交集结果完全一致；每组各卡的 L14 和 L39 导出 shape、dtype、format、核类型、block 字段逐项一致。L39 是对照任务，不证明两个层实际有效数据或执行工作相同。

| digest | 选中零交集 EP，L14 时长 ms / L14÷L39 | 选中完整覆盖 EP，L14 时长 ms / L14÷L39 |
|---|---|---|
| `7ddf7ae9` | 0、1：25.501–25.562 / 2.645–2.648 | 4–7：25.573–25.605 / 2.637–2.644 |
| `db852bff` | 0–2：22.471–22.502 / 2.789–2.794 | 4–7：22.480–22.526 / 2.785–2.799 |

这说明这些选择的相近量级 L14 时长及其 L14/L39 差异出现时，**本卡该条已选择的 L23 AI CPU 记录覆盖 L14 区间并非必要条件**。这里有意排除零交集但 L14 时长较短的 `7ddf7ae9` EP2、3 和 `db852bff` EP3；表格不能代表全部零交集样本、整体 rank 或其他采集。完整 32 对的覆盖情况仍见上一节。上述接近不是因果对照：跨卡通信或等待、其他流、有效 token、路由、计算状态、未导出的活动、采集扰动及时间坐标契约仍未知，不能由此排除真实通信争用或任何总体的通信反事实。

复算 `python3 agents/diagnosis/check_mc2_selected_overlap_counterexample.py > /tmp/mc2-selected-overlap-counterexample.json`。脚本 SHA-256 `bd9c804d75a5f127cc679b1d62ce8b8fabd953706aa0276bdecfcc4243331a62`，验证上述两份输入既存摘要并输出逐卡身份、时长和选择规则，结果 SHA-256 `ece4f4f615365206f3dbfb8f11908d219dfca106a14fde919234bb3195e3a258`。只用离线数据，未启动设备。

## 另一个可比较口径：EP0–7 的 L23 提交记录时长

此前对 `7ddf7ae9…` 和 `db852bff…` 只按 L14 MC2 的相交与否分类；现在复用同一份已提取的 16-rank JSONL，查看 **L23 host scope 分别选到的 AI CPU 记录自身时长**。各 digest 独立分组，对 EP0–7 每卡恰好取到一条 `reduce_scatterAicpuKernel`，采集名和核心类型相同；来自既有同线程 scope → flow → device → CSV 的关联。该任务在 EP0 的 device stream 为 90，在 EP1–7 为 92，trace track 与 CSV stream 相符。CSV 中的输入 shape 与 dtype 均为 `N/A`，因此不能用它们证明工作量相同。

| digest | EP0–3 单条记录时长范围 | EP4–7 单条记录时长范围 | 后者最短 / 前者最长 |
| --- | ---: | ---: | ---: |
| `7ddf7ae9…` | 0.176–0.221 ms | 27.891–27.981 ms | 约 126 倍 |
| `db852bff…` | 0.204–0.259 ms | 24.815–24.891 ms | 约 96 倍 |

这是一次 profiler 采样中的事后分组，对比的是**每卡自身的记录持续时间**，没有跨卡时间戳相减。它补充了：即便名称、被标记的提交层和摘要一致，所选 AI CPU 记录也可呈现很大的 rank 间差异；此前 L14 同范围中部分 rank 的 MC2 时长仍相近。记录时长不是物理网络传输时间，长短也不能直接认作等待、异步隐藏或争用。相同名称、scope 和 digest 不证明这些记录是同一 collective、同请求或相同 payload；通信组、rank 的真实职责、实际 payload、系统状态、导出器以及 profiler 扰动均未绑定，无法据此解释差异原因或推断端到端成本。为判断是否同一通信事务，需要同次采集的 group、op ID 和连接标识等可绑定证据；当前不为此启动设备采集。

复算：`python3 agents/diagnosis/recheck_l23_aicpu_rank_durations.py`，脚本 SHA-256 `90b97ef77fb43de9d8487a35af8330edc8fbe19f38adcab2af442479bb96c2bf`；它验证源 JSONL SHA-256 `0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4`、每组每卡唯一、指定 stream/core、缺失 shape/dtype、CSV 和 trace 时长一致到 0.01 微秒。输出 SHA-256 `3cecd1ac380e38ec59c5c48b11c441d55a9f4367e54db69effdbba3349bef5fd`。只用既有数据，不启动 NPU、不修改推理代码。

## 两组 EP0–7 的通信关联：组与导出数量一致，但仍非事务证明

对上述两组各 8 张卡各一条 L23 选中 AI CPU 记录，以已有冻结 JSONL 的 rank、PID、stream、task ID 和设备起点，在对应 rank 的历史 profiler DB 只读查询 `TASK`，每条均唯一；再通过该任务的 `CONNECTION_IDS` 反查关联 `PYTORCH_API` 和 `COMMUNICATION_OP`。16 条每条在所查 API 表均关联一条 `HcclReduceScatter`，并在可达连接中关联一条通信 op。字段如下：

| digest | 每卡关联的通信 op 名 | opId（同库出现次数） | count（导出值） | dtype（同库枚举） | rankSize / 组 |
| --- | --- | --- | ---: | --- | --- |
| `7ddf7ae9…` | `hcom_reduceScatter__503_351_1` | 351（1） | 3046400 | 11 → BFP16 | 8 / TP |
| `db852bff…` | `hcom_reduceScatter__503_209_1` | 209（1） | 5237760 | 11 → BFP16 | 8 / TP |

两组中各卡 `parallel_group_info` 的 TP 列表相同（全局 rank 0–7），`group_rank` 所指成员均与各卡 `RANK_DEVICE_MAP` 的该 device 行一致。每库每个选定 opId 在 `COMMUNICATION_OP` 表中只有一行，降低了**同库**重复 ID 导致误匹配的风险。这些结果比单纯“名称、层号相同”更支持所选八卡记录关联同一组、相同导出 count 的同类通信；但数据库之间 opId/通信组生命周期、connection 语义及实际 runtime communicator 尚未闭合，不能断言八条就是同一物理事务、相同有效数据或同一请求，更不能从 AI CPU 时长推知链路传输耗时或因果。

连接集合存在多对多，不能当作天然一一对应。EP1 的 `7ddf7ae9…` task connection 关联两个 id，其中一个在所查 `PYTORCH_API` 表匹配 `HcclReduceScatter`，另一个无匹配行；后者关联的另一 connection 在同库 `CANN_API` 查到 `aclrtRecordEvent`。在该库 `CANN_API` 中通信 op 对应 connection 与 task connection 各有 `launch` 记录，导出区间呈嵌套关系；它们的语义和事件对象尚未绑定，不能把连接集合当成只含一个 launch 的完整事务。

仅作有条件的算术核对：`3046400 = 4760 × 640`；`5237760 = 8184 × 640`，而 `8184 = ceil(8180 / 8) × 8`。数值与此前的两组调度数量和 8 路行数推导相容，但尚未绑定实际 tensor、count 在该版本的确切单位、rank 的有效输入或 padding；不证明这些数即物理数据流量或实际补齐。

复现入口：`python3 agents/diagnosis/run_l23_rank_communication.py`。wrapper 验证 JSONL SHA `0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4`、stage 的容器/image 和 `/work` 挂载；容器中只读 SQLite、禁自动索引，分别计算 DB 读前后 hash 并确认无 WAL，逐组检查 16 条唯一关联、枚举、组和 count。DB 摘要在本次首次观察并保存在结果中，不是历史事先冻结的证明。结果 `agents/diagnosis/l23-rank-communication.json` SHA `f920ad422a443c28a1f34878c44472ec203618d5598380ef2de01b53edf32633`；wrapper SHA `3bed6799a0abb6c489b4b3b80ec65fbb552652fd36163ca27d69e97813597ac0`；容器侧查询源码 SHA `4924dbb1c6a6d4e0418d204aedf07b66da2ae673988c023ac50ef72a330e63d2`。未跨卡相减时间戳；无新 NPU 负载或推理代码变更。

## 两组所选通信 op 的现有分项导出

以上一节的 16 个 **同库唯一 `COMMUNICATION_OP`** 为锚点，在各自 EP0–7 的 `communication.json` 中以 op 名前缀查找，16/16 各有一个候选条目；每个 JSON 的 `Elapse Time(ms)` 与该库 op 的 `(endNs−startNs)/10⁶` 数值完全一致，JSON 显示起点与数据库起点相差不超过 0.233 微秒。JSON 浮点时间戳精度有限、名称后缀的组映射和 exporter 版本未绑定，这增加同库匹配可信度，但不证明跨库同一事务。JSON 文件本次首次记录 hash，并确认读取过程中大小、mtime、inode 与 hash 不变；没有历史冻结文件内容供对照。

| digest | Elapse ms 范围 | Transit ms 范围 | Wait ms 范围 | Synchronization ms 范围 |
| --- | ---: | ---: | ---: | ---: |
| `7ddf7ae9…` | 0.294974–0.308014 | 0.193136–0.213176 | 0.090098–0.106997 | 0.054480–0.068259 |
| `db852bff…` | 0.453031–0.481130 | 0.303414–0.343113 | 0.122878–0.165877 | 0.085680–0.108119 |

这些仅是 **通信 op 的导出字段范围**，不是 AI CPU 记录时长的阶段分解，也不是请求延迟。当前没有独立校准 AI CPU 轨与通信 op 轨的时钟与事件边界，本节不据两者的相对时间推导先后、数据依赖、等待或隐藏。同一数据来源的 JSON/DB 一致不是独立重复测量。不能将 Synchronization 与 Wait 自动相加；[MindStudio 6.0 通信分析文档](https://www.hiascend.com/document/detail/en/mindstudio/600/msug/msug_000227.html)说明对应标签的 ms 单位、通信和同步含义，但文档版本未证等同历史 CANN 9.2 采集与 exporter，不能由此声称字段互斥或无误差。

各条 JSON 都有 HCCS/SIO/SDMA 等标注，所选 HCCS `Transit Size(MB)` 在两组分别均为 20.3369140625 和 34.9658203125；这里仅记录字段数值，未验证 MB 单位基数、链路实测或 exporter 对 HCCS/SIO/SDMA 的重叠统计，**不能将这些类别相加当物理网络字节**。

每条 op 在本库 `COMMUNICATION_TASK_INFO` 中以 local opId 查得 166 或 168 行，globalTaskId 各自均不重复。EP2 的两条 op 均是 166 行（其中 66 条 `Notify_Wait`）；其余各 168 行（68 条 `Notify_Wait`）。该差异是有限导出覆盖/字段统计，不足以断言丢失、缺同步、通信异常或真实同步次数。名称、重复 notifyId、任务表关系和跨 rank 身份均不单独证明依赖或通信完成。

复算：`python3 agents/diagnosis/run_l23_comm_json_metrics.py`；入口验证既有 JSONL 与上一节匹配文件 SHA，限定 EP0–7 两个 digest、容器 image 和挂载，读取已存 DB/JSON；每库读取前后验证先前观察 DB hash、无 WAL、JSON 前后 hash 一致，限制 child 行数。输出 `agents/diagnosis/l23-comm-json-metrics.json` SHA `815d6a52fd16e2537bec9e849ed11574564cf1b6411fb9c515e489b8b7222e3c`；容器侧查询 SHA `f65097e8ef695a03c15867f735362d103d35392435a6caeafec78145d74588c0`，wrapper SHA `a4dc039536d420d3cf77cae3b6596aead78cb9c4210596c97b4790e20bfdf446`。结果仅为单次高扰动 profile，未运行新设备任务或更改推理实现。

## 同设备通信子任务起点与所选 AI CPU 区间：导出坐标复核

基于上一节已绑定的每卡所选 AI CPU `TASK` 与可达的 `COMMUNICATION_OP`，在各自数据库中按唯一 local opId 查询 `COMMUNICATION_TASK_INFO`。16 组中每组有 166 或 168 个子任务，`globalTaskId` 组内唯一；所选 AI CPU 自身的 `globalTaskId` 在对应子任务表中均无直接匹配。所有比较**只在同一 rank 的同一数据库**内进行，不跨卡相减，也不据此认定同一 replay 或物理事务。

| digest | EP0、1、2、4、5、6、7：最早子任务起点 − AI CPU 结束，导出坐标范围 | EP3：子任务起点落在 AI CPU 区间的行数 | 全 8 卡：最早子任务起点 − 通信 op 起点，导出坐标范围 |
| --- | ---: | ---: | ---: |
| `7ddf7ae9…` | +6.415532～+210.329657 ms | 3 | −0.439772～−0.156720 ms |
| `db852bff…` | +8.467911～+227.903226 ms | 13 | −0.674406～−0.098300 ms |

因此，**单看大通信 op 的开始时间不能排除更早的关联子任务**：16/16 所选组的最早子任务起点都早于大 op 的导出起点。另有 14/16 组中所查子任务起点都在 AI CPU 结束之后，余下 EP3 两组有部分起点落在 AI CPU 导出区间内；这只是对所查 local opId 子表的记录，不覆盖其他通信或未导出活动。起点落入区间也不能证明持续时间重叠、依赖或等待。

[CANN 9.0 的 profiler 数据库说明](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/devaids/Profiling/atlasprofiling_16_1144.html)将库内时间说明为本地 Unix ns，定义 `TASK` 起止、`COMMUNICATION_OP` 起止并说明子任务 globalTaskId/opId 的表关联。该公开版本并未证明本次历史 9.2 导出的各轨转换实现和误差。**条件性解释：**如果此次 AI CPU 与对应子任务起点的相对时钟误差已证明小于 6.415532 ms，那么上表排除 EP3 后的 14 组可判定所查首个子任务起点确实晚于该 AI CPU 结束；现有资料并未给出此误差证明，所以仍不能将结果升级为实际先后，也不能用它排除提前启动的其他操作、整体通信或依赖。要进一步确认，至少需要该次 exporter 的轨道时间换算与相对误差界，以及实际 replay/事务的身份与事件对象映射；不在当前只读阶段启动新采集。

复算入口 `python3 agents/diagnosis/run_l23_op_child_timing.py`：校验上轮 JSONL 与 16 项关联输入 hash、stage 容器和挂载，并在 stage 中只读 8 个固定库、检查先前观察的库 hash 与无 WAL；每组按 opId 精确取至多 2000 个子任务起点。结果 `agents/diagnosis/l23-op-child-timing.json` SHA `d20cf9de6f3b5a965ed31a16ca1d0be7873041796a2bba17782ec4bf99254af3`，容器查询 SHA `b0f6e7bb30a48138775ed397d4ce63320a8b85e45b265190a6b87996f9ee81df`，wrapper SHA `7ca528319f3cfd314ae7b342fa458ab62f0ff991b2ced776ec3ffff4f5387d45`。没有新 NPU 负载或推理代码改动。

## 现存转换代码与所选 AI CPU 的覆盖边界

本次定向查阅 stage 镜像内 profiler 的 Python 源（CANN analysis 759 个、torch_npu profiler analysis 90 个 `.py`；仅匹配表名及时间转换关键字，文件上限 1 MB，不含二进制实现），发现当前安装的 torch_npu `_trace_step_time_db_parser.py` SHA `7058eeef718b899657cd0738722e46e4d544c2dc9520375f22c21592130c5719`：第 141–184 行读取同库 `COMPUTE_TASK_INFO JOIN TASK` 的区间，以及按 connection 关联到单个 device 的 `COMMUNICATION_OP` 区间；第 93–110 行按 device 聚合并做区间 overlap 计算。这说明**该当前分析器在它的输入范围内将两类导出区间按同设备坐标比较**，本身并不提供硬件时钟误差界，也不能证明历史 10 月 29 日 DB 一定由该版本或这段代码产生。

为检查能否把此逻辑用于本节目标，在各自原库按 `TASK.startNs + globalTaskId` 精确重验上面 16 条 AI CPU 记录，均各唯一；随后按相同 `globalTaskId` 查同库 `COMPUTE_TASK_INFO`，**16/16 均无匹配**。因此此分析器的 compute 表路径不包含这些所选 AI CPU 任务，不能用其 overlap 实现直接证明所选 AI CPU 与通信子任务的跨轨时钟同步、物理顺序或误差；该缺失也不代表 `TASK` 不存在或设备没有执行任务。

同一 stage 的 CANN 9.2 `analysis/mscalculate/ts_task/ai_cpu/aicpu_from_ts_collector.py` SHA `b570f75742e1671fabc7002cc3f567b5f6426ac80ffc3421e141085c17f99cfc` 的 49–55 行，对该 collector 的一类 AI CPU TS 数据调用 `InfoConfReader.time_from_syscnt` 转换。已有 `sources/primary/frozen-image-clock/cann_info_conf_reader.py` 快照 SHA `044899552770c109ea9e18ad0282c097cccce22b22b564af2ce52ac9566b23e4` 第 269–280 行显示基于 device 计数器、频率及 host monotonic 锚点的转换；但本次所选 16 条 `TASK` 与该 collector、目标通信子任务生成代码/版本、锚点采集误差和漂移均未绑定，不能从源码形式推出统一时钟或数值误差上限。

若需要把上节条件先后提升为实际结论，最小补证是：取得该次采集实际生成 `TASK` 与 `COMMUNICATION_TASK_INFO` 的版本/代码或原始时间数据及转换锚点，给出同设备两轨**相对误差上界**，并绑定目标 op、子任务与 AI CPU 的 replay / event 实例；仅补一个同名算子或某条时间线并不足以完成身份与误差证明。需要新增设备采集时须另行取得授权；本轮到此为止。

复算 `python3 agents/diagnosis/run_l23_aicpu_compute_membership.py`：检查固定 JSONL 与前述关联结果 hash、容器及挂载，stage 中按已观察 DB hash 和无 WAL 只读查询；结果 `agents/diagnosis/l23-aicpu-compute-membership.json` SHA `35d3d4eb4049cdfe80e9b9efcd989b0ba6e0711ead97b6cf3f4752fd3b019ae5`，容器查询 SHA `0609ffe6b649cb1596b40a2794a81c1cbbf2e117e2987945ce26a03b722a0fcc`，wrapper SHA `e3d616d26cd7024ebf5de3cef99d4a4544990b5c478c48c2646f932e96eb4e5b`。已复跑并逐字节核对；未启动 NPU 或修改推理代码。
