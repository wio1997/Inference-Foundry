# F07：Q 投影与 partial RoPE 的分块和跨流边界

此报告是读取既有代码与采样后的候选审查，没有新实现或设备工作。为何选择 F07：`evidence/q-projection-16rank-diagnosis.json` 在 16 rank × 3 个已捕获内部采样轮 × 40 层中，按同形状和来源约束选出 1920 个投影任务，中位数 42.0 μs；这是频繁的大任务的优先线索，不证明每条记录都已精确绑定 `wq_b`、后接 rotary 的耗时、端到端暴露或任何收益。该报告的 nominal Cube 比值属于狭窄敏感性假设，不是可实现加速比。

## 源码边界和消费者

冻结路径相对 `sources/vllm-workspace/vllm-ascend/vllm_ascend/`：

- `attention/dsa_v41.py:350–389` 根据 `num_actual_tokens` 截取 query/cache，空 query 单独更新 cache 并返回 None。非空时先取得各自 cos/sin；普通分支经 `q_norm(wq_a(...)) → wq_b → unflatten`，多流分支通过 `multistream_preprocess` 获得 q 和 qr。无论上述哪种非空分支，在 `:374` 都对 q 原地执行 partial RoPE，随后传出 `q.to(query_hidden.dtype)`、qr 和可选 indexer 数据。
- `:708–711,657–682` 最终将 q 送 sparse attention；在有长上下文的 index source，`qr` 还作为 `indexer.project_query` 的输入（`:394–409`），不能将主 Q 的投影/旋转与 indexer 的输入误当同一张量。`_prepare_indexer_inputs` 使用对应 cos/sin 来旋转其各自 query；`dsa_v41.py:375` 特别要求 local Query RoPE 在 compression 复用 source tables 之前消费。`q.to(...)` 是否是别名或实际 dtype conversion 未知，重组不得无证据删除。
- `device/device_op.py:75–91` 的 `inplace_partial_rotary_mul` 使用 `interleave` 与 `partial_slice`，且修改目标的存储；按冻结模型配置 `head_dim=512`、`qk_rope_head_dim=64`，范围是 `[448,512)`，头内其他 448 维不旋转。`model.py:377–415` 在 CP 条件下 `wq_b` 是 replicated 的 `64×512` 输出，非 CP 为 column-parallel；实际进程对象和 tensor alias 仍需证据。投影输出的原有 dtype/舍入、每对旋转通道和正确 cos/sin、原地写的可观察副作用、attention 输入 layout 与所有 graph 地址须保持。

## 已有多流交错，新的重组位置

`dsa_v41.py:440–495` 在 `multistream_dsv4_dsa_overlap` 条件下已安排：main 流的 q norm/量化与 auxiliary 流 KV matmul 交错；然后 main 发起 Q_b matmul，auxiliary 做 KV norm/RoPE 和条件 SWA 写。main 在 `:492` 等待 aux，函数返回后 `_preprocess:374` 才执行 Q RoPE。必须保留这一条件分支、`kv_matmul_done`、cache 写和 attention 前的真实可见性。源码并未展示一个现成的 Q_b + Q RoPE 复合调用；不代表运行图 lowering 未融合，也不能把 Q_b 或 KV norm/RoPE 的已有工作重新计成新融合。

可研究两种**不同候选**：

1. **投影 epilogue 与 Q RoPE**：若输出 tile 含旋转配对所需的完整结果，量化 GEMM 先保持当前反量化/偏置和原中间 dtype 舍入，再在输出接合 RoPE；未旋转部分按原值写出。需要确认 GEMM 权重 scale/量化实现、各 N tile/head 的映射、旋转对齐、cos/sin 就绪和设备接口支持。当前结构允许讨论，但源码与采样未证明 epilogue 能执行或数值逐位一致。
2. **把 Q RoPE 从 aux join 之后提前至之前**：main 的 Q_b 完成、所需 cos/sin 已就绪，且 Q 与 aux 的 KV 缓冲、rotary 表及事件无冲突时，Q RoPE 数学上不需要等 KV 的结果；可条件研究与 aux 的 KV norm/RoPE/cache 尾部重叠，并在 attention 需要 cache 前保留 join。这是从现有显式等待位置导出的候选，不表示可简单删掉 wait，亦不证明两流物理并发或收益。KV 与 Q 的 V/MTE 资源可能争用；该改动还须保留 compression 前的 Q/indexer RoPE 顺序、跨流 tensor 读写与 graph capture 生命周期。

两者可分别考虑、也可共同考虑，但不能将同一个 Q RoPE 的潜在省时重复计账。条件会计例：假设 CP 路径、M=3、64 heads、旋转 64 个 BF16 数值/head，且原旋转分别从某存储层读取并写回全部旋转部分，则有 `2 × 3 × 64 × 64 × 2 = 49152` 字节逻辑访问可供审计。这不是已测 HBM 字节、必省流量或时间收益；cache/片上转发、实际 kernel 访问布局、Q 输出 dtype、tile 额外搬运及融合代价均未证。

## 仍缺什么

未捕获该候选的同一 replay 的 Q_b 输出 tensor ID、rotary 目标 alias、独立 rotary task 的具体输入形状/stream/耗时、KV aux 尾部和 event 的绑定、对应 quant GEMM 内核 epilogue 接口；仅按任务名称或时间相邻不能补齐。不能把额外 output 转换、RoPE、等待或 Q 投影本身的全部 42 μs 当成必可消除。已有资料足以登记条件候选；只有原 trace 含可关联字段时，才有必要做有界只读查询。


## stage126：运行配置及一个 rank 的九个历史窗口

`ascend_config.py:569` 的默认 `multistream_dsv4_dsa_overlap=True`；冻结启动参数未指定该开关，`dsa_v41.py:262,358` 只有配置为 true 且 `cache_hidden.shape[0]` 非零才进入多流。若无运行时覆盖且缓存有行，源码推导支持此分支；现有记录没有捕获该进程实例最终配置。

stage126 由 ZCode 只读已保存 EP8 CSV（exit 0、模型 `deepseek-flash` 验证），结果 `f07-q-kv-streams.json` SHA-256 `04f9f65cb91b5dbf76456b411923f1a8032e1d7897d93a66ca1aa7c49ab82ec5`；脚本、窗口、trace manifest 和 CSV SHA 经本机复核。三个内部轮次 × 层 0/2/24 = 九个已归属 target 窗口，均有同模型的以下候选，均通过形状和窗口内时序确定，**不是直接 tensor 绑定**：

- 主流 89 中一个 `QuantBatchMatmulV3` 输入 `3,1280;1024,80,16,32;...`，与 Q_b 的配置/已往筛选相容；主流同形状 `[3,1,64,512]` 的 `InplacePartialRotaryMul` 在 attention 之前恰好一次，attention 之后另一次不能混入主 Q RoPE 统计。
- 另一流 87 中有 `InplacePartialRotaryMul` 输入 `[24,1,1,512]` 和带 `[24,512]` 更新的 `ScatterNdUpdateSk`，与 CP 的缓存侧计算相容；它们与主流同形状 Q_b task 在设备同 rank 上时间交叠，支持该样本有多流活动，但不自动证实算子属于同一源码调用或开关真实值。
- 九个窗口的 scatter 在 Q_b 完成后还有 10.540–12.140 μs，主流 pre-attention RoPE 自身 3.860–4.460 μs；scatter 结束至该 RoPE 开始为 0.540–0.820 μs。时间间隔使用同一 rank 的设备时间。`summarize_f07_streams.py` 对形状、同模型及源角色条件下的唯一性检查并可复算，输出 `f07-q-kv-stream-summary.json` SHA-256 `0f87c2b80f99c62225c9b422dd9658f4d8faf38728190324eea99c6a2eb31137`。

观察顺序与多流源码的“Q_b 与 aux KV 操作交错，等待 aux，再做 Q RoPE”相容，且表面上存在候选时隙；但没有事件 ID / wait 的精确关联、tensor 地址或实际执行分支的直接证据，不能把散点时序称作已证的因果或实际等待时间。提前 Q RoPE 是否能和 scatter 共存、是否因资源争用拉长散写、最终 attention 路径是否缩短，均未知；更不能将 10–12 μs 归为可删除或声称直接提升 4 μs。9 个窗口只来自一个 rank、三个层和三个内部采样轮，不代表全部负载。

## 事件证据的初步范围

现有 EP8 profiler `TASK` schema 有时间、stream、task ID、model 和 connection ID，没有明确的 event-object ID 字段。之前 `agents/diagnosis/ep8-trace-event-args.json` 只针对 Engram 相关时间点，所选 `EVENT WAIT/RECORD` 参数没有提供事件对象 ID；该结论**不能外推到 F07 的九个窗口**。后续只读 `query_f07_event_objects.py` 已在同一 EP8 九窗口、主流和候选 KV 流以及 Q_b 到 Q RoPE 的时间区间取得事件 task 和 trace 参数；结果见下节。即使观察到事件及相邻时间，缺少事件对象身份或明确 graph 依赖仍不能证明 record 与 wait 配对或该等待由 KV scatter 导致。

## stage129：事件记录支持时间线，但尚不能建立事件对象配对

ZCode stage129 只读了同一个 EP8 保存的数据库、CSV 和 trace（exit 0、模型 `deepseek-flash` 验证）；`f07-event-object-evidence.json` SHA-256 `12ba1743e2d8914b93572eadb485eb712c40ea6c8608e86167dc0e7be187567a`，本机核对查询脚本、窗口和三份原文件的记录 hash 与已有冻结值一致。仍是 3 轮 × 层 0、2、24 的 9 个 source-window 和形状候选，不是全负载或其他 rank。

每窗均有 3 个关联时间段的导出 task：主流 89 的 `EVENT WAIT`、另一流 87 的 `EVENT RECORD` 和该流之后的 `EVENT WAIT`。九个主流 wait 的导出持续时间为 **9.980–11.560 μs**（中位 11.179 μs）；对应主流 pre-attention Q 形状 RoPE 在主流 wait 结束后 0.020–0.240 μs 开始。另一流 `EVENT RECORD` 在候选 scatter 之后，主流 wait 结束时间比该 record 开始晚 0.500–0.600 μs。此前看到的同窗 Q_b/scatter 时间交叠和这些记录一起，与源码中的跨流 join **相容并强化候选解释**。

27 条 DB 事件 task 均在导出 trace 中找到唯一名称、stream、model、task ID 一致的匹配，27 个时间戳恰好相同；查询允许 ±2 μs 的匹配容差，因此这是关联证据，不升级为无条件身份或因果证明。匹配事件 `id/id2` 均为空，经过限定键过滤后的数字标识只有 `Batch Id`、`Model Id`、`Physic Stream Id`、`Subtask Id`、`Task Id`、`connection_id`，没有 event-object ID；所有样本的 `connection_id=184330` 共享，不能作为配对事件的唯一标识。没有取得明确的 graph record→wait 依赖，因此目前**不能证明**另一流 record 与主流 wait 是同一个事件、候选 scatter 就是被等待的写入，或者这个 wait 的全部持续时间可被改动消除。`EVENT WAIT` 是 profiler 标签和时间区间，并非排他空闲周期的直接计量。

这次结果可以把“主流中无等待证据”排除在所选九个窗口之外，但仍须保留 tensor 身份、事件关联及优化收益 UNKNOWN。查询只读历史数据，不执行设备任务。

## 固定基线的 16 rank 交叠覆盖（2026-10-30）

为检查上述 EP8 九窗的现象是否只出现在单卡，复用同一冻结基线、16 rank × 3 个内部轮次 × 40 层的 1920 个目标窗口。按每卡本地导出时间、同一 model 49、任务完整落入对应窗口：主流中选 stage202 的完整物理签名 `[3,1280]` INT8 × NZ 权重 → BF16 `[3,32768]` 的 `QuantBatchMatmulV3`；其他数字 stream 中选输入形状末尾为 `;24,2;24,512` 的 `ScatterNdUpdateSk`。所有窗口均恰有 1 条前者和 1 条符合条件的后者；**1920/1920 的两个导出区间相交**。窗口、manifest、签名摘要与 16 份 CSV 均逐项校验既有 SHA-256。匹配范围是已导出的指定形状和类型，不证明没有其他 scatter 或未导出活动。

所选记录中主流矩阵任务时长的中位数为 **42.000 µs**，scatter 的中位数为 **34.319 µs**；对每一窗先计算两个记录交集，再取中位数，得到 **22.8295 µs**（范围 7.120–30.000 µs）。scatter 结束均晚于矩阵记录结束，尾部跨度中位 **11.600 µs**、范围 9.180–19.200 µs。对每一对计算 `矩阵时长 + scatter 时长 − 交集` 后再取中位数为 **53.559 µs**，而逐对时长之和的中位数为 **76.179 µs**；中位数各自可能来自不同窗口，不能互相相减或相加。16 rank 各 120 窗均有同类区间交叠；这些观察来自同一次 profile，不是独立重复实验。

这为 F07 的重组约束提供更广的**时间线证据**：评估将某个候选跨流阶段移到主流时，不能简单把现有两条记录的时长相加，也不能先假定原有交叠仍然保留。然而，两个已导出区间相交只说明记录在同一设备导出时间坐标上有交集；跨轨时钟契约和采集完整性尚需核实，也不能证明同时执行的硬件资源、张量或模块身份、event 依赖、是否争用、是否处于请求关键路径或可实现的提速。前述源码条件支持潜在路径，但历史执行分支仍未绑定；没有计算、融合或调优变更。

本地脚本 [query_q_scatter_16rank.py](query_q_scatter_16rank.py) SHA-256 `1a3c934fe5fecc0516b5c841f6e87609f5eb070ad33bf3d446d8be9dc549f858`；只读查询结果 [q-scatter-16rank-overlap.json](q-scatter-16rank-overlap.json) SHA-256 `4d58d6c00eadb7c146670be2ea49157a1beb666d5745e7fe06859c4ce9c8fb7a`，逐窗保存两任务 ID、duration、交集、尾部和 16 份 CSV 哈希。独立按旧 EP8 九窗结果复算任务 ID、时长、交集、尾部 **9/9 一致**；按 rank 计数均为 120 窗。本机 ZCode 另核对九窗中的 `[3,4608]` 目标：所选其他流事件没有与它相交，与本机复算一致；这仅覆盖所选事件，不能推广全部流。本次 16 rank 查询由 GPT 编写，stage 容器只读历史 CSV，没有操作 NPU。

### 16 rank 同窗 scatter 尾部与主流 pre-attention RoPE 候选

在上一节相同的 1920 窗、同模型主流中筛 `SparseFlashMla` 及输入形状以 `[3,1,64,512]` 开头的 `InplacePartialRotaryMul`；限定前者**在窗内唯一**，而 RoPE 候选必须在它开始前结束且**满足该条件的候选唯一**。这按 F07 的形状与阶段规则选取 pre-attention 候选，不因名称推断所有 RoPE 都是同一角色，也不将另一条同形状记录自动解释为 post-attention 反旋转。每窗分别复核原 Q/scatter task ID、Q 时长、交集和尾部均与前一查询一致；16 CSV 及前述窗口、清单、签名和旧结果 SHA 均一致。

| 选定窗口 | 每窗 attention / 同形状主流 RoPE / pre-attention 候选数 | scatter 结束至候选 RoPE 开始 |
|---|---|---|
| 1920，16 rank × 3 轮 × 40 层 | 分别均为 1 / 2 / 1 | **全部为正**，最小 / 中位 / 最大 **0.520 / 0.620 / 1.020 µs** |

每个 rank 各 120 窗满足这些选取条件；每个内部轮次各 640 窗也均为正。RoPE 候选记录时长中位为 **4.020 µs**。没有以跨设备时间戳计算差值：差值来自单卡 CSV 的不同设备流记录，跨流时钟与导出边界仍须校准。这一有界集合比旧的 EP8 九窗更广，显示同一导出坐标里候选先后稳定；并不能证明 RoPE 等待 scatter、二者共享 cache、一个事件导致另一事件或存在可释放的时间。两个候选来自同一层窗，不保证同一 tensor、源码角色或完整请求；未选行和未导出活动仍可能影响结果，三轮属于同一次 profile，也不能说明所有线上请求如此。

查询 [query_q_scatter_rope_16rank.py](query_q_scatter_rope_16rank.py) SHA-256 `d544eea8c9c7eb886f763b6570c39266bbd8bff6035d8deeaca5528b712f5974`，逐窗结果 [q-scatter-rope-16rank.json](q-scatter-rope-16rank.json) SHA-256 `ef1a5fbd918defea6e433708e6181e163897505b0bd1ac5195667fc2c337946a`；16 rank 各 120 条与选取分母均已复核。另独立对旧 EP8 九窗复算候选 task ID、RoPE 持续时间、scatter→RoPE 间隔，**9/9 与既有记录一致**。本机 ZCode 对已保存九窗间隔算术复核得到 9/9 正、范围 0.54–0.82 µs；此验证不是对 1920 窗因果的证明。仅只读 stage 容器既有 CSV，未启动 NPU 或修改推理代码。

### 对候选时隙逐窗复核：长度充足不等于当前可执行

将上面两个同源聚合按 `(rank, run, layer)` 做唯一键连接，分别核对 1920 条键无重复、键集合相等、16 份输入 CSV 的哈希表完全相同，并逐条复核 `Q 结束到 RoPE 开始 = scatter 尾部 + scatter 结束到 RoPE 开始`。对每一窗计算 `scatter 尾部时长 − pre-attention RoPE 记录时长`，**1920/1920 为正**，最小 / 中位 / 最大为 **4.899 / 7.360 / 14.940 µs**。例如这些统计说明，在既有导出时间坐标和所选候选下，仅从两个区间长度看，每窗的 Q 结束到 scatter 结束区间足以容纳该窗的 RoPE 记录时长。

这比只比较两个总体中位数更严格，但不是可执行性或收益证明。冻结多流源码 `multistream_preprocess` 在 Q_b 之后明确 `main_stream.wait_stream(aux_stream)`，返回后 `_preprocess` 才发出 Q RoPE；当前源码调度本身**有等待顺序**，不能宣称现有执行可以直接插入该时隙。重组须另证 Q、cos/sin、缓冲和所有其他依赖已就绪、允许调整等待位置且保持 attention/cache 可见性；特定 `write_cache_on_main` 分支还会在 join 后由主流写 cache。即使重排合法，资源争用、其他未选任务及计时扰动也可能吞掉全部收益。所选 Q、scatter、RoPE 的角色仍只是形状和同窗候选，不是 tensor 或事件的绑定；跨流时间戳是否有严格可比较的时钟契约也未核实。这里将同卡导出坐标的相对时间用于条件性账本，不解释为硬件可见空闲或必可消去时间。

复算时从 [q-scatter-16rank-overlap.json](q-scatter-16rank-overlap.json) 和 [q-scatter-rope-16rank.json](q-scatter-rope-16rank.json) 读取 `samples`，使用上述三列键连接，在每窗对 `scatter_tail_after_q_ns - rope_duration_ns` 求最小、中位及最大。两文件 SHA-256 分别为 `4d58d6c00eadb7c146670be2ea49157a1beb666d5745e7fe06859c4ce9c8fb7a` 和 `ef1a5fbd918defea6e433708e6181e163897505b0bd1ac5195667fc2c337946a`；无跨卡时间相减，仅 CPU 读取已有材料。
