# SP attention 输入 gather 与 Q/KV 投影的跨模块重组候选

## 本次增量和范围

既有 F02 仅记录 attention input RMSNorm 后的 SP all-gather 是必须考虑的边界，F05/F06 研究的是 attention 内部 Q/KV 投影与量化。此处考察一个未在 F01–F31 作为独立结构列出的**前移 Q_a/KV 投影、改变 token 轴上通信对象**的候选；与 F02/F06 的同一段工作有重叠，不可累计成独立收益。纯源码条件分析，不代表实际性能排序或可运行实现。

冻结 `model.py:879–905` 中：HC pre 的混合 x 先经 `input_layernorm`；SP 条件下 `sp_all_gather(x)[:positions.shape[0]]` 后送入 `self_attn`，输出再 reduce-scatter，并与保留下来的原 HC residual 和系数经 HC post 合流。此 decoder 方法在 self-attn 返回后不再显式读取这份已归一化并 gather 的 x；它仍保留**另一份**层入口四路 residual 和 HC 系数。不能把其余 residual、跨层辅助状态或返回值删掉。

`model.py:390–416` 声明 `wq_a` 为 `ReplicatedLinear(hidden_size,q_lora_rank)`，`wkv` 为 `ReplicatedLinear(hidden_size,head_dim)`；`dsa_v41.py:350–389,440–494` 的正常 preprocess 分别由 query_hidden、cache_hidden 产生 Q_a 和 KV，后续 Q norm/Q_b、KV norm/RoPE/cache 等工作仍存在。ReplicatedLinear 的类名是**复制语义的源码线索**，不是历史各 rank 参数实值、量化 scheme、TP/DP 组或二进制已绑定的证明；169 描述哈希对应的本地 `source172/model-metadata/quant_model_description.json` 对这两个投影标 `FLOAT` 也只是文件标签，不等于实际 runtime 一定无动态量化。全量消费者、hook、custom op 附加副作用和 graph 的 tensor 使用不能只由这一 Python 函数概括。

## 可能结构和行域条件

对某个满足条件的 non-source 层，若 all-gather 沿 token 行拼接、每个投影是对应完整 hidden 行的行独立函数、通信组内权重和其他参数一致，且以改变后的形状执行得到与原要求一致的输出，可研究先对本 rank 的 `input_layernorm` 行执行 Q_a/KV 投影，接着通信并恢复各 rank 实际需要的行；再进行原有 q_norm、Q_b、KV norm/RoPE 与 cache 等步骤。不能把数学上 `project(gather(x)) = gather(project(x))` 当成设备逐位等价：GEMM 分批形状/tiling、FP32 累加、BF16 输出舍入、量化/scale、rank 上权重对象、padding 及 compiler 可能改变结果。该研究需要重构 custom-op 的输入/输出接口；现有实现并不提供这一方式。

- **普通路径**：`_preprocess` 使用 query/cache 两份输入，两者可能是同一 hidden tensor 的不同视图；需保证每条输入行到各 rank 的还原、完整 KV 写入、空 query 时仍处理非空 cache 以及全量 attention 的原可见行为。即使没有直接数值消费者，原 hidden 的 shape/dtype 仍被输出分配、slicing 等使用；调用侧必须保留或用等价 metadata 替代，不能宣称无需原 tensor 的全部语义。
- **CP 路径**：`attention/context_parallel/dsa_v41_cp.py:189–207` 把 hidden 的局部后缀作 query、完整 hidden 作 cache；两个行域不同。全组 gather 两个完整投影可能额外传输无用的 Q_a；只能先建立 SP 行所有权与 CP query/cache 分片的准确映射，再决定实际需要传哪些行。另需满足 source 层、head 交换及输出 layout 的全部条件。仅凭同 rank 编号不能证明组或所有权一致。
- **PCP 路径**：`:445–482` 针对特定 metadata 另做 PCP all-gather 并按 `hidden_restore_idx` 恢复全局 cache hidden，空 query rank 也参加 hidden 和权重 collective；这个额外消费者不能略去，必须在各通信组建立 token 的排列映射并确保 collective 次序匹配。改变或移除它需要新的等价证明，不得推广普通 non-PCP 分析。
- **source 层（已按下文重新区分）**：配置 index source 为 2、8、14、20、24、28、32、36；KV source 为 2、8、14、20。legacy CP 的 indexer weights 和 query 使用 local query hidden/qr；不能因所有 index source 都需要 indexer 就一概认为必须收集完整 hidden。KV source 的 compressor 则读取 global cache hidden，直接只传 Q_a/KV 不足以生成原 compressor 状态；也可研究预先投影后通信，但须保留全局数据依赖和语义。具体详细条件、空 query 分支、类型和可见性见下文 source 层纠正章节。其余层也仍需检查所有消费者与运行绑定。

若任一投影实际走 W8A8 等动态量化，必须绑定 scale/offset 的确切输入域：逐行归约在同一行完整到达且使用相同舍入、饱和和异常值时才可能独立于 batch 切分；若 scale 跨 token/组，或 quant 与 GEMM 的 packing 随分块变化，则不能把 local shard 的 scale 当原 full batch 的 scale。原 FLOAT 标签不足以直接排除这些替换、包装器通信或实际精度；wq_b/QLI 自己的动态量化也不能被忽略。

## 仅作设计比较的逻辑载荷

保存配置 hidden=5120、q_lora_rank=1280、head_dim=512。**假设**三种值均为稠密 BF16，每个全局逻辑行单次物化输入是 `5120×2=10240 B`，两个投影逻辑输出合计 `(1280+512)×2=3584 B`，约 35%。这是不同 tensor 的逻辑大小，不是 collective 网络传输字节、净节省、带宽时间或下界：它忽略 TP/DP/CP group、实际投影 dtype、远端 row 需求、padding、打包、两次 collective 的轮次和启动、缓存/重传，以及在 source/PCP 分支仍需 hidden 的额外成本。传输 Q_a/KV 还可能改变已有 Q/KV 的 Cube/Vector 多流并行及权重读带宽，导致收益消失或负收益。

## 适用证据与缺口

现有采集体现的 SP 真实对象、通信 group 的 rank/路由、该层规范化张量与 gather 入参/投影结果 storage、每层 weight/value 一致性、实际 dtype/scale、跨 stream 和 graph 绑定尚缺。具体历史采样中不能仅用 group 大小、同形任务或时间邻近证明上述条件。下一步应优先在已保存的运行对象/graph/trace 中定位同层 `sp_all_gather` 的 producer/consumer storage 或 flow，以及该层 CP 分支、通信组和投影 scheme；没有这些资料就保持 UNKNOWN，不启动额外 NPU 测试。该候选仅供后续评估，未修改目标实现。

## 量化解析与已存 EP8 行域证据

对 `source172/model-metadata/quant_model_description.json` 的本地副本重新核验 SHA-256 `2f4020a0a12daa621cfd2ff9d09a2c73e5674af7c624ef47df5c66796c3774bf`；`quant-metadata-summary.json` 和此前 169 容器只读对照分别显示，40 层 `layers.N.attn.wq_a.weight`、`layers.N.attn.wkv.weight` 都为 `FLOAT`，副本和当时 169 文件摘要相同。`evidence/model-config.json` 是维度/角色配置，不是量化描述。

保存的 `configs/modelslim_config.py:185–198,213–214,265–318,550–568,655–706` 为 `deepseek_v41` 定义 runtime 到 checkpoint 前缀/子串映射；如果实际前缀、映射、所用描述和该源码的 `LinearBase` 类型确实对应 `model.layers.N.self_attn.wq_a/wkv` → `layers.N.attn.wq_a/wkv`，其 `FLOAT` 在 `get_quant_type_for_layer` 返回 None，`get_quant_method` 会选择 `AscendUnquantizedLinearMethod`。保存的 `ops/linear.py:100–139` 的该类 `apply` 调 `torch.ops.vllm.unquantized_gemm(x, weight, bias)`。因此这比单独看到 FLOAT 标签更具体地支持**源码条件下的非量化方案**；没有绑定历史进程配置对象、mapper 是否及如何预先变换了整个字典、权重对象、运行入口或二进制，不能声称历史每次都命中。普通 unquantized GEMM 的行独立性是它的数学接口结构，但同一行在不同 batch 的设备实现可能选择不同 tiling、归约和舍入，不能据此证明逐位相等。

若另一个配置/对象实际命中 W8A8，冻结 `cv_linear.py:93–155` 在无通信且识别动态 W8A8 时调用 `npu_dynamic_quant(x)`，所得 `pertoken_scale` 送 `npu_quant_matmul`，并指定 output dtype 为 `weight_scale.dtype`；`w8a8_dynamic.py:104–132` 的直接 `apply` 同样调用动态量化，但指定输出为原始 `x.dtype`。源码的注释及变量名说明预期 per-token，尚未证明当前 NPU 实例的精确归约维、全零与特殊值行为和批处理/rounding 恒等；更不能因为输出接口名称相同就把 wrapper 和 scheme 的 dtype 语义混同。需要检查真实 scheme、scale shape 与值域、同版本 native quant 合同及 tiling 才能应用。

现有 `f07-q-kv-streams.json` 的 EP8 layer 0 run 2：main stream 89 task 27 的 RMSNorm 输出 BF16 `[3,5120]`，task 31 的 MatMul 从 BF16 `[3,5120]×[1280,5120]` 得 `[3,1280]`；另一个 stream 87 task 3 的 MatMul 从 BF16 `[24,5120]×[512,5120]` 得 `[24,512]`。这些物理 shape 和 dtype 与 CP 下局部 query、全局 cache 及 FLOAT 投影预期相容，也说明两类已选记录的行数不同；没有 tensor storage、Python callchain、collective 参与组和完整 trace event 身份，不能将它们绑定到该层源码变量、证明 SP gather 是否发生或确认其前后数据依赖。现有 F07 查询只列指定 operator 类型，因此不能因记录没包含 all-gather 推断它不存在。既有采样没有升级 F32 为真实运行可用或证明大小比较适用；后续只读定位调用级张量/流 event 和通信组是必要条件。

## Legacy CP 和 SP 在同一 TP 组上的条件行域对齐

本轮只读取 169 自己的 stage 容器源码，不触发 NPU，也只读取 baseline 容器同一文件的哈希。`sequence_parallel.py` 在两容器 SHA-256 均为 `73bcf07050bce8b049dd0bbdb0a7504f6eba157a1e97a5033849ef31ef05fcb0`，stage 源码快照 `sequence_parallel_stage_snapshot.py`。这只说明检查时文件内容一致，不证明历史运行已加载、未被 monkey patch 或采用同一个 device communicator。

快照的 `sp_shard` 调 `get_tensor_model_parallel_world_size/rank`，将 token 轴尾补齐到 TP 倍数，再按 rank 连续切块；`sp_all_gather` 优先调用 `get_tp_group().device_communicator.custom_all_gather`，如不可用或返回 None 则调用 `tensor_model_parallel_all_gather(x, 0)`。冻结 `dsa_cp.py:1139–1155` 的 legacy CP `_local_token_range` 直接使用 `get_tp_group().world_size/rank_in_group`，对 `common.num_input_tokens` 以相同 ceil 规则算 `start,end,per_rank,padded`；`dsa_v41_cp.py:110–153,188–192` 用 local metadata 的实际区间处理 query，同时将完整 hidden 交给全局 cache preprocess。

因此，在 **SP 切分及该时刻 local norm 输入仍保持原 token 顺序、两处使用相同 T、TP rank/world 映射一致，且实际 all-gather 按 rank 顺序拼接** 时，本 rank 的 SP 局部行和 legacy CP 本 rank 的 query 区间按相同公式对齐。在此特定条件下可以研究本地生成本 rank 的 Q_a，**不需要先把所有 rank 的 Q_a 全量 all-gather**；全局 cache 的 KV 仍需取得所有有效行所需信息。该结果比“两个 group 未知是否相同”的初步表述更具体，但并未证明历史满足条件：实际 producer 的 token 顺序、T 与 metadata 的绑定、custom gather 实例、rank membership、tensor 和执行图均未知。stage 现有 vLLM `base_device_communicator.py:247–270` 的常规 all-gather 代码使用 `dist.all_gather_into_tensor` 和 rank 维 reshape；这不能替代实际 custom 入口、后端及历史 group 的绑定。按该快照 legacy DSACP 和 PCP 同时启用会抛错；PCP 另有 gather/restore 路径，不能套用此结论。

例如假设 T=24、TP=8、无 padding，则 CP r 的 query 行与同号 SP shard 都是 `[3r,3r+3)`。已有 EP8 layer0 的 `[3,5120]` Q 投影形状与 `[24,5120]` KV 投影形状和此算术相容，**不证明**真实 T=24、TP=8、成员/排列及调用身份。若 T=5、TP=8，则 padded T=8、per-rank=1；rank 5–7 的 CP 实际 query 为空，但它们的 padded SP shard 非空，是否执行投影及各 collective 必须依赖有效行规则，不能让空 query rank 缺席仍需要它参与的 collective。该例仅是源码公式的构造，不代表历史样本。

反例显示“只算本地 KV 且不取得其他 rank 的信息”不能满足源码中复制 cache 的一般接口：假设 TP=2、每 rank 各一有效 token、KV 投影为标量恒等、两行分别为 0 和 1；rank0 原路径要为全局 cache 行输出 [0,1]，若新的路径只接收本地行、又没有远端任何等价信息，则对于远端行 0 与 1 两种情况看到相同输入却需要不同 cache 值。仍可研究以投影后的 KV 或其他经证明等价的信息通信，或利用已知不变状态；该反例不证明特定物理 all-gather 必要，更不表明历史通信量或性能。换分块须完整保留位置/slot、写顺序、padding、空行和缓存生命周期。

## 当前基线的配置门槛与通信顺序证据

`evidence/frozen-serve.sh` 记录的启动命令请求 TP=8、DP=2、EP、`enable_flashcomm1=true`、`enable_dsa_cp=true` 和 DSpark；这是启动请求，不是 worker resolved 状态。已有 `CONFIGURATION-REVIEW.md` 对上游及 Ascend patch 两套 `use_sequence_parallel_moe` 谓词的核查显示，在该模型配置、`allgather_reducescatter` 默认 backend 且无覆盖时两者均为 true；实际 backend 和属性对象未读取。`ascend_config.py:795–806` 最终将 DSA CP 设为请求值 AND 模型有 `hf_text_config.index_topk` AND `use_sequence_parallel_moe`，`get_v41_cp_classes()` 还要求 PCP 不同时启用。若当时确用该源码且选择 legacy CP，则 SP 是一项必要配置条件；反之只凭请求不能确认 CP 选择或本次样本的 resolved 分支。

华为 [CANN Community 8.5 集合通信说明](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/850/commlib/hcclug/hcclug_000009.html) 指出 HCCL AllGather 将各输入按**通信域 rank ID 升序**拼接。PyTorch [当前 distributed 文档](https://docs.pytorch.org/docs/stable/distributed.html) 描述 all-gather-into-tensor 要求每 rank 相同输入尺寸并生成拼接或堆叠输出。结合 stage 的 vLLM 常规实现用 `dist.all_gather_into_tensor` 并按 rank 维重排，可为先前假设的常规 gather 顺序提供接口级支持；但这些是不同版本公开合同与当前 stage 源码，不证明历史采集采用相同 9.2 实例、同一 communicator 的 rank 编号/成员、custom gather 实现或 SP/CP 真正共用相同 token 起点。需特别区分 communicator 内 rank、全局 device rank 和 EP rank，不能以设备标签直接代入。

现有 `MC2-EP3-L36-NOTIFY-INTERPRETATION.md` 在同一个诊断 EP3 数据库把一条 `HcclAllGather` 与 host 栈的 `sp_all_gather`、`model.py(1563)` 连接；若该历史栈与冻结源码对应，它位于 DSpark 辅助 hidden 状态的 gather，提供那个调用走 SP 的局部证据，但**不是** `model.py(898)` 的 attention 输入 gather，不能用于绑定 F32 的投影及张量或判断每层都执行。旧 EP8 Q 3 行 / KV 24 行仍只属形状相容。

曾对该 EP3 数据库尝试精确定位 `model.py(898)` 调用栈；初始全库重复扫描在 45 秒观察时限内未完成，检查确认其 CPU 查询进程仍存活后，为避免长期占用资源显式终止了本任务创建的查询进程。该尝试没有得出匹配数量，不能把超时或未返回当作“未命中”。后续若确有价值，应从已绑定的指定层、host API 或 connection ID 入手做单次有界查询，而不逐 callchain 重扫大表。现阶段对 F32 的判定仍是**条件性可研究、当前基线是否实际命中未知**；不能判定已排除、可实施、节约网络字节或可测得提速。

## EP3 层 36：同一诊断 scope 内的输入通信与注意力

限定已有 EP3 诊断 trace，以之前已绑定到 MC2 的 host connection `431408` 为锚，在同一 host thread 向前 100ms 只查 `HcclAllGather`、`aclnnMatmul`，并检查连接及其包含 scope。`query_f32_ep3_layer36_host.py`、`query_f32_ep3_layer36_scope.py` 分别输出 `f32-ep3-layer36-host-window.json`、`f32-ep3-layer36-scope.json`，脚本 SHA 记录于 JSON。SQLite 各查询均设 25 秒 progress 限制，完成后关闭；无请求值/权重读取与设备操作。

`wio_diag:DeepseekV41DecoderLayer.forward:2128-42` 同时包含候选 host `HcclAllGather` connection `431371` 和 MC2 host anchor `431408`；`sidecar-hierarchy-20261029.json` 同一次进程 `2128-42` 归属 EP3、layer=36、同一 scheduler digest，角色标记为 `conditional_target`。候选 gather 栈中有 `sequence_parallel.py(29):sp_all_gather` → `model.py(898):forward`；冻结 `model.py:898` 是 decoder 的 attention input gather。作用域内还有 431372、431373 等 `aclnnMatmul` 栈包含 `model.py(899):forward`；这证明**相同已归属 decoder host scope 内出现与所考察路径相符的调用栈**，不证明这些 MatMul 的输入输出是 gather 结果或分别是 Q_a/KV。

`CONNECTION_IDS` 中 host 431371 有两个 flow（102373、102386）；前者匹配到一条 `COMMUNICATION_OP`，opId=246、rankSize=8、count=5237760、dataType=11。其 count 数值恰为 `1023×5120`，与该 scope 下其他已存 `[1023,5120]` 形状相容，**不据算术巧合认定张量身份或物理网络流量；dtype 和 count 的后续限定解析见下文**。该 op 的 groupName 是数据库符号 ID；rankSize=8 不独立证明该组等于 TP group 或建立每 rank 的成员映射。一个 host connection 链到两个 flow，不能不查第二个 flow 即假定一一对应或排除其他任务。host 与 device 的记录时间、同步与 profiler 对齐限制不因该关联而消失。

该证据比先前只知 SP 一般适用更强：它支持历史诊断样本的特定 conditional_target 层出现 attention 输入 SP gather；它仍不能证明历史源码二进制与快照相同、Q/KV tensor 的直接数据依赖、组内 token 顺序、运行 shape 或 F32 可以安全重组和收益。总体服务基线也不能单凭此一次带 profiler 的诊断样本认定每层相同。

## 纠正：index source 与 KV source 的不同原 hidden 消费域

之前把 8 个 index source 都列为可能必须保留**全局 hidden gather**过于宽泛，现以已冻结 legacy CP 和非 PCP 代码分清：

1. `dsa_v41_cp.py:188–196` 将 `hidden_states[start:]` 作为 query_hidden，完整 hidden 作为 cache_hidden；`dsa_v41.py:350–355` 再用 local/global 各自的 `num_actual_tokens` 截断。`_prepare_indexer_inputs(query_hidden,qr,...)` 中 index query 从 local `qr` 投影，index weights 从同一 local `query_hidden` 投影（`:394–409`）；`_select_sparse_indices` 在 index source 分支使用这些准备结果。若前述 SP 本地行与 CP 本地 query 行和所有消费条件已证明对齐，则 index source 所需的原 hidden **是本地 query 的 normalized 行**，不能因该消费就断定必须 gather 全局 hidden。但本地 hidden 值及其生命周期仍必须保留；Q_a/KV 两个输出并不足以恢复 index weights。weights 缩放精度、QLI query/quant/metadata、topk 与候选副输出及事件仍必须保持。
2. 配置层 24、28、32、36 是 index source，但不是 KV source；`_preprocess` 不执行 source compressor，代码不因这些 indexer 的 weights 消费本身要求全局原始 hidden 数值。它们仍然需要全局 cache 的 SWA KV、正确行与 slots，以及各未封闭的 tensor/graph/custom op 消费者。上面的 EP3 L36 gather 落在此静态类别；不能据此证明这次请求可删 gather 或判断收益。
3. 配置层 2、8、14、20 既是 index source 也是 KV source；前者同样使用 local query，但 `is_kv_source` 会将 global cache hidden 交给 compressor。ratio 2 的 2、8、14 需从 global hidden 的 FP32 值生成两路 `wkv/wgate`，经跨 token pool、历史 ring 和 norm 后发布 index key 与 long KV；20 的 ratio 1 则通过 BF16 compressor `wkv→norm`，再发布相应两路 cache。`indexer.update_keys` 消费 compressor latent、positions/slots/cos/sin 并生成 key 和 scale cache；它不是单靠 Q_a/KV 可重建的输出。可以研究按 local 行提前计算 compressor 所需投影并通信等价数据，但要先证明独立点积、权重一致、FP32/BF16 舍入、全局 token 顺序、ring 先读后写和跨层 cache 语义，不能只收集 SWA KV。
4. 空 query 时 `_preprocess` 早转 `_update_caches`：即使没有 index query，也要按 global cache 行更新 SWA，KV source 还要更新 compressor/long/index key；不能跳过必要 collective。PCP 的 `_gather_and_restore_hidden_states`、普通非 CP 分支、packed 分支、外部 hook/图捕获需单独分析，不可把这份 legacy CP 的 local 结论推广为全局。

### 层 36 的固定 host→flow→设备投影记录

对上节同一 scope 的固定 connection，只读 `CONNECTION_IDS`、`TASK`、`COMPUTE_TASK_INFO` 和三个 host callchain，结果在 `f32-ep3-layer36-device-flows.json`、`f32-ep3-projection-stacks.json`；对应脚本和哈希记录于两个 JSON。匹配如下：

| host connection → flow → globalTaskId | host 栈关键帧 | 设备任务已导出输入 → 输出 |
| --- | --- | --- |
| 431372 → 102408 → 3162 | `dsa_v41.py(467):multistream_preprocess` | BF16 `[1023,5120] × [1280,5120] → [1023,1280]` |
| 431373 → 102416 → 3163 | `dsa_v41.py(475):multistream_preprocess` | BF16 `[8180,5120] × [512,5120] → [8180,512]` |
| 431383 → 102482 → 3173 | `indexer.py(174):project_weights` → `dsa_v41.py(405)` | BF16 `[1023,5120] × [32,5120] → [1023,32]` |

三个 host 栈都有 `dsa_v41_cp.py(192):_prepare_inputs_and_caches` 和 unquantized_gemm 相关帧；按冻结 Python 源码，467 行是 `wq_a.matmul`，475 行是 `wkv.matmul`，indexer 174 行是 `weights_proj`。同一 host scope 和实际 flow/形状与 layer36 的 legacy CP、多流及 local Q/global KV/local index weights 路径相容，且该证据比只按 shape 归属更强。运行时实际加载源码 SHA、张量 storage 地址、callchain 导出与源码版本的直接绑定仍缺：不能将这三条严格升级为已证明由同一个 gather 张量输入、或认定 CP/SP 的 token 排列已被精确绑定。`8180` 和 `1023` 也不能仅靠形状推导每个位置的有效性及同步；既有对应 scheduler 摘要另记总量 8180，但仍缺每行身份。

前述 gather 的另一个 flow `102386` 与多个设备 TASK 行关联，`102373` 与通信 OP 行关联；同一个 host collective 的多类导出不能重复计作多个独立 AllGather 或证明哪段耗时可节省。设备 task 所有时间取自同一带 profiler 的诊断，不能直接将它们差值归因为传输、队列等待或可优化成本。

### EP3 样本组与张量元数据的边界核查

对同一数据库精确查询 host connections 431371/431372/431373/431383：`PYTORCH_API` 这四条的 `inputShapes`、`inputDtypes` 均为 NULL；该表结构没有输出 tensor 地址或 storage ID。已有 device `COMPUTE_TASK_INFO` 的形状和 dtype 可绑定 flow，不能仅靠与 host 调用相邻或等形来填补张量身份。

通信 OP 246 的 `groupName` 解码字符串与该数据库 `parallel_group_info` 的一个 communicator key **完全相等**；匹配项 group 名分类 `tp`、`group_rank=3`、`global_ranks=[0,1,2,3,4,5,6,7]`。全库 `META_DATA` value 中该字符串只出现于唯一一条 `parallel_group_info`；其 JSON 没有重复 key，故该 key 在该条记录中唯一。实际字符串仅以 SHA-256 `78b8cb99bf220764b2db33e53042912fddc0744a98a943481567352dd319b63a` 记录，不公开通信端点。匹配是同库强关联线索；仍需相同版本 exporter 的字段契约、运行对象绑定及每个 token 的实际排列才能称完整通信组/行域已证。

OP 的 `dataType=11` 在同库 `ENUM_HCCL_DATA_TYPE` 只有一条映射，显示 `BFP16`。`count=5237760` 数值等于 `1023×5120`，与同 scope local Q_a/index weights 的输入 shape 和 dtype 相容。[CANN 9.0 profiler 文档](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/devaids/Profiling/atlasprofiling_16_1144.html) 将 `count` 描述为传输的 `dataType` 类型数据量；这仍是比历史 9.2 早的版本，不能仅凭数值与之相容就证明 gather shape、buffer 内容或精确 exporter 单位。**一次 count 不是物理网络字节数**，也不是端到端收益。

有界查询脚本 `query_f32_ep3_tensor_group.py` SHA-256 `37780774ac72a71a01765d907db6ab3588dd486580920d279ac8035ae22f66be`，结果 `f32-ep3-tensor-group.json` SHA-256 `a2838df19b4353a8899851cc8cd82db0d2128c1e85a912e506928ead8d0d57db`。新证据不能证明 gather 输出与投影输入是同一 storage，也不能独立证明逐 rank 的 token 位置一致；没有新的设备采集。

### 固定 host 范围与原始 TorchOp 字段检查

EP3 同层的 `c10d::_allgather_base_` host scope 包含候选 `HcclAllGather`，其记录的两个 BF16 tensor 参数 shape 依次为 `[8184,5120]`、`[1023,5120]`。已安装版本的 `torch/include/torch/csrc/distributed/c10d/ProcessGroup.hpp:363–378` 在调用 dispatcher 的 `c10d::_allgather_base_` 时传入 `outputBuffer, inputBuffer`；`torch/distributed/_local_tensor/_c10d.py:503–512` 亦注释同一 schema 的 output/input 顺序。若当时实际接口与该已安装源码一致，则这两个形状分别描述输出和输入缓冲；不由此确认其和后续投影 storage 的关系。

同线程 gather host call 结束至 KV matmul host call 之间，以固定算子名称有界查询 host `PYTORCH_API`，可见多条 BF16 `[8184,5120]` 的 `aten::slice` 输入记录。但该表对这些记录没有输出 shape、slice 的数值参数或 tensor storage 标识，故**不能据此认定**其中某条截成 `[8180,5120]` 或其输出是 KV 投影输入。`f32-ep3-slice-ops.json` SHA-256 `c7d9fdf4619a1d79ba4586b29170b0647ee029e566de911f2254a1b042d50d7c`，对应查询 `query_f32_ep3_slice_ops.py` SHA-256 `537351d5a2a27257de55717d115bd0bf10b0e76fda9f5066ea79cfdec7af9831`。

安装的 torch_npu `_torch_op_bean.py` 为原始 op-range 定义 `INPUT_TENSORS` 类型 5；`_event_tree_parser.py` 对该字段的张量元数据格式有 TensorImpl、storage pointer、dtype、shape、stride 等描述。但这只是解析器可识别的字段结构，并不证明本次采集都写入该字段。

对本 rank 已保存 `FRAMEWORK/torch.op_range` 做一次有界低优先级流式元数据检查：按记录头跳过不在固定线程/时间窗的 payload，只对接近三个 host 目标的条目读取 op name 与类型 5，不读取 tensor 内容或 scalar 字段，限制每条记录大小并核对文件前后属性。该方法未将大型 trace 整体加载到内存。`c10d::_allgather_base_`、两条 `aten::linear` 各有一条同名候选，其开始、结束经近似时钟换算与相应 DB 记录均相差 4.720 μs，均是 10 μs 窗口内唯一候选；时钟转换与采集源版本仍未完全绑定。这**三条候选**都没有类型 5 字段，故没有可比较的 tensor/storage/stride；此结果**不证明整个采集或所有其他 op 都没有此字段**。脚本 `query_f32_ep3_oprange_tensor.py` SHA-256 `e8209cc5a2ca3679c5d56ff8ddb0952b9f0b4afa1b4d03288e30f50ec07d63aa`，结果 `f32-ep3-oprange-tensor.json` SHA-256 `9674ada48a35202bd610ff660fdd72198a97862df1fb9d4413edcc608181d72a`。即使未来见到相同 storage pointer，还需区分 offset、分配代际、生命周期、原位写入和 copy；不凭地址或时间邻近独立认定数据流。

在当前只读既有数据、不得新跑 NPU 的范围内，这个固定样本的 gather 输出→投影输入 storage 身份未获证明；停止沿此固定字段路径重复查询，保留后续有同版本、已绑定对象事件或用户允许的针对性采集时再验证的条件。

### EP3 层 36：仅收集 KV 投影的条件性张量体积比较

此前的 **35%** 是对每行同时收集 Q_a 和 KV 两种 BF16 投影输出的设计比较：`(1280+512)/5120=35%`。针对 legacy CP 行域对齐后 Q_a 可在本 rank 保留、而全局只收集 KV 投影的另一种结构，分子只含 512 维 KV，条件性逻辑体积比为 `512/5120=10%`；两种数字对应不同结构，不能相加，也都不是实际通信节省。

以 EP3 层 36 所选 host scope 的 BF16 `[1023,5120]` 输入形状和 8 成员 TP 组记录为算术示例，**额外假设每个 rank 均有同样 1023 行且行序及分组满足同一 gather 合同**，候选先将每 rank 的该输入投影为 BF16 `[1023,512]`，再收集该投影，得到：

| 对比项 | 原 hidden gather 的逻辑 tensor | 候选 KV gather 的逻辑 tensor |
| --- | --- | --- |
| 单 rank 输入 | 1023×5120×2 = 10,475,520 B | 1023×512×2 = 1,047,552 B |
| 8 rank 拼接输出 | 8184×5120×2 = 83,804,160 B | 8184×512×2 = 8,380,416 B |

以上输入/输出大小是逻辑 tensor 的字节数，**不是链路传输字节、物理访存量、实际消除量、时延界或性能收益**。表中的 8184 行是按 8×1023 计算的候选总行数；现有同 scope KV 矩阵任务输出 8180 行，但仅凭形状差不能证明多出的 4 行是 padding、位于尾部或与这些 tensor 相关。候选必须证明实际 padding 与裁剪、各 rank 行排列、有效 token 与同一请求/批次一致。每 rank 假设、同一通信组和真实值映射都不能从 EP3 单卡记录推定。

该结构还须证明每个 rank 的权重一致，局部投影与原全局批次投影在允许的精度、量化和舍入契约下等价；本地 query、indexer、输出分配及其他 hidden 消费者都得到所需数据或等价 metadata，且 SWA cache 更新、slot/position、collective 次序、stream/event 与生命周期不变。第 36 层在保存配置中是 index source 而非 KV source，不能将此候选推广到需要 compressor 的 KV source 或 PCP 分支。原 gather 输出与投影输入的 storage 绑定仍缺，实际源码对象和完整数据依赖也未闭合；不由此宣称可实施、可省掉现有 gather 或获得任何提速。

### 同摘要 EP0–7 的形状、flow 与等长约束复核

只读 `target-layer-all-kernels-16rank.jsonl`，核对 SHA-256 `0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4`。限定 layer 36、scheduler digest `db852bff987d2b98d3dc0734e3731ec5528e37dcf6e874860d1ebb4fd12eefc8` 和 EP0–7：每个 EP 恰有一个所选 scope，scope 在 trace 中唯一、sidecar 线程相符；每个 scope 恰有一条 BF16 `[*,5120]×[1280,5120]` 和一条 BF16 `[*,5120]×[512,5120]` MatMul。既存提取器仅保留所属 scope 内唯一 host→device flow 起止、唯一设备事件、唯一 CSV 行及匹配时长的记录。两条候选在各 rank 均有该关联。EP3 再按设备开始时间和 task ID 交叉核对既存数据库设备记录，各唯一命中：前者 task 41415 / connection 102408，后者 task 13531 / connection 102416。trace flow ID 与 DB connection ID 不属于同一编号域。

前一类物理输入行数按 EP0–7 为 `1023,1023,1023,1023,1023,1023,1023,1019`；后一类均为 `8180`。`7×1023+1019=8180` 只是算术相等，既不证明是同一个请求、通信组、tensor 或 token 顺序，也不证明有效 token 数，尤其 EP7 的 1019 可能是裁剪后的物理行数。除 EP3 已有具体 host 栈与设备记录外，其他七个 EP 的两类任务只按形状筛选，不能据形状独立断定实际调用名和 tensor 身份。EP3 的 OP246 到 TP `[0,1,2,3,4,5,6,7]` 是前述同库 metadata 字符串关联，仍受 exporter 版本和运行对象绑定限制；其余七个 scope 的对应 gather、实际 communicator 身份及 collective 配对尚未证明，不可将 EP 标签直接当组内 rank。

华为 [CANN Community 9.1 的 HcclAllGather 接口文档](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/commlib/hcclug/docs/en/api_ref/comm_op_interface/HcclAllGather.md) 明确规定组内各 rank 的 `sendCount` 和 dtype 相同，并按通信域 rank 顺序拼接。这个 9.1 文档不是本次历史 9.2 采集的同版本合同，尚不能凭它证明当时具体后端调用及语义。构造性反例：**假设**把上述 8 个局部输入分别投影为 512 列 BF16，保持其物理行数，再直接作为同一次该接口 gather 的发送缓冲，则七个 `sendCount=1023×512=523776`，另一个为 `1019×512=521728`，不满足文档的等长条件。若要研究该方案，需先证明这些输入属于同一通信组和目标行域，再设计并证明等长打包、顺序和裁剪；或另行核实相应版本与设备确实支持的变长协议。此反例不说明原历史 gather 有错误，也不证明本地投影的行数必须使用这些 Q 形状、padding 值或重组能带来性能收益。

复核脚本 `recheck_f32_ep3_projection_rows.py` SHA-256 `47db6a4711d1a0311b1a50bbf4aaa0866f22c890b79bf93190fef48cc385da84`，结果 `f32-ep0-7-projection-rows.json` SHA-256 `87fa9a16d2acad858fe1da3f216915c1497bd1964a28671d39883386d400e831`。本机 ZCode 简单核对运行 90 秒超时、无结果，以上由主控脚本复核；未启动 NPU、未修改推理实现。

### 同卡已选投影的跨流导出时间

复用上节同一摘要、L36 的 EP0–7 两类唯一匹配设备记录，分别在每张卡内部用十进制计算 `KV 形状任务开始 − Q 形状任务结束`。每卡两项的导出 stream 不同：Q 为 EP0 的 45、其余的 47；KV 为 EP0 的 33、其余的 35。八卡该差值分别为 **1.0405、1.2595、1.00025、1.220、0.98025、0.9805、1.0195、1.0205 微秒**，最小/中位/最大为 **0.98025 / 1.0200 / 1.2595 微秒**。按这些导出时间，八对均为 Q 记录结束后 KV 记录开始，而不是两个选定区间相互重叠。

EP3 的 Q / KV 记录还可分别按完整起点和时长匹配前述 DB flow 102408 / 102416；其同库 AllGather OP246 的导出结束比 Q 记录开始早 **16.420 微秒**。这仅是所选设备记录与通信记录的时间相对位置，不把多个 flow 合并为多个独立 collective，也不将 host 时间与 device 时间相减。

这些观察依赖**同设备不同轨导出时间可比较**；该采样的跨流和通信轨时钟偏差未单独校准，微秒级差值尤其不能当高精度物理同步测量。其他任务或流可能并行，现有记录不揭示任务内部流水、tensor/storage 身份或 event 依赖；不能据此认定两项必然串行、实际空闲或可实现收益。EP3 有先前 host 栈绑定，其余七卡仅是形状候选；跨卡只是分别求自身差值，并非证明同一次 collective、请求或相等工作量。带 profiler 的记录也不能直接替代无 profiler 基线。

在运行根目录执行 `python3 agents/fusion/recheck_f32_projection_trace_order.py` 可只读复算；脚本 SHA-256 `433b64576b0f43d4b892069aa52179fd53ea589dee54fe023fb434a32b29080f`。脚本核验上节行域 JSON 的 SHA 和 `f32-ep3-layer36-device-flows.json`、`f32-ep3-layer36-scope.json` 的 SHA-256 分别为 `d8bae5c11c49f16d2c305fc6b23a6667766a1356131b7dcafd01b8a488774021`、`210a74e86892d4a11c0b127eb16149fbb1c2256c90e7272c1a15d63c88be2b84`；核对 EP3 精确 task 时间、flow 及 OP246，不读取设备或修改服务。

### F32 不同通信对象的信息界：限定接口与输入族

F32 的选择是将完整 normalized hidden 按 token 收集，还是在源 rank 局部先投影、让目标 rank 得到所需 KV 投影。仅有逻辑 tensor 的 10% 大小比不证明跨卡必要通信量之比：真实权重可能降低投影可区分结果数；已有缓存、副本、局部重算、其他隐藏消费者、输出误差合同和所有旁路都可能改变必要量。

对于一个指定源 rank、一个指定接收 rank 及本窗口，固定接收侧初态和所有与输入无关的预处理，定义有限可能输入集 `X`；源侧有 r 个需要被接收侧处理的完整 token 行，权重 W 固定且双方可预知。若契约要求接收侧在窗口内**逐位精确观察这些行的 KV 投影** `f_W(x)`，且接收侧获取 x 相关信息的全部路径（含其他 rank、旁路、共享状态、控制、结束标记、可观察长度与时序）均已计入同一个有限离散 transcript，那么 `K = |{f_W(x): x∈X}|` 个不同的投影结果必须能被区分。对所有输入的 transcript 若都短于 `floor(log2 K)` bit，可用有限二进制串总数不足 K；故至少有一个输入的跨 cut 可区分信息不少于 `floor(log2 K)` bit。允许投影前移、改变分块、源侧或接收侧重算、计算共享与无损压缩；这个界不要求使用 AllGather，也不是网络包字节或每个输入都必须传输的数量。这里假设确定性或对有限 X 可固定一个对所有输入都精确正确的随机种子；交互回复必须由已计入的信息及固定初态决定，否则须一并计入全部信道。已被接收侧掌握的输入相关副本或摘要必须纳入初态并重新计算 K，不能只拿 tensor shape 计算。

可复算的**形状相容、权重和输入均虚构**例子：以 r=1023，W 使 `f_W(x)=x[:512]`，每行 5120 个 BF16 分量分别独立取 256 个有限正数编码 `0x3f00…0x3fff`，并假设这些值在本接口可达。若契约只要求接收侧得到 512 维精确投影，则 `K=256^(1023×512)`，最坏输入的可区分信息下界为 `1023×512×8=4,190,208 bit`（523,776 B）。若另行要求接收侧完整逐位观察 5120 维输入，则联合像变为 `256^(1023×5120)`，对应 `41,902,080 bit`（5,237,760 B）；这两个数分别对应**不同接口契约**，不应相加。它们还与 BF16 存储大小 `1,047,552 B` / `10,475,520 B` 不同：输入族每元素只有 256 个可能，而不是全部 16 位编码均独立可达。即使模型真实权重不是零、可达集也可能远小于此假设；若 W 为全零且不含使输出随 x 变化的 bias，投影输出恒定，K=1，此投影接口本身给不出正的通信信息下界。

纯 CPU 复算：`python3 -c 'r=1023; print(r*512*8, r*5120*8, r*512*8//8, r*5120*8//8)'`。输出 `4190208 41902080 523776 5237760`。这些数没有使用或推断实际值、带宽、初态信用、执行时长或收益。如果另有该 cut 覆盖所有通信路径的已证包络 `I(t)≤β+Bt`（`B>0`，bit/s），则对人工输入族的最坏情况才可写 `T≥(floor(log2 K)−β)_+/B`；与可重叠的计算/其他通信下界取最大，只有已证明的依赖链才能相加。对于多个接收 rank，各自 cut 的信息可以被一次广播或共享链路携带，不能将逐接收端界直接相加。

已保存的 EP3 L36 关联提供 `[1023,5120]` 与 `[8180,5120]→[8180,512]` 等形状及同 scope 的 TP 组线索，但尚未证明相互间 tensor/storage、每行有效性、精确 KV 输出的必要可观察性、真实 W 的可达像、可用副本和网络包络。故 r、K、β、B 对该次真实跨卡 cut 均为 UNKNOWN，不能把上面示例代为填值，不能把该条件最坏输入与当次 profiler 的时间对比。此 cut 位于 **F32 的 SP 输入/KV 投影**，不同于 F16 的 attention 输出 O 投影；如将本界与其他对同一 KV 或同一远端输入的 cut 界合并，须先构造联合可区分输出像和共享信息边界，避免重复计数。若最终仅需后续 attention 的值而不要求物化 KV 投影，也必须重新定义可观察输出和像集合。

### PCP gather 与恢复顺序：真实行不一定是 rank-major 缓冲的前缀

在需要全局 metadata 的 PCP 条件分支，冻结 `dsa_v41_cp.py:398–417,438–441,465–482` 先调用 `AscendDSAPCPMetadataBuilder._prepare_graph_pcp_context`，随后 PCP all-gather 并以 `hidden_restore_idx` 在 token 维执行 `index_select`，得到 scheduler 顺序的 global cache hidden。`dsa_cp.py:2265–2285` 对 restore 的前 `global_batch.num_tokens` 项拷贝原索引，并把后续 padding 项的**索引**填 0；`dsa_cp.py:2365–2393` 将 global 实际 token 数设为 `global_batch.num_tokens`、graph 后长度设为 `num_tokens_after_padding`。`dsa_v41.py:350–355` 的 `_preprocess` 再按 global `num_actual_tokens` 截取 cache 行。由此应区分 **rank-major 发送缓冲、经过索引恢复的 scheduler 顺序和按实际 token 截取后的行域**；padding 的 restore 索引为 0，并不意味着恢复出的 padding tensor 数值为 0。

下例是符合以上接口约束的**构造性行索引例**，不是已保存历史的 PCP 请求，也未证明所选索引可由实际 PCP manager 生成：假设两个 rank 各发送三行，rank-major gather 为 `[10,20,P,30,40,P]`，其中 `P` 表示各 rank 的 padding；真实 scheduler 顺序是 `[10,30,20,40]`，实际 token 数 4、恢复长度 6，restore 为 `[0,3,1,4,0,0]`。对这个例子先恢复得到 `[10,30,20,40,10,10]`，然后截断为真实四行；若错误地先截 gather 的前四行，再使用原 restore，就因索引 4 越界。若直接用前四行代替恢复结果，则得到 `[10,20,P,30]`，行序和有效行都错。把行独立投影前移并在投影后正确 all-gather、restore、截断仍可研究：前提是上述行语义、权重及数值合同、各 rank collective 次序和所有其他消费者都成立。不能因例子而断言某个通信或中间物化必须发生；也不能将已经在截断后完全不可见的 padding 值差异单独认定为最终输出差异，需考虑别名/graph/其他消费者。

纯 Python 复算（这里用恒等标量投影仅演示行映射，不涉及 NPU 或真实权重）：

```bash
python3 - <<'PY'
gathered = [10, 20, 'P', 30, 40, 'P']
restore = [0, 3, 1, 4, 0, 0]
actual = 4
restored = [gathered[i] for i in restore]
print('restored:', restored, 'valid:', restored[:actual], 'wrong_prefix:', gathered[:actual])
try:
    print([gathered[:actual][i] for i in restore[:actual]])
except IndexError:
    print('prefix-before-restore: IndexError')
PY
```

此条不扩大到前文已绑定的 EP3 layer 36：其已有栈与 legacy CP 相容，而 PCP runtime metadata、真实 `hidden_restore_idx`、请求和张量身份均未绑定。源码 SHA-256：`attention/context_parallel/dsa_cp.py` `7e8dd93eeda988866138342523ecfa7cc9c42ce67669d4b1507ee8c5f3452305`；`attention/context_parallel/dsa_v41_cp.py` `039ec9d18e0a8a4aa39f9d5b4bea15204cea3f397a3fce97a5097b266bc9822e`；`attention/dsa_v41.py` `3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd`，均位于本 run 的 `sources/vllm-workspace/vllm-ascend/vllm_ascend/`。


## Q_b 按 head 分片与 attention 分布重排的条件候选（2026-10-30）

CP 构造 ReplicatedLinear(1280,64×512)：model.py:377-415；非 CP 是 ColumnParallelLinear。dsa_v41_cp.py:118-155,189-207 当前按 local-token/all-head metadata 执行，dsa_cp.py:77-91 的 restore_tp_heads all-to-all 转为 all-token/local-head。条件 TP8、每rank有效M3 时，Q_b [3,1280]×[1280,32768] 与 [24,1280]×[1280,4096] 均为125,829,120 MAC/rank，INT8权重本体40→5MiB，输出元素同为98,304。候选须前移分片并同步改query metadata、Indexer共享indices、因果位置、SWA/压缩cache可见性、量化scale/NZ与输出head/offset/padding；新增聚合和cache/通信未计，不能独立删restore或承诺加速。与F15保留输出交换的方案互为替代，和F32输入gather有交互，不可相加收益。无目标实现修改或该重排的设备测试。
