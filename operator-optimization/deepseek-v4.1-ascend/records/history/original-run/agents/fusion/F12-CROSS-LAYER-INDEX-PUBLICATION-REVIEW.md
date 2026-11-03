# F12：QLI 输出发布与跨层注意力的真实边界

## 为什么先看这里

`captured-layer-attribution/operator_metrics.csv` SHA256 `2a8d39f101c7e8949931390d49a3b2dcce58aee13bfe0bf799f789b81f8f55f4` 的 `INTERIOR_RUNS_1_2_3`，在配置中的 8 个 index source 层 2/8/14/20/24/28/32/36 各归属 48 条 `VllmAscendQuantLightningIndexerV2`，合计 384 条、任务时长合计 81935.706 µs、单条最大 504.61 µs。该表是 16 rank × 3 个窗口的角色/时间归属；跨 rank 时间求和不是墙钟时间或关键路径，也不能证明每条调用的输入 tensor 身份与具体候选模式。同期每个 40 层均有 SparseFlashMla 记录。同一 `evidence/TRACE-SUMMARY.json` 的 EP0 历史整段中另有 80 条 `_prepare_indexer_indices_kernel`，不能与上述 384 条跨 rank/窗口记录直接一一绑定或将其时长当成此窗口收益。这里研究被采样支持的高频边界，不据此声称已量化改进。

## 分支、生产者及已知消费者

- 冻结镜像 A3 build 信息按正常导入、对象未替换时映射 STANDARD/non-packed，详见 `INDEXER-RUNTIME-REVIEW.md`；历史单次实例和具体数据条件仍未直接绑定。`indexer.py:233–297` 的非 packed 正常路径：输入包括量化 query/scale、INT8 key/scale、FP16 weights、paged block table、sequence/QLI metadata，先等待 `DeviceMetadataStage.INDEXER`，再调用 QLI。mode 1 为 candidate source、mode 2 为使用 candidate filter、mode 3 为普通选择。QLI 输出 `selected` 和条件性的 candidate block 列表；`prepare_indexer_indices` 将 token position 按可见范围过滤并排序、写到 `indices_output`，在长度输出条件满足时同一 Triton kernel 写长度。这些过滤、排序和部分长度计算已经融合，不能重复计入。
- `model.py:490–502, 941–952, 990–1016`：每个 model 所有 decoder 层引用同一个 `DeepseekV41SharedAttentionState`；其中 INT32 `topk_indices`、candidate `[max_tokens,1,2048]`、两个 `[max_tokens,1]` 长度缓冲是容量级持久对象，`reset()` 故意不清空，依赖 source 覆盖活跃行及固定地址图重放。允许清理或生产复用时必须保持不同 forward/graph 重放及无效行的行为。
- 配置 `index_topk=512`、candidate source=20、candidate blocks=2048、block size=8（`evidence/model-config.json:114–117`）。`model.py:527–550` 的 index source 刷新索引；非 source 复用前一个 source 的索引。层 20 生成 candidate block IDs，24/28/32/36 的 source 依条件使用这些 block IDs 做过滤；非 source 层沿用各自 source 结果。这些是配置/源码角色，并非证明每次调用都走非空过滤。
- `dsa_v41.py:567–611, 640–680, 690–715`：source 在 QLI 后写稳定 topk、candidate source 将返回 candidate 拷入共享 candidate buffer；本层与后续非 source 层的 SMLA 从相应共享 topk 得到 `cmp_sparse_indices`。QLI 的 candidate block ID 与给 SMLA 的 token 位置不是相同语义，不可把二者当同一中间量。长 KV cache / metadata 也单独被注意力读取；不能用索引融合删除这些状态。CP 中 `context_parallel/dsa_v41_cp.py:187–207` 的 source 用本 rank 局部 token 切片，非 source 用 `qr.shape[0]` 界定该 rank 查询行，不能用 full-batch 行索引直接代替。
- `spec_decode/llm_base_proposer.py:468,709–724` 条件下把 draft model 的 `topk_indices_buffer` 属性指向 target buffer；这是已知可能发生的属性级别共享，**并不单独证明** draft 的 `shared_attention_state.topk_indices` 被同步重绑定，或某个采样有此共享。改稳定地址前必须检查实际绑定、消费顺序与 target/draft 生命周期，不能推测它们互不别名。

## 具体可重组边界与反例

1. **QLI→filter/sort→直接发布到稳定索引 buffer。** helper 已支持 `indices_output` 且 `indexer.py` 已传入共享 buffer；因此把 helper 输出重新“优化为直接写 buffer”本身不是新增机会。潜在更大候选是 QLI 内部最后阶段完成原 helper 的可见性判断和按位置排序、直接写固定地址，同时保留 mode 1 的 candidate block 输出，或按行交付给 attention。但当前 Python 的 QLI 调用没有接收 `indices_output` 参数，算子内部实现、可扩展 out 契约和是否按行产生全局 topk 尚未知；这需要新接口或改变算子，不能推断现成能力。每行排序必须处理完整 QLI topk 集合并按 `position` 和 `compress_ratio` 保持 `-1` 及原序规则；在该行 topk 决定前无法让该行的 SMLA 使用完整结果。
2. **分块生产消费。** 在 QLI 对某行完成完整排序且该行的 query、正确 cache/page、metadata 均就绪后，数学上可以研究行块交给 SMLA。当前 QLI / SMLA 是整张量接口，跨 kernel 的逐行 ready、chunk 输出生命周期和 event 尚无证明；同一共享 buffer 后续 source 会覆盖行，而前一个 source 对应后续层仍要用其结果，因而不能提前覆盖仍在使用的区间。跨层 hidden/query 还依赖上游完整模型计算，不存在把所有后层选择都在层 20 一次算出的依据。
3. **长度计算是条件精简候选，不是已证收益。** 常规非 packed 分支仍把 `topk_lengths` 传入 helper，然而 `_forward_attention` 仅 packed `backend.qsmla` 分支显式读取它；普通 SMLA 调用并不传 `topk_lengths`。同一源码中 `candidate_lengths` 在非 packed 正常 QLI 调用亦未传入 native operator，但空 cache 分支会对其清零；packed 逻辑会用这些长度。局部显式消费者搜索无法排除框架外部属性读取、图捕获副作用、别名及运行时替换，不能认定可删。即便所有这些证实不需要，helper 在 `num_rows*max(next_power_of_2(topk),32)<=524288` 时已融合写长度，减少的也只是部分归约/写入，不自动等于少一次 launch；超过阈值的 fallback 另有 sum，实际形状及是否命中需单独证明。
4. `pad_sparse_indices` 仅在宽度不足 topk 时 padding，之后 `unsqueeze(1).contiguous().int()`。若输入确为 contiguous INT32 `[T,512]` 且 topk=512，前者无需填充，后三者通常可作为 view/identity，不可凭源码将其记为一次实际 NPU 复制或可节省的 HBM 流量；空/不同宽度、非连续输入、执行框架 lower 后的行为另行判定。

此外，source QLI 选择依赖 query quantize 完成事件，`dsa_v41.py:580–583` 明确等待该 event；每次 QLI 前也等待 metadata 对象，SMLA 前等待 attention metadata。压缩 key、candidate 和稳定输出的正确 stream/producer、真实 event 身份及向后共享读完成时间尚未由现有 trace 绑定。不能因为设备记录在时间上靠近就删除同步或声称能并行/提速。数值上需保持 QLI 的量化和 scale、权重 FP16、有效位置过滤、topk tie/排序与最终 INT32 值；算子内部 tie 及跨算子舍入的准确契约仍未知。

## 可复核来源

`indexer.py` SHA256 `d5d6a438bbced4e63d729dac94c08869df30dab4a71ceacecdea211386b3db33`；`dsa_v41.py` `3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd`；`dsa_v41_cp.py` `039ec9d18e0a8a4aa39f9d5b4bea15204cea3f397a3fce97a5097b266bc9822e`；`model.py` `101f96b38e9fc47e7be147ddfe56976aee896867c73a33a7d95c3e3c833a6e91`；`llm_base_proposer.py` `458fb50e6839f3a353ddc7181ace4fdae02e35093d8ccd1c4327dc24ae5fff31`。路径在 `sources/vllm-workspace/vllm-ascend/vllm_ascend/`。helper 已冻存在 `agents/fusion/frozen-source/vllm_ascend/ops/triton/prepare_indexer_indices.py`，来源校验见 `INDEXER-HELPER-REVIEW.md`。以上只读已有资料，没有设备运行和推理代码变更。

## EP3 层 36：QLI、索引整理与注意力的主机栈到设备记录关联

对先前 `f32-ep3-layer36-scope.json` 已归属的 EP3、layer 36、decoder host scope `2128-42`，只读同一 profiler 数据库（打开路径见 `query_f32_ep3_layer36_host.py`）。限定同一 globalTid、scope 内的所选 operator host API，查 callchain，再用 `CONNECTION_IDS` 连接相应设备 `TASK`。结果比此前仅由形状或邻近时间推测模块归属更强：

| 调用记录与命名设备 task | host 调用栈的关键帧 | 物理输入、输出摘要 | 设备持续 |
|---|---|---|---:|
| QLI：host 431389 → flow 102518 → global task 3179 | `indexer.py:275 select_projected` → `dsa_v41.py:585 _select_sparse_indices` | INT8 query `[1023,32,128]`、INT8 paged key `[17315,128,1,128]`、若干 FLOAT16 scale/weights、INT32 metadata 等；首输出 INT32 `[1023,1,512]` | 5074.260 µs |
| 索引整理：host 431390 → flow 102521 → global task 3180 | `prepare_indexer_indices.py:123 prepare_indexer_indices` → `indexer.py:291 select_projected` | INT32 `[1023,512]`、INT64 `[1023]`、INT32 `[1023,1]` → INT32 `[1023,512]` | 217.160 µs |
| SparseFlashMla：host 431391 → flow 102528 → global task 3181 | 外层 `_C_ascend::npu_sparse_flash_mla` 栈含 `dsa_v41.py:657 _forward_attention` | BF16 query `[1023,64,512]`、两项 BF16 paged KV `[17315,128,1,512]`、INT32 索引 `[1023,1,512]` 及其他 metadata → BF16 `[1023,64,512]`、FLOAT 输出 | 898.260 µs |

三条命名 task 均处 stream 47，按导出时间前后排列，前两段间隔 0.860 µs、后两段间隔 2.320 µs。QLI 和 SMLA 的 flow 还各关联同流同 taskId 的无名 TASK 行，不应将其再算成独立的命名算子耗时。设备时间来自重 profiler 采集，不是在线基线或可兑现收益；小间隔不证明 stream 空闲，host 的 scope 时间也不能直接与 device 时间相减。INT8/FP16 缓存、query 与 metadata 的具体逐参数对应，仍须结合真实接口和导出字段，表格不将同类型输入擅自映射到某个物理对象。

根据保存的 `indexer.py:256–295`、`dsa_v41.py:567–607,613–680`，非 packed 条件路径先使 QLI selected squeeze 后进入 `prepare_indexer_indices`，写稳定 `indices_output`，再供本层 SMLA；该稳定 buffer 还可能被后续非 source 层复用，candidate block ID 和 SMLA token ID 含义不同。`prepare_indexer_indices` 已在指定条件中完成过滤、排序和长度写入，故**单纯提出再把这些相同步骤合到一个独立 helper 不是新增融合**；更大重组将 QLI 最终选出、过滤排序与发布合一仍是候选，须保持 candidate 输出、无效/排序规则、输出地址及所有后继读取。根据保存配置，36 是 index source 且候选过滤角色来自先前 source；历史运行实际源码 hash、candidate_mode 值、输入内容和缓存对象未绑定，不能据此声称采样确定选择 mode2。

此处 host 栈及 flow 只把三次调用分别关联到各自设备 task，**不直接证明**三者首输出/输入是同一 tensor、共享 buffer 或依赖已由事件正确建立；shape 相容和同流顺序不是 storage 证明。`1023` 也是物理行数，不是已确认有效 query 数。最低补证是同次记录中的 tensor/storage、view/offset、共享 buffer 版本和具体运行源码/参数身份，以及与跨层消费者相关的事件/生命周期；若缺失则该候选保持条件性。

复核入口：同 `query_f32_ep3_layer36_host.py` 的只读数据库、globalTid `9139690408016`、scope `1793241329100209040–1793241329168402770 ns`，host callchain 分别 `1020978`、`1020982` 和外层 `1021010`，按上表 connection 从 `CONNECTION_IDS`、`TASK`、`COMPUTE_TASK_INFO` 和 `STRING_IDS` 查询。此结果仅限该 scope 和三条指定调用，无 NPU 工作、无权重或请求内容读取。

## 同一 EP3 scope 的 mode、有效长度与 helper 阈值：能确认到哪里

基于上一节的同一 EP3/L36 scope、QLI host 外层 callchain `1020979` 和设备 global task `3179`，定向只读 `PYTORCH_API`、`STRING_IDS` 并复用既有 `qli-32-binding-fields.json`：

- `_C_ascend::npu_quant_lightning_indexer_v3` 外层记录的 tensor 类型与物理 shape 与设备 QLI 报告相容，包括一项 INT32 `[1023,1,2048]`；其他位置有若干 `Scalar`、`None` 类型，但这条 host 记录不包含 scalar 的值。仅凭该张量形状**不能**判断候选 mode、有效候选数、压缩率、mask 模式、有效序列长度或 tensor 身份。
- 既有 32 组绑定审计精确覆盖这条 task：EP3、scope `2128-42`、stream 47、taskId 41433、globalTaskId 3179，解码的 `attrInfo` 为空。它没有可读的算子属性，并不说明 mode 为默认值或采集之外没有属性。
- 针对此 `_C_ascend` 外层 host 事件的 raw `FRAMEWORK/torch.op_range`，使用与 `query_f32_ep3_oprange_tensor.py` 相同的定点时钟换算，遍历 165722 条头记录，只解码 PID/TID 2128 且两个端点均距 DB 事件不超过 10 µs 的候选的名称、TLV 类型及长度；唯一同名候选的两个端点都偏移 **+4.720 µs**，其字段类型为 `2,3,4,8`，没有类型 5 tensor metadata。已存 `tensorinfo-export-schema.json` 显示当前解析器将 3、4 用于输入类型和形状；当前解析器、近似时钟和历史 raw 之间的完整版本关系仍未闭合。该限定查询**未读取 scalar 数值或 tensor 内容**，也不能证明其他 raw 记录缺少相关数据。
- scope `2128-42` 与匿名 scheduler digest `db852bff…` 的关联已在 `target-layer-all-kernels-16rank.jsonl` 中保存，现与 `F12-SELECTED-SCHEDULER-BINDING.md` 的既有 32 窗口核查对上：该批 scheduler total Q 为 8180，匿名 descriptor 分组为 `(3328,29440)`、`(1652,29568)`、`(3200,29568)`，此处的第二项为保存的 computed token 描述，不是已绑定的设备 effective cache length。该数据不能推出 rank 内每行具体位置、分片有效性、QLI 实际候选和 native 计算量，不能据此证明后续过滤冗余。

唯一可以进一步收紧的是**源码条件下的 helper 分支**。上节已由该次 host→flow→设备匹配看到 prepare 的物理首输入为 INT32 `[1023,512]`。若历史运行确是所保存 `prepare_indexer_indices.py` 的相同版本、该记录的首输入对应源码 `selected` 参数且以该 shape 启动，则 `num_rows × max(next_power_of_2(topk),32) = 1023 × 512 = 523776 ≤ 524288`，仅差 512 到阈值；源码规定此时 **只有** `lengths_output is not None` 才设 `WRITE_LENGTHS=True`，否则不写长度。若其中任一绑定未证明，则不能把判断升级为实际执行路径；特别是额外看到一个 `[1023,1]` 输入不足以证明它就是长度指针及其值。即便条件成立，也只说明长度计算位于同一个 helper kernel，并不能证明过滤、排序或这个 helper 可省略。

复核：同一数据库按 globalTid `9139690408016` 和 callchain `1020979` 获取外层类型/shape；task 的 attrInfo 引用 `qli-32-binding-fields.json` 及其输入 DB 摘要。raw 限定目标为名称 `_C_ascend::npu_quant_lightning_indexer_v3`、起止 `1793241329139097330–1793241329139752130 ns`，读取方式参照现有 `query_f32_ep3_oprange_tensor.py`，没有创建新任务、设备采集或推理实现变更。

## layer 20 与 24/28/32/36：缓存来源、相对块坐标与候选掩码的细化

依据保存的 `evidence/model-config.json:96–117`，KV source 为 2、8、14、20，index source 为 2、8、14、20、24、28、32、36，candidate source 为 20。保存的 `model.py:519–550,739–746` 对有压缩的每层分别按 `_latest_source` 选 KV source 与 index source，但构造 `index_k_source_prefix` **使用 KV source**。因此在这些静态配置和同一个 model/context 的条件下，第 20、24、28、32、36 层选择的 index-K 均通过层 20 的 `indexer.k_cache` 名称获取（`dsa_v41.py:575–578`）；而 24/28/32/36 是各自重新选择 topk 的 index source，使用同一个 shared candidate buffer。这个结论说明逻辑上同一个 cache 名称被取用，**不证明历史 trace 中的 storage 地址、内容、版本、调用时机或运行中的源码与配置身份**；同名引用也不能用来证明跨模型、跨 forward 或不同 request 共用缓存。

另外，保存的 arch22 `quant_lightning_indexer_v2_service_vector_arch22.h:448–520,775–920` 给出比“过滤候选块”更精确的条件语义：consumer 对每行读取 candidate 块号、排序；按 `cuBaseS2Idx / candidateBlockSize` 计算 tile 的**相对块号**，再将命中关系展开为位置 mask。source 的 candidate 块输出不加 `output_idx_offset`，而 sparse token 索引可加行 offset（同文件 899–920 行）；因此两类结果不能直接视作同一坐标，融合/重排须保留行索引、块大小和 offset 约定。

尤其注意：该源码中 mode 2 在 `mm1ResGm` 被读出并乘 key scale 之后，才根据候选外 mask 将 score 降到有限的 `NEG_HUGE`；候选外可见位置保留实际索引，在候选不足 topk 时仍可进入结果作为填充。故将 mode 2 改成“候选外一律 -1”或无条件只计算候选块，可能改变结果。若要以跳过非候选块重组计算，必须另证在实际合法输入和 ties/有限分数边界下保留这种候选不足时的行为；现阶段既未证充分候选，也不能断言消除了相应计算或计算占比。

`quant_lightning_indexer_v2_kernel_arch22.h:701–815` 的源码还显示，合法 `curRunInfo` 的 `ComputeMm1` 在 AIC 上执行，AIV 在跨核同步后对前一块运行 `ProcessVec1`；该调用点没有按候选 mask 跳过 `ComputeMm1` 的分支。**这是保存源码的控制流观察**，不是对历史采样是否执行此二进制、其全部可优化路径、实际被处理块数或耗时原因的证明。source 文件 SHA256：vector `fcddf4e682e5c742f1190dbe673da909bcdf745dbd6dc1cb66f700fa9b41b8a8`，kernel `d35c26e6cb781aebad163a2c85cbd84f68c7c838ebe5ef9f67e6a42777949df6`。后续如需将源码推导绑定历史运行，最低补证包括加载二进制/hash、mode 和逐行 candidate/effective-length、storage/view 身份与同步版本。

进一步限定该观察的覆盖：`kernel_arch22.h:540–567,595–603,735–758` 的循环边界由有效序列长度、attention mask 和 `s2Start/s2End` 等切分信息确定，局部计算 `s2LoopEnd` 未读取 candidate mask；但这些切分信息由更早的 tiling 提供，**仅凭这里不能排除上游 tiling 已删除某些工作**，也不能因此声称运行时对全部候选外块都执行了矩阵计算。`Process()` 依次调用 `ProcessMain()` 和 `ProcessDecode()`（`669–679`），后者这里是 AIV 的 LD 合并（`822–830`），而不是另一个所见 MM1 调用。

对合法块，`service_vector_arch22.h:825–856` 在载入 MM1 输出、转换并乘 K scale 后，才为 mode 2 构造候选 mask 并执行有限负数惩罚；该值的位型定义在 `quant_lightning_indexer_v2_vector.h:26`，注释为 `-1e30`。源码以算术形式实现惩罚，涉及浮点舍入及非有限值，不能在未经值域约束时简单假定对所有输入都严格得到某个固定精确值。`869–879` 的源码区分有效位置的真实索引与对齐位置 `-1` 填充；候选外的有效位置可能在候选不足时进入 TopK。source 分支 `526–533,551–553,592–625` 对块内分数做归约、对无效块设 `-inf`，并将该行最后可见块的分数置 `+inf`，设计意图是使该块保留；对非有限分数及相同优先级的排序仍需另证。重组输出或变更分块时须保留原路径语义。以上均限定为所保存源码中的条件路径，不证明历史实例实际 mode、tiling、参与计算的块数或输入值域。

## 候选块号排序的跨层复用：源码条件候选（2026-10-30）

补充上一节的 candidate 块 ID 数据流：保存的 arch22 `quant_lightning_indexer_v2_service_vector_arch22.h:448–510` 中，mode 2 的 `BuildCandidateMask` 在 `info.isFirstS2InnerLoop` 成立时，将该行 `candidateTopkBlocks` 个 INT32 块号读入、转换为 FP32，与索引组成 pair，调用 `SortAll` 降序排序，再以 V→S flag 等待其对标量读取可见。后续 `CountGE` 和块命中循环使用排序后 pair 的值字段 `GetValue(2*j)`；该局部函数没有使用 pair 的原始序号作命中判断。这与 mode 1 按块分数选择 candidate 不同，不能把 mode 1 的 score 排序视为块 ID 数值已排序。

由冻结配置和 Python 源码可条件推导 L20 的共享 candidate 行可能被 L24/28/32/36 的 index source 再次消费。因此，可研究由 producer 或独立阶段按原有比较规则对每个 candidate 行排序一次，供这些层的 mode 2 复用；这需要修改或扩展 QLI 接口/实现，让消费者接受其所需的排序表示或跳过现有 SortAll，**没有现成开关或已验证收益**。已有源码是否使每个逻辑行只排序一次、每个物理分片如何复用、排序结果从 UB 跨层保存是否引入额外 GM 搬运与同步，均未闭合，不能写成必然节省 3 次排序或一个确定时间。

必要条件包括：在**同一 forward、相同有效行映射和相同 candidate 版本**中验证 L20 输出与每个 consumer 输入的 storage/offset/内容及生命周期；保留完整块号、重复项、`-1`、INT32→FP32 转换、排序比较值、候选不足时的有限惩罚与非候选填充行为、原 ABI 和其他观察者；发布时处理跨核/流可见性、固定图地址与覆写等待。源路径的 mode1 score 排序、尾块 pin 与 topk 语义不能改变。当前 `isFirstS2InnerLoop`、多 S2 切分、每个 consumer 的 block offset、LD 和可用 UB 资源需要分别绑定，不能凭配置和 shape 假定历史调用的四层均执行同一操作或拥有同一候选行。

异常/无效输入还需要逐行处理：`service_vector_arch22.h:324–345` 的 `CleanInvalidOutput` 在 source mode 将对应 candidate 行填充 `INVALID_IDX`；`kernel_arch22.h:419–440` 的 `DealActSeqLenIsZero` 存在逐行调用该 helper 的路径。它与 Python 对整个 `max_cache_seq_len==0` 返回全 `-1` 的分支不同；并不证明所有其他无效/LD 分支均已覆盖或历史路径执行该代码。共享缓存不能只判断整个 batch 非空，也不能复用上一 forward 的旧排序行。源文件 SHA 分别为 `fcddf4e682e5c742f1190dbe673da909bcdf745dbd6dc1cb66f700fa9b41b8a8` 和 `d35c26e6cb781aebad163a2c85cbd84f68c7c838ebe5ef9f67e6a42777949df6`。以上为可证伪的结构候选，尚未进行 NPU 测试、数值验证或实现修改。

### 排序视图的别名、空间和数值条件

`model.py:991–1012` 构造的 `candidate_buffer` 以同一个 Python tensor 同时赋给 `self.candidate_indices_buffer` 和 `shared_attention_state.candidates`；因而若直接对它原地重排，两个属性观察到的内容次序都会改变。局部 Python 显式引用检索不是对自定义算子、hook、图捕获或外部消费者的穷尽证明，不能仅凭 mode 2 中的 membership 逻辑认为可以改变原始发布顺序。另建只保存 FP32 排序值的独立视图，在条件 shape `[1023,1,2048]` 下的逻辑容量为 `1023×2048×4 = 8,380,416 B`（7.9921875 MiB）；若对配置上限 8192 行完整预分配则为 64 MiB。另存双 lane 的 value/index pair 分别为 15.984375 / 128 MiB。这只是不同设计的容量账本，不是实际分配、必需空间或 HBM 流量；原地排序、压缩表示与分块生命周期可能改变账本，但必须另证语义及代价。

non-packed `BuildCandidateMask` 对 INT32 候选执行 FP32 转换，块基址及索引亦以浮点形成后比较。不能不加条件改用精确整数 membership。作为仅验证数值概念的 CPU 示例，IEEE FP32 最近偶数下整数 16777217 与 16777216 均表示为 16777216；若块基址为 16777216，相邻块 ID 的 FP32 表示可能相同，精确整数比较则区分它们。CANN 目标 `CAST_NONE`、`Adds` 的确切舍入需同版本验证；已保存请求的模型长度为 34816，但未绑定有效块号或 tiling，不证明该大值实际可达。这个反例不排除保留原转换规则后对值预排序；若进一步证实每个有效块 ID、块基址及相关加法都在精确整数范围，还可在该范围重新验证整数实现。

不同后端不能直接共享同一证明：`mixed_quant_attention.py:82–95,156–166,198–214` 的 packed 路径同时传 candidates 与 candidate_lengths；此处 non-packed arch22 函数在构造 mask 时读取整个 candidate 行，不使用这一长度参数。假如某接口定义只有前 L 项有效，直接对全行排序可能改变有效前缀（如 L=1、`[1,2,0]` 重排为 `[2,1,0]`）；这是条件例子，不是关于 packed 实际 ABI 的断言。packed 的长度含义、排序契约、消费者和运行绑定须独立取证。

### source candidate 发布与 LD 分支的未解可达性

为了审查上文排序结果复用的 producer 条件，有限核查冻结 QLI `op_kernel`、`op_host`、`torch_extension/csrc` 中 candidate 输出符号和 arch22 LD 控制流。`service_vector_arch22.h:324–345,696–709` 的显式 GM 写点分别是无效行清 `INVALID_IDX` 和 `CopyOutCandTopkIndex`；后者在 `:890–921` 仅由 `needCopyOutGm && !info.isNeedLD` 分支调用。`:925–935` 与 `ProcessVec2:1000–1125` 的已检查 LD 分支操作普通 sparse index workspace / `indiceOutGm`，未见对应 candidate 合并输出。torch extension `quant_lightning_indexer.cpp:230–240` 的 candidate output 分配使用 `at::empty`。这些是限定文件及符号的源码观察；不能排除其他版本、别名写入、框架外初始化或本次运行不同二进制。

是否有有效 source mode1 行能进入 LD 未证。保存的 [qli_v2_two_level_topk_design.md](frozen-source/csrc/attention/quant_lightning_indexer_v2/docs/qli_v2_two_level_topk_design.md) SHA-256 `759330f97b4bab326507ee549857ae7ac5f248746c86b558751494e2c7a8dae4` 第 69、226–230、371 行声明该设计中 **910b** 的 metadata `supportFd_` 不启用、S2 不跨核、`isNeedLD` 不激活；不能自动把该文档绑定到 169 910C 的实际 metadata 二进制。`kernel_arch22.h:539–563` 中 LD 需要 `s2Start>0 || s2LoopEnd<s2BlockNum`，来源于 metadata 与有效行。所查 `op_host/quant_lightning_indexer_v2_tiling.cpp:427–480,1592–1605` 中没有找到 mode1+LD 的禁止条件；有限目录检索未找到对应 metadata AICPU 实现。故 source 在实际 910C 路径是否保证完整 candidate 发布仍是有界缺口，**不能断言 LD 可达、线上有 bug 或排序复用失效**；也不能将此存档内非 LD 分支的行为扩展为所有可能模式的完整发布证明。后续需要同版本 metadata 规则/实现或同次 mode 和分片字段，以及加载对象与代码身份，才能判断该条件与排序复用适用性。

各源码 SHA-256：kernel `d35c26e6cb781aebad163a2c85cbd84f68c7c838ebe5ef9f67e6a42777949df6`；vector `fcddf4e682e5c742f1190dbe673da909bcdf745dbd6dc1cb66f700fa9b41b8a8`；tiling `e82b98c0d406c376513b614fca77fcf2fb21d3ad29abe2b54a9c009812773066`；torch extension `3754467288227b4b4f63f6018af56a5ee1a94966b802ca35c93be0e31f78823d`。未操作 NPU 或目标实现。

### 逐行位图作为候选 mask 的条件性重组

限定保存的 non-packed arch22 `BuildCandidateMask`，候选项只取 `-1` 或 `[0,B)` 的整数，查询 tile 块号在 `[0,B)` 且 tile 右端不大于 `B≤2^24`；假定该目标版本的 INT32→FP32 转换、算术、正无穷和排序在该域按源码意图准确执行，排序完整且降序。源码 `:431–445,493–518` 对候选值排序，通过 `CountGE` 二分取 tile 范围内的候选，并对每个 tile 块求与这些候选值的最小绝对距离；距离为 0 表示命中，否则非零整数至少为 1，空集合为正无穷，最后 clamp 到 1。故该函数生成的候选外 mask，等价于对该行非负候选块建立 bitset 并查询成员：有则 0，无则 1；重复块和候选项顺序不影响 mask，`-1` 不会成为有效查询块。源码 `Brcb` 还将该结果按位置展开，重组必须保留同一块/位置映射与有效范围。

这里的块号是相对 S2 逻辑坐标，不是物理 cache page 或 token 位置；保存 host tiling 约束 candidate blockSize 为 8、候选数上限 2048 且为 64 的倍数。该等价只证明 mask，不允许删除或原地改动仍有其他消费者的 candidate 数组；跨层共享 bitset 还需相同行、forward、候选版本和同步。原源码在 MM1 和 key scale 后对候选外位置施加有限惩罚，候选不足时仍可能参与 TopK，不能据此跳过候选外计算。packed 路径、其他负数、值域外整数、非完整候选、不同二进制及实际执行身份都不在此证明范围。额外 bitset 构建、内存和同步可能有开销，不保证速度或任何收益。

`python3 agents/fusion/f12_candidate_mask_cpu_witness.py` 对人工 3 项候选、三块及所有非空 tile 区间做 640 次标量成员判断，脚本 SHA-256 `d34df2c0a9b40358958ca4e8efce002dc84b3037c1a29aad3d6a5476aa73d36d`。这只是辅助示例，其小 tile 不代表设备合法 tiling，也不能替代上述条件证明或设备数值验证。向量源码 SHA-256 `fcddf4e682e5c742f1190dbe673da909bcdf745dbd6dc1cb66f700fa9b41b8a8`。

#### 位图尾块与构造同步

冻结 arch22 kernel 的 `S2_BASE_SIZE=2048`（`kernel_arch22.h:93,201`），保存 host tiling 的 candidate block size 为 8，因此 `BuildCandidateMask` 构造一个完整 tile 的 256 个块、2048 个位置 mask。随后 `ProcessVec1` 的 penalty 算术传入的是实际 `cuS2Len`（`service_vector_arch22.h:820,852–856`），不能据此忽略向量指令实际对齐访问及其他观察者。条件例：若有效长度 2049，尾 tile 的完整块号范围为 256–511，但仅按有效长度分配的 bitset 有 257 个 bit；直接无边界查询完整 tile 会访问界外。要么保证 padded bitset 覆盖全部查询、要么证明边界屏蔽和仅生成实际使用部分对所有消费者、对齐和缓冲访问安全；上节 `base+tileBlkNum≤B` 的条件已经规避此例，但可能过于保守，不能未经新证明直接放宽。

构造 bitset 也有写竞争：同一 word 初值为 0，两线程分别非原子执行 `word |= 1` 和 `word |= 2`，若二者均先读到 0，最后可能为 1 或 2 而不是 3。需要逐 word 单写者、正确原子 OR 或等价无竞争归约，并让消费者等待结果就绪；集合去重不自动消除这个问题。`Brcb` 的 8 倍位置展开与源码注释、社区 A2 参考相容，但尚未绑定同版本硬件契约。空 query、最大 cache 长度为零、全 `-1` 和 non-packed 全行读取等特殊分支仍应保持；packed 的 candidate_lengths 属于不同接口，不能混用。
