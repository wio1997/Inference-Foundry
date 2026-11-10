# 171 空档：具体操作、数据用途与实际消费者

本轮将已有函数诊断与冻结源码连接，没有新跑模型或 profile。重点是解释“在准备什么数据、供谁使用”，再据此判断哪些工作能删除、共享或提前。

## 先区分两次采样：空档没有被证明从38%升到50%

|采样与用途|窗口ms/卡|Free ms/卡|Free占窗口|
|---|---:|---:|---:|
|原生阶段标记：总体分布|839.320|322.148|38.38%|
|细函数探针：定位具体函数|1025.503|491.193|47.90%|

两次采样不能直接比较为真实性能变化。细函数诊断新增标记边界约90.35ms与准备空档重叠，还有未量化扰动；简单减去90.35ms不能还原无探针时序。下文65.10%的分母是细诊断中的输入准备空档，不是整个推理。

## 已经能确定的结论

- 细函数诊断每卡 1025.503ms / 11周期，其中 Free 491.193ms；prepare input 与 Free 相交 477.800ms。
- CP build、base build、local-token helper 三个函数自身合计 311.050ms，占上述准备空档 65.10%。它们主要构造局部边界/长度、组地址、掩码、对象及原生参数，不是 CPU 在读取整份历史 KV。
- block table 是历史缓存的分页地址簿；slot mapping 是新 token 的写入地址。实际 KV payload 在后续设备 Scatter / indexer / attention 中读写。
- 28 项语义记录按 8 个阶段组织。这是读写关系清单，不是 28 个都已独立计时的 kernel。各字段的真实值、传输完成时间尚未采集。
- 沿源码发现的重复工作可先删再谈隐藏：loop046 的后25次命中长度投影没有值读者；首投影、已有共享和每资源读者仍保留。该补丁尚未通过原生集成验收。

## 数据交付顺序

这是依赖顺序，不是把并行 CPU、H2D 和设备工作强行画成串行耗时。地址表拷贝在源码中已经提前启动。

**S1 更新本轮请求**：确定哪些请求继续、每条处理几个 token、上轮实际接受多少。 交付：请求顺序、调度数量、上一轮有效计数

**S2 拼好输入与位置**：把多条请求展平，取 token ID，建立每条请求在扁平数组中的边界。 交付：input_ids / query_start_loc / positions

**S3 确定长度与执行档位**：修正投机接受结果，协调 DP 的输入规模和图档位，处理 padding。 交付：有效上下文长度 / padded 大小 / 图模式

**S4 建立缓存地址**：上传逻辑页到物理块的映射，算出新 token 应写入的槽位。 交付：block_table / slot_mapping；不是 KV 数据本体

**S5 生成每卡 CP 视图**：按该卡负责的 query 区间重算边界、局部长度、切片和最大值。 交付：local QSL / local seq_lens / token range

**S6 生成各缓存组参数**：准备压缩长度、二维槽位、RoPE、C2 ring；已有 batch / group 共享。 交付：分组地址、压缩状态、位置系数

**S7 生成并发布原生参数包**：收集回调，执行 SMLA / QLI / compressor 参数生成，发布就绪事件。 交付：native packet / stage-ready event

**S8 算子真正读写 KV**：Scatter 写入新 KV，索引器与注意力读取历史缓存并计算。 交付：更新的缓存、稀疏索引和 attention 输出；主要属于设备计算

## 函数时间账：只有这一张可以加总

每卡平均、11个闭合周期，函数自身区间与 prepare input ∩ device Free 的交集；已单列可辨识新增 FT 标记边界，仍受诊断扰动。

函数自身已排除被标记子函数；仍可包含 Python、库调用、异步提交与等待，不能称为纯 CPU 算力。调用次数来自完整诊断窗，含 target/draft；不能全部当 target 层数。字段卡片重复引用本表的同一函数，不额外加价。

|函数|准备空档交集 ms/卡|每周期 ms|占准备空档|
|---|---:|---:|---:|
|AscendDSAV41CPMetadataBuilder.build|138.641891|12.603808|29.017%|
|AscendDSAV41MetadataBuilder.build|118.491047|10.771913|24.799%|
|AscendDSACPMetadataBuilder._build_local_token_metadata|53.917078|4.901553|11.284%|
|NPUModelRunner._build_attention_metadata|27.377687|2.488881|5.730%|
|_ReplicatedCacheMetadataBuilder.take_device_metadata_tasks|8.246469|0.749679|1.726%|
|AscendDSAV41MetadataBuilder.build.build_c2_metadata|7.468938|0.678994|1.563%|
|_ReplicatedCacheMetadataBuilder._build_global_metadata|6.531719|0.593793|1.367%|
|GPUModelRunner._preprocess|5.035516|0.457774|1.054%|
|DeviceMetadataExecutor.submit|3.943750|0.358523|0.825%|
|AscendDSAV41MetadataBuilder.build.build_smla_metadata|2.997438|0.272494|0.627%|
|AscendDSAV41MetadataBuilder._build_batch_metadata|2.615953|0.237814|0.548%|
|NPUModelRunner._prepare_inputs|2.449594|0.222690|0.513%|
|AscendDSAV41MetadataBuilder._publish_task|2.007656|0.182514|0.420%|
|AscendDSAV41MetadataBuilder.take_device_metadata_tasks|1.955828|0.177803|0.409%|
|NPUModelRunner.execute_model|1.687078|0.153371|0.353%|
|AscendDSACPMetadataBuilder._local_token_range|1.239609|0.112692|0.259%|
|AscendDSAV41MetadataBuilder.build.build_qli_metadata|0.938859|0.085351|0.196%|
|AscendDSparkProposer.set_per_group_attn_metadata|0.799016|0.072638|0.167%|
|NPUModelRunner._sanitize_placeholder_input_ids_for_forward|0.401703|0.036518|0.084%|
|NPUModelRunner._sync_metadata_across_dp|0.231234|0.021021|0.048%|
|NPUModelRunner._calc_spec_decode_metadata|0.150016|0.013638|0.031%|
|MultiGroupBlockTable.commit_block_table|0.098922|0.008993|0.021%|
|NPUModelRunner._copy_spec_decode_metadata_to_device|0.064031|0.005821|0.013%|
|MultiGroupBlockTable.compute_slot_mapping|0.049484|0.004499|0.010%|
|NPUModelRunner._determine_batch_execution_and_padding|0.046078|0.004189|0.010%|
|GPUModelRunner._update_states|0.032547|0.002959|0.007%|
|NPUModelRunner._pad_query_start_loc_for_fia|0.026172|0.002379|0.005%|
|NPUModelRunner._update_states|0.007375|0.000670|0.002%|
|函数自身合计|387.452687|35.222972|81.091%|
|可辨识新增标记边界|90.346922|8.213357|18.909%|
|准备空档总计|477.799609|43.436328|100%|

这不是无探针生产时间，也不是可兑现收益。旧原生阶段样本的 76.302ms/周期、28.161ms 准备空档不与本轮绝对时间混算。

## 具体操作与数据字典

### D01 · 维护请求状态与本轮顺序（S1）

字段：`req_ids / num_scheduled_tokens / num_computed_tokens_cpu`。CPU 请求对象、索引表、每请求整数；B 表示本轮请求数。

**为什么需要**：让本轮扁平 token、缓存地址、采样结果都对应同一条请求，处理加入、退出和续跑。

**读取 → 生成**：scheduler_output 的新/继续/结束请求、上一轮状态、缓存块分配。 → 更新 InputBatch 的请求槽位、token 进度及缓存引用。

**实际消费者**：_prepare_inputs 的 req_indices、位置和长度；后续 metadata。

**依赖与提前边界**：调度决策和相应上轮状态就绪；不能用上一轮相同 B 代替相同请求顺序。

**优化判断**：先区分状态真正变化与相同数据重复封装；请求增删和槽位重排仍须保留。

**证据与计时**：构造函数已在本轮诊断执行；字段读写由冻结源码定位，未逐字段采样。字段独立时间：未测。
- 归属函数 `GPUModelRunner._update_states`：自身准备空档交集 0.032547ms/卡/11周期；这是同一函数总账，不是本字段的耗时。
- [冻结源码 model_runner_v1.py:961–997](../idle-attribution-171/source/model_runner_v1.py)；SHA256 `568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c`。
- [冻结源码 gpu_model_runner.py:1215–1257](../framework-timing-171/source/gpu_model_runner.py)；SHA256 `87c29d08c0bbf66993d8b984811e7325e35e6436242a773feb55ec88f16b2c51`。

### D02 · 把多请求展平成 token 所属关系（S2）

字段：`req_indices / query_pos / cu_num_tokens`。CPU→NPU；req_indices[T] 指明第 t 个 token 属于哪条请求；query_pos[T] 是本轮请求内偏移。

**为什么需要**：把按请求存放的输入，转换成算子使用的扁平 token 顺序。

**读取 → 生成**：每请求本轮 token 数 n_i。 → repeat(request_id,n_i)、前缀和 cu_num_tokens、请求内偏移；拷入持久设备缓冲。

**实际消费者**：input ID gather、设备 positions、slot mapping；QSL 也由同一前缀和生成。

**依赖与提前边界**：本轮请求顺序和 n_i 已确定即可准备结构部分，不必等 KV 数值生成。

**优化判断**：检查一次展平结果能否贯穿后续各消费者；不同顺序/不同 token 数必须另算。

**证据与计时**：构造函数已在本轮诊断执行；字段读写由冻结源码定位，未逐字段采样。字段独立时间：未测。
- 归属函数 `NPUModelRunner._prepare_inputs`：自身准备空档交集 2.449594ms/卡/11周期；这是同一函数总账，不是本字段的耗时。
- [冻结源码 model_runner_v1.py:1304–1336](../idle-attribution-171/source/model_runner_v1.py)；SHA256 `568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c`。
- [冻结源码 model_runner_v1.py:1533–1545](../idle-attribution-171/source/model_runner_v1.py)；SHA256 `568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c`。

### D03 · 取本轮 token ID 并交付 embedding（S2）

字段：`token_indices / input_ids`。CPU token 池→扁平 input_ids[T]→NPU；值是词表编号，不是 embedding 或 KV。

**为什么需要**：让 embedding 读到本轮实际 token；异步投机路径可能继续在设备侧修补输入。

**读取 → 生成**：token_ids_cpu、请求归属、位置、上轮采样/草稿所需 ID。 → token_indices=position+request_index×token_pool_stride；index_select 得到 input_ids，交 _prepare_input_ids。

**实际消费者**：模型 embedding 和输入预处理；placeholder 清理。

**依赖与提前边界**：CPU 已有的 token 可提前 gather；依赖上轮结果的 ID 必须等其实际生产者，不能只看 shape。

**优化判断**：分开稳定 prompt 输入与反馈 ID，避免把整段都判定可提前。继承 helper 内部未在本项逐行展开。

**证据与计时**：构造函数已在本轮诊断执行；字段读写由冻结源码定位，未逐字段采样。字段独立时间：未测。
- 归属函数 `NPUModelRunner._prepare_inputs`：自身准备空档交集 2.449594ms/卡/11周期；这是同一函数总账，不是本字段的耗时。
- [冻结源码 model_runner_v1.py:1375–1395](../idle-attribution-171/source/model_runner_v1.py)；SHA256 `568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c`。
- [冻结源码 model_runner_v1.py:1462–1465](../idle-attribution-171/source/model_runner_v1.py)；SHA256 `568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c`。
- [冻结源码 model_runner_v1.py:2484–2535](../idle-attribution-171/source/model_runner_v1.py)；SHA256 `568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c`。

### D04 · 标出每条请求的 query 边界（S2）

字段：`query_start_loc / cu_seqlens_q / QSL`。CPU 与 NPU 的整数前缀和，语义形状 [B+1]；实际缓冲可能含图 padding。

**为什么需要**：说明扁平 Q 数组中哪一段属于哪条请求；它不是历史 KV 内容，也不是序列总长度。

**读取 → 生成**：本轮请求顺序、n_i 和前缀和。 → QSL[0]=0；QSL[i+1]=Σ(j≤i)n_j；复制到设备并处理尾部。

**实际消费者**：CP 区间裁剪、slot mapping、C2 ring、原生 SMLA/QLI 参数及最终注意力。

**依赖与提前边界**：本轮 n_i 和顺序确定；异步 H2D 的 CPU 源缓冲在读完前不可覆写。

**优化判断**：同代同语义边界可共享计算；是否可删某份 H2D 还要核对其 first owner、持久地址和图读者。

**证据与计时**：构造函数已在本轮诊断执行；字段读写由冻结源码定位，未逐字段采样。字段独立时间：未测。
- 归属函数 `NPUModelRunner._prepare_inputs`：自身准备空档交集 2.449594ms/卡/11周期；这是同一函数总账，不是本字段的耗时。
- [冻结源码 model_runner_v1.py:1435–1465](../idle-attribution-171/source/model_runner_v1.py)；SHA256 `568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c`。
- [冻结源码 ascend-dsa_v41_cp.py:125–132](../idle-attribution-171/source/ascend-dsa_v41_cp.py)；SHA256 `039ec9d18e0a8a4aa39f9d5b4bea15204cea3f397a3fce97a5097b266bc9822e`。
- [冻结源码 dsa_v41.py:1171–1206](../framework-timing-171/source/dsa_v41.py)；SHA256 `3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd`。

### D05 · 获取上一轮真正接受的 token 数（S1）

字段：`valid_sampled_token_count / prev_num_draft_tokens / prev_positions`。设备采样计数及其 CPU 镜像、上轮请求位置和草稿宽度；是反馈状态。

**为什么需要**：投机 token 不一定全接受，下一轮真实进度须按实际有效数恢复。

**读取 → 生成**：上一轮 target 校验/采样结果、请求位置映射。 → side stream 等待主流后异步 D2H count，记录 event；保留 GPU count 供下轮纠偏。

**实际消费者**：GPU num_computed_tokens 修正、CPU optimistic length 修正及 scheduler bookkeeping。

**依赖与提前边界**：必须等实际接受计数产生；CPU 读之前等 D2H event。代码注释说稳态常已完成，不等于实测等待为零。

**优化判断**：计数产生后可与后续独立工作重叠；不能把反馈依赖本身消除，也不把整份 KV 搬回 CPU。

**证据与计时**：源码存在且被本负载使用的准备路径引用；此 helper 没有独立细函数计时。字段独立时间：未测。
- [冻结源码 model_runner_v1.py:1848–1896](../idle-attribution-171/source/model_runner_v1.py)；SHA256 `568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c`。
- [冻结源码 model_runner_v1.py:1502–1532](../idle-attribution-171/source/model_runner_v1.py)；SHA256 `568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c`。

### D06 · 生成设备权威进度与上下文长度（S3）

字段：`num_computed_tokens / seq_lens`。NPU 每请求整数 [B]；seq_lens 是本轮更新后可见的有效 token 长度。

**为什么需要**：防止 attention/indexer 把被拒绝草稿或 padding 当有效历史。

**读取 → 生成**：CPU 初始进度、前一轮 GPU accepted count、prev position/draft width、当前 scheduled count。 → 先纠正 computed，再令 seq_lens=computed+scheduled；无效尾部置零。

**实际消费者**：positions、CP 局部长度、base 持久长度、压缩长度、SMLA/QLI。

**依赖与提前边界**：反馈 count 和本轮请求映射就绪，且输入缓冲可安全复用。

**优化判断**：可以共用同代权威结果；不能用未经证明的 CPU 乐观上界替代设备有效值。

**证据与计时**：构造函数已在本轮诊断执行；字段读写由冻结源码定位，未逐字段采样。字段独立时间：未测。
- 归属函数 `NPUModelRunner._prepare_inputs`：自身准备空档交集 2.449594ms/卡/11周期；这是同一函数总账，不是本字段的耗时。
- [冻结源码 model_runner_v1.py:1502–1532](../idle-attribution-171/source/model_runner_v1.py)；SHA256 `568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c`。
- [冻结源码 model_runner_v1.py:1598–1625](../idle-attribution-171/source/model_runner_v1.py)；SHA256 `568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c`。

### D07 · 维护 CPU 长度镜像与上界（S3）

字段：`optimistic_seq_lens_cpu / _seq_lens_cpu / max_seq_len`。CPU [B] 镜像，部分路径先假设草稿全接受，再在 guard 下纠偏；与设备权威值分开记。

**为什么需要**：减少为了 Host 分支、最大值和构造参数而额外读取设备长度的需要；不同字段有不同合同。

**读取 → 生成**：CPU computed+scheduled；必要时读已完成 D2H 的 valid count。 → 乐观长度、纠正后的 CPU 镜像、max 长度及 common 中的不同字段。

**实际消费者**：丢弃采样 mask、CP CPU helper、max_seq_len/原生属性；async 下普通 seq_lens_cpu 字段可能置 None。

**依赖与提前边界**：结构上界可先算；精确 CPU 长度必须等有效计数 copy 完成。

**优化判断**：分别追上界、已纠正 CPU 值和设备值的读者；不把三者按名字合并。

**证据与计时**：构造函数已在本轮诊断执行；字段读写由冻结源码定位，未逐字段采样。字段独立时间：未测。
- 归属函数 `NPUModelRunner._prepare_inputs`：自身准备空档交集 2.449594ms/卡/11周期；这是同一函数总账，不是本字段的耗时。
- 归属函数 `NPUModelRunner._build_attention_metadata`：自身准备空档交集 27.377687ms/卡/11周期；这是同一函数总账，不是本字段的耗时。
- [冻结源码 model_runner_v1.py:1449–1459](../idle-attribution-171/source/model_runner_v1.py)；SHA256 `568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c`。
- [冻结源码 model_runner_v1.py:1608–1620](../idle-attribution-171/source/model_runner_v1.py)；SHA256 `568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c`。
- [冻结源码 model_runner_v1.py:3656–3668](../idle-attribution-171/source/model_runner_v1.py)；SHA256 `568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c`。

### D08 · 生成绝对位置（S2）

字段：`positions`。NPU int64 [T]；CPU 也有暂存位置；表示每个 query 在原序列的位置。

**为什么需要**：控制 RoPE 系数、因果可见区间、缓存地址和 C2 完成组判定。

**读取 → 生成**：已纠正 num_computed_tokens、req_indices、query_pos。 → position[t]=computed[owner[t]]+query_pos[t]；CP 再取实际 token 切片。

**实际消费者**：slot mapping、RoPE、C2 complete mask 和 source positions。

**依赖与提前边界**：设备 computed 与 token 归属就绪；CPU 的早期估计不自动等于最终位置。

**优化判断**：共同位置向量可复用；不能跨 speculative rejection、请求重排或代次复用。

**证据与计时**：构造函数已在本轮诊断执行；字段读写由冻结源码定位，未逐字段采样。字段独立时间：未测。
- 归属函数 `NPUModelRunner._prepare_inputs`：自身准备空档交集 2.449594ms/卡/11周期；这是同一函数总账，不是本字段的耗时。
- [冻结源码 model_runner_v1.py:1598–1606](../idle-attribution-171/source/model_runner_v1.py)；SHA256 `568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c`。
- [冻结源码 ascend-dsa_v41_cp.py:143–153](../idle-attribution-171/source/ascend-dsa_v41_cp.py)；SHA256 `039ec9d18e0a8a4aa39f9d5b4bea15204cea3f397a3fce97a5097b266bc9822e`。
- [冻结源码 dsa_v41.py:1337–1356](../framework-timing-171/source/dsa_v41.py)；SHA256 `3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd`。

### D09 · 协调 DP 输入规模与执行模式（S3）

字段：`packed_tensor / num_tokens_across_dp / cudagraph_mode`。CPU 小整数通信包 [2,DP]：各 DP token 数和 graph mode，不是 KV 或激活交换。

**为什么需要**：保证跨 DP 协作时规模和执行路径兼容，避免集体调用失配。

**读取 → 生成**：本 DP token 数、图模式、MoE 通信策略。 → CPU group all_reduce 后得到各 DP 数量、最大数量和统一模式；按策略决定 padding。

**实际消费者**：_determine_batch_execution_and_padding、图调度及相应 MoE 执行。

**依赖与提前边界**：所有参与方发布本轮控制值；等待可能来自对端就绪，不能等同本地纯计算。

**优化判断**：先判断该策略是否需要每轮一致性和 uniform input；不能直接删除 collective。

**证据与计时**：构造函数已在本轮诊断执行；字段读写由冻结源码定位，未逐字段采样。字段独立时间：未测。
- 归属函数 `NPUModelRunner._sync_metadata_across_dp`：自身准备空档交集 0.231234ms/卡/11周期；这是同一函数总账，不是本字段的耗时。
- [冻结源码 model_runner_v1.py:825–884](../idle-attribution-171/source/model_runner_v1.py)；SHA256 `568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c`。
- [冻结源码 model_runner_v1.py:3325–3432](../idle-attribution-171/source/model_runner_v1.py)；SHA256 `568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c`。

### D10 · 补齐图回放形状和无效请求（S3）

字段：`num_reqs_padded / num_tokens_padded / PAD / dummy QSL`。捕获图档位及 padded 请求/token 边界；有效数量与容量分别保存。

**为什么需要**：固定图地址和形状，同时保证 padding 不读写真实 cache。

**读取 → 生成**：实际 B/T、DP 协调结果、已捕获 graph 大小。 → QSL dummy 边界、无效 seq_lens=0、slot=-1、block table 尾部清理。

**实际消费者**：图回放、native metadata、注意力与缓存写入。

**依赖与提前边界**：模式与档位已确定；清零/掩码要先于可能读取该尾部的消费者。

**优化判断**：可按真实有效数量减少工作，但必须保留图合同及空 query 事件/清零行为。

**证据与计时**：构造函数已在本轮诊断执行；字段读写由冻结源码定位，未逐字段采样。字段独立时间：未测。
- 归属函数 `NPUModelRunner._pad_query_start_loc_for_fia`：自身准备空档交集 0.026172ms/卡/11周期；这是同一函数总账，不是本字段的耗时。
- 归属函数 `NPUModelRunner._build_attention_metadata`：自身准备空档交集 27.377687ms/卡/11周期；这是同一函数总账，不是本字段的耗时。
- [冻结源码 model_runner_v1.py:1060–1108](../idle-attribution-171/source/model_runner_v1.py)；SHA256 `568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c`。
- [冻结源码 model_runner_v1.py:3561–3590](../idle-attribution-171/source/model_runner_v1.py)；SHA256 `568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c`。

### D11 · 上传逻辑页到物理块的地址表（S4）

字段：`block_table / block_table_tensor`。CPU→NPU 整数表 [B,可容纳逻辑块数]；每项是分配器给的物理 cache 块 ID。

**为什么需要**：让算子按序列逻辑位置找到历史 KV 的实际设备地址；上传的是地址表。

**读取 → 生成**：请求当前持有的各物理 KV group 块列表。 → 各组 commit_block_table，将有效请求行复制到设备；准备阶段最先启动以覆盖后续 CPU 工作。

**实际消费者**：slot mapping、C2 ring 的块所有者、QLI 和 SMLA 的分页寻址。

**依赖与提前边界**：本轮请求及块分配已确定；CPU 源和目标 device buffer 的寿命都要安全。

**优化判断**：评估只更新变化部分或同组共享；物理 group 不同，不能只因 shape 相同共用表。

**证据与计时**：构造函数已在本轮诊断执行；字段读写由冻结源码定位，未逐字段采样。字段独立时间：未测。
- 归属函数 `MultiGroupBlockTable.commit_block_table`：自身准备空档交集 0.098922ms/卡/11周期；这是同一函数总账，不是本字段的耗时。
- [冻结源码 model_runner_v1.py:1302–1305](../idle-attribution-171/source/model_runner_v1.py)；SHA256 `568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c`。
- [冻结源码 ascend-block_table.py:519–521](../idle-attribution-171/source/ascend-block_table.py)；SHA256 `215305c699a9df9c118f47c6f7b71d8457ef54e98b5b8571536fd373de7b21ed`。
- [冻结源码 dsa_v41.py:644–674](../framework-timing-171/source/dsa_v41.py)；SHA256 `3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd`。

### D12 · 计算新 token 应写入的缓存位置（S4）

字段：`slot_mapping`。NPU 整数 [T]；通常是物理块和块内 offset 的展平地址，PAD=-1。

**为什么需要**：Scatter 等写入新 KV 时避免搜索页表；与“读历史 KV”用途不同。

**读取 → 生成**：本轮 QSL、最终 positions、各组 block table/block size、循环缓存配置。 → 分页/环形规则映射到每个 token 的目的槽；已有 fused-groups 快路径，guard 不满足则逐组。

**实际消费者**：base 的二维/压缩映射；KV cache 写入算子。

**依赖与提前边界**：设备位置和块表就绪，且目的 cache 槽可以被当前代合法覆写。

**优化判断**：先核对实际快路径；不能把源码中已有 fused-groups 宣称为新融合。

**证据与计时**：构造函数已在本轮诊断执行；字段读写由冻结源码定位，未逐字段采样。字段独立时间：未测。
- 归属函数 `MultiGroupBlockTable.compute_slot_mapping`：自身准备空档交集 0.049484ms/卡/11周期；这是同一函数总账，不是本字段的耗时。
- [冻结源码 model_runner_v1.py:1622–1626](../idle-attribution-171/source/model_runner_v1.py)；SHA256 `568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c`。
- [冻结源码 ascend-block_table.py:469–502](../idle-attribution-171/source/ascend-block_table.py)；SHA256 `215305c699a9df9c118f47c6f7b71d8457ef54e98b5b8571536fd373de7b21ed`。

### D13 · 把全局 query 区间裁到本卡（S5）

字段：`cp_token_range / local_query_start/end / local QSL`。CPU 计算的区间 start/end、每请求局部 query 数与局部 QSL；再 pin/H2D 到持久 buffer。

**为什么需要**：当前 CP 实现把 query token 分给不同卡；每卡只为自己负责的 query 构造 attention 参数。

**读取 → 生成**：全局 QSL、B/T、TP rank/size、actual token 数。 → 区间交集得到 local query 长度；cumsum 得 local QSL；裁掉 padding，slice positions/slot/RoPE。

**实际消费者**：local common、local SMLA/QLI；全局 KV 侧仍有独立 metadata。

**依赖与提前边界**：rank、B/T/QSL 可先准备结构部分；helper 同时含长度相关部分，不能整函数都视为独立。

**优化判断**：共享同代重复区间/前缀和；保留实际因果/非因果、空尾、每资源地址和输出写入合同。

**证据与计时**：构造函数已在本轮诊断执行；字段读写由冻结源码定位，未逐字段采样。字段独立时间：未测。
- 归属函数 `AscendDSACPMetadataBuilder._build_local_token_metadata`：自身准备空档交集 53.917078ms/卡/11周期；这是同一函数总账，不是本字段的耗时。
- 归属函数 `AscendDSAV41CPMetadataBuilder.build`：自身准备空档交集 138.641891ms/卡/11周期；这是同一函数总账，不是本字段的耗时。
- [冻结源码 ascend-dsa_cp.py:1146–1225](../idle-attribution-171/source/ascend-dsa_cp.py)；SHA256 `7e8dd93eeda988866138342523ecfa7cc9c42ce67669d4b1507ee8c5f3452305`。
- [冻结源码 ascend-dsa_v41_cp.py:112–153](../idle-attribution-171/source/ascend-dsa_v41_cp.py)；SHA256 `039ec9d18e0a8a4aa39f9d5b4bea15204cea3f397a3fce97a5097b266bc9822e`。

### D14 · 生成本卡真正可见的上下文长度（S5）

字段：`suffix / local_seq_lens / local_common`。CPU suffix[B]→pin/H2D；NPU local length[B]；还包含 query 有效性 mask。

**为什么需要**：本卡只处理请求中某段 query，因果上下文不能包含位于这段之后的 query。

**读取 → 生成**：全局 query 结束边界、本卡 actual_start/end、设备权威 seq_lens、本卡 QSL。 → suffix=query_end−clamp(query_end,local_start,local_end)；local_len=max(seq_len−suffix,0)，无本地 query 的请求置零。

**实际消费者**：首 base batch producer 的持久长度，再到压缩长度和原生 packet；命中后可能不再消费新投影。

**依赖与提前边界**：结构 suffix 可先算；最终减法依赖设备权威长度；noncausal 路径 suffix=0。

**优化判断**：loop046 已确认源码上的死值：目标同 frame 26 次中首投影保留，后 25 次 batch hit 新长度值无读者；补丁待原生集成验收，draft 不套 26 倍。

**证据与计时**：构造函数已在本轮诊断执行；字段读写由冻结源码定位，未逐字段采样。字段独立时间：未测。
- 归属函数 `AscendDSAV41CPMetadataBuilder.build`：自身准备空档交集 138.641891ms/卡/11周期；这是同一函数总账，不是本字段的耗时。
- [冻结源码 ascend-dsa_v41_cp.py:125–154](../idle-attribution-171/source/ascend-dsa_v41_cp.py)；SHA256 `039ec9d18e0a8a4aa39f9d5b4bea15204cea3f397a3fce97a5097b266bc9822e`。
- [冻结源码 dsa_v41.py:902–927](../framework-timing-171/source/dsa_v41.py)；SHA256 `3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd`。
- [冻结源码 dsa_v41.py:961–979](../framework-timing-171/source/dsa_v41.py)；SHA256 `3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd`。

### D15 · 封装共同 batch 坐标和按组对象（S6）

字段：`common_v41_batch_metadata / common_v41_metadata / metadata_by_layer`。Python 字典、对象和设备持久 tensor 引用；batch 数据与物理 cache group 数据分域。

**为什么需要**：连接模型层、物理缓存和原生算子。多数是组织数据及触发小操作，不是在 CPU 重算注意力。

**读取 → 生成**：common 输入、builder/层分组、各 group 表和 slots。 → 每轮重置共享字典；首 producer 复制持久 seq_lens，后续 batch hit 使用 canonical 坐标；metadata 交给所属 layer。

**实际消费者**：各组 builder、native callback 和模型层 forward。

**依赖与提前边界**：CPU 标量可按其输入独立准备；持久设备字段要等相应设备输入及旧读者退役。

**优化判断**：已有共享必须计入基线；继续查共享前仍被重复构造但最终没有读者的工作。

**证据与计时**：构造函数已在本轮诊断执行；字段读写由冻结源码定位，未逐字段采样。字段独立时间：未测。
- 归属函数 `NPUModelRunner._build_attention_metadata`：自身准备空档交集 27.377687ms/卡/11周期；这是同一函数总账，不是本字段的耗时。
- 归属函数 `AscendDSAV41MetadataBuilder._build_batch_metadata`：自身准备空档交集 2.615953ms/卡/11周期；这是同一函数总账，不是本字段的耗时。
- 归属函数 `AscendDSAV41MetadataBuilder.build`：自身准备空档交集 118.491047ms/卡/11周期；这是同一函数总账，不是本字段的耗时。
- [冻结源码 model_runner_v1.py:3760–3827](../idle-attribution-171/source/model_runner_v1.py)；SHA256 `568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c`。
- [冻结源码 dsa_v41.py:902–979](../framework-timing-171/source/dsa_v41.py)；SHA256 `3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd`。

### D16 · 转换每组二维和压缩缓存槽位（S6）

字段：`slot_mapping_2d / flat_slots / valid mask`。NPU [T,2] 的 (物理块,offset) 以及 [T] flat 地址；无效项=-1。

**为什么需要**：不同原生 cache writer 使用不同地址布局；不能把未完成的一对 token 当压缩 KV 写入。

**读取 → 生成**：原 slot、positions、storage block size、压缩比、actual B/T、skip_update。 → 压缩坐标变换→valid mask→div/remainder→where→copy；C2 只在压缩组完成的位置写入。

**实际消费者**：SWA / long KV / index K 的 cache 写入和访问；compressor_state 改用 ring 所有权。

**依赖与提前边界**：最终槽位、位置及实际数量就绪；物理组内同 layout 可复用，跨组不可默认共用。

**优化判断**：现有 group-local key 已共享 long KV/index K 映射；需从剩余分组转换和共同掩码判断增量价值。

**证据与计时**：构造函数已在本轮诊断执行；字段读写由冻结源码定位，未逐字段采样。字段独立时间：未测。
- 归属函数 `AscendDSAV41MetadataBuilder.build`：自身准备空档交集 118.491047ms/卡/11周期；这是同一函数总账，不是本字段的耗时。
- [冻结源码 dsa_v41.py:979–1039](../framework-timing-171/source/dsa_v41.py)；SHA256 `3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd`。

### D17 · 生成压缩缓存长度与未满组残数（S6）

字段：`cache_seq_lens / cmp_residual`。NPU 每请求整数；C2 时 floor(seq_lens/2) 和 seq_lens mod 2。

**为什么需要**：告诉索引和注意力：有多少完整压缩条目，末尾还有多少未满组 token。

**读取 → 生成**：canonical 设备长度、压缩比。 → div/remainder 写入持久长度/残数；同一 batch 的 lengths:c2 已共享。

**实际消费者**：SMLA / QLI metadata 及最终 SMLA / QLI kernel。

**依赖与提前边界**：canonical 精确长度就绪；不能把 CPU 上界除以 2 当精确结果。

**优化判断**：优先保留已共享的首 producer，排查共享前重复投影，避免再建议一次已存在的 div/remainder 共享。

**证据与计时**：构造函数已在本轮诊断执行；字段读写由冻结源码定位，未逐字段采样。字段独立时间：未测。
- 归属函数 `AscendDSAV41MetadataBuilder.build`：自身准备空档交集 118.491047ms/卡/11周期；这是同一函数总账，不是本字段的耗时。
- [冻结源码 dsa_v41.py:1038–1056](../framework-timing-171/source/dsa_v41.py)；SHA256 `3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd`。
- [冻结源码 dsa_v41.py:1178–1200](../framework-timing-171/source/dsa_v41.py)；SHA256 `3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd`。

### D18 · 按位置取旋转位置系数（S6）

字段：`cos / sin / rope_views`。位置索引→NPU cos/sin 系数视图；不是 KV 或 attention 输出。

**为什么需要**：为 Q/K 位置旋转、压缩器所需源位置提供系数。

**读取 → 生成**：positions、RoPE 表/配置、global metadata。 → 取所需位置的系数；CP local 复用 global slice，避免再次 gather 覆写仍在使用的全局 buffer。

**实际消费者**：RoPE/attention 前处理和 C2 source cos/sin。

**依赖与提前边界**：位置就绪且被引用的系数 buffer 未被下一次 gather 覆写。

**优化判断**：全局/local 已有 slice 复用；剩余优化须区分取系数、实际旋转和源 buffer 生命周期。

**证据与计时**：构造函数已在本轮诊断执行；字段读写由冻结源码定位，未逐字段采样。字段独立时间：未测。
- 归属函数 `AscendDSAV41MetadataBuilder.build`：自身准备空档交集 118.491047ms/卡/11周期；这是同一函数总账，不是本字段的耗时。
- 归属函数 `AscendDSAV41CPMetadataBuilder.build`：自身准备空档交集 138.641891ms/卡/11周期；这是同一函数总账，不是本字段的耗时。
- [冻结源码 dsa_v41.py:1057–1070](../framework-timing-171/source/dsa_v41.py)；SHA256 `3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd`。
- [冻结源码 ascend-dsa_v41_cp.py:155–164](../idle-attribution-171/source/ascend-dsa_v41_cp.py)；SHA256 `039ec9d18e0a8a4aa39f9d5b4bea15204cea3f397a3fce97a5097b266bc9822e`。
- [冻结源码 dsa_v41.py:1358–1378](../framework-timing-171/source/dsa_v41.py)；SHA256 `3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd`。

### D19 · 描述 C2 ring 状态更新（S6）

字段：`ring_meta / complete_mask / source_positions`。设备 ring_meta 五行及 token mask/源位置数组；是状态操作的描述，不是压缩 KV 本体。

**为什么需要**：让压缩器知道从哪条请求/哪段输入读取、更新哪个状态，以及何时生成一对 token 的压缩结果。

**读取 → 生成**：QSL、精确 seq_lens、input positions、block table 首块、actual B/T、skip_update。 → 五行分别写已有前缀长度、本轮有效数量、输入起点两份、状态块 ID；奇数位置且有效为完成组，源位置取 position−1。

**实际消费者**：global C2 compressor；local CP 的 compressor 通常关闭，不按每层每卡都重复计价。

**依赖与提前边界**：精确长度/位置/表就绪；状态被上一位读者释放，且本轮确实允许更新。

**优化判断**：metadata 构造与真正压缩运算分开定价；跨组 ring 所有者和副作用不能合错。

**证据与计时**：构造函数已在本轮诊断执行；字段读写由冻结源码定位，未逐字段采样。字段独立时间：未测。
- 归属函数 `AscendDSAV41MetadataBuilder.build.build_c2_metadata`：自身准备空档交集 7.468938ms/卡/11周期；这是同一函数总账，不是本字段的耗时。
- [冻结源码 dsa_v41.py:1304–1357](../framework-timing-171/source/dsa_v41.py)；SHA256 `3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd`。

### D20 · 生成 SMLA 原生参数包（S7）

字段：`smla_metadata / npu_sparse_flash_mla_metadata`。原生 metadata packet，持久设备缓冲；包含原生执行所需参数，内部二进制字段未解码。

**为什么需要**：提前把 attention 的动态形状/可见范围等转为原生算子需要的执行参数；不是计算 QK/softmax/PV。

**读取 → 生成**：QSL、原/压缩有效长度、残数、head 数/维度、mask/window、topk、layout、实际数量。 → 调用 metadata op，copy 到持久 packet；通过 smla:c{ratio} key 发布任务。空 query 保留原清零/事件行为。

**实际消费者**：实际 npu_sparse_flash_mla，使用前 wait ATTENTION/group。

**依赖与提前边界**：packet 所需 CPU 属性与设备长度就绪，且 packet buffer 可覆写。

**优化判断**：按真实 first owner 和完整 attrs 合并；不能凭同 shape 复用旧代 packet。不能从 Python 签名猜完整内部 tiling 格式。

**证据与计时**：构造函数已在本轮诊断执行；字段读写由冻结源码定位，未逐字段采样。字段独立时间：未测。
- 归属函数 `AscendDSAV41MetadataBuilder.build.build_smla_metadata`：自身准备空档交集 2.997438ms/卡/11周期；这是同一函数总账，不是本字段的耗时。
- [冻结源码 dsa_v41.py:1171–1216](../framework-timing-171/source/dsa_v41.py)；SHA256 `3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd`。
- [冻结源码 dsa_v41.py:650–682](../framework-timing-171/source/dsa_v41.py)；SHA256 `3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd`。

### D21 · 生成 QLI 原生参数包（S7）

字段：`qli_metadata / quant_lightning_indexer_metadata`。索引选择原生参数 packet，独立于 SMLA packet；910C 与 A5 的生成分支不同。

**为什么需要**：规定索引器如何解释各请求/缓存和压缩候选；还没有得到本轮实际 top-k 选择结果。

**读取 → 生成**：QSL、压缩有效长度/残数、index heads/dim、topk、压缩比、mask、layout。 → 按后端生成 QLI packet，copy 到持久输出并发布 INDEXER stage。

**实际消费者**：后续 QLI/indexer kernel 的 metadata 输入；实际索引结果再交给 attention。

**依赖与提前边界**：有效长度、QSL 和 index 配置就绪；与相应 cache/state 读取的就绪条件分别满足。

**优化判断**：不能因为无本地 query 就省略已要求的清零/发布；不把 A5 分支计入当前 910C。

**证据与计时**：构造函数已在本轮诊断执行；字段读写由冻结源码定位，未逐字段采样。字段独立时间：未测。
- 归属函数 `AscendDSAV41MetadataBuilder.build.build_qli_metadata`：自身准备空档交集 0.938859ms/卡/11周期；这是同一函数总账，不是本字段的耗时。
- [冻结源码 dsa_v41.py:1218–1303](../framework-timing-171/source/dsa_v41.py)；SHA256 `3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd`。
- [冻结源码 indexer.py:255–289](../gap-dataflow-171/source/indexer.py)；SHA256 `d5d6a438bbced4e63d729dac94c08869df30dab4a71ceacecdea211386b3db33`。

### D22 · 收集回调并按阶段发布（S7）

字段：`DeviceMetadataTask / inputs_ready / stage_ready / buffer_reusable`。Python closure/task list、stage+group 标识、NPU stream/event；是控制依赖。

**为什么需要**：让准备任务与模型计算有序交错，同时防止读未完成 packet 或覆写仍在读的 buffer。

**读取 → 生成**：各 builder 产出的延迟回调、图 descriptor、旧 buffer 退役事件。 → 排序→主流输入 event→metadata stream wait→执行回调→按 stage/group 记录就绪→consumer wait→release。

**实际消费者**：compressor、attention、indexer 的相应第一读者及下一代 writer。

**依赖与提前边界**：CPU 回调已生成只是“可提交”；设备输入、生产完成、消费结束、pinned 源退役是不同时间点。

**优化判断**：先排死工作，再按字段拆提前部分。额外流/线程需要独立资源和双端寿命证明，移动 submit 不等于可覆盖。

**证据与计时**：构造函数已在本轮诊断执行；字段读写由冻结源码定位，未逐字段采样。字段独立时间：未测。
- 归属函数 `DeviceMetadataExecutor.submit`：自身准备空档交集 3.943750ms/卡/11周期；这是同一函数总账，不是本字段的耗时。
- [冻结源码 device_metadata.py:74–151](../framework-timing-171/source/device_metadata.py)；SHA256 `efe13f0e801b70d7fba7498ac44062a8eccfd78e5b6303cf658d33dd22217580`。
- [冻结源码 dsa_v41.py:879–901](../framework-timing-171/source/dsa_v41.py)；SHA256 `3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd`。
- [冻结源码 model_runner_v1.py:3898–3907](../idle-attribution-171/source/model_runner_v1.py)；SHA256 `568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c`。

### D23 · 写入新产生的 KV 数据（S8）

字段：`swa_cache / scatter_cache_sk / slot_mapping`。设备上的实际 cache payload；本模型还分 SWA、long KV、index K 和 compressor state。

**为什么需要**：保存本轮结果，供本轮后续/未来 token 的注意力读取。

**读取 → 生成**：本轮新 K/V 或对应潜在表示、目标 slot 和 cache buffer。 → scatter_cache_sk/cache backend 把新 value 放入指定槽；压缩 cache 使用各自路径。

**实际消费者**：当前/后续 attention、indexer 或 compressor。

**依赖与提前边界**：新值计算完成、写入地址和缓存生命周期安全；不能在对应数据生成前完成写入。

**优化判断**：可考察生产者直写或与消费者融合以删中间物化；必要 cache 状态副作用仍须保留。

**证据与计时**：设备 ScatterNdUpdateSk 类型已测；具体每条任务到 SWA/long/index 的绑定未全部完成。字段独立时间：未测。
- [冻结源码 dsa_v41.py:297–311](../framework-timing-171/source/dsa_v41.py)；SHA256 `3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd`。

同窗设备类型 `ScatterNdUpdateSk`：18.444625ms/卡/11周期（不属于该字段的 Host 时间）。导出中的完整 shape 样例。cache 张量形状是池容量/视图，不等于每次实际读取字节数；有效请求、长度、mask、索引和 PAD 决定访问。例中 padded 槽位数不等于实际写入 token 数。

原样例 EP0：输入 `"17305,128,512;24,2;24,512"`；dtype `DT_BF16;INT32;DT_BF16`；输出 `"17305,128,512"`。全部去重样例保存在 JSON。

### D24 · 读取历史 KV 并完成注意力（S8）

字段：`ori_kv / cmp_kv / block_table / SparseFlashMla`。真实设备 cache 张量 + 寻址表 + 长度/稀疏索引 + 原生 packet。

**为什么需要**：复用历史 token 的 K/V，避免每个新 token 都重算全历史；通过寻址和稀疏规则限制有效读取。

**读取 → 生成**：Q、历史 SWA KV、可选压缩 KV、block tables、QSL、有效长度、mask/window 和 sparse indices。 → 实际 npu_sparse_flash_mla 读取 cache，完成注意力并写 output。

**实际消费者**：O 投影及后续层；新 output 不是 metadata。

**依赖与提前边界**：Q、所需 cache 写入和 native packet 均就绪；先 wait ATTENTION/group。

**优化判断**：若要判断 KV 读取带宽瓶颈，需算子内部访存量/有效带宽证据；22.042ms 的 SMLA 总时间不能直接当纯 KV 读时长。

**证据与计时**：同窗设备 SparseFlashMla 已测 22.042ms/卡/11周期；包含实际计算，不是 gap 时间。字段独立时间：未测。
- [冻结源码 dsa_v41.py:613–683](../framework-timing-171/source/dsa_v41.py)；SHA256 `3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd`。

同窗设备类型 `SparseFlashMla`：22.041875ms/卡/11周期（不属于该字段的 Host 时间）。导出中的完整 shape 样例。cache 张量形状是池容量/视图，不等于每次实际读取字节数；有效请求、长度、mask、索引和 PAD 决定访问。例中 padded 槽位数不等于实际写入 token 数。

原样例 EP0：输入 `"3,64,512;17305,128,1,512;17305,128,1,512;;3,1,512;4,272;4,272;5;;;;4;4;;;;64;1024"`；dtype `DT_BF16;DT_BF16;DT_BF16;DT_UNDEFINED;INT32;INT32;INT32;INT32;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;INT32;INT32;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED;FLOAT;INT32`；输出 `"3,64,512;"`。全部去重样例保存在 JSON。

### D25 · 按当前 query 选择历史候选（S8）

字段：`compressed_indices / QLI output`。设备稀疏索引，是本轮数据相关计算结果；与生成 QLI packet 是两件事。

**为什么需要**：避免 attention 扫描所有长上下文条目，选出本 query 需要的候选。

**读取 → 生成**：本层量化 query/query_scale、实际 index K cache/key_scale、weights、block/长度信息、QLI metadata 和可选候选过滤集合。 → 等待 INDEXER/group 后执行 npu_quant_lightning_indexer_v3，再 prepare_indexer_indices；attention 对齐/补齐后按其读取压缩 KV。

**实际消费者**：npu_sparse_flash_mla 的 cmp_sparse_indices。

**依赖与提前边界**：本层实际 query 和 index cache 就绪；不能用仅由 shape 得到的 metadata 提前替代最终索引。

**优化判断**：跨层可以研究候选表示共享；实际选择依赖当前 query，不可直接复用成同一个 top-k。

**证据与计时**：同窗 QLI 类型已测 27.259ms/卡/11周期；框架调用名 v3 与 trace 类型名 V2 不作为两个独立 kernel 计价；内部访存/计算未拆分。字段独立时间：未测。
- [冻结源码 indexer.py:255–298](../gap-dataflow-171/source/indexer.py)；SHA256 `d5d6a438bbced4e63d729dac94c08869df30dab4a71ceacecdea211386b3db33`。
- [冻结源码 dsa_v41.py:613–647](../framework-timing-171/source/dsa_v41.py)；SHA256 `3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd`。
- [冻结源码 dsa_v41.py:665–674](../framework-timing-171/source/dsa_v41.py)；SHA256 `3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd`。

同窗设备类型 `VllmAscendQuantLightningIndexerV2`：27.258547ms/卡/11周期（不属于该字段的 Host 时间）。导出中的完整 shape 样例。cache 张量形状是池容量/视图，不等于每次实际读取字节数；有效请求、长度、mask、索引和 PAD 决定访问。例中 padded 槽位数不等于实际写入 token 数。

原样例 EP0：输入 `"3,32,128;17305,128,1,128;3,32;3,32;17305,128,1;5;;;4;;4,272;;1024;3,1,2048"`；dtype `INT8;INT8;FLOAT16;FLOAT16;FLOAT16;INT32;DT_UNDEFINED;DT_UNDEFINED;INT32;DT_UNDEFINED;INT32;DT_UNDEFINED;INT32;INT32`；输出 `"3,1,512;;"`。全部去重样例保存在 JSON。

### D26 · 保护跨轮输入与草稿 ID 拷贝（S1）

字段：`prepare_inputs_event / draft_token_ids_cpu / copy event`。共享 CPU/pinned 缓冲、设备输入和草稿 ID 镜像；事件是寿命保护而非有效数据。

**为什么需要**：避免下一轮 Host 写入覆盖上轮尚未完成的传输；只为实际 Host 读者拷草稿。

**读取 → 生成**：上轮异步 H2D 的未结束读者、草稿输出、当前 async/structured/PP 条件。 → input prep 进入时等旧事件，退出时 record；draft ID D2H 在部分 async 条件直接 return。

**实际消费者**：下轮 input prep、需要 CPU draft ID 的结构化输出/PP 等分支。

**依赖与提前边界**：必须核对事件实际覆盖了哪些流/读者；不能把一个 release 当作全部异步读者退役。

**优化判断**：已有异步免拷分支要先复用；双缓冲必须同时保护 Host 源和所有 device consumer。

**证据与计时**：源码条件分支；未单测这些事件各自等待时间或逐次草稿拷贝人口。字段独立时间：未测。
- [冻结源码 gpu_model_runner.py:3841–3854](../framework-timing-171/source/gpu_model_runner.py)；SHA256 `87c29d08c0bbf66993d8b984811e7325e35e6436242a773feb55ec88f16b2c51`。
- [冻结源码 model_runner_v1.py:2163–2205](../idle-attribution-171/source/model_runner_v1.py)；SHA256 `568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c`。

### D27 · 其它可能分支：搬页、恢复、清零、多模态（S6）

字段：`kv_transfer / sparse offload / blocks_to_copy/zero / M-RoPE / GDN`。可能涉及真实 KV 块搬运、状态清零或其它模型专用输入。

**为什么需要**：支持卸载/恢复、缓存整理和其它模型结构，不是所有 prepare input 的固定步骤。

**读取 → 生成**：特性开关、调度器搬块请求和模型类型。 → 只有对应 guard 命中时执行；本轮未逐分支记录人口。

**实际消费者**：相应 cache/模型分支。

**依赖与提前边界**：按本轮实际 flag/动态请求逐项核准，不能因为源码里存在就计入已见空档。

**优化判断**：需分支命中证据后再定价；generic DCP manager 不等于此处 TP query CP。

**证据与计时**：只列覆盖边界；未证明这些分支在该诊断窗执行，不认领其成本。字段独立时间：未测。
- [冻结源码 model_runner_v1.py:961–997](../idle-attribution-171/source/model_runner_v1.py)；SHA256 `568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c`。
- [冻结源码 model_runner_v1.py:1394–1434](../idle-attribution-171/source/model_runner_v1.py)；SHA256 `568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c`。
- [冻结源码 model_runner_v1.py:3470–3525](../idle-attribution-171/source/model_runner_v1.py)；SHA256 `568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c`。

### D28 · 实际压缩并更新 ring 状态（S8）

字段：`projected kv / scores / state_cache / ring_pooled`。设备 projected 数据、分数、持久 ring 状态与 pooled 输出；与 ring_meta 描述表不同。

**为什么需要**：将新 token 对应投影与历史状态组合为压缩表示；未来索引和注意力读取其缓存结果。

**读取 → 生成**：已计算的 kv/scores、state_cache、c2_ring_metadata。 → compressor_from_projected 读取/更新 ring 状态并生成 pooled，再执行 norm。

**实际消费者**：后续压缩 cache 写入和相应 index/attention 路径。

**依赖与提前边界**：kv/scores、ring metadata、旧状态以及合法更新顺序均就绪。

**优化判断**：计算/状态融合可研究，但不能把 metadata 已就绪误认为压缩数据已就绪；实际工作不能只靠移动 Host 组表隐藏。

**证据与计时**：当前源码消费者与历史同 SHA 审阅匹配；没有在此表给 compressor 绑定独立设备时长。字段独立时间：未测。
- [冻结源码 compressor.py:49–71](../gap-dataflow-171/source/compressor.py)；SHA256 `31516e5ba234e34fa01e73e864fbc88bd6473d4a5274a1d43da8a79bab3cf494`。
- [冻结源码 dsa_v41.py:1304–1357](../framework-timing-171/source/dsa_v41.py)；SHA256 `3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd`。

## 用一个例子理解这些参数

仅演示字段含义，不是从本次 trace 提取的实际数值；忽略投机纠偏和分组压缩。

两条请求本轮处理 [2, 3] 个 token，已有进度 [100, 200]：

- `req_indices = [0, 0, 1, 1, 1]`：每个扁平 token 属于哪条请求。
- `query_start_loc = [0, 2, 5]`：请求0占 [0,2)，请求1占 [2,5)。
- `query_pos = [0, 1, 0, 1, 2]`：各请求内部的本轮偏移。
- `positions = [100, 101, 200, 201, 202]`：各 token 的绝对序列位置。

普通非环形、未压缩的单组示例：块大小128，位置130→逻辑块1、块内offset2；若 block_table[request,1]=11，则 flat slot=11×128+2=1410。当前各物理组、环形/压缩缓存按其真实规则处理。

注意力随后用 QSL 分请求，用有效长度限制范围，用 block table 寻址实际 KV；cache writer 用 slot mapping 写入新值。参数已经就绪不代表 Q/KV 等真实计算输入也已就绪。

## 如何据此判断能被计算隐藏多少

先将每项拆成稳定结构、反馈数值与资源发布三类依赖，再找各自真实生产时刻。删除已经没有读者的值；对仍必要的工作，只有在输入、CPU执行资源和缓冲都可用后才能启动，并须在实际消费者需要它之前完成。

对单个既定资源顺序的任务：`开始=max(数据就绪, 缓冲可写, 执行资源可用)`；`完成=开始+耗时`；`暴露等待=max(0, 完成−消费者其余依赖就绪时刻)`。消费者需求时间不能直接使用已经被这项准备拖迟后的观察起点。多个任务重新排依赖，比较最终交付点，不能逐项相加覆盖收益。

本轮已补齐“工作是什么、读谁、交给谁”，没有足够证据给每个字段编造可提前比例。尤其 accepted count、设备权威长度、当前 query 的 top-k 都有真实反馈依赖。

## 保留的缺口

- 没有逐字段取值/字节数/H2D 完成时间，不能把函数总价拆给每个字段。
- 没有每次事件的原因与 off-CPU 细分，不能把函数自身未覆盖 CPU-op 的部分全叫 Python 或锁等待。
- 设备 KV 真实读流量和带宽未单独测出；SMLA/QLI 时间包含计算。
- 本表是已定位准备链的源码语义清单，不声称全部框架分支或请求完整因果 DAG 已闭合。

机器入口：[view-data.json](view-data.json)（含逐项源码片段与 SHA） · [operations.csv](operations.csv) · [旧细函数诊断](../framework-timing-171/REPORT.md) · [死投影读者证据](../execution-search/共同设备长度投影的死工作与真实消费者判断.md)。
