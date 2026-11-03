# Compressor / Engram 融合边界复核

本次只审计冻结源码与配置，未运行 NPU、未改推理代码。输入文件和 SHA 在 `compressor-engram-input-hashes.json`，先前模板是结构图，不能替代下面的具体分支。选择 F18、F25 作为进一步研究对象是因为数据边界和必要条件清楚，**不是性能优先级或已证明可实现的判断**。

## 1. F18：ratio-2 pooling → RMSNorm，明确是当前调用的两个边界

`dsa_v41.py:523-532` ratio 2 路径：`hidden_states.float()` 送入 FP32 的 wkv 和 wgate，两组 FP32 输出进入 `compressor.pool_projected`。`compressor.py:55-67` 将它们及 FP32 ring cache、设备 metadata、BF16 预留输出传给 `compressor_from_projected`，然后对返回 pooled 执行独立 `self.norm(pooled)`。

`compressor_triton.py:675-720` 中 `compressor_from_projected` 先 zero 输出，再启动 `_pool_kernel` 与 `_cache_update_kernel`；该调用明确 `TOKEN_ALIGNED=True`。`_pool_kernel:310-320` 的 token-aligned 分支只写 BF16 pooled，**不在该分支做 RMSNorm**；文件里另一个 `TOKEN_ALIGNED=False` 分支包含归一化，不可误认为这次调用也已做了。pool 里已有按位置/score 的 exp、softmax 和加权归约组合；pool 和 ring 更新在当前路径仍是不同的 kernel，不能称它们已被合成一个 kernel。上层独立 RMSNorm 的具体底层 lowering 和运行时是否被其他机制消除仍未知。

条件候选是将 token-aligned pooled 的归一化与 pooling 重组，在同一合适粒度输出规范化结果；此候选不是移除 ring 更新。必须逐项保留：

- 先将 FP32 pooled 按当前路径舍入为 BF16，再按原 `self.norm` 的实际精度、epsilon、权重和归约顺序计算；非 token-aligned 分支使用的 `1/sqrt` 并不自动证明和 `self.norm` 数值一致。
- 所有 zero 初始化、无有效组、空 tokens、无效 cache block、padding 和各 token 行输出语义；不能只给完成的 pair 写结果后忽略其余行。
- pool 读取前一个时段的 ring residual，随后 cache_update 写入当期 kv/score；源码明确要求**先读再写**。处理分块、多请求、block ID、竞争写入时必须证明无跨块数据竞争，不能简单将两个 kernel 合并并去掉顺序。
- normalized latent 先被 `attn.indexer.update_keys` 读取，继而 `latent.view` 原地 RoPE 并写 long KV（`dsa_v41.py:544-564`）。必须保留未做 long-KV RoPE 的 indexer 消费版本、持久索引及 scale、所有后续 layer 对 long KV 的读取，不能以“唯一消费者”删除分流。
- `wait_for_device_metadata` 和既有 index-query 辅助流量化与 WKV matmul 重叠，以及本设备上可能的 Cube/Vector/访存争用；源码中的调度不是收益证明。

模型配置里 target 层 2、8、14 是 ratio-2 KV source；目标执行涉及真实 shape、请求元数据、输入是否在缓存与实际数据流尚未逐次证实。ratio-1 的 `compressor.forward` 是 BF16 投影加 norm，并无同一个 C2 ring 分支，不能套用本候选。

## 2. F25：Engram 投影后的 gate / residual

配置的 Engram 层为 1、14。`engram/layer.py:34-55` 中 `wkv(rows)` 给出 kv，分成 key 和 value；`common.py:61-72` 的普通路径进行必要的 FP32 变换、两个均方根归一化、逐通道权重乘积及 dot reduction、符号开方、sigmoid、token mask，最后与 hidden residual 组合并转回原 dtype。普通路径在 rotation 非空时先在 FP32 还原 hidden 基；改变 rotation、epsilon、weight 与各阶段舍入或减小的中间精度都需另证，不能把 key/value 当作单一结果丢掉。

已有 `fused_engram_gate` 仅在 rotation None、NPU、profile 具备 `ENGRAM_UNROTATED_GATE` 时由源码选择。此前 stage 093 的 `_build_info` 为 A3；已快照 `hardware_profile.py:249-281,298-308,361` 中 A3 capabilities 不含该能力（A5 列表才包含）。因此**正常加载该 A3 profile 且未被替换时，不能把这项已有 fused gate 当作 A3 路径已经命中**。Engram 是否真实启用、rotation、compiled graph 内部进一步融合及各次 tensor 消费仍需运行证据，源码 API 分开并不意味着未被 compiler 融合。

条件候选是保留 `wkv` 输出的 key/value 两路以及全部数学步骤，将 gate 的部分规约、pointwise 与最终 residual 重组成少数设备阶段，或研究 projection epilogue 接入。不能把“q_weight × k_weight 提前计算”视作默认合法：必须证明权重在执行期间不可变、浮点乘法结果与原路径逐元素一致及加载/更新生命周期；涉及乘法重排须证明既定精度语义。还必须保留 mask ready 和 per-layer rows ready：`model.py:1565-1592` 在实际消费前等待，满足 SP 时先 wait 才 shard；`prepare_engram` 的 event 在相关 lookup、DP/TP 交换及 padding 之后发布。不得把等待删掉仅因希望异步。

## F17、F19、F24 的边界和纠正

- F17 可以在 ratio-2 条件下评估共享输入 cast 的两组 FP32 投影及 tile 化送入 pooling。两输出、分片、权重和原矩阵乘法数值要保留；准备 indexer query 的辅助流量化与其中投影当前存在交叠，改变竞争与关键路径的结果未知。
- F19 的 latent 同时有 index-key 分支和 long-KV 分支；`update_keys` 在 latent 被 long-KV RoPE 原地修改前使用它。cache 写、slot、scale 及跨层复用是状态输出，不能把其中任一分支视为无用中间结果。
- F24 旧 `candidates/F24.json` 将 `hash` 和 `lookup` 均笼统标成 CPU，不能拿这个资源标签当本基线结论。`engram/hash_state.py:102-114,132-145` 明确给出 Triton request-index 与 hash kernel；`engram/embedding.py:394-401` 在 `_codes_uva` 存在时调用 `gather_dequantize_host_uva`，后者 `engram/npu.py:196-247` 明确是通过已注册主机地址的 NPU Triton gather/dequant。那条路径中**表驻留主机**不代表查询计算必然由 CPU 执行。CPU fallback、device table、DP/TP 交换及实际存储类型均须逐分支判定。gather + dequant 本身已在对应 Triton 调用组合，不能重复算新融合；全局 hash 已覆盖多层，不能把每层 hash 看成独立重复计算。

所有候选的历史执行次数、实际设备流量、数据别名、计算/通信重叠、可融合接口、数值等价与性能收益均未直接证实；先前模型图与源码的覆盖不等于穷尽所有可能组合。

## F24 补充：request 搜索与 cache 写入的并行边界

冻结 `engram/hash_state.py:73–170` 的有 slot-cache 条件分支顺序是 request 搜索、写 hash cache、计算所有层的 hash。搜索读 `query_start_loc` 并写 `req_ids`；cache 写入接收 `input_ids`、`token_map`、`dead_mask`、`slot_mapping` 和 cache，接口没有使用 `req_ids`。hash 再读取 `req_ids` 与 cache；注释明确要求 cache 写完成后，其他线程块才能读取 fallback 历史。`num_tokens == 0` 直接返回；不开 slot cache 则没有该写入环节。

**可研究的结构**：在证明底层访问集合及别名无冲突之后，使 request 搜索和 cache 写入分流并行、在 hash 前等待两者，或研究把两个互不依赖的 producer 组成一次分工执行后再整体发布的设备阶段；hash 不能在 cache 尚未可见时开始。它不等同于直接把 request 搜索、cache 写入和 hash 三者合成一个不带全局同步的 kernel：不同线程块读写 fallback 历史的顺序必须保留。cache 的状态、重复 slot、buffer 生命周期、跨流事件和任意后续消费者均需证明；两个 producer 可能争抢 AIV/访存且事件开销可能大于节省。接口关系支持依赖假说，不代替未快照内部设备实现、实际别名和 compiler 验证。源码另明确说当时 Triton-Ascend 不能编译合并 request-search/hash 的 kernel，故不可把两者直接融合当成已具备的接口或既有能力；这一源码注释也不是对后续任何实现均不可能的证明。

现有 `agents/diagnosis/engram-producer-events-exact.json` 在 EP8 两段图开始前的固定 10ms 回看窗口里各有一组同 stream 47、相继的 `_engram_req_index_kernel`、`_write_hash_cache_kernel` 和 `_hash_ids_kernel` 名称。第 2 组任务时长分别 **6.160、12.699、127.698 µs**，第 3 组分别 **6.140、12.420、126.938 µs**；第 1 个窗口未检出这些精确名称。名称和相邻顺序与上述源码相容，但没有逐调用 tensor、请求、历史二进制和完整依赖绑定；不证明这些任务就是同次调用、处于端到端关键路径或可实现任何时间节省。三条 task 区间之间仍有较长间隔，不能把间隔解释成纯调度或空闲。该边界仅在观察到相容任务的意义上值得进一步查，不声明它是影响最大的融合点，也不重复计算已组合的 UVA gather/dequant。

可复核输入：`hash_state.py` SHA256 `62dd8271db66f4a735de7e9f764b3a1a4b7e66b56c352352f27093f3ef40eb38`；`engram-producer-events-exact.json` SHA256 `ea31f7d421d10109a6cab2a14d13c084dda169761b4085177a658741821964cb`，其内部含 DB 和边界文件的 SHA；只比较单卡同源时间。剩余缺口是 producer 的完整设备访问及别名、实际 slot-cache 分支及身份、流事件、关键路径和各资源竞争；当前材料不支持量化端到端收益，也未穷尽其他模块的多算子融合。

## Hash 后选择本地行：可研究融合与历史状态边界

冻结 `engram/model_state.py:99–110` 在 `pcp_context` 非空且非 dummy 时，将输入 batch 换成 `global_batch`，并传入 `local_token_indices`；代码注释明确要求先对连续请求 hash，再选择本地行。`model.py:1163–1193` 先由全量 `input_ids` 及可变 `valid_token_count` 构造 `dead`；keep mask 可按 indices 选择并提前发布，然后 `hash_state` 接收全量 IDs、positions、query starts、dead、lookback 和可选 slot/block 映射，最后才对 hashes 的第 0 维执行 `index_select`。`hash_state.py:90–99` 给出完整输出的逻辑形状 `[T,L,(max_ngram-1)*num_heads]`、dtype INT32；若选择向量有 S 个元素，`index_select` 得到的逻辑形状是 `[S,L,(max_ngram-1)*num_heads]`，其行次序和重复均由选择向量决定；之后 `model.py:1203–1217` 再进入 DP gather、各层 lookup 及各自的 ready 发布。

**候选结构**：可研究保留完整输入和请求、lookback、dead 等语义，同时令 hash 的输出按选择次序直接写入 `[S,L,C]`，避免完整 `[T,L,C]` 输出与后续选择的逻辑中间值；也可研究将选择与后续 gather 组合。这个候选依赖能够表达有效索引、原输出顺序和重复行、跨 rank collective 及各层 table 的真实接口；`token_indices=None`、空输入、dummy 路径、capture padding、图地址与早发布 mask 的行为也需保持。源码中已有的“所有 layer 一次 hash launch”和已处理的 `gathered[:,slot].contiguous()` 是另一阶段，不能把中间值节省重复记作多个独立收益。

**不能简单将 token 筛选推到 hash 输入之前。** `hash_state.py:90–94` 的接口说明：历史依次来自当前 chunk、runner lookback，再按条件来自 slot cache；传入 hash kernel 的还包括请求边界、positions、dead、block table 和 stride。比如一个请求的当前 chunk 为 `[A,B,C]`，本地只选择 C 时，仅把 C 作为新 chunk 输入会丢失原 chunk 中的 A、B；除非重建等价的历史和请求映射，否则就不再满足原接口的历史条件。这是依赖反例，不声称 A、B 的变化必然让本模型某个具体 hash 输出不同，也不假定本次历史请求包含这样的选择。

有 slot cache 时，`hash_state.py:115–131` 写入以完整 `num_tokens` 启动，并按注释要求写入完成后其他 block 才可读取 fallback 历史。就算某些本轮 hash 行最终没有被选择，也不能直接省掉对应 cache 写入而不证明这些 slot 对本轮读取或以后调用不可见；而把 cache 写入与 hash 放进无全局同步的一次 kernel 也不能自动保持这项约束。相反，在不使用 slot cache 的分支，该副作用约束不适用，但仍有当前 chunk、请求边界和 lookback 的依赖。mask 在 hash 前单独 record，联合设备阶段如只能整体完成后才发布 mask，可能减少原来与后续工作的重叠；这不是无需测量即可认定的延迟收益或损失。

源码哈希：`model_state.py` `677cb13bb86ad80a154e0ff9a43620aa3344e7e93170a05e27aba0cc1f10aa8b`，`model.py` `101f96b38e9fc47e7be147ddfe56976aee896867c73a33a7d95c3e3c833a6e91`，`hash_state.py` `62dd8271db66f4a735de7e9f764b3a1a4b7e66b56c352352f27093f3ef40eb38`。该路径是冻结源码的条件分析；实际历史 PCP、索引内容、hash binary、buffer 身份和选择耗时均未绑定，不作为历史瓶颈、物理流量或性能界。本次只读已有源码，没有运行 NPU 或修改推理实现。

## Engram 请求定位与 slot-cache 写：双生产者 join 的条件

此前本文件已分析不能把 token 筛选无条件移到 hash 前，也指出 slot cache 写后 hash 可能需要读取；这里细化 hash 前**两条不同生产者**的条件性数据关系，不将其与已有 F24 查表/指针表重组或 mask ready 的收益重复计算。冻结 `sources/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/engram/hash_state.py` SHA-256 `62dd8271db66f4a735de7e9f764b3a1a4b7e66b352352f27093f3ef40eb38` 的 `:95–145`：当 `T=num_tokens>0`，先新建 INT32 `[T]` 的 `req_ids`，以 `query_start_loc` 调用 `_engram_req_index_kernel`；只有 `use_slot_cache` 为真时，再向 `_write_hash_cache_kernel` 传 input IDs、token map、dead mask、slot mapping 与 cache；随后 `_hash_ids_kernel` 消费 req_ids、当前输入、positions、block table、cache、lookback 等。输出为 INT32 `[T,num_layers,(max_ngram−1)*num_heads]`；若冻结 config 和实际实例一致，是 `[T,2,24]`。源码的注释表示 request search 从 hash body 拆开是 Ascend 编译器 workaround，不能假定当前 Triton 能直接把两者重新组合。

从**Python 可见的参数**看，request search 的新输出与可选 cache-write 的参数集没有显式数据边：若实际 kernel 的完整读写集合、输入准备和不同参数的 storage 均已证明互不冲突，可探索让两条生产者在不同流推进，在 hash 读取 `req_ids` 和可被本次 fallback 使用的 cache 前分别等待对应完成事件。没有 slot cache 时这一分支不存在；前两个 kernel 的完整实现、副作用、buffer alias、执行流及历史绑定当前不在该代码片段中，不能将其称为已证独立、已存在并发或必然有收益。跨流方案还须保持当前 chunk、lookback/dead mask、request 边界、block table 与 slot 映射、写后读可见性及未来轮 cache 生命周期；共享 AIV/访存可能竞争，且不能根据数值不依赖推断物理可并行。

一个只用于说明 join 的有限反例：假设待算 hash 需要 fallback 读取 slot 5，原 cache 在该槽的 token 为 9，而本轮 cache-write 应将 42 写进该槽。若只等 request search 完成，hash 在 cache-write 前读取，得到旧值 9；若在合法新写完成且对读取流可见后读取，得到 42。若当前 hash 对两值可区分，最终结果也可能不同。这个条件例子不证明历史 slot 重复、此读取落在原 kernel 路径或两值必生成不同 hash；也不排除经过证明的逐槽/逐行依赖替代整批屏障。

已有 `agents/diagnosis/engram-producer-events-exact.json` SHA-256 `ea31f7d421d10109a6cab2a14d13c084dda169761b4085177a658741821964cb` 在 EP8 重 profile 的图前 10 ms 指定回看中，run2 记录 stream47 的 `_engram_req_index_kernel` task53677（6.160 µs）、`_write_hash_cache_kernel` task53678（12.699 µs）、`_hash_ids_kernel` task53679（127.698 µs）；run3 对应 task55034/55035/55036（6.140/12.420/126.938 µs）。这些是同设备导出记录、相容于源码次序，不证明三个任务属于同一请求或所存源码，也没有连接 tensor、事件或 ready 信号。查询只筛指定名字；任务间隔可能含其他活动或未导出记录，不能当空闲、等待、可隐藏时间或潜在收益。未运行 NPU 或修改目标实现。
