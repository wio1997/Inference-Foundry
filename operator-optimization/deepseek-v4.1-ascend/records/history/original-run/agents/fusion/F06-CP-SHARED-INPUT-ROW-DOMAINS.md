# F06：Q_a / Wkv 共同输入与 CP 行域

## 范围与此前审查

`QUANT-CONSUMER-REVIEW.md` 已指出本地描述中 40 层的 Wq_a/Wkv 都标记 FLOAT，不能把源码的输入复用称为已经省掉动态量化。这里新增分析实际 **legacy DSA-CP 的 metadata 对象身份与局部/全局行域**，不是重新登记已有的“拼接权重”候选。`F07` 已查 Q_b/RoPE 和 KV 多流，本记录只考虑更早的 Wq_a / Wkv 共同输入。

## 源码确切分支

- 普通 `_prepare_inputs_and_caches` 以相同 metadata 对象传给 `_preprocess`；进入多流时 `share_inputs=query_metadata is cache_metadata` 为真，后续 `share_quant` 还需要两 wrapper `_quant_method` 外层类型相同、通信检测标记相同（`attention/dsa_v41.py:347–369,440–465`）。这个判断是 Python 对象身份，不等于逐 tensor 数据值的运行验证。是否真的复用一次量化还取决于 wrapper 方法/配置：`CVLinearWrapper.quantize` 对非动态或含通信路径直接返回输入与空 scale。现有两份量化描述一致性只证明文件标签，不证明历史实例执行的 quantization method。
- legacy CP 的 `forward` 先由 `_get_layer_metadata` 为本层新建 `DeepseekV41LayerMetadata`；`AscendDSAV41CPImpl._prepare_inputs_and_caches` 又通过 `_global_layer_metadata` 调用同一个构造函数新建全局 wrapper，将 `hidden_states[start:]` 和 `hidden_states` 分别传给 `_preprocess`（`dsa_v41.py:274–288,690–703`；`context_parallel/dsa_v41_cp.py:173–192`）。因此**如果正常运行该冻结 CP 实现、未替换方法且该调用进入多流**，局部与全局 wrapper 是两个新建对象，`query_metadata is cache_metadata` 为 false，已有 `share_quant` 不命中；这即使底层某些 metadata 或 hidden view 恰好别名也成立。配置请求 CP 并不单独证明某个历史任务的真实 backend 实例、是否多流或该执行分支。
- false 分支仍安排 aux 等待 `q_quant_done` 后做 KV quantize 并记录 `kv_quant_done`；主流推进 Q_a，同时以 `part2_start` 协调 aux 的 Wkv 与 Q norm/量化，后续 event 控制 Q_b 与 KV norm/RoPE/cache 发布。量化 wrapper 若只是原样返回，Python 仍记录相关 events；不能将 event 消失、设备 task 或 cost 自动推定为某一个值。直接拼成一个大 GEMM 会改变这条已有 Cube/Vector 多流交错，不能只比较 GEMM 数量。

`share_quant` 对不同对象只是按外层类型和 `_has_communication` 相等判断；这两项不直接证明两个方法的实际 scale 算法、inner wrapper、group size 等配置相同。冻结 wrapper 的 MX quant 显式使用方法实例的 `dynamic_mx_quant_scale_alg`。没有实际实例与量化参数对照，不能把该判断升级为所有类型、所有配置下的量化数值等价证明，也不能据此宣称本基线发生错误。

## 已有样本与不同重组

`f07-q-kv-streams.json` 的 EP8 第24层 run2 同窗口，主流 89 的 BF16 `MatMulV2` task1257 输入 `[3,5120];[1280,5120]` → `[3,1280]`；aux stream87 task275 输入 `[24,5120];[512,5120]` → `[24,512]`。它们与该源码的 local Q_a、global Wkv 的 3 行与 24 行物理 shape 相容；但没有 tensor 地址/对象、实际有效行数或 CP 实例绑定，不能仅凭维度确认这两个任务为对应调用。这是单个已有历史窗口，不是全负载数量。

**条件示例**：若这两个记录确实分别是同一 CP 调用、query 的 3 行均为 global 24 行内的同一输入子区间、精度/布局允许共同权重读取：

1. 为 3 个交叠行做 `[Wq_a;Wkv]` 合并投影，其他 21 行只做 Wkv。按每输出元素各自执行相同长度 5120 的点积计算，输出元素总数是 `3×(1280+512)+21×512=16128`，恰与两次分开做的 `3×1280+24×512=16128` 相同。仍需保持各投影原输出 dtype、舍入、bias、量化方法及权重布局；多输出分配、临时内存、实际 kernel 与调度开销未知。KV 全部 24 行依然要用于 cache，Q 分支只属于这 3 行；不能把剩余的 21 行删掉。
2. 如果粗暴把合并投影对 24 行全算，则输出元素变为 `24×(1280+512)=43008`，是上例的 `8/3`；多算的 Q 元素 `21×1280=26880`、若每元素是长度 5120 点积则额外 `137625600` 次标量乘加。其余真实开销包括激活布局、权重、padding/通信和实现不同，不是性能预测。不是说一个大 GEMM 必然更慢，而是不能把额外工作忽略。
3. 若只分析 BF16 交叠输入的潜在共同读取，3 行 × 5120 × 2 字节是 `30720` 逻辑字节；只有在当前两 GEMM 确实分别从同一存储层读取、融合后一份可被共同使用等额外条件下才可能减少对应重复访问。该数不是 HBM 强制流量、必省字节或时间收益。

## 必须保持和未证明

空 query 分支仍调用 `_update_caches` 处理非空 global KV，故不可要求每个 KV 行都有 Q；检查相应截断与 padding。有效 query 行必须映射 global 正确 offset；同时保存 SWA/long-KV 写、RoPE 的不同 local/global 位置与对应 metadata、Q norm 的 BF16 和后续 Wq_b/indexer 多消费者。对重复行/别名、quant 算法及校准、不同 batch 下的输入行不相等、图 buffer、event、算子通信和数值舍入必须分别检查。若线性层只允许完整矩阵输出，按行重组需额外接口，硬件实际逐块 ready 和资源竞争均未知。两 GEMM 合并可重排浮点累加与 kernel 实现，不能只依据实数代数说逐位相等。

来源：`agents/fusion/f07-q-kv-streams.json`、`QUANT-CONSUMER-REVIEW.md`、已冻结 `attention/dsa_v41.py`、`attention/context_parallel/dsa_v41_cp.py`、`agents/fusion/frozen-source/vllm_ascend/ops/cv_linear.py`；前两处源码 SHA 见 `frozen-layer-coverage.json` / `evidence/actual-sku-and-cp.json`。仅本地读证据，无 NPU、无目标代码变更；是否值得进入开发与实际收益仍 UNKNOWN。

## 进一步收紧：非空 CP 分支的行别名可由同源源码推导

这是对上面第 1 项交叠行候选的**条件强化，不是新增收益项**。冻结 `context_parallel/dsa_v41_cp.py:112–153,173–180` 的 legacy CP builder 对同一个 `common` 先建立 global metadata，设全局实际行数 `N=common.num_actual_tokens`，取非负 `start≤end`，并令 `actual_end=min(end,N)`、`actual_start=min(start,actual_end)`，local 行数 `L=actual_end-actual_start`；冻结 `dsa_v41.py:941–949,1390–1401` 中 global `swa.num_actual_tokens` 取自同一 `common.num_actual_tokens`。CP 的调用把 `hidden_states[start:]` 与 `hidden_states` 分别交给 `_preprocess`，而后者 `:353–355` 各自按 local / global `swa.num_actual_tokens` 截断。

如果这些对象确由这套对应 builder 建立、原有方法/metadata 未被替换，`hidden_states` 至少有 N 行，且 **L>0**，那么必有 `actual_start=start`，两个输入分别是相同 base tensor 的 `hidden_states[start:actual_end]` 与 `hidden_states[:N]`。因此 local Q 行恰为 global KV 输入的连续子视图，索引映射 `q[i] ↔ kv[start+i]`（`0≤i<L`），既非按值猜测，也不需要假定两个 metadata 对象是同一对象。`L=0` 情况不能套用：此前 source 的 `_update_caches` 分支仍要处理非空 global KV。N、start、L 具体取值、现存 EP8 3/24 的任务归属和该历史调用是否满足上述前提仍未证。

这一事实允许讨论**不同于完整权重拼接的条件数据重组**：若两侧确实使用语义一致、每 token 独立的动态量化，且 `q` / `kv` 覆盖的同一 base 行在量化前未被写改，可考虑只量化全局输入一次，将量化 tensor 及其每行 scale 的 `[start:actual_end]` 视图交给 Q、全量交给 KV。若原量化对于行批次大小、padding、layout、特殊值或设备选择有不同结果，则不等价；还须证明相同数值配置、量化 kernel 实际归约域与输出形状、storage stride、图稳定地址、生产者事件和两个 GEMM 对 buffer 的最后使用。量化一次全局 N 行可能改变原有 Q_a 与 KV quant 的启动顺序及 Cube/Vector 重叠；物理效率和性能未知。**冻结量化描述将 Wq_a 与 Wkv 标为 FLOAT**，故没有证据表明本次基线有这段可消除的重复动态量化，更不能用这个条件例子计算实际省时；它也不能用于不同 dtype、MXFP8 scale 参数不一致或有通信等情况。F06 的行共享/投影重组本身适用性仍取决于实际执行绑定和数值/流依赖。

只读依据 SHA-256：`dsa_v41.py` `3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd`；`context_parallel/dsa_v41_cp.py` `039ec9d18e0a8a4aa39f9d5b4bea15204cea3f397a3fce97a5097b266bc9822e`；`frozen-source/vllm_ascend/ops/cv_linear.py` `d2bf17c189f8748a9b7e5cbd42325c798eeb238da894755065792da982450b6d`。没有运行设备或修改目标实现。

## CP 局部／全局 RoPE 的共享 buffer：重排时的覆盖反例

这是 F06 的**正确性约束**，不是新增算子或性能收益。`context_parallel/dsa_v41_cp.py:112–119,156–165` 先构造 global metadata；若其中 cos/sin 存在，local builder 使用 `global_metadata.cos[actual_start:actual_end]` 与对应 sin 切片，并注明第二次 cached RoPE gather 会覆盖仍被 global KV 引用的 process-wide buffer。`dsa_v41.py:180–181,353–379,1051–1059` 表明，layer 分别通过 query/global metadata 读取 RoPE；builder 在没有现成 `rope_views`/batch 共享 rope 时，可调用 `get_cos_and_sin_dsa`，其缓存模式受 prefill/graph 条件控制。`ops/rope_dsv4.py:107–188` 的实现按 config、group 取 `_ROPE_STATE.runtime_buffer`（或 draft buffer），在 `use_cache=True` 分支将 gather 结果写入 `buf_cos[:num_tokens]` / `buf_sin[:num_tokens]`，并返回这些 buffer 的视图；非缓存分支则通过 indexing 生成结果，不套用以下覆盖结论。字符串索引通过 proxy 选择层对应 config/group，故这里必须明确**两次引用同一 config、group、buffer 和相同 draft 域**。

具体反例：假设一个合法的三行 global 位置为 `[10,11,12]`，三 TP rank 每 rank 一行，此 rank 为 2，local 范围 `[2:3]`，该 group 的位置 10 和 12 的至少一个 RoPE 分量不同；先在 cached 模式生成 global 结果，其行 0 对应位置 10。若重组时将目前的 local slice 改成对位置 `[12]` 再做同一缓存 gather，则会写该共享 buffer 的 `[:1]`，覆盖 global 的行 0；之后再按原 global metadata 将第 0 行用于 KV RoPE，则该位置拿到的是位置 12 的分量。这个例子是从源码写入／视图规则推导的条件构造，**不是历史故障、真实位置或实测数值**；实际并发还要考虑两次写入及消费者跨流的 happens-before，不能把 enqueue 顺序当作设备完成。

因此，任何把 Q/KV 投影、Q RoPE、KV RoPE、compression 或 metadata gather 拆开并重排的设计，都要为相关 buffer 建立最后读取和下一次写入的依赖：可研究保留 global→local 的 slice，或为 local 配置独立且寿命足够的存储，或在真实完成读取后复用；跨流用能证明可见性的完成事件，graph 重放要求固定地址且不能被下一次或其他调用提前改写。是否有公开、可用的逐块事件接口、持有额外 buffer 的开销，以及历史实际分支／config、位置、alias 与时间都未知。不能因当前源码明确规避此反例，就声称当前运行已发生覆盖或现有瓶颈由此造成。

只读来源：`context_parallel/dsa_v41_cp.py` SHA-256 `039ec9d18e0a8a4aa39f9d5b4bea15204cea3f397a3fce97a5097b266bc9822e`，`attention/dsa_v41.py` SHA-256 `3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd`，`ops/rope_dsv4.py` SHA-256 `9cbcd2cb2a1123dcbf5afc3999025035d155dfd3b160f186c36aebb5aa7c706b`。仅阅读已冻结文件，未调用 NPU 或变更目标实现。
