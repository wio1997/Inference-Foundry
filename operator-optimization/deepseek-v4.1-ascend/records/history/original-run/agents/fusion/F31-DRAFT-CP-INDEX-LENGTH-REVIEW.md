# F31：DSpark CP 的全局索引、局部长度与 metadata 共享

这次只分析先前 F31 清单未展开的 **DSpark 的 CP metadata builder 全局→局部边界**；不是重复 per-group 输入展开或预测提速。启动脚本显式请求 `dspark`、`num_speculative_tokens=5`、`enable_dsa_cp=true`、draft `enforce_eager=true`。既有 `agents/diagnosis/DRAFT-EP7-QUERY-METADATA.md` 的一个严格 draft bracket 已导出 `SparseFlashMla`，其余严格样本汇总在 `evidence/draft-module-diagnosis.json`；这说明该次 profile 采到 draft attention 的条件样本，**不证明**其元数据对象是下文具体 V4.1 CP builder、非 causal 分支、底层 helper 或每个候选均真的执行。126/288 未归属样本也不能据此补成同分支。

## 索引与长度：一条尚未审查的具体重复计算候选

路径相对于冻结 `sources/vllm-workspace/vllm-ascend/vllm_ascend/`：

1. `spec_decode/dspark_proposer.py:313–347` 在准备 first-pass draft metadata 时，先依据 rejected 更新 device seq_lens，重建 query start/positions 和 group slot，`cad.causal` 若模型提供 `get_draft_attn_causal` 则取其结果，否则设为 false。不能仅从缺少某个本地定义推定真实模型没有继承/动态提供该方法；运行 causal 值仍 UNKNOWN。
2. 当采用 `attention/context_parallel/dsa_v41_cp.py:103–167` 的 CP builder 时，先用 `_global_builder` 构造 global metadata，再按 `actual_start:actual_end` 创建 local common。`dsa_v41.py:1071–1096` 的非 causal 且未传 indices 分支调用 `build_dspark_swa_indices(...)` 同时取得 global `ori_sparse_indices` 和 `ori_topk_length`，再将它们分别 copy 进持久 builder buffers，取 `[:num_actual_tokens]` 返回；block table、query starts、seq lens、sliding-window、block size、K、逻辑索引模式与实际 token 数都是输入。注释指出图捕获需要稳定地址，但草稿 eager 的实际 graph/buffer 别名仍须独立确认。
3. `dsa_v41_cp.py:155–166` 仅在 `global_metadata.ori_sparse_indices is not None` 时将其切片放入 `kwargs["ori_sparse_indices"]`，没有同时传 global `ori_topk_length`。local 的 `dsa_v41.py:1069–1070,1115–1116` 对传入 indices、非 causal 情形重新计算 `(ori_sparse_indices >= 0).sum(dim=-1, dtype=torch.int32)`。`dsa_v41.py:651–671` 将 local 的 indices、lengths 和 SMLA metadata 作为独立参数交 sparse attention；构造 SMLA metadata 的 `:1171–1207` 也消费 local lengths。两者不能只提供其中一个或改成其他 group 的数据。

**有条件重组**：如果 global helper 返回每行长度严格等于这行 `indices >= 0` 的 int32 计数，并且 global 缓冲中这些值在本 rank 范围对应一致、索引不会在 local 首个消费者前被原地修改，可研究将 global lengths 的同一 `actual_start:actual_end` 切片传给 local，省去重复 bool+reduce 的候选工作。local builder 目前只读取 `kwargs["ori_sparse_indices"]`，并在本地初始化长度为 None；要这样做需要实际支持 length 参数，不是当前已存在的复用。不可直接删现有 sum，直到核实 helper 的计数是否包含 padding/sentinel、重复或特殊编码、empty query、数据类型、每组独立 block table 和 DCP 配置；切片后的稳定地址、copy/alias、跨流 ready、下一次构建覆盖和 concurrent batch 必须保持。

该条件不依赖“多 group”，一组 CP builder 即可形成 global/local 两个层次；但具体运行是否走到它目前仍未知。stage133/134 已取得入口和 Triton helper 源码；两条实现与 local 重算的条件等价和不同形状见下节。尚未绑定运行对象与值域，也无法计算节省的 launch、字节或时间。

## 共享字典：区分 runner 的意图与 V4.1 builder 实际接口

`spec_decode/llm_base_proposer.py:2576–2631` 创建 `shared_dsa_draft_cache` 并在 group 循环中用 `**extra_attn_metadata_args` 调用 `builder.build_for_drafting(...)`。但本次冻结 `dsa_v41.py:839–842` 的 `build_for_drafting(common_attn_metadata, draft_index, **kwargs)` 返回 `self.build(0, common_attn_metadata)`，**没有转发 kwargs**。而 `dsa_v41.py:955–960` 只有收到名为 `common_v41_metadata` / `common_v41_batch_metadata` 的映射才跨调用共享，否则新建私有字典；`dsa_v41_cp.py:87–95` 也只在收到这些 kwargs 时给 global 子域独立子字典。

这说明不能仅据 proposer 的注释宣称其共享字典已经跨组传入 V4.1 builder；也不能由它反推一定存在重复：`shared_dsa_draft_cache` 使用的键域还需核查、是否选中这个 builder/存在几个 group/运行实例 alias 均未知。如果考虑跨组共享，必须证明**完整缓存键覆盖 batch、local/global 域、长度、位置、压缩比、block layout 和用于 SMLA 的具体参数**，并且将 per-group `block_table/slot_mapping` 保持独立；共享后持久 buffer 不能被另一 group 或下一步提前覆盖。现有 `batch_shared` / `shared` 在同一 build 内已经复用满足 key 的 buffer，不能把这些已有复用重记成新增机会。

## 下一步与不可推论

现有严格 draft 样本说明 attention 实际运行，但其 causal、builder 类、indices/lengths 的指针、group 数与完整消费者尚未由样本直接证实。下一步优先读取两容器固定 helper 源码确认 length 计算及索引契约，再查看已有 trace 能否识别同窗 builder 的实际调用；仍不匹配就保留 UNKNOWN。未修改推理代码、服务或设备。

## stage133：fallback 数值、形状及非泛化条件

ZCode stage133 exit 0、验证模型 `deepseek-flash`；`f31-dsa-v1-provenance.json` 的 SHA-256 为 `f824f32965b49731a4ddb7cf006efcdf0c023cbb8915dc91242d5e90b211101c`。两个容器中 `dsa_v1.py` 相同，快照 SHA `698765a3662332ad108fd9ef128eb87fd313544cd887f5154f924a5915ee123a` 已在本机重算。

`dsa_v1.py:466–480` 在 logical 索引、query_start_loc 在 NPU、num_decode_tokens 非 None、未提供 indices_output/buffer 且 Triton helper 可用时调用 `build_dspark_swa_indices_triton`。当前 V4.1 调用使用 logical 且传 num_actual_tokens（可为 0），但实际 tensor 设备与 helper 可用性尚未从运行对象绑定；这个条件成立时**不是 fallback**。

对 `:482–504` fallback，令请求 query 行数 `q`，序列长度 `S`，窗口 `W`，宽度 `D`；`start=max(S-q-W,0)`，`visible=S-start`。logical indices 每行按 `0<=j<D` 输出 `start+j`（当 j<visible），否则 -1；由 `start>=0`，local 的有效项计数等于 `min(max(visible,0),D)`，而 global 的返回值是 `visible` 本身。因而只有 `0<=visible<=D` 才能逐值相等。原始 global indices 形状 `[T,1,D]`、global lengths 形状 `[T]`；local sum 形状 `[T,1]`、dtype 明确 int32，global lengths 的 dtype 由输入 seq_lens 推导。故直接传 global 一维切片就算数值等价也不保持形状，必须先证明或调整输出 shape/dtype/布局与消费者接口。

可复核的非目标域反例：W=128、K=5，width=ceil((128+5)/128)×128=256；q=200、S=1000，`start=672`、`visible=328`，global length=328，而 local count=256。这个例子不声称实际 DSpark 使用 q=200。相反，冻结 proposer `dspark_proposer.py:39–43,317–318` 在 K=5 的条件下每请求 q 为 5 或 6；若每请求 `S>=q>=0`、该行真实有效且 width=256，则 `visible<=W+q<=134<256`，fallback 值域条件成立。是否实际满足这些输入和是否调用 fallback，均未观察；同时还是要处理 shape/dtype、padding、缓冲生命周期和观测消费者。

stage134 已按两容器 SHA 校验 `ops/triton/spec_decode/dspark_swa_indices.py`；Triton 与 fallback 的差异及条件等价见下节，仍不得升级为整个基线的无条件证明。

## stage134：Triton 输出契约与条件等价

ZCode stage134 exit 0、验证模型 `deepseek-flash`；`f31-swa-kernel-provenance.json` SHA `b3f27ad8ddc59d89e8b6e1a56a0abe60ea7e8bc7d9226fae720bd518252d0839`，两容器快照源码 SHA `335be10476943d4ad429f4d52b7a0e069f3b65511d28265b0e9d9f7d2e94111c`，本机重算一致。

`dspark_swa_indices.py:20–44` 对每个 token 依 request 的 query 末尾找到所属请求，计算 `start=max(seq_len−query_len−window,0)` 与 `visible=seq_len−start`，对小于 `visible` 的列写入 `start+col`，其余写 -1；同时将 `visible` 写到该 token 的 length。`:55–65` 返回 **indices `[T,1,D]` int32、lengths `[T,1]` int32**，零 token 直接返回空张量；这与 local 的 `(indices>=0).sum(dim=-1,dtype=int32)` 在形状和 dtype 上一致，区别于上述 fallback 的 `[T]` lengths。有效请求定位及无效 padding 行需要正确的 `query_start_loc` 契约；对 `valid_req=False` 的 masked loads，源码得 `seq_len=query_len=visible=0`，写 -1 与 length 0。

对一个已生成的 Triton 行，如果 `0<=visible<=D`，且各已写的非负逻辑位置能无溢出表示为 int32，则正数计数必为 `visible`；因此 global length 与 local 重算的对应行**数值、shape 和 dtype 一致**。给定目标配置 `W=128, K=5`、D=256；若每请求 q 为 5 或 6、有效 `S>=q` 且有效逻辑位置不超 int32，值域成立（上文推导）；无效 padding 行为 0。上述条件是对源码的数学推论，实际 helper 分支、有效长度、query 元数据和 int32 边界在目标样本未逐次捕获，因此不是已验证的运行等价。若 visible>D，global 返回 visible 而 local 最多计 D；或 int32 位置溢出导致符号改变，也不能直接替换。

即便上述数值契约成立，local metadata builder 仍需要显式接收 global lengths 的切片，且须审计两阶段 index/length buffers 的同步、可观察 alias 和下游持久地址；实际是否改变性能和是否在此负载执行仍 UNKNOWN。该 Triton kernel 已一次生成 indices 与 lengths，不应将这两项重新记为尚未融合；候选仅是跨 global→local 避免再次统计。

## 独立审查：数值等价不等于 buffer 行为等价

bounds_work 独立审查指出：local 的 `(ori_sparse_indices >= 0).sum(dim=-1, dtype=torch.int32)` 在 Python 语义上产生独立结果；如果改为引用 global 持久长度 buffer 的切片，**会改变对象和潜在 storage alias、写入及生命周期**。global builder 在之后的 `build` 中会再次向其长度 buffer 执行 `copy_`（`attention/dsa_v41.py:1086–1096`）；若先前 local 的 attention 或 metadata 构建仍在读取共享 view，下一次写入与先前读取可能冲突。这是必须排除的数据竞争场景，不声称实际运行已发生。

`wait_for_device_metadata(DeviceMetadataStage.ATTENTION, id(op_metadata))` 是 metadata 消费前的等待点；现有代码与证据并未证明它同时建立了“所有使用该 lengths view 的异步消费者都完成 → 下一次 global builder 允许覆盖该缓冲”的反向生命周期约束。不能把前向 ready wait 当作 buffer reuse 完成信号，也不能假设 eager 模式或同流执行覆盖了所有 stream、graph 与 concurrent batch 情况。

因此上文数学条件只证明**对应读取时刻的值、shape 与 dtype 可相等**；方案还需要完成一份所有读者清单与 happens-before / buffer-reuse 证明，或通过独立稳定缓冲、必要同步和其他等价机制保留原行为。任何新增复制或同步的性能成本也未知。实际 storage identity、图降解和执行器排程尚未绑定，此项仍是候选，不能据此宣布可实现提速。

## EP7 既有 trace 中的实际记录（stage136）

证据是 `f31-ep7-metadata-trace.json`（SHA256 `f4089a7e07ca0ce1332e5e97671357f96be218fbe8886078c6ce353c565389b0`），来自同一 rank 的历史 EP7 CSV（SHA256 `9aa04e4cbbe7edc2160befcec0b133fdaf3b3e2571a1d2f1d7c92c6998b9bdc1`）；查询在两个预选 bracket 起点之前回看 20 ms，加上 bracket 本身，仅搜索固定的五类算子名。窗口、样本不构成全负载代表。起点为之前选定的 HcPre 开始时间，以下为相对此起点的微秒，负数表示之前。

| 预选窗口 | 事件 | 起止相对时间（µs） | stream | 关键输入 → 输出，记录 dtype |
|---|---|---:|---:|---|
| strict recorded SMLA | `_build_dspark_swa_indices_kernel` | -2183.717 ～ -2179.097 | 47 | [4];[3] INT32 → [15,1,256];[15,1] INT32 |
| strict recorded SMLA | `GreaterEqual` | -2131.918 ～ -2129.558 | 47 | [1,1,256];scalar INT64 → [1,1,256] BOOL |
| strict recorded SMLA | `ReduceSum` | -2127.318 ～ -2123.418 | 47 | [1,1,256] INT32; axis [1] INT64 → [1,1] INT32 |
| strict recorded SMLA | `SparseFlashMlaMetadata` | -2121.258 ～ -1773.325 | 37 | 记录包含 [1,1] INT32 输入；不能从 CSV 确定该项对应哪个参数或张量 |
| 未分类窗口 | `_build_dspark_swa_indices_kernel` | -6880.163 ～ -6877.663 | 47 | [2];[1] INT32 → [5,1,256];[5,1] INT32 |

strict bracket 内此前记录的 SMLA Q 形状是 `[1,64,512]`，因此附近 `[1,1,256] → [1,1]` 的比较、归约形状与已分析的单本地 token、窗口宽 256 的长度重算相容；同流先后顺序也与源码的 global build → local compare/sum 顺序相容。它们是该算子种类实际出现在采样中的证据，**并不足以证明这些记录属于此 draft bracket 的同一次调用，或证明是该源码行及同一数据链**。特别是 builder 输出 INT32，而 `GreaterEqual` 的输入记录为 INT64；`GreaterEqual` 输出 BOOL 而 `ReduceSum` 输入记录为 INT32。运行时可能涉及转换或其他路径，但现有证据未确定转换或数据身份。附近的 metadata 在 stream 37，既不能从时间先后推断跨流同步，也不能把它的 `[1,1]` 输入认作该 sum 的输出。所有这些记录的 model ID 都是 `4294967295`，不是请求唯一标识。源码允许一条解释，但只有进一步取得运行时算子到调用栈、张量或事件依赖绑定后才能确认。

未分类窗口里 builder 形状为 `[5,1,256]`，但所查名称内没有匹配 `[T,1,256] → [T,1]` 的比较和归约；这不是 local sum 没运行的证明：该窗口未被确定为 draft、固定名字查询会遗漏融合或替代实现。整个 CSV 的 8 条 builder 记录也只表明有 8 条 trace 记录，不能据此推断请求数、全部分支或性能收益。原先数值等价及独立 sum 与复用持久 buffer 的异步生命周期限制仍然成立，不能因形状和邻近时间把候选记为已证实。


## F31 键名契约：转发 kwargs 仍不足以建立共享（2026-10-29）

对前文“键域还需核查”补充限定源码证据：保存的 `spec_decode/llm_base_proposer.py:2576–2587,2601–2631` 在 `use_compress` 为真时将 `shared_dsa_draft_cache` 设为含 `common_ratio_to_sas_metadata` 的字典，每组复制其外层并额外加入 `block_size`，然后在 DSpark 分支交给该组 `build_for_drafting`。外层字典被复制，但其中的 `common_ratio_to_sas_metadata` 内层对象仍由各次浅复制引用；这说明该键可传递同一对象，不证明任何实际 builder 读取它。

`attention/dsa_v41.py:839–842` 的 V4.1 builder 接受 `**kwargs` 但调用 `self.build(0, common_attn_metadata)` 时丢弃它们；如果**仅假设**把该一处调用改为同时传入原 kwargs，`dsa_v41.py:955–960` 读取的仍是 `common_v41_metadata` 和 `common_v41_batch_metadata`，而以上 proposer 传入的共享键是 `common_ratio_to_sas_metadata`，当前调用点没有前两项。故原条件下仍会走构造私有 `shared` 字典，不能因为“只转发 kwargs”就认定跨组共享生效。CP builder 继承 V4.1 的该 draft 方法（`context_parallel/dsa_v41_cp.py:48,103`）；其 `_build_global_metadata:87–95` 也只依据两项 V4.1 键创建 `cp_global` 子域。该结论只描述已检查的调用与方法，不能排除其他调用方、覆写、对象状态或共享机制。

这是一个需要同时满足的接口条件，而非实际修改建议：要研究共享，除了显式建立正确的键与作用域，还须证明 group 的 block table、slot、压缩比和 local/global 域不会错误复用，并满足原 buffer 的 producer/consumer 和跨步生命周期；也要核实 `block_size` 的传递改变不会破坏构造语义。实际 draft group 数、所选 builder/运行代码与已存 snapshot 的绑定、真正重复工作及收益均未知，不能把源码不匹配称为线上缺陷或实际瓶颈。

本地快照 SHA-256：`llm_base_proposer.py` `458fb50e6839f3a353ddc7181ace4fdae02e35093d8ccd1c4327dc24ae5fff31`；`dsa_v41.py` `3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd`；`dsa_v41_cp.py` `039ec9d18e0a8a4aa39f9d5b4bea15204cea3f397a3fce97a5097b266bc9822e`。只读文件，未启动 NPU 或修改实现。

## 索引生成与图稳定缓冲的跨接口直写候选（2026-10-30）

与上述 global→local 长度复用不同，这里关注**同一次 global builder 内的生成→复制**。冻结的 `dsa_v41.py:1071–1096` 非 causal 且未传 indices 的分支首先调用 `build_dspark_swa_indices`，之后取得持久的 `_dspark_swa_indices[:num_actual_tokens]` 和 `_dspark_swa_lengths[:num_actual_tokens]` 并分别 `copy_`，再将这两个 view 作为后续数据；注释说明图捕获需要稳定地址。辅助 Triton 源码 `dspark_swa_indices.py:47–75` 在入口以 `torch.empty` 创建临时 INT32 `[T,1,D]` 和 `[T,1]`，kernel 将每行索引及长度写入其中。两份源码只能条件性串联：需要 V4.1 调用命中 `dsa_v1.py:466–480` 的 logical 模式、NPU tensor、非 None token 数、helper 可用以及两个输出参数皆为 None 等条件，尚未证明历史该具体样本执行此调用链。

可研究让相同生成逻辑**直接写入两个已取得的持久 view**，以避免单独生成临时张量后分别复制；这需要扩展生成器接口并保持稳定地址，并非当前已有的开关。特别是 `dsa_v1.py:470–480,506–524`：传入现有 `indices_output` 或 `buffer` 会使 Triton 条件为假，走另一条先构造 `per_token_slots`、再 copy 的 fallback；其中 `buffer` 只承接 indices，不是供 Triton 同时直写 indices 与 lengths 的接口。所以简单传入现有参数不能证明消除复制，反而改变路径、shape/alias 和设备任务。

重排必须保留每个有效行的逻辑位置及 `-1` 填充、INT32 值及输出 shape，以及 padding/空 token 行行为；两个 view 必须逐次全部覆盖所需区域且不能与输入或并发消费者冲突。尤其写入时间会从“临时结果生成之后 copy”移到“生成开始时”，因此必须证明上一轮对持久地址的最后异步读取已完成后才允许该轮写入，同时本轮对 SMLA 和 metadata 的消费者必须等待对应索引和长度都可见；capture/replay 时 buffer 身份、跨流 event、并发 batch 和生命周期也不能改变。现有前向 metadata ready 等待不自动证明 buffer 可安全覆盖。若实际 ABI 或其他调用方观察临时对象及其异常行为，亦需保留其契约；不能将该候选当作当前实现或已有收益。

按源码表达式作**条件逻辑字节会计**：若两项均为上述 INT32 形状且两次 `copy_` 真正分别读取临时数据并写持久数据，则可被研究消除的复制读写为 `8*T*(D+1)` 字节；例如 `T=15,D=256` 为 30,840 字节。该例的形状与 stage136 保存的一条 helper task 相容，但**没有**将该 task 绑定到后续两次 copy 或同一次历史调用；这不是实际 HBM 流量、必要下界、设备耗时或提速。实现可能融合、复用或以其他方式完成 copy，而直写也可能改变资源争用与图行为。此候选与 global→local 长度复用存在共同缓冲及潜在重叠，不能将收益相加。只检查已保存源码和已有 trace 摘要，没有运行 NPU 或修改推理实现。


## DSpark draft cache 到 V4.1 builder 的源码选择链（2026-10-29）

在保存的源码中，`models/deepseek_v41/dspark.py:165–173` 创建 DSpark decoder，`:137–140,101–104` 指定 DSpark attention 和 DSpark SWA cache；`models/deepseek_v41/model.py:797–805,627–647` 按对应类创建 attention 和缓存层，并保存到 `dsa_attn.swa_cache_layer`。`dspark.py:278–279,443–444` 返回每个 draft layer 的 SWA cache prefix。`spec_decode/dspark_proposer.py:127–175` 使用这些 draft cache 名称，在匹配 KV cache group 的 layer 上取得 backend，建立 AttentionGroup 并调用 `create_metadata_builders`；仅在相应 layer 名真实属于 group 时构造，不能据此推断历史 group 个数。

该 DSpark SWA cache 的 `get_attn_backend` 返回 `DeepseekV41CacheBackend`（`dspark.py:97–98`）；后者的 `get_builder_cls` 调用 `get_v41_cp_classes()[0]`（`attention/dsa_v41.py:1440–1444`），并依赖构造时的当前配置。`context_parallel/dsa_v41_cp.py:34–45` 在 legacy DSA CP 真且 PCP 假时返回 `AscendDSAV41CPMetadataBuilder`，在二者都真时抛错，其他情形返回对应 PCP 或普通 builder。当前 169 stage 安装的 vLLM `v1/worker/utils.py:271–316` 也已只读核对：`AttentionGroup.create_metadata_builders` 通过 `self.backend.get_builder_cls()` 取得类并实例化，`get_metadata_builder` 返回实例；该文件 SHA-256 `0ca3ec6bf4d20076145b7fe64e564962adebedc8f75a856b517ba0b2f2d77acf`。这是检查时 stage 文件，不证明历史进程加载了它。

因此，**若**冻结 DSpark 类在历史实例中实际加载、这些 cache prefix 属于相应 draft groups、该组使用所查 AttentionGroup 创建路径、当前配置确实满足 legacy CP 真及 PCP 假，源码选择链可将相关 group 的 builder 收窄到此前分析的 CP builder；这比仅凭启动参数猜测更具体，但仍不是历史实例绑定。已保存启动请求 `enable_dsa_cp=true` 不等于构造时生效配置或 class identity；其他 cache/backend、group member、动态覆写和运行版本须分别核实，也不能据此断言重复构建或性能影响。

复查时本机快照与 stage 中 `dspark.py`、`model.py`、`dspark_proposer.py` 的 SHA-256 分别一致：`fcee256905928917154d8dac94d6611bb83d02da441e8aa154a525ef51e6a601`、`101f96b38e9fc47e7be147ddfe56976aee896867c73a33a7d95c3e3c833a6e91`、`70d216de21875e679d510153ebdb0f0ccf130875a0e644f0673d60b67b6b633c`。文件相同不能证明历史进程所加载对象和本次相同；仅做 CPU 只读核验，无 NPU 任务。

## EP7 生成后复制任务：关联更近，但数据身份仍未绑定

此前只检查了 kernel CSV 中指定算子名称，可能漏掉非 compute 的复制。本次针对同一预选 helper task `39195`，只读 EP7 保存的 profiler 数据库，查询 stream 47 的 task `39195–39201` 及前三条 CANN API 对应的同线程 host 包含区间。数据库 SHA-256 `8b8e72a61e2521f0d91050bc53ab852ddf26dd4e40e2d5ec5d3973652fd807c6` 已重新校验；[查询脚本](query_f31_ep7_copy_link.py) SHA-256 `b60019a78702ed1548cebf7c1450f5bb82f11f422fc51381c6e1bcc9c264c643`，[结果](f31-ep7-copy-link.json) SHA-256 `850f7c0b813b12ab919c9479e9dc8466d0dd293cc6e75a0af799a17e86a6d8bb`。

`39195` 是 `_build_dspark_swa_indices_kernel`，历史 CSV 导出其两个 INT32 输出形状为 `[15,1,256]` 和 `[15,1]`。其后同 stream 紧接 task `39196`、`39197`，分别对应 CANN `aclnnInplaceCopy`，设备记录时长 0.800 和 0.760 微秒；三条任务对应 host CANN API 均被相同线程的 `Dequeue@...` 区间包含，顺序与任务顺序一致。这比仅比较宽窗口的时间邻近更具体，**仍不能证明**两条复制分别读取 helper 的两个输出、写入源码中的持久缓冲，或者这三条属于同一次 Python builder 调用：当前记录没有复制 tensor 的源/目标地址、shape、大小和 Python 调用栈。两条时长也不是可消除的端到端收益。

同流后续还有两条 `aclrtMemcpyAsync` 和各自 `aclrtRecordEvent`；存在这些记录不证明 event 身份、其他流是否等待或某个 buffer 的复用安全。对同一窗口的 CSV 附加核查发现，先前 `[1,1,256]` 的 `GreaterEqual` 前有相容形状的 INT32→INT64 `Cast`，其后 BOOL→INT32 `Cast` 紧邻 `ReduceSum`；这解释了先前导出 dtype 不连续的一种可能，但仍缺张量身份，不能把它们确定为同一数据链。

尝试只针对前三个 dequeue 的已知 connection group 回查 enqueue/其他关联时，SQLite 常规只读和 immutable 只读查询均返回 `disk I/O error`；该失败不否定已成功读取的固定 task 记录，也不证明整个数据库均不可读。不要原样重复失败查询；若之后能使用可验证的一致副本或另一个有效已有 trace，可按已知 connection group 有界重试。确认 F31 实际生成→复制→消费关系最少还需可绑定到该调用的 host 栈/queue 关联与复制 tensor 身份、大小、stream/event 依赖，以及运行源码与实例绑定；本轮未启动 NPU、未修改推理实现。

## 非 causal 索引重建不能用局部 query 行数替代全局行数

复核冻结 `dsa_cp.py:1139–1144`、`dsa_v41_cp.py:115–166` 与 `dspark_swa_indices.py:20–38`。前者将 token 行补齐到 TP 组大小的整数倍并按 rank 切分；noncausal 情况下 CP builder 将 suffix 设零，保留该请求的 device seq_len，并在 global indices 存在时将 `[actual_start:actual_end]` 的行切片传给 local builder。helper 则以该请求的 `query_end−query_start` 和 seq_len 推出 `start=max(S−q−W,0)`，每行生成从 start 到 S−1 的逻辑索引。

一个满足该源码分区规则的条件反例：假设 3 个请求各有 global q=6，共 18 行，TP=8，补齐后 24 行、每 rank 3 行；第一个请求的 6 行分别落在 rank0 和 rank1，各自 local q=3。假设该请求 device S=1000、window W=128、indices 宽度 D=256，global helper 为该请求各行生成 `866..999` 共 134 项；若错误地只用 local q=3 且保留 S=1000 重建，同一局部行变成 `869..999` 共 131 项，丢失 `866,867,868`。当前条件路径按行切 global indices 避免此简化；重排/复用必须保留 global q 或等价的 global start、每请求的映射和正确 device seq_len，不能仅靠 local 行数与局部长度重建。

另一个更细条件反例：同一请求真实 device S=1000，若 CPU mirror 假设陈旧为 1003，q=6、W=128 时两者生成的有效长度均为 134，但索引分别为 `866..999` 和 `869..1002`；**长度一致不足以证明索引一致**。这只证明值不一致时的后果，不说明历史 mirror 曾陈旧或当前实现有缺陷。CP builder 的注释强调 device seq_lens 在 speculative rejection 后为权威，而 proposer 也期望 runner 已按 accepted count 修正 canonical host mirror；相关更新时点和实际分支仍未绑定。

源码 SHA-256：`dsa_cp.py` `7e8dd93eeda988866138342523ecfa7cc9c42ce67669d4b1507ee8c5f3452305`，`dsa_v41_cp.py` `039ec9d18e0a8a4aa39f9d5b4bea15204cea3f397a3fce97a5097b266bc9822e`，`dspark_swa_indices.py` `335be10476943d4ad429f4d52b7a0e069f3b65511d28265b0e9d9f7d2e94111c`，`dspark_proposer.py` `70d216de21875e679d510153ebdb0f0ccf130875a0e644f0673d60b67b6b633c`。上述是源码条件推导和整数算术，不声称历史调用出现上述 q/S、不同步或错误，也没有新增 NPU 工作。
