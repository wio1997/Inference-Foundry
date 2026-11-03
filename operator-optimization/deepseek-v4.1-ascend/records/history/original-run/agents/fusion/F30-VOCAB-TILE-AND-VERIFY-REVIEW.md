# F30：词表分块采样与 target 验证边界

## 这次补查的范围

先前 `DSPARK-COMPUTE-AND-STATE-REVIEW.md` 已指出 base logits 可以批量计算、Markov token 按位置串行，且把 bias + argmax 作为条件候选。本记录不重复这条结论，进一步拆解 **一次 greedy 采样内部的词表分块合并**、跨位置的交错边界及草稿输出到验证的依赖，不改实现或设备。

启动快照 `evidence/frozen-serve.sh` 指定 DSpark、K=5、`enforce_eager=true`；基线何时 greedy、每个请求的实际温度与 sampling metadata、模型对象和每次分支并未被启动脚本证明。`llm_base_proposer.py:1534–1606` 指定 batched base logits 与逐位置 Markov bias/采样；`models/deepseek_v41/dspark.py:315–339,410–438` 使用 norm+LM head 的 logits processor、Markov head 的 embedding 和 bias，且该具体 V4.1 draft 对象的 `map_draft_to_target` 是 identity。**实现中判断是否存在 vocab mapping 使用另一个运行时属性，identity 方法本身不能证明该属性为空或该轮采用概率采样。**

## 已有融合、采样和边界

- 已有 trace summary 的 EP7、EP8 历史整段各记录 7 个 `rejection_greedy_sample_triton`、分别 65 和 70 个 `ArgMaxV2`；它们只证明这些 task 类型出现过，不能把总数按 K 分配到每个 DSpark 请求、断定采样顺序、对应此源码调用、target 验证模式或具体 logits。rejection kernel 是**已有单个复合 task 名称**，内部比较、prefix 筛选和结果写入是否全融合、具体逐位语义需看对应实际 kernel 源码，现有本地快照未包含它。没有源代码契约时不能根据名称重新声称已实现或未实现某个细节。
- 指定 greedy 分支已有 `logits[:,idx].add_(logits_bias)` 后调用 `argmax(dim=-1)`，也就是 Python 源级两个调用；不证明设备实际发生了两次 HBM 往返，也不证明 compiler 没融合。`raw_logits.view` 切片与 `add_` 会修改 base logits 的对应行，新的方案若不保留被任何后续代码观测的变化，要证明其他所有消费者、alias、callback 和 graph 生命周期。
- probabilistic 分支依 `sampling_metadata`、条件 vocab mapping 等生成 token，并可能生成并保存每步的概率分布到 `_last_draft_probs`；与 target 的拒绝采样使用兼容概率空间，不能把只返回 greedy token 的重组套进去。采样顺序、RNG 的使用与状态、temperature/top-k/top-p 等实际采样配置均需保持；本地文件没有可绑定的 target rejection 源码/输入对象，不对具体实现作假设。

## 新增：一个 token 的词表分块归约

把已经通过 `compute_draft_logits` / logits processor 的第 i 位置基础向量记为 `L_i[v]`，前一个草稿 token 决定 Markov 偏置 `B_i[v]`，则该步源码先按原 dtype 做 `S_i[v] = round_dtype(L_i[v] + B_i[v])`，再全词表 argmax。为明确舍入顺序，此处 `round_dtype` 指原算子实际写入类型及原实现规则，非对特殊值/混合 dtype 的额外假设。

**条件重组**：将 vocab 分成互不重叠且覆盖整个有效词表的块，对每块直接得到同样 `S_i[v]` 的局部候选 `(score,index)`，最后按原 argmax 对 score、全局 token ID、ties、NaN 的完全相同规则做归约；若该规则能表示为一致的确定性全序，局部最优再全局合并在数学上可保持最终 token。该方法可不必把 `S_i` 物化为完整中间向量，但仅当没有其它可见消费者、输入 logits 的 in-place 效果允许消除或以等价结果保留，且 logits processor/并行词表重排语义保持。实际 FP16/BF16/FP32、NaN 规则、是否要求逐位 logits、被采样端实际绑定均未知；不声称接口已支持。按条件使用的全序不能自己假设为“最小 token ID 优先”。

不能先对基础 logits 做 top-1 再只计算其 Markov 偏置：一个无需运行设备的反例 `L=[10,9]`、`B=[0,2]`，基础 top-1 为第 0 项，而修正后 `[10,11]` 为第 1 项。固定只选 top-M 也需要一个能证明未选项不会因偏置翻盘的**全局上界**，不能因平均偏置小而跳过其余词表。每个词表块的基础计算与偏置计算在相应输入准备后可局部交错，但最终 token 必须包含所有可能胜出的块的信息。`LogitsProcessor` 与 `ParallelLMHead` 的 gather/词表重映射、约束与 padding 是否允许块级执行在已有本地源码中未展开；应保留整个已处理 logits 的接口，不能直接把分块推到其前面。

## 跨位置和验证的依赖

第 i 步需要 `draft_token_ids[:,i]`，第 i+1 步的 Markov embedding 依赖本步产生的实际 token，所以一般情况下 **Markov 选择链不能在不同位置无条件并行**。但各位置的 base logits 在 draft hidden states 完成后批量产生；在控制数据生命周期和接口的条件下，可研究将后续位置 base-logit 的不同 vocab tile 与当前步 Markov/归约交错。源码当前 `raw_logits = compute_draft_logits(...)` 先返回完整张量，未给 chunk-ready API 或事件；跨阶段重叠是研究候选，不是历史运行事实，且会争抢 Cube/Vector/访存。

对于条件化 target 模型，验证位置 j 的条件前缀包含更早草稿 token；在不预测分支、不做冗余尝试的前提下，不能在依赖 token 值已知之前决定对应输入。理论上可以在某个前缀确实确定后研究按前缀/块提前验证，同时其他后续 draft 工作继续；这要求 target KV 与拒绝后回滚、请求调度、验证概率、终止条件和批量布局等与原实现等价，不能承诺当前接口或现有采样已支持。概率验证尤其需完整对应 draft 分布；只保留 argmax 是不够的。target 模型对于该条件前缀的所有输出不能单凭草稿置信度略去。

## 结论与可恢复输入

这不是新的性能估算：大词表分块有潜在中间 materialization 候选，但完整覆盖所有 vocabulary、原有精度/排序、接口、stream 依赖及概率输出必须保持，实际 HBM、流重叠与可节省时间未知。冻结源 SHA256：`llm_base_proposer.py` `458fb50e6839f3a353ddc7181ace4fdae02e35093d8ccd1c4327dc24ae5fff31`，`dspark.py` `fcee256905928917154d8dac94d6611bb83d02da441e8aa154a525ef51e6a601`；既有 `evidence/TRACE-SUMMARY.json` `af2a29b27353a93826f15f8393595d9ab77bea3f1d96ec05f7705be591a0c9cf`。如要继续，先从现有已保存数据定位 logits processor、Markov head、rejection kernel 的具体契约与样本调用链；没有则标 UNKNOWN，不做无目的测试。整个记录只读历史资料、没有 NPU 运行。

## 概率路径：原位 logits、全批噪声与跨轮概率的具体契约

冻结的上游 `vllm/v1/spec_decode/llm_base_proposer.py:261–265,453–479,1848–1885` 与 Ascend proposer 的调用链补充了前文“具体随机数逻辑未知”的源码范围。条件是 `_enable_probabilistic_draft_probs` 由 `rejection_sample_method == "standard"` 且 `draft_sample_method == "probabilistic"` 同时成立，DSpark 的 `draft_id_to_target_id` 又为空；实际运行值、历史二进制、采样实例均未绑定。V4.1 草稿类的 `draft_id_to_target_id = None` 是类属性默认（`dspark.py:345–347`），不能排除运行对象覆盖。虽然上游概率函数前有旧注释称其未使用，当前冻结 `_sample_from_logits` 在特定分支明确调用它；不能把注释当作运行事实，也不能因源码路径存在就断言样本命中。

具体而言，`sampling_metadata is None` 时 Ascend wrapper 返回 argmax/无 probs；上游配置未启用概率或 `sampling_metadata.all_greedy` 时也返回 argmax/无 probs。进入概率函数时，若 temperature 行数不同于 logits 行数，先要求整除并按 request-major `repeat_interleave`；混合 greedy/random 时把低温 greedy 行的除数临时替成 1。对这些实际进入概率函数的行，源码再对 logits 进行 **原位**温度除法，计算 FP32 softmax，按 `probs` 的 shape 分配指数噪声并执行 `q.exponential_()`；将 `probs.clone()` 传给带噪采样，从而保留原概率，混合批次最后用 `where` 选择 greedy 或随机结果。只使用 temperature 是当前函数自身的源码行为，不能替代 target sampler 的完整语义。源码对 RNG seed 还有 TODO，没有显式传 generator；历史 RNG 实例与状态未知。

DSpark 内部每步先对 `logits[:,idx]` 原位加 Markov bias，再在概率路径对同一 slice 原位除 temperature；两项都会改变底层 `raw_logits` 所代表的存储。跨 token、跨词表 tile 的重排必须同时保留原数值顺序和所有可能的 alias/可见消费者，不能只看加 bias。每步的噪声覆盖整个该步 batch×vocab：即便某些请求最终通过 `where` 选择 greedy 分支，**混合批次仍先为整批生成噪声**；仅给随机行生成、改变词块遍历/并发顺序或生成器使用方式，都可能改变随机数到元素的对应及后续 RNG 状态。若验收仅要求某种分布等价而非相同随机流，需明确该契约并另证；不能把数学分布相似当作原实现逐请求结果及 RNG 状态相同。

当每个 step 都产生 `probs` 时，Ascend `llm_base_proposer.py:1585–1606,1636–1643` 先保存每步 `[num_blk,V]` 概率，再 stack 为 request-major `[num_blk,K,V]` 并整理给 `_last_draft_probs` 的早返回路径。已存 `speculative.py:1461–1462` 在相应配置逻辑中将 dspark 置为 `parallel_drafting=True`，属于条件性配置证据而非历史 resolved 对象；每步没有返回 probs 时不能假设存在该张量。已存 `ascend_runner.py:1914–1917,2098–2120,2985–2993` 显示：每次 propose 前清空旧 `_draft_probs` / `_draft_prob_req_ids`；产生新概率时保存它并复制当前 request id 列表；后续 `_sample` 从 `_get_spec_decode_draft_probs` 取值并把结果、spec metadata、target logits 与 sampling metadata 传给 rejection sampler。该 getter 的完整 request 重排/校验实现和 rejection 内部算法在此快照未绑定；不能把同一次生成结果错误地复用于后续批次或未经许可丢弃完整分布。

仅作形状会计：**如果**一次实际 batch 恰有 8 个有效请求、K=5、V=129280 且概率为 FP32，那么 `[8,5,129280]` 逻辑大小为 **20,684,800 字节（19.73 MiB）**；一轮并发 8 不证明实际采样 batch 恒为 8。这不是实测显存、额外 HBM 流量、peak、必要分配量或理论上界。研究分块生成/验证需考虑 request id 归属、每步可见完整概率与存储生命周期，不能仅保留 argmax；也不能以概率分布的大小估计节省时间。

本段静态证据 SHA-256：上游 proposer `5bd11c41f0a8cf1272b5880d9878043f876bee6c7e04fb781e613f92930b15ab`；Ascend runner `568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c`；Ascend proposer SHA 见上文。未执行 NPU、不修改目标实现，不从这些路径推断历史分支、瓶颈或收益。

## 首个 Markov seed 与草稿 decoder / base logits 的分叉

先前已分析各草稿位置 base logits 与逐步 Markov token 链，但**第 0 步的 seed 来源**与后续位置不同：`dspark_proposer.py:226–246` 在 first-pass 输入准备时将 target 已提供的 `next_token_ids` 复制到持久 `_dspark_seed_buffer[:n]` 并把余下位置填零；这份 seed **依赖 target 采样已产生 next_token_ids**，不能在目标采样结果未知时凭猜测提前执行。此文件:45–47 中 seed 和 draft token 都是跨调用复用的预分配 buffer。

在 `Ascend .../llm_base_proposer.py:1479–1495,1534,1577–1589` 的源码顺序中，模型输入准备后先调用草稿 decoder，选取 `sample_hidden_states`，再完整调用 `compute_draft_logits`；拿到 `raw_logits` 并 reshape 后才由该结果的第一维得到 `num_blk`、复制 `_dspark_seed_buffer[:num_blk]` 到 draft 第 0 列，并运行第一步 `markov_embed`、`markov_bias`。第 0 步的数学输入 seed 与 base logits 的当前草稿 hidden 有不同的直接来源；**如果**确知同一次目标结果、有效 request 行顺序、精确 `num_blk` 和 padding，并保证模型权重及所用共享组件就绪，可研究把 `markov_embed(seed)` 或连同 `markov_bias` 的准备安排在 draft decoder 或 base logits 进行期间；第 0 次加 bias 和采样仍必须等待对应 base logits 与 Markov 结果。第 i>0 步还依赖上一步实际生成并写入的 token，不能照搬 seed 规则提前确定。

这个分叉比前文把“后续位置 base-logit tile 与当前 Markov 步交错”表述得更具体，但不构成已实现 overlap。注意以下依赖：

- 源码的 `num_blk` 是计算完 base logits、处理可选 lm-head TP padding 后由 `logits.shape[0]` 得到；提前阶段不能简单把并发数、`cad.num_reqs` 或 seed 非零数当作它。特别是 `num_blk>n` 时源码 seed 的尾部为零，且 lm-head TP 分支先裁切 logits；需证明各 rank 对所有 padding 与真实请求采用一致的行映射和 collective 次序。
- `dspark.py:320–324,414–437` 将 `markov_bias(markov_emb)` 和 base logits 都交由同一个模型内的 `logits_processor` 对象间接参与。这里仅能证明对象被传给两个调用，不证明 `DSparkMarkovHead.bias` 内部的所有计算、是否有 collective、该对象无状态、线程安全或能与自身另一调用并发。该段撰写时本地缺少 `qwen3_dspark.py`；后续 stage244 已取得 head 实现，见下节，但 processor 全部缩放/通信和历史实例仍未绑定。若两条路径涉及 TP/DP collective，所有参与 rank 必须以协议允许的一致顺序发起；各自同样的输入也不允许无证明重排 collective。
- seed 在每次 first-pass 被覆盖、尾部重置；早启动的读必须等对应次写完成，且在该次 read 和所有使用结束前不得被下一轮覆盖。同步还要保留原 `draft_token_ids[:,0].copy_` 及可见状态，或证明改动后的存储、padding 和 downstream 无差异。动态 confidence 支路可复用 embedding 的候选早已记录在 `DSPARK-COMPUTE-AND-STATE-REVIEW.md`，但需要单独验证；不能当作本条新增可相加收益。
- base 与 Markov 路可能同时消耗矩阵/向量计算、访存和通信；`logits` 的 in-place 修改、概率/RNG 状态以及 later rejection 保留前节约束。不知道硬件时间线、队列与实际等待，就不能把延迟相加减或声称一定隐藏。

stage241 的两个 EP8 宽窗口各记录三个已筛选 bracket、随后五个 ArgMax 类型任务，只提供同流时间顺序；没有把具体 seed、Markov 或 request 身份绑定到这五条记录，因此不能把五条直接认成上述五次迭代，也不能计算此候选的实测覆盖或收益。全部分析基于既存源码，未操作 NPU；若需闭合优先寻找已存在的 head 源码/图、collective 协议、buffer/事件关联与采样配置，缺失则保持 UNKNOWN。

## stage243：两个窗口、八个间隔中的物理形状与 Markov 条件预测

基于 stage242 固定的 EP8 两个窗口、五个后置 ArgMax 之间八个间隔，stage243 对每间隔 10 条已选任务按同一 rank、model、stream、task 和起止时间在冻结 CSV 唯一查回元数据；**80/80** 匹配，结果 `agents/diagnosis/draft-argmax-intervals-shapes-ep8.json` SHA-256 `59930fe7433cb31708f906713b8f24685dd307cb43ad1ab7a0b2d35310766dfe`。这些仍只是导出的物理 shape/dtype 与同流顺序，不是 tensor 地址、内容、请求身份或模型调用链。stage241 的相邻 ArgMax 本次没有直接绑定到 Markov。

八个间隔中，每个都有以下形状相同的记录：INT32 `[4]` cast 到 INT64 `[4]`；`Slice [4,6]→[4,1]`；`GatherV2` 从 BF16 `[129280,256]` 和 INT64 `[4]` 等输入得到 BF16 `[4,256]`；随后 `MatMulV2` 输入 BF16 `[4,256]`、`[129280,256]`，输出 BF16 `[4,129280]`；另有 `Slice [4,646400]→[4,129280]`，以及 BF16 两个 `[4,129280]` 输入的 Add 与同形输出。查询还见到长度 `2585600` 的 BF16 ViewCopy 导出记录；名称和形状不能证明物理复制量或 in-place 存储行为。

源码 `dspark_proposer.py:45–47` 构造 `[max_batch_size,1+K]` INT64 token 缓冲，`llm_base_proposer.py:1577–1602` 将 base logits 按 `[-1,K,V]` 解释并逐步对 `logits[:,idx]` 添加 Markov bias；已存启动脚本请求 `K=5`，model config 声明 `vocab_size=129280`、`dspark_markov_rank=256`。若历史真实对象确为该 DSpark 路径、真实词表大小相同且这组记录分别是所述子操作，则物理代数关系相容：`6=1+5`、`646400=5×129280`、`2585600=4×5×129280`，中间 rank 维 256 和词表宽 129280 也与配置匹配。**4 是这些任务的物理首维，不证明四个有效请求、该次 sampling batch 的实际值，更不与用户发起并发 8 等同。**该段撰写时 `qwen3_dspark.py` 还未保存在本地；后续 stage244 已取回实现，见下节。即使有源码，shape 一致仍不证明这些设备记录是 Markov lookup/projection、权重存储相同或直接生产/消费关系。

这比仅凭五条 ArgMax 数量更具体，但仍不能宣称这些任务属于真实 DSpark、某次 Markov 第 i 步、某个请求、同一组张量或特定参数，也不能把八个间隔的时长归因为 Markov、更不能估算融合收益。可证伪的条件预测是：若完整的 Add 记录确实实现所述 eager `logits[:,idx].add_(bias)`，两个输入必须能按该调用的逻辑 `[num_blk,V]` 广播且结果满足其输出约束；若调用先经过图分解、padding、切片或子任务，单一设备 task 的物理元数据不一定是完整 Python 输入，因此不能用任一物理 shape 不符就无条件排除全路径。要进一步收紧，需找到已存或当前容器的 head 实现、量化/词表与并行契约，以及历史调用图、tensor/storage 或事件与该任务绑定；对这些 80 行只读提取不会提供上述缺失信息。本次未操作 NPU。

## stage244：Markov head 源码和 stage243 形状的条件关联

本机 ZCode stage244 只读获取当前固定镜像 `vllm/model_executor/models/qwen3_dspark.py`，保存 `agents/fusion/frozen-source/vllm/model_executor/models/qwen3_dspark.py`，SHA-256 `caf8287a55d2af786487a66246118fc02c2d05166e042cfc9798206c4e8427f1`。它补齐了前文缺失的 head；当前快照不证明历史采样使用此源码或实例。

`DSparkMarkovHead:61–93` 构造 `nn.Embedding(vocab_size,markov_rank)`，以及 `ParallelLMHead(draft_vocab_size,markov_rank,bias=False,quant_config=quant_config,disable_tp=True)`。`embed` 将 token ids 交给 embedding；`bias` 将 embedding 结果和第二个 head 交给传入的 `LogitsProcessor`。V4.1 `dspark.py:205–212` 用 `config.vocab_size`、`draft_vocab_size or vocab_size`、`dspark_markov_rank` 创建该类，调用时未传 `quant_config` 和 `retain_weight_for_gather`，故在该构造调用及当前默认签名成立的条件下分别是 None 和 False。类把两个权重声明为不同模块参数，但没有历史权重对象/存储绑定，不能断定 stage243 两个同形输入是同一或不同物理 buffer。

如果实际对象采用上述 V4.1 构造、完整词表为 129280、rank=256、一次物理输入 B=4，并且 embedding 与 unquantized head 的物理实现分别被 profiler 以整任务导出，则预测 lookup 表为 `[129280,256]`、ids `[4]`、embedding `[4,256]`；全词表 bias 物理值为 `[4,129280]`，标准线性后端可将 head 权重作为 `[129280,256]` 使用。stage243 八个间隔中的 `GatherV2`、`MatMulV2`、Add 形状和上述条件预测都相容，且进一步吻合每步依赖前一个 token 的源码顺序；这比旧记录只有接口名和 rank 的相容性更强。预测可被该具体路径的已绑定 weight 元数据、输入行数、call graph 或 tensor identity 否证；仅名称/shape 相同并不反过来证明绑定，也不能用底层子任务或不同 padding 导致的 shape 差异无条件否定整个 Python 调用。`disable_tp=True` 是所存 constructor 参数，仍须核对 Ascend 覆写、processor 的通信和映射及 resolved 对象，不能仅凭它认定历史调用完全无 collective。输出 dtype、processor 缩放、词表 padding、有效请求数、实际权重分片或编译结果亦未知。

仅依以上物理维度做 **标准稠密**算术：一次 B4、V129280、r256 投影为 `4×129280×256 = 132,382,720` 逻辑 MAC；若权重与输出假设 BF16，一个 `[129280,256]` 权重是 **63.125 MiB**，一个 `[4,129280]` 输出是 **1,034,240 B（约 0.986 MiB）**，`[4,256]` 则为 **2,048 B**。若两张表各独立、全量且均 BF16，则它们的总逻辑大小为 126.25 MiB；这是假设，并非已证实际物理存储。以上均不是实际硬件指令数、有效行数、HBM 必读字节、缓存缺失、运行时峰值或可兑现收益，不能乘以五次当作已证实的一次生成成本。

该源码已有另一个 `apply_bias_gathered` 方法（95–161）：给定候选 `index/values`，对选中词表行计算 Markov 修正再 scatter 到调用者准备的 logits；注释描述未选项留 `-inf`。CUDA NVFP4 分支及 fallback 的条件各异，不能说 A3 走 CUDA 实现。它服务截断提案，**不是自动与全词表逐项 bias→argmax 等价**：若被跳过词项的 Markov 修正能改变最大值（前文 `[10,9]+[0,2]` 反例），截断会改变 greedy；概率路径还要求相应分布和 RNG/验证契约。`Qwen3...` wrapper 可调用该方法（288–301），但当前已审的 V4.1 `dspark.py` 暴露 `markov_bias`，Ascend proposer DSpark 普通循环用完整 bias；没有证据当前 V4.1 历史运行启用了 gathered 方法，不能把它称作新增实现或已确认执行。实际配置和其它动态绑定仍须证据。

下一步如要证明八个间隔是该路径，需历史调用与两个权重/tensor 的身份、实际 processor 分支与 batch/token 映射；无需再通过名称/shape 重复加固。同流顺序、当前源码和参数相容均不足以单独确证历史角色或收益。本轮无新 NPU 工作。

## 条件性贪心截断证书与数值舍入反例

考虑一次 **greedy** 采样，把最终实际参与比较的有限、非空有效词表记为 `V`，已经选出的非空集合记为 `C`；必须先把 processor 的 scale、bias、mask、padding、词表映射及过滤后的索引与比较顺序按真实路径确定。令 `S[v]` 为原实现对每个有效词实际会用于 argmax 的最终分数，候选 `C` 的分数必须按原实现求得，或有证据证明替代计算完全等价；`m=max_{v∈C} S[v]`。若对每个 `v∈V\C` 都证明 `S[v]≤U`，且 **`m>U`**，则任何被排除词都不能与最终最大分数相等或超过它；在候选内部继续使用原 argmax 的 tie 规则，所得 winner 与全词表相同。若 `m≤U`，证书仅是不成立，不能据此判断候选错误。可用小词表完整计算 `S` 逐项反驳任何给出的错误上界；不能以经验均值当证明。

一个仅在相应模型语义得到证明时成立的上界构造：设 `L[v]` 是加 Markov bias 前、已处理且与最终有效词表正确对齐的 base logits；如果实际路径可以证明其最终分数满足 `S[v]≤L[v]+α·⟨e,w_v⟩+c_v+ε_v`，其中 `e` 为当前 embedding、`w_v` 为对应词权重、`α` 是已核实的投影缩放，且已证明对所有排除词 `|c_v|≤C0`、`ε_v≤E`，由 Cauchy–Schwarz 可得

`U = max_{v∉C} L[v] + |α|·||e||₂·max_{v∉C} ||w_v||₂ + C0 + E`。

`E` 必须覆盖**实际**投影、量化/反量化、processor 所有适用处理及最终加法的舍入和其他误差；若非线性后处理不能由上述关系覆盖，须先另证上界，不能套用该式。范数、最大值和整个证书的计算也要有可核实的保守误差或向上舍入，避免有限精度把真实上界算低。使用负缩放需要绝对值；任何 NaN、无穷、溢出、未知词表重排、跨 rank 合并、动态 mask、真实权重/误差未知时，以上有限实数条件不成立，应返回 UNKNOWN 或使用已证明处理这些情况的其他证书。空排除集是单独的平凡情形。这个证书仅证明 greedy token：源码中的 logits 原位修改和其他可能的可见消费者仍须保留或证明无影响；概率采样与完整概率/RNG 契约不能由它证明。计算 max、范数、误差界及跨 rank 归约本身有成本，无法假定免费；当前并未构造实际实例或证实收益。

现有 `apply_bias_gathered` 的 fallback 是对 `weight[index]`、`markov_embed` 与 `values` 使用 `baddbmm_` 后 scatter（`qwen3_dspark.py:132–160`）；完整路径则经 processor 产生 bias，再对基础 logits 原位 `add_`。即使候选覆盖完整词表、数学表达式相同，也**不能由源码名自动证明舍入顺序相同**。例如在假设 bias 输出 BF16、标准最近偶数舍入、无其他 processor 变换、点积只有一项时，取 BF16 `e=129/128`、`w=65/64`、`L=-1`：精确乘积是 `8385/8192`，先将乘积舍入 BF16 再加 `L` 得 `3/128=192/8192`；假设融合实现保留精确乘加直到最终 BF16 一次舍入，则得 `193/8192`。两者均可用有理数复算，结果不同。这个例子只说明中间舍入是需要证明的边界，**不声称**当前任何 NPU 的 `baddbmm_` 实际采用后一种舍入，也不声称历史调用命中 gathered 分支或产生过该差异。若要证明 token 等价，应使用实际内核与 processor 的数值契约，或用覆盖误差的 winner 证书；不得将已有方法直接视为精确融合。
## 同一步不同请求的重复 Markov token：仅共享 bias 的条件

此前 F30 讨论了逐词表 tile、seed 提前和截断；`DSPARK-COMPUTE-AND-STATE-REVIEW.md` 讨论的是**同一请求** Markov embedding 给动态 confidence 再用，均未分析**同一步不同请求 token 重复**。冻结 `qwen3_dspark.py:61–93` 用同一个 `nn.Embedding` 按 token 取行，再经 `ParallelLMHead`/`LogitsProcessor` 生成 Markov bias；Ascend `llm_base_proposer.py:1581–1602` 对该步完整 batch 算 bias、逐行原位加到各自 base logits，再逐行采样。V4.1 wrapper 在 `dspark.py:433–437` 转发这两项调用。这里的 `disable_tp=True` 和静态构造不证明历史实例、processor 实现或无通信；stage243 的物理 batch=4 也未证明所有行有效或 token 值。

**条件结构**：同一投影版本下，若不同有效行的前一个 target-space token ID 精确相同，且能够证明 head/processor 在该调用是确定性、逐行相同函数（包含 scale、bias、词表映射、padding 与任何 collective），可研究对这些重复 ID 仅算一次 embedding/bias，再按原请求行映射复制给每行。目标必须是原计算契约要求的 bias 值及所有外部可见状态相同；仅假设模型数学相同不够。若 processor 使用行相关参数、多个 rank 的内容/顺序不一致、对应输入值不同，或计算去重改变 kernel/累加和舍入而违反精度契约，则不能直接复用。必须保留原 batch 对应的行域、无效/padding 行、重复项反向索引、图缓冲和异步生命周期，不能因两行相同就删掉可能有其他观察者的中间结果或变更 collective 参与及调用顺序。

**不能共享最终采样结果。** 例如两请求前一 token 相同、Markov bias 均为 `[0,0]`，但各自 base logits 分别为 `[2,0]` 与 `[0,2]`；greedy 的下一 token 分别为 0 和 1。即使 bias 相同，base logits、temperature、mask、采样状态和 target 验证也仍按请求独立；概率分支还须保留完整分布及整批随机数消耗/映射，不能对相同 token 的请求共用一次随机采样。相同步只允许在各自前一 token 已经确定后去重；跨步复用还需要新的权重版本、RNG 及 buffer 生命周期证明，不得借此跳过串行 token 依赖。

收益没有观测：历史样本没有请求 token identity、重复率、精确 processor 与张量身份；`unique`、映射、scatter、可能的重新 batch、collective 及权重/激活访问都可能产生额外工作。仅作内存规模对照：若试图为全部 129280 个 token 预计算完整 129280 词表、每值 BF16，逻辑表大小为 `129280²×2 = 33,426,636,800 B`（约 31.13 GiB），不含索引、版本或执行开销。这不是实际分配建议、必要存储量或收益估算，也不排除按需缓存和不同表示；不能用无限 precompute 代替实际工作量。下一条可证伪证据是绑定该步每个有效 ID、processor 的逐行语义与实测重复数/数值契约；现有数据不能用 shape 代替 token identity。本次只读，无 NPU 操作。

## target logits 到 greedy 验证：bonus 与 logprobs 的额外消费者

前文讨论的是草稿 Markov bias 和逐 token 采样；这里补充 target LM head 到 rejection sampler 的另一条条件性边界，不是同一 draft argmax 的新收益。冻结 `sources/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/model.py:1778–1783` 的 `compute_logits` 返回 `logits_processor(self.lm_head, hidden_states)`。保存的 Ascend `sample/rejection_sampler.py` SHA-256 `bb1d39de851540c16c2dd2b16df60c60a1fc4957cedb88bdce9417efc0d55eda`：`forward:183–249` 将输入 logits 分别按 `bonus_logits_indices` 和 `target_logits_indices` 选行；bonus 交给 sampler，target 转 FP32，视 logprobs 模式克隆，再经过 logits processors 和 sampling constraints，之后交给 rejection。`rejection_sample:508–554` 在非 all-random 分支对其目标行做 argmax，all-greedy 时在对应 greedy 验证后返回。

另在 `sampling_metadata.max_num_logprobs is not None` 时，Ascend forward 调用继承的 `_get_logprobs_tensors`。保存的上游 `sources/vllm-workspace/vllm/vllm/v1/sample/rejection_sampler.py:218–265` SHA-256 `0bb9d224887ccb1355e16199005cc23a4afc6fa75156eca419851b7247d99750` 将 target/bonus logits 按行重组，按配置选择 logits 或对所选行算 logprobs，再 gather 已接受 token 和所需候选。Ascend 类在所存源码中继承上游类；历史对象和所加载 upstream/Ascend 源码、输入身份、实际模式与各调用之间的绑定仍未知。若请求 `max_num_logprobs` 为 None，上述 `if` 分支不运行；不能据此证明 sampler 的其他中间输出或其他调用方没有可观察要求。`bonus_sampler_output` 在该 forward 中始终先被生成，但其 logprobs 只在列出的该条件分支被使用；不能只因源码中有该结果就将它当成用户始终请求的 logprobs。

因此，研究将 target LM head 与 greedy 验证接合、只为目标行保留最终 `(score, global_token_id)` 时，必须分别覆盖 bonus 行，且保留需要的 logprobs、processor 和所有其他消费者；该建议不要求物化完整词表，也不预设通信算法。两个可复算的有限例子：

1. 设每个请求一个 draft token=0，target greedy argmax=0，且走标准非 synthetic 的 spec_len=1 PyTorch 路径；`:819–822` 在接受后使用 bonus token。bonus 行 `[1,0]` 与 `[0,1]` 的 argmax 分别为 0 和 1，输出分别为 `[0,0]` 与 `[0,1]`。两种情况下 target 的 top1 完全相同，无法仅用 target top1 恢复 bonus。
2. 如使用的是需要 log-softmax 的 logprobs 模式，词表行 `[0,-1]` 与 `[0,-100]` 的最大值和词 ID 均为 `(0,0)`，但对应 winner 的 log-prob 分别为 `-log(1+exp(-1))` 与 `-log(1+exp(-100))`。因此只留 top1 值与 ID 不能确定 logprobs；可研究保留经证明足够的 logsumexp、所需 top-k/请求项或等价重算，但要满足原模式、precision 与输出合同。

```python
import math
assert [0, max(range(2), key=lambda i: [1,0][i])] == [0,0]
assert [0, max(range(2), key=lambda i: [0,1][i])] == [0,1]
a = -math.log1p(math.exp(-1))
b = -math.log1p(math.exp(-100))
assert a != b
print(a, b)
```

这些是源码条件和 CPU 数学反例，未证明历史服务的请求要求 logprobs、实际采用该 PyTorch fallback、synthetic/processor 配置或上述值。F30 已有概率拒绝路径的完整 draft 分布与 RNG 约束亦不可用于忽略。Ascend `rejection_sample` 参数注释写 target logits 末维 `vocab_size//tp_size`，`:508` 注释说 global argmax 前需要 allgather，但紧接实际语句直接对传入 tensor 做 argmax；这并不证明在此位置真实执行了 collective、词表实际分片、rank-to-token 映射或跨 rank 时钟关系。若研究每 rank 局部 top1/合并，仍须绑定实际 processor 顺序、padding、词表索引、有效行、tie/NaN、rank 通信和所有输出消费者，不能凭静态注释估算通信收益。未访问 NPU或更改推理实现。

## 草稿概率到 rejection：跨批 request-ID 重排与整批缺失回退

此节补齐本记录概率路径中此前标记为未知的 getter 源码；它与前文概率分布、RNG 和 target bonus 的约束属于同一验证链，不单独累加收益。冻结 `agents/bounds/metadata-path-snapshot/ascend_runner.py:108–110,364,2098–2120,2985–2993`：`NPUModelRunner` 继承 `GPUModelRunner`；条件下 proposer 的 `take_last_draft_probs()` 返回非 None 时，保存 `_draft_probs`，并复制该时刻 `input_batch.req_ids`；后续 `_sample` 调 `_get_spec_decode_draft_probs`，并将结果交给 rejection sampler。保存的上游 `vllm_runner.py:4990–5017` **确有**该 getter，收紧本记录此前“该 getter 完整逻辑未知”的表述；历史加载源码、未列出的覆写、概率分支、request ID 和实际执行仍未绑定。

该 getter 做的是：

1. 如果缓存的概率或保存的 request ID 列表任一为 None，直接返回 None。
2. 以**保存时**的 request ID 映射到概率第 0 维行号，再按**当前** `input_batch.req_ids` 与 `spec_decode_metadata.num_draft_tokens` 的 zip 顺序逐项处理；draft 数量为 0 的请求直接跳过，不查其缓存。
3. 某个 draft 数量非零的当前请求在保存列表中找不到时，立即返回 **None（整次 getter 的回退）**；不能只丢弃该行或把该位置的概率填零继续。
4. 已找到时按保存行的 `[:num_draft]` 取值，按当前请求顺序把这些片段 `torch.cat(..., dim=0).contiguous()`；所有项均被跳过时也返回 None。

该实现用 Python 字典与 zip，只有在 request IDs、长度和概率缓存满足运行契约时才可以按通常的一一对应理解；源码本身对重复 ID 使用最后一次字典赋值，对 zip 两侧不等长也不会自动完成剩余项校验。不能把这些异常条件臆测为历史状态；提出替代实现时应分别保留或明确其允许域及原行为。之前 proposer 把每步概率 stack 成请求维、step 维、词表维，实际 dtype/shape/值及当前请求顺序仍需逐调用绑定。

**两种纯 CPU 行域反例：**假设保存请求顺序 `[A,B,C]`，缓存每行依次标作 `[A0,A1]`、`[B0,B1]`、`[C0,C1]`；当前顺序 `[C,A,B]`、draft 长度 `[2,1,0]`，getter 输出顺序是 `[C0,C1,A0]`，而按原缓存顺序直接 flatten 为 `[A0,A1,B0]` 会将不同请求/步骤的分布配给验证行。若当前 `[D,A]`、长度 `[0,1]`，缺失 D 被跳过，仍返回 `[A0]`；若当前 `[C,D,A]`、长度 `[1,1,1]`，D 缺失则整体返回 None。这里 A/B/C/D 是构造标签，不是历史请求或概率值。

```python
def assemble(saved_ids, rows, current_ids, lengths):
    lookup = {rid: i for i, rid in enumerate(saved_ids)}
    output = []
    for rid, count in zip(current_ids, lengths):
        if count == 0:
            continue
        if rid not in lookup:
            return None
        output.extend(rows[lookup[rid]][:count])
    return output or None

ids = ['A', 'B', 'C']
rows = [['A0', 'A1'], ['B0', 'B1'], ['C0', 'C1']]
assert assemble(ids, rows, ['C', 'A', 'B'], [2, 1, 0]) == ['C0', 'C1', 'A0']
assert assemble(ids, rows, ['D', 'A'], [0, 1]) == ['A0']
assert assemble(ids, rows, ['C', 'D', 'A'], [1, 1, 1]) is None
```

可以条件地研究把按请求 gather/concat 与 rejection 的概率读取合并：提供证明正确的 request→保存行映射和每请求 step 边界，让验证按原顺序索引缓存，或预先生成完全等价的紧凑行布局。它必须保留原 None/全跳过语义、具体 shape/dtype、概率值、每 request 和 step 的对应、版本/跨调用缓存生命周期、所有读完成事件以及概率/RNG/target 对齐，不能只按 batch 下标或 shape 复用。尤其当前 Ascend `sample/rejection_sampler.py:400–423,558–625` 接受 None 或二维 **contiguous** draft probabilities；随机验证/恢复路径在有无 draft_probs 时选择不同的 `NO_DRAFT_PROBS` / `IS_NGRAM` 分支。若提供非连续 3-D 源加间接索引，需扩展接口或在内核内部等价重构，不能把它直接当作现有 2-D 接口；也不能以各片段可视为 view 推断 `torch.cat` 没有物化或一定存在 HBM 读写。整个分支在 greedy 时可能不使用概率验证，但源码前端仍可能准备 getter，实际如何优化须按具体可见副作用和分支证明。

证据 SHA-256：上游 runner `87c29d08c0bbf66993d8b984811e7325e35e6436242a773feb55ec88f16b2c51`；Ascend runner `568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c`；Ascend rejection sampler `bb1d39de851540c16c2dd2b16df60c60a1fc4957cedb88bdce9417efc0d55eda`；Ascend proposer `458fb50e6839f3a353ddc7181ace4fdae02e35093d8ccd1c4327dc24ae5fff31`。这是源码条件约束，不证明历史服务启用了概率路径、发生请求重排、某次 dtype/归一化、cat 成本或可兑现收益；无 NPU 工作、未改目标实现。

### 全 greedy 时概率缓存的源码条件与例外

保存的 `metadata-path-snapshot/ascend_runner.py:1914–1917` 在每次 `propose_draft_token_ids` 入口清空 runner 的 `_draft_probs` 和对应 request IDs；`draft-constraints-snapshot/ascend_base_proposer.py:1466–1467` 在 `_run_merged_draft` 入口清空 proposer 的 `_last_draft_probs`。若 DSpark 此次所有步骤都走同一 all-greedy metadata，保存的上游 `vllm/v1/spec_decode/llm_base_proposer.py:453–461` 返回 `(argmax,None)`；当 DSpark 使用该概率采样 wrapper 时，只有 step probs 非 None 才加入列表并 stack；若没有启用该路径则直接 argmax。结合 runner 仅在取得非 None 时保存（`:2116–2120`），在这些调用、顺序与版本均成立、没有其他写入或并发覆盖且按对应返回路径完成的条件下，下一次 getter 将因 runner 概率缓存为空直接返回 None，而不会走按 request-ID 的 `cat`。这不证明历史每次都符合这些条件；基准客户端配置 `temperature=0` 不能单独证明运行时所有 `sampling_metadata.all_greedy`、实际调用链或混批状态。

反之，也不能因为 all-greedy 就声称原实现完全没有概率或随机数操作：保存的 Ascend `rejection_sample:450–460` 在 greedy 返回判断前，若启用 entropy verify 且传入 `ori_target_logits`，仍先计算其 softmax；`:486–495` 在 synthetic mode 且存在 greedy 路径时生成 uniform 随机数；`:508–554` 后续才执行 greedy 验证并在 all-greedy 时返回。真实配置与运行分支未知，上述是源码条件，不证明这些计算在该基准中发生，也不能将其当允许融合、换算法、预计算的宽域必要工作。为删改任何实际执行的部分仍须考虑可观察输出、RNG 状态、副作用和异常契约；本节不估算加速或容量。

## 对冻结基线的适用性：概率重排不应当默认列为本负载热点

两轮已保存 benchmark manifest 的 `effective_config.aisbench.model.generation_kwargs` 均是 `temperature=0`、`ignore_eos=true`；`evidence/frozen-serve.sh:28` 请求 DSpark、K=5，其 speculative JSON 没有显式提供 `draft_sample_method` 和 `rejection_sample_method`。保存的 `vllm/config/speculative.py:517,588` 默认分别是 `standard` 和 `greedy`。因此，**如果**运行实例确使用该快照的配置解析与 proposer、上述字段没有其他覆盖，则 `vllm/v1/spec_decode/llm_base_proposer.py:261–264` 的 `_enable_probabilistic_draft_probs` 为 false，Ascend `spec_decode/llm_base_proposer.py:1564–1606` 的 DSpark loop 使用 argmax，不构造并 stack full draft probabilities；`:1466–1467` 对进入该执行函数的每轮重置旧 `_last_draft_probs`。同时 Ascend runner `:1914–1917,2116–2120` 在每次 propose 前清空本轮 `_draft_probs`/IDs，仅在 drafter 真返回非 None 时保存。**在此完整条件下**，下一次进入 `_sample` 时 getter 会被调用，但其空缓存 guard 返回 None，不执行上节的 request-ID 索引、`cat` 和 contiguous 概率重排。因此不能以这条概率重排作为已确认的 `temperature=0` 基线性能瓶颈；它仍是其他概率配置和真实生产条件下的审查候选。

另有一个独立的条件：即使 probabilistic 配置被开启，若**产生缓存的那次 draft batch** 的 `sampling_metadata.all_greedy=True`，且源码分支和缓存重置均如上，则每步 `super()._sample_from_logits` 在 `vllm/v1/spec_decode/llm_base_proposer.py:453–461` 返回 argmax、None，当前 DSpark loop 不积累概率。请求 `temperature=0` 与 `all_greedy` 的关系也有更具体的静态支持：保存的 `structured-request-snapshot/sampling_params.py:27,533–538,781–784` 把小于 `1e-5` 的 temperature 归类为 GREEDY；但是该材料没有完整的服务参数变换与 input-batch 聚合实现、实时 metadata、源码身份及其他并发请求绑定，不能将两个客户端 manifest 自动等同于这次 worker 的 `all_greedy=True`。

**不要把当前验证 batch 的 all-greedy 与缓存是否为空混为一谈。** getter 在验证时按当前 request 与缓存 ID 匹配，但缓存来自前一次 proposer。构造例：上一次是混合 batch，包含 greedy 请求 A 与 random 请求 B，概率模式按源码可能保存两行；之后当前 batch 只剩 A 且 A 有非零 draft 数，getter 仍可取出 A 的概率，返回非 None，即使当前 metadata 的 `all_greedy=True`。此时 `sample/rejection_sampler.py:508–553` 先做 greedy 验证并在随机验证/恢复前返回，故该分支不需要消费 draft 概率；getter 可能已经做了重排。上述不同批次状态是源码可表达的条件例，未证明这次服务实际出现，亦不证明请求采样参数或重排行为会如此变化。

若当前 batch **mixed**，即 `all_random=False` 且 `all_greedy=False`，同一代码会先处理 greedy 行，再继续进入随机路径；不能因出现 `rejection_greedy_sample_triton` 就排除该次随机验证或 draft probabilities 的消费。`entropy_verify` 在 greedy 的提前返回之前可对原 target logits 执行 FP32 softmax；`synthetic_mode` 即使有 greedy 行也可能先生成 uniform 随机数。它们不意味着 full draft probabilities 已生成，也说明不能把“greedy”扩大成所有 softmax、RNG 或验证工作均不存在。metadata 为 None 的 profile/dummy 分支在 `Ascend .../llm_base_proposer.py:1405–1424` 则明确回退 argmax、None。

**现有历史证据边界：**旧 benchmark `service-log-tail.txt:100` 的 `SpeculativeConfig` 摘要列出 dspark 与 5 token，并未列出这两个 sampling 字段；这既不能单独确认默认值已解析，也不能用作诊断进程 resolved 状态。另一个 Oct29 诊断的 `diagnostic-20261029/profile_client.py:34–36` 构造 `temperature=0`，但脚本文件本身不证明历史每次实际请求及其服务实例，不能和旧 benchmark 混为一谈。现有 `agents/diagnosis/DRAFT-EP7-QUERY-METADATA.md` 所查 EP8 两个指定图间隔各有一条导出 `rejection_greedy_sample_triton`：task 名称、stream 和窗口不能绑定到相同请求的 `all_greedy`、proposer 缓存状态、概率张量、None 与否或所有批次；该窗口筛查有限，不能把未列 `cat` 作为不存在。现有 API/trace 摘要中没有已绑定 getter 输入、Python 返回 None 与 request ID/概率存储及 resolved config 的证据；这只描述所查资料，不声称所有未检查 profiler 原始字段必然缺失。最小补证需同进程同一次 proposer→getter→rejection 的配置、采样 batch 标志、缓存状态/请求映射及可关联调用或张量事件；当前不做新采集。

本轮只读复核 SHA-256：冻结启动 `c297201185b5676254af4d7693279f89bfa60f9c76eb340058a78ecf730b7bba`；`speculative.py` `9c642a87264339c0dbc1d0b57330572149df0f9d5e16c6893b9895e50a619e19`；上游 proposer `5bd11c41f0a8cf1272b5880d9878043f876bee6c7e04fb781e613f92930b15ab`；Ascend proposer `458fb50e6839f3a353ddc7181ace4fdae02e35093d8ccd1c4327dc24ae5fff31`；Ascend runner `568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c`；rejection sampler `bb1d39de851540c16c2dd2b16df60c60a1fc4957cedb88bdce9417efc0d55eda`；sampling params 快照 `f21efc98bf8ce8aab88eb67adb4207285e509a167d976e14d3e2e8282e024bea`；旧 benchmark 日志 `64bfa3d8523ad86e1c466fcca166cdcfe1dc4a3676573fc51b65279a69aa8aba`；measure1/measure2 manifest 分别 `b2c55c214bb48d701a3466603c6798f2614c5528ea678dfb9ae280f7fc930c33` / `7689a189df0895ddf219d414981ab080277bcf7e3655f6b2d2c4ef4aa733aca6`；Oct29 profile client `e3528495c48ae02f114dc3943263ba67c6494f795b67db4bfef9079408d9c2a3`；上述 draft 图间隔报告 `00c3ca42fd53fcbd3e5a72caed5375bca9607a429de351fda41ade97427080e6`。无设备访问或目标实现修改。
