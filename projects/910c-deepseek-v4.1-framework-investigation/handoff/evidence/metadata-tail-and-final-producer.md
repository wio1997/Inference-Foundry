# 末读完成条件与最后两层 producer 入口

**Goal ACTIVE；loop-015 PIVOTED（限定源码/时间证据，metadata方案仍待确认）；当前loop-016 EVALUATING。全链路未扫全，每个替换原语逐位一致。** 本轮新NPU0、模型加载0、服务推理0、新profile0、远端写入0。

## 介绍

先沿metadata实际consumer补足共享frame的退役条件，再从最后可见输出反推更大producer。只读当前冻结源码和原C8时间账；已有方案与未消费行必须分别证明，不能用最终输出相同替代原语逐位，也不能用一个小范围的价格剪更大的交付。

## 到了哪里

### V1 metadata实际末读

[16源身份](metadata-readers-execution-identity.json)包含实际indexer、CP与模型；[读者前沿](metadata-reader-frontier.json)列出具体行段。引用扫描仅用于定位，不是穷尽覆盖证明。

- C2 source writer在 `dsa_v41.py:523–550` 等COMPRESSOR后读取pool/state/cos；它在较早stage，不能证明INDEXER尾部完成。
- 非空query的SMLA在653–681等待ATTENTION，并读qsl/seq/物理表。model.forward:704–716的空query分支跳过该consumer，仍参加CP输出exchange和O投影的形状协议。
- 实际A3 indexer在273等INDEXER，但 `query.shape[0]==0` 或 `max_cache_seq_len==0` 在此前返回；non-index-source复用旧topK。相对地，A3 `build_qli_metadata:1253–1271` 没有空query early return。因此 `release` 不是全提交任务已完成的证明，不能靠“所有consumer通常都等了”允许新frame覆写输入。

[原C8定点读取](metadata-producer-tail-receipt.json)保留最初62个完整3SMLA+2QLI匹配、10个未匹配记录。随后按源码与原HcPre几何分类：小decode在跨DP降级eager后仍保留24行桶，每卡3行；DP0 actual6使TP2–7无query，DP1 actual12使TP4–7无query。10个窗口恰为2QLI、无3个native SMLA，符合SMLA zero-fill但QLI仍运行的实现，未改原始回执/原始字节以强行通过。

合计72个source-qualified窗口的最后QLI→copy→ready marker均早于body开始，最小2.68574ms；早于body结束最小46.48836ms。eager标记为EVENT RECORD，FULL为MEM_WRITE_VALUE。[逐窗归类](metadata-reader-frontier.json)。这是source+唯一顺序motif，缺event handle/张量地址，不是事件身份或原生集成证明；没有对六普通负载、异常或新栅栏开销作外推。

更大共享frame应在全部producer/result-copy入队后记录本代 `submission_done`，在normal release/下一代device覆写前join；CPU H2D lease、metadata输出复用与data-ready继续分开。部分提交失败时传播原错误并隔离frame到实际完成或重建，不能把release当可复用。现有72窗没有暴露新增device等待，但新增record/wait/API成本和其它负载仍计入h。metadata路线保留**待确认、值得开发**，留下alias/原语逐位/graph/error/成本缺口，不再重复旧CPU-plan或自动快照测试。

### 从最后aux向前扩边界

当前配置 `dspark_target_layer_ids=[37,38,39]`，实际 `layer.layer_idx+1` 判断意味着aux采集于零基36/37/38层**入口**。修正了早期[拓扑导航](target-layer-plan-corrected.json)中硬编码aux_before_layer=37/38/39的偏移；不是重新证明全模型数值。

所以最后一份aux在38层body之前已经产生，**38和39两个body**都可纳入后续行需求分析。两层都不是long-KV/index-K source，读取long-KV20和topK36；各自SWA128仍有当前/未来读者。CP preprocess已明确分开query_hidden与cache_hidden；CP非index-source直接按本地原query行取topK36前缀，缩小query集合时必须显式重映射原CP-local行，不能把选择后的行号当旧前缀行号。

在“每请求仅末token一个query、连续因果位置、无prompt-logprobs/其它观察者”的简化场景中：39层query1行需SWA末128行；因此38层输出/query需末128行，38层SWA KV union可能为末255行（原left=127）。这只是反推逻辑域，**不是已证明的物理写集合**。还需检查chunk边界、旧cache、页/slot别名、淘汰、prefix共享、推测拒绝/补写和同一步所有query。完整prompt-logprobs仍保留完整prompt行；DSpark aux38入口仍完整。所有保留HC/normalization/router/GMM/Combine/Post的替换值必须逐位；原样compact FFN已存在位差，不能重试到最终norm相同就放行。

## 结果

| 范围 | 状态 | 本轮判断 |
|---|---|---|
| metadata共享frame直接依赖release复用 | 已停止该原样协议 | 有合法QLI producer未被consumer等待的源码反例；需独立本代完成条件。 |
| metadata共享+producer_done/alias协议 | 待确认，值得开发 | 72窗尾部时间支持normal join可能不暴露device等待；新API/host开销、全alias、原语逐位及graph/error集成仍缺。 |
| 只按单个首输出贡献筛整个末层producer | 已停止该筛法 | 口径过窄。必须覆盖会一起改变的先前prefill波次和更早producer。 |
| aux38之后两个body的query/output/FFN与cache联合生产 | 待确认，值得深入 | 边界已扩到38/39；物理SWA与精确保留域未完成，不能删全KV或按任意新M直接重算。 |

[末端入口及价格](final-producer-entry.json)绑定原C8 **56个real eager rank-body/head窗口**，不为8个idle DP窗口认领收益。保存8-rank clock约束和每个task边界：

- 只认各请求自身首eager一次，完整39层→head-input全免，P90条件0.6741–0.6767%；这个窄口径不剪更大parent。
- 单39层按独立最有利DP、全部inclusive eager波次串行传播，P90组件包络34.73991ms、2.32864%。
- 扩到最后aux之后38/39两层同口径全免，P90组件包络56.86208ms、3.81150%。保留head GEMM、verify与DSpark；所有必要SWA、存活query/FFN和替代开销还未扣除。

粗净模型为每请求 `rho×两层inclusive包络 − h×受影响eager波数`，rho含实际可删且暴露比例。h=.5ms/wave时，rho0保留-2ms回退；rho.25约净12.21552ms、P90 .81881%；rho.5约净26.43104ms、1.77169%。这是有利敏感性，不是expected/matched实测加速；实际rho取决于物理cache写域与逐位数值几何。1%仅历史排序参考，精度门槛不随收益放宽。

价格reader首版误把8-rank DP子集传给硬编码16-rank的旧helper，在生成结果前IndexError，error Run保留；随后显式按8-rank矩阵计算。源码peek一次越过298行EOF也保留探索说明。均无NPU或候选正确性结果。

**下一段**：追38/39 SWA128物理writer/page/slot和selected-query/topK36映射，把必须保留的cache生产从可删的query/O/FFN中剥离，再确定保持原数值几何的具体实现及净成本。metadata开发缺口已留具体协议，继续主图；不回到原样compact FFN或用旧窄尾段预算结束整个两层范围。
