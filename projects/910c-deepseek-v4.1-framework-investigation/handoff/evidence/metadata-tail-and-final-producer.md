# metadata末读与末层producer：aux完整接线更正

**Goal ACTIVE；loop-015 PIVOTED（限定源码/时间证据，metadata方案仍待确认）；当前loop-016 EVALUATING。全链路未扫全，每个替换原语逐位一致。

**更正/撤回**：先前只用model的`layer_idx+1`推导aux，遗漏NPU runner先给ID加1。撤回aux36/37/38与两个body都可删的判断；两层3.8115%及1.77169%情景不能用作候选收益。原记录保留并附失效说明，metadata末读结论不受影响。** 本轮新NPU0、模型加载0、服务推理0、新profile0、远端写入0。

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

### 完整aux配置接线与实际行需求

[完整调用链](aux-config-call-chain.json)执行了两份pinned selector方法，未导入vLLM/Torch：冻结配置没有上游eagle/dflash覆盖，因此base selector返回None；NPU runner:787–802将`dspark_target_layer_ids=[37,38,39]`加1，得到38/39/40。GPU setup:5497–5516调用这个virtual selector，再调用model setter；EagleModelMixin:1577–1581仅排序去重，model:1560再判断layer_idx+1。故实际采集于零基**37/38/39入口**，只有**39层body**在最后aux之后。

当前Docker日志未找到INFO记录，缺日志不作为接线证明。本结论是冻结配置与实际源码的确定性推导，不冒充live对象dump。早期拓扑硬编码37/38/39恰好正确，本轮第一次改成36/37/38是错误修正；[当前拓扑](target-layer-plan-corrected.json)已改成沿runner转换推导。

39层不是long-KV/index-K source，读long-KV20/topK36；其SWA128仍有当前/未来读者。CP preprocess明确分开query_hidden与cache_hidden。连续因果、每请求只选末token、无prompt-logprobs/其它观察者时，39层query可从一行反推所需SWA末128行；仍需物理slot/淘汰/chunk/prefix共享/推测拒绝与补写证明，不能直接删其它KV写。

**38层全行输出mean供39入口aux使用**，所以不能把38层query/output/FFN缩到末128行，更不能借此给出末255行写集合。跨38/39仍可研究“38的HCPost与完整aux-mean物化交付”融合，但必须计算全部aux行、保留原BF16结果后再按原mean归约；它是不同机制，不能认领整层38计算删除的预算。

## 结果

| 范围 | 状态 | 本轮判断 |
|---|---|---|
| metadata共享frame直接依赖release复用 | 已停止该原样协议 | 有合法QLI producer未被consumer等待的反例；需独立本代完成条件。 |
| metadata共享+producer_done/alias协议 | 待确认，值得开发 | 72窗尾部支持normal join未暴露device等待；新增API/host、alias、逐位和graph/error仍缺。 |
| aux36/37/38及两个body可按行裁剪 | **已撤回/已停止该方案** | 遗漏runner ID转换；38层全行输出有aux39读者。先前source/value review由后续证据声明失效，原Run不可变记录保留。 |
| 第39层query/output/FFN与SWA联合生产 | 待确认，值得深入 | 只有body39在最后aux之后；物理SWA写域、CP-local topK36映射与逐位数值几何仍未完成。 |
| 38 HCPost→完整aux-mean→39输入交付 | 待确认，独立范围 | 保留全部aux计算和每个原语rounding；不能用被撤回的两body全免价格。 |

[当前价格](final-producer-entry.json)只为第39层候选计价，绑定原C8 **56个real eager rank-body/head窗口**；8个idle DP窗口无收益。单个请求自身首eager贡献的P90全免仅0.6741–0.6767%，这个窄口径不能剪全部prefill传播。独立最有利DP、所有inclusive eager波次串行传播下，第39层→head-input全免包络为34.73991ms、P90 **2.32864%**，保留head GEMM、verify和DSpark。

粗净模型为每请求`rho×39层inclusive包络−h×受影响eager波数`，rho包含实际可删且暴露的比例。h=.5ms/wave时，rho0为-2ms；rho.25约净6.68498ms、P90 .44810%；rho.5约净15.36995ms、1.03026%。都是有利敏感性，不是实测或expected加速；必要SWA/存活行/数值几何决定实际rho。1%仍只是排序参考，精度门槛不随收益放宽。

[被撤回的两层价格](withdrawn-two-layer-price.json)仅保留错误推导的反事实数据和原commit，**不再进入当前候选收益**。原compact FFN曾有原语位差，即使最终norm相同也继续停止该配方。

8-rank误用16-rank旧helper、一次EOF peek和selector AST漏future-annotations的reader错误均保留说明。Taskctl拒绝覆盖已结束Run或已有knowledge delta，故以新Run、显式review-invalidation与superseding knowledge记录更正，不抹去原记录。均非候选native测试。

**下一段**：追第39层SWA128物理writer/page/slot及未来读者、selected-query到topK36原CP-local行映射，保留38输出的完整aux39；拆必需cache与可删query/O/FFN，再核保持原M/tiler/rounding的实现和净成本。metadata缺口已留协议，主图继续，原两层裁剪与两层预算均停止使用。
