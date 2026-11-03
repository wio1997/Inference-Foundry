# Key缓存、MoE汇合与输出布局：Profile分析 v5

本轮先遍历全部1280个已存target scope，再核对原始EP3数据库的明确端点、提交API、完整栈和通信连接。只有F23的返回/归约分支不能由profile定案，才定点读3个冻结源码文件的相关方法。没有新增推理或设备采集。

## F11：缓存任务的调用位置已补强

128个KV-source scope均各有一个相容key投影、128维norm、RoPE、DynamicQuant、INT8 key scatter和FP16 scale scatter。物理key行数为4760/5144/8180，分别32/32/64组；它们不是已证明的有效压缩行数。

所选EP3/L2/scope `2128-8` 的5个明确TASK端点，经connection映射绑定到：

|任务|提交API|栈中位置|
|---|---|---|
|key投影|aclnnMatmul|indexer.update_keys:110|
|key归一化|aclnnRmsNorm|indexer.update_keys:110|
|key量化|aclnnDynamicQuantV2|indexer.update_keys:119|
|key cache写入|aclnnScatterNdUpdateSk|update_keys:120 → scatter_cache_sk:242|
|scale cache写入|aclnnScatterNdUpdateSk|update_keys:121 → scatter_cache_sk:242|

这是实际提交位置的证据，强于同名/同形猜测；没有证明tensor/slot身份，也不把所选5条栈推广为全部128组已逐条绑定。RoPE仍是同scope/stream的相容候选。

进一步融合继续保留native quant的数值契约、FP32→FP16 scale边界、key/scale两份持久输出、真实page stride及无效槽规则。重复slot情况下，等待两次写完与两张cache选择相同赢家是不同条件；不能未经原允许结果集合和实际索引证据，就把成对原子发布或只写一张cache视作等价。QLI的读取版本及跨步生命周期仍是缺口。

## F23：所选样本应采用runner内合并图

1280个scope都有一个`ForeachAddList`和两个`HcPost`候选。foreach到末次HC的导出间隔中位20ns、最大23.640µs；这些数字不代表启动开销或可节省时间。

EP3/L36的merge task41461明确关联`aclnnForeachAddListV2`、`AscendMoERunner.forward:240`；HC task41462关联`model.py:919`。进一步读未过滤栈顶部，runner与模型MoE调用之间只有PyTorch的`_call_impl/_wrapped_call_impl`，没有另一层forward包装。该记录的源码解释因此需要采用以下图：

`routed/shared准备与归约 → runner缩放/输出transform → foreach_add → final reduction/truncation → 可选zero-expert输出 → Tensor返回 → 模型Tensor分支 → HC post`

冻结源码中runner的返回及其两个后置helper均保持Tensor；模型`isinstance(...,tuple)`为另一条legacy路径。不能再把外层legacy `muls_add_triton`及其缩放公式直接套给这个已定位的runner返回路径。旧tuple分析保留为条件分支，不覆盖所选活动图。

相关源码SHA已与源台账匹配：Ascend fused_moe为`2f42ce56…`，upstream runner为`05893188…`，模型为`101f96b3…`。这证明冻结实现的分支合同，单独不构成历史加载对象hash证明。

当前候选若吸收foreach和HC，必须模拟原add的输出dtype舍入，再进入HC的FP32运算及原累加顺序；合并后的归约、截断、zero-expert或transform存在时也必须包括。实际runner scalar值没有从profile绑定，不能从配置中的1.5直接推断此处又乘一次。

`_fused_output_is_reduced`依赖当前MoECommType；Ascend的最终归约helper明确避免把capture时动态布尔值当所有replay的事实，改用静态层策略及context-aware op。重组需保持这种跨迭代语义，也不能因Tensor返回就断言所有通信都不存在。

## F14/F15：侧车漏掉了真实存在的通信边

1280个scope的attention结束到第一输出投影间，均有一个相容RoPE及一个Transpose，但侧车没有匹配到通信。原始DB在所选EP3/L36区间补到一条rankSize=8的`hcom_alltoall__503_247_1`。

这次又补了明确连接：host `HcclAlltoAll` API connection431394 → device connection102545 → COMMUNICATION_OP opId247，并有`dsa_v41_cp.py:215 _project_output`调用栈。因此这个样本的零通信计数是提取关联缺口，不是可删通信或通信成本为零的证据。连接身份只在该DB/scope内成立，未推广成跨rank同一物理消息证明。

保留TP行/head交换、valid rows、两类padding、metadata及graph稳定输出合同后，可联合研究inverse-RoPE与send-layout写入；它属于F14/F15的重叠变体，不能计两份收益。

**条件逻辑访问账**：所选局部输出为1023×64×512个BF16值，RoPE尾维64。整tensor为67,043,328字节，旋转tail为8,380,416字节。在“RoPE仅读写tail、Transpose完整物化”的模型中，若必须保留原in-place状态，联合写布局最多讨论减少一次额外tail读；若原中间量确实不再可观察，才可讨论额外tail读写合计16,760,832字节。不能把整个tensor的读写都算成RoPE融合可省，也不把逻辑接口账当HBM流量或实际提速。

## 状态与下一步

本轮F11/F14/F15/F23纳入细粒度profile和局部合同，当前分级轮次还有9项待继续对账：F02/F03/F21/F24/F25/F26/F27/F28/F29。这个数量不是完成比例；已经查看的候选也保留数值、所有消费者、runtime alias、跨步状态及资源证据缺口。

后续先把mHC post→pre、尾部和draft初始化的既存profile放在一起核查，再继续Engram、DSpark与MC2资料。所有32候选和26模块范围仍保留，0个整模块关闭，整体目标未完成。
