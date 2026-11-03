本轮补齐 F01–F07、F01–F11、F01–F19：HC 与主 Query RoPE、IndexKey、旧 latent 分叉的共同调度及表示选择。继续采用profile优先，只定点核对直接调用。整体目标仍未完成。

960组正Query的eager前沿关联了Q_b、Query RoPE、SWA norm/RoPE/cache写入，共4800个TASK；另复核96组C1/C2前沿的10类旧latent/key/scale/long消费者，共960个既有TASK。它们是固定profile的关联与复核，不是新增设备运行或原生字段关闭。

首先排除了一个假设：Query RoPE并没有等待Indexer准备。源码在第374行做主Query RoPE，第376行才准备indexer；192组对应TASK顺序也支持这一点。真正候选位置是更早的SWA辅助流join：Q_b在第491行完成后，第492行等待aux，函数返回后才做Query RoPE。960组中有880组SWA写入结束晚于Q_b，另外Query M830的40组没有这个正窗口。

在固定Q_b/SWA结束时间、固定RoPE服务时间R、C/S已正确就绪且不增加竞争的三节点参考图中，跨join重排能隐藏的这一部分最多是min(R,max(SWA_end−QB_end,0))。它不是融合或端到端提速的物理上限，也不能跨层/卡/迭代相加。单独前移现有RoPE仍有原GM接口；Q_b epilogue融合则是另一个实现，需保持Q_b的BF16舍入、tail64旋转及BF16舍入、配对坐标与合法handoff，完整512维Query仍被需要，当前路径没有可删除的post-QB RMS512。

若在multistream路径内部把Query RoPE移到SWA join之前，需传入正确的query cos/sin，保持源码要求的“Query读完后才能复用compression源表”关系，并重估两路对Vector与访存的竞争。正Query分支不能代替empty local Query但global cache非空时仍须更新缓存的分支分析。

旧latent的核心约束也更具体：C1/C2产生的BF16 latent512先给IndexWk512→128读取；IndexKey随后做独立128维norm、RoPE、native DynamicQuant、FP16 scale转换及key/scale写入。long-KV则旋转latent512的tail64。只有被覆盖旧值的实际读取需要先完成，不应添加通用“整个Index发布完才能做long RoPE”的barrier。当前96组确实都在两份index store之后才开始long RoPE，并在long store后才开始QLI；这是当前控制顺序。

在实际数据、缓存字节范围、slot和世代条件成立时，QLI需要IndexK及scale，SMLA的压缩分支需要long-KV，两者有不同就绪期限。QLI有条件在long发布完成前推进；SMLA仍须等待自己将读取的long行。所选模型用完整前沿作充分调度，没有声称所有N行或整个算子都是不可拆的必要barrier。

旧值可用以下不同表示处理，不能叠加领取收益：

| long分支表示 | 需要补齐的合同 |
|---|---|
| 原位旋转 | 重叠tail旧读先于覆盖，可按行/范围做有证据的流水 |
| 完整512维out-of-place结果 | 保存448维原前缀与64维旋转尾部，另计完整buffer及前缀读写 |
| 仅64维旋转尾部buffer | 新writer读取旧latent前缀和独立tail；旧前缀需保留到该reader结束 |
| 保存旧64维tail再原位旋转 | 新IndexWk读取未变前缀和旧tail，保留原K/累加顺序；原连续GEMM不能直接消费两份地址 |
| 只读旧latent的直接RoPE-cache writer | writer内旋转tail并写cache，旧latent不变，可与IndexWk作只读分叉；需关闭其它后旋转latent观察者并验证新writer |

对N行512维BF16 latent，完整旋转buffer内容为1024N B，tail-only为128N B；若完整out-of-place实现显式搬运未变前缀，相对tail-only旋转多1792N B的前缀值读写。保存旧tail另有256N B copy值读写。直接long-cache writer的tail私有接口消除参考为256N B。这些是声明图的逻辑内容，不是额外分配或HBM流量下界；普通单一连续512维Scatter或GEMM不能未经新接口就消费分离的前缀与tail。

IndexKey norm/RoPE/quant/cache也可比较融合，但须保留512 latent norm与128 key norm的不同值和参数、BF16边界、native key量化的实际FP32 scale/量化政策、最终FP16 scale及定义内的invalid/duplicate-slot语义。Query量化先用FP16 scale再量化，不能直接替代key DynamicQuant。IndexKey128和long512虽共享source C/S表达式与tail64宽度，旋转的值却不同，不能复用旋转结果；C2 source positions也不等于当前Query/SWA positions。

新的分叉还改变生命周期。旧latent要保留到IndexWk以及direct/segmented writer的实际读取结束；source C/S要等IndexRoPE与LongRoPE等所有读者，不能在第一个分支结束时就复用。缓存新世代在相应QLI/SMLA读取前可见，复用页和buffer仍须等待最后读者及排队写者。HC原始X继续供当前系数和Post残差，pre4仍供下一FFN，C2 ring未来状态也不会因这次分叉自动退休。

2304个Query参考DAG包含HC提前Y、RoPE跨SWA join和竞争重估；给定假设服务中1428组较快、494组较慢、382组相同。1536个latent参考DAG包含延后long join、旧读后分叉、只读direct与保存旧tail版本；只读direct有964组较快、192组较慢、380组相同。这些仅说明新buffer、writer、队列和资源代价可以改变结论，不是实测胜率、native精度或最优时间。模型终点是HCPost输入就绪，共同Post服务与下一pre4/FFN未计价。

新增REL140-01/02/03，条件关系150→153、待核对445→442；HG93加入F07、F11、F19，联合组仍26，全部子集和native联合合法性保持开放。35主候选、7历史框架记录、358语义行/26模块、4472次固定SP关联和854节点/1346边/48提示的部分图保留。9类/1352次原生证据、849个PMU缺口、16个phase未绑定图、3136项归属歧义未补齐，整模块关闭0，全部模块的物理MFU和最低时间仍未建立。
