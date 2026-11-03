# 融合、拆分与重组候选清单

42个候选族：35个模型相关、7个框架边界相关。全部为CONDITIONAL，未实现为新kernel，也没有优化后实测提速。族内方案可能互斥，不叠加收益。

逐条逻辑、证据、纠正和待办见[进展看板](progress/README.md)；当前讨论的是[F01 EARLY-Y](progress/F01.md)。本表保留历史候选口径，进展页不把登记或源码澄清算作实现完成。

|编号|对象|思路|关键条件|
|---|---|---|---|
|F01|注意力前 mHC 复合边界|现有HC复合算子内部拆分系数和Y分支，提前交付Y，使norm/Q路径与系数计算并行。|已有融合不能重算收益；保留Sinkhorn、系数输出和新交接时序。|
|F02|embedding → 首层 mHC 系数投影；mHC pre → 输入归一化|embedding/HC首层展开、初始化、系数投影共读输入；Y产出后接norm。|保留HC4残差/premix及舍入；首层特例不能推广到后续层。|
|F03|attention 或 FFN mHC post → 下一次 mHC pre，跨边界重组|Post结果分块交给下一Pre/norm，减少残差中间物化。|Engram、aux和其他残差读者保留；不能穿过非线性预乘权重。|
|F04|FFN 归一化与类型转换|在已有RMSNormCast双输出基础上与router/shared/routed输入接合。|BF16与其舍入后FP32扩展值都保留，不重算已有融合。|
|F05|Q norm → 动态量化 → Q 投影，多算子链|组织Q的norm、动态量化和GEMM输入准备，减少中间读写。|整行统计/scale/舍入及Indexer的BF16读者保留。|
|F06|Q 与 KV 投影，共享输入或合并投影|Q/KV共享输入读取，匹配行域时组织多输出投影。|本地Q与全局KV的行域可能不同；保留原有并行。|
|F07|Q 投影 → RoPE，生产者与消费者组合|Q投影epilogue完成规定尾部RoPE。|保持投影舍入、旋转维度及真实head/布局。|
|F08|KV 投影 → norm → RoPE → SWA 缓存写入|KV投影、norm、RoPE后直接写最终cache。|整行归约、invalidslot、pagedlayout和缓存读者保留。|
|F09|Index query 投影 → RoPE → 量化|Indexerquery投影输出接RoPE和量化。|保留query特定scale/舍入，不套用key量化器。|
|F10|Index weight 投影 → 转换与缩放|在Indexweight投影收尾完成转换与缩放。|维持BF16结果->FP32缩放->FP16次序。|
|F11|Index key norm → RoPE → 量化 → key/scale 缓存写入|Indexkeynorm/RoPE/量化直接产出key与scalecache。|持久写供后续层使用，不能删除；key量化规则不变。|
|F12|Indexer TopK → 索引修整 → 共享索引发布；候选块号排序跨层复用或位图 mask|TopK修整/过滤/padding/长度直接发布；候选membership研究位图表示。|位置TopK512与候选块2048是不同状态，保留所有读者与代际。|
|F13|稀疏注意力复合边界|已有稀疏attention内研究SWA提前、压缩分支接入及分块复用。|保留sink/mask/索引重复贡献及softmax/exp/PV数值次序。|
|F14|Attention → 逆 RoPE → 布局恢复 → 输出投影|Attention输出接逆RoPE、布局和输出投影。|跨卡head恢复依赖不能当成本地边。|
|F15|Attention 输出填充/恢复 → 投影|上游直接产通信layout/padding，就绪分块交WO_A。|分块就绪需证据，保留有效前缀/零尾/参与者。|
|F16|两级输出投影与 reduce-scatter 的分块重组|WO_A、WO_B和ReduceScatter分块接力，减少中间物化。|保留两次GEMM和中间/每rankBF16舍入及归约次序。|
|F17|Compressor 两路输入投影，共享转换或分块|在现有一次cast基础上，共读输入并组织Compressor两路投影。|两组权重/输出独立，C2的FP32不能降成BF16。|
|F18|Compressor 状态池化 → norm|pool输出按原舍入接norm，重组ring维护分支。|先读旧ring再覆写；zero/partialrows和完整norm保留。|
|F19|Compressor latent 分叉 → index key 与压缩 KV|同读未旋转latent，分别供Indexkey及压缩KV分支。|另一分支不能消费被RoPE改写的值；两套cache输出保留。|
|F20|MoE router → TopK → dispatch；输入 ID 准备融合与跨层复用|Router收尾/TopK/ID清理打包直接供dispatch，同代ID准备按条件共享。|评分/tie/bias用途/renorm和原始ID读者不变。|
|F21|MoE 内 dispatch、两次分组矩阵计算、激活、combine 的分块流水|审查已有融合MC2内部按expert/tile的数据到达、GMM及回传流水。|保留接收量/credit/flag/clip协议；sender0不等于receiver0。|
|F22|共享 expert 的投影 → 激活量化 → 下投影|共享expert的gate/up收尾接SwiGLU、clamp、量化与down输入。|clamp/全行量化统计/scale/舍入保留，GEMM争用一起计。|
|F23|共享 expert 与 routed expert 汇合 → scaling/add，可延伸至 HC|共享/路由expert就绪tile完成scale/add并接HCPost。|保留两路舍入、残差/系数及真实join依赖。|
|F24|Engram hash/lookup → ready → 行视图或分片；host-UVA 寻址|复用公共history/hash解析，组合NPUhost-UVAlookup/dequant及行组织。|当前overlapFalse；保留slot/owner、完整输出写入与事件代际。|
|F25|Engram 投影 → gate/残差|WKV投影分块接gate/残差，按不变参数复用准备工作和统计。|完整hidden归约/rotation/舍入/mask/NaN及原residual保留。|
|F26|最终 HC collapse → norm 及 logits 边界|HCcollapse/norm接head输入；按完整消费者行集合组织。|rawHC、DSpark、logprobs等读者和舍入都保留。|
|F27|DSpark 辅助上下文投影 → norm，候选流水|辅助HCmean共读输入、多输入投影替代concat，按条件组织aux通信并接norm。|三个checkpoint值不同，未来层未算完不能使用。|
|F28|DSpark embedding → HC 重复/初始化 → decoder|草稿2Dembedding先pad/shard，本地再HC展开和premix初始化。|保留独立可写残差及mask；target已有前置shard不重复计收益。|
|F29|DSpark 上下文 KV 投影 → norm/RoPE → 组装与缓存写入；按请求复用 SWA 索引|共同context共享读取或三路联合投影，各自接norm/RoPE/cache。|三层权重/cache/group/slot独立，共同输入不是共同KV结果。|
|F30|DSpark collapse/norm → Markov bias、logits/置信度/验证|base/Markovbias/Add/后处理/argmax及ID交接组合；研究分片或owner协议。|保留五步回馈、rawconfidence、RNG、score读者和targetverify。|
|F31|Attention/DSpark 相关 metadata 构造、共享与发布|共同坐标/长度/slot一次算多输出，直写最终metadata bank。|query/cache/group/阶段/空queryplan和就绪代际不能混用。|
|F32|SP 输入 gather → Q_a/KV 投影，研究前移投影后按需通信|本地先算可分投影，再只通信消费者需要的投影结果。|保留行映射/权重归属/数值/原hidden读者，不盲拼不同M。|
|F33|C1 compressor 与 SWA KV 的同域投影共享、联合分块与前移通信|C1source层SWA与Compressor共读输入，组织两路投影，可结合F32。|两组512输出分别norm，不能合成一次1024norm。|
|F34|C2 ring 状态的可观测性裁剪、稀疏维护与表示重组|按未来可观察读者裁剪/紧凑维护C2ring，重组owner物理表示。|接受回滚/SWAanchor/owner/ABA代际需证明，不直接承诺容量减半。|
|F35|目标采样的按需raw logits、行选择/转换/argmax与拒绝输出重组|等价argmax/ID转换、行选择/cast/填充/rejection组合，强方案再研究按需head/score。|保留处理后tie/NaN、raw/logprob/entropy读者与RNG；temp0不是score全死。|
|H01|响应等待路径的轮询开销与重复等待工作|合并重复ready/deadline/check记账，研究已有通知/等待组织。|原来已有busy/idle机制；FIFO/fence/cancel保留，采样占比不是可省延迟。|
|H02|输入准备中剩余小拷贝、锁页和元数据工作的合并重组|兼容字段小拷贝/metadata生产合并，复用持久/锁页缓冲，消已证重复生产者。|请求remap/accepted代际、地址/范围/先就绪拷贝及旧读者保留。|
|H03|前缀块哈希的跨请求重复工作|完全相同token前缀与命名空间的跨请求hash链结果复用。|请求内增量hash已有；block/parent/salt/LoRA及失效规则要匹配。|
|H04|前端长提示 tokenization 的精确复用、排队与批处理|完全相同输入与选项复用tokenization；研究已有异步/batch排队。|tokenizer版本/预处理/特殊token/截断/offset和BPE边界保留。|
|H05|前端 DecodeStream 早段长调用与输出就绪流水|decoder准备、按请求串行卸载或语义完成后的输出批处理/发布。|token/stop/EOS/UTF8恢复/异常不变；长wall不是可删CPU。|
|H06|固定 speculative 宽度下的 padding 域拆分与图 bucket 重组|拆分attentionquery域与MoE/EP全局padding域，再研究图bucket组织。|当前uniform6与TP8对齐为24，不能直接改16或删接收工作。|
|H07|原始 token ID 图回放与图外 embedding 的单一生产路径|rawID图内部embedding与外部embedding->inputs_embeds图二选一，避免重复生产。|绑定真实capture/replay入口，保留prompt/MM/eager和Engram原ID读者；方案互斥。|

结构化详情：[records/latest/records.json](records/latest/records.json)。各候选图位于[records/latest/candidates/](records/latest/candidates/)。历史原文中有过期状态和错误修正，以v180当前入口及其条件为准。
