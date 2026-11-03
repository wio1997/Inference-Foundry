**当前整体分析：继续推进，完整范围尚未通过验收。**

本页汇总当前有效结论，替换旧报告中的过期状态。指定模型仍为DeepSeek V4.1 Flash目标网络、DSpark与Engram，冻结负载32K输入/1K输出、C8、P90；环境已迁移到171，CANN9.2。驱动/固件差异按用户接受的范围保留。各采集cohort单列，旧硬件/旧epoch不回填为新观测。

登记范围含26个功能模块、360条语义行、35个模型候选族和7个框架边界候选族。这些是分析对象，不等于42项可直接实施或可叠加的融合收益。现有F21已融合dispatch、量化、双GMM、SwiGLU及return；当前greedy验证也已有Triton实现。

已有的1101处精确静态点完成算子效应审查；全部42个登记族有组合约束，27份源码模板的854节点/1346边完成一轮7规则条件审查。完整模型范围仍需把精确条件集合与这些答案对齐，并核齐分层G0的消费者、别名、状态与跨owner端口。

| 要求 | 当前状态 |
|---|---|
|MODEL_VERSION_HARDWARE_LOAD|FIXED_SOURCE_AND_MIGRATED_CASE_DOMAIN_WITH_EXPLICIT_NATIVE_MODE_UNKNOWN|
|ALL_MODULE_CONDITIONAL_BRANCHES|INCOMPLETE_EXACT_GUARD_TO_EXISTING_OPERATOR_CONTRACT_RECONCILIATION|
|CURRENT_FINITE_GRAPH_LEDGER|INCOMPLETE_HIERARCHICAL_G0_INTERFACE_AND_ALIAS_PORT_RECONCILIATION|
|CROSS_OPERATOR_LAYER_CARD_ITERATION|REGISTERED42_COMPOSITION_AND_DECLARED_SOURCE_RULE_ROUND_ANSWERED;FULL_SCOPE_DEPENDS_ON_G0|
|NATIVE_OPERATOR_REGIMES|CONDITIONAL_NATIVE_ELIGIBILITY_AND_UNSUPPORTED_PRIVATE_EDITS_EXPLICIT|
|THEORY_AND_MEASURED_COMPARISON|PROTOCOL_P5_EXPLICIT_CONDITIONAL_SYMBOLIC_COMPARISONS_PRESENT|
|LOWEST_MODULE_UTILIZATION|PROTOCOL_P4_COMPARABLE_MEASURED_GROUP_MINIMA_ESTABLISHED|
|CAUSES_AND_CRITICAL_PATH|PROTOCOL_P6_SUPPORTED_LOCATION_AND_COMPETING_CAUSES_WITH_LIMITS|
|CRITICAL_EVIDENCE_AND_FINAL_CLOSURE|NOT_ACHIEVED_TWO_EXACT_SOURCE_COVERAGE_CHECKS_REMAIN|

本轮已用profile调用与已加载类固定72份源码、573个选中函数及1867个条件引用，缺失/哈希不匹配为0。67份已有profile源码哈希匹配，另5份为新绑定的已加载类文件。条件数量不表示同等数量的新优化任务；先复用已有合同，只分析未匹配的算子效应。同名条件定义、属性访问器和传给executor的嵌套回调均按源码位置保留。

已登记的候选族如下；具体条件、必要工作、候选图和未知项由当前台账及组合合同给出。

| 编号 | 对象 | 结论范围 |
|---|---|---|
|F01|注意力前 mHC 复合边界|条件分析；实际适用性与收益分别核对|
|F02|embedding → 首层 mHC 系数投影；mHC pre → 输入归一化|条件分析；实际适用性与收益分别核对|
|F03|attention 或 FFN mHC post → 下一次 mHC pre，跨边界重组|条件分析；实际适用性与收益分别核对|
|F04|FFN 归一化与类型转换|条件分析；实际适用性与收益分别核对|
|F05|Q norm → 动态量化 → Q 投影，多算子链|条件分析；实际适用性与收益分别核对|
|F06|Q 与 KV 投影，共享输入或合并投影|条件分析；实际适用性与收益分别核对|
|F07|Q 投影 → RoPE，生产者与消费者组合|条件分析；实际适用性与收益分别核对|
|F08|KV 投影 → norm → RoPE → SWA 缓存写入|条件分析；实际适用性与收益分别核对|
|F09|Index query 投影 → RoPE → 量化|条件分析；实际适用性与收益分别核对|
|F10|Index weight 投影 → 转换与缩放|条件分析；实际适用性与收益分别核对|
|F11|Index key norm → RoPE → 量化 → key/scale 缓存写入|条件分析；实际适用性与收益分别核对|
|F12|Indexer TopK → 索引修整 → 共享索引发布；候选块号排序跨层复用或位图 mask|条件分析；实际适用性与收益分别核对|
|F13|稀疏注意力复合边界|条件分析；实际适用性与收益分别核对|
|F14|Attention → 逆 RoPE → 布局恢复 → 输出投影|条件分析；实际适用性与收益分别核对|
|F15|Attention 输出填充/恢复 → 投影|条件分析；实际适用性与收益分别核对|
|F16|两级输出投影与 reduce-scatter 的分块重组|条件分析；实际适用性与收益分别核对|
|F17|Compressor 两路输入投影，共享转换或分块|条件分析；实际适用性与收益分别核对|
|F18|Compressor 状态池化 → norm|条件分析；实际适用性与收益分别核对|
|F19|Compressor latent 分叉 → index key 与压缩 KV|条件分析；实际适用性与收益分别核对|
|F20|MoE router → TopK → dispatch；输入 ID 准备融合与跨层复用|条件分析；实际适用性与收益分别核对|
|F21|MoE 内 dispatch、两次分组矩阵计算、激活、combine 的分块流水|已有融合；内部拆分、重排与载荷分配为条件候选|
|F22|共享 expert 的投影 → 激活量化 → 下投影|条件分析；实际适用性与收益分别核对|
|F23|共享 expert 与 routed expert 汇合 → scaling/add，可延伸至 HC|条件分析；实际适用性与收益分别核对|
|F24|Engram hash/lookup → ready → 行视图或分片；host-UVA 寻址|条件分析；实际适用性与收益分别核对|
|F25|Engram 投影 → gate/残差|条件分析；实际适用性与收益分别核对|
|F26|最终 HC collapse → norm 及 logits 边界|条件分析；实际适用性与收益分别核对|
|F27|DSpark 辅助上下文投影 → norm，候选流水|条件分析；实际适用性与收益分别核对|
|F28|DSpark embedding → HC 重复/初始化 → decoder|条件分析；实际适用性与收益分别核对|
|F29|DSpark 上下文 KV 投影 → norm/RoPE → 组装与缓存写入；按请求复用 SWA 索引|条件分析；实际适用性与收益分别核对|
|F30|DSpark collapse/norm → Markov bias、logits/置信度/验证|条件分析；实际适用性与收益分别核对|
|F31|Attention/DSpark 相关 metadata 构造、共享与发布|条件分析；实际适用性与收益分别核对|
|F32|SP 输入 gather → Q_a/KV 投影，研究前移投影后按需通信|条件分析；实际适用性与收益分别核对|
|F33|C1 compressor 与 SWA KV 的同域投影共享、联合分块与前移通信|条件分析；实际适用性与收益分别核对|
|F34|C2 ring 状态的可观测性裁剪、稀疏维护与表示重组|条件分析；实际适用性与收益分别核对|
|F35|目标采样的按需raw logits、行选择/转换/argmax与拒绝输出重组|条件分析；实际适用性与收益分别核对|
|H01|响应等待路径的轮询开销与重复等待工作|条件分析；实际适用性与收益分别核对|
|H02|输入准备中剩余小拷贝、锁页和元数据工作的合并重组|条件分析；实际适用性与收益分别核对|
|H03|前缀块哈希的跨请求重复工作|条件分析；实际适用性与收益分别核对|
|H04|前端长提示 tokenization 的精确复用、排队与批处理|条件分析；实际适用性与收益分别核对|
|H05|前端 DecodeStream 早段长调用与输出就绪流水|条件分析；实际适用性与收益分别核对|
|H06|固定 speculative 宽度下的 padding 域拆分与图 bucket 重组|条件分析；实际适用性与收益分别核对|
|H07|原始 token ID 图回放与图外 embedding 的单一生产路径|条件分析；实际适用性与收益分别核对|

模块最低者已有同口径结果：12个带MAC活动模块中M12最低，合并报告AIC/MAC周期加权比为2.41549%，两个正式窗口分别为2.55250%和1.17269%，对已记录的归属歧义仍稳健。20个带Vector活动模块中M23算术活动最低，合并值约0.01791%；它承担通信，不据此判定无用。

这些是报告计数器活动比例，不能称为物理MFU、整卡忙时或可移除比例。未测host、3510条缺PMU、87495条归属歧义和未记录工作明确保留，不能填0或强行与不同资源排名。

| 模块 | 功能 | 报告MAC活动 | 报告Vector活动 |
|---|---|---:|---:|
|M00|注册、平台替换、模型与算子选择|UNKNOWN/不适用|UNKNOWN/不适用|
|M01|请求、tokenization与采样契约|UNKNOWN/不适用|UNKNOWN/不适用|
|M02|前缀复用、KV分配与回收|UNKNOWN/不适用|UNKNOWN/不适用|
|M03|EngineCore、调度与异步队列|UNKNOWN/不适用|UNKNOWN/不适用|
|M04|worker输入准备与持久batch状态|UNKNOWN/不适用|0.63818%|
|M05|embedding、模型wrapper和图像条件|UNKNOWN/不适用|14.86994%|
|M06|mHC初始化、pre/post与跨层残差|18.55886%|47.76641%|
|M07|Q/KV投影、norm、量化与RoPE|43.77610%|38.02437%|
|M08|compressor、ring状态与latent分叉|85.31651%|19.20898%|
|M09|Indexer评分、候选与跨层索引|17.25753%|65.25212%|
|M10|SWA、压缩KV与稀疏attention|30.96945%|20.84985%|
|M11|attention输出、head恢复、两级投影与归约|74.46228%|1.40725%|
|M12|MoE路由、dispatch、专家计算和combine|2.41549%|0.68474%|
|M13|共享专家、路由分支汇合与scaling|51.68150%|42.40911%|
|M14|Engram hash、host/UVA lookup、投影与门控|61.92930%|7.35831%|
|M15|最终mHC collapse、norm及MTP辅助状态|UNKNOWN/不适用|18.08709%|
|M16|目标/草稿词表投影与跨卡重建|11.79996%|0.18673%|
|M17|目标采样、拒绝/接受、bonus与logprobs|UNKNOWN/不适用|9.80047%|
|M18|DSpark上下文投影、共享KV及输入发布|69.20902%|8.90436%|
|M19|DSpark decoder、mHC及共享计算|UNKNOWN/不适用|1.62276%|
|M20|DSpark base logits、Markov和confidence|20.28561%|7.01101%|
|M21|draft/target metadata、slot与索引准备|UNKNOWN/不适用|1.38304%|
|M22|graph捕获、replay、更新与多流生命周期|UNKNOWN/不适用|UNKNOWN/不适用|
|M23|DP/TP/EP/PCP通信与分区所有权|UNKNOWN/不适用|0.01791%|
|M24|跨迭代状态、接受回滚、预取和重用|UNKNOWN/不适用|1.03496%|
|M25|采样结果回传、响应、流式输出与终止|UNKNOWN/不适用|UNKNOWN/不适用|

M12的低活动与主要累计kernel时间位于F21内部：两个窗口中F21占M12累计native时间98.7649%/99.1925%，占报告AIC周期99.7213%/99.9279%。这排除了“只需把外围小算子合并”的解释。小receiver工作、专家/rank不均和跨卡到达/credit/polling是有源码或跨cohort机制支持的竞争原因；当前PMU不能独自分解其因果比例，累计跨卡kernel时间也不是请求walltime。

理论对照现有94条当前规范记录：26个模块合同、42个未实现组合、26组匹配的局部稠密模式。各对象均显式给出Q、T、L、Q/T、U、L/T、T/L、活动指标、支配约束和证据状态；缺值保留条件/符号/UNKNOWN。旧76条对照独立保存在历史快照，不能与当前cohort直接合并。

Cube直接稠密域的形式为L=Q_MAC/(24·r_mode·f_upper)，INT8的r=8192、BF16/FP16的r=4096、受限FLOAT模式按各自1024/2048条件。1800MHz是额定参考，未当物理fmax。整体融合按必要且不重复的计算、逐层必要I/O、共享资源/通信cut和已证明依赖重算，各独立资源下界取max。权重/缓存/workspace分配大小不自动等于每次HBM或wire流量。

部分可复算对照如下，T为同一所选执行集合的累计kernel服务时间，跨设备累计，不能当请求延迟；FLOAT模式行是互斥假设，不能相加。参考L/T不是设备活动率或可达提速。

| 对象/窗口/模式 | 必要MAC（条件域） | 匹配累计T(ms) | 1800MHz参考L/T |
|---|---:|---:|---:|
|M07/direct-dense/20261102152817/BF16|6022719078400|81.306645|41.86228%|
|M07/direct-dense/20261102152817/INT8|38509163315200|286.876942|37.93104%|
|M07/direct-dense/20261102152835/BF16|3303014400|2.311586|0.80753%|
|M07/direct-dense/20261102152835/INT8|15099494400|5.238709|0.81445%|
|M08/direct-dense/20261102152817/FLOAT_HF32_OFF|2887526645760|78.178785|83.49364%|
|M08/direct-dense/20261102152817/FLOAT_HF32_ON|2887526645760|78.178785|41.74682%|
|M08/direct-dense/20261102152817/BF16|481254440960|5.000115|54.39401%|
|M09/direct-dense/20261102152817/INT8|962508881920|11.243323|24.19003%|
|M09/direct-dense/20261102152817/BF16|78203846656|6.777423|6.52108%|
|M11/direct-dense/20261102152817/BF16|38535126056960|290.523252|74.96045%|
|M11/direct-dense/20261102152835/BF16|15099494400|2.247910|3.79612%|
|M12/direct-dense/20261102152817/FLOAT_HF32_OFF|1806544404480|72.263908|56.51237%|
|M12/direct-dense/20261102152817/FLOAT_HF32_ON|1806544404480|72.263908|28.25618%|
|M12/direct-dense/20261102152835/FLOAT_HF32_OFF|283115520|2.255051|0.28381%|
|M12/direct-dense/20261102152835/FLOAT_HF32_ON|283115520|2.255051|0.14190%|
|M13/direct-dense/20261102152817/INT8|32527425208320|208.943583|43.98929%|
|M13/direct-dense/20261102152835/INT8|15288238080|6.849743|0.63068%|
|M14/direct-dense/20261102152817/BF16|7225108070400|72.811097|56.07936%|
|M14/direct-dense/20261102152817/FLOAT_HF32_OFF|30104616960|2.284089|29.79452%|
|M14/direct-dense/20261102152817/FLOAT_HF32_ON|30104616960|2.284089|14.89726%|
|M16/direct-dense/20261102152817/BF16|101934694400|20.262395|2.84307%|
|M16/direct-dense/20261102152835/BF16|174745190400|16.420400|6.01420%|
|M18/direct-dense/20261102152817/BF16|3250984058880|27.943676|65.74877%|
|M18/direct-dense/20261102152835/BF16|20384317440|2.814740|4.09274%|
|M20/direct-dense/20261102152817/BF16|29124198400|8.729653|1.88544%|
|M20/direct-dense/20261102152835/BF16|31771852800|7.514879|2.38933%|

当前公开绑定证据还确认：目标/草稿共用16MiB索引bank，需保留writer/reader/rollback代际；C2mask的24/1704/3200/8176元素视图来自同8KiB backing，不能按固定24B临时allocation或恒值处理；四个embedding/head存储对象不同但未读数值，不能只由shape或指针作数值CSE判断。16worker共1104次实际采样调用绑定greedy、无penalty/entropy/synthetic/max_num_logprobs前提；这只支持对应观察人群和已有源合同，不证明所有score消费者消失。

v176采集触及单worker ET限额，完整C8校验失败。已恢复原服务；其部分窗口只作绑定证据，性能结论继续使用先前有效profile。恢复及终止记录可核查。此后仅CPU离线解析，没有新建NPU上下文、推理或重启。

下一步只做两项范围检查：精确条件到已有算子合同的覆盖对应；分层G0的调用、消费者、别名和共享状态端口对应。不得把已回答的条件未知、历史私有ABI、物理fmax或未实施提速重新当作通用阻塞；也不得在真实未分析范围仍在时因本页成稿标完成。

当前验收表：[current-requirement-audit.json](/Users/wio/work/166-vllm-analysis-continuation-20261030/shared-profile-analysis-v177/current-requirement-audit.json)；当前模块/对照：[module-analysis.json](/Users/wio/work/166-vllm-analysis-continuation-20261030/shared-profile-analysis-v177/module-analysis.json)；条件精确引用：[exact-source-guard-coverage-frontier.json](/Users/wio/work/166-vllm-analysis-continuation-20261030/shared-profile-analysis-v177/exact-source-guard-coverage-frontier.json)；条件源码规则轮：[declared-source-template-rule-review.json](/Users/wio/work/166-vllm-analysis-continuation-20261030/shared-profile-analysis-v176/declared-source-template-rule-review.json)。
