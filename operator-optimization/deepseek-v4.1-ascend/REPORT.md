**整体优化机会分析已通过验收，停在发现与诊断阶段。**

范围为 DeepSeek-V4.1-Flash-1017-w8a8、DSpark、Engram，171上的CANN9.2环境，冻结32K输入/1K输出、C8、P90文本负载。覆盖26个功能模块、360条语义行、原1101处静态审查和另外1867处精确条件引用。40层目标网络、3层草稿网络及跨算子、跨层、跨卡、跨迭代关系均纳入当前公共条件图。驱动/固件差异按用户接受的条件保留。

共保留35个模型候选族和7个host边界候选族，42个均为条件式候选。每个族包含原图/候选图映射、消费者/状态/别名约束、工作与I/O变化及资源条件。没有实施优化，没有融合后的实测提速。当前F21已经融合dispatch、量化、双GMM、SwiGLU及return，不能把同一融合再次计为新收益。

停止依据是完整的声明范围、Case与7条规则扫描及语义复核：两两、三算子、更多算子、拆分重组、共享输入、跨模块和多消费者均有结论；最后一轮未出现新的非等价候选。42个族的互斥、依赖、共享资源与生命周期约束同时保留。状态为SCAN_COMPLETE_IN_SCOPE；这是有限规则下的覆盖结果，不是所有可能算法已穷尽的证明。

实质核验修正了四处会影响判断的关系：

- DSpark三层共享上下文输入，分别使用各自WKV、norm/RoPE、cache/group/slot。联合输入读取不等于复用一份KV输出。
- source20的2048候选块列表有fresh output及发布拷贝，供24/28/32/36四个过滤读者使用；它与512位置TopK/prepare输出的别名关系不同。
- 批状态修正的三次gather先于computed写入，accepted的else读取在该写入之后。融合保留跨行旧值和潜在跨平面别名顺序；逐行原地重映射反例产生[21,22]，正确值为[21,11]。
- SP提供mask的True前缀及True padding尾部保留；不能凭BOOL shape把SP mask与C2 mask当成同一状态。

另已纠正Engram的历史CPU异步标签：当前正向证据为NPU host-UVA gather/dequant，记录的overlap标志为False。在已捕获EP0 worker中，目标/草稿共用16MiB TopK backing；8个源层值代际供38层注意力读取，重排保留重叠范围的RAW/WAR/WAW及下一轮覆写约束。跨rank不能按相同虚拟地址推断物理共享。

最低利用率需分组表述。按报告周期加权的AIC MAC活动率，M12在12个有效计算模块中最低，合并窗口为约2.41549%，两个窗口分别约2.55250%和1.17269%；对已记录的归属歧义仍稳健。M23在20个有效Vector算术组中最低，通信搬运/轮询并不因此无用。没有一个可信的host/Cube/Vector/通信全体最低值；这些比例也不是整卡忙碌率或物理MFU。

M12的低活动位置集中在已融合F21：两个窗口中，它分别占M12累计native时间约98.76%/99.19%，占报告AIC周期约99.72%/99.93%。小接收tile、rank/expert不均衡以及peer到达/协议等待仍为竞争假说；现有指标不支持“单纯HBM饱和解释整个区间”。累计kernel时间不是请求墙钟，也不能直接视为可消除时间或端到端提速。

当前171无profiler参考完成8个32768输入/1024输出请求，客户端墙钟16.346353521秒，返回输出约501.1515 token/s；这是单次参考，不是稳定性或提速结论。正式profile在16个逻辑设备采两段窗口，共32份profile；全部8份响应与参考文本匹配。全量聚合288846条native执行，201351条可唯一功能归属，87495条歧义单列，3510条缺PMU不填零。v176未完成的完整C8采集仅提供部分绑定证据，不用于替代有效性能数据。

理论与实际对照共94条：26个完整功能模块的条件/符号记录、42个未实现候选记录、26个同口径密集矩阵子集记录。360条语义行都有对照路由。所有对象保留T、L、Q/T、U、L/T、T/L、活动指标、支配约束和证据状态；候选T为空。26个匹配子集已用精确分数重新计算。1800MHz仅为额定参考，不是物理最大频率；未知的f_upper、带宽、模式或必要工作继续保留符号/条件或UNKNOWN，未伪造完整模块的有限数值上限。L/T不是设备利用率，T/L不是可达提速。

|模块|名称|唯一归属native条数|AIC MAC活动率|AIV Vector活动率|
|---|---|---:|---:|---:|
|M00|注册、平台替换、模型与算子选择|UNKNOWN|UNKNOWN|UNKNOWN|
|M01|请求、tokenization与采样契约|UNKNOWN|UNKNOWN|UNKNOWN|
|M02|前缀复用、KV分配与回收|UNKNOWN|UNKNOWN|UNKNOWN|
|M03|EngineCore、调度与异步队列|UNKNOWN|UNKNOWN|UNKNOWN|
|M04|worker输入准备与持久batch状态|2224|UNKNOWN|0.63818%|
|M05|embedding、模型wrapper和图像条件|2688|UNKNOWN|14.86994%|
|M06|mHC初始化、pre/post与跨层残差|17888|18.55886%|47.76641%|
|M07|Q/KV投影、norm、量化与RoPE|10786|43.77610%|38.02437%|
|M08|compressor、ring状态与latent分叉|1240|85.31651%|19.20898%|
|M09|Indexer评分、候选与跨层索引|4544|17.25753%|65.25212%|
|M10|SWA、压缩KV与稀疏attention|4225|30.96945%|20.84985%|
|M11|attention输出、head恢复、两级投影与归约|7257|74.46228%|1.40725%|
|M12|MoE路由、dispatch、专家计算和combine|31286|2.41549%|0.68474%|
|M13|共享专家、路由分支汇合与scaling|10208|51.68150%|42.40911%|
|M14|Engram hash、host/UVA lookup、投影与门控|4736|61.92930%|7.35831%|
|M15|最终mHC collapse、norm及MTP辅助状态|616|UNKNOWN|18.08709%|
|M16|目标/草稿词表投影与跨卡重建|384|11.79996%|0.18673%|
|M17|目标采样、拒绝/接受、bonus与logprobs|1144|UNKNOWN|9.80047%|
|M18|DSpark上下文投影、共享KV及输入发布|9624|69.20902%|8.90436%|
|M19|DSpark decoder、mHC及共享计算|520|UNKNOWN|1.62276%|
|M20|DSpark base logits、Markov和confidence|5824|20.28561%|7.01101%|
|M21|draft/target metadata、slot与索引准备|72157|UNKNOWN|1.38304%|
|M22|graph捕获、replay、更新与多流生命周期|0|UNKNOWN|UNKNOWN|
|M23|DP/TP/EP/PCP通信与分区所有权|11624|UNKNOWN|0.01791%|
|M24|跨迭代状态、接受回滚、预取和重用|2376|UNKNOWN|1.03496%|
|M25|采样结果回传、响应、流式输出与终止|0|UNKNOWN|UNKNOWN|

上表只汇总已记录的唯一归属执行和正分母活动字段。host、未执行、未采到或无法归属的部分不自动为0；完整分母、窗口、rank及歧义见原始聚合索引。

|候选族|登记的机会范围|状态|
|---|---|---|
|F01|注意力前 mHC 复合边界|CONDITIONAL|
|F02|embedding → 首层 mHC 系数投影；mHC pre → 输入归一化|CONDITIONAL|
|F03|attention 或 FFN mHC post → 下一次 mHC pre，跨边界重组|CONDITIONAL|
|F04|FFN 归一化与类型转换|CONDITIONAL|
|F05|Q norm → 动态量化 → Q 投影，多算子链|CONDITIONAL|
|F06|Q 与 KV 投影，共享输入或合并投影|CONDITIONAL|
|F07|Q 投影 → RoPE，生产者与消费者组合|CONDITIONAL|
|F08|KV 投影 → norm → RoPE → SWA 缓存写入|CONDITIONAL|
|F09|Index query 投影 → RoPE → 量化|CONDITIONAL|
|F10|Index weight 投影 → 转换与缩放|CONDITIONAL|
|F11|Index key norm → RoPE → 量化 → key/scale 缓存写入|CONDITIONAL|
|F12|Indexer TopK → 索引修整 → 共享索引发布；候选块号排序跨层复用或位图 mask|CONDITIONAL|
|F13|稀疏注意力复合边界|CONDITIONAL|
|F14|Attention → 逆 RoPE → 布局恢复 → 输出投影|CONDITIONAL|
|F15|Attention 输出填充/恢复 → 投影|CONDITIONAL|
|F16|两级输出投影与 reduce-scatter 的分块重组|CONDITIONAL|
|F17|Compressor 两路输入投影，共享转换或分块|CONDITIONAL|
|F18|Compressor 状态池化 → norm|CONDITIONAL|
|F19|Compressor latent 分叉 → index key 与压缩 KV|CONDITIONAL|
|F20|MoE router → TopK → dispatch；输入 ID 准备融合与跨层复用|CONDITIONAL|
|F21|MoE 内 dispatch、两次分组矩阵计算、激活、combine 的分块流水|CONDITIONAL|
|F22|共享 expert 的投影 → 激活量化 → 下投影|CONDITIONAL|
|F23|共享 expert 与 routed expert 汇合 → scaling/add，可延伸至 HC|CONDITIONAL|
|F24|Engram hash/lookup → ready → 行视图或分片；host-UVA 寻址|CONDITIONAL|
|F25|Engram 投影 → gate/残差|CONDITIONAL|
|F26|最终 HC collapse → norm 及 logits 边界|CONDITIONAL|
|F27|DSpark 辅助上下文投影 → norm，候选流水|CONDITIONAL|
|F28|DSpark embedding → HC 重复/初始化 → decoder|CONDITIONAL|
|F29|DSpark 上下文 KV 投影 → norm/RoPE → 组装与缓存写入；按请求复用 SWA 索引|CONDITIONAL|
|F30|DSpark collapse/norm → Markov bias、logits/置信度/验证|CONDITIONAL|
|F31|Attention/DSpark 相关 metadata 构造、共享与发布|CONDITIONAL|
|F32|SP 输入 gather → Q_a/KV 投影，研究前移投影后按需通信|CONDITIONAL|
|F33|C1 compressor 与 SWA KV 的同域投影共享、联合分块与前移通信|CONDITIONAL|
|F34|C2 ring 状态的可观测性裁剪、稀疏维护与表示重组|CONDITIONAL|
|F35|目标采样的按需raw logits、行选择/转换/argmax与拒绝输出重组|CONDITIONAL|
|H01|响应等待路径的轮询开销与重复等待工作|CONDITIONAL|
|H02|输入准备中剩余小拷贝、锁页和元数据工作的合并重组|CONDITIONAL|
|H03|前缀块哈希的跨请求重复工作|CONDITIONAL|
|H04|前端长提示 tokenization 的精确复用、排队与批处理|CONDITIONAL|
|H05|前端 DecodeStream 早段长调用与输出就绪流水|CONDITIONAL|
|H06|固定 speculative 宽度下的 padding 域拆分与图 bucket 重组|CONDITIONAL|
|H07|原始 token ID 图回放与图外 embedding 的单一生产路径|CONDITIONAL|

|匹配子集|必要密集MAC|累计kernel时间ms|额定1800参考L/T|
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

保留的未证项已按影响分类：物理fmax/模式/带宽限制绝对数值上限；私有参数、native别名与retirement限制相关私有编辑；host当期成本和请求关键路径暴露限制因果提速判断；F21的等待机制仍有竞争解释。它们均未被当成独立、无效、零开销或已证可优化。所有声明范围均已有分析答案；更强的无条件融合、物理上限或实现收益需要新的验证，不属于本次发现与诊断的已证结论。

本次验收为SELF_REVIEW，不声称独立外部审阅或形式化证明。原始大数据保留原路径和SHA，历史169/166及失败采集单列，未混入当前171指标。原服务已在取证阶段恢复，最后服务校验在v177；本次收尾未执行远程命令、推理或重启，也未把旧health当作新的在线检查。

当前交付入口：

- [合同](records/latest/contract.json)、[原图](records/latest/graphs/G0.json)、[42候选及条件](records/latest/records.json)、[扫描台账](records/latest/fusion-scan.json)。
- [模块及94条理论/实测对照](records/latest/module-analysis.json)、[界输入](records/latest/bounds-input.json)、[精确分数复算](records/history/continuation/shared-profile-analysis-v180/exact-rational-P5-recalculation.json)。
- [profile小摘要](records/latest/profile-summary.json)、[完整数据索引（原材料未随归档提供）](inventory/excluded-materials.jsonl)、[26模块汇总](records/latest/module_metrics.csv)、[84算子汇总](records/latest/operator_metrics.csv)。
- [逐项完成验收](records/history/continuation/shared-profile-analysis-v180/final-completion-audit.json)、[360条语义与状态复核](records/history/continuation/shared-profile-analysis-v179/semantic-contract-sufficiency-review.json)。
