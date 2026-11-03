本轮补齐 F28–F32：第一层 draft embedding/HC 初始化如何与 SP 前移投影、Q_a 行归属及跨迭代复用组合。全部工作基于已有 profile、固定源码和本地模型检查。

按 Tile 原始 TASK 键把104次初始化接到第一层 HcPre、RMSNorm、SP gather；每组前面的 embedding AllReduce 也通过精确 CID 和 dspark249 源码栈确认。分支有69次 Q_a 与全局KV同时出现、19次正面 preprocess354 栈确认的本地查询为空/全局KV仍更新、16次正面 metadata=None 清零。19次空查询不能按 dummy 路径删掉 KV 或输出/head交换。顺序窗口与源码证明范围不等于真实 tensor、指针、请求或 cache 世代绑定。

既有 v35/v67 已提出 embedding AllReduce→匹配 SUM ReduceScatter、先 SP 后 repeat、局部或虚拟 HC 初始化。本轮把它们放入一个图，避免将全局 Tile/Pad、局部初始化和虚拟 residual 的完整收益重复相加。SUM 的成员、rank顺序、舍入、量化及全部 embedding 读者须匹配；ReduceScatter 的 padded 输入内容可能更大，不能单凭输出变小宣布 wire 流量或延迟更低。功能 custom-op 的 clone、别名、capture 和旧读者合同仍须保留。

第一层 HC coefficient/pre/post/comb 以及 first HcPost residual 仍活跃。外部 onehot premix 的 Y 与当前 coefficient 生成是两个数值依赖分支，提前交付 Y 需要新的 split/handoff。它也不是无条件返回 embedding：1×Inf+0×Inf+0×Inf+0×Inf 得到 NaN，直接 Inf 则不同。原 BF16 Y、完整5120维 RMS 的 BF16输出、full4H coefficient 计算和后续舍入不能因融合而消失。

SP owner 的 padded 行与 CP 查询 owner 的行未被实际操作数绑定。若它们不一致，仅把 KV512前移仍不够：须交付缺失的归一化 raw5120行，或由 SP owner 计算并交付 row-tagged Q_a1280；也可在新ABI中统一发送512+1280=1792维平面。Q_a1280 norm、后续查询投影、RoPE与head角色仍保留，全部 raw消费者要闭合。3816组原始 BF16 位模式/掩码用例确认 copy重排保留 signed zero、Inf、NaN payload与 supplied True前缀；864组 owner置换用例核对查询行标签与顺序。这些不是 native数值或真实 CP/SP map 证明。

参考 TP8 均匀 owner 图中，N10需 A16行：原 SP sender内容20480字节，KV-only为2048，KV+Q_a为7168。全卡 KV dense参考行数80→16，但若查询集合原来不重叠且覆盖10行，Q_a参考行数10→16。N20/A24对应30720→3072或10752字节，KV160→24、Q_a20→24。前移通信带来计算与padding取舍；这些是接口内容和指定 dense算法工作，不能当 HBM必要字节、网络wire、关键路径收益或物理上限。

跨迭代还可登记第一层 token行复用：在完整 embedding/reduction/量化、HC初态、norm与WKV、math/nativeM、参数/adapter及capture世代稳定时，预计算或按需缓存 pre-RoPE normalized KV512；可另存 raw Q_a1280。position/RoPE、slot和cache发布仍在 lookup 后执行。一个固定key版本下，完整词表129280行的 KV值内容为132382720字节（126.25MiB），KV+Q_a为463339520字节（441.875MiB），还不含索引、队列、fill与epoch维护。若需同时保留不同nativeM/math/参数版本，值内容还要相应增加，不能假定一张表对全部M等价。原始 embedding、HC coefficient及residual仍有需求，不能因缓存KV就删除。SP padding 的零行也不能拿 token0 的 embedding代替。

owner按需缓存仍可能需要 KV gather；全复制表加全部所需 global IDs 则可研究在每个cache接收者直接lookup，代价变为各卡lookup、全局ID寿命、表容量与维护。当前没有实际重复token/命中率/参数稳定性证据，也没有 native cachefill等价性，因此它只是有明确条件和成本的候选。不能将后续层已改变的HC状态或已加位置的KV按同一token缓存。本轮没有运行设备或修改模型。

新增 REL132-01，条件关系138→139、待核对457→456；HG85 已含两候选，联合组仍26。并修正 G10 文本遗留的135/460旧计数，该修正不算优化发现。35主候选、7历史框架记录、358语义行/26模块和854节点/1346边/48提示的部分图不变；9类/1352次原生字段、849个PMU缺口、16个phase未绑定图、3136项归属歧义继续保留。整模块关闭0，整体目标仍未完成。
