# 算子融合与结构重组候选清单

当前持续分析台账（v115）为 **35项模型候选、7项框架候选**，完整范围仍未闭合。以下早期32项及G1/G2统计保留历史口径；F33–F35及F20的后续变体见表。最新条件、证据及未知项以[当前台账](/Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/candidate-profile-ledger-v115.json)和[本轮记录](/Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/PROFILE-FINDINGS-v115.md)为准。


本清单依据冻结源码、配置、既有结构图和少量历史 profile 编制，供后续判断用。**候选不等于可融合、已执行、能够加速或数值等价**。未修改目标实现。本轮在隔离诊断容器完成指定真实负载及官方双窗采集，并做过一次合成输入的单组投影接口对照；原常驻容器已恢复。以下候选仍是研究判断，不把诊断采集当作优化完成。

## 如何阅读与覆盖范围

共登记 **32 个主编号**：3 个是源码中已有复合调用的边界记录；27 个是条件性候选；2 个当前语义未知。F30 的早期单项记录仍标 `CONDITIONAL`，此处采用后续总扫描和层覆盖审计的 `UNKNOWN`，不把早期标记当作证明。每个 F 代表一条分析边界，可能有不同图版本的派生记录；它们不是彼此独立、也不能把预估收益相加。

后续源码条件补充图单独标记第 20 层向第 24、28、32、36 层索引来源发布候选块 ID，并把首层 `pre_mix` 初始化改为依赖形状和设备而非 embedding 内容；残差值依赖保留。原图与下文统计未被覆盖或重新生成。候选边连接的四对层原已有 KV 相邻边，因此简单的无向邻接枚举不会因加入候选语义边而发现新增相邻对；需要基于值、条件、消费者与生命周期重新审阅，不能把连接数量当作融合覆盖证明。补充图依赖冻结配置和源码分支，历史运行关联仍未知。

旧清单的边引用是原 G1 数组下标：删除被替换的边后，62 个旧引用指向该边，5,673 个大于该下标的引用会错位。因此旧清单与原图 SHA 绑定，新图需要重建索引或可靠映射，不能直接沿用。四条候选边对应非空、非 packed 分支；逐行发布粒度和同步尚未证实，不能把逻辑行域依赖自动转为可实施的无整调用屏障方案。

补充图已另建边映射及结构索引，并复核全部旧引用：旧 22,802 个节点集合完整保留，新索引共 22,820 条，其中新增 18 条是含初始化节点的结构组合；四条候选 ID 边与同端点 KV 边分别保留，形状属性不用于计算连通性。它们不是经过验证的融合机会；G5 模板、动态分支及运行时数据身份仍未完整展开。下文历史统计继续按原 G1 口径展示。

扫描覆盖现有源码模板中的两节点、三节点、部分多节点、拆分重组、共享输入和跨模块边界。G1 复合图上跨模块连通 2–4 节点集合共 **22,802 个**（分别为 385、2,376、20,041）；这些只是有界结构组合，**不是 22,802 个可行融合，也不是全模型穷举**。不同图版本的结构枚举不能相加计作独立机会。

后续覆盖审计发现：在 DSpark 的特定 fallback 条件下，现有 G1 图的辅助状态层号与该条件源码不一致；G5 片段还没有接上 context positions 和分组 slot 的生产者与消费者。历史运行是否采用该 fallback 未绑定。已在本地建立独立条件修订包并重建受影响的结构索引，旧图和旧索引保留供追溯；校验只证明图结构和引用一致，不证明运行身份、全部消费者或融合可行。下文历史统计仍使用旧图，不能作为这些路径完整覆盖的证据。

在该修订包基础上，另行建立了 hidden、positions 和按层选择的 group slot 三条跨片段条件数据映射，并在枚举时排除与细化片段重复的外围复合节点。生成的结构集合只是检查范围，不代表新融合数量；空输入、其他分支、buffer 身份和事件仍待绑定。

以下依据采用任务内部的 `fusion-scan.json`、`candidates/Fxx` 和 `LAYER-COVERAGE.md`；这些原始材料未随本清单公开。个别证据来自历史 profiler 选样，其范围和限制以总报告为准。**读者可以审阅候选与限制，但仅凭公开摘要不能独立复现实例级结论。**

下表“已有复合调用”表示现有源码调用了复合算子，不代表其内部不能重新组织；“条件”表示结构上值得考察，关键语义及运行条件尚未证明；“未知”表示当前连必要的语义或调用绑定也不足。每项均需保留原有数据类型、量化 scale、舍入和归约顺序，或另行证明改变后满足精度契约。

| 编号 | 位置与候选结构 | 状态 | 现有依据与主要缺口 |
|---|---|---|---|
| F01 | 已有注意力 mHC Pre 复合边界及其系数/外部Y分叉 | 已实现边界／条件重组 | 当前post/comb必须供本层Post，pre供下一FFNPre。Y/系数拆分是既有方向；本轮联接3200原生块并补共享权重/截止点合同。不能把同一HC工作、warm额度或观测跨度重复记为收益。 |
| F02 | embedding → 首层 mHC 系数投影；mHC pre → 输入归一化 | 条件 | 首层四路输入若确实相同，可研究预合并权重；实数工作量减少不保证浮点等价，预处理和后续四路消费者须保留。pre 与 norm 的分块、归约及精度另需满足；运行身份未绑定。 |
| F03 | attention 或 FFN mHC post → 后续pre / 最终collapse重组 | 条件 | 内层保留后续残差、aux及Engram读者。最终FFN Post仅在剩余快照/输出观察者和生命周期条件闭合时可私有融合collapse；保留BF16舍入及Post输入残差/系数。 |
| F04 | RMSNorm/类型转换与路由矩阵、非线性、TopK融合 | 条件 | 已有双输出norm不重复计收益。2201FLOAT/HF32与A3Vector指令速率已建分支模型；保BF16边界、全E选择、无偏权重及舍入合同。实际模式、联合吞吐和物理收益待证。 |
| F05 | Q norm → 动态量化 → Q 投影 | 条件融合／精度受限 | 保留RMS输出BF16舍入与行amax，可在UB衔接量化。公开融合API已查分支直接量化FP32归一化值，不能据名称认定位级等价；与Index Query共用量化时按同一接口图计账。 |
| F06 | Q与KV投影共享输入或组合 | 条件 | v43核对metadata身份/量化方法/通信共享条件；CP局部Q与全局KV不能按同形整份复用。部分行量化重用需相同行函数、参数、scale、row map及就绪；当前unquantized原语无新量化kernel收益计数。 |
| F07 | Q投影与RoPE的producer组合 | 条件 | v44补graph边界：replay使用捕获参数，runner kwargs输入不由位置参数DEBUG检查完整覆盖；需保持实际kwargs/nested bank、标量代际、事件与所有读者，不能仅以地址assert通过证明合法。 |
| F08 | KV投影、norm、RoPE、缓存写入及条件延后发布 | 条件 | 当前attention先需候选KV，不能仅延后Scatter。保留数组overlay、PA池COW、片上保留与重算是不同图；对原Scatter少搬运不代表优于直接写缓存。实际读者/前缀/tuple别名/精度/容量仍open。 |
| F09 | Index query 投影 → RoPE → 量化 | 条件 | 源码链；量化归约、scale 和下游消费者/就绪关系需核对。 |
| F10 | Index weight 投影 → 转换与缩放 | 条件 | 非 packed 路径的 weight 与 query scale 分支各有 FP16 舍入；将缩放跨分支移动可能改变结果，query scale 的量化消费者及实际 kernel 绑定需保留。 |
| F11 | Index key norm → RoPE → 量化 → key/scale 缓存写入 | 条件 | FP16 scale 舍入可能影响候选排序；分页读取还须遵守 block table 与真实页 stride，保留持久 key/scale、位置、消费者及就绪。历史布局未绑定。 |
| F12 | Indexer TopK、索引修整与共享发布 | 条件 | v43补role/候选/长度/emptycache/本地row合同；direct indices_output已存在。普通同宽I32连续bank的SMLA格式可只是view，不虚算pad/cast工作。copy省略需实际return alias/fullwrite/version/capture证明；block候选与token索引分开。 |
| F13 | 稀疏注意力复合边界 | 已有复合调用 | 源码已有 sparse flash 复合调用；宽实现域不必然独立计算每个分支，须保留可观察输出、sink/mask、缓存副作用与舍入。 |
| F14 | Attention → 逆 RoPE → 布局恢复 → 输出投影 | 条件 | 源码边界；保留张量布局、别名、通信及既有精度；运行实例需绑定。 |
| F15 | Attention 输出填充/恢复 → 投影 | 条件 | 特定 CP 分支存在源码级重组可能；分片行域、跨 rank 操作与实际运行分支未完全确认。 |
| F16 | 两级输出投影与reduce-scatter重组 | 条件 | v42补PCP full/shard权重与activation调度域；保留warmup alias、handles、wait及finally切回，不能假定权重每轮all-gather。实际代际与通信成本待绑定。 |
| F17 | C2两路FLOAT投影及与SWA BF16联合投影/分区 | 条件 | 原cast已共享；全BF16重编码须精确性证明。N24/M3有效算术不同，但声明固定B/L1/原类型/一次初始驻留下的权重服务界仍为8022周期。输入、通信与分块收益另核，不能用HF32容量推定实际模式或加速。 |
| F18 | Compressor 状态池化 → norm | 条件 | 池化本身已有复合调用；状态、历史页、槽位须保留。条件性反例表明仅凭各输入行的平方和不足以推出池化行的 RMS；可研究按通道累积或重算，需保留 BF16 舍入。历史值域及收益未知。 |
| F19 | Compressor latent 分叉 → index key 与压缩 KV | 条件 | 两条消费路径及持久状态来自源码；不得消去某一分支或更改其缓存语义。 |
| F20 | MoE router→TopK→dispatch；ID准备融合与跨层复用 | 条件 | v29补齐4472路由人口、80图×40层ID四步处理及5088原生ID准备TASK；仅-1→0与MC2零pad/行选择可交换，保留图像负数、逐层参数/行对齐、原始IDs和每次input generation；F20-ID是原项变体，不新增主号，无提速结论。 |
| F21 | MoE dispatch、专家FFN、combine及内部路由映射 | 已有复合边界／条件重组 | 已有dispatch到combine融合保留。小行数full-load的第二次排序只求逆，可在保留首个排列时直接构造；共享映射或前移到TopK另需mask原地副作用、sentinel、容量/顺序和发布合同，未证明提速。 |
| F22 | 共享 expert 的投影 → 激活量化 → 下投影 | 条件 | 采样已有 `DequantSwigluQuant` 复合任务；在半区 gate/up 布局条件下，256 通道分块能整块配对，512 通道分块的首块跨越两个 up 投影块。实际张量、分块及 scale 发布未闭合；需保留各自 clamp、量化边界和中间舍入，不能重复计算已有融合收益。 |
| F23 | 共享 expert 与 routed expert 汇合 → scaling/add，可延伸至 HC | 条件 | 跨 rank 归约重排须保留 shared 和 routed 的归约域及 BF16 舍入；历史通信分支未绑定，已有 scaling/add 不计作新增收益。 |
| F24 | Engram hash/lookup/ready；预清零、行头交换与状态重组 | 条件 | v30全范围3632原生TASK+160×38图段、176 TP8通信；当前UVA输出全覆写支持预清零条件省去，TP头gather→SP行切可改目的行/头交换但需全消费者/新bank/ready合同；cache写只能在全生命周期历史覆盖且无其他观察者时省去。 |
| F25 | Engram WKV、分块旋转、gate/残差与参数代际复用 | 条件 | v30补齐208 body/624参数准备TASK；实际32×32分块旋转，保留两路5120维RMS、FP32算术、原始hidden残差和4HC共享value。参数缓存/WKV tile/投影分区为重叠条件替代，无实现或提速结论。 |
| F26 | 最后层Post、collapse、Norm、选行及条件Q/FFN重组 | 条件 | 完整读者并集可条件前传到最后Q/输出/FFN；KV/cache/aux及EPLB状态另保留。TP8的U12可能仍需16～24行槽；head已有K行、权重和MC2接收方成本不可再按U/N缩放。实际读者/索引/数值/容量仍open。 |
| F27 | DSpark 辅助状态投影、完整行 norm 与 context 发布 | 条件 | v34绑定88条实际proposal及列分片640→TP8→完整5120维norm；间接三源投影、gather布局/norm直写bank、完整权重行分区和feature统计协议为条件变体。当前分片直接前移会漏off-diagonal计算；保留行序、BF16边界和所有bank/aux读者，逻辑通信不作收益。 |
| F28 | DSpark embedding、HC 初始化与 decoder 布局 | 条件 | v35绑定104完整forward模式及TP归约来源；先shard后repeat、正确padding的ReduceScatter、本地ID/mask发布与广播residual表示有条件。19组noQ仍需globalKV/输出交换；mask输入值、首post residual和统一collective域保留，逻辑字节不是收益。 |
| F29 | DSpark context KV、norm/RoPE、缓存写与组索引复用 | 条件 | v34覆盖312条完整source-order KV模式，分别264proposal/48dummy；ReplicatedKV声明、实际group-slot列表、纯FP32 RoPE查表/过滤及Proxy已定点核对。联合/流水、查表hoist、同组slot-pair和owner-KV交换均有独立条件；不自动删除624内部Cast或每层cache写。 |
| F30 | DSpark Markov、confidence 与验证预算重组 | 条件 | v36核对104五步/520步，绑定seed-inclusive embedding复用、raw hidden FP32confidence、global生存概率/预算和取回时prefix裁剪。device预算/批量长度发布与严格上界post-decoder剪枝为条件；当前dynamic flag/耗时未知，不把逻辑长度当物理收益。 |
| F31 | Metadata、slot、draft与graph发布重组 | 条件 | v47补normal adapter与canonicalDSACP源码；A3 packed getterNone为冻结normal-path条件证明，不冒充当前166依赖/生成值。193已连但原166/105资产未匹配、原rawprofile未恢复。3136为模块归属歧义执行而非已定价候选资源行；实际bank/epoch/分母和完整scope仍未闭合。 |
| F32 | SP/PCP输入gather与逐行Q_a/KV投影重排 | 条件 | v42补PCP投影→gather/restore数学合同；必须覆盖SWA KV、compressed-source和multistream所有global hidden消费者，否则原gather仍需保留。682整数例仅验证行恒等式，native舍入/实际rowmap/拓扑/收益未证。 |
| F33 | C1 compressor与SWA KV的同域投影、联合分块及前移通信 | 条件 | v16补充：source层20两路BF16投影、独立512维norm；保留布局、消费者、ready和数值合同，无收益证明。 |
| F34 | 压缩KV状态、slot与物理缓存布局 | 条件 | v47区分normalA3与A5packed软件分派；已有SWA anchor不变时不能承诺单独slot压缩省显存，整体pool/ABI变体需实际所有权/布局/内存分母。历史A3构建不是新166runtime快照。 |
| F35 | 目标采样行选择、概率过滤、race及输出重组 | 条件 | 默认原P阈值与上游pivot合同分开。Pivot已有多项scan融合，1e-9近似计数非普通并行归约；可条件保原tile有序拆分、复用同表达式Exp、虚拟最终mask及隔离scratchbank。原生精度、读者/生命周期与收益仍待证。 |

后续还单独检查了 target rejection 路径的取行、FP32 转换与原值副本初始化：可作为条件性组合候选；原值可能被 entropy 验证或 raw logprobs 消费，不能无条件删除副本。此项不是上表 F30 的独立收益，也尚未绑定历史运行。

F30 的特定概率恢复源码路径中，拒绝后的恢复选择会依赖词表中其他项的信息。存在两组草稿分布对已选 token 的概率相同，但在相同 target 概率和噪声下恢复出不同 token 的反例，因此不能仅凭已选 token 的概率保证逐结果等价；这不排除经证明等价的融合、压缩或重新计算。该概率分支是否由历史请求使用仍未知。

## 未覆盖与下一步

- 图上节点是复合模块，编译后二进制或自定义算子的原始指令、内部通信和所有消费者并未完整展开；跨迭代依赖也不在单次调用连通集合枚举内。超过初始四节点限制的变换只覆盖少数已知复合边界。
- 层角色映射不等于运行时每条分支都执行。要把候选变成实际适用结论，需要同一次调用的配置、输入形状、张量/别名、事件、持久状态和算子实现身份；再证明全部消费者及精度语义。
- 选样测得的任务时长、低活动字段和潜在节省不得直接叠加或外推到整模型。没有同口径的实际时间与必要工作下界，不能计算可靠加速或全局理论吞吐上界。
- 优先结合既有轨迹追查 F12 的输入与索引消费者、F08 的持久写入身份，以及 F21 的通信计算依赖，以及 F32 的隐藏状态消费者和通信组；如果已有证据缺失，明确记录缺口，不凭结构相邻推断数据依赖。

## 当前证据支持的派生方向

- F12 的 token K=512 特化只缩短 token 累加列表，保留 tile/candidate=2048；每满 tile merge 输入对数 4096→2560，状态回拷 16→4 KiB。不是近似 TopK，也不是已经测得同等时延下降。
- F15/F16 的单组投影后端：合成 BF16、交替各16次 eager 事件区间中位 66.0/48.8 µs，输出最大差 0.000244140625，存在漂移。仅支持继续考察二维 GEMM；未证明实际图与真实权重精度或吞吐收益。
- F05/F15/F32 的跨模块替代方案：CP 分支 Q_b 从全 head 复制权重改为前移 head 分片，条件账每 rank 权重 40→5 MiB、MAC 相同；需同步重构 query/index/KV cache 的 token/head 所有权及通信。与保留后部 head 交换的方案不能简单叠加。

新官方采集与文献复析对上述方向的影响以 [研究报告](README.md) 正文为准。

v28持续分析记录：35项模型候选、7项框架候选；表中旧G1/32项枚举统计仅属历史范围，未证明当前全图闭合。当前证据见 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/sampling-and-coverage-summary-v28.json。

v29持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/FFN-norm-and-routing-ID-summary-v29.json；历史枚举统计不代表当前全范围闭合。

v30持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/Engram-whole-frontier-summary-v30.json；旧枚举/报告成稿不代表全范围闭合。

v31持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/M15-final-tail-and-consumer-summary-v31.json；局部reader/行域问题回答不等于全范围闭合。

v32持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/M16-vocab-ownership-layout-and-score-summary-v32.json；旧枚举/报告与局部恒等式不证明全范围完成。

v33持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/M17-distribution-and-recovery-summary-v33.json；局部证据/报告成稿不证明全范围完成。

v34持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/M18-aux-context-reorganization-summary-v34.json；局部证据/报告成稿不证明全范围完成。

v35持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/M19-decoder-ownership-and-HC-summary-v35.json；局部证据/报告成稿不证明全范围完成。

v36持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/M20-confidence-and-next-ID-summary-v36.json；局部证据/报告成稿不证明全范围完成。

v37持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/M21-metadata-and-host-publication-summary-v37.json；局部证据/报告成稿不证明全范围完成。

v38持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/M21-slot-rejection-state-summary-v38.json；局部证据/报告成稿不证明全范围完成。

v39持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/M21-C2-metadata-and-initialization-summary-v39.json；局部证据/报告成稿不证明全范围完成。

v40持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/M21-field-reconciliation-summary-v40.json；局部证据/报告成稿不证明全范围完成。

v41持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/M21-original-scope-audit-summary-v41.json；局部证据/报告成稿不证明全范围完成。

v42持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/M21-fixed-case-analysis-summary-v42.json；局部证据/报告成稿不证明全范围完成。

v43持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/M21-neighbor-reconciliation-summary-v43.json；局部证据/报告成稿不证明全范围完成。

v44持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/M22-graph-lifetime-analysis-summary-v44.json；局部证据/报告成稿不证明全范围完成。

v45持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/M22-registry-reconciliation-summary-v45.json；局部证据/报告成稿不证明全范围完成。

v46持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/M22-mode-padding-analysis-summary-v46.json；局部证据/报告成稿不证明全范围完成。

v47持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/dispatch-access-resource-summary-v47.json；局部证据/报告成稿不证明全范围完成。

v48持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/profile-evidence-capability-summary-v48.json；局部证据/报告成稿不证明全范围完成。

v49持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/M17-fallback-summary-v49.json；局部证据/报告成稿不证明全范围完成。

v50持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/F21-placement-publication-summary-v50.json；局部证据/报告成稿不证明全范围完成。

v51持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/F21-checkpoint-commutation-summary-v51.json；局部证据/报告成稿不证明全范围完成。

v52持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/F21-callback-registration-summary-v52.json；局部证据/报告成稿不证明全范围完成。

v53持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/restored-profile-and-output-summary-v53.json；局部证据/报告成稿不证明全范围完成。

v54持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/request-resolution-summary-v54.json；局部证据/报告成稿不证明全范围完成。

v55持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/operator-fusion-and-ceiling-summary-v55.json；局部证据/报告成稿不证明全范围完成。

v56持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/MC2-operator-analysis-summary-v56.json；局部证据/报告成稿不证明全范围完成。

v57持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/shared-expert-ceiling-summary-v57.json；局部证据/报告成稿不证明全范围完成。

v58持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/HC-operator-ceiling-summary-v58.json；局部证据/报告成稿不证明全范围完成。

v59持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/HC-regime-clarification-summary-v59.json；局部证据/报告成稿不证明全范围完成。

v60持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/attention-indexer-ceiling-summary-v60.json；局部证据/报告成稿不证明全范围完成。

v61持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/projection-ceiling-summary-v61.json；局部证据/报告成稿不证明全范围完成。

v62持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/state-engram-ceiling-summary-v62.json；局部证据/报告成稿不证明全范围完成。

v63持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/router-vector-ceiling-summary-v63.json；局部证据/报告成稿不证明全范围完成。

v64持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/index-context-ceiling-summary-v64.json；局部证据/报告成稿不证明全范围完成。

v65持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/heads-sampling-ceiling-summary-v65.json；局部证据/报告成稿不证明全范围完成。

v66持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/communication-operator-ceiling-summary-v66.json；局部证据/报告成稿不证明全范围完成。

v67持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/embedding-initialization-ceiling-summary-v67.json；局部证据/报告成稿不证明全范围完成。

v68持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/metadata-field-ceiling-summary-v68.json；局部证据/报告成稿不证明全范围完成。

v69持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/remaining-field-ceiling-summary-v69.json；局部证据/报告成稿不证明全范围完成。

v70持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/selected-input-ceiling-summary-v70.json；局部证据/报告成稿不证明全范围完成。

v71持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/planner-ceiling-summary-v71.json；局部证据/报告成稿不证明全范围完成。

v72持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/pad-ceiling-summary-v72.json；局部证据/报告成稿不证明全范围完成。

v73持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/candidate-composition-summary-v73.json；局部证据/报告成稿不证明全范围完成。

v74持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/remaining-fusion-edge-summary-v74.json；局部证据/报告成稿不证明全范围完成。

v75持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/V41-query-ceiling-summary-v75.json；局部证据/报告成稿不证明全范围完成。

v76持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/router-predicate-ceiling-summary-v76.json；局部证据/报告成稿不证明全范围完成。

v77持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/padding-route-ceiling-summary-v77.json；局部证据/报告成稿不证明全范围完成。

v78持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/shared-layout-scope-ceiling-summary-v78.json；局部证据/报告成稿不证明全范围完成。

v79持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/ring-storage-lifetime-summary-v79.json；局部证据/报告成稿不证明全范围完成。

v80持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/A3-datapath-ceiling-summary-v80.json；局部证据/报告成稿不证明全范围完成。

v81持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/A3-datapath-verification-recovery-summary-v81.json；局部证据/报告成稿不证明全范围完成。

v82持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/quant-epilogue-summary-v82.json；局部证据/报告成稿不证明全范围完成。

v83持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/attention-composition-summary-v83.json；局部证据/报告成稿不证明全范围完成。

v84持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/Engram-HC-summary-v84.json；局部证据/报告成稿不证明全范围完成。

v85持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/DSpark-tail-summary-v85.json；局部证据/报告成稿不证明全范围完成。

v86持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/terminal-layer-summary-v86.json；局部证据/报告成稿不证明全范围完成。

v87持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/KV-publication-summary-v87.json；局部证据/报告成稿不证明全范围完成。

v88持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/graph-mixed-summary-v88.json；局部证据/报告成稿不证明全范围完成。

v89持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/typed-capacity-summary-v89.json；局部证据/报告成稿不证明全范围完成。

v90持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/greedy-fallback-summary-v90.json；局部证据/报告成稿不证明全范围完成。

v91持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/sampling-filter-race-summary-v91.json；局部证据/报告成稿不证明全范围完成。

v92持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/pivot-filter-summary-v92.json；局部证据/报告成稿不证明全范围完成。

v93持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/HC-attention-summary-v93.json；局部证据/报告成稿不证明全范围完成。

v94持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/norm-quant-summary-v94.json；局部证据/报告成稿不证明全范围完成。

v95持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/route-quant-summary-v95.json；局部证据/报告成稿不证明全范围完成。

v96持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/route-map-summary-v96.json；局部证据/报告成稿不证明全范围完成。

v97持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/receiver-budget-summary-v97.json；局部证据/报告成稿不证明全范围完成。

v98持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/raw-operator-summary-v98.json；局部证据/报告成稿不证明全范围完成。

v99持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/projection-panel-summary-v99.json；局部证据/报告成稿不证明全范围完成。

v100持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/attention-projected-state-summary-v100.json；局部证据/报告成稿不证明全范围完成。

v101持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/recovery-fallback-summary-v101.json；局部证据/报告成稿不证明全范围完成。

v102持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/sampling-applicability-summary-v102.json；局部证据/报告成稿不证明全范围完成。

v103持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/selection-attention-summary-v103.json；局部证据/报告成稿不证明全范围完成。

v104持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/operator-boundary-summary-v104.json；局部证据/报告成稿不证明全范围完成。

v105持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/coverage-dependency-summary-v105.json；局部证据/报告成稿不证明全范围完成。

v106持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/index-projection-summary-v106.json；局部证据/报告成稿不证明全范围完成。

v107持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/operand-evidence-summary-v107.json；局部证据/报告成稿不证明全范围完成。

v108持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/stop-mask-summary-v108.json；局部证据/报告成稿不证明全范围完成。

v109持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/operand-probe-summary-v109.json；局部证据/报告成稿不证明全范围完成。

v110持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/penalty-histogram-summary-v110.json；局部证据/报告成稿不证明全范围完成。

v111持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/pad-provider-summary-v111.json；局部证据/报告成稿不证明全范围完成。

v112持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/clock-capacity-summary-v112.json；局部证据/报告成稿不证明全范围完成。

v113持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/draft-cache-join-summary-v113.json；局部证据/报告成稿不证明全范围完成。

v114持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/sample-hash-summary-v114.json；局部证据/报告成稿不证明全范围完成。

v115持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/coverage-metadata-summary-v115.json；局部证据/报告成稿不证明全范围完成。
