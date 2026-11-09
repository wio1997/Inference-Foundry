# 输入embedding与DSpark context整段判断

介绍：沿实际runner、文本wrapper和draft继承关系追查输入交付，重新形成aux选行→main→metadata/context→embedding→第一层draft HcPre的完整parent。**Goal ACTIVE，每个替换原语逐位一致；本轮无候选native准入。** loop-017仅限源码/价格判断后PIVOTED，loop-018继续三层draft body与TP/EP联合交付。

到了哪里：[15份运行源码身份](embedding-context-execution-identity.json)、[逐段边界/价格](embedding-context-frontier.json)、[原C8 SQL回执](context-parent-receipt-summary.json)。原reader首次在构造idle DP规格时KeyError、SSH/DB访问前退出，error Run184933Z保留；184951Z修正idle分支后匹配3696窗，其中3624 active、453个TP8组、227波。检查的是特定任务/shape/order边界，不能当全链路覆盖率或张量所有权证明。

结果：不存在的重复、直接跨模型共用和已有位差配方停止；单独小预算不再细拆。更大host/context联合producer仍待确认，与metadata-host共享同一部分成本，不能叠加收益。下一段继续三层decoder，不能因当前缺口已有名字就宣布扫全。

## 输入与embedding的真实读者

介绍：相同ID、同H5120和wrapper签名均不足以认定计算可共享。

到了哪里：`GPUModelRunner._preprocess:3502–3655`先clamp占位ID，多模态wrapper即使纯文本也查表并copy到持久embedding bank，仍传raw IDs。`vl_model.py:45–60,190–210,261–277`要求raw tokens；`model.py:1519`仅在没有inputs_embeds时内部查表，Engram/history/routing继续读IDs。FULL raw-ID capture由`model_runner_v1.py:4192–4206`选择，重放不会重走Python supplied-embed条件；旧native中明确见到图外与图内lookup。eager supplied-embed只算一次。soft prompt、MM merge、PP owner、graph key与bank其它读者均独立，不能仅跳整个preprocess。

目标先SP shard再HC4 repeat；draft先repeat再SP shard/pad。`sp_shard`无pad时clone、有pad时返回分块view，所以不能仅改变view而忽略所有权。第一HcPre的四路初值相同，可以考虑虚拟广播少搬，但不能把原四项/Gram或权重K归约重排后称逐位等价；整个mHC算法不被这个入口小范围剪掉。

draft的实际类为AscendDSpark→AscendDflash→AscendEagle及AscendSpecDecodeBaseProposer。`load_model:381–390`按embed API签名关闭其MM预计算，`dspark.forward:404–413`虽忽略inputs_embeds，活路径却没有此前重复lookup可删。`_maybe_share_embeddings:534–560`与load_weights/process_eagle_weight保留自己的embedding；原保存的四份embedding/head存储也相互独立。因此target/draft不能直接共用相同ID的查表值。

结果：**“draft忽略inputs_embeds，所以删重复查表”“target/draft按ID共享embedding值”已停止**。原named embedding全免费，C8有利TTFT5.956654ms/.39928%、整批93.115478ms/.50087%；六普通的embedding-only条件变体最高.61973%。25/14/23项旧输入pin检查均匹配，重复pin不当独立覆盖数。这里只结束相同独立1%优先预算的embedding/SP/外lookup小配方；更大host publisher和HC计算仍独立判断，1%不是用户硬验收线。

## Aux、main和三份cache：少算与共享到哪里

介绍：从三个context cache真实消费者向前推，不能只融合末端slot helper。

到了哪里：padded `prepare_inputs_padded:2388–2444`返回未改的INT32 arange/QSL前缀；GPU accepted count另外生成rejected数和sample-row选择器。no-spec prefix、非padded迟到selector、K0、skip/fit、dummy和DP参与保留各自条件。main是BF16完整K15360、TP列640→feature AG→全5120 RMS，输出copy进入context bank。三层各自Wkv/RMS/RoPE/store，不能共用不同权重或gamma的结果。

`copy_and_expand…:100–194`先复制context位置/物理slot，再用device `seq_len-rejected`生成query的物理linear slot；RoPE position另算。Part1不能读另一program同launch尚未发布的Part2结果。实际DSpark indices的可见域为`[max(0,P−128),P+5)`，P是校正prefix；每层自己的query store在其attention之前。suffix覆盖和window外退休可以减少上游行需求，但需要三层物理byte-span/owner/group/代次、prefix/CoW和排队读者共同证明。c=0、new/resume或group变更不能继承不中断generation的单调证明。

相同M、原行顺序的packet有旧原语样例；“先全M投影，再按不同M selector取行”的通用配方有4/2个位差反例。本轮独立核了原native回执SHA、同镜像和7份当前实现SHA，复用的是一NPU虚拟rank下的原语反例，**不是完整TP8准入**。旧派生aux报告的flow-index pin已失配，保留该失配，不继承其覆盖声明。main按三份aux拆K后相加、跳RMS或BF16边界也没有逐位资格。

结果：**通用changed-M提前投影配方停止**。保留原M/tile的行mask、完整内容key的cache、同M提前生产、共享metadata/consumer发布是待确认的不同机制；需要原数值几何、所有读者和净成本，不能用少了多少行直接换算耗时。

## 用相同原C8重算完整parent

介绍：为较大的交付单独定价，避免用小writer结果剪掉host/context联合范围。

到了哪里：原16个DB只读immutable SQL，复用先前全hash回执并核对size/mtime、保存的first-context/first-HcPre任务ID和时间戳；限定同stream/model、唯一Cat/main/RMS与准确行数。没有新profile。parent从三次aux IndexCheck或Cat开始，到第一draft HcPre；含main、feature AG/RMS、DP/输入metadata、三cache panel和query embedding/pad/premix。**之前target中的aux mean/SP AG、之后三个draft body和score都不在此parent内。**

结果：每波在active DP中独立取最有利八rank frontier，再按原客户端deadline传播；必要工作、idle peer限制、重叠与新成本均在毛包络中放松：

| 原范围全部免费 | P90 TTFT条件信用 | 整批条件信用 |
| --- | ---: | ---: |
| main GEMM/feature AG/RMS整段 | 4.539297ms / .304272% | 33.069756ms / .177883% |
| aux选行至首context Wkv之前 | 12.724520ms / .852933% | 119.728076ms / .644019% |
| 完整parent至第一draft HcPre | 23.118418ms / 1.549643% | 271.438628ms / 1.460072% |

单独early-main/删main即使全免仍属于低值独立C8预算，不为它重跑逐位用例；这不是对较大融合的普遍上界。完整parent不能按低于1%剪掉：rho=.5、每波新增h=.1ms的**条件净值**为11.059209ms/.741306% P90、113.019314ms/.607932%整批。rho未知，负收益保留。h=.1ms时若要达到旧1%优先参考，实际可省且暴露比例需约66.69%/76.85%。这说明只省main或末端小copy不足以支撑该parent的开发。

更大host/context共享交付**待确认，保留价值问题**。现有主机span包含必需计算、通信与等待，不可全部删；必须同时扣DP控制、各cache writer、有效query embedding、代次/lease/错误协议和新调度成本。与metadata BatchFrame、提前输出的同一段host时间只记一次。六普通已有小embedding/context条件值继续适用，但本次完整host parent没有该负载的匹配原生时间，不能用N相同便声称已覆盖，也没有因此重跑服务。

## 下一段

介绍：loop-018继续真正三层DSpark decoder作为一个producer。

到了哪里：从`DeepseekV41DSparkModel.forward:249–274`进入三次body，经非因果SWA/CP、HC、128专家/top3/EP16与共享分支，区分实际B×5和DP/SP物理padding；目标层local24/top6与M416服务资格不能借给draft local8/top3和小M。

结果：先匹配原完整body预算和TP/EP关键等待，再考察数学共享、通信排程、较大提交/megakernel；proposer强制eager不构成硬件“不可能capture”的证明。当前仍未完成全链路，没有新增“已确认可实现”项。本轮NPU0、模型加载0、推理0、新profile0、远端写入0，无服务或环境变化。
