# 输入尾部、MC2有效行与MoE完整输出

介绍：沿target/draft输入尾部追实际MoE有效行交付。到了哪里：30当前源身份，38原Python前缀/尾部/MC2行控制；从actual前缀到SP切片、量化payload和原生DFC mask写者。结果：target旧尾部可保留，DSpark query有真实用途；DFC已有独立有效行处理。原156位仍122受限绑定/34动态待追，全链路未闭合，新原语/框架准入0，每替换原语逐位一致。

已停止：按position=0/ID=0或图shape认整行恒零；按5个DSpark query只保留anchor；把router未读is_padding推成DFC没有有效行mask，再重复计算“新增跳过padding expert”的收益；盲删空DP整卡执行。target只写actual ID/embedding前缀，clamp只改负数，positions尾部清零不能清除非负旧ID/旧embedding。DSpark eager固定，query尾部显式填parallel drafting ID/position0/slot-1，与SP新补零行不同。

待确认：大row交付保留，但拆开已存在的mc2_mask与model is_padding。target实际count、draft实际query count经forward context生成mc2_mask，MC2按本卡原SP行再补DP长度，W8A8 payload传到DFC x_active_mask。当前DFC源核在init routing前把无效行expertIdx storage改为expertNum哨兵；不是-1。原生变异、expert counts、完整routed输出、shared输出/归约/scale/add、HC残差/系数、aux/MTP和下一轮租约都属于替换边界义务。torch schema仅声明out/count可变，源码kernel却写expertIdx；包装层到loaded op实际alias仍须核实，不能据此宣称已观测caller Tensor变异。

旧EP0/8 localM3窗的DFC全免均16.758165%、norm→Post全免均22.504527%；它们包含已有mask和必要数学，不是padding行可省比例。保留DFC必需时，parent span减DFC的极乐观包络均5.746362%（两代表卡，global24/480calls），仍含必要norm/router/shared/HC和等待，不能全计为候选收益。新row请求收益信用默认0；六普通及原C8请求曝光/净价待绑，不迁移不同cohort。

继续深入的边界：真实AscendMoERunner的shared/routed reduction→scale→foreach_add→HC Post整体交付，保持每一处BF16/FP32舍入和事件/collective顺序；不先假定把多个数学式合并就逐位相同。把已有mask可省之外的具体工作与完整原语输出闭合后，才给出开发准入。大metadata/gid12和Engram owner/input候选保留。本轮仅源码、行tag与旧数据，无新Torch/NPU/模型/profile/请求/服务/远端写入。

[本段判断](输入尾部与MC2有效行及MoE完整输出判断.md) · [buffer与真实消费者](row-delivery-review.json) · [38原源码控制](row-delivery-controls.json) · [分cohort价格](row-delivery-price.json)。

下一条待扫：loop047 Run182623Z PLANNED：继续M06真实AscendMoERunner.forward/_forward_impl→shared_experts各执行/等待/归约→scale/foreach_add→HC Post的完整输出；保留原生DFC有效行、expertIdx潜在变异/count与全部EP参与。先走较大交付再用匹配旧profile拆必要工作/新增开销/请求曝光，若整段乐观低才剪同预算子项。然后转下一未扫递归消费者；34动态原位/全callee仍未闭合，不重复122冻结约束或已停小配方。每替换原语逐位一致，无新Torch/NPU/模型/profile/请求/服务或远端变更。

| 行域 | 实际交付 | 已追消费者 / 缺口 |
|---|---|---|
| target actual prefix | current CPU gathered IDs / previous sampled IDs / previous draft scatter; corrected GPU position | GPUModelRunner._preprocess and target full/eager model input；Full capture entry/read-address and all error/lease retirement remain; source cases are not observed frame population. |
| target FIA/graph/DP tail | nonnegative old IDs or old external embedding may remain; position tail zero; graph internal lookup differs from eager external embedding | target model hidden/HC, Engram/hash selected reads, attention padded metadata；No claim every tail reaches same consumer; graph/eager chosen bytes, history and retained outputs need exact binding. |
| DSpark real query | K=5 source default sample_from_anchor=True: query0 bonus +4 parallel IDs, five positions and sample indices; alternate False uses K+1 queries | _run_merged_draft:1476-1534 -> full draft layers -> gathered selected states/LMHead；Do not discard4noise as padding. Alternate config legal source leg, frozen currentHF selection inherits previous model/source evidence. |
| draft DP padding | parallel draft ID / position0 / query slot-1; whole context suffix slot-1; no hidden/external-embed writer in that method | DeepseekV41DSparkModel.forward raw ID embedding, local HC; mc2 active mask uses pre-padding num_tokens；Different from target old tail and SP modulo zeros; preserve cached physical views. |
| SP modulo padding | only extra rows beyond model shape are newly zero-filled; clone behavior when aligned preserved | HC, normalization/router, shared/routed; mc2 mask separately excludes inactive actual domain；Zero mathematical input alone is no native bit proof; downstream coefficients, outputs, counts and leases remain. |
| MC2 valid rows | True before actual count, False suffix; slice original SP shard then DP-pad false | routed_experts:717-750 -> W8A8.apply:218-234 -> payload.mc2_mask -> fused DFC:590-608 / CANN MegaMoe alternative:514-551；model is_padding and mc2_mask are separate contracts. Original controls prove geometry only. Native caller alias, selected loaded op and packet/bits remain. |
| native DFC mask mutation | inactive expertIdx storage overwritten with expertPerRank*EP before init routing, followed by sync | routing/group-list/EP GMM and combine; expert_token_nums output；Python schema declares only out/count mutable. Adapter/API forward input; runtime op executor/format transformations and caller-visible alias must be verified, not inferred. Other native internal routing/unpermute branches not fully audited here. |
| retained whole MoE/HC/aux | shared/routed selected reductions+scale+foreach_add -> HC Post; residual/pre_mix and aux checkpoints retained | next layer, aux DSpark context, MTP buffer and final collapse/norm, next iteration consumers；Still must follow inherited reduction/scale hooks, all shared streams and quant/reduction boundaries. Cannot zero whole row or skip an empty EP rank with remote work. |

30身份不等于30全文审阅；38控制均为整数/布尔/opaque行tag，不是native逐位对拍。原122/34账未新增绑定，递归callee覆盖另列，不伪造全链路完成。CANN MegaMoe替代leg及本地DFC核分列，旧DFC任务证明历史人口，不证明当前loaded op。
