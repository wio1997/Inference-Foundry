# 2026-11-11 Engram实际hash与UVA内核：补齐覆盖，完整父段继续保留

介绍：从实际prepare_engram调用追V1 history写入、两层hash、共享UVA查表/反量化、TP heads交付，到SP行上的Wkv/rotated gate与后继HC。到了哪里：17当前源码身份；补齐实际适配器、上游history和Triton hash/lookup主体，并与exact-SHA旧合同去重；旧两代表卡24完整双层交付定价。结果：整段native包络1.954322%代表窗，不能停止整个父段；lookup本体0.031226%、私有预零0.008845%，同预算小方案不重测。该父段与机制已有旧记录，本轮是覆盖/组合价补全，不冒充新发现。新增原生准入0，全链路未闭合。

实际调用与状态：DeepseekV41Model.prepare_engram一次hash/gather供层1/14，再直接table.embed_gathered；Embedding.forward不是这处服务入口。shared-memory=true令table.dp_size=1，绕过每步DP hash gather/rowexchange；TP8每rank3个head，仍需完整head-order交付再SP切行。V1 adapter继承NgramHashState.ensure_cache：未绑KV返回false并丢history，同一live KV对象保留，换对象重置；request-index一次供两层，独立_write_hash_cache先写mapped/dead token，再hash读取。当前chunk>known lookback>slot fallback，known dead会阻断且不回退；阻断对后续shift保持。层/head multipliers、prime和offset不同，不能共用两层hash结果。合hash/lookup或结果cache仍须保存所有未来history读者和原INT64溢出/XOR/modulo次序；普通program顺序不能替代旧kernel完成屏障。

UVA与反量化的完整边界：当前共享映射每rank各自CANN注册、构造INT64 chunk指针；leader只在全员attach/失败汇总及barrier后unlink。codes/scales是独立映射，关闭成功后才撤对应alias，unregister失败保留可重试owner。指针表不是每token建立。lookup将ID先转INT64，按全局vocab区间判断归属，减vocab_start后按2^22行chunk定位；INT8/FP32 group32读数、乘法和BF16舍入已在一个内核。invalid ID安全读row0，最终select为+0；不能改成0乘可能NaN的结果。选中pad_heads=local_heads，全部本地head每次写完，局部out的zeros可作为冗余候选；通用wrapper仅写本地head前缀，不能删pad尾/固定buffer尾零化或初始化表零。两表、两层Wkv及其结果均不同，已有DP共享/反量化/ID widening不得重复记收益。

数值与消费者：BF16 rows先TP heads收齐，再SP按原行/填充规则交原M3、K6144、N25600的BF16 FLOAT Wkv，随后原FP32 block32旋转/RMS、左结合dot、signbit/sqrt/sigmoid、mask、乘V加hidden、最终BF16四路残差。原Wkv BF16边界和最后残差仍有HcPost读者。直接K分片再求和、擦掉舍入、mask false直通、stock AIVAGMM或改变M/N/tiling的失败配方不复跑。原数值panel/N-owner复用B与直接SP-owner去掉全局bank是不同布局，需要的新交换与生存期不能同时免费。本轮匹配旧hash/lookup/state合同SHA并复用其有限检查；没有新数值验收。旧M3/M15常量/hidden widening复用资格仍只限其fixture，不能扩大到全父段。

完整价格：旧配平C8 EP0/8、global24/local3中24完整双层分支，含48个Wkv+37gate链。整段native均1.954322%窗口，原Wkv0.717463%，其余1.236860%；mask/hash/history0.200640%，lookup0.031226%，private zeros0.008845%，hash slice0.009031%，TP AG+transpose0.557507%。同起止到层14 HC跨度19.915867%，夹着13层必要target工作，不能当融合收益。TP交付native也含arrival wait，不是可删纯传输。若保留原Wkv、只省其余50%并另付20us/完整双层交付，条件净5.357–5.724ms/0.569763–0.615753%窗口；付50us为0.531473–0.577029%。这些是敏感性情景，不能外推请求或六普通。

旧负载另列：原C8 lookup全部免的固定模型P90 TTFT条件信用最高0.118045%、makespan0.046259%；旧Wkv/gate/copy全部免最高1.115388% TTFT，非全Engram上界。原owner-only六普通tuned约0.427–0.929%，default0.323–0.726%；联合input+gate+external tuned0.968–2.073%含其它父段，不相加。查表/compact scale/预零/hash小helper、旧独立gate与readiness不变tuple cache、原样overlap/staged prefix均沿其原停止预算；当前1.954322%大包络并不替代这些旧子范围，也不足以证明任何新实现会达到净收益。

归档去重：旧loop012的Engram完整交付报告已记录同一大父段、history/invalid/padding/alias合同、容量路线和native位反例；本轮补实际函数/内核与原递归worklist的精确交集及代表窗组合价。待确认仍是较大input/owner/HC/iteration与NUMA/capacity：原vendor M/N/K tile和Fixpipe逐位路径、全部新packet/collective/pack成本、全16rank ready与资源竞争、persistent bank最后读者/失败退役、六普通和请求曝光。INT8保留E8M0的17.16665GiB物理共享host容量路线仍按旧CPU穷举资格和NPU/成对加载/映射缺口保留；纯lookup延迟小预算停止。未知请求信用为0。

旧完整合同与请求/六普通价：[Engram完整交付与融合判断](engram-execution-review.md)。

下一条待扫：loop047下一Run：先把旧源全文范围、当前实际receiver与aux/Compressor/Engram侧表合成累计递归覆盖前沿，保留每项的精确SHA/分支/消费者/历史价及未闭合原因，避免重复扫已有主体。再从未归位的AscendMoERunner.apply_routed_input_transform、_maybe_reduce_*、shared/routed transform与最终combine实际消费者继续，绑定冻结SEDP/DFC与合法替代分支；按完整输出交付审共享、数学/量化/布局和megakernel，再用已有同负载价。不要用319节点、34输入谓词或具名缺口声称全链路闭合；每替换原语逐位一致。本源码Run无新Torch/NPU/模型/profile/请求/服务/远端写入。
