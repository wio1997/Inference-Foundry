# DSpark词表、Markov与完整拒绝结果交付

介绍：从DSpark选行/LMHead→Markov五步→概率/confidence guard，追到target bonus/greedy拒绝与下一轮ID/count读者。到了哪里：21当前源、冻结策略白名单、19原源码loop/guard/原生内循环标量及有限反例；两代表卡24完整head链/120个W2。结果：当前confidence不执行、greedy draft不导出概率；完整Markov/目标词表父段保留有条件分片/融合，原样cache fixture不重跑。原122源码约束/34动态位未变，全递归调用图未闭合，新原语/框架准入0。

已停止：把定义/分配confidence head当作当前耗时；把proposer._compute_confidence和DynamicSpecScheduler的实际helper当两次计算；并行五个依赖前一步ID的Markov transition；只取base top1再加bias；擦掉W2/BF16 add舍入；将Markov复制head按base LMHead TP路径重复gather。冻结serve c297只有dspark/K5/eager，未设draft_sample_method和dynamic_spec；源默认greedy、dynamic method=None，full vocab无d2t。base head TP8，Markov W2 disable_tp=True先决策为复制，最后ID写回B×6持久buffer的1:视图，种子列/别名和下一轮宽度必须保留。

待确认：较大的base-head/Markov/argmax完整交付可改词表owner或融合W2后处理。原base TP-local GEMM保持M/N/K/舍入；Markov按词表分片、局部bias+BF16 add+argmax后做全局(value,global-ID)选择，省稠密base gather但付五次串行小通信。tie选全局最小ID，padding/NaN/输出dtype与原native必须一致；当前V129280、base每片16160，不以通用rank×local-V代替所有LoRA/pad映射。有限例证明base top1不保留加bias后的winner；省bias GEMM的BF16舍入可把winner由1变0。原语完整输出/中间可见变异、原N-shard native位、packet/同步/lease与付成本净价仍缺。

完整尾部定价：旧配平C8 EP0/8、B4/K5/R256/V129280，draft Post-ready→最终ID写完均1.084138%该窗；norm→ID仅0.991113%、Markov五步0.600790%，不能把较窄数字当全parent。保持复制W2/base gather，仅新后处理20us/步的情景净0.234237–0.236032%；假设W2实删87.5%且每步小通信10us+后处理20us，净4.177–4.185ms/0.444258–0.450156%；通信30us变0.316624–0.321075%，100us变负。87.5%是理想逻辑分工的时间敏感性，不是实测8倍加速，首轮冷/后续热W2及五轮host/通信成本都不能省略。

请求/历史分列：旧原C8固定之前所有迭代和请求开始，只推进当前carrier Markov到下一Fill的全免信用，按旧TP/latest-DP为最大0.531087% TTFT，最宽全16卡span为0.694546%；不是新客户端完成或整个decode/makespan预测。旧score→seed 112周期整段全免约0.442–0.608%周期，只剪其中固定原LMHead外部的独立clone/argmax/count小配方；不能剪掉更大projection/布局/Markov共同父段，也不拿它当六普通上界。六普通全parent/请求曝光及新HCCL延时未绑定，新增实际请求信用0。

已有Markov cache证据可复用：当前四源SHA与旧真实权重native fixture全相同，M2/3/4对M1/排列/重复邻行/选列有0位差，固定cache128行全hit分数/IDs通过。原全hit单步约省19.841us，静态1/2/3 miss混合反而多26.127–27.171us；未收费classifier时也要约56.8–57.8%的全hit step才抵消对应混合代价。保存的345个scalar placeholder中可用前驱ID为0，不能推命中率。只是历史有限fixture、不是完整动态cache/owner/六普通准入；不重跑同配方，只留真实命中/分类器/退役与N分片位缺口。

target词表→完整拒绝IDs另立父段：baseline先bonus私有选行、target转FP32且raw模式clone，再constraints/argmax/原Triton首拒绝与bonus补位。可考虑在无分数/grammar/非argmax不变processor等严格guard下，用原local GEMM+小摘要直接交原拒绝kernel，保留所有INT32[B,K+1] placeholder/bonus槽、有效数、RNG与异步代次。raw/processed logprobs、entropy、penalty/bad words/thinking budget、NaN诊断和随机/混合分别有读者，不能无条件删完整logits或raw副本。19控制中的原Triton内循环涵盖全接受、首拒绝、零draft、混合greedy mask；不是native编译/位或向量prologue读界证明。

[本段判断](DSpark词表与Markov到拒绝采样的交付判断.md) · [真实消费者与缺口](draft-output-review.json) · [19原源码/有限控制](draft-output-controls.json) · [完整尾部/付成本/历史native交集](draft-output-price.json)。

下一条待扫：loop047 Run191838Z PLANNED：继续M09完整target vocab→Sampler.forward/各processor/bonus/拒绝→AsyncModelRunnerOutput/有效数与CPU输出→Engine/scheduler下一轮消费者；闭合guarded摘要路径的全部logit/NaN/RNG读者及新代次退役。并对全模型递归callee建立明确已扫/未扫worklist，区分34未定输入谓词与尚未走读的分支腿，不能用有名字的缺口或122冻结约束宣布扫完。保留add-HC、shared-DFC、metadata/gid12、Engram owner与Markov整段候选，不重开旧固定小配方。源先定大交付后匹配旧C8/六普通，每替换原语逐位一致；无新Torch/NPU/模型/profile/请求/服务或远端写入。

本轮仅21身份/限定body、19源码/标量/有限控制与旧profile/native证据交集。无新框架/Torch/NPU/模型/profile/请求/服务/远端写入，未重测live健康。历史fixture资格原样保留，本轮新增原语/框架准入0。
