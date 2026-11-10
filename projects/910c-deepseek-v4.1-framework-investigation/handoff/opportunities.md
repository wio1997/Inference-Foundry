# 融合优化逐项说明

## 当前断点 · 2026-11-11 三份aux到context cache：解析真实receiver，共享准备按原预算停止

介绍：从target层37/38/39入口的HC均值，追三次SP交付、选行/concat、main投影/RMS和三个draft context cache，到各层SMLA实读。到了哪里：19当前源码身份、3份旧精确SHA回调源、15原Python/有限布局控制；24完整aux/main/三cache链、72panel。结果：A3 main投影不量化，RoPE有重复准备；同窗独立aux打包/准备共享沿原预算停止，更大的host/context共同producer保留。原122约束/34未定输入不变，native准入0，全链路未闭合。

真实字段派发：文本wrapper的language_model_cls→AscendDeepseekV41LLMForCausalLM，model_cls→DeepseekV41Model；PP1拥有40层，DSpark wrapper→DeepseekV41DSparkModel→三个DSparkDecoder。后者继承原Decoder.forward，attention_cls指向DSparkSWAAttention，原attention前向别名和CP实现保留。具名receiver与原工作表匹配边、额外展开body另列，不把“字段名字相同”当相同计算。

aux是真实三个不同时间的输入：模型在进入zero-based37/38/39之前取HC4均值、SP gather并裁full_num_tokens；不是三个相同Tensor，也不是层输出。runner的无spec metadata前缀分支先裁num_scheduled，验证分支按token_indices选行；随后按配置顺序concat，保留重复行、原顺序、discard/rejected与DP/SP尾部关系。DFlash继承的build_model_inputs_first_pass把固定context隐藏/位置buffer和各层slot交给三层cache producer，不能把B×5 query当作全部context行。

main数学边界：DSpark main_proj传入的quant_config仅在packed-cache设备且checkpoint标为fp8时非None；A3 packed=None，所以这里是原BF16/FLOAT unquantized GEMM，不能按模型W8A8标签省quant。main_proj前缀不命中特殊column op，ColumnParallelLinear保留K15360、TP8列640，feature gather回5120后原RMS；原return_bias=False。Replicated wkv经CustomReplicatedOp调用各自quant method，无这条feature TP gather。wkv的全局quant_config仍要按各prefix描述和实际native任务判断，不能从main不量化外推所有wkv。RMS还有bias_loaded分支，保持原norm/add边界；不把三份aux拆K后相加、改变M/分片权重或穿过RMS作代数折叠当作逐位等价。旧changed-M有4/2位差反例，原样配方不重跑。

实际少算点：每层_project_shared_kv都调用_apply_dsv4_rope；其get_cos_and_sin_dsa(context_positions)默认use_cache=False、layer_names=None，遍历全部已注册配置并构造新cos/sin，再只取本层名字。当前匹配链每层有4个表Index及其4个IndexCheck，三个panel共12次表Index；保留原第一份完整proxy、后两层共享准备，可删后8次及对应检查。它不要求三层config相同，但要求同一position/registry/full-cache代次、只读cos/sin与覆盖三消费者的租约。若再按layer_names限制配置，必须保留各层真实config，不能猜所有名字共用一套频率。15控制执行了原getter/proxy、不同config和缺层错误，并给出position变代次的否定例；没有原生数值验收资格。原consumer内cast、RotaryMul、KV/RMS数学和写入仍保留，不能把准备共享说成RoPE本体全免。

能少搬什么：先保持三个均值的BF16结果不动，考虑本卡aux按每row的三个H段打包，单次SP gather成全局[N,3H]，再选行；其值与原三次gather后选行/concat可做坐标对应。总aux逻辑payload不减少，仍要付pack、等体积collective和选行成本。三个legacy aux若以strided view暴露，还须闭合全部reader/别名、graph固定地址与native stride支持；只证明数值排列不足以准入。full_N裁切必须先于可能的负索引，TP8/N11与重复索引有限tag例保留了这点。

追到真实cache读者：_context_slot_mapping_buffers按_layer_group_idx重建，gid相同可引用同一slot buffer，三个cache Tensor仍是不同层的物理view。_store_standard_swa_kv在slot=None/empty时只停止store；上层仍已算过该层Wkv/norm/rope。A3一维linear slot先clamp/除模并把负数两列置-1；二维坐标保留。scatter_cache_sk把values转cache.dtype并contiguous，原SK算子保留cache实际page stride和[-1,-1]跳行，不能按shape假设平面连续。随后各DSpark层经显式dsa_v41_forward side-effect边界、query cache更新和metadata ATTENTION wait，把同一swa_cache_layer.kv_cache[0]作为SMLA ori_kv。packed A5、long-compressed reader不属于这个冻结draft角色。跨层/跨流须证明物理byte-span、cache/query读写次序及producer/consumer退役；不是前缀/gid不同就可并发。

按整段定价再剪枝：旧配平C8 EP0/8、M24/local3的完整aux均值+三AG+选行/main+三cache native分支，均0.893095%该窗；从最早aux到cache末写的墙钟跨度达5.374410%，其中夹着必需target末三层、head/rejection与metadata，不能当融合预算。aux AG/选行/concat全免仅0.158914%，context全部90任务0.583920%；三次全RoPE getter0.243367%，保留第一份而共享后两份的native信用0.161580%。新增pack10us+AG30us+选行25us、共享控制5us/call，二者不相交的条件净2.119–2.193ms/0.225392–0.235898%该窗；较贵110+10us仅0.161575–0.171357%，210+20us只0.021179–0.029368%，aux packing单项已转负。均为付成本情景，非实测新kernel收益。

历史/请求分开：原C8全16卡112周期中，全部90 context任务全免最多0.866501%周期，沿用原stage-confined row/RoPE/slot停止决定。固定所有此前迭代和请求开始、只推进当前carrier的旧context全免，TP/latest-DP最大个体1.759380%不代表P90；重算该模型P90只有2.175160ms/0.146108%，全16时钟宽包络为5.134580ms/0.344896%。它们都不是全迭代或客户端收益上界。旧逐波传播的更大host/context父段仍有P90全免23.118418ms/1.549643%、整批271.438628ms/1.460072%的宽包络，含必需工作且与metadata/frame重叠，不与本轮比例相加。六普通的完整新父段/净曝光仍缺，新增请求信用0；1%只是旧独立优先预算的引用，不是新用户硬阈值。

结果分档：已停止——相同预算下独立aux打包、context RoPE/slot/row小方案及既有changed-M反例配方，不做原样native重验。待确认——更大的aux/host/context共同producer、保持原数值tile的owner或megakernel，以及既有metadata/gid12/add-HC/sharedDFC/Engram/Markov候选；要扣必需数学、通信、query/cache写入及全部租约，再看六普通和请求曝光。已确认可实现——本轮0，原15协议/坐标控制不是逐位准入；旧封闭CPU原型资格独立保留。

[本段判断](evidence/三份aux到DSpark_context_cache的字段派发与共享判断.md) · [真实receiver与分支](evidence/aux-context-review.json) · [逐边派发](evidence/aux-context-edge-dispositions.json) · [15原Python/有限控制](evidence/aux-context-controls.json) · [完整分支与付成本价](evidence/aux-context-price.json)。

下一条待扫：loop047下一Run：转DeepseekV41Compressor.forward及其实际设备实现，沿输入缓存/压缩KV、index-K发布到跨层SMLA实际消费者，先绑定冻结role与receiver，再按exact-SHA旧证据去重。覆盖合法ratio/short-input/append/reset/缓存复用分支和全部可见输出；先审整段数学/搬运/等待共享，再匹配原C8/六普通，低整段预算才结束同预算子项。保留大metadata/gid12、add-HC/sharedDFC、Engram owner、Markov/host-context具体缺口；每替换原语逐位一致，不重开本轮停止的小配方。本源码Run无新Torch/NPU/模型/profile/请求/服务/远端写入。


## 当前断点 · 2026-11-11 目标采样到下一轮：完整score读者与递归callee分账

介绍：沿原target词表→bonus/拒绝采样→完整ID/count/logprob/NaN输出→异步复制、RPC与Engine下一轮消费者。到了哪里：27当前源、12原Python协议/有限反例；两代表卡24完整target head/reject/seed链，另立全模型递归callee工作表。结果：逐位门槛不变，完整score有条件才能消掉；旧固定投影小方案继续停止，更大owner/Markov交付保留待确认。122/34只说明原输入谓词，全链路仍未闭合，新原语/框架准入0。

已停止：把“greedy draft”当作target全greedy；给原bonus Sampler.forward(max_num_logprobs=-1)换成仅IDs而漏掉其全score输出；只因argmax结束便删raw logits、NaN、logprobs或调度状态；用本卡全-inf替代MinTokens的全词表恢复条件；把thinking写1e9当作强制ID；在随机/混合/合成路径删RNG或忽略discard回退offset。冻结DSpark的processor factory只建MinTokens，custom processor报错，不能把未选中的MinP/LogitBias当当前耗时；普通无spec metadata的首步仍用同一个spec配置factory。

到了真实消费者：AscendRejectionSampler先取bonus私有行，强制max=-1返回raw/processed全scores；target私有选行转FP32，raw模式再clone，之后penalty/allowed/badwords/MinTokens/thinking与采样constraints。外层max_num_logprobs非None才合并target+bonus全scores并取logprobs；source getter仍有完整词表/排名读者。原bonus输出本身不可缩成IDs；只有把更大的完整拒绝交付作为替换原语、并证明对应score结果在该guard下死掉，才能消除内部全score副本。保留所有仍活的BF16/FP32边界、输入变异和输出。

合法条件分支：target all_greedy是每批sampling metadata，不能从draft策略推出。全greedy constraints直接返回；mixed/random会缩放、top-k/p、softmax、uniform与recovered exponential noise，按原per-request generator顺序执行；discard还回退offset4。synthetic+block明确抛错。entropy原softmax在greedy早返前计算，若只看该函数返回则该概率在全greedy分支死掉；冻结默认未启用，不给这条未匹配负载的新分支计价。MinTokens structured恢复先看屏蔽后全行isneginf，再只恢复有限stop值；分片要全局all-inf与原stop值。thinking状态按bonus→target顺序更新/改force_index与bonus_token_forced，并写FP32 1e9，inf/NaN仍可能改变winner。这些路径保留原实现或扩大摘要，不能静默略过。

交付一直追到下一代：sample_tokens先清12字段暂态，grammar可能把完整logits往返CPU，bookkeeping在采样后读原logits的NaN，并保留prompt-logprobs所需hidden。ID INT32[B,K+1]含bonus/placeholder，count/seed/padded、draft与后续状态都读它。AsyncGPUModelRunnerOutput持有device refs，复制流等生产流、复制ID/logprob/NaN再record event；get_output先sync再parse/发布。MP的WorkerProc.enqueue_output实际调用get_output，失败转FAILURE；future按队列顺序交给Engine，先处理abort再scheduler.update_from_output。InputBatch另持CPU/event，在下一次_sample前修复重排后的输出尾部，spec修复还会原地截短draft CPU列表。parse_output是(!=-1)&(<V)逐位置过滤，不能改成>=0或遇首-1即停止；logprobs过滤/丢弃顺序也保留。

下一轮状态不能吞掉：Scheduler先处理inflight/stale与延迟free，按原request→output索引取ID，扣rejected computed/placeholders，逐token append/check_stop并在stop截尾；AsyncScheduler对非stale扣已交付placeholder并cache_blocks。首拒绝/全接受/零draft、discard、abort/完成、stale preemption与structured接受分别有分支。摘要buffer不能复用原output未完成的槽；最省证明的边界是继续保留原fresh ID输出、Fill/reject/count/seed/D2H/RPC/Engine过程，仅替换闭合guard下的score交付。

完整价格分列：旧配平C8 EP0/8、M24/B4/D20/V129280的24链，target Post-ready→reject完成均0.825173%该窗，到seed/padded完成0.882463%，含必需GEMM/数学与等待，不能全算可省。保留原local GEMM/Fill/reject/seed和全部后续，只给已匹配score gather/transpose/选行/cast/clone/argmax族全免预算均0.352432%。新增local reduce+小通信+merge总40us/call，条件净2.778–2.852ms/0.298852–0.303327%该窗；100us降0.221403–0.226747%，240us只0.040690–0.048060%，350us转负。都是收费情景，不是新kernel实测。原C8全16卡另有112周期：head-end→seed整段全免最多0.608161%该周期，旧固定投影小配方不重开；不把此子段上界迁给更大的HC/LMHead布局/Markov/context共同父段。CPU完成/请求与六普通曝光未绑定，新增请求信用0。

待确认：较大词表owner/Markov/下一代交付可继续合并判断。最窄安全起点需同时满足all_greedy、standard、无grammar/full-score/logprob/NaN额外读者、无生效penalty/allowed/badword/MinTokens/thinking，或显式保留各读者。原local GEMM的M/N/K、dtype、舍入不动；native tie/NaN/all-inf/pad与global ID、完整输出/可见变异、capture/别名、collective及summary临时租约、实际eligible人口与新增成本仍缺。任何末位差异都不准入；12协议控制没有数值验收资格。

递归覆盖分账：工作表把模型原17个选中/待角色绑定root（排除冻结PCP）与16条C3/alias绑定、target输出消费者一起展开；80份带来源的Python源，319个(receiver,body)上下文、2739处静态callsite。这里只对可解析direct/self/super调用继续展开，字段receiver、闭包、构造、外部import与native边界逐条挂起；AST展开不是走读。34未定输入谓词仍单列，不能当34条未扫源码，也不能据319/2739宣布扫全。当前明确走读和旧SHA原文相交证据另列，未闭合边由后续逐项归并/核实。

[本段判断](evidence/目标词表到下一轮调度的完整输出与递归缺口判断.md) · [原语全部输出与缺口](evidence/target-output-review.json) · [12原Python协议控制](evidence/target-output-controls.json) · [完整父段与付成本价格](evidence/target-output-price.json) · [递归callee工作表](evidence/recursive-source-worklist.json) · [明确走读/旧原文交集/未闭合边](evidence/recursive-source-disposition.json)。

下一条待扫：loop047下一Run：优先收敛recursive-source-worklist/disposition里的已选模型字段receiver、闭包与import/native边界，先按exact-SHA旧原文证据去重，再选第一个没有完整走读凭证的真实大交付；逐条记录分支leg、实际消费者、可省工作/原语全部输出与价格，不重做旧固定投影/clone/argmax/count或相同native fixture。34未定输入与未走读callee分账；add-HC/sharedDFC、metadata/gid12、Engram owner、完整Markov继续留具体缺口。每替换原语逐位一致；源先定范围再匹配旧C8/六普通；本源码Run无新Torch/NPU/模型/profile/请求/服务/远端写入。


## 当前断点 · 2026-11-11 DSpark输出：confidence未启用，Markov分片须付五轮通信

介绍：从DSpark选行/LMHead→Markov五步→概率/confidence guard，追到target bonus/greedy拒绝与下一轮ID/count读者。到了哪里：21当前源、冻结策略白名单、19原源码loop/guard/原生内循环标量及有限反例；两代表卡24完整head链/120个W2。结果：当前confidence不执行、greedy draft不导出概率；完整Markov/目标词表父段保留有条件分片/融合，原样cache fixture不重跑。原122源码约束/34动态位未变，全递归调用图未闭合，新原语/框架准入0。

已停止：把定义/分配confidence head当作当前耗时；把proposer._compute_confidence和DynamicSpecScheduler的实际helper当两次计算；并行五个依赖前一步ID的Markov transition；只取base top1再加bias；擦掉W2/BF16 add舍入；将Markov复制head按base LMHead TP路径重复gather。冻结serve c297只有dspark/K5/eager，未设draft_sample_method和dynamic_spec；源默认greedy、dynamic method=None，full vocab无d2t。base head TP8，Markov W2 disable_tp=True先决策为复制，最后ID写回B×6持久buffer的1:视图，种子列/别名和下一轮宽度必须保留。

待确认：较大的base-head/Markov/argmax完整交付可改词表owner或融合W2后处理。原base TP-local GEMM保持M/N/K/舍入；Markov按词表分片、局部bias+BF16 add+argmax后做全局(value,global-ID)选择，省稠密base gather但付五次串行小通信。tie选全局最小ID，padding/NaN/输出dtype与原native必须一致；当前V129280、base每片16160，不以通用rank×local-V代替所有LoRA/pad映射。有限例证明base top1不保留加bias后的winner；省bias GEMM的BF16舍入可把winner由1变0。原语完整输出/中间可见变异、原N-shard native位、packet/同步/lease与付成本净价仍缺。

完整尾部定价：旧配平C8 EP0/8、B4/K5/R256/V129280，draft Post-ready→最终ID写完均1.084138%该窗；norm→ID仅0.991113%、Markov五步0.600790%，不能把较窄数字当全parent。保持复制W2/base gather，仅新后处理20us/步的情景净0.234237–0.236032%；假设W2实删87.5%且每步小通信10us+后处理20us，净4.177–4.185ms/0.444258–0.450156%；通信30us变0.316624–0.321075%，100us变负。87.5%是理想逻辑分工的时间敏感性，不是实测8倍加速，首轮冷/后续热W2及五轮host/通信成本都不能省略。

请求/历史分列：旧原C8固定之前所有迭代和请求开始，只推进当前carrier Markov到下一Fill的全免信用，按旧TP/latest-DP为最大0.531087% TTFT，最宽全16卡span为0.694546%；不是新客户端完成或整个decode/makespan预测。旧score→seed 112周期整段全免约0.442–0.608%周期，只剪其中固定原LMHead外部的独立clone/argmax/count小配方；不能剪掉更大projection/布局/Markov共同父段，也不拿它当六普通上界。六普通全parent/请求曝光及新HCCL延时未绑定，新增实际请求信用0。

已有Markov cache证据可复用：当前四源SHA与旧真实权重native fixture全相同，M2/3/4对M1/排列/重复邻行/选列有0位差，固定cache128行全hit分数/IDs通过。原全hit单步约省19.841us，静态1/2/3 miss混合反而多26.127–27.171us；未收费classifier时也要约56.8–57.8%的全hit step才抵消对应混合代价。保存的345个scalar placeholder中可用前驱ID为0，不能推命中率。只是历史有限fixture、不是完整动态cache/owner/六普通准入；不重跑同配方，只留真实命中/分类器/退役与N分片位缺口。

target词表→完整拒绝IDs另立父段：baseline先bonus私有选行、target转FP32且raw模式clone，再constraints/argmax/原Triton首拒绝与bonus补位。可考虑在无分数/grammar/非argmax不变processor等严格guard下，用原local GEMM+小摘要直接交原拒绝kernel，保留所有INT32[B,K+1] placeholder/bonus槽、有效数、RNG与异步代次。raw/processed logprobs、entropy、penalty/bad words/thinking budget、NaN诊断和随机/混合分别有读者，不能无条件删完整logits或raw副本。19控制中的原Triton内循环涵盖全接受、首拒绝、零draft、混合greedy mask；不是native编译/位或向量prologue读界证明。

[本段判断](evidence/DSpark词表与Markov到拒绝采样的交付判断.md) · [真实消费者与缺口](evidence/draft-output-review.json) · [19原源码/有限控制](evidence/draft-output-controls.json) · [完整尾部/付成本/历史native交集](evidence/draft-output-price.json)。

下一条待扫：loop047 Run191838Z PLANNED：继续M09完整target vocab→Sampler.forward/各processor/bonus/拒绝→AsyncModelRunnerOutput/有效数与CPU输出→Engine/scheduler下一轮消费者；闭合guarded摘要路径的全部logit/NaN/RNG读者及新代次退役。并对全模型递归callee建立明确已扫/未扫worklist，区分34未定输入谓词与尚未走读的分支腿，不能用有名字的缺口或122冻结约束宣布扫完。保留add-HC、shared-DFC、metadata/gid12、Engram owner与Markov整段候选，不重开旧固定小配方。源先定大交付后匹配旧C8/六普通，每替换原语逐位一致；无新Torch/NPU/模型/profile/请求/服务或远端写入。


## 当前断点 · 2026-11-11 MoE完整输出：SEDP已复制，保留BF16边界的add-HC值得定点验证

介绍：沿真实AscendMoERunner/shared量化与归约追到HC Post输出完成。到了哪里：18当前源、冻结serve白名单、13原Python factory/layout/完整shared opaque控制；旧两代表卡1032完整交付、原C8全16卡2560 add-HC对。结果：SEDP复制权重且无shared TP通信，缩放在路由权重侧；add→HC保留中间BF16舍入的融合值得定点验证。原122源码约束/34动态位未变，全链路未扫全，新原语/框架准入0。

已停止：把disable_tp=is_sequence_parallel直接推成权重已复制；按默认SEDP=False给冻结负载补出shared TP gather/归约；寻找不存在的后置scale；擦掉ForeachAdd的BF16舍入再合并HC数学式；只开multistream flag就认为DFC与shared可重叠。实际serve enable_shared_expert_dp=True；OOT get_parallel_op按shared开关独立决策，shared.parallel_mode为SP_SEDP。patch factory默认apply_routed_scale_to_output=False，TopK原生ABI接收routed factor，runner factor1；DFC配置强制shared multistream=False。

待确认、值得开发验证：完整ForeachAdd→HC Post原语。共享/路由BF16两输入先做原add的BF16舍入，再按HC DSplit原FP32 Mul/Add顺序及最终CAST_RINT产生残差输出；不能分配乘法/使用FMA擦掉中间舍入。有限IEEE例(1+2^-8)*3，原BF16 add边界结果0x4040，直接塌缩0x4041；这只否定通用代数替换，不是native对拍。M3/D5120逻辑上可省中间BF16写回+读回61440B/卡/call及一次launch；残差读、HC输出写、系数/下一层pre_mix/aux与DFC counts/潜在expertIdx变异都保留。完整native Foreach舍入/NaN/零符号、格式/tiling/别名/图捕获和全部输出逐位、增量成本/六普通回归仍缺。

先把整段算全：上轮5.746362%止于Post开始，不能当完整输出上界。补Post执行后，旧EP0/8 global24窗保留DFC的宽松包络均6.126260%；shared开始→Post完成4.445635%，Down-ready→Post完成1.498290%。Foreach本体1.066954%、HC Post0.379898%；若保留原HC Post成本、每call新增5us，480次局部均净7.572826ms/0.810240%该窗；新增20us只剩0.040098%。这是条件价，不是实测融合收益，外流allgather未归给shared。global20另列，不迁移target价。

请求另算：原C8首token16卡2560对，按每层最大卡信用同时给全体、固定原carrier/请求次序串行累计的极宽松模型，Foreach全免最大0.371417% TTFT，新增5us为0.316674%；add+HC全部免最大1.353714%，但含必须HC数学/写出，不可实现为零。该预算不是下一EP arrival模型，也不用于预测decode/makespan。六普通/全请求曝光仍未知，实际请求信用默认0；不把两卡steady百分比外推。

更大父段仍保留：保持原量化数学的shared与DFC联合调度/megakernel、共同输入读与双quant格式交付；已有DequantSwigluQuant已融合activation+requant，不能当未融合重复计。DFC和shared的quant舍入政策不同，不能直接共用量化Tensor。新增重叠须改当前mutex、证明EP参与/资源与跨流读写租约，不能复用“已打开”的假设。优先继续剩余路径，留下以上具体native/净价缺口。

[本段判断](evidence/MoE完整输出到HC_Post的融合与净价判断.md) · [源/候选边界](evidence/moe-output-review.json) · [13原源码控制](evidence/moe-output-controls.json) · [1032局部交付及原C8请求价](evidence/moe-output-price.json)。

下一条待扫：loop047 Run184748Z PLANNED：转M08真实DSpark _run_merged_draft的选行/LMHead/confidence→采样/拒绝→下一轮调度交付，追全输入、实际消费者和合法动态分支。已扫MoE输出源边界保留add-HC完整原语native逐位/格式/捕获/净价与大shared-DFC联合调度缺口；不重做不存在的shared TP通信/后置scale或已停小配方。34原动态位及递归callee未闭合，保留metadata/gid12、Engram owner/input候选。源先定大交付后匹配旧C8/六普通，每替换原语逐位一致；本源审Run不新开Torch/NPU/模型/profile/请求/服务或远端写入。


## 当前断点 · 2026-11-11 输入尾部到原生DFC：两套mask不同，padding省算不可重复计价

介绍：沿target/draft输入尾部追实际MoE有效行交付。到了哪里：30当前源身份，38原Python前缀/尾部/MC2行控制；从actual前缀到SP切片、量化payload和原生DFC mask写者。结果：target旧尾部可保留，DSpark query有真实用途；DFC已有独立有效行处理。原156位仍122受限绑定/34动态待追，全链路未闭合，新原语/框架准入0，每替换原语逐位一致。

已停止：按position=0/ID=0或图shape认整行恒零；按5个DSpark query只保留anchor；把router未读is_padding推成DFC没有有效行mask，再重复计算“新增跳过padding expert”的收益；盲删空DP整卡执行。target只写actual ID/embedding前缀，clamp只改负数，positions尾部清零不能清除非负旧ID/旧embedding。DSpark eager固定，query尾部显式填parallel drafting ID/position0/slot-1，与SP新补零行不同。

待确认：大row交付保留，但拆开已存在的mc2_mask与model is_padding。target实际count、draft实际query count经forward context生成mc2_mask，MC2按本卡原SP行再补DP长度，W8A8 payload传到DFC x_active_mask。当前DFC源核在init routing前把无效行expertIdx storage改为expertNum哨兵；不是-1。原生变异、expert counts、完整routed输出、shared输出/归约/scale/add、HC残差/系数、aux/MTP和下一轮租约都属于替换边界义务。torch schema仅声明out/count可变，源码kernel却写expertIdx；包装层到loaded op实际alias仍须核实，不能据此宣称已观测caller Tensor变异。

旧EP0/8 localM3窗的DFC全免均16.758165%、norm→Post全免均22.504527%；它们包含已有mask和必要数学，不是padding行可省比例。保留DFC必需时，parent span减DFC的极乐观包络均5.746362%（两代表卡，global24/480calls），仍含必要norm/router/shared/HC和等待，不能全计为候选收益。新row请求收益信用默认0；六普通及原C8请求曝光/净价待绑，不迁移不同cohort。

继续深入的边界：真实AscendMoERunner的shared/routed reduction→scale→foreach_add→HC Post整体交付，保持每一处BF16/FP32舍入和事件/collective顺序；不先假定把多个数学式合并就逐位相同。把已有mask可省之外的具体工作与完整原语输出闭合后，才给出开发准入。大metadata/gid12和Engram owner/input候选保留。本轮仅源码、行tag与旧数据，无新Torch/NPU/模型/profile/请求/服务/远端写入。

[本段判断](evidence/输入尾部与MC2有效行及MoE完整输出判断.md) · [buffer与真实消费者](evidence/row-delivery-review.json) · [38原源码控制](evidence/row-delivery-controls.json) · [分cohort价格](evidence/row-delivery-price.json)。

下一条待扫：loop047 Run182623Z PLANNED：继续M06真实AscendMoERunner.forward/_forward_impl→shared_experts各执行/等待/归约→scale/foreach_add→HC Post的完整输出；保留原生DFC有效行、expertIdx潜在变异/count与全部EP参与。先走较大交付再用匹配旧profile拆必要工作/新增开销/请求曝光，若整段乐观低才剪同预算子项。然后转下一未扫递归消费者；34动态原位/全callee仍未闭合，不重复122冻结约束或已停小配方。每替换原语逐位一致，无新Torch/NPU/模型/profile/请求/服务或远端变更。


## 当前断点 · 2026-11-11 模型guard与实际派发：122源码约束，34动态位未闭合

介绍：从5个模型/attention入口guard追SP mask、实际router、Engram旋转和generic runner/cache两条backend派发。到了哪里：27当前源身份、21进程初始单项env、85原Python selector/布尔行/opaque控制；新增3处，原156位122受限绑定/34动态待追。结果：5处源码条件已说明，不是5处原生人口闭合；全callgraph仍未扫全，新增原语/框架准入0，每替换原语逐位一致。

已停止：把未选中的分解vision/hash fallback当A3热路径再融合（当前cap已含原生hash/vision）；将A5 unrotated gate替当前nonNone rotation；寻找一个该generic selector源链并不需要的use_compress注入；原样重跑既有独立Engram Wkv/gate/常量小配方。Wkv有quant_config参数但描述仍FLOAT，旧原生任务BF16；初版quantized标签已纠正并保留原件，整段范围未缩小。

旧原C8 first-token匹配：当前6源与旧身份相交、量化描述SHA同2f4020；完整gate全免最大0.489683%TTFT，Wkv0.615615%，常量/重复widening0.037375%。Wkv+完整gate+copy1.115388%、再含紧邻HC Pre1.207905%仍包含必要数学/残差，不是预计净收益，更不是lookup/TP/host上界。50%保留工作情景见原arrival模型，既有有限M3/M15 native fixture按历史资格保留，本轮不重跑或新增准入。

待确认、继续：更大的跨模块row来源→route/dispatch→expert/shared→HC/aux交付。旧EP0/EP8代表rank、localM3形状窗中，global24整DFC全免15.979–17.537%，norm→Post均22.505%；这只是两代表卡旧窗口，非全16卡/请求上界。若真正删掉DFC时间2.5%且每call新增5us，局部均净1.516ms/0.162240%该窗；25%为36.759ms/3.932827%。时间份额不是padding行比例，未知默认0时新增费用为负收益；请求实际eligible calls/曝光和六普通未绑，新增row请求信用默认0，不迁移1025ms诊断或原C8/TTFT。

SP helper只补模型shape对TP的余数；输入mask=None、shape8时不会自动知道actual6。当前Ascend router/BaseRouter所读路径不直接消费is_padding，不能因flag默认True就认padding expert已被跳过，也不能未闭合runner/dispatch/可选reader便删发布。maskNone的profile路径与本卡query0不同，后者保留global cache准备、TP交换、补齐和O投影。generic runner源链选MLA sibling、V41逐层cache backend独立；其updatable predicate不是GQA路径，gid12该源歧义收窄，但捕获地址/全部读span、task reset/异常与native位/净价仍待确认。

已确认可实现：本轮0；旧封闭CPU租约原型1及历史有限native常量fixture分别保留原资格。source、初始env、opaque控制均不当loaded对象/真实事件或数值逐位验收；错误读路径、AST helper扩展、2代表rank误数及量化标签纠正均有记录。

[本段判断](evidence/模型guard与SP行及旋转Engram交付判断.md) · [122/34与实际消费者](evidence/model-gate-source-review.json) · [85源码控制](evidence/model-gate-source-controls.json) · [先验大row范围](evidence/model-pad-row-parent-primary.json) · [分cohort局部/条件请求价](evidence/model-gate-parent-prices.json)。

下一条待扫：loop047 Run174546Z PLANNED：沿真实V1 _prepare_inputs/_preprocess追target/draft ID、position、hidden尾部的所有写者/末读与行映射，到M06 router/dispatch/shared/HC/aux/下一轮消费者。区分actual/FIA/graph/DP/SP，不把is_padding写入当路由已跳过padding；先核大row交付的实际人口/完整输出义务。34动态原位及递归callee未闭合，不重扫122固定约束；保留大metadata/gid12和Engram owner/input候选。source-first整段边界后用旧C8/六普通，单独核费用/曝光，不跨cohort相加；每替换原语逐位一致。此源审Run无新Torch/NPU/模型/profile/请求/服务或远端变更。


## 当前断点 · 2026-11-11 worker状态与合法gate：119源码约束，37动态位未闭合

介绍：沿真实execute→状态更新/CPU copy lease→异步修正→metadata provider→forward/错误→12字段暂态→sample消费协议，排除固定配置不合法gate。到了哪里：24当前源身份；61原worker guard源码leg走读，13原方法opaque协议控制+13字面量/factory控制；新增29受限冻结约束，原156位119绑定/37动态待追。结果：源码leg不等于分支人口或完整callee覆盖，全链路仍未闭合，新增原语/框架准入0。

已停止：仅同配平C8详细诊断下，保持CPU计数生产/最终记账与GPU metadata原语不变的“删CPU位置影子/结构索引共享/早期等待后移”独立小方案。先定义完整父交付，再放宽到整个execute输入准备self，其prepare-Free仅1.687078ms/11循环（0.164512%诊断窗）；全命中/全转移/零新增的条件整批代理仅35.272ms/0.220889%。50%实删暴露、每frame新增0.02ms，局部净0.623539ms/0.060803%诊断窗；全命中/50%诊断转移的请求代理13.036ms/0.081637%。非实测收益，不迁移原不同prompt C8/TTFT/prefill/六普通；不作为删除CPU计数D2H整个协议、CP共同frame或gid12大段的上界，不与旧候选相加。

待确认：37位仍有动态输入/人口或后续参数缺口；其中32worker位已有条件leg记录，另外5个model/attention guard继续。递归input/preprocess/lookback/MM/DBO/cascade及数值callee尚须追到全部消费者。A3 V1 model-call错误会release metadata；preprocess/Engram准备错误在该try之前，不能把V2 join当V1全错误保证。空scheduler/有效批提前返回与全局活跃但本CP query0不同，后者仍有global cache准备与O交换/补齐；CPU H2D guard也不等于producer_done。gid12整段和大共同frame继续保留图读span、task reset/异常、完整packet/逐位与无探针净价缺口。

已确认可实现：本轮0，旧封闭CPU租约原型1保留。逐位门槛不放宽。控制架两次初版遗漏字段/误数13字段已按原源码修为12并留记录，不是后端错误或native数值失败。

[本段判断](evidence/worker状态协议与早期CPU坐标交付判断.md) · [119/37逐位与61源码leg](evidence/worker-protocol-source-review.json) · [13原协议控制](evidence/worker-execution-protocol-controls.json) · [先验整段范围](evidence/worker-position-parent-primary.json) · [整段放宽上界/条件价](evidence/worker-position-parent-price.json)。

下一条待扫：loop047 Run170051Z PLANNED：先核5个model/attention guard的初始化与实际读者（MoE skip-padding、Engram rotation、no-metadata/空query），再沿未闭合递归callee走实际_prepare_inputs/_preprocess/lookback/Engram及M03/M06/M08大交付。32个worker动态位已分列源码leg与运行人口，不重扫29个冻结gate。补actual OOT selector patch/use_compress注入与graph backend，影响gid12整段候选合法性。先独立源范围再匹配旧C8/六普通；保留大frame/gid12、全部CPU/producer/consumer租约与native逐位缺口。此Run无新Torch/NPU/模型/profile/请求/服务或远端变更。


## 当前断点 · 2026-11-11 target阶段draft metadata整段：值得深入，保留图与状态缺口

介绍：从真实13KV/26builder源计划独立追target阶段最后gid12的整次CP/global/local metadata交付。到了哪里：15当前源码身份、34原Python控制；16旧详细trace/176次last-of26，完整保留两次base与helper成本并去匹配标记边界。结果：墙钟中位1.663035ms，留时钟余量的prepare-Free15.223438ms/11循环，占诊断窗1.484485%，整段值得深入；原156位90受限绑定/66待追未变，不冒充新增全链路覆盖。

待确认：target只按本层prefix取值且actual draft会fresh build，但仍须核selected图backend/task注册/捕获buffer读span、私有buffer写后读、MM/offload字典读者、两provider task reset及异常恢复、native完整bytes与无探针净价。CM选择/unpad、rawphysical setter和真正draft producer保留。当前legacy FULL update因use_compress绕开，不等于捕获地址无读者。

局部条件净价：实删且暴露50%/新增0.1ms每frame，11循环净省6.511719ms，占该诊断窗0.634978%。同配平轻量15.967999s、229/231 metadata frame，只作条件请求代理：命中50%/诊断转移50%/新增0.1ms每frame，平均56.574ms/0.354297%；命中100%则136.148ms/0.852633%。未知命中/转移默认0；不是实测净增益，不迁移原不同prompt C8、TTFT/prefill或六普通，不与helper/后25长度投影/大frame相加。

已停止：在CM/rawphysical setter之前跳整group；只跳build却保留无reset的take（控制发现会返回旧task；同时删build/take须另证无读/提交）；删除earlier packet首owner；以旧native小岛价剪这段host parent。已确认可实现：本轮原语/框架替换0，旧CPU封闭租约原型1保留。全链路未闭合，每替换原语逐位一致。

[本段判断](evidence/target阶段draft_metadata整段交付判断.md) · [源/状态与34控制](evidence/target-draft-metadata-review.json) · [完整176调用/区间原件](evidence/target-draft-metadata-price-raw.json) · [局部/请求条件价](evidence/target-draft-metadata-price.json)。

下一条待扫：loop047 Run163605Z PLANNED：继续66原静态位及递归callee，从selected execute_model状态/DP空批与异步修正、provider submit/release及KV/EC/offload/Mamba/EPLB/DBO gate走到真实读者。gid12整段候选保留图backend/task注册、捕获读span、覆写前末读、CM/rawphysical、task reset/异常及native位/无探针净价具体缺口，不阻塞剩余路径。优先下一处未扫大交付，先独立源边界再匹配旧C8/六普通；每替换原语逐位一致。无新Torch/NPU/模型/profile/请求/服务或远端变更。


## 当前断点 · 2026-11-10 真实资源组与坐标退役：90源码约束，66位待追

介绍：沿实际Ascend cache patch/原spec与planner→V1实际QSL/物理地址→Engram hash与DSpark context/query读者，再核旧原生任务。到了哪里：30当前源码身份；756文件仅关键词盘点；原源码计划54资源/13KV组/26runner builder/1draft builder，19退役/CPUguard控制、2坐标域反例，3624活跃rank窗各1次扩展；新增13处，原156位90绑定/66待追。结果：停止不存在的多组重复、当前A3基线mask-ready等待后移和固定native小岛独立配方；更大CPU/scalar/coordinate frame继续。原语/框架新准入0，全链路未闭合，每替换原语逐位一致。

待确认：大frame的actual/FIA/draft正确视图、原physical/packet owner、CPU H2D读租约与producer_done+consumer_done、最早storage覆写、完整4096B packet/尾部和净成本/六普通曝光。V1 finally只release metadata，V2 finish/captured helper join不能当V1保证；当前A3 overlap默认False、savedserve未覆盖，无辅助流竞态观测。新增overlap/提前producer必须另闭合prep/model/错误退役。

已停止：重复draft公共生成（source1组且原C8每活跃窗1kernel）；当前不存在的mask wait后移；盲共享FIA padded req-map给Engram（尾行clamp不同）；原C8固定hash3kernel+唯一expansion及内部gap全免仅0.365136%TTFT/0.524144%整批，50%实删暴露/新增0.02ms每波条件净0.175865%/0.237651%，非实测，不作全host/Engram上界、不迁移普通负载或与大parent相加。已确认可实现：本轮0，旧封闭CPU租约原型1保留。

[本段判断](evidence/实际资源组与Engram_DSpark坐标及退役判断.md) · [90/66逐位账/消费者](evidence/input-resource-source-review.json) · [13KV/26builder/1draft源计划](evidence/input-resource-source-plan.json) · [原C8逐Task资格/价](evidence/input-kernel-windows-receipt.json) · [条件价格](evidence/input-kernel-island-price.json) · [19原方法控制](evidence/input-retirement-controls.json)。

下一条待扫：loop047 Run155707Z（PLANNED）：先核target阶段gid12 draft builder整段输出的数值/私有cache/first-owner task/FULL及后续draft读者，保留CM与原物理view，用旧逐call整段价判去留；再继续66位及递归callee，走selected execute_model状态协议、DP空批/异步修正、KV/EC/offload/Mamba/EPLB/DBO gate与metadata provider submit/release到真实消费者；区分合法动态源码分支审阅与运行人口。已绑定13KV/26runner/1draft、A3 Engram同步默认；保留actual/FIA/draft坐标域、CPU copy lease、producer/consumer双退役及完整packet位。先用旧C8/六普通，不重开已停小岛；无新Torch/NPU/模型/profile/请求/服务或远端变更。


## 当前断点 · 2026-11-10 实际embedding入口与首层静态段：77源码约束，79位待追

介绍：独立沿V1预处理→OOT注册→STANDARD TP8/FLOAT查表→持久input→首HC/QKV及DSpark公共/物理地址交付，再与旧范围相交。到了哪里：23当前Python源+1prior HF精确SHA，29原方法受限控制；新增7处，原156位77绑定/79待追；原C8 3624活跃rank窗的Qb/KVnorm末端全资格。结果：eager用外部embeds，FULL捕获rawIDs/None并执行内部lookup；SP mask入口等待与Engram producer仍属大交付。新原语/框架准入0，全链路未闭合，每替换原语逐位一致。

待确认、值得深入：大input/Engram/graph-entry交付原C8活跃请求gross2.356753%TTFT/5.058344%整批，含必要工作不能全删；实删且暴露25%、新增0.1ms/波，条件净0.555673%/1.142482%，50%为1.144861%/2.407068%。未知比例默认0，非已兑现。首HC到pre-RoPE单独gross0.710251%/0.456940%，不再含空query动态Oproj等待；静态cache仍缺native精确位/shape-key/热行成本/六普通整段价。453活跃DP组不等于454物理组，synthetic8rank未价；不迁移并发padding行比例或另cohort诊断。

DSpark公共IDs/positions/sampleindices不依赖物理组，slot依赖group table/size/DCP；5query为anchor+4noise，cad属性重绑不是旧Tensor退役。实际draft组人口/次序、全部末读、producer/consumer双退役和packet完整bytes继续待确认。已确认可实现：本轮0，旧封闭CPU租约原型1资格保留。已停止：旧lookup/copy/SP-only同预算小方案继续停；不以纯文本排除MM、不以runtime embeds推断FULL内部lookup跳过、不按5×实际请求认shape、不以BF16尾位容差放行。较大parent保留。

[本段判断](evidence/词嵌入实际派发与首层静态计算整段判断.md) · [77/79逐位账及29控制](evidence/embedding-static-source-review.json) · [原C8整段价](evidence/first-static-parent-price.json) · [源行段](evidence/embedding-static-source-excerpts.json)。

下一条待扫：loop047 Run150110Z（PLANNED）：继续79位及递归helper，从selected资源组构造走Engram V1 slot-cache hash输入、DSpark公共/每组输出到全部异步末读和最早storage写；含空/synthetic DP、FULL与错误退役。优先大input-frame交付，保留DP/TP与physical first-owner次序；先复用旧记录核人口/净价，每替换原语逐位一致，无Torch/NPU/模型/新profile/请求/服务或远端变更。


## 当前断点 · 2026-11-10 V1输入实际读者与packet边界：70源码约束，86位待追

介绍：从V1实际caller的默认参数、PP/async、Engram准备字典和QSL输入→packet交付继续扫。到了哪里：16当前Python依赖/11当前原生source；新增21处受限源码绑定，原156位现70绑定/86仍待追；原Engram方法16个stub控制、原base32个owner控制及3个反例、原GenMetadata12个本机C++写范围控制。结果：canonical输出QSL不代表所有闭包输入QSL已共享；5个packet key首owner与每物理组C2地址映射首owner不同。显式重绑incomingQSL的大parent值得深入，尚无原语/框架新准入，未扫全。

待确认：较大CPU/scalar/canonicalQSL/device长度frame，保留原物理映射/packet owner并核全部读者、输入不可变租约、producer_done+consumer_done及最早写前acquire；逐原语完整bytes、空query/尾部/错误/FULL和真实净价仍缺。原source packet struct只含整数，Gen分别写3600/3456B而wrapper各分配4096B，496/640B外部初始化未闭合，不能默许忽略padding或当NPU实测。MM预处理即使纯文本也可能先embed再copy，配置与实现待下一段绑定。

已有同serve SHA诊断：CP/base/helper互斥self-Free合计311.050016ms/卡/11周期，占独立1025.503ms窗30.3315%，含必要首producer/物理工作，不是全可删。假设实删10%/转移50%/新增0.1ms每frame，条件净1.314ms/frame、镜像配平C8轻量整批1.8924%；默认未知比例0，非实测，不迁移原不同prompt C8 TTFT或六普通，不加旧device-only/CPU原型收益。

已确认可实现：此前受限CPU原型1保留，本轮融合替换0。已停止：删除首全局QSL H2D（中间slot mapping/logits已读），只因batch命中就删各privateQSL而不重绑incoming（SMLA/QLI/C2反例）。这不停止按owner共享及更大frame；86位不是86个算子，递归调用图/全部消费者仍未闭合。

[本段判断](evidence/V1输入到原生packet的owner与完整字节判断.md) · [70/86逐位账](evidence/v1-input-source-review.json) · [owner控制](evidence/v1-qsl-owner-controls.json) · [packet写边界](evidence/v1-packet-write-extent.json) · [整段条件价](evidence/v1-input-parent-price.json) · [限定源行段](evidence/v1-input-source-excerpts.json)。

下一条待扫：loop047 Run133903Z（PLANNED）：沿继承的V1 _preprocess绑定supports_mm_inputs/配置convertor/embedding真实实现→持久input→首HC消费者；并追globalQSL/物理block-slot经Engram和DSpark per-group视图的全部末读/最早覆写。继续86位及递归helper，不以具名缺口当扫完；已有匹配profile优先，每替换原语逐位一致，无Torch/NPU/模型/新profile/服务动作。


## 当前断点 · 2026-11-10 全模型注册与继承绑定：49源码约束，107输入控制位仍待扫

介绍：从真实注册、C3/别名、冻结配置与逐层职责建立执行入口，再与旧范围相交。到了哪里：40当前源码、config43d3/serve c297未变；16方法解析与Python C3对照、40target角色/3draft与私有128专家TopK3标量控制；156原静态分支/循环/上下文位逐项记录。结果：49受限源码/角色/排除绑定，107仍未闭合，不是107算子或已完成路径。DSpark继承/别名与实际_propose已补，rawMTPbank存在但固定DSpark数值路径不消费；同预算旧terminal小方案不重开。新框架/原语准入0，全链路未扫全，每替换原语逐位一致。

待确认、值得深入：较大共同CP frame的CPU/scalar/canonical QSL/设备长度交付；冻结V1 executor不capture producer，A3 builder不启用metadata_in_graph，producer输入地址与model graph地址分开审查。不能据此跳过native位、全部读者/事件或输入租约，增加共享读者可延长末读；旧device-only2.098217%不是整parent上界。107处还须继续callsite/人口/消费者，不把源码行交集当搜索完成。

配置补充：原HF类体展平控制保留vision_n_layers32，而draft arch-info的is_mm_prefix_lm=False；两者不是同一字段。纯文本/DSpark不能推出vision-router bias不存在，实际HF/text/gate与token谓词仍待绑定。已确认可实现：仅上轮受限CPU原型1保留，本轮新增融合0。已停止：静态18入口当全覆盖、flag1当CANN MegaMoe、raw bank不存在、凭text-only删除vision-bias与同预算独立terminal copy小方案；更大parent保留。

[完整判断](evidence/全模型派发继承与合法分支覆盖判断.md) · [注册/C3/40角色](evidence/whole-forward-dispatch.json) · [156位与旧范围交集](evidence/whole-forward-coverage-delta.json) · [配置形态对照](evidence/whole-forward-config-shape-controls.json) · [源行段](evidence/whole-forward-binding-source-excerpts.json)。

下一条待扫：loop047 Run run-20261110T124045Z：继续实际V1 worker metadata/input的107剩余位，核输入Tensor来源、selected builder/资源组、canonical QSL与每个first owner→全部base/native/model graph消费者和CPU/producer/consumer退役及覆写；区分V2/A5和固定/动态条件。40源派发Run114154Z已PASS/NA；新Run仍PLANNED，配置两形态标量补充不当真实对象或原生位。只读源码/旧数据，无Torch/NPU/模型/新profile/服务动作。


## 当前断点 · 2026-11-10 共同设备投影命中后死工作：保留原首owner，转全模型分支覆盖

介绍：设备权威长度经CP投影→原base缓存→C2/native与最终消费者。到了哪里：17当前源/11限定body/18行段、39个common读取点，原base符号哨兵960例与miss反例；16旧trace共4576次目标调用/4400命中段。结果：已有batch命中后的25段新长度值无消费者，原首投影/持久复制保留，独立补丁固定保留各QSL但未证明每份必要；精确补丁可审查但未安装。缩入时钟余量后平均45.234547ms/11周期Free，转移.5/新增.1ms每周期条件净2.098217%独立诊断窗。无原生/框架新准入，全链路未扫全，每替换原语逐位一致。

待确认，值得开发：已有local batch hit的死投影删除；源码边界闭合，仍缺私有clone真实Torch/NPU/原生packet和地址/空尾与capture逐位、无探针成本、全请求与六普通人口/转移。已确认可实现：上轮受限CPU原型1保留，本轮新增框架替换0。已停止：删首投影、CPU上界替设备值、无owner判别删全部QSL、writer-only异步化、跨代cache与空query跳QLI的具体配方；按owner有条件删非首QSL/pin/H2D及CPUscalar/setup的大parent保留，此投影价不当整段上界。

请求另列：同镜像配平C8轻量参考15.967999s，真实metadata frame229/231；人口.5/转移.5/h.1ms的条件整批省1.336721%，人口1则2.817479%，默认未知比例0，非实测，不迁移原不同prompt C8/TTFT/普通。CPU微基准、不同窗口及并发时间线收益不相加。

[完整判断](evidence/共同设备长度投影的死工作与真实消费者判断.md) · [原源码消费者](evidence/common-device-source-review.json) · [精确待审补丁](evidence/common-device-hit-elision.diff) · [局部与请求条件价](evidence/common-device-price.json) · [下一段静态入口](evidence/whole-forward-entry.json)。

下一条待扫：loop047 Run run-20261110T114154Z：沿当前模型/DSpark/Engram/attention的18个静态入口递归解析callee与合法guard，绑定冻结flag后再与同SHA旧审阅范围相交，分别记录源码已读/分支人口/消费者/原生证据；挑最大未扫交付。入口Run114347Z已PASS：16当前源、295语法callsite/156分支位，均不当唯一算子或执行覆盖。只读源码/配置/旧数据，无Torch/NPU/模型/新profile/服务动作。


## 当前断点 · 2026-11-10 共同frame分层key与26caller：CPU原语受限通过，继续device读者

介绍：共同frame按rawCPU/postCPU/device/每资源交付分层，追真实caller、变异和图地址。到了哪里：12当前源/11限定body/29行段；16worker旧CPUparent日志确认每frame目标26+draft1；原CPU AST标量1728例，实际TorchCPU1728例/44928字节比较，guard变体3模式/78比较。结果：封闭不可变租约内CPU复用原型已确认可实现，服务框架替换准入0；v1 inference Tensor版本读取失败原件保留，模式guard通过但不证明不可变。CPU v1六组26call配对净省2.226–2.445ms/frame，非请求收益；真实caller/保留首body的新条件价1.964%/3.275%诊断窗。全链路未扫全，每替换原语逐位一致。

已确认可实现：仅封闭CPU输入租约下的隔离原语原型，含optional-out旁路和inference-aware key；未安装到服务。待确认：框架实际集成/全部保护负载、较大公共device投影的所有读者/地址/原生位与关键路径净价。已停止：全局helper cache漏out写、CPU双mirror合并、shape/版本号当不可变证明、遗漏TP/causal/dtype/actualTokens，以及假定V41draft继承Legacy缓存或有26次重复。旧大frame异步writer-only配方仍停，不剪原主流完整方案。

[分层key、caller与CPU原语](evidence/共同frame分层键与真实caller及CPU原语判断.md) · [CPU模式资格补充](evidence/共同frame_CPU模式资格补充.md) · [真实caller与付成本价](evidence/common-frame-caller-price.json) · [实际CPU字节/局部成本](evidence/common-frame-torch-cpu.json) · [inference反例与guard](evidence/common-frame-cpu-modes.json)。

下一条待扫：loop046 Run run-20261110T105846Z：一次原主流CP suffix/clamp/where→base首batchcache与全部common字段/原生closure读者→稳定QSL和capture/event firstowner。保留各builder QSL复制与物理输出，查是否能只删重复device长度投影；只读源码/已有profile，不新增NPU/模型/服务动作。CPU模式门槛Run110329Z已PASS，不能以guard省略输入租约。


## 当前断点 · 2026-11-10 DFC原路由与末读闭合：转有函数归因的共同metadata大交付

介绍：DFC原mask/排序/inverse/count发布→远端搬运→GMM与Unpermute末读，再转同代共同metadata/frame。到了哪里：11当前DFC源/10限定body，454完整EP16路由组/7264ranklayer调用、80原生inverse witness、21792纯字节映射；5当前metadata源/4入口body与新16rank函数自身账、144付首producer/新增成本情景。结果：原R与dummy/zero-group flags保留，删gmA写不等于释放activation arena；历史5路由免费输入1.543–6.922us低于旧小预算，原样输入-only维持停止。新metadata仅helper共享的条件净1.995%诊断窗，CP自身25%也共享为3.622%，全base保留；非请求预测。新准入0，全链路未扫全，每替换原语逐位一致。

待确认：改变GMM读法/提前publication/largeM/shared资源的原生位、phase/资源/末读与净价；metadata共同字段的实际caller兼容性、不可变CPU快照、device权威长度及每资源发布、图事件和原语位。已确认可实现：本轮0。已停止：历史固定GMM/flags的M3输入-only，sorted position/topK代原行、删dummy inverse/zero-group flags的具体配方；跨代shape-only或跨物理组整metadata复用仍停。保留不同GMM/大parent，未把旧排序参考改成用户门槛。

[DFC路由与末读](evidence/DFC原路由行映射与GMM末读判断.md) · [共同metadata函数自身与净价](evidence/共同metadata函数自身归因与保留首生产者的净价.md) · [454路由/5原生旧实验](evidence/dfc-row-map-delivery.json) · [144条件价](evidence/common-frame-measured-price.json)。

下一条待扫：loop045 Run run-20261110T100641Z：同代frame最小字段key→CP/global/localbase实际caller与闭包→CPUhelper返回值mutation→每资源固定deviceQSL/slot/table/C2/nativeattrs发布；复用原acquire-before-mainwrite及CPU/producer/consumer双退役，价实际保留工作和新增成本。只读源码/已有诊断，不做NPU/框架进口/新profile/服务动作。


## 当前断点 · 2026-11-10 Engram全局owner付成本剪枝与DFC输入行共享：继续真实路由读者

介绍：全局Engram rows→WKV/gate/HC付成本判断，再沿较大DFC的BF16输入/量化/路由/跨EP搬运追到首GMM。到了哪里：Engram9源、56几何/4字节oracle、108C8/216短窗价、544比例与六普通代理；DFC9源18限定行段、原host DoTiling CPU12形状、1032匹配父段。结果：Engram仅明确小预算子情景停止，大parent待确认；DFC单核已共享但跨核仍重复，localM3有18次量化/3输入行，prefill416仅多20次，不能统一取6倍。targetDFC占自身两窗15.979%/17.537%，删整段10%另付h5条件净1.343%/1.496%窗口，量化内部占比未知。新准入0，全链路未扫全，每替换原语逐位一致。

待确认：更大Engram lookup/globalowner/WKV/gate/HC的原生位/资源/传输/末读与请求曝光；DFC稳定行映射、mask/dummy/zero-work完整分支、peer/gather或GMM A loader、原量化位与内部phase和新增成本。已确认可实现：本轮0。已停止：Engram已明确低于旧排序预算的独立子情景、N512/1024单AIC八panel同驻、flag-only分片及错误8倍MAC/B读取信用；DFC全形状unique-once/每路由独立重算的错误信用、把整个DFC当量化phase。只停止这些范围，不剪更大parent。

[Engram全局owner与付成本价](evidence/Engram全局行Nowner几何与付成本请求判断.md) · [DFC输入行与首GMM判断](evidence/DFC输入行量化共享与首GMM交付判断.md) · [原host回放](evidence/dfc-input-host-cpu.json) · [DFC完整父段价](evidence/dfc-input-delivery-price.json)。

下一条待扫：loop044 Run run-20261110T094003Z：原expandedRowIdx sort/active mask/dummy/count生产→unique sender row map→远端gather→receiver连续gmA与GMM首读；保留原flags/容量/K/Fixpipe/舍入，追bank末读，付新map/gather/并行度成本。先用旧profile，不做NPU/框架进口/新profile/服务动作。


## 当前断点 · 2026-11-10 HC双parent跨层与FULL图：转Engram全局行和WKV大交付

介绍：HC系数尾部到跨层/Engram/aux/final读者及FULL事件，再转全局Engram rows→WKV→gate→HC。到了哪里：15当前源/11限定body，两入口AST seed，6有限正常DAG/2004顺序对及3反例；144双parent价、16 PMU压力价；另9当前Engram源/6入口body、2权重header，payload0，每rank24条WKV+37gate完整链/54价格。结果：纠正实际首层None解释，target和DSpark首层均有premix；ownPost前整作业join支配后续读者。双parent无争用条件2.672%/2.740%窗口，PMU代理可降至.1%–.4%或负，不能当实测新争用。Engram大段全免1.169%/1.176%窗口，半省h5仅.572%/.575%；旧独立小预算不重开。新准入0，全链路未扫全，每替换原语逐位一致。

待确认：HC native位、独立流/FULL捕获/allocator-workspace/真实争用；Engram全局owner/WKV/gate/HC联合原生几何、原语位与付成本净价。已确认可实现：本轮0。已停止：实际首层当None、借无关ExternalEvent作coeffdone、缺producer fence或跳过/异常不join的具体配方；旧WKV-only/gate-only/inactive-passthrough与原样overlap/staged-prefix停止保留。

[HC双支路读者与FULL图](evidence/HC双支路系数跨层读者与FULL图判断.md) · [双parent/PMU价](evidence/hc-both-delivery-price.json) · [9源Engram入口](evidence/engram-joint-entry-review.json) · [完整WKV/gate价](evidence/engram-joint-entry-price.json)。

下一条待扫：loop043 Run run-20261110T084215Z：现存全局complete-K Engram bank→保留原M3 padding/K/Fixpipe的WKV Nowner几何→新row-restore AllToAll→rotated gate和持久HC残差；比较direct-SP-owner的冲突，先用完整原C8/普通/prefill价大parent。只读源码/旧数据，不做NPU/框架进口/服务动作。


## 当前断点 · 2026-11-10 O投影资源剪枝与HC premix提前交付：继续真实系数和跨层读者

介绍：OA/OB→SP→HC/FFN大交付，再沿实际HCPost/HCPreV3/RMS走到MLP和FFNPost。到了哪里：37O投影源/23限定body、20HC源/17限定body，两rank各516条源一致链，144 parent价、108资源价、144 fork/join价；原profile六条1800MHz记录。结果：当前M24单owner81.636us下界高于全部旧OA+OB，停止该schedule及擦掉OA舍入的预合成；多owner/大prefill待确认。premix存在时collapse不依赖本次新系数，提前交付e5/h5、无争用条件净1.340%/1.368%窗口，40us MLP争用会翻负；原BF16残差、collapse与RMS舍入/FP32扩宽均保留。新原语准入0，全链路未扫全，每替换原语逐位一致。

待确认：多owner OA/OB的活跃key、原语位、聚合服务与净价；premix提前交付的原系数尾部/跨层代次/独立事件/FULL图/实际MLP争用；大prefill和普通请求曝光。已确认可实现：本轮0。已停止：擦掉OA舍入的普遍预合成、当前M24单owner全部OA/OB、原kernel间gap-only小范围；舍入前FP32直接喂router、删除后续FFNPost残差、无premix原样提前或跳过活跃coeff。只停止具体变换，不剪大parent。

[O投影原生资源与大段价](evidence/O投影原生资源与SP残差FFN交付判断.md) · [HC原生与提前交付](evidence/HC原生舍入与premix提前交付判断.md) · [144 fork/join情景](evidence/hc-early-delivery-price.json)。

下一条待扫：loop042 Run run-20261110T080543Z：HCPre stage2 post/comb/Sinkhorn/newpre到下一decoder、Engram与final读者及实际多流FULL图split/join；保留原coeff完整工作、原舍入/归约，查能改变去留的事件/资源/争用边界。仅只读源码/旧profile，不进口框架、不做NPU/服务/采样。


## 当前断点 · 2026-11-10 共同frame双端退役与packet净价：继续O投影大交付

介绍：同代CP共同frame沿metadata producer/consumer走到下一代覆写，并补head8 query packet成本。到了哪里：15当前source identities、11限定body行段，Python回调提交时点与device/CPU存储寿命分开；20场景1033顺序模型、两rank各12帧/144成本情景；另7当前OA/OB→SP→mHC/FFN入口源已核。结果：release-only/devicewait-only pinned复用缺边，明确acquire须早于runner持久输入覆写，CPU快照与producer/consumer完成分别保护。三目标packet旧参考616–620us/迭代，不按层重复；替换与共存成本不同。大frame/head候选保留待确认，转O投影大段；新原语准入0，全链路未扫全，每替换原语逐位一致。

待确认：common frame的native事件代次/alias/逐原语位与净价，head全局packet/压缩mask/graph资格；较大OA/OB→SP→mHC/FFN原生几何与净价。已确认可实现：本轮0。已停止：release-only输入复用、devicewait-only pinned源复用，以及挪走commonwriter但保留eager依赖读者的原样方案；不停止完整owner/fence或更大parent。

[真实覆写/双端退役与成本](evidence/共同frame的真实覆写与双端退役判断.md) · [顺序模型](evidence/metadata-frame-order-cpu.json) · [packet价格](evidence/global-query-packet-price.json) · [下一段入口](evidence/o-projection-native-entry-review.json)。

下一条待扫：loop041 Run run-20261110T070506Z：实际OA TransposeBatchMatMul、所选OB BF16的TorchNPU/CANN tile/fixpipe/舍入→SP collective→mHC/FFN双读者；保留大prefill原M/全部权重集与rounded残差，按匹配旧数据价整段。7源入口/局部调用链不当native准入；不重测已停M3 OA/小逆RoPE打包。只读源码/已有数据，无新NPU/框架进口/profile/服务动作。


## 当前断点 · 2026-11-10 head布局分支与共同metadata：继续真实寿命和消费者

介绍：贯穿Qb→全局query SMLA→inverseRoPE→TP O投影，并转向同代CP共同metadata交付。到了哪里：17当前head/native源、原标量25packet fixture与有限坐标；两个代表rank各480目标样本拆456CSA/24纯SWA；另9当前metadata源追到真实helper、base共享和SMLA/QLI闭包。结果：CSA的KV逻辑访问扩大8倍、Cube M padding2倍，纯SWA不同；原条件净价须付新增成本。共同plan仍有大范围价值，但31ms提交前边界非全可删或精确函数；首段源码/72条件价格已记。新原语准入0，全链路未扫全，每替换原语逐位一致。

待确认：贯穿head布局的native位/metadata/NZowner/资源/请求净价；共同frame的alias/双完成栅栏/实际归因与净成本。已确认可实现：本轮0。已停止：cache-only packet借作globalquery、跨物理组整对象或shape-only跨代缓存、CPU上界替device长度；不剪完整动态协议或更大parent。

[head分支与净价](evidence/head布局贯穿SMLA的分支与净价判断.md) · [共同metadata实际生产/价格](evidence/同代CP共同metadata的实际生产与价格判断.md)。

下一条待扫：loop040 Run run-20261110T063608Z：共同QSL/CPUlength/device length/positions的全部observer与mutation、闭包attrs，canonical frame在主流覆写前的producer_done+consumer_done栅栏；emptyquery/emptyDP/error/abort/FULL事件集合与顺序。保留每资源slot/ring与原native attrs，沿实际读者补证后继续下一交付。仅只读源码/旧数据，不重采、不操作服务。


## 2026-11-10 QKV原生次序与head布局贯穿Attention：继续较大consumer

介绍：沿真实Q/KV输入、Wq_a/WKV、QR量化、Wq_b到SWA/query读者，扩大到head布局贯穿SMLA并直接交付TP O投影。到了哪里：当前源码/量化描述/142个checkpoint header（payload0），两代表rank各516完整Q/KV段、480目标样本和96真实QLI producer；CANN pertoken/basic/opt舍入边界。结果：保留原三GEMM、其它全免仅1.020%–1.025%窗口；只拆Qb仍被SWA挡住。贯穿SMLA删除原transpose/A2A、Qb取1/4耗时、用旧AG参考付新通信并另付h5us，SWA不变条件净1.228%–1.299%窗口，SWA半省为2.646%–2.723%；不是请求预测。INT32 K1280无溢出不代表最终BF16等价，CPU反例0x44d2对0x44d1证明scale重结合不能盲换。新准入0，全链路未扫全，每替换原语逐位一致。

[当前原生/付成本判断](evidence/QKV原生反量化与贯穿Attention的head布局判断.md) · [有限公式反例](evidence/qkv-dequant-order-cpu.json) · [SMLA head8入口](evidence/attention-head-layout-entry-review.json)。

下一条待扫：loop039 Run run-20261110T054436Z：GSize8全局query host/AICPU packet→arch22 QK/PV行/head地址、原MMAD/softmax→inverseRoPE→原TP O投影布局；全局selected-ID的96次producer收集/同代复用、common TP24资格、zero-query/draft20/ordinary回退与图collective顺序。8当前SMLA/CP源SHA匹配，两处head guard允许8只属入口条件，不是完整合法性/原语位准入。继续完整付成本范围，不把新短窗或35MiB读范围兑换成请求/容量收益。

待确认：head布局贯穿Attention/O投影的逐位、global query/selected-ID代次、资源/graph和请求净价；已确认可实现：本轮0；已停止：两个当前窗固定cache到达的Qb-only和普遍scale重结合，不剪更大parent。compiled source/guard/CPU反例均不当NPU或模型验收。


## 2026-11-10 原生缓存发布与SWA主辅流join：局部3%不等于净省

介绍：真实OOT/native的indexK/longKV双发布扩大到SWA整段与Qb重叠。到了哪里：39当前source pins/29非空ranges；两个代表rank各48条十算子链、516条SWA三算子链与Qb/queryRoPE/SMLA一一关联，已有完整native CSV各35664行。结果：十算子链全免仅.427%–.432%窗口，独立旧小预算继续停止。SWA局部全免3.065%–3.080%，但固定Qb到达的join最多仅提前.648774%–.660879%窗口；不把局部3%当收益。保留RMS BF16与RoPE BF16边界、FP32三角表、原乘乘加及stride/skip；SK已融合地址计算。更大WKV/Q/input-ready资源与数学共享待确认，本轮准入0；窗口不是请求收益，全链路未扫全，每个替换原语逐位一致。

[原生交付与局部价格](evidence/原生缓存发布与SWA整段融合判断.md) · [Qb重叠与join净价](evidence/SWA主辅流join与较大输入交付判断.md)。

下一条待扫：loop038 Run run-20261110T050354Z：沿CVLinearWrapper/实际层方法与量化守卫→TorchNPU npu_quant_matmul→CANN9.2 QuantBatchMatmulV3 tiling/epilogue→FP32-trig query RoPE→SMLA实际读者。7当前磁盘source pins/5匹配，CV13–189与CP/量化入口已读；CP local-query/global-cache metadata不同，不能盲共享全张量。继续较大Q/KV整段价格和精度/布局/资源。owned容器最近读到exited，同image/id；用docker cp stdout只读源码成功，不启动/干扰其它采集，不冒称当前live服务健康。

状态：待确认——更大Q/KV/input-ready与原生逐位/资源/请求净价；已确认可实现——本轮0；已停止——旧独立index尾段，固定Qb的SWA-only不以局部3%立项，不剪ordinary/任意资源重排或更大parent。全部大trace/逐链完整原件留本地，公开summary projection明确列SHA，不冒称全原件公开。


## 2026-11-10 Engram成对owner与lookup整段停止边界

**配平数据补充**：旧loop036原件冻结后获知：配平复采已完成并恢复原服务。新集合为前4条prompt镜像，不能直接沿用旧8条请求比例；32worker窗口中steady lookup union中位约0.03034%/0.03029%。四个native算子族全部站点的松包络约5.009%/5.083%窗口，含其它target/draft，下一步需绑定实际indexK/longKV调用，不能作为候选收益。 [分负载补充](evidence/engram-storage-balanced-supplement.json)。

**Goal ACTIVE；loop036限定source/原生旧价PIVOTED，loop037 RUNNING。全链路未扫全，每个替换原语逐位一致。**

介绍：Engram checkpoint code/E8M0scale→成对加载→共享UVA注册owner/pointer bank→lookup/dequant与退役。到了哪里：12当前source/header pins、11非空ranges；3236Python字面caller搜索；2当前I8/E8M0header且payload读0；7248原native lookup/906TP8group与固定C8端点匹配。结果：weight回调成对写、scale回调no-op；910C flags为MAPPED|PINNED，normalclose有同步/注销失败保留owner，但热graph/reload与半初始化事务仍未证明。仅改dtype/开MXFP8/热复用旧ptr非法。先价整个lookup：全免且替代0，TTFT条件最多1.761056ms/.118045%，整批最多8.599814ms/.046259%；同旧1%固定延迟预算的compactscale/groupbroadcast/lookup-only开发停止，不再补小NPU实验。共享host请求payload差18,432,548,400B单列，非实际释放RAM或NPUKV容量；较大input-owner/WKV/iteration仍待确认。本轮新原语准入0，全链路未扫全，每个替换原语逐位一致。

[整段判断](evidence/Engram成对UVA寿命与整段lookup价值判断.md) · [源审阅](evidence/engram-storage-source-review.json) · [当前前沿](evidence/engram-storage-frontier.json) · [完整旧价与906组记录](evidence/engram-storage-price.json) · [原件/摘录范围](evidence/engram-storage-public-source-manifest.json)。未做新原语NPU验收，不再为已低值lookup小范围补证。

loop037 Run run-20261110T035955Z：indexK/longKV的RMS→inplacepartialRoPE→DynamicQuant→code/halfscale scatter实际原生交付。6当前source匹配；已读indexer.update_keys、AscendRMSNorm.forward_oot无residual与bias_loaded分支、DeviceOperator旋转/散写和ops导入。继续OOT注册/实际选中callee、TorchNPU/CANN/自定义kernel tile/舍入、slot/cache代次与末读。归入已有input-ready/query/SWA较大parent，条件全免2.1327%/half-h20us1.0449%不是小epilogue收益；先整段净价，不干扰配平profile控制器。 [新入口](evidence/cache-epilogue-entry.json)，Run仍PLANNED。此源码扫描无新NPU/模型/请求/profile/显式远端写入/服务动作，独立配平控制器未操作。


## 2026-11-10 原序L1与尾块联合判断、Engram下一入口

**Goal ACTIVE；loop035限定source/CPU/条件价格PIVOTED，loop036 RUNNING。全链路未扫全，每个替换原语逐位一致。**

介绍：selected IDs/可见长度→原V0物理pair/半区压紧→三槽GM panel→QK/PV真实L1交付。到了哪里：11实际source/限定ranges，333原V0 CPUfixture/46184行地址；12旧nativeQLI×PA情景与1239552活跃NZ坐标；54旧价复现、108联合与20指令成本情景。结果：910C ND2NZ srcDValue为uint16元素，物理行距128会溢出，保留64位base并按原顺序拆分；朴素逻辑序104例不符，7例活跃洞回退。发现两块L1保留末128行、第三块work的较大联合方向，需跨QK(i)/PV(i-1)任务及新flags。K512/.75参考/h5us/12B factor1、全尾合资格时最有利请求条件1.6663%，非P90/实测，负值保留；高指令数可转负。原生ND2NZ padding/LoadData/MMAD逐位、真实K/页代次/分组、预取与净价仍待确认。停止逻辑序直替、A5宽stride套910C、原7块全驻K512与旧stockGM prepack的具体配方。本轮新增原语0，全链路未扫全，每个替换原语逐位一致。

[整段判断](evidence/SMLA原序直达L1与尾块复用判断.md) · [源审阅](evidence/smla-direct-l1-source-review.json) · [待确认/已停止前沿](evidence/smla-direct-l1-frontier.json) · [全部价格](evidence/smla-direct-l1-price.json) · [原件与公开范围](evidence/smla-direct-l1-public-source-manifest.json)。CPU证明不作NPU原语逐位准入。

loop036 Run run-20261110T032750Z：Engram紧凑scale checkpoint→host/sharedUVA注册owner→device pointer bank→lookup/dequant→末读/close/reload。3当前source SHA匹配；已读storage/scale类型分配、I8与MXFP8加载分支和lookup入口。继续SharedUvaBuffer注册/失败回滚/各rank close、成对code/scale bank寿命和图稳定地址；保留原INT8×FP32→BF16与255 NaN位合同。只为会改变18.4325GB容量/大交付净价去留的参数补证；不干扰进行中的配平profile控制器。 [新入口](evidence/engram-storage-entry.json)，Run仍PLANNED，不当成已完成。此源码扫描无新NPU/模型/请求/profile/显式远端写入/服务动作；本机另一个已授权配平profile控制器未操作。



## 2026-11-10 SMLA分核包与付成本原序计划

介绍：沿长度/掩码/平台/确定性→AICPU分核包→arch22 FA/FD与QK/PV实际读者，判selected-KV大交付。到了哪里：21实际源pin/18限定ranges；原标量323–1631的7个本机CPUfixture；1840原C8 native/2560窗/30typed组，6旧值精确复现、54付plan/协调成本情景。结果：packet已按batch ratio共享，900词已定义/124尾词未写；默认整行，q3 level3可生成6FA/3FD/48vec。更正旧“未传FLASH_DECODE”原因：入口会传，但host/key选择FD0；未继续消费的是BATCH_CONSISTENCY算术。停止flag-only安全跨S2和重复38层metadata信用；不是服务故障/NPU精度结论。12B计划/.75稀疏参考/h5us，K256条件最有利请求.727%–.875%，K512为1.101%–1.395%；后者继续深入。13负请求信用保留，均非实测/P90。新增原语准入0，全链路未扫全，每替换原语逐位一致。

loop035 Run run-20261110T023853Z：原序selected-KV64位地址+transfer计划→Cube QK/PV精确L1布局。已读Vector pair/large-stride fallback、Cube panel ND2NZ和DataCopyPA片段；继续vendor Nd2NzParams字段/stride/padding、PV布局/LoadData/flags、原ring与源/page/index代次，先找匹配旧搬运数据，再只补影响去留的门槛。保留原metadata/逻辑TopK及所有MMA/softmax/FP16-BF16边界。

待确认继续开发：保留原metadata/逻辑TopK和全部数值边界，原序selected-KV计划→Cube精确L1；K、ND2NZ/PV/flags、稀疏issue率与逐位是实际门槛。停止同K256/.75/h5情景不剪大parent。[判断](evidence/SMLA分核包与原顺序交付判断.md) · [具体缺口](evidence/smla-schedule-frontier.json) · [下一入口](evidence/smla-direct-kv-entry.json)。


## 2026-11-10 原生workspace容量与大融合

介绍：从实际GetWorkspaceSize→Byte empty→rawptr/OpCommand→ACLNN/设备末读，判断容量、共享和大融合。到了哪里：安装TorchNPU声明commit绑定24参考源，15文件限定ranges；9安装header逐SHA一致，16worker初始expandable_segments:True；原rank0 DB无分配/释放或PYTORCH_API表。结果：原NPU empty走同流CachingAllocator；scope析构不等于设备完成/物理释放，queued闭包没有workspace Tensor lease；capture用private pool且replay不逐节点重跑empty。停止逐层容量累加、captured每步malloc收益与scope末读计法。分桶DFC/QLI缩容量支持更大融合仍待确认，缺全sender界/实际ACLNN-inner scratch/全流末读/物理峰值/逐位与净价。大DFC旧共同尾段全免条件P9020.09%–22.69%，阶段占25%/半省/h5us为2.458%–2.783%，容量本身不兑换成该收益。新增已确认原语0；全链路未扫全，每替换原语逐位一致。

loop034：SMLA metadata AICPU的1024×INT32分核包→arch22 FA/FD字段和workspace末读；8实际source SHA匹配，已读adapter35–89、AICPU Compute/Prepare25–110与packet布局，BalanceSchedule/GenMetadata、actual core attrs/合法determinism/空KV/ori_sparse及读者仍待扫。36/72只是源码packet容量常量，不代表实际launch核数；不按小metadata成本剪掉selectedKV/QK/PV大parent。

**范围更正**：三处审阅尾行标签超过EOF，已按实际文件长度更正；不存在行不计覆盖。[审阅v2](evidence/native-workspace-source-review-v2.json)与[原记录更正](evidence/native-workspace-source-review-correction.json)保留原Run证据。

待确认开发：按全sender界分桶缩容量，先核实真实pool/峰值和全流末读，再判断能否支持大DFC/QLI融合；停止逐层声明累加和captured每步malloc的收益计法。新增已确认可实现原语0。[详细判断](evidence/原生workspace寿命与融合容量判断.md)。

## 2026-11-10 DFC原生交付与24/48资源更正；继续workspace实际分配

**介绍**：沿receiver行→GMM1 FP16→SwiGLU INT8/scale→GMM2 FP16→combine追真实交付、别名和同步，再价整段。

**到了哪里**：31实际source pin、所列ranges；16旧DB逐SHA及2,560原eager任务时间/globalId精确匹配，全部 `MIX_AIC / blockNum24 / mixBlockNum48`。8 typed成本和24专家计数逐窗绑定。旧“DFC固定20/40留4/8空核”解释撤回：40是内部routing tiler参数，wholeDFC从平台取资源。独立shared4/8的原语字节/发核事实保留，不能推共驻。

**结果**：已停止——依赖固定空闲4/8、当前arch2201直接L0C→UB/换TSCM名就删GM、仅compact4608/把既有别名当四bank节省的配方。C1/C2已同640MiB、A/activation已同320MiB；2171计数窗有潜在WAR见证，960MiB公式不是已释放显存，不乘层/卡。原FP16、2304维max、clamp/Exp/Div/RINT和scale末读都保留。

**待确认**：大DFC共同尾段全免条件P9020.090%–22.690%；若相关曝光阶段占25%、半省/h5us，为2.458%–2.783%。半省/h20us要同1%排序参照，阶段需占10.70%–12.09%。这些是固定原调度情景，非实测/预计；计数不等于阶段时间。第二V2C等待可研究移到真实group21消费者前，但需实际DMA/旗标/位一致与曝光。显式24→20让核若多花20%×Cube敏感阶段，并非免费空核；全局负收益已保留。六普通/完整decode与实际阶段时间仍缺。已确认可实现本轮新增0。

**下一处**：loop032有限PIVOTED，loop033 RUNNING；6实际source pin，已读 `GetWorkspaceSize→at::empty Byte→rawptr capture→OpCommand.SetCustomHandler/Run` 入口。继续queue/allocator/stream/capture/现有memory记录和末读；不凭tensor作用域断言use-after-free，也不凭声明容量断言物理释放。无新NPU、模型、请求、profile、显式远端写入或服务改动。全链路未扫全，每替换原语逐位一致。

[介绍、到了哪里、结果](evidence/DFC原生交付与24核资源更正.md) · [资源更正证据](evidence/dfc-resource-correction.json) · [净价前沿](evidence/dfc-gemm-tile-frontier.json) · [实际分配入口](evidence/native-workspace-entry.json)

## 2026-11-10 latent双缓存整段净价；继续DFC专家GMM内部

**介绍**：从四KV源输入追到FP32投影/pool/ring、BF16 latent、index-K量化双发布和同latent原地long-RoPE/发布，先按整段再判小配方。

**到了哪里**：256首token源窗，224发布，其中40窗query0仍更新cache；133个typed子集条件匹配。另112个完整decode诊断周期、448源窗、38条件；它们是candidate周期样本，非全C8请求/六普通。实际C2为token-aligned清零后pool、原独立RMS；pool读旧ring必须早于尾写，wk末读前不能在另一流原地RoPE。native RMS/quant/scatter内部未读，未用pin冒充覆盖。

**结果**：已停止——同独立1%预算的尾段（span全免条件P90 .2802%）、保留原GEMM的其它计算/间隙（.4856%）、仅挪早生产段理想重叠（.7990%）。完整root含cast/zero/span全免1.1518%，half/h20us .5545%；扩大到含保留query/SWA的输入ready联合parent为2.1327%、half/h20us1.0449%，仅待确认，须补真正ready/资源/位一致/通信净成本，不与旧C2/host/QLI/SMLA相加。graph样本root span全免周期最大.9972%，非请求收益。本轮“已确认可实现”新增0，无新NPU/请求/profile/远端写入/服务改动。

**下一处**：loop031有限PIVOTED，loop032 RUNNING；实际DFC GMM1接收端M/轮转core/两组C2V发布入口已读，10源hash匹配。继续GMM2→BlockMmad异步fixpipe/FP16→SwiGLU INT8+scale→combine真实消费者，先整段价。GMM1 logicalN与expertC strideK解释尚未核，不能直接认作可删padding；独立4/8原语不证明与20/40 DFC共驻。每替换原语逐位一致，全链路未扫全。

[介绍、到了哪里、结果](evidence/latent双缓存整段与保留GEMM净价判断.md) · [净价前沿](evidence/index-key-publication-frontier.json) · [下一处入口](evidence/dfc-gemm-tile-entry.json)

## 2026-11-10 QLI原生末读者、容量与有成本门槛；继续latent双交付

**介绍**：沿整段QLI继续host分核→Cube/Vec→token/candidate最终输出及PA物理读者，判断少算/少搬与资源再分配。每替换原语逐位一致。

**到了哪里**：18 native源核对，arch22主body、原Sort32/MrgSort接口、host parser/tiler主路径和非FD metadata分配已读列明ranges。16当前worker映射custom opapi，QLI JSON/.o f1dd0d93…相符、4key；磁盘/映射/initial env不冒充每调用device binary/tiling或loaded内存证明。

**结果**：已停止——910C所选metadata源不生成S2/LD归并payload，删除该数据信用0。待确认——末尾仍Reset/init/prefetch/SyncAll的空epilogue；声明workspace与显式region差额；保留原half链、candidate2048/pin/filler/tie次序的top512 accumulator及shared membership/大QLI。C24且host核数=kernel block数时，源声明99.0234375MiB、显式G32区域4.6875MiB；这不是实际resident可释放显存，不乘八，LibApi L/隐藏scratch/实际alias与高水位待核。

有成本筛查：若accumulator占原kernel时间25%、阶段时间删75%，e1/h0、5、20us每index source时条件P90为1.017481%/1.006756%/.974581%。h20us要过1%需f≥25.6246%(e1)或51.2491%(e.5)。阶段比例/曝光/新成本会改变去留；PMU比率不当阶段耗时，表非预计/因果/实测收益。旧byte-mask局部机制保留，旧batched-slot C8稳健门槛未过；不认领新服务收益。

下一条已经从实际caller读到：KV源2/8/14/20同latent→indexwk/RMS/RoPE/quant/key-scale两scatter，再原地longRoPE/第三scatter。跨流必须保护indexwk对preRoPE latent的末读；query的half-scale-first quant与key先quant后scalehalf也不能混用。loop031继续真实行量/C2完整位与整个多输出交付价格，再追native consumers。**Goal ACTIVE，全链路未扫全，无新primitive/性能基线准入。**

[QLI原生/资源与净价判断](evidence/QLI分核末读者与容量及净价判断.md) · [native覆盖/待确认](evidence/indexer-native-contract-frontier.json) · [下一条入口，未价全段](evidence/index-key-producer-entry.json) · [地图](evidence/source-frontier.json)。本轮只读，模型/NPU/请求/新profile/远端写入/服务变更0；未操作其它controller。


## 2026-11-10 启动交付已判，热Indexer整段继续native

**介绍**：从完整启动交付追四helper、workspace及MM容量，再按实际消费者定热Indexer融合范围。每个替换原语逐位一致。

**到了哪里**：四helper/early guard、workspace capture锁定及MM profile/KV预算已读；16rank旧启动wrapper .329–.378s（不相加）。纯文本启动仍profile8192 encoder tokens/8images。142直接safetensors头部匹配266视觉tensor、485268480 BF16元素，checkpoint925.576MiB；构造dtype约925.605MiB。张量payload读取0，非live allocator上界。热路径368组/50独立成本条件吻合，实际每卡rows≤1024，长度已融合。Python V3→native V2 adapter及arch22 half舍入、算后membership/filler、crosscore flags追至所列局部ranges；源pin不当加载库或全分支完成。

**结果**：已停止——冷开销删除和locked热增长对暖请求直接信用0；同1%独立预算下量化+chronological排序即使全免费，条件P90仅.594571%，额外大行sum在这些调用本就不存在。待确认、值得深入——QLI整段有利P9089.693940ms/6.012243%；删50%且每source付5us为44.686970ms/2.995397%，均非预计/实测收益。不能用这份小段价格剪掉大QLI，也不能相加其它候选。视觉参数单bank不足2240MiB/卡，联合容量仍需allocator/headroom/KV policy/逐位与请求价。

**逐位限制**：原FP16 query scale、weights×scale half、QK DEQF16+ReLU half、FP32 head sum/key scale次序保留；候选外真实位置可作不足top512填充，无条件压掉/改-1停止。chronological bank已在八index sources共享，每消费者重算不获免费信用。下一条继续host tiler/SplitCore、candidate pin/tie/merge、LD末输出、PA物理gather和当前priority/loaded库-key；随后核有成本净价。原ordinary/跨迭代cache与其它全链路缺口保留，**Goal ACTIVE、全链路未扫全**。

[启动/容量判断](evidence/启动预热与容量交付判断.md) · [热Indexer整段深入](evidence/热Indexer整段交付与逐位边界判断.md) · [覆盖地图](evidence/source-frontier.json)。本轮只读：模型/NPU/请求/新profile/远端写入/服务变更0，未操作其它controller。header-index失败及analyzer字段错误原样留证，无新原语/性能基线准入。


## 2026-11-10 实际mode更正与公共前缀原语：复用原Encoding，保留冷miss

**介绍**：沿输入renderer→native encode_batch→完整Encoding→每请求IDs的实际消费者查重复计算。每替换原语逐位一致，先价整段，再深入稳定前缀共享。

**到了哪里**：C8八任务在2.473533ms提交，单worker native466.209239ms/父方法487.110933ms，最后input492.603961ms。显式tokenizer-mode=deepseek_v41已沿两registry追完限定分派：tokenizer为V41 factory+cached HF，renderer却是DeepseekV4Renderer→BaseRenderer。**撤回“实际服务已有HfRenderer副本池、只调线程数即可安全并行”的解释**；HfRenderer不在这条路径，flag-only配方停止，显式独立backend/调度仍待确认。实际无pool CachedDSV41TokenizersBackend+原scalar方法重放8份IDs同hash；from_pretrained bootstrap/实际加载对象及wire尚未准入。Generic/pool回放只属反事实能力。初稿报告及旧JSON解释由actual-tokenizer-dispatch-frontier更正，原件保留。

**结果**：
- **已确认可实现（给定prefix的受限CPU原语）**：固定pipeline的117067字符/29385 IDs安全数字cut，不跨AddedToken；1256cases/269698 IDs含24fallback、Unicode/special/cap/error/list ownership，差异0。实际公共前缀117286字符；预选prefix是点证，不是在线发现/LRU实现。
- 进一步从首次原完整Encoding的字符token_to_chars和已有IDs提取prefix，8份完全相同，增量插入中位.182108ms/max.261111ms，避免另encode prefix。char117067≠UTF8byte117187，错误bytecut控制拒绝。局部9对old8 781.565971→保留首个完整miss+7hits166.072773ms；这是孤立CPU原语观测，不是实测请求收益。第一组另一cohort的759.538/79.447/额外build129.943ms原件单列，跨Run标不具可比性，不与旧profile466ms相加/替换。
- **待确认、值得开发：在线prefix cache**。缺一般prefix发现/插入、byte-LRU/epoch/实际backend指纹失效、完整bootstrap/wire、真实CPU关键暴露和六ordinary。每请求独立list，原text/arrival/cache_salt、validation与EngineCore/Engram/H2D消费者保留，无模型/H2D删除信用。
- **有成本粗价**：旧FIFO/其它父work/gap不变，冷miss保留第一job全部native，只给7hits删50%，每job付.5ms并首次插入.182108ms，最后input条件提前199.150528ms、firstinput延迟.682108ms。固定下游lag/未知request配对的P90情景−.682108～144.505989ms，非因果界/预计或实测收益；不能把input提前改名为batch收益，不叠加旧seed/metadata/parallel预算。
- **已停止**：依赖不存在pool的flag-only并行、裸共享可变backend、任意prefix拼接和错误bytecut。保留显式pool与在线cache的不同净成本/代次问题。

**下一段**：loop029已pin7源/5旧match，读到kernel_warmup wrapper，四个rejection_sampler/penalties/rms/indexer helper body仍未读；之后workspace/encoder_budget真实producer、消费者与cleanup。loop027只做mode前置守卫，不能记这些helper已扫。无新startup/model/profile/服务改动；全链路未扫全，不用函数/证据数量作完整性证明。

失败原件保留：AutoTokenizer尝试含模型/NPU包进口，未观测其设备初始化，不把“无显式计算请求”当进口无设备初始化证明；成功native点无Torch，成功scalar replay无torch_npu。未请求模型/device计算，owned cache路径核实不存在，无远端文件写入。配置/21进程initial-exec flags与native trace支持FASTOKENS关闭，但不当加载中Python对象证明。

[实际分派/原语/净价与缺口](evidence/实际tokenizer分派与prefix原语判断.md) · [受限边界证明](evidence/shared-prefix-tokenizer-proof.md) · [下一启动入口](evidence/startup-warmup-entry.json)

## 2026-11-10 文本/SSE整段与ByteLevel长prompt预热点证

**介绍**：沿真实FastIncrementalDetokenizer/stop/logprobs→RequestOutput/collector→completion SSE→Starlette UTF8/Uvicorn HTTP交付，先价整段，再深入首步长prompt重复decode。每替换原语逐位一致。

**到了哪里**：437834完整Python事件唯一ID及父边包含校验通过，父子区间并集去重；8个约10ms的native大首步合计84.266439ms。installed tokenizers0.22.2/ByteLevel JSON/binary已pin。候选保留原构造的完整ID转换，只在list prompt IDs最后有效token以ASCII结束且tokenizer不可变时选择空seed；其它边界回退。129280词表ID、两skip标志、多步UTF8/special/error等共1302780次逐步字节/None/异常比较无差异；576个原vLLM wrapper场景/2976次update也无差异。未完成UTF8盲删seed有反例，未放宽guard。None/prompt_embeds入口尚未准入，接入前必须绕过原型guard。

**结果**：
- **已确认可实现（限定CPU原语/原wrapper fixture）**：8份同输入CPU重建都命中，9对交错配对中位合计101.113973→4.124833ms，包含原构造和新增guard/空stream，非服务TTFT/NPU实测。原构造4.392ms不认领为已删。
- **待确认**：实际renderer IDs/kwargs/backend/不可变资格、完整SSE wire回归与真实关键暴露/新成本、六ordinary。旧profile native8全免是宽松固定schedule敏感性：最高84.266439ms/P90约5.86%/整批.482%；rho.5、每seed .1ms条件净41.33322ms，非预计收益，不能拿CPU点96.989ms替换旧cohort预算。
- **已停止**：JSON-only同C8独立1%预算，全免首tokenP90信用仅.255192ms，整批约.0851%；不剪whole output/seed/ordinary。整输出609.977ms并集有大量保留/重叠工作，不当实测可省。
- **下一段**：实际输入renderer→async tokenize/executor→HF native。已pin8源并读局部入口：真实调用tokenizer(prompt, kwargs)，有max_input_tokens+1截断、token IDs/embeds分支；8后台encode_batch共466.209ms尚不是event-loop阻塞或收益。之后四warmup helper内部/startup MM仍未扫完。

Goal ACTIVE。loop023/024 PIVOTED；第三Run原语对拍通过，前两host/container路径ERROR/INVALID保留，Task禁止含invalid Run的Loop标ACCEPTED。CPU词表加载3次，模型权重/NPU/服务请求/新profile/远端写入/服务改动0；未操作其它controller。源码范围/哈希/fixture计数均不当全链路穷尽证明。

[整段与价格](evidence/文本到SSE交付与长prompt预热判断.md) · [CPU定点/边界/净价缺口](evidence/ByteLevel长prompt预热定点判断.md) · [下一输入入口](evidence/input-renderer-entry.json)

## 2026-11-10 请求代次、stop与终止轮：保守host界与整段价格

介绍：从真实internal ID/归一化stop参数到abort双队列、旧future、worker tombstone与异常消费者，反推no-next-demand整段交付。到了哪里：18运行树源14旧hash匹配/4新取；原方法控制fixture验证external abort可作用多internal代、EOS/显式stop/min-token集合不同、GPU valid数不等于可见cap裁剪数，async空CPU valid列表也不等于无输出。默认随机后缀/opt-out是source条件，不当live值或唯一性证明。

结果：scheduler已有保守终止界，会抑制即使全拒绝也必然达到max_tokens的额外target；不给重复40层信用。同一post-schedule数学界可随immutable frame提供终止保守子集，不需新accepted-count D2H；实际资格频率还缺。原C8七终止波把所有GPU-facing尾部全免也仅32.152328ms/整批0.172948%、TTFT0，rho.5/每波.25ms只14.326164ms，停止同独立1%小预算。该边界是target采样/verification最后native→draft最后score，不能作为更大host/queued/EOF父区间上界。

待确认：no-next-demand DSpark/provider/output/queued-input联合，保留context/prefix cache活写者、真实peer专家服务、全DP/TP决定、zero placeholders/valid counts、原算术几何与前轮lease。调用_skip_drafting本身仍跑dummy body，不是整段删除。普通六负载用exact9144 client SHA eba6确认caps126/13/300/20/126/13、ignoreEOS True；18端点各arm保留，四名称跨arm输出hash不同，不作位一致/性能前后对。短code-single末交付松弛约60ms不等于本地native时长或净省，缺实际资格、native/CPU parent和新增fence/协议成本；不套C8低预算剪掉普通。

已确认可实现：本轮无新原语准入；控制fixture/review PASS不是NPU逐位或模型通过。Goal ACTIVE、每替换原语逐位一致、全链路未扫全；下一条loop-023追hot frontend detokenizer/stop-string、logprobs、RequestOutput到SSE字节与buffer/error消费者，之后startup/warmup/MM。NPU0、模型0、服务请求0、新profile0、远端写入0，未改服务/环境。

[本轮介绍、到了哪里、结果和证据](evidence/请求代次与终止轮交付判断.md)。

## 2026-11-10 Prefix-cache/退役：实际guard排除空GPU配方

介绍：沿request hash→多组prefix命中/touch→分配/发布→worker cache消费者→finished/abort/reset/复用，先核整段已有共享与真实工作。到了哪里：27运行树源19旧hash匹配/8新取；不变serve.sh/default和完成profile启动记录确认prefix caching开启。普通KV/EC transfer、generic权重预取与sparse KV offload未启用，Engram CPU表仍活跃。全16既有trace SHA/size不变；该capture没有专属cache/prepare-input标签，不能据此定CPU父区间为0。

结果：实际V1＋DSpark不初始化通用KV block zeroer；无Mamba、128对齐cached specs、circular32私有/noncacheable，使当前partial-hit CoW copy也不触发。删这两类现有GPU工作的local/request信用0（固定host），停止同配方；CPU ID收集/drain仍待大host范围定价。仅翻cascade/trailing-drop或通用changed-M hidden复用不能继承逐位一致；不剪新的layout/owner/shared-read机制。

待确认：typed physical-cache＋immutable BatchFrame退役，以及保留原row算术的跨请求selected-KV读取共享，分别与既有metadata-host、SMLA/QLI/物理cache范围重叠，不相加。缺整个engine CPU父区间、实际选中物理行交集/现有L2共享、owner代次/最后读者及净成本。原free_blocks fixture证明refcnt0仍可prefix命中且不可写；当前无KV consumer，defer_block_free=False，新早交付不能继承不存在的通用fence。

已确认可实现：本轮无新增原语准入。review/fixture PASS不是NPU或模型通过。NPU0、模型0、请求0、新profile0、远端写入0；Goal ACTIVE、每替换原语逐位一致、全链路未扫全。下一条loop-022沿normalized internal request ID/stop参数→abort旧future、同ID新代及异常消费者；4个warmup helper内部/startup MM缺口仍保留。

[本轮介绍、到了哪里、结果与证据](evidence/Prefix-cache与退役交付判断.md)。

## 2026-11-10 整inactive-DP与实际就绪：限定判断，继续全链路

介绍：沿engine本地无工作→target40＋draft3→peer专家与最后输出，先评估整轮、合成sender和host联合交付。到了哪里：171 hostname核对恢复，20运行树文件16旧hash匹配/4新取＋5依赖；原C8的43个idle F21入口全部早于active。独立新profile全16 CPU/native每轮40＋3关联完成，26个共同Gloo ordinal形成208对；实际Ascend target/draft共用INT32[2,DP]数据包，8个CPU pair并非单个16rank reduce。源范围/身份计数不证明全链路完成。

结果：原C8固定active到达/EP服务/资源的独立idle prefix/tail停止；252ms整轮的32ms非DFC补集不能直接算收益。新录制C8 Gloo backend-only全免P90仅5.586240ms/0.388793%，rho=.5、每阶段新增.1ms为1.393120ms/.096959%，停止同独立TTFT1%预算。DP1独有末尾约715ms等待在所有首token后，且跨同步profiler stop/export；不得将Free/Gloo总量归为生产收益。

待确认：全局无真实工作＋BatchFrame整轮快路、单侧真实EP service-only/资源重写。前者仅两DP代表marker与frontend-ready夹住的有利启动上界21.561858ms（P90 1.500668%、整批.123345%；half/h.1为10.580929ms），还免费抹掉下一真实准备，不是预计净收益。缺实际全16工作标志/TP一致、inbox与在途future/lease、协议代次/成本和六普通负载；后者缺eager M416接收计数与改sender行序/M的逐原语逐位证明。与已有metadata-host范围不相加。Engram全false mask直接returnhidden不具通用IEEE位等价，非新NPU失败。

已确认可实现：本轮无新增原语准入。NPU0、模型0、请求0、新profile0、远端写入0，未操作另一个已完成profile控制器。Goal ACTIVE、每替换原语逐位一致、全链路未扫全；下一条loop-021沿prefix/cache/provider/finished/error退役，先核冻结guard再追最后读者；冷启动warmup/启动MM缺口仍保留。

[完整介绍、到了哪里、结果及证据](evidence/整inactive-DP与Gloo就绪交付判断.md)。

## 2026-11-10 实际worker家族与有效配置更正

介绍：沿真实V1 profile/capture/dummy与消费者，见[本轮判断](evidence/worker-capture-review.md)。到了哪里：22份局部源范围、15旧identity匹配；3632参与窗分为3136真实Q>0、488有metadata但Q0、8个None-metadata dummy，SP/head/三层sender全部核对。结果：仅翻DSpark flag、按M3共图、仅翻shared flag均已停止；真实metadata图模板/dispatcher/代次与显式资源重写仍待确认，无新native准入。

**更正此前“shared已重叠”**：固定enable_fused_mc2=1会强制multistream_overlap_shared_expert=False，源码SHA与旧启动排除回执一致；可选helper不是已生效重叠，旧candidate样本0、无speedup。整段价格与已有原语资格保留，不能因此宣称共驻已支持。

下一条loop-020：整inactive-DP target40+draft，追真实跨卡expert服务、原mask和死输出再定价。原C8只一波idle，连真实peer整三body parent免费也仅P90 .647747%/整批.051980%，停止同小预算，不剪六普通/target40。新授权framework profile已独立完成，先绑定Gloo与target/draft就绪；不混入旧227波价格。Goal ACTIVE，每替换原语逐位一致。



## 2026-11-10 三层body限定进展；全链路仍进行中

介绍：沿真实三层DSpark HC→CP/SWA→TP恢复→EP16/top3/shared→下层与head输入，见[本轮完整判断](evidence/dspark-three-body-review.md)。到了哪里：原C8全部3696窗与11088 F21计数；227活跃波保留3632参与窗（3624自身active），原observer epoch为DP各自历史标签，错误断言与更正保留。结果：全parent免费有利P90 1.424587%/整批3.569201%；rho.5、h.1ms每波条件净.678778%/1.662497%，不是预计/实测，不与host/context/score相加。每替换原语逐位一致；本轮无新native准入。

待确认：同原语body graph/提交+shared/routed资源安排；具体切口、persistent metadata、wrapper同步与DFC末读已下钻。已停止：零sender跳EP、空expert新增GMM收益、三层同值合并及同小TTFT预算配方。已确认可实现：本轮没有新增候选准入。

下一条：loop-019真实V1 worker _dummy_run/profile_run/capture_model及继承caller，派生target/draft warmup、dummy、metadata/cache代次和collective顺序，之后继续prefix/provider/error剩余路径。当前171认证失败，本轮只用固定源码/旧profile，无当前运行身份核验、NPU或远端写入。


## 2026-11-10 当前：输入/context整段价格，继续三层draft body

**介绍**：从实际loader/runner/selector走到三cache与query embedding。**到了哪里**：15运行源；原C8 SQL3696窗（3624 active、453 TP8组），不是地址或覆盖率证明。draft MM预embedding被loader关闭，own embedding保留；padded aux全QSL arange与accepted/rejected/sample selector分开。**结果**：完整aux选行→main→metadata/context→embedding→第一draft HcPre全免，有利P90 1.549643%/整批1.460072%；rho.5/h.1ms每波条件净.741306%/.607932%，非预计/实测/准入。大host/context待确认，与metadata-host不叠加；main单独全免.304272%/.177883%停止同独立C8小预算。不存在的重复、跨模型共值、通用changed-M原配方均停止；旧反例native回执和7当前源pin核对，旧派生flow-index失配保留。

loop-017 PIVOTED仅限源码/价格判断，loop-018继续真正三层draft的noncausal SWA/CP、HC和128/top3/EP16、提交/graph合法性与megakernel范围。物理三cache末读、原M/tile逐位、联合新成本及完整host-parent六普通时间仍是具体缺口。Goal ACTIVE，全链路未扫全，每替换原语逐位一致；本轮NPU0、推理0、新profile0、远端写入0。

[介绍、到了哪里、结果与下一条交付](evidence/embedding-context-review.md)。


## 2026-11-10 当前：末层物理cache与原行映射，继续输入/embedding

**介绍**：保留body38完整aux39，body39沿writer/page/未来读者到topK/TP-SP返回。**到了哪里**：16运行源pin；slot3/group11、padded stride、非模128环；页按已处理提交prefix退役。3328单请求末行=CP7/旧local415，不能压成一行后取topK前缀。**结果**：尾128 writer且原prefix发布不变、新topK前缀且原restore不变两配方已停止；全原cache+显式旧行/原数值几何联合query/O/FFN仍待确认。56原C8窗口保留六cache原语旧时长的有利组件P90包络2.163059%；rho.5/h.5ms每wave条件净.947468%，非预计/实测/逐位准入，也不剪改变cache producer的大范围。混合波的全部验证行仍有读者。

loop-016 **PIVOTED仅限源码/价格判断**，loop-017继续文本输入/embedding与DSpark context producer；实际多请求QSL/device长度、SMLA/HC/GEMM/EP原几何、逐位实现/净成本/六普通缺口保留。Goal ACTIVE，全链路未扫全，每替换原语逐位一致。本轮NPU0、推理0、新profile0、远端写入0，服务不变。

[介绍、到了哪里、结果及下一条交付](evidence/final-cache-review.md)。


<!-- FINAL-PRODUCER-20261110-START -->
## 当前前沿：aux完整接线更正，继续body39（loop-016）

**Goal ACTIVE；全链路未扫全；每个替换原语逐位一致。更正并撤回：此前aux36/37/38与两个body可删的判断漏了NPU runner先加1，两层3.8115%/1.77169%情景不得再用于候选收益。**

- **介绍**：重新沿GPU setup virtual selector、NPU override、setter到model采集点追完整接线，保留原错误记录和失效说明。
- **到了哪里**：冻结ID37/38/39，上游无eagle/dflash覆盖，NPU选择38/39/40，setter排序去重不换ID，model再判断layer_idx+1；真实aux入口为37/38/39。body38全行输出mean供aux39，只有body39在最后采集之后。这是源码/冻结配置推导，不冒充live对象dump；缺Docker INFO日志未被当成分支证据。
- **结果**：原两body裁剪已停止/撤回，原Run不可变记录由新Run及superseding knowledge明确声明旧解释失效。当前body39候选仍待确认；56个real eager窗口下inclusive最有利DP全免包络P90 2.32864%，h=.5ms/wave、rho.5情景净1.03026%，rho.25仅.44810%，负收益保留。非实测或原语准入。metadata72窗末读结论不变，producer_done/alias/native/graph/error仍待确认。
- **下一处**：body39 SWA128物理writer/page/slot/未来读者、selected-query到topK36原CP-local映射；保留KV20/topK36与body38完整aux39。38 HCPost→完整aux-mean物化只能另研、须算全aux行，不能认领38整层计算删除。新NPU0、推理0、新profile0、远端写入0。

[更正与当前判断](evidence/metadata-tail-and-final-producer.md) · [完整接线](evidence/aux-config-call-chain.json) · [当前body39价格](evidence/final-producer-entry.json) · [撤回档案](evidence/withdrawn-two-layer-price.json)

<!-- FINAL-PRODUCER-20261110-END -->

<!-- METADATA-HOST-20261110-START -->
## 上一段：metadata与host联合交付（loop-015已PIVOTED）

**Goal ACTIVE；loop-014 PIVOTED（限定结论），loop-015 PIVOTED（限定结论）；全链路未扫全。每个替换原语逐位一致。**

- **介绍**：从target DP协调、global/local metadata，经实际DSpark query到producer/consumer，先核早知字段与整段净值。
- **到了哪里**：13源SHA；DP为8组[r,r+8]CPU Gloo pair。实际DSpark override返回B×Q（冻结Q5），不能复用target24桶；GPU reject后的length/slot仍晚知。CP同代batch可共享，但global/local域及每资源physical state分开。V1每次host submit、FULL用external event、finally release，不能借V2全task finish证明V1末读。
- **结果**：旧CP/CPU-plan renderer精确重放，21项profile输入hash核对；已有方案不重复认领，旧文本验收不满足逐位门槛。同代BatchFrame+group组装/发布**待确认，值得深入**。原样shape-only返回缓存/mini graph已停止；device-only小预算不剪较大host方案。
- **净模型**：保留两个parent builder、每组return、first producer。旧B4诊断组件毛预算1.689–2.490ms；f=.5/rho=.5/h=.25ms时条件净.79357ms/迭代，C8整批.969%、P90.266%；rho1时1.83714ms、2.243%/.616%。这是有利情景而非matched实测。无暴露的负收益保留。原C8串行敏感性到整批1%需净.81898ms/wave，P90需2.98371ms；首输出一次另列，不多认异步下一波。
- **下一处**：BatchFrame observer/alias/closure和V1 release到实际末读，再决定具体grouped实现；DP packet保留empty/dummy/K0/input-fits/error序列及Gloo净成本缺口。其后回末层attention/FFN row-demand和剩余输入/embedding/Engram。新NPU0、推理0、新profile0、远端写入0，无部署变化。

[完整判断](evidence/metadata-host-review.md) · [具体联合协议](evidence/metadata-batch-frame-protocol.md) · [有成本净模型与请求门槛](evidence/metadata-host-net-price.json) · [源码与缺口前沿](evidence/metadata-host-frontier.json)

<!-- METADATA-HOST-20261110-END -->

<!-- INPUT-STAGING-20261110-START -->
## 上一段：输入原保护与条件 dead-backup（loop-014已PIVOTED）

**Goal ACTIVE；loop-014 PIVOTED（限定结论）；每个替换原语逐位一致。**

- **介绍**：继续沿物理 CPU 源追 H2D 末读，判断提前输出真正需要的代次边界。
- **到了哪里**：21份源码身份核对。继承的 synchronize_input_prep 在 target 准备前等待、结束后记录，覆盖 block table、zeroer 和 target CPU 源；已知晚到持久源为 proposer backup INT32[4]（每卡16B），CP qsl/suffix 为本次新结果。device metadata 另有 stage/join/reuse。
- **结果**：无 discard greedy 下每行至少一个有效ID，backup臂条件无活读；可仅消去 fallback fill/H2D，作为提前输出协议深入，仍属**待确认**。原 C8 回复 rank0/8 的2641/2677条日志不含 discard 数/all_greedy 标志，受限分支的实际资格未证。原7–15ms仅重叠组件预算，未扣新协议成本，不能全额授予此分支。原样 early-copy 补丁继续**已停止**；扩展复用栅栏或 live-source snapshot/ring 亦待确认，需支付 lost host overlap / event / backpressure；不能预设全部输入双份。
- **下一段**：共享 metadata / host submission 的较大交付联合定价，保留本段全caller/委托末读、实际资格和native逐位缺口即可继续。随后回末层attention/FFN producer。不重复自动快照实验。
- **环境**：新NPU0、推理请求0、新profile0、远端写入0，无部署变更。AST遍历顺序reader错误保留，修正后仅review通过，未认领实现或性能准入。

[完整判断](evidence/input-staging-review.md) · [行段与所有权前沿](evidence/staging-execution-frontier.json) · [协议修订](evidence/early-output-generation-protocol.md)

<!-- INPUT-STAGING-20261110-END -->

## 上一轮：提前输出扩大到staging所有权（未完成全图）

**Goal ACTIVE；loop-014 EVALUATING；全链路未扫完。每个替换原语逐位一致。**

- **介绍**：沿实际sampler/processor/RNG/verify→输出线程→下一输入staging，判断等待能否与DSpark重叠。
- **到了哪里**：28源SHA，13类条件族；原C8只读SQL72窗口。真正回复来自rank0/8的9个输出等待，63非回复窗口不当客户端收益。spec只实例化MinTokens；custom拒绝，block与普通首拒规则不同，RNG形状/offset/代次不能合并改写。
- **结果**：保留sampling ready、原CPU返回、copy/event代价，short TTFT条件3.395%、P90约0.5373%，没有吞吐/response-end实测结论。更大提前输出+输入staging所有权路线**待确认**。native证明pending pinned H2D不会快照：shared INT32[24]/INT64[16]全部24/16元素变新值，独立代次control字节0差；这仅是底层所有权参数补证。缺代次协议的原样early-copy补丁**已停止，不能应用**。
- **下一处**：枚举CpuGpuBuffer、block-table及metadata可写pinned源和最后H2D读者，设计lease/不可变frame并扣ring/event/回压成本；然后继续metadata/host及末层attention/FFN大producer。不重测自动快照，不重开低值greedy或当前inactive random小方案。

新NPU隔离进程1、服务推理请求0、新profile采集0，health前后200。a1在NPU启动前setup筛选错误保留；修正后a2匹配当前16 worker环境。服务/全局环境未改。

[介绍/到了哪里/结果](evidence/sampler-output-review.md) · [代次协议](evidence/early-output-generation-protocol.md) · [前沿与价格](evidence/sampler-output-frontier.json) · [native原件](evidence/native-pinned-h2d-a2/evidence.json)


## 本轮终端与下轮交付判断（未完成全图）

**Goal ACTIVE；loop-014 EVALUATING；每个替换原语逐位一致。全链路未扫完。**

- **介绍**：最终HC→全部真实读者→target head/接受拒绝→异步输出与下一输入，按完整交付定价。
- **到了哪里**：18源SHA匹配；13,520文件仅是三个bank/getter标识的引用扫描。DSpark不消费MTP bank getter值；prompt完整行、req-ID快照/bonus-draft scatter、Worker输出线程、CPU stop和common退役接口已读。
- **结果**：bank源码容量320MiB/卡（16卡共5GiB），保留full AG/full-M数学形状的具体补丁已形成，未部署、native/graph待确认。Post→head输入全免C8 P90最高0.96564%、六普通情景0.64242%；target head+验证连GEMM全免最高个体响应0.58204%、六普通0.52940%，停止相同独立1%延迟小预算。原compact FFN原语位差已停止，即使normalized相同也不放行。更大末层producer与reject-count/下轮host plan保留。
- **下一处**：sampler processors/RNG、random block verifier/fallback；再联合定价下轮host-device大交付及末层attention/FFN producer。connector provider内部仍条件待扫。

[介绍/到了哪里/结果](evidence/terminal-execution-review.md) · [结构化前沿](evidence/terminal-execution-frontier.json) · [价格](evidence/terminal-cost-replay.json) · [未部署补丁](evidence/terminal-dspark-dead-bank.patch) · [原native引用资格](evidence/terminal-native-archive-qualification.json)

容量项与设备运行资格分开：没有新native已确认替换。具体失败配方和限定1%小预算归已停止；更大末层及跨迭代producer归待确认。

## DSpark完整score预算与terminal读者前沿（未完成全图）

- **介绍**　沿aux/main/context、三decoder、完整score→五ID及接受拒绝边界建立范围。
- **到了哪里**　21源/实际128-top3-3层patch，8组typed成本与3632个C8 rank-score绑定。W2是复制的，五轮没有AG；BF16 W2及加法边界不能跨过。
- **结果**　整score含六GEMM全免费，C8个体最高0.9602%、六普通B1情景最高0.8027%，停止同一独立1%预算小方案；aux/main/decoder/大提交仍待确认。terminal入口发现DSpark仍分配MTP HC bank，容量320MiB/rank只是待核读者切口；并未确认可删。继续验证/async/下一输入。

[完整判断与停止范围](evidence/dspark-execution-review.md) · [terminal缺口](evidence/terminal-entry-frontier.json)


## Engram独立源码与CPU表示门槛，DSpark入口推进

- **介绍**　沿实际Wkv/gate残差消费者反推hash、INT8/FP32 UVA、TP-SP和迭代退役。
- **到了哪里**　19源身份、128边界/4736 gate任务和10组Wkv成本；body全免最高条件TTFT1.1154%，gate→HC仍保存BF16残差的后续HcPost读者。DSpark进入base logits一次/五次Markov反馈，主体和验证末读未闭合。
- **结果**　独立小预算/原样失败变体停止；E8M0容量候选净少18.43GB共享host，编码255不能直接用现成decoder，修正CPU穷举0差、NPU待验。owner交付/提前投影/内容cache/HC联合保留明确缺口，全图继续。

[三类处置及证据](evidence/engram-execution-review.md) · [DSpark下一处](evidence/dspark-entry-frontier.json)


每项按介绍、进度、结果阅读。展开可查代码、适用条件和证据。[更新方式](README.md#阶段说明与上传流程) · [早期 42 族](evidence/priority-42.md)

## 前次MoE深入增量（全链路未完成）

- **介绍**　完整shared4/8与DFC共驻、DFC返回到残差大consumer，均由真实交付形成范围后再匹配历史。
- **到了哪里**　M416真实共享权重八项字节比较0、actual BlockNum8/4/8/4（DownMix4），仅确认本case原语准入；返回段明确FP16/BF16舍入、原TopK序和屏障后peerD末读。
- **结果**　完整共驻及前缀/deferred返回融合待确认具体净价/生命周期/剩余逐位覆盖。原样空credit变体复用EP16 native价值失败，已停止；DQ-only条件0.2864%停止独立1%路线，merge+Post全免1.3537%不是预计净省。继续SMLA内部和其余未扫路径。

[原语事实/缺口](evidence/shared-fourcube-primitive-gate.md) · [返回判断/停止依据](evidence/moe-return-consumers.md)

## SMLA与服务激活实现的独立增量

- **介绍**　从真实自带包和QK/softmax/PV读者形成KV交付+首partial联合parent，补齐shared实际激活实现门槛。
- **到了哪里**　C1/C2快pair/大步长回退的不同顺序有8个原helper CPU case；30组成本已匹配。a3服务custom实际对象打开，M416八输出字节0且发核4/8。
- **结果**　SMLA联合K256/512情景有1.219%/2.031%条件信用，保留K/速率/逐位与生命周期，非预计收益。shared仍只准入单M416原语，DFC共驻待确认；a2实现资格已补充。下一段Engram继续。

[完整判断](evidence/smla-execution-review.md) · [服务门槛](evidence/shared-4cube-service-gate-summary.json)

## C01

**M00、M11 · 未变block表跳过/增量发布**

- **介绍**　标记真实变更的 block 表行，只上传脏区，保持各组物理表地址和图回放后缀。把减少的 H2D 时间与少提交的关键路径时间分别计价。
- **到了哪里**　待确认。初步判断。待定点复核
- **结果**　部件预算：C8 旧整批 23.66 ms（0.127%），仅计上传数据。

<details><summary>代码与依据</summary>

- **代码定位**　`block_table.py:298` `commit_block_table`、`block_table.py:519` `commit_block_table`、`model_runner_v1.py:1281` `_prepare_inputs`
- **适用负载**　13组、B≤4；dirty与graph后缀状态保留
- **成立条件**　全部13×453可删仍需额外净提交162.24ms，≥27.55µs/copy
- **证据**　[S02 代码与定价](evidence/S02.md)
- **接续**　按实际调用点核对剩余可删的提交时间。关联 M11

</details>

## C02

**M00、M11 · 请求/接受数/ID/位置/长度/spec多输出publisher**

- **介绍**　遍历一次请求映射与接受数，同时写出新 ID、位置、长度和 spec 元数据。扣除仍有读者的 CPU 镜像及各目的地必需的写入。
- **到了哪里**　待确认。初步判断。待定点复核
- **结果**　设备端上界：已定价部件全免为 61.73 ms（旧整批 0.332%）。

<details><summary>代码与依据</summary>

- **代码定位**　`gpu_model_runner.py:1790` `_prepare_input_ids`、`model_runner_v1.py:1771` `_calc_spec_decode_metadata`、`model_runner_v1.py:1876` `_copy_valid_sampled_token_count`、`llm_base_proposer.py:2192` `prepare_next_token_ids_padded`
- **适用负载**　target输入到seed/count及下一轮；前缀/验证各守合同
- **成立条件**　其余净省≥124.17ms/旧整批；raw ID/CPU镜像/分目的count不能丢
- **证据**　[S01 代码与定价](evidence/S01.md) · [S02 代码与定价](evidence/S02.md)
- **接续**　找出必须保留的 CPU 镜像和索引读者。关联 M00

</details>

## C03

**M00、M11 · 共同CP计划与按组slot多输出批写**

- **介绍**　让同轮 slot 写入共用计划和输入，将 26 项工作批量提交，保持每个 cache bank 的地址、无效行和代次。复用历史46/F02 的实验。
- **到了哪里**　已确认可实现。服务实测·限定基线。复用；小批量另查
- **结果**　历史服务实测：B4 两组整批耗时分别下降 3.39% 和 2.22%。

<details><summary>代码与依据</summary>

- **代码定位**　`block_table.py:469` `compute_slot_mapping`、`block_table.py:504` `compute_slot_mapping_draft`、`model_runner_v1.py:3561` `_get_block_table_and_slot_mapping`
- **适用负载**　C8 B4、真实26-job caller；其它job集合另判
- **成立条件**　不是与CP+slot、CPUplan其它基线相加；B1/empty需caller净省
- **证据**　[S03 代码与定价](evidence/S03.md)
- **接续**　复用；小批量见历史46/F03。关联 M00

</details>

## C04

**M00、M10、M11 · target/draft共用CPU DP计数协调packet**

- **介绍**　在目标模型与草稿的 DP 计数可预测时共用一次协调结果。统计真正位于关键路径且可取消的第二次协调，各 DP 保留一致回退。
- **到了哪里**　待确认。初步判断。待定点复核
- **结果**　门槛：旧整批若命中 200 次，每次净省至少 0.930 ms 才达 1%。

<details><summary>代码与依据</summary>

- **代码定位**　`model_runner_v1.py:825` `_sync_metadata_across_dp`、`model_runner_v1.py:3325` `_determine_batch_execution_and_padding`、`llm_base_proposer.py:1013` `_propose`
- **适用负载**　固定K5、padded eager draft、角色/计数可预测
- **成立条件**　所有DP一致fallback，保留EP顺序；CPU barrier等待未必可消失
- **证据**　[S03 代码与定价](evidence/S03.md) · [S10 代码与定价](evidence/S10.md) · [S11 代码与定价](evidence/S11.md)
- **接续**　核对各 DP 是否能一致决策及回退。关联 M10/M11

</details>

<!-- WORKBENCH-CONTINUATION-20261109-START -->
### 2026-11-09 交付合同续扫

- **介绍**　从目标DP协调追到DSpark query、dummy/skip和EP集合调用，判定可共用计划的完整字段。
- **到了哪里**　待确认。静态消费者核对/离线模型；无新设备实验。
- **结果**　目标24行桶不能决定草稿5/20行；直接复用子项C04a停止。扩展packet须含B/K/角色/epoch及一致fallback；协调净关键省仍未知。

[本次源码与完整判断](evidence/workbench-continuation.md) · [复算结果](evidence/workbench-continuation-results.json) · [源码摘录与hash](evidence/workbench-continuation-source-review.json)

### C04a

- **介绍**　直接用目标padded计数替代草稿计数协调。
- **到了哪里**　已停止。B/K/角色信息不足。
- **结果**　目标桶24对应草稿5或20行；不否定含完整字段的一次共同协调。
<!-- WORKBENCH-CONTINUATION-20261109-END -->

## C05

**M06、M08、M10 · 取消没有最终贡献的行：idle/小桶/末层/context分角色**

- **介绍**　按输出读者划分本地假行、末层有效行、已退休 context 行。只跳过无人消费的部分，同时保留远端专家服务、raw 快照和最后一次 cache 写入。
- **到了哪里**　待确认。初步判断。待定点复核
- **结果**　上界：末层 FFN 与读出合计 22.992 ms（长 prefill）；idle 波若有 100 次，旧整批达 1% 需净省 1.859 ms/次。

<details><summary>代码与依据</summary>

- **代码定位**　`prepare_finalize.py:252` `prepare`、`fused_moe.py:485` `_forward_impl`、`model.py:1505` `forward`、`dspark.py:295` `precompute_and_store_context_kv`
- **适用负载**　idle专家服务；B1/2；末层head行；retired context
- **成立条件**　末层先退役raw快照；≥64.89%上界净兑现；context须物理final writer/窗口/代际；不删远端专家
- **证据**　[S03 代码与定价](evidence/S03.md) · [S08 代码与定价](evidence/S08.md) · [S09 代码与定价](evidence/S09.md) · [S10 代码与定价](evidence/S10.md)
- **接续**　数真实 idle 波次；保留远端专家和 raw 读者。关联 M08/M10

</details>

## C06

**2026-11-10 资格更新**：介绍：完整结果复用与lookup表示是不同交付范围。到了哪里：当前成对加载/共享注册/指针bank和7248旧lookup已追。结果：lookup-only全免条件TTFT最多0.1180%，同1%独立延迟小开发停止；不据此剪去完整WKV/owner/input/iteration联合。[当前前沿](evidence/engram-storage-frontier.json)。

**M01 · Engram hidden-independent lookup/WKV/统计结果缓存与发布**

- **介绍**　按完整内容键和权重版本缓存与 hidden 无关的 lookup、WKV 和统计结果。按层回放实际命中，再扣失效、未命中协调和容量成本。
- **到了哪里**　待确认。初步判断。待定点复核
- **结果**　门槛：本地周期降 1% 需消除原可缓存耗时的 71.1%–96.3%（中位 75.4%），另扣缓存管理。

<details><summary>代码与依据</summary>

- **代码定位**　`hash_state.py:79` `forward`、`embedding.py:333` `embed_gathered`、`embedding.py:378` `lookup`、`layer.py:34` `forward`
- **适用负载**　层1/14、TP8共享CPU-UVA表；完整tuple/权重代际key
- **成立条件**　还要扣cache/miss协调/容量；position重叠不是命中率
- **证据**　[S04 代码与定价](evidence/S04.md)
- **接续**　按层、权重版本及完整键回放

</details>

<!-- WORKBENCH-CONTINUATION-20261109-START -->
### 2026-11-09 交付合同续扫

- **介绍**　区分完整内容结果命中与保留dense WKV的局部miss，修正缓存性能单位。
- **到了哪里**　待确认。静态消费者核对/离线模型；无新设备实验。
- **结果**　71.1%–96.3%是需消除的可缓存耗时比例，不是row-hit率。部分miss可保留整次WKV；缺完整键命中和残余调用成本。

[本次源码与完整判断](evidence/workbench-continuation.md) · [复算结果](evidence/workbench-continuation-results.json) · [源码摘录与hash](evidence/workbench-continuation-source-review.json)
<!-- WORKBENCH-CONTINUATION-20261109-END -->

## C07

**M02、M05、M07 · mHC norm/coeff主计算、rounded producer→consumer常驻**

- **介绍**　核查 Gram、系数计算及工作区，寻找可共用的归约或可直接交付下一个 Pre 的舍入残差。保留独立的 Post、aux 与 Engram 读者。
- **到了哪里**　待确认。初步判断。待定点复核
- **结果**　预算：HcPre 占同周期约 5.10%；周期降 1% 需净删 HcPre 耗时的 19.6%。

<details><summary>代码与依据</summary>

- **代码定位**　`model.py:858` `hc_pre`、`model.py:871` `hc_post`、`model.py:879` `forward`、`hc_pre_cube_compute.h:398-477`
- **适用负载**　40层80次Pre；初始/Engram/aux/终端例外；draft6次扣重
- **成立条件**　保存FP32 post/comb、HF32与BF16边界；两版row-norm和QA-only原实现已关闭
- **证据**　[S06 代码与定价](evidence/S06.md)
- **接续**　定位可删主体并复核舍入；具体反例见独立 HC 记录。关联 M05/M07

</details>

## C08

**M02、M03 · Attention producer与通信重排（具体数值路线分别判定）**

- **介绍**　在 SP 行所属卡先算 Wkv，将 H5120 的宽 AllGather 换成窄 KV。36 层传 512 维，层 20 传两份 512 维，三处 C2 源保留原路径。
- **到了哪里**　待确认的大范围；已测Wkv切M及stock AIV AGMM直接替换因位差停止，不能照旧包络领取这些原型。
- **结果**　原60.54ms/30.27ms只保留为旧local-M条件模型；其数值未准入，不是当前可开发收益。同M重排须重价，不能继承计算删除信用。

<details><summary>代码与依据</summary>

- **代码定位**　`model.py:879` `forward`、`dsa_v41.py:327` `_project_kv`、`dsa_v41.py:497` `_write_compressed_source`、`dsa_v41_cp.py:189` `_prepare_inputs_and_caches`
- **适用负载**　长提示prefill；小decode保留旧图
- **成立条件**　净兑现≥24.64%；换M数值、raw-reader、packing/争用；非P90重算。用四个 prefill 执行批次原宽通信与重复投影的节省作毛预算，扣除新 pack、同步与资源争用。
- **证据**　[S07 代码与定价](evidence/S07.md)
- **接续**　验证单层 BF16、行序、cache 和净通信时间。关联 M02

</details>

### 2026-11-09 独立搜索与C2扩展深入

- **介绍**　从真实Decoder/DSACP读者独立反推行owner producer；C2考虑SWA BF16与两份FP32投影共同发布，保留原全局pool/ring。
- **到了哪里**　待确认，正在深入。旧保守价排除三个C2源；不能用低值pool/ring小helper剪掉FP32主计算，不能据当前缺口关闭总搜索。
- **结果**　同C8 eager的两个FP32投影，四carrier/三源的原耗时中位串行参考9.113ms；非请求可省量。本地M数值/耗时和原重叠仍须核对，不与旧30.27ms包络相加。[实际消费者、搜索前沿与下一步](evidence/execution-derived-search.md)。

### 2026-11-09 C2逐调用与native数值深入

- **介绍**　按实际三源读者绑定全部336次FP32调用和192个分支窗口，对大producer的Wkv换M做最小native数值诊断。
- **到了哪里**　待确认；C2单独两FP32的C8首token1%独立路线停止。ND primitive换M位等价有反例，大producer需同M重排、原归约实现或既定容差验收。
- **结果**　两FP32全部免费、原peer/其它工作固定时，八请求TTFT信用最高0.636%，cache-only该项0。真实层0 Wkv/finite synthetic BF16的3328→8×416有302/1703936个位差，maxabs2.44e-4；后续NZ编译缓存报ERROR，6540/NZ未验。1组ND比较完成，无服务请求/性能benchmark。原30.27ms仅条件模型，不升格可开发收益。[全文、全部ID、原件及复算](evidence/C2-deep-screen-and-Wkv-numerics.md)。


### 最新逐位合同与通信深入

- **介绍**　默认gather可能模拟AllGather；另对原TP8使用现成AIV AGMM做固定两shape原语对照。
- **到了哪里**　用户要求每个替换原语逐位一致。Wkv切M的302个位差、stock AGMM的526/1005个位差已停止对应直接替换；历史段的“容差待定”已被新合同取代。
- **结果**　AGMM接口本身可执行、gather及finite检查完成，但不满足逐位门槛；不再补其Q争用或服务收益。保持full-M的真实root/P2P及保持原算术的其它流水仍待协议和净价。[完整判断](evidence/attention-transport-review.md)。

## C09

**2026-11-10 资格更新**：介绍：保留原序的selected-KV直接L1与尾块联合。到了哪里：原pair/半区fallback、910C uint16元素stride、QK/PV布局及CPU源地址检查已核。结果：两块L1 pin和较大joint待真实K/页分组、原生padding/LoadData/MMAD逐位、新flags与净预取代价；旧17.78ms不作实测，不复用逻辑序helper或A5宽stride。[当前前沿](evidence/smla-direct-l1-frontier.json)。

**M03 · SMLA direct KV reader＋首partial交付；shared panel为替代**

- **介绍**　按原选择顺序从分页 cache 直读 KV，消除 selected KV 在显存中的暂存和重读；首段 PV 直接交付。把不连续读取导致的 QK/PV 降速计入净值。
- **到了哪里**　待确认。条件估算。待领原型
- **结果**　情景：有效 K=256、QK/PV 等速，TTFT 省 17.78 ms（1.19%），整批省 21.93 ms（0.118%）。

<details><summary>代码与依据</summary>

- **代码定位**　`dsa_v41.py:613` `_forward_attention`、`sparse_flash_mla_csa_block_vector.h:606-653`、`sparse_flash_mla_csa_block_cube.h:416-436`、`sparse_flash_mla_csa_block_cube.h:704-726`
- **适用负载**　长提示G64，真实有效K256/512
- **成立条件**　K256相对QK/PV读速≥约0.805或等速75.9%行；保持原顺序/容量。分别比较原暂存与直读路径的整体耗时，保留相同 mask、softmax 和 partial 顺序。
- **证据**　[S07 代码与定价](evidence/S07.md)
- **接续**　比较完整 kernel 的精度与耗时

</details>

<!-- WORKBENCH-CONTINUATION-20261109-START -->
### 2026-11-09 交付合同续扫

- **介绍**　追到Vector实际物理packed顺序及两个Cube读者，收窄分页直读验收合同。
- **到了哪里**　待确认。静态消费者核对/离线模型；无新设备实验。
- **结果**　原KV pair有物理偏移重排、尾部fallback和负索引停止；新直读须匹配packed plane或过数值验收。原17.78ms仍为条件情景，实际K/整核速度待证。

[本次源码与完整判断](evidence/workbench-continuation.md) · [复算结果](evidence/workbench-continuation-results.json) · [源码摘录与hash](evidence/workbench-continuation-source-review.json)
<!-- WORKBENCH-CONTINUATION-20261109-END -->

## C10

**M03、M04 · SMLA/RoPE/OA直接产send布局，再交换窄OA结果**

- **介绍**　让逆 RoPE/OA 的生产者直接写发送布局，再交换较窄结果。已有显式 pack 变体关闭；按实际可覆盖形状核算额外权重与 KV 容量。
- **到了哪里**　待确认。初步判断。待定点复核
- **结果**　成本余量：理想 OA 前移后，新传输与融合的额外开销中位仅容许 0.214 ms；新变体净值待定。

<details><summary>代码与依据</summary>

- **代码定位**　`dsa_v41_cp.py:209` `_project_output`、`dsa_cp.py:77` `restore_tp_heads`、`dsa_v1.py:1732` `_wo_a_bmm`、`dsa_v1.py:1759` `_forward_o_proj`
- **适用负载**　需新合法shape覆盖与owner权重容量；旧guard命中0
- **成立条件**　stock+pack新增1.010ms已失败；全owner约2560MiB/rank不得挤KV
- **证据**　[S07 代码与定价](evidence/S07.md)
- **接续**　查实际可覆盖形状及权重额外容量。关联 M03

</details>


### 最新D-A2大行交付判断

- **介绍**　实际BF16 OA后接BF16 OB。query owner先算全部8组OA、再窄交换，保留OB/RS和OA舍入。
- **到了哪里**　待确认；14个原OA/OB typed成本组匹配。需要总2560MiB/额外2240MiB OA权重/rank，并保持KV容量；大行416/818、完整40权重逐位一致及OA+pack增量未验，不能继承M3结果。
- **结果**　payload 65536→16384B/query；同cohort最快通信代理、h50/h100µs/层场景最高条件信用1.938%/1.391%。旧M3 stock+pack数值虽过但价值未过，保持停止；只消逆RoPE+pack全免最高0.723%，停止独立1%路线。SMLA/OA联合producer及OA/OB/HC仍保留原舍入/归约门槛。[原读者、预算与具体缺口](evidence/attention-output-review.md)。

## C11

**M03 · QLI逐key压实评分与成员发布**

- **介绍**　压实有效 key 后评分，再按原 TopK 语义恢复顺序、并列和 filler。保留 gather、压实和恢复成本；四层共享成员转换的历史收益另有基线。
- **到了哪里**　待确认。初步判断。待定点复核
- **结果**　工作量：理想少 50.09% 评分 MAC；若暴露 223 波，旧整批达 1% 需每四次 QLI 净省 0.834 ms。

<details><summary>代码与依据</summary>

- **代码定位**　`indexer.py:181` `select_projected`、`dsa_v41.py:567` `_select_sparse_indices`
- **适用负载**　长C1 mode2；保持score/TopK/tie/filler/nonfinite
- **成立条件**　约旧1.981ms的42.1%；原member-only组合未过批量slot服务门槛
- **证据**　[S07 代码与定价](evidence/S07.md)
- **接续**　扣除 gather、压实和次序恢复

</details>

<!-- WORKBENCH-CONTINUATION-20261109-START -->
### 2026-11-09 交付合同续扫

- **介绍**　把评分主体、压实降速和gather/恢复新增开销写成同一净收益门槛。
- **到了哪里**　待确认。静态消费者核对/离线模型；无新设备实验。
- **结果**　223完全关键波情景需每四次净省0.833667ms。等速零开销需评分主体占旧段≥84.024%；极有利全段评分时新增成本≤0.158514ms。缺c/ρ/h。

[本次源码与完整判断](evidence/workbench-continuation.md) · [复算结果](evidence/workbench-continuation-results.json) · [源码摘录与hash](evidence/workbench-continuation-source-review.json)
<!-- WORKBENCH-CONTINUATION-20261109-END -->

## C12

**M05、M06 · norm/router GEMM/路由的共同producer**

- **介绍**　让 norm、router GEMM 和 TopK 共用输入并直接交付路由结果，消除中间重读和提交。保留动态路由必需的点积与选路数值顺序。
- **到了哪里**　待确认。初步判断。待定点复核
- **结果**　预算：Router 与 RMS 合计占同周期约 1.88%；周期降 1% 需净删该段约 53%。

<details><summary>代码与依据</summary>

- **代码定位**　`model.py:844` `rms_norm_cast`、`fused_moe.py:401` `_apply_router_linear`、`fused_moe.py:412` `_compute_router_logits`、`fused_topk_router.py:140` `_compute_routing`
- **适用负载**　动态sqrtsoftplus384/K6；无固定hash表
- **成立条件**　小ID/TopK-only全免费仍低值；新GEMM/数值/替代成本必须计入
- **证据**　[S08 代码与定价](evidence/S08.md)
- **接续**　分开核算必要 GEMM 和可删交付。关联 M05

</details>

<!-- WORKBENCH-CONTINUATION-20261109-START -->
### 2026-11-09 交付合同续扫

- **介绍**　按同窗分母将必要GEMM与可消边界彻底分账，结束同预算的小方案。
- **到了哪里**　待确认。静态消费者核对/离线模型；无新设备实验。
- **结果**　固定GEMM的子项C12a停止：非GEMM全免目标最高0.8963%周期，连草稿最高0.9637%。大方案若改变GEMM主体，须另证净差。

[本次源码与完整判断](evidence/workbench-continuation.md) · [复算结果](evidence/workbench-continuation-results.json) · [源码摘录与hash](evidence/workbench-continuation-source-review.json)

### C12a

- **介绍**　固定router GEMM，消除packet、TopK、RMS等全部边界任务。
- **到了哪里**　已停止。同rank同周期全免费仍低值。
- **结果**　目标最高0.8963%，连草稿最高0.9637%；不外推成服务上界，也不排除实质改变GEMM的大方案。
<!-- WORKBENCH-CONTINUATION-20261109-END -->

## C13

**M06、M07、M10 · shared/DFC、返回producer与草稿三层联合调度**

- **介绍**　分析共享专家与路由专家各自占用 Cube、Vector 和通信的时间，按能同时执行的部分重排；分别核算目标和草稿。K03 是其中一个具体实现。
- **到了哪里**　待确认。初步判断。待定点复核
- **结果**　情景：F21 独占资源且保留路由时可省 0.594 ms/周期，达到 1% 后新增成本余量中位仅 0.0427 ms。

<details><summary>代码与依据</summary>

- **代码定位**　`fused_moe.py:485` `_forward_impl`、`shared_experts.py:670` `forward`、`dispatch_ffn_combine_kernel.hpp:912-977`
- **适用负载**　target EP16和draft EP16分别计；所有必要expert服务保留
- **成立条件**　不能把wait-inclusive当全free；旧fork/分段/return/deferred/merge原变体停止
- **证据**　[S08 代码与定价](evidence/S08.md)
- **接续**　核对资源窗口；原 fork 不重测。关联 M07/M10

</details>


### C13最新资源范围：DFC20/40与完整shared4/8

- **介绍**　冻结host固定20AIC/40AIV，研究其余4/8承接shared DQ、INT32 Up、activation和Down；区别于仅把Up塞前段或尾段，两处不重复计同一Up。
- **到了哪里**　待确认，有继续判断价值。原四阶段全隐藏的条件信用最高2.248%；4/8实际伸长、原NZ权重与控核API要求的ND约束、逐原语bit identity及真实BlockDim/共驻仍待核。core数相加不是资源独占证明。
- **结果**　只藏Up的C8诊断cohort最高条件信用0.906%，停止该C8独立1%路线的τ补证；decode/其它负载不外推。whole shared至少须兑现约44.5%最有利信用才到1%且无新增成本，非预计收益。下一步先查接口/原语实现，再决定固定4/8 native门槛。[完整判断](evidence/moe-delivery-frontier.md) · [预算](evidence/shared-4cube-residency-budget.json)。

## C14

**M09、M10 · aux owner→main/norm→三个context KV共同发布**

- **介绍**　从 aux 所在卡组织 main、norm 和三套 context KV 的输入与发布，复用前缀读写。三套 KV 权重、规定的 BF16 边界和各 bank 的发布时点均保留。
- **到了哪里**　待确认。初步判断。待定点复核
- **结果**　预算：原 context 前缀 1.050 ms（同周期 1.897%）；周期降 1% 需净省该段约 52.7%。

<details><summary>代码与依据</summary>

- **代码定位**　`dspark.py:281` `combine_hidden_states`、`dspark.py:284` `_project_shared_kv`、`dspark.py:295` `precompute_and_store_context_kv`、`dspark_proposer.py:226` `set_inputs_first_pass`
- **适用负载**　三份pre-layer aux；保留main norm与三个不同KV权重
- **成立条件**　多付main权重约137.63MB/rank；新M精度、行退休、输入/银行协议；原primitive余量4.61µs
- **证据**　[S09 代码与定价](evidence/S09.md) · [S10 代码与定价](evidence/S10.md)
- **接续**　扣除权重移动、行重排和同步。关联 M09

</details>

## C15

**M09、M11 · 分离sample-copy-ready与draft/cache-write-retired**

- **介绍**　将 token 可交付事件与草稿完成、cache 可回收事件分开。按客户端原响应端点计提前量，后台继续完成必要写入并处理取消和页退役。
- **到了哪里**　待确认。初步判断。待定点复核
- **结果**　上界：C8 即使去掉全部可释放等待也仅省 12.807 ms；普通 code-single 整响应达 1% 需净省 4.666 ms。

<details><summary>代码与依据</summary>

- **代码定位**　`model_runner_v1.py:1876` `_copy_valid_sampled_token_count`、`model_runner_v1.py:2163` `_copy_draft_token_ids_to_cpu`、`model_runner_v1.py:2764` `sample_tokens`、`gpu_model_runner.py:345` `get_output`、`scheduler.py:1874` `update_from_output`
- **适用负载**　async无logprobs/grammar；客户端输出与安全回收分开
- **成立条件**　保持页回收/abort/stale/bank退役；其它请求按各自整响应1%
- **证据**　[S11 代码与定价](evidence/S11.md)
- **接续**　核对普通请求端点、页生命周期及取消路径。关联 M09

</details>

## K00

**M06 · 当前DFC megakernel**

- **介绍**　BF16 expert input + IDs/weights -> dispatch/EP/GMM1/activation/GMM2/return/unpermute。基线已融合routing quant/EP/GMM1/act quant/GMM2/return/unpermute。已有一调用；必要专家结果和通信仍做
- **到了哪里**　已确认可实现。既有实现。复用
- **结果**　基线事实；无第二份融合收益

<details><summary>代码与依据</summary>

- **代码定位**　`moe_comm_method.py:563` `fused_experts`、`dispatch_ffn_combine.h:244-287`、`dispatch_ffn_combine_kernel.hpp:912-977`
- **适用负载**　target384/K6和draft128/K3按各自参数
- **成立条件**　保留CombineV1/V2、权重/scales、mask、peer计数/flags与退休
- **证据**　[实现切口与公式](evidence/megakernel.md) · [逐窗口模型](evidence/megakernel-window-model.json)
- **接续**　作为参照，不重复计收益

</details>

## K01

**M02、M03 · Post/HcPre/RMS到本地KV-ready的前端**

- **介绍**　rounded Post -> HcPre -> RMS -> local KV/C1 projection/norm/RoPE -> KV-ready packet + Y/coeff outputs; Qb sibling outside base launch。局部转发和少spill/调用；owner重排另消重复投影与宽AG。保留rounded残差/迟到系数；窄KV交付是独立通信边界
- **到了哪里**　待确认。设计与条件估算。与C08联合评估
- **结果**　owner机制已有60.54ms毛包络，50%情景30.27ms；不是megakernel新增独占收益

<details><summary>代码与依据</summary>

- **代码定位**　`model.py:871` `hc_post`、`model.py:879` `forward`、`dsa_v41.py:327` `_project_kv`、`dsa_v41.py:350` `_preprocess`
- **适用负载**　长prefill；三C2源保持raw；入口/Engram/aux各守时点
- **成立条件**　净兑现≥24.64%旧TTFT目标；基础launch不塞Qb拖KV-ready；换M精度/tiling；只40个Attention Pre需净删39.05%旧耗时
- **证据**　[实现切口与公式](evidence/megakernel.md) · [逐窗口模型](evidence/megakernel-window-model.json)

</details>

<!-- THREE-MEGA-PERFORMANCE-20261109-START -->
### 2026-11-09 净性能补价

- **介绍**　到KV-ready为止的行owner前端；本轮区分owner总收益与同owner基线下的megakernel边际。
- **到了哪里**　待确认。离线条件模型已复算，没有新内核或设备/服务实测。
- **结果**　owner毛包络分为计算22.412ms、通信38.132ms；各50%暴露再扣额外5ms省25.272ms（旧TTFT尺度1.694%）。megakernel独立边际待定。 保持KV-ready早截止、原BF16/FP32/HF32和读者；同owner多kernel先作基线，不能把owner收益重复归给fusion。50%情景额外成本余量15.354ms。

[详细性能分析](evidence/three-megakernel-performance.md) · [输入](evidence/three-megakernel-inputs.json) · [结果](evidence/three-megakernel-performance.json)
<!-- THREE-MEGA-PERFORMANCE-20261109-END -->

## K02

**M03、M04 · SMLA direct/partial到inverse RoPE与可选OA send producer**

- **介绍**　selected KV -> QK/softmax/PV -> inverse RoPE -> owner OA -> send layout。KV gather/partial交接；更大分支输出payload65536→16384B/query。保留原partial合并/BF16边界和跨8head组的OA依赖；之后窄交换
- **到了哪里**　待确认。设计与条件估算。与C09联合评估
- **结果**　direct+partial K256等速旧TTFT条件约17.78ms；direct+OA联合收益未知

<details><summary>代码与依据</summary>

- **代码定位**　`dsa_v41.py:613` `_forward_attention`、`sparse_flash_mla_csa_block_vector.h:606-653`、`sparse_flash_mla_csa_block_cube.h:416-436`、`sparse_flash_mla_csa_block_cube.h:704-726`、`dsa_v41_cp.py:209` `_project_output`
- **适用负载**　长prefill G64；实际有效K；小shape不外推
- **成立条件**　QK/PV相对速度≥约0.805；新OA transport/fusion余量模型中位0.214ms；额外OA权重2240MiB/rank；原partial/rounding与资源账
- **证据**　[实现切口与公式](evidence/megakernel.md) · [逐窗口模型](evidence/megakernel-window-model.json)

</details>

<!-- THREE-MEGA-PERFORMANCE-20261109-START -->
### 2026-11-09 净性能补价

- **介绍**　SMLA direct/partial至可选OA-send producer；本轮给出reader降速和OA净新增成本的共同门槛。
- **到了哪里**　待确认。离线条件模型已复算，没有新内核或设备/服务实测。
- **结果**　K256等速direct基准17.777ms；OA净增关键惩罚≤2.859ms才守住旧TTFT尺度1%。仅PV读速75%时余量0.883ms；两侧75%时基准仅13.827ms。 先保护direct/partial速度与真实K；同窗测OA增量δ，保持KV容量/原rounding。2240MiB额外owner权重及另一cohort的0.214ms门槛不得直接加到direct收益。

[详细性能分析](evidence/three-megakernel-performance.md) · [输入](evidence/three-megakernel-inputs.json) · [结果](evidence/three-megakernel-performance.json)
<!-- THREE-MEGA-PERFORMANCE-20261109-END -->

## K03

**2026-11-10 资源更正**：介绍：DFC与共享分支的整段共驻。到了哪里：wholeDFC的2560旧任务均为24AIC/48AIV，40仅内部routing tiler；旧20/40闲核前提撤回。结果：4/8独立primitive bit事实不等于共驻，待实际资源/flags、每替换原语逐位与净请求成本；旧局部τ/ρ模型保留为历史场景。[资源与整段证据](evidence/DFC原生交付与24核资源更正.md)。

**M06 · DFC内部Cube尾部承接shared GateUp**

- **介绍**　原 DFC 的 routed GMM2 完成后，让 Cube 承接 shared GateUp，与 Vector 的 combine、跨卡同步和 unpermute 重叠。两侧完成后再做激活、Down 和合并。每周期净省 = Σ min(原 GateUp 耗时, 可用尾部时长) − Σ 新同步与争用。
- **到了哪里**　待确认。设计与条件估算。优先补尾部时长证据
- **结果**　每层可用尾部 τ=20 µs、新增开销 h=5 µs 时，估省 0.60 ms/本地周期，中位 1.08%（112 个窗口中 96 个达到 1%）。

<details><summary>代码与依据</summary>

- **代码定位**　`dispatch_ffn_combine_kernel.hpp:912-977`、`shared_experts.py:471-482`、`shared_experts.py:534-564`
- **适用负载**　target SP本地M3；prefill/draft另判
- **成立条件**　h5µs时τ门槛18.41–22.26µs，中位18.89；τ尚未测得；量化、AIC仲裁/flags、Vector全核屏障与争用须成立
- **证据**　[实现切口与公式](evidence/megakernel.md) · [逐窗口模型](evidence/megakernel-window-model.json)
- **接续**　先量可用尾部与新增争用。关联 C13 的实现分支

</details>

<!-- THREE-MEGA-PERFORMANCE-20261109-START -->
### 2026-11-09 净性能补价

- **介绍**　把shared GateUp嵌入DFC Cube尾部；本轮将Up吞吐、其它开销、层覆盖和端点次数分账。
- **到了哪里**　待确认。离线条件模型已复算，没有新内核或设备/服务实测。
- **结果**　τ20µs/h5µs、40层时：Up吞吐ρ1省0.600ms/周期（中位1.080%）；ρ0.9仅0.486ms（0.876%）。同一局部周期情景，不是整请求实测。 先核τ/ρ/h/适用层数和关键迭代数；ρ已含Up降速，h不双扣。τ20/h5需ρ中位约0.9584；0.60ms/关键迭代到旧整批1%需310次完全暴露。

[详细性能分析](evidence/three-megakernel-performance.md) · [输入](evidence/three-megakernel-inputs.json) · [结果](evidence/three-megakernel-performance.json)
<!-- THREE-MEGA-PERFORMANCE-20261109-END -->

## K04

**M10 · DSpark三层或whole-draft常驻**

- **介绍**　three independent context banks -> one five-query three-layer forward -> collapse/head/Markov。重复调用/交接/初始化；必要TP/EP、cache及GEMM保留。保留3次EP服务、每层SP/CP/TPcollective和五步ID依赖；需设备TP协议
- **到了哪里**　待确认。设计与条件估算。待协议
- **结果**　三层仅gap免费后仍需删native24.32%；whole draft需11.62%；都是同周期门槛

<details><summary>代码与依据</summary>

- **代码定位**　`dspark.py:248` `forward`、`dspark.py:403` `forward`、`llm_base_proposer.py:1013` `_propose`、`llm_base_proposer.py:1454` `_run_merged_draft`
- **适用负载**　三层一次五query；EP16/128/K3；头五步依赖
- **成立条件**　需接入设备TP/CP协议、metadata/flag epochs、资源和精度；不能把host层循环当成megakernel
- **证据**　[实现切口与公式](evidence/megakernel.md) · [逐窗口模型](evidence/megakernel-window-model.json)

</details>

## K05

**M09、M10 · aux owner到main norm与三套context KV**

- **介绍**　three local aux rows -> main15360-to5120 -> RMS -> three distinct KV projections -> cache store。宽aux交付、重复重组与全局KV投影；主norm和不同WKV保留。main norm不能穿线性；保留三个cache组的slot/代次；需要row-owner权重
- **到了哪里**　待确认。设计与条件估算。待证
- **结果**　原prefix1.050ms；到同周期1%需净删52.71%；尚无新实现预计值

<details><summary>代码与依据</summary>

- **代码定位**　`dspark.py:281` `combine_hidden_states`、`dspark.py:284` `_project_shared_kv`、`dspark.py:295` `precompute_and_store_context_kv`、`dspark_proposer.py:226` `set_inputs_first_pass`
- **适用负载**　DSpark context；实际所需行、三个physical组
- **成立条件**　完整main权重多137.63MB/rank；BF16/main norm不得穿越；旧primitive余量4.61µs，不做原样拼接
- **证据**　[实现切口与公式](evidence/megakernel.md) · [逐窗口模型](evidence/megakernel-window-model.json)

</details>

## K06

**M01 · Engram固定位置plane融合**

- **介绍**　history/hash/CPU-UVA -> WKV -> rotation/gate -> rounded residual。中间plane/局部lookup/gate交付。权重/完整tuple不同不复用；内容缓存收益与单纯fusion分开
- **到了哪里**　已停止。设计与条件估算。停止原范围
- **结果**　所列gate+lookup全免最高约0.743%同周期，局部停止

<details><summary>代码与依据</summary>

- **代码定位**　`hash_state.py:79` `forward`、`embedding.py:333` `embed_gathered`、`embedding.py:378` `lookup`、`layer.py:34` `forward`
- **适用负载**　层1/14 TP8 CPU-UVA/rotated BF16 WKV
- **成立条件**　完整内容缓存或数学/owner重写另判；需旧cacheable work消失71.1%–96.3%再扣新成本
- **证据**　[实现切口与公式](evidence/megakernel.md) · [逐窗口模型](evidence/megakernel-window-model.json)

</details>

## K07

**M00、M11 · 跨组slot与共同输入publisher**

- **介绍**　accepted old state/request map -> ID/position/length/spec/slot publication。重复索引/slot任务、提交与共同plan交付。host镜像与CP CPU plan仍有读者，已有13组linear slot已融合
- **到了哪里**　待确认。B4 批写有历史实测，更大输入发布仍在判断。复用
- **结果**　历史B4整批3.39%/2.22%；共同输入已定价设备全免61.73ms

<details><summary>代码与依据</summary>

- **代码定位**　`block_table.py:469` `compute_slot_mapping`、`block_table.py:504` `compute_slot_mapping_draft`、`model_runner_v1.py:3561` `_get_block_table_and_slot_mapping`
- **适用负载**　B4真实26-job已有历史价值；B1另判
- **成立条件**　更大输入publisher还需其它净省124.17ms/旧整批；host镜像/DP协调保留；已有13组linear slot不重复计
- **证据**　[实现切口与公式](evidence/megakernel.md) · [逐窗口模型](evidence/megakernel-window-model.json)

</details>

## K08

**M09、M10 · 五Markov与局部head融合**

- **介绍**　needed target logits -> greedy/rejection; draft base + five dependent Markov heads -> IDs。局部add/argmax/copy/调用；必要dot保留。必要dot和五步依赖不消失；末层dead行裁剪另外定价
- **到了哪里**　已停止。设计与条件估算。停止原范围
- **结果**　五步55任务即全免最高0.8504%周期；保持dot的target/draft联合旧整批179.925ms<185.908ms

<details><summary>代码与依据</summary>

- **代码定位**　`dspark.py:311` `hc_head`、`dspark.py:320` `markov_embed`、`dspark.py:323` `markov_bias`、`dspark.py:326` `compute_logits`、`llm_base_proposer.py:1400` `_sample_draft_from_logits`
- **适用负载**　五步全vocab greedy；ID依赖保留
- **成立条件**　局部整组停止；更大controller只计新增可删工作
- **证据**　[实现切口与公式](evidence/megakernel.md) · [逐窗口模型](evidence/megakernel-window-model.json)

</details>

## K09

**全模型 · 40层或跨迭代无限常驻**

- **介绍**　complete target or target/draft loop controlled by device descriptors。潜在跨阶段提交/驻留，尚无可信净收益值。现有host scheduler、CPU DP协调、output/default stream完成与cache退休必须重做协议
- **到了哪里**　待确认。概念设计，尚无可报告净值。待协议
- **结果**　当前未具备可接入实现或可报告性能值，不排在局部方案前

<details><summary>代码与依据</summary>

- **代码定位**　`model.py:1505` `forward`、`model_runner_v1.py:2229` `execute_model`、`scheduler.py:1874` `update_from_output`
- **适用负载**　全部TP8/EP16参与与CPU scheduler交互
- **成立条件**　需设备TP、host命令/epoch、output-ready与cache-retired分离；当前default-stream完成/页回收合同不能直接沿用
- **证据**　[实现切口与公式](evidence/megakernel.md) · [逐窗口模型](evidence/megakernel-window-model.json)

</details>

## K10

**M06、M08 · 末层需求驱动selected query/FFN/readout producer**

- **介绍**　layer39 KV all actual rows + selected Q/FFN/head rows -> sampled IDs。无读者的非logit行query/output、FFN/readout；不删远端服务。先合法退役raw bank；保留aux与全量SWA cache、远端真实专家
- **到了哪里**　待确认。设计与条件估算。待读者核对
- **结果**　FFN+readout有利上界22.992ms；需净兑现64.89%；不是预计值

<details><summary>代码与依据</summary>

- **代码定位**　`model.py:1505` `forward`、`model.py:1778` `compute_logits`、`fused_moe.py:485` `_forward_impl`
- **适用负载**　layer39长prefill；SWA全实际行保留
- **成立条件**　先合法退役raw bank/observer/图发布；query metadata/换M精度/EP及全SWA状态保留
- **证据**　[实现切口与公式](evidence/megakernel.md) · [逐窗口模型](evidence/megakernel-window-model.json)

</details>

## history46-F01

**M00、M11 · slot 坐标与 flat 索引双输出**

- **介绍**　block table+position → 坐标与flat slot → cache writers；接受计数/位置 → 26项slot双输出 → 目标图。同一输入除模、索引生成及一次独立提交。
- **到了哪里**　已确认可实现。服务实测·限定基线。已有实测。双输出已有实现，沿用同轮基线。
- **结果**　各自CP基线：15.034s→14.217s，整批-5.43%

<details><summary>代码与依据</summary>

- **代码定位**　`block_table.py:469` `compute_slot_mapping`、`block_table.py:504` `compute_slot_mapping_draft`、`model_runner_v1.py:3561` `_get_block_table_and_slot_mapping`
- **适用负载**　冻结C8；匹配CP条件及slot银行
- **成立条件**　两份输出值、地址、无效行和代次不变；不与F02百分比相加
- **证据**　[module-fusion-estimates](../baselines/module-fusion-estimates.json)
- **接续**　已有实现，按该基线复用。关联 M11

</details>

## history46-F02

**M00、M11 · 26 项 slot writer 批量调用**

- **介绍**　block table+position → 坐标与flat slot → cache writers；接受计数/位置 → 26项slot双输出 → 目标图。26次提交合一次，保留所有写入。
- **到了哪里**　已确认可实现。服务实测·限定基线。已有实测。26项批写已有实现。
- **结果**　13.083/12.926s→12.639s，整批-3.39%/-2.22%

<details><summary>代码与依据</summary>

- **代码定位**　`block_table.py:469` `compute_slot_mapping`、`block_table.py:504` `compute_slot_mapping_draft`、`model_runner_v1.py:3561` `_get_block_table_and_slot_mapping`
- **适用负载**　冻结C8 B4
- **成立条件**　基线已含CP+双输出+CPU计划；适用26项写入
- **证据**　[M11](../modules/M11.md) · [ep16-batched-slot-c8](../baselines/ep16-batched-slot-c8.json)

</details>

## history46-F03

**M00、M11 · 小批量 13/26 项 slot 批写**

- **介绍**　block table+position → 坐标与flat slot → cache writers；13全局writer+活跃本地writer → 后续decode。实际caller提交与可暴露等待。
- **到了哪里**　待确认。已完成初筛。不独立立项。属于F02的小批量扩形状，13项分支尚无足够净省证据。
- **结果**　26项0.961ms/步时，13项需0.880ms/步才两轮整段均≥1%

<details><summary>代码与依据</summary>

- **代码定位**　`block_table.py:469` `compute_slot_mapping`、`block_table.py:504` `compute_slot_mapping_draft`、`model_runner_v1.py:3561` `_get_block_table_and_slot_mapping`
- **适用负载**　普通B1后续对话；B2/B3待时序
- **成立条件**　单请求caller净省尚未实测；短代码独立降级，不把B4外推为B1
- **证据**　[ordinary-input-and-slot](../ordinary-input-and-slot.md) · [slot-batch-by-population](../slot-batch-by-population.md)
- **接续**　查 13 项分支的 caller 净时间。关联 C03 的扩形状

</details>

## history46-F04

**M02、M05、M07 · 两处 Post→mHC premix/RMS 共读残差**

- **介绍**　WKV/gate → BF16四路残差 → 下层mHC；上一层FFN HcPost → Y/premix → RMS → TP8 AllGather；embedding或更新残差/均值读者 → Attention HcPre。残差读取、Y写读和合法多输出交付。
- **到了哪里**　已停止。已完成初筛。暂不立项。若新增扩宽写入，残差共读接近或低于历史请求门槛；跨层服务收益不稳定。
- **结果**　同速字节模型9.12–18.23ms；宽写+25%时14.59ms；原生直连中位10.53/14.02ms

<details><summary>代码与依据</summary>

- **代码定位**　`model.py:858` `hc_pre`、`model.py:871` `hc_post`、`model.py:879` `forward`、`hc_pre_cube_compute.h:398-477`
- **适用负载**　长提示TTFT；34直接Attention入口+40FFN入口
- **成立条件**　保持原舍入、FP32系数和旁路；全量18.23情景要兑现≥81.8%；跨层服务未稳定
- **证据**　[hc-paired-boundaries](../baselines/hc-paired-boundaries.json) · [cross-layer-hc-fusion](../baselines/cross-layer-hc-fusion.json)

</details>

## history46-F05

**M03 · 四 QLI 消费层共享一次候选成员转换**

- **介绍**　第20层candidate → 24/28/32/36层共享成员表示。重复mask/成员表示生成。
- **到了哪里**　已确认可实现。服务实测·限定基线。已有实测。四层共享单改有效，已有批写基线下的增量尚不稳定。
- **结果**　单改整批-1.50%/-1.48%；slot基线上-1.27%/-1.56%，样本重叠

<details><summary>代码与依据</summary>

- **代码定位**　`indexer.py:127` `select`、`indexer.py:181` `select_projected`、`dsa_v41.py:567` `_select_sparse_indices`
- **适用负载**　长上下文C8，24/28/32/36层
- **成立条件**　保留各层query/score/TopK；后者增量未确立
- **证据**　[qli-service-c8](../baselines/qli-service-c8.json) · [qli-batched-slot-c8](../baselines/qli-batched-slot-c8.json)
- **接续**　已有实现；叠加批写的增量未确立

</details>

## history46-F06

**M03 · QLI 四消费层内截短列表归并**

- **介绍**　四消费层token归并 → 稳定index bank。归并比较及列表/分数面写读。
- **到了哪里**　已停止。已完成初筛。暂不立项。归并占旧核时的比例未证实；不能按整个QLI核时计收益。
- **结果**　七非生产层TTFT过线需约20%核时净减；修正版联合整批需消费层净减60.6%–63.5%

<details><summary>代码与依据</summary>

- **代码定位**　`indexer.py:127` `select`、`indexer.py:181` `select_projected`、`dsa_v41.py:567` `_select_sparse_indices`
- **适用负载**　长提示TTFT；修正B4图整批另计
- **成立条件**　若归并提速75%，需原消费QLI核时81%–85%属归并；profile未分出；前三层+源层免费仍不足
- **证据**　[four-mixed-fusion-edges](../four-mixed-fusion-edges.md) · [joint-attention-frontiers](../joint-attention-frontiers.md)

</details>

## history46-F07

**M03 · SMLA KV 直读与首段 PV 直写**

- **介绍**　indices/gather → SWA与compressed QK/softmax/PV；KV源页及共享TopK → 后续层；分块max/sum/PV → 最终attention结果。BF16临时面板写读、FP32 PV部分结果写读。
- **到了哪里**　待确认。已完成初筛。优先原型。沿C09，从真实分页KV写回和QK/PV重读入手，先做局部对照。
- **结果**　256位置、50%差额暴露：17.77–17.81ms(1.19%)；512位置KV单项23.72–23.77ms(1.59%)

<details><summary>代码与依据</summary>

- **代码定位**　`dsa_v41.py:613` `_forward_attention`、`sparse_flash_mla_csa_block_vector.h:606-653`、`sparse_flash_mla_csa_block_cube.h:416-436`、`sparse_flash_mla_csa_block_cube.h:704-726`
- **适用负载**　长提示38压缩层TTFT；短/空cache回退
- **成立条件**　256位置PV+KV需稀疏读速达参考80.4%–80.6%；两次QK/PV读取、sink、归约和发布不变
- **证据**　[smla-partial-handoffs](../smla-partial-handoffs.md) · [smla-panel-occupancy](../baselines/smla-panel-occupancy.json) · [sparse-selection-delivery](../baselines/sparse-selection-delivery.json)

</details>

## history46-F08

**M04 · 大行 OA→OB 同核分块流水**

- **介绍**　OA BF16块 → OB → ReduceScatter输入；a=roundBF16(xWa); y=aWb。舍入后BF16块交付与可兑现投影重叠。
- **到了哪里**　待确认。已完成初筛。备选原型。大prefill有已绑定形状和可删除的投影交接，但依赖搬运/计算重叠；M24排除。
- **结果**　长 prefill 的 TTFT 估省 15.34–17.94 ms（1.03%–1.20%），整批约 0.08%–0.10%。假设两种资源均达参考速率的 90%，短阶段可重叠 50%。

<details><summary>代码与依据</summary>

- **代码定位**　`dsa_v1.py:1732` `_wo_a_bmm`、`dsa_v1.py:1759` `_forward_o_proj`、`dsa_v41_cp.py:209` `_project_output`
- **适用负载**　大prefill TTFT；M24回原图
- **成立条件**　两次乘法和BF16中间舍入保留；资源/权重搬运与同步成本需扣；纯边界5.70–6.51ms不足
- **证据**　[oa-ob-shared-cube](../oa-ob-shared-cube.md) · [oa-ob-fusion](../baselines/oa-ob-fusion.json)
- **接续**　先查片上容量、权重搬运与重叠

</details>

## history46-F09

**M06 · 固定路由专家跨核激活交付**

- **介绍**　TopK/hidden → DispatchFFNCombine → routed out；FP16 gate/up → SwiGLU+Quant → INT8激活+scale → GMM2。GMM1 FP16写读、INT8激活的实际跨核传输。
- **到了哪里**　已停止。已完成初筛。暂不立项。多核仍需同一激活；没有可兑现的新广播或驻留成本依据。
- **结果**　单核最多3次读含前侧11.13ms(0.75%)；全部逻辑传输免费敏感性29.35ms(1.97%)

<details><summary>代码与依据</summary>

- **代码定位**　`dispatch_ffn_combine_kernel.hpp:793-823`、`dispatch_ffn_combine_kernel.hpp:912-977`
- **适用负载**　长提示TTFT；固定EP16、专家/权重分布
- **成立条件**　要过1%，前侧全省后每分派需净省约6.74次INT8单向整行传输；跨核发送与同步要付；20逻辑读≠20次HBM miss
- **证据**　[ffn-merge-premix](../ffn-merge-premix.md) · [routed-activation-readers](../baselines/routed-activation-readers.json)

</details>

## history46-R01

**M00、M01 · 词嵌入/Engram 输入交付**

- **介绍**　token ID → 查表/掩码/SUM → 持久输入 bank；lookup head → SP行所有者/WKV。重复查表、输入重排、SP交付。
- **到了哪里**　已停止。已完成初筛。不立项。按原范围停止。图外bank有其它读者则扣回；不同层权重不能缓存成同一WKV
- **结果**　embedding具名设备+launch整批0.482%，普通≤0.667%；协议常见组分别≤0.083%/0.085%

<details><summary>代码与依据</summary>

- **代码定位**　`model.py:1392` `prepare_engram_inputs`、`model.py:1505` `forward`、`vl_model.py:190` `embed_input_ids`
- **适用负载**　冻结C8及六普通请求
- **成立条件**　图外bank有其它读者则扣回；不同层权重不能缓存成同一WKV
- **证据**　[ordinary-input-and-slot](../ordinary-input-and-slot.md)

</details>

## history46-R02

**M00、M11 · 元数据 tiling 多输出**

- **介绍**　公共tiling → 多份固定地址metadata → 图/eager读者。重复构造与copy/launch。
- **到了哪里**　已停止。已完成初筛。不立项。按原范围停止。原就绪信号早于读者，不计整段异步等待
- **结果**　全部具名API免费：整批0.141%，普通≤0.161%

<details><summary>代码与依据</summary>

- **代码定位**　`dsa_v41.py:879` `_publish_task`、`dsa_v41.py:929` `build`、`device_metadata.py:74` `submit`、`device_metadata.py:130` `wait`
- **适用负载**　当前图与eager
- **成立条件**　原就绪信号早于读者，不计整段异步等待
- **证据**　[profile-boundary-audit](../profile-boundary-audit.md)

</details>

## history46-R03

**M01 · Engram 局部 gate/lookup 与尾部**

- **介绍**　历史token/slot → hash → lookup → WKV；FP32(q_weight*k_weight)、块旋转 → 加权dot；WKV/gate → BF16四路残差 → 下层mHC。统计共读、静态权重乘积、尾部回写。
- **到了哪里**　已停止。已完成初筛。不立项。按原范围停止。这是两种不同范围/分母；不能相加；联合尾部已在F04预算
- **结果**　gate+lookup全免≤0.743%本地周期；尾部TTFT1.30–2.05ms

<details><summary>代码与依据</summary>

- **代码定位**　`hash_state.py:79` `forward`、`embedding.py:333` `embed_gathered`、`embedding.py:378` `lookup`、`layer.py:34` `forward`
- **适用负载**　两个实际Engram入口
- **成立条件**　这是两种不同范围/分母；不能相加；联合尾部已在F04预算
- **证据**　[initial-hc-engram](../initial-hc-engram.md)

</details>

## history46-R04

**M00、M02 · 首层虚拟四路与草稿 padding**

- **介绍**　embedding或更新残差/均值读者 → Attention HcPre；draft embedding → mHC/attention/MoE三层 → collapse/norm。repeat、初始化、重复原行读取。
- **到了哪里**　已停止。已完成初筛。不立项。按原范围停止。HcPre/HcPost都要适配原BF16行，不省后续残差输出
- **结果**　C8整批15.88ms(0.085%)，普通≤0.081%

<details><summary>代码与依据</summary>

- **代码定位**　`model.py:879` `forward`、`model.py:1505` `forward`、`dspark_proposer.py:226` `set_inputs_first_pass`
- **适用负载**　目标/DSpark首层
- **成立条件**　HcPre/HcPost都要适配原BF16行，不省后续残差输出
- **证据**　[initial-hc-engram](../initial-hc-engram.md)

</details>

## history46-R05

**M02、M03 · 独立提前本地 Q 与空侧提早**

- **介绍**　Y/RMS → 本地Q或全局KV → 汇合。仅隐藏Q投影。
- **到了哪里**　已停止。已完成初筛。不立项。按原范围停止。不把排程移动当作删掉计算
- **结果**　TTFT最有利9.011ms<14.919ms；空侧对EP16到达0

<details><summary>代码与依据</summary>

- **代码定位**　`dsa_v41.py:320` `_project_q`、`dsa_v41.py:333` `_project_q_kv`、`dsa_v41_cp.py:189` `_prepare_inputs_and_caches`
- **适用负载**　当前34直连和空侧
- **成立条件**　不把排程移动当作删掉计算
- **证据**　[attention-service-budget](../baselines/attention-service-budget.json)

</details>

## history46-R06

**M03 · Attention 输入多投影/旋转/量化/cache 融合组**

- **介绍**　QA/QR、Qb、index Q/W、KV norm/RoPE → cache。重复输入、qr量化、旋转发布中间面。
- **到了哪里**　已停止。已完成初筛。不立项。其中Qb/index-Q共享动态量化：八站点第二次量化合计22.58µs/图，未扣共享开销；不单开项目。
- **结果**　直接搬运整批7.33–9.27ms；半向量工作+间隙全省0.68%–0.70%

<details><summary>代码与依据</summary>

- **代码定位**　`dsa_v41.py:320` `_project_q`、`dsa_v41.py:327` `_project_kv`、`dsa_v41.py:440` `multistream_preprocess`、`indexer.py:167` `project_query`、`indexer.py:176` `quantize_query`
- **适用负载**　实际主辅流、KV-only与普通形状
- **成立条件**　相同提前排程；不同quant/scale不合并；两流等候不可无条件删
- **证据**　[attention-input-fusions](../baselines/attention-input-fusions.json)

</details>

## history46-R07

**M03 · C2 与索引输出接口联合**

- **介绍**　FP32 WKV/Wgate → pool/ring/RMS → indexK与longKV。pool/ring及索引准备中间面。
- **到了哪里**　已停止。已完成初筛。不立项。按原范围停止。持久索引有后读者；第二流水组可并行须另证
- **结果**　TTFT0.353ms、整批0.461ms；索引准备全免TTFT6.56–6.70ms

<details><summary>代码与依据</summary>

- **代码定位**　`dsa_v41.py:497` `_write_compressed_source`、`compressor.py:49` `prepare_ring_compressor`、`compressor.py:55` `pool_projected`、`indexer.py:106` `update_keys`
- **适用负载**　C1/C2实际源层
- **成立条件**　持久索引有后读者；第二流水组可并行须另证
- **证据**　[c2-selection-pipeline](../c2-selection-pipeline.md)

</details>

## history46-R08

**M04 · Attention 直接写发送布局/接收逆旋转**

- **介绍**　attention output → inverse RoPE/布局 → TP8 AllToAll。pad/transpose/copy或旋转接口。
- **到了哪里**　已停止。已完成初筛。不立项。按原范围停止。两者替代；448/64与正零及接收路径保留
- **结果**　发送整批0.49%–0.51%；接收端新增免费也仅TTFT4.84–5.33ms

<details><summary>代码与依据</summary>

- **代码定位**　`dsa_v41_cp.py:209` `_project_output`、`dsa_cp.py:77` `restore_tp_heads`、`dsa_v1.py:1732` `_wo_a_bmm`、`dsa_v1.py:1759` `_forward_o_proj`
- **适用负载**　有/无AllToAll、空发送
- **成立条件**　两者替代；448/64与正零及接收路径保留
- **证据**　[attention-send](../baselines/attention-send.json)

</details>

## history46-R09

**M02、M07 · 宽残差工作区交付**

- **介绍**　RS结果+原残差 → HcPost → 下个HcPre；BF16残差 → FP32/HF32投影工作区 → Cube。少一份BF16读，或只存FP32。
- **到了哪里**　已停止。已完成初筛。不立项。按原范围停止。后读Y/FFNPost各多读宽面抵消；不支持免费UB→L1假设
- **结果**　双输出TTFT0.36–3.93ms；只存宽平面净省字节0

<details><summary>代码与依据</summary>

- **代码定位**　`model.py:858` `hc_pre`、`model.py:871` `hc_post`、`model.py:879` `forward`、`hc_pre_cube_compute.h:398-477`
- **适用负载**　40FFN/相邻Post
- **成立条件**　后读Y/FFNPost各多读宽面抵消；不支持免费UB→L1假设
- **证据**　[hc-workspace-delivery](../baselines/hc-workspace-delivery.json)

</details>

## history46-R10

**M05、M06 · Norm 同时产三种输入**

- **介绍**　归一化 → router宽输入/共享专家INT8+scale。absmax与匹配量化输入写读。
- **到了哪里**　已停止。已完成初筛。不立项。按原范围停止。router宽输入是规定舍入；六份dispatch行不是本地稠密输入
- **结果**　TTFT慢2.01至快3.09ms；给路由首段50%附加净省仍8.40ms

<details><summary>代码与依据</summary>

- **代码定位**　`model.py:844` `rms_norm_cast`、`fused_moe.py:389` `_prepare_router_input`
- **适用负载**　router FP32/共享INT8
- **成立条件**　router宽输入是规定舍入；六份dispatch行不是本地稠密输入
- **证据**　[activation-quant](../baselines/activation-quant.json)

</details>

## history46-R11

**M06 · Router/TopK/ID/dispatch 入口**

- **介绍**　归一化宽输入 → gate → TopK/ID/权重 → dispatch。分数和ID中间面。
- **到了哪里**　已停止。已完成初筛。不立项。按原范围停止。MM与选中权重保留；归档未传固定表；主机晚提交另记
- **结果**　TTFT1.65–1.85ms、整批0.27%–0.36%

<details><summary>代码与依据</summary>

- **代码定位**　`model.py:844` `rms_norm_cast`、`fused_moe.py:401` `_apply_router_linear`、`fused_moe.py:412` `_compute_router_logits`、`fused_topk_router.py:140` `_compute_routing`
- **适用负载**　动态routing及实际hash调用
- **成立条件**　MM与选中权重保留；归档未传固定表；主机晚提交另记
- **证据**　[router-score-delivery](../baselines/router-score-delivery.json)

</details>

## history46-R12

**M06 · 路由专家单核激活承接**

- **介绍**　TopK/hidden → DispatchFFNCombine → routed out；FP16 gate/up → SwiGLU+Quant → INT8激活+scale → GMM2。FP16写读+最多3次INT8读。
- **到了哪里**　已停止。已完成初筛。不立项。按原范围停止。其余核仍读GM，不能删整行写回；跨核门槛另归F09
- **结果**　TTFT11.13ms(0.75%)，普通整段≤0.148%

<details><summary>代码与依据</summary>

- **代码定位**　`dispatch_ffn_combine_kernel.hpp:793-823`、`dispatch_ffn_combine_kernel.hpp:912-977`
- **适用负载**　当前GMM2列块分配
- **成立条件**　其余核仍读GM，不能删整行写回；跨核门槛另归F09
- **证据**　[ffn-merge-premix](../ffn-merge-premix.md)

</details>

## history46-R13

**M06 · 共享专家两处交接联合**

- **介绍**　Gate/Up INT32 → DequantSwigluQuant → INT8+scale → Down BF16。INT32 gate/up及INT8+scale写读。
- **到了哪里**　已停止。已完成初筛。不立项。按原范围停止。保留激活顺序和点积；W8A8已有DequantSwigluQuant，无独立clamp/cat可再省
- **结果**　有利TTFT8.83ms(0.59%)、整批62.10ms(0.33%)

<details><summary>代码与依据</summary>

- **代码定位**　`shared_experts.py:185` `part1`、`shared_experts.py:192` `apply_activation`、`shared_experts.py:195` `part2`、`shared_experts.py:670` `forward`
- **适用负载**　[M,4608]→[M,2304]→[M,5120]
- **成立条件**　保留激活顺序和点积；W8A8已有DequantSwigluQuant，无独立clamp/cat可再省
- **证据**　[shared-mlp-fusion](../baselines/shared-mlp-fusion.json)

</details>

## history46-R14

**M06、M07 · 共享 Down+加和+HcPost+直接 premix**

- **介绍**　ForeachAddList → BF16 X → HcPost。三处合法中间面。
- **到了哪里**　已停止。服务反例·限定变体。不立项。按原范围停止。先BF16加法舍入；premix读与F04重叠，不相加
- **结果**　三项有利TTFT7.30ms(0.49%)；M3服务慢0.31%/0.19%

<details><summary>代码与依据</summary>

- **代码定位**　`shared_experts.py:195` `part2`、`fused_moe.py:485` `_forward_impl`、`model.py:871` `hc_post`
- **适用负载**　实际q=3/416/419/816/817/818
- **成立条件**　先BF16加法舍入；premix读与F04重叠，不相加
- **证据**　[ffn-merge-premix](../ffn-merge-premix.md)

</details>

## history46-R15

**M08、M10 · 三检查点打包归集与草稿取行拼接**

- **介绍**　四路残差 → 三处mean/TP gather、末层raw buffer；aux Index/Concat → main_proj/TP gather → RMS。两次TP8启动及合法前缀中间面。
- **到了哪里**　已停止。已完成初筛。不立项。按原范围停止。所有发送字节与原均值源时点保留
- **结果**　整批8.02–31.27ms；极有利全额最多0.39%

<details><summary>代码与依据</summary>

- **代码定位**　`model.py:1505` `forward`、`model.py:1798` `get_mtp_target_hidden_states`、`dspark.py:281` `combine_hidden_states`
- **适用负载**　三处aux、同batch代次
- **成立条件**　所有发送字节与原均值源时点保留
- **证据**　[dspark-checkpoint-packet](../baselines/dspark-checkpoint-packet.json)

</details>

## history46-R16

**M08 · 末层 Post 到词表 MM 整段**

- **介绍**　末层Post → collapse(Cast/Mul/Sum/Cast) → RMS → 词表选行。collapse/RMS/选行中间面。
- **到了哪里**　已停止。已完成初筛。不立项。按原范围停止。同段子融合一起排除；仅裁行不算融合，24槽最长分片不变不省通信
- **结果**　整段全部免费TTFT14.41ms(0.966%)、整批78.69ms(0.423%)

<details><summary>代码与依据</summary>

- **代码定位**　`model.py:855` `hc_collapse`、`model.py:1778` `compute_logits`、`logits_processor.py:136` `_apply_head`
- **适用负载**　raw/aux/完整行分支
- **成立条件**　同段子融合一起排除；仅裁行不算融合，24槽最长分片不变不省通信
- **证据**　[terminal-head-delivery](../baselines/terminal-head-delivery.json)

</details>

## history46-R17

**M08、M09 · 目标词表选词与分片摘要**

- **介绍**　TP8 vocab projection → gather/布局/ArgMax。布局扫描、ArgMax，或稠密gather。
- **到了哪里**　已停止。已完成初筛。不立项。按原范围停止。完整raw/logprob读者不能删；tie/NaN/私有处理保持
- **结果**　一次摘要交换整批27.48–47.78ms；完整logits额外ID半扫描4.23ms

<details><summary>代码与依据</summary>

- **代码定位**　`logits_processor.py:122` `_gather_logits`、`logits_processor.py:201` `get_top_tokens`、`sampler.py:83` `greedy_sample`
- **适用负载**　仅greedy ID与完整分数分别处理
- **成立条件**　完整raw/logprob读者不能删；tie/NaN/私有处理保持
- **证据**　[vocab-selection-fusion](../baselines/vocab-selection-fusion.json)

</details>

## history46-R18

**M09、M11 · 目标验证直接生成 seed/count**

- **介绍**　target/bonus ID → 连续接受比较/seed/count。重复检查及count D2H等待。
- **到了哪里**　已停止。已完成初筛。不立项。按原范围停止。seed原位置、完整输出ID及host镜像保留
- **结果**　即使计数回传免费整批10.14ms；普通≤0.065%

<details><summary>代码与依据</summary>

- **代码定位**　`rejection_sampler.py:146` `forward`、`llm_base_proposer.py:2192` `prepare_next_token_ids_padded`、`model_runner_v1.py:1876` `_copy_valid_sampled_token_count`
- **适用负载**　greedy接受与discard
- **成立条件**　seed原位置、完整输出ID及host镜像保留
- **证据**　[draft-submit-waits](../baselines/draft-submit-waits.json)

</details>

## history46-R19

**M09、M10、M11 · 验证+草稿ID+下一轮状态联合**

- **介绍**　target/bonus ID → 连续接受比较/seed/count；已验证ID/计数 → CPU发布 → 草稿/下轮输入；验证计数 → effective length/slot/input pack → 下一迭代。窄ID、有效性、位置及具名等待。
- **到了哪里**　已停止。已完成初筛。不立项。按原范围停止。时间轴去重；不同主机线程空档和换批不能照抄
- **结果**　设备+等待模型63.86ms(0.344%)；满批主机全给足约0.992%

<details><summary>代码与依据</summary>

- **代码定位**　`rejection_sampler.py:146` `forward`、`llm_base_proposer.py:2192` `prepare_next_token_ids_padded`、`model_runner_v1.py:1876` `_copy_valid_sampled_token_count`、`model_runner_v1.py:2163` `_copy_draft_token_ids_to_cpu`
- **适用负载**　当前稳定batch；换批另U10
- **成立条件**　时间轴去重；不同主机线程空档和换批不能照抄
- **证据**　[draft-state-handoffs](../draft-state-handoffs.md)

</details>

## history46-R20

**M09 · top-k/top-p 双阈值融合**

- **介绍**　温度/惩罚 → 概率排序/双阈值 → 随机抽样/拒绝恢复。相同滤前概率、合并mask写回。
- **到了哪里**　已停止。已完成初筛。不立项。按原范围停止。top-p依赖原概率累积；RNG及raw分数不能用argmax等价替代
- **结果**　B4每次18.72us；随机外推整批0.046%/0.134%，普通<0.06%

<details><summary>代码与依据</summary>

- **代码定位**　`rejection_sampler.py:321` `apply_sampling_constraints`、`rejection_sampler.py:378` `rejection_sample`、`sampler.py:92` `forward_native`
- **适用负载**　默认Ascend随机分支，冻结greedy无命中
- **成立条件**　top-p依赖原概率累积；RNG及raw分数不能用argmax等价替代
- **证据**　[random-sampling-fusion](../baselines/random-sampling-fusion.json)

</details>

## history46-R21

**M09 · 有限 token logprobs 与私有mask**

- **介绍**　bonus/target私有分数 → logsoftmax/TopK/rank。扫描归一化、TopK/rank及拼行。
- **到了哪里**　已停止。已完成初筛。不立项。按原范围停止。prompt长行另U09；K=-1或raw读者继续完整发布
- **结果**　整批7.99–15.98ms；最有利附带mask24.53ms(0.132%)

<details><summary>代码与依据</summary>

- **代码定位**　`logits_processor.py:98` `forward`、`model_runner_v1.py:2973` `_sample`
- **适用负载**　冻结几何假设全启用
- **成立条件**　prompt长行另U09；K=-1或raw读者继续完整发布
- **证据**　[score-readers](../score-readers.md)

</details>

## history46-R22

**M00、M10 · 输入bank与DSpark主投影转置**

- **介绍**　aux Index/Concat → main_proj/TP gather → RMS。一处输入copy、rank-major转置。
- **到了哪里**　已停止。已完成初筛。不立项。按原范围停止。RMS仍归约全5120并读gamma；图copy命中未知用有利代理
- **结果**　合计有利TTFT0.623ms、整批8.05ms(0.043%)

<details><summary>代码与依据</summary>

- **代码定位**　`model_runner_v1.py:1281` `_prepare_inputs`、`dspark.py:281` `combine_hidden_states`
- **适用负载**　冻结7 eager+446图TP8调用
- **成立条件**　RMS仍归约全5120并读gamma；图copy命中未知用有利代理
- **证据**　[optional-paths-and-raw-head](../optional-paths-and-raw-head.md)

</details>

## history46-R23

**M10 · 三层context RMS/RoPE/cache与提交**

- **介绍**　每层WKV → RMS → RoPE → scatter cache；draft embedding → mHC/attention/MoE三层 → collapse/norm。拼接、位置查表、地址与中间写读。
- **到了哪里**　已停止。已完成初筛。不立项。按原范围停止。三层WKV及各自norm不共用；计数等待与submit依赖保留
- **结果**　半设备预算27.94–30.82ms；含主机整批0.21%–0.24%

<details><summary>代码与依据</summary>

- **代码定位**　`dspark.py:281` `combine_hidden_states`、`dspark.py:284` `_project_shared_kv`、`dspark.py:295` `precompute_and_store_context_kv`、`dspark_proposer.py:226` `set_inputs_first_pass`
- **适用负载**　453次TP8调用
- **成立条件**　三层WKV及各自norm不共用；计数等待与submit依赖保留
- **证据**　[draft-context-fusion](../baselines/draft-context-fusion.json)

</details>

## history46-R24

**M10 · 五步 Markov 校正到 ID 及W1/W2复用**

- **介绍**　W1查表 → W2 bias → Add/round → ArgMax/ID → 下一步。Add/扫描/ID写回；命中时替代查表投影。
- **到了哪里**　已停止。已完成初筛。不立项。按原范围停止。五步串行自回归、BF16舍入和原词表映射保留
- **结果**　具名任务含末轮全免整批67.65ms(0.364%)；普通≤0.170%；W1/W2免费0.213%

<details><summary>代码与依据</summary>

- **代码定位**　`dspark.py:311` `hc_head`、`dspark.py:320` `markov_embed`、`dspark.py:323` `markov_bias`、`dspark.py:326` `compute_logits`、`llm_base_proposer.py:1400` `_sample_draft_from_logits`
- **适用负载**　五个前驱ID序列
- **成立条件**　五步串行自回归、BF16舍入和原词表映射保留
- **证据**　[draft-state-handoffs](../draft-state-handoffs.md)

</details>

## history46-R25

**M10 · DSpark 可选共享 TopK hook**

- **介绍**　llm_base_proposer hasattr → compact_topk_indices。本路径无对应调用。
- **到了哪里**　已停止。已完成初筛。不立项。按原范围停止。hasattr为假；其它MTP模型的功能不能算到DSpark
- **结果**　发生次数0，可省工作0

<details><summary>代码与依据</summary>

- **代码定位**　`dspark.py:414` `compute_draft_logits`、`llm_base_proposer.py:1400` `_sample_draft_from_logits`
- **适用负载**　固定DeepseekV41DSparkModel
- **成立条件**　hasattr为假；其它MTP模型的功能不能算到DSpark
- **证据**　[optional-paths-and-raw-head](../optional-paths-and-raw-head.md)

</details>

## history46-U01

**M00 · 视觉条件入口**

- **介绍**　视觉编码 → 空间重组/投影 → 语言输入。patch/Transformer/空间投影与语言输入布局。
- **到了哪里**　已停止。已完成初筛。范围外。冻结文本请求无视觉输入。
- **结果**　当前收益0；图像负载未知

<details><summary>代码与依据</summary>

- **代码定位**　`vl_model.py:190` `embed_input_ids`、`vl_model.py:261` `forward`
- **适用负载**　当前14文本请求无命中
- **成立条件**　仅图像请求命中且N·净省关键时间≥0.01·T时重估
- **证据**　[M00](../modules/M00.md)

</details>

## history46-U02

**M01 · Engram 自然跨层/跨步缓存**

- **介绍**　lookup head → SP行所有者/WKV。相同lookup/常量投影。
- **到了哪里**　已停止。已完成初筛。暂不立项。按C06，完整内容命中且收益覆盖协调及容量才重开。
- **结果**　原本地周期1%需消除约75.4%可缓存部分（常量免费假设）

<details><summary>代码与依据</summary>

- **代码定位**　`hash_state.py:79` `forward`、`embedding.py:333` `embed_gathered`、`embedding.py:378` `lookup`、`layer.py:34` `forward`
- **适用负载**　相同token、位置、历史、层权重版本
- **成立条件**　自然命中率×每次实际可省量扣缓存维护、通信后过线；不同层WKV不直接共用
- **证据**　[M01](../modules/M01.md)

</details>

## history46-U03

**M02、M05 · Gram 对角化、范数归约与精度约束**

- **介绍**　XW与diag(XXᵀ)、Sinkhorn → pre/post/comb/Y；分块max/sum/PV → 最终attention结果；BF16残差 → FP32/HF32投影工作区 → Cube。不需要的Gram非对角、行归约合并。
- **到了哪里**　已停止。已完成初筛。暂不立项。有位差先例；尚缺满足精度的具体替代实现。
- **结果**　M3例Gram 5242880 MAC vs 61440平方；mix投影7864320 MAC仍保留；无通过精度的性能值

<details><summary>代码与依据</summary>

- **代码定位**　`model.py:858` `hc_pre`、`model.py:871` `hc_post`、`model.py:879` `forward`、`hc_pre_cube_compute.h:398-477`
- **适用负载**　所有80处mHC，历史严格四输出合同
- **成立条件**　旧改版FP32 post/comb有位差，停该实现；保留HF32/K归约或先有明确模型误差预算，不能放宽后宣称等价
- **证据**　[M02](../modules/M02.md)

</details>

## history46-U04

**M02、M05 · Sinkhorn 真正定点提前停止**

- **介绍**　XW与diag(XXᵀ)、Sinkhorn → pre/post/comb/Y；归一化统计及Sinkhorn → 系数/Y。已逐位不变后的剩余迭代。
- **到了哪里**　已停止。已完成初筛。暂不立项。提前退出需满足真实定点及判定成本，尚无过线的实现证据。
- **结果**　两处需兑现中位差额77.6%；单侧最有利TTFT≤9.96ms

<details><summary>代码与依据</summary>

- **代码定位**　`hc_pre_m_k_split_core.h:330-435`、`model.py:858` `hc_pre`
- **适用负载**　Attention+FFN两处合计
- **成立条件**　完整FP32 tile在一整轮后逐位稳定才可精确停；稳定深度与检测/同步成本未知
- **证据**　[mhc-iteration-cost](../baselines/mhc-iteration-cost.json)

</details>

## history46-U05

**M03 · 同代 KV 面板驻留**

- **介绍**　KV源页及共享TopK → 后续层。重复gather面板，不减少模型KV逻辑状态。
- **到了哪里**　已停止。已完成初筛。不独立立项。C09的一种替代实现，共用KV交付预算。
- **结果**　连续copy参考TTFT4.02–5.77ms；有效稀疏速率<约465GB/s可能改变优先级

<details><summary>代码与依据</summary>

- **代码定位**　`dsa_v41.py:613` `_forward_attention`、`sparse_flash_mla_csa_block_vector.h:606-653`、`sparse_flash_mla_csa_block_cube.h:416-436`、`sparse_flash_mla_csa_block_cube.h:704-726`
- **适用负载**　同源后续层；足够容量与自然命中
- **成立条件**　192MiB L2报告不证明可用容量/共享；命中×省流量扣占用竞争需过线
- **证据**　[sparse-kv-retention](../baselines/sparse-kv-retention.json)

</details>

## history46-U06

**M06 · 其它合法 MoE 后端/selector**

- **介绍**　prepare → dispatch → apply_moe_mlp → combine/finalize。仍真实存在的prepare/dispatch/MLP/combine中间面。
- **到了哪里**　已停止。已完成初筛。范围外。备选后端没有冻结负载命中证据。
- **结果**　该固定profile无匹配后端预算，Nbranch及净开销未知

<details><summary>代码与依据</summary>

- **代码定位**　`moe_comm_method.py:563` `fused_experts`、`routed_experts.py:693` `forward_impl`
- **适用负载**　AllGather/AllToAll/分离MC2/CANN MegaMoe条件
- **成立条件**　仅该selector实际可达且Nbranch·Δcritical>0.01T才深查；保持mask/capacity/HCCL状态
- **证据**　[M06](../modules/M06.md)

</details>

## history46-U07

**M06 · 固定布局的专家返回有序直写**

- **介绍**　GMM2 → return/reducer → 本地routed output。return暂存或重复归约/同步。
- **到了哪里**　已停止。已完成初筛。暂不立项。必要的远端服务与有序归并仍保留，可删的净关键时间未证实。
- **结果**　不把专家等待或冷热放置当成可删；新增净省量未证实

<details><summary>代码与依据</summary>

- **代码定位**　`dispatch_ffn_combine_kernel.hpp:912-977`、`prepare_finalize.py:316` `finalize`
- **适用负载**　保持专家选择、权重及EP16所有权
- **成立条件**　保持原加权累加顺序；直写多读者与每轮返回代次明确，净减≥1%才重开
- **证据**　[M06](../modules/M06.md)

</details>

## history46-U08

**全模型 · 全层/跨层 megakernel 额外空间**

- **介绍**　Post → mHC → attention → FFN → 下层；draft embedding → mHC/attention/MoE三层 → collapse/norm。仅能指认的新读写/launch/同步。
- **到了哪里**　已停止。已完成初筛。不独立立项。需指明已有局部候选以外可新增消除的工作。
- **结果**　局部已归F04/F07/F08/F09；没有额外通用加速百分比

<details><summary>代码与依据</summary>

- **代码定位**　`model.py:1505` `forward`、`model_runner_v1.py:2229` `execute_model`、`scheduler.py:1874` `update_from_output`
- **适用负载**　全40目标+3草稿层
- **成立条件**　非线性、cache与TP/EP barrier不能凭常驻消失；若声称额外空间须量出Δsubmit/Δtraffic并扣资源退化
- **证据**　[M07](../modules/M07.md)

</details>

## history46-U09

**M08、M09 · 有限 prompt 评分共同扫描/TP摘要**

- **介绍**　vocab FP32扫描 → LSE/TopK/指定词rank；bonus/target私有分数 → logsoftmax/TopK/rank。LSE、TopK、指定词rank扫描及稠密分数交付。
- **到了哪里**　已停止。已完成初筛。范围外。原负载未命中有限提示评分。
- **结果**　本地0.79–2.20ms；含跨卡2.42–4.82ms；固定请求无命中

<details><summary>代码与依据</summary>

- **代码定位**　`logits_processor.py:122` `_gather_logits`、`logits_processor.py:180` `_get_logits`、`model.py:1778` `compute_logits`
- **适用负载**　prompt logprobs开启；例1464行K5
- **成立条件**　三次小通信可能抵消；Nscore与K满足Nscore·每行净省−通信>0.01T；全量K保持全输出
- **证据**　[target-readout-and-prompt-scores](../target-readout-and-prompt-scores.md)

</details>

## history46-U10

**M09、M10、M11 · 随机/概率草稿/换批/动态长度的状态交付**

- **介绍**　温度/惩罚 → 概率排序/双阈值 → 随机抽样/拒绝恢复；已验证ID/计数 → CPU发布 → 草稿/下轮输入；draft probs或ID → target映射/动态长度 → verifier。该分支重复扫描、状态copy与实际submit等待。
- **到了哪里**　已停止。已完成初筛。暂不立项。不同选择和换批条件混合，缺匹配负载的独立收益。
- **结果**　当前无该分支端点估值；不外推greedy/caller大间隙

<details><summary>代码与依据</summary>

- **代码定位**　`rejection_sampler.py:146` `forward`、`rejection_sampler.py:378` `rejection_sample`、`llm_base_proposer.py:2389` `prepare_inputs_padded`
- **适用负载**　未测合法配置；保持RNG、vocab映射与代次
- **成立条件**　Nbranch·每次净暴露量超过目标预算才取证；概率映射不兼容时源码回退greedy；分数与confidence读者保留
- **证据**　[draft-state-handoffs](../draft-state-handoffs.md)

</details>

## history46-H01

**M00、M11 · CP 局部长度与 CPU 计划公共项**

- **介绍**　批计划/局部长度 → 26组metadata builder。重复长度、后缀处理与CPU计划。
- **到了哪里**　已停止。已完成初筛。不独立立项。已有调度与CPU计划基线。
- **结果**　CP单改整批6.10%/4.89%；CPU复用10.36%/9.56%

<details><summary>代码与依据</summary>

- **代码定位**　`dsa_v41_cp.py:87` `_build_global_metadata`、`dsa_v41_cp.py:112` `build`、`model_runner_v1.py:3449` `_build_attention_metadata`
- **适用负载**　原历史CP及CPU计划各自基线
- **成立条件**　属于公共工作/主机共享，不能全写成新增设备融合收益
- **证据**　[module-fusion-estimates](../baselines/module-fusion-estimates.json)

</details>

## history46-H02

**M06 · 前缀换 DP、冷热专家与权重放置**

- **介绍**　现有KV前缀 → 新请求任务划分。工作量或负载均衡改变。
- **到了哪里**　已停止。已完成初筛。范围外。部署放置调整。
- **结果**　C8额外命中上界0 token；专家布局收益不纳入融合

<details><summary>代码与依据</summary>

- **代码定位**　`scheduler.py:561` `schedule`、`routed_experts.py:693` `forward_impl`
- **适用负载**　固定C8当前前缀与专家分布
- **成立条件**　固定专家与权重放置，只回收能指认的物化或重复计算
- **证据**　[c8-prefix-ceiling](../baselines/c8-prefix-ceiling.json) · [fusion-attribution](../fusion-attribution.md)

</details>

## 冻结源码索引

在相同快照按以下路径定位，可用 SHA-256 核对版本。表内未公开源码的文件，使用工作台环境中已有的对应快照。

| 文件路径 | SHA-256 |
|---|---|
| `vllm_ascend/worker/block_table.py` | `215305c699a9df9c118f47c6f7b71d8457ef54e98b5b8571536fd373de7b21ed` |
| `vllm_ascend/models/deepseek_v41/compressor.py` | `31516e5ba234e34fa01e73e864fbc88bd6473d4a5274a1d43da8a79bab3cf494` |
| `vllm_ascend/worker/device_metadata.py` | `efe13f0e801b70d7fba7498ac44062a8eccfd78e5b6303cf658d33dd22217580` |
| `vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/dispatch_ffn_combine/dispatch_ffn_combine.h` | `d2b5712ad97f7505233f340e18660627424e5e1fe4c33be68f8e34b51a479ab1` |
| `vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/dispatch_ffn_combine/dispatch_ffn_combine_kernel.hpp` | `7dcef4c08c4c47e38bdd01aef0dee997798658ee4d7b871c8e6c70f7ffe596ba` |
| `vllm_ascend/attention/context_parallel/dsa_cp.py` | `7e8dd93eeda988866138342523ecfa7cc9c42ce67669d4b1507ee8c5f3452305` |
| `vllm_ascend/attention/dsa_v1.py` | `698765a3662332ad108fd9ef128eb87fd313544cd887f5154f924a5915ee123a` |
| `vllm_ascend/attention/dsa_v41.py` | `3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd` |
| `vllm_ascend/attention/context_parallel/dsa_v41_cp.py` | `039ec9d18e0a8a4aa39f9d5b4bea15204cea3f397a3fce97a5097b266bc9822e` |
| `vllm_ascend/models/deepseek_v41/dspark.py` | `fcee256905928917154d8dac94d6611bb83d02da441e8aa154a525ef51e6a601` |
| `vllm_ascend/spec_decode/dspark_proposer.py` | `70d216de21875e679d510153ebdb0f0ccf130875a0e644f0673d60b67b6b633c` |
| `vllm_ascend/models/deepseek_v41/engram/embedding.py` | `a7741ed16522be4fe607e062e1c988c4e2435af7d258fe7db99f672d4508c4c9` |
| `vllm_ascend/ops/fused_moe/fused_moe.py` | `2f42ce56a7636bf618204dbc2c1aeb2a2c95fed69fe9e5bcfac995354a00beb4` |
| `vllm_ascend/ops/fused_moe/router/fused_topk_router.py` | `06c47fd91e8c36f6f5465cea0b32a184280f842833c2bff27a18fc6d17fcb838` |
| `vllm/v1/worker/gpu_model_runner.py` | `87c29d08c0bbf66993d8b984811e7325e35e6436242a773feb55ec88f16b2c51` |
| `vllm_ascend/models/deepseek_v41/engram/hash_state.py` | `62dd8271db66f4a735de7e9f764b3a1a4b7e66b56c352352f27093f3ef40eb38` |
| `vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/hc_pre/hc_pre_cube_compute.h` | `9056b512f88569025bd3cec598a66d5461b5fd4c14bf6a7eb6c963ff6537ed69` |
| `vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/hc_pre/hc_pre_m_k_split_core.h` | `4d697b15e541216722369284cbd8c9bd5f4154cd0d2251cdb06a6cd46f42fc14` |
| `vllm_ascend/models/deepseek_v41/indexer.py` | `d5d6a438bbced4e63d729dac94c08869df30dab4a71ceacecdea211386b3db33` |
| `vllm_ascend/models/deepseek_v41/engram/layer.py` | `dc9457c88aa3c90133115bc608447e037b0693f8db71bb8da37f1ead44021f80` |
| `vllm_ascend/spec_decode/llm_base_proposer.py` | `458fb50e6839f3a353ddc7181ace4fdae02e35093d8ccd1c4327dc24ae5fff31` |
| `vllm/model_executor/layers/logits_processor.py` | `6b0603d67b0c756253c2fdc882a3896d2e873a16e9aa2ef877aabca8d36bdb5f` |
| `vllm_ascend/models/deepseek_v41/model.py` | `101f96b38e9fc47e7be147ddfe56976aee896867c73a33a7d95c3e3c833a6e91` |
| `vllm_ascend/worker/model_runner_v1.py` | `568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c` |
| `vllm_ascend/ops/fused_moe/moe_comm_method.py` | `7040f691355bd9acb3c668e698de3f7d3854edbc22e3f979f9b03d99800364d7` |
| `vllm_ascend/ops/fused_moe/prepare_finalize.py` | `dff4fac292200fc06f60aef48505f1ee38314ea5adb53560b325901db308eb24` |
| `vllm_ascend/sample/rejection_sampler.py` | `bb1d39de851540c16c2dd2b16df60c60a1fc4957cedb88bdce9417efc0d55eda` |
| `vllm_ascend/ops/fused_moe/routed_experts.py` | `f9db74836ae210a541432b923118696c7a15be9a10d5776834e1d3b64ce11d77` |
| `vllm_ascend/sample/sampler.py` | `867f1770614bb594baf5ecfb858aa4a20dc413cbfbe4f94e06b3786900b581ee` |
| `vllm/v1/core/sched/scheduler.py` | `c1db45f3bbad3a875dd8638331b8870c4380ac1ab4734265ba5c863833a82884` |
| `vllm_ascend/ops/fused_moe/shared_experts.py` | `471db26e9e9e4ba70c2ccd43502ad39bff6699bbd4ca191319a9b66fc20458f9` |
| `vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/sparse_flash_mla/arch22/sparse_flash_mla_csa_block_cube.h` | `55210741e16d8cc455215849c73e2c06db10f23efeb73ca87b4a16847339ccc3` |
| `vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/sparse_flash_mla/arch22/sparse_flash_mla_csa_block_vector.h` | `685190e19519b9a44da7fa95832c06e4b41849577488f4efd0d2df80e1158757` |
| `vllm_ascend/models/deepseek_v41/vl_model.py` | `bb98a2e2a0e24c18aafe17229eb593937aaed2e37c1fd7cc2003c45ce872d282` |
