# 长输入prefill的临时张量峰值

G501完成四个真实原生PD检查：算术答案4以及127/129/131输入各32输出；无retraction，实际DSPARK verify计数为正，131请求报告128个cached tokens。这些是选定短例，尚不等于完整长输入或缓存一致性验收。D八rank完整target/draft图及八组PD内存注册通过并开放API。

随后8191输入在P的原生mHC `_mix_stats_impl` 的FP32 square申请632MiB时OOM，日志可用228–286MiB。P清理并释放八卡。G523只把P静态预算0.92改成0.88，保留chunk8192、EP8、DSPARK5、BF16KV、权重、前缀和已有数学helper，D501保留0.80及完整图。P池由3282816降为2353024 tokens，初始化日志可用量5.48→7.96GB；这是原生日志单位，不与OOM的GiB/MiB混算。

新P再通过算术，8191随后在适配的Engram `rotated_gate` 最终FP32输出求和申请642MiB时OOM，可用176–235MiB，55.90GiB active/56.71GiB reserved。预算调整越过了此前mHC位置，但没有完成长输入可用性，不能称性能改善。两轮失败和原始HTTP500均保留，P八卡释放。D后续日志显示旧请求deferred KV释放超时后释放；还须新P/D状态与功能验收。

源码显示门控计算后，key及original两份大FP32中间量已不再被读取，却仍保持局部引用到函数返回。G533候选只在最终输出表达式前删除两个引用，算式和张量形状不变。实际八rank64例零容差数值比较和分配门槛通过：8191/8193行的pre-output活跃分配各降低约1280MiB，PyTorch峰值分配各降低约640.4MiB。这只证明该独立算子的张量存活机制，不是总HBM、更大batch或端到端收益。

G533校验器的torch.equal只比较数值，不能区分正负零，原始记录的bitwise_equal字段表述过宽；原始文件保留，由显式更正覆盖其范围。G537随后在同一case/seed实际完成八rank共64例的BF16 int16位模式检查，包含符号零、mask、FP32 KV、非连续布局和8191/8193行；所有位模式与输入不变门槛通过。长形状pre-output分配降低1342177280/1342342144 bytes，峰值降低671516672/671599104 bytes，八rank一致；原始框架PyTorch分配计数不等于总HBM。组件结束已核验八卡清空。

G542 P-only完整模型于07:32:55UTC实际启动，保持G523的0.88/chunk8192/EP8/DSPARK5/缓存/权重，D501 0.80保留。源码AST核对仅增加已验证的del及每worker一次、无张量操作的大形状路径记录；完整模型长输入、缓存与功能验收尚未完成，没有恢复或端到端性能通过结论。

- [两次峰值的父级判断与范围](../evidence/prefill_memory_assessment_G535.json)
- [P501原始OOM](../evidence/full_model_G501/2203/server.log)
- [四个短例及8191失败](../evidence/functional_G506/functional/state.json)
- [P523原始Engram OOM](../evidence/full_model_G523/2203/server.log)
- [P523退出与八卡释放](../evidence/full_model_G523/2203/state.json)
- [保留D的图/注册/API实际证据](../evidence/full_model_G501/2204/actual_graphs_registration_API.json)
- [G533数值与内存原始rank0](../evidence/engram_lifetime_G533_numeric_only/evidence/rank0.json)
- [比较器范围更正](../evidence/engram_lifetime_G533_numeric_only/comparator_scope_correction.json)
- [G537严格位模式探针](../code/engram_lifetime_G537_strict_bit_gate/probe.py)

- [G537严格64例及释放的父级复核](../evidence/engram_lifetime_G537_strict_bits/parent_acceptance.json)
- [G537实际rank0](../evidence/engram_lifetime_G537_strict_bits/evidence/rank0.json)
- [G542实际P-only启动](../evidence/full_model_G542/2203/actual_start_parent_receipt.json)
- [G542部署源码与AST前置检查](../code/full_model_candidates/prepare_prefill_lifetime_g542.py)


G542后续实际结果：八rank均在8192行（8191请求被执行路径padding）记录已进入新的Engram删除引用后路径，实际adapter SHA与候选完全相符；算术再次通过。但8191请求仍HTTP500，07:44:21–22UTC在原生DeepseekV41Indexer.scores的relu/加权头归约临时分配1.00GiB时OOM，可用约791.92–852.02MiB，55.18GiB active/56.10GiB reserved。不是原Engram位置复发，也不能称完整长输入恢复。P于07:45:51UTC已核验八卡清空，D501保留。

从精确停止的P542容器取回原生scoring/full_topk/dsv41_sparse源码，SHA留证。原生_score_chunks将单个BF16[rows,heads,lc]上限设为1GiB，源码明确临时量约为其3倍；改变该原生常量可仅调整独立query行分块，评分、因果mask、topk算式保持。G563候选准备128MiB上限，实际模型32个索引头、head_dim128，仍须实际NPU逐位score/原生unsorted-topk/峰值门槛通过后再考虑完整模型；目前没有新模型或端到端收益。

- [P542完整原始OOM与释放](../evidence/full_model_G542/2203/state.json)
- [父级新OOM与八rank实际路径判断](../evidence/full_model_G542/2203/parent_indexer_OOM_assessment.json)
- [原生score预算与3倍临时量注释](../code/fixed_native_score_sources_G559/dsv4/v41_indexer/scoring.py)
- [G563原生评分分块探针](../code/native_score_budget_G563/probe.py)


G563严格探针已执行并失败留证：八rank均完成前六组形状的score逐位、原生unsorted-topk和输入不变检查；8191/lc4095组实际工作峰值3322135552→415173120 bytes。但第七组8193/lc4097、FP32头权重在八rank均未通过FP32 score整数位模式比较，探针停止，最后一组没有执行。不能把该峰值下降称作候选通过，也不能据此部署。08:00:40UTC八卡已清空。后续G570仅准备诊断score误差量、原生topk索引/集合是否改变，没有放宽原严格验收或完整模型部署。

- [G563严格门槛失败的实际rank0](../evidence/native_score_budget_G563_failed/evidence/rank0.json)
- [G563组件退出与八卡释放](../evidence/native_score_budget_G563_failed/state.json)


G570后续诊断实际完成八rank64例：56例score逐位相同，8例FP32权重的8193/lc4097组仍有每rank4–9个score值不同，最大绝对差0.00167083740234375、最大相对差0.0012020663125440478；全部64例原生unsorted-topk索引与集合相同。大形状工作峰值约3.3–4.4GB降为0.41–0.55GB，仅为PyTorch分配计数。组件已结束、八卡清空。原G563严格逐位score门槛仍为失败，不以topk选定例相同覆盖失败或宣称完整模型可部署。实际native queries经过fake_quant_fp4、index-K来自FP4 pack/dequant，而本诊断使用一般BF16输入；下一步核对实际操作数域是否改变数值判断，尚未作新验收。

- [G570完整64例诊断父级复核](../evidence/native_score_difference_G570_diagnostic/parent_diagnostic.json)
- [G570实际rank0](../evidence/native_score_difference_G570_diagnostic/evidence/rank0.json)
- [G570八卡释放](../evidence/native_score_difference_G570_diagnostic/state.json)


后续G581按实际native per32 UE8M0 FP4-rounded query/index-key操作数域进行同样严格检查；源码来自固定镜像，实际导入fake_quant_fp4文件SHA逐项核对，数值门槛仍为FP32 score整数位模式和原生unsorted-topk完全相同，峰值门槛未改。八rank64例全部通过，包含两种头权重类型、8191/8193/lc4095/4097和lc131072。08:15:32UTC八卡清空。这是限定实际生产操作数域的所选tensor fixture通过，不能恢复一般BF16等价结论，原G563失败和G570差异量保持；也不是完整checkpoint激活分布或模型质量证明。

G587完整模型候选只把native_score_chunks原始code对象放入NPU专用globals副本，将预算设128MiB；CPU原函数和原module 1GiB常量保留，native评分/mask/topk算式不改。限定固定FP4 producer源码、模型32头/维128、BF16 q/k与原生indexer类型；host-only一次真实large-chunk路径记录用于后续验收。CLI/env、P0.88/chunk8192/EP8/DSPARK5、之前Engram liveness及D5010.80保持。完整模型运行、8191/8193/32K、缓存重放及有效图执行仍需真实验收，尚无端到端收益。

- [G581实际FP4域64例严格门槛父级复核](../evidence/actual_FP4_score_gate_G581/parent_scoped_acceptance.json)
- [G581实际rank0](../evidence/actual_FP4_score_gate_G581/evidence/rank0.json)
- [G581八卡释放](../evidence/actual_FP4_score_gate_G581/state.json)
- [限定NPU的原code对象分块预算](../code/full_model_candidates/sg_pd_g587/indexer_score_budget_g587.py)


G587已实际运行：算术通过，八rank均记录native128MiB score chunk返回，原module全局1GiB保留、native code对象相同。但8191仍HTTP500，在更后面的Engram gate点积表达式`(original*(q_weight.float()*k_weight.float())*key).sum(-1)`申请642MiB时OOM，可用约155–215MiB、55.28GiB active/56.71GiB reserved。不是此前索引score位置复发；没有完成长输入恢复。08:38:21UTC P八卡清空，D501保留待新peer/状态复核。

G597接下来只准备独立token行分块（256行），维度/H头的归约和原始Quarot计算不分片，完整输出预分配、全部输入仍处理。参考函数与实际G587算术AST相同，去掉host-only路径记录。预注册仍要求BF16 int16输出逐位相同、输入及权重不变、mask原值保持；大形状working peak<768MiB且比参考降低>1GiB，14形状/布局/类型×8rank，未通过不部署。更多kernel launch可能增加时延，不能从显存峰值直接宣称性能收益。

- [G587完整模型失败/八卡释放](../evidence/full_model_G587/2203/state.json)
- [G587父级新点积OOM判断](../evidence/full_model_G587/2203/parent_Engram_dot_OOM_assessment.json)
- [G587真实8rank路径记录样例](../evidence/full_model_G587/2203/native_score_budget_hits/worker135.json)
- [G597行分块探针与预注册边界](../code/engram_rowchunk_G597/mechanism.json)


G597行分块被严格门槛拒绝：rank1在513行非连续布局出现BF16 int16位模式差异，其他rank在各自11例左右被torchrun中断，未完成14例。没有放宽或忽略该case，也未将行分块放入完整模型。08:58:46UTC组件八卡清空。D501于08:58UTC实际controller仍存活且/ready=200，旧P失败请求的deferred KV释放超时后释放记录保留；新peer/cache与长形状仍需后续验收。

G600是不同的实现：保持原行数/形状及归约次序，rstd计算后仅复用由原矩阵乘法新分配的original FP32缓冲区，按原顺序原地乘q/k权重积和key，再sum。key、输入和权重都不原地修改。AST反向重建与原参考相同；重新用相同14case/seed检验输出逐位及输入不变，不删除失败的stridedcase。由临时量模型预注册：BF16 KV大形状峰值应从约4个fullFP32 buffer降至输出表达式约3个，至少降低0.8个buffer且低于2.3GiB；FP32 KV的key原已别名输入、输出可能主导，只要求不回归超过1MiB，不声明该型峰值下降。此处尚无组件通过或模型收益结论。

- [G597实际位差异失败rank1](../evidence/engram_rowchunk_G597_rejected/evidence/rank1.json)
- [G597八卡清空](../evidence/engram_rowchunk_G597_rejected/state.json)
- [D501当前只读就绪复核](../evidence/full_model_G501/2204/actual_retained_readiness_G599.json)
- [G600原尺寸buffer复用及预注册](../code/engram_owned_inplace_G600/mechanism.json)


G600实际完成八rank112例：BF16 int16输出位模式全同、输入与权重不变，包含原G597失败的513非连续布局及FP32 KV。8191/8193 BF16 KV工作峰值分别2684486144→2013825536、2684815872→2014072832 bytes；FP32 KV8193也实测下降，但不扩大其预注册声明。全部门槛通过，09:09:26UTC八卡清空；这仍是选定eager张量和PyTorch分配计数，不是完整质量、总HBM或端到端收益。

G603 P-only完整模型09:13:28UTC启动；原始数值AST与G600候选逐项相同，新增每worker一次host-only KV dtype/stride路径记录。CLI/env与G587相同，保留native128MiB score预算、dead-reference删除、P0.88、EP8、chunk8192、DSPARK5及D501。D501在09:13UTC同一容器controller仍存活且ready200。完整长请求、缓存、功能与性能仍未验证，后续原始执行记录决定状态。

- [G600严格112例与释放的父级验收](../evidence/engram_owned_inplace_G600/parent_acceptance.json)
- [G600实际rank1含原失败布局](../evidence/engram_owned_inplace_G600/evidence/rank1.json)
- [G600八卡清空](../evidence/engram_owned_inplace_G600/state.json)
- [G603源码固定的完整模型适配](../code/full_model_candidates/sg_pd_g603/adapter.py)
- [G603实际P-only启动](../evidence/full_model_G603/2203/actual_start_parent_receipt.json)
- [D501当前身份与只读就绪](../evidence/full_model_G501/2204/actual_retained_readiness_G606.json)


G600实际八rank112例严格BF16 int16输出位模式与输入/权重不变门槛全部通过，原G597失败的513非连续case仍保留且通过。8191/8193 BF16 KV的PyTorch工作峰值分别由2684486144/2684815872降至2013825536/2014072832 bytes；FP32 KV大形状也观察到下降，但预注册只要求不回归。09:09:26UTC八卡清空。这支持所选eager Engram缓冲区复用机制，不代表总HBM、整模型质量或端到端收益。

G603 P-only完整模型于09:13:28UTC实际启动；rotated_gate去除host-only记录后的数值AST与已测G600完全相同。新增一次actual path元数据含KV dtype/stride，保留native score128MiB、deadref release、P0.88、chunk8192、EP8、DSPARK5。D501同一容器/进程在09:13UTC仍实际存活且ready200。G604只提交选定功能、8191/8193/32K及缓存重放诊断，尚无完整模型通过或AISBench结果。

- [G600严格112例父级验收](../evidence/engram_owned_inplace_G600/parent_acceptance.json)
- [G600原始rank1含非连续布局](../evidence/engram_owned_inplace_G600/evidence/rank1.json)
- [G600组件退出/八卡释放](../evidence/engram_owned_inplace_G600/state.json)
- [G603实际启动凭证](../evidence/full_model_G603/2203/actual_start_parent_receipt.json)
- [G603完整模型适配源码](../code/full_model_candidates/sg_pd_g603/adapter.py)
- [G604所选功能/长输入/缓存重放采集器](../code/full_model_candidates/functional_PD_P603_D501_g604.py)


G603/G604父级已检查7个完整原始响应、usage与保存SHA：算术4、8191/8193/127/129/131各32输出和OpenAI短32输出均满足所选门槛。8191越过原Engram dot OOM，八worker实际新buffer复用记录与adapter SHA一致，KV实际BF16及stride已取证；native128MiB score路径也全8命中。此处证明所选完整请求恢复，不是端到端性能收益或完整质量。

32K诊断09:29:07UTC提交，09:34:07触发客户端300秒读超时；无完整响应，后续三个缓存重放未提交。P随后记录clientdisconnect并完成Abort，D也确认AbortReq。不能记为32K通过，也不把客户端超时断言为服务OOM或因此盲目重启。09:37两端连续idle快照确认running/waiting/active/queues及所有PD请求计数为0，router两个workerload0。按用户更新vLLM镜像后，原owned controller收到stop.request，P603在09:42:02UTC、D501在09:41:44UTC核验八卡释放；终态failed_released的Stoprequested是有意框架切换，不是新服务缺陷。原始来源、日志和所有失败保留，无AISBench formal。

- [7例原始响应与SHA父级核验](../evidence/functional_G604/parent_partial_acceptance.json)
- [32K客户端超时/未完成状态](../evidence/functional_G604/functional/state.json)
- [P实际8worker路径/BF16KV/idle快照](../evidence/controlled_switch_G629/2203/actual_idle_and_path.json)
- [D实际idle快照](../evidence/controlled_switch_G629/2204/actual_idle_and_path.json)
- [P有意切换的8卡释放](../evidence/controlled_switch_G629/2203/owned_release.json)
- [D有意切换的8卡释放](../evidence/controlled_switch_G629/2204/owned_release.json)
