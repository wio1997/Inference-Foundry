# glm5-3 HANDOFF

当前正式workload已按用户最后更正改为80K输入/0.6K输出/93%共享前缀、KV cache命中率条件93%（80000/600/0.93）；声明命中率与实测HBM/external计数分账。166实际AISBench prefix driver和两机五文件已同步、原配置备份，SLO默认600；未启动压测，历史冻结Run不改。GitHub main/research五文件独立提交已atomic fast-forward成功，read-only remote refs/实际config核验main `c0159929`、research `e8e30671`，见[同步证据](records/points/GLM-OPT-0002/jobs/GITHUB-WORKLOAD-SYNC-20261007/result.json)。已恢复checkpoint160记录的用户明确公开授权，先前再次等待确认是多余阻塞；本次只同步workload文件，未夹带私有研究祖先/raw。

最新[checkpoint171](records/points/GLM-OPT-0002/research/decode_path_20261006/CHECKPOINT171.md)：Run266 H9完整比较不重复增益，H9off；原controller跨host收尾失败保留，独立terminal reconciliation已completed/phase0，总21请求，H6/H5和纯KV/API修复驻留。当前D266 root419075/start309643580，P249保护；现场按Run266 guards/retained_stack核验。下一唯一候选H10提前greedy padded MTP提交，336实际CPU状态/4096guard通过，Run267已于06:14:38Z启动唯一controller1079159/start309928676，spec202a2a25/51pins、80installed-source gate通过；correctness/性能仍待实际裁决。现场用Run267 state/active_epoch/guards/retained_stack，不按旧D266 PID操作。

checkpoint170终态：[Run264错误归因与Run265 clean replay通过](records/points/GLM-OPT-0002/research/decode_path_20261006/CHECKPOINT170.md)。Run265于05:03:04Z completed/phase0；独立raw归约确认canonical API500、全16失败请求零target/draft调用、两个clean PD exactIDs/EOS/KV、全16动态FULL实际12次完整请求replay。H6/H5全16驻留，pure logical-ID/API修复驻留；Graph配置/观察器仅诊断支持，未宣称matched性能或完整API/SLA。P249保护，D265 root3490723/start309425199，实际按Run265 guards/retained_stack。旧Run264恢复及以下运行中根均是历史。下一源码候选[H9受限SFA stream-ordered replay](records/points/GLM-OPT-0002/research/decode_path_20261006/stream_ordered_replay/REVIEW_PACKAGE.md)两文件patch/768 actual AST CPU组合+6动态revocation通过，尚未安装/NPU比较；最大剩余FULL gap和同步安全按仓库trigger6交独立Challenger复核，H6/H5与研究持续保留。API/SLA/Current未完成不停止研究。

checkpoint169更新：Run263已FAILED并于04:04:41Z完成一次H6/H5 eager恢复，exact2334/8 gold和all16 stack/health/idle通过。错误输出同一请求存在rank5 Mooncake/HCCL传输失败，不能归因于Graph数值错误；[terminal与源码归因](records/points/GLM-OPT-0002/research/decode_path_20261006/CHECKPOINT169.md#run263-terminal-closure-and-decisive-kv-confound)。已提出逻辑KV失败ID最小correctness patch，20 actual AST pipeline用例与7 wrapper/实际决策CPU用例通过。[Run264 Reset](records/points/GLM-OPT-0002/research/decode_path_20261006/RESET_PD_IDENTITY_AND_REPLAY.md)限定三条PD请求验证失败API与成功传输replay；当前准备/冻结，实际执行以Run264 state.json为准。H6/H5始终保留，H4/H8 off，P249保护，Current/API/SLA另验。以下Run263运行中段落是历史，不能继承其PID或诊断结论。

checkpoint169运行中：H6/H5保留，Run262 H8 all16 bytes/gold/工作量通过，但两对完整PD不重复增益，独立INCONCLUSIVE/H8 off。[固定bucket2 Reset](records/points/GLM-OPT-0002/research/decode_path_20261006/RESET_BUCKET2_REPLAY_ON_H6_H5.md)已冻结Run263：唯一controller PID1138367/start309059523，spec d7148e60c7d6cf03b594da99dda0cbeeb6397b18a88ebd71fef0a12bdde3c4f5；一次owned D重载+两个PD请求，H6/H5固定、H8 off，验证capture→真实动态KV/DCP/HCCL replay，观察器无性能宣称。实际现场以Run263 active_epoch/guards/retained_stack为准，不能继承旧D PID。P249保护。CPU观察器6case通过，源消费hash/健康空闲/all16 ownership preflight通过。正式API/SLA/Current仍另验，不停止研究。下方历史运行中状态不能作为当前现场。

最新checkpoint168：[H6+H5研究栈保留与继续归因](records/points/GLM-OPT-0002/research/decode_path_20261006/CHECKPOINT168.md)。Run261已completed/phase0；独立raw reducer核验H5在H6上完整PD D TPOT下降6.77/7.04%，D saving0.314704/0.313984s > drift0.017733s，四完整累计11/11/11/11/0、八短4/4/4/4/0，IDs/EOS/KV/all16 helper/cache/map/same-worker均通过。研究stack [H6,H5]、MC2/event1实际保留，H4 off；最终Current/API/SLA另验。Run262已由唯一controller启动H8局部投影复用比较，正常一次D重载/20请求，H6/H5全程固定。最新现场用Run262 active_epoch/guards/retained_stack，不继承Run261或更旧D PID；P249保护。42实际CPU AST和84 selector用例通过，H8未入stack。下面运行中描述是早先状态，保留历史。

2026-10-07用户要求保留验证性能栈持续研究。H6已实际恢复并驻留：Run259初始化因遗漏空witness目录失败，唯一冻结恢复成功；原失败保留，不重放。Run260无重载、同H6 workers完成H6对H6+H4：正确性/全rank bytes通过，短TPOT改善6.06/9.10%，但完整B1工作量12draft/11accepted不匹配、B2同工作量仍慢0.265656s，冻结INCONCLUSIVE；H4关闭，H6始终mode1。

当前唯一性能比较[Run261 H5 on H6 Reset](records/points/GLM-OPT-0002/research/decode_path_20261006/RESET_H5_ON_H6.md)：一次owned D重载、MC2 mode1固定、event0/1同库/源码/workers，20条请求按correctness→A/B/A/B→自然23token完整PD。controller运行中，D已ready；实际现场以Run261 guard/active_epoch/retained_stack为准，不能继承下面旧D PID。P249保护不变，无新profile/扫描。

离线继续已有trace：MLA 173.377ms暴露分成indexer/remap/wait74.718、A2A32.479、SFA18.696、V-up12.342ms等，非可删预算。[实际AST FULL审计](records/points/GLM-OPT-0002/research/decode_path_20261006/replay_control_flow_CPU.json)证明目标Python forward在重复FULL调用不重入，MLA不是FULL eager段；metadata/PD外层/logits/sampler/GLM eager MTP在wrapper之外，未证实设备capture正确性或删同步。另发现native indexer两次相同wk_weights投影，[最小局部复用patch](records/points/GLM-OPT-0002/research/decode_path_20261006/indexer_projection_reuse.patch)已有42 CPU实际AST cases通过，尚未安装/进入stack；Run249仅110/395 MLA执行该indexer，第二GEMM暴露D15约1.196ms，不是最大Gap。继续研究/验证；Current与最终产品KEEP仍另验。下文checkpoint167等为历史，不授权清空研究stack或停止。

2026-10-07 checkpoint167：[同次 native 因果闭环与层间源码拆分](records/points/GLM-OPT-0002/research/decode_path_20261006/CHECKPOINT167.md)。Run258唯一stock2334/8请求、9120边界/zero lost关闭MC2 availability归因：D15 predicate257.133ms，占nativeCPU77.64%，并非整步；Run256两处缓存patch的correctness→A/B/A/B→自然EOS完整PD收益13.62/14.20% TPOT保留。继续已有Run249拆开post-combine finalize＋shared W8A8、MLA、prepare/route和control；1901通信producer源码联结排除重复TP最终reduce，36CPU TLS用例不支持boxed丢inference mode。不新增大NPU/profile/scan，H6限域POSITIVE/正式PARKED、H4/H5不叠加，CurrentNone/stack空。本阶段没有新增代码级性能 KEEP。现役P249/stockD256，Run258 guards/native before==after，controller和自有probes已退出/清理；下面旧checkpoint保持历史。

2026-10-07 checkpoint166：[H6已完成及继续源码归因](records/points/GLM-OPT-0002/research/decode_path_20261006/CHECKPOINT166.md)。Run256同workers/common native binary A/B/A/B与23token自然EOS完整PD通过；D TPOT下降13.62/14.20%、完整PD wall11.51/14.41%（P drift单列），H6限域POSITIVE/正式PARKED，原库已恢复。Run257唯一source-stack补证执行/golden通过但reader46/37errors，定量INCONCLUSIVE且不重试；继续已有raw/source。新EP归约760/760最晚host==device，后三轮均D15，强化层内eager host晚供给归因。local prepare3888CPU通过但PARKED/source-only；外层boxed入口CPU差约6.5–6.7us，STOP作为主因，无新增NPU Run/scan/graphflag。现役P249/stockD256，最新已核验guards见Run257，两个旧controller均已退出。Current=None/stack空；本阶段没有新增代码级性能 KEEP。下方checkpoint165及旧现场保持历史，不能再继承build-running或旧D PID。

2026-10-07 checkpoint165：[现有raw到MC2最小native缓存候选](records/points/GLM-OPT-0002/research/decode_path_20261006/CHECKPOINT165.md)。一次最小Run254已完成，保存D16的3223 CPU IP/callchains；248查询样本全部定位到两处MC2 V4 predicate，最小patch及native CPU并发/负路径检查通过。H6为唯一候选，隔离CPU build E运行中，尚未安装、重载D或执行模型A/B。实际graph/replay/PD hooks也已核对，未翻graphflag。P249/D253 stock健康空闲、原CPU审计before/after guard一致。继续source/raw归因和correctness→matched A/B→E2E，不因无formalKEEP结束研究。Current=None/active stack空；本阶段没有新增代码级性能 KEEP。下面旧checkpoint保持历史，不重放已完成controller或按旧PID操作。

2026-10-06 checkpoint164：[逐轮critical path与H5实测](records/points/GLM-OPT-0002/research/decode_path_20261006/CHECKPOINT164.md)。Run249五轮最晚供给从D13切到D15；1901次中1896次最晚host==最晚device，主要Gap位于模型内eager host供给。D15 preEnqueue747.341ms中MoE531.027/MLA173.377ms；不能作为profOFF265ms/token可删预算。H4copy/split路径仍正式INCONCLUSIVE/PARKED；后续H5单流MoE event五文件最小patch已走176native+64worker见证+same-resident A/B/A/B+相同23token自然EOS完整PD。D generation TPOT229.346→218.544、226.595→219.210ms/token，下降4.71/3.26%；D wall下降4.83/3.41%，P drift单列。正向限域代码证据成立，但标准动态workload/SLA未验收，正式PERF_KEEP INCONCLUSIVE/PARKED，Current=None、active stack为空。本阶段没有新增代码级性能 KEEP。

最新15:48:54Z[fresh现场](records/points/GLM-OPT-0002/research/decode_path_20261006/final_site.json)：P249 root1916718/start303049910不变；D253 root1117497/start304582938替代D251；两health200/idle、各16owned worker、67相关original源码hash一致，253controller/phase已退出。D selector0及五原磁盘源码恢复，驻留诊断wrapper保持stock语义；H4未叠加。不要按下方旧D PID操作或重放spec。Run252一个CPU-clock补证表明main高CPU但含SHM/runtime spin，不可等同纯frontend。下一最高价值源码问题是MC2分配/descriptor/EXEC_NPU_CMD的剩余frontend成本；workspace query仅19.921ms不能解释391.290msinclusive。继续已有源码/raw分析；不自动新增profile、扫描或大Run，不因没有formalKEEP结束研究。以下checkpoint163及更早均为历史。

2026-10-06 checkpoint163：[现有 Run249 critical path / 最小 patch / 实测裁决](records/points/GLM-OPT-0002/research/decode_path_20261006/CHECKPOINT163.md)。已定位最大局部区域为层内 eager host 供给晚；实际 SFA/DCP/TP/EP/MTP 依赖与 peer/event 等待已拆分，不能把早 rank 的通信/事件 union 当可删 wall。H4 MoE list-gather→direct gather 删除380×16输出拷贝，final512B guard 的240 CPU用例、320 native HCCL字节用例及两次D16真实warm gate通过。同一驻留 D 的 A1/B1/A2/B2 短 TPOT 中位数263.843→246.900、260.413→233.540ms/token；完整相同3token自然EOS PD wall1.693→1.660、1.746→1.566s，首对接近波动且第二对含P快31ms，正式Product Gain INCONCLUSIVE/PARKED，不进Current/stack。Run250 gate失败/未匹配完整74A/8B保留，Run251另行恢复与比较，不改写旧raw。无新大规模Run、profile、参数扫描或kernel。本阶段没有新增代码级性能 KEEP。

当前驻留为P Run249、D Run251；251controller已completed退出。14:04Z独立核验两health200、idle、各16owned NPU worker；D mode0/磁盘原源码54e8ac…已恢复，内存保留safe diagnostic selector。最新[现场](records/points/GLM-OPT-0002/runs/GLM-RUN-0251/final_site.json)记录fresh D root2532689；不要按下方Run249/250 D PID操作或重放完成spec。剩余最大可删除profOFF预算及Scheduler/Executor/ModelRunner分别时长unknown；继续现有源码/raw的producer成本归因，不从边界几ms或CPU inclusive推断整体收益。以下checkpoint162及更早为历史。

2026-10-06 checkpoint162：实际vLLM/Ascend CPU+NPU profile已采集并离线解析全部P16/D16rank非空设备trace。Run249替换Run246并成为当前驻留P1669081/D1679900，布局/模型/算子保持；profiler已stop、两health200且running/waiting/KVusage/preemptions0。2334prompt/8output暖态profOFF单请求TTFT1.3091s、首末token平均265.29ms/token，尚非SLA分位数，Decode远于18/40ms目标；完整API/unified8000未验收。H3 KV晚观测parked，当前唯一源码问题为D逐步通信/peer供给与host提交关系，最大可删除Gap/time仍unknown。无性能patch，Current=None；本阶段没有新增代码级性能 KEEP。见[checkpoint162](records/points/GLM-OPT-0002/research/profiling_installed_20261006/CHECKPOINT162.md)、[Run249实际证据](records/points/GLM-OPT-0002/runs/GLM-RUN-0249/execution_summary.json)。以下checkpoint161及更早状态为历史；不要再把Run246根PID当现役，也不重放已完成spec。



2026-10-06 checkpoint161实际执行完成：用户已确认两机上传完成，gate解除；全部182文件/177474tensor结构再次核验。旧完整副本/public/observer已由Run245按fresh身份退休，历史状态保留。5.3各角色DP1/TP16/PP1/EP16/DCP16、MTP K1、eager、2GiB KV、native Mooncake fail-policy的P166/D167实际ready。Run247 P完成2334prompt+1internal token导出，Run248复用同一export后D接收2334外部KV命中、HTTP200和32生成token IDs，MTP接受10/12draft；结束后两health200、running/waiting/KVusage/preemptions均0。controller248已completed退出，32worker/两native角色保持驻留；8000旧gateway停止，新完整公共入口尚未验收。详见[checkpoint161](records/points/GLM-OPT-0002/research/pd_critical_path_20261006/CHECKPOINT161.md)、[Run248实际结果](records/points/GLM-OPT-0002/runs/GLM-RUN-0248/execution_summary.json)。Run245缓存预算失败、Run246代理阻塞和Run247客户端engineUUID断言失败均保留原raw/源码；后两者没有D请求，不当真实PD或性能失败。不要重放任何已完成spec，后续先核验当前身份并复用ready角色。

唯一H3仍conditional；没有device-safe全贡献者/合法D机会/commit关联，不能凭cache计数或D单段10.72s推导可删除Gap。完整API入口的独立Astra Review保留D-native canonical解析/状态边界，否定简单单hook完整性；fanout取消/P export ownership、prompt-logprobs结果消费者需闭环。未接入任何性能patch或新kernel，完整API/SLA/matched E2E仍未验收，Current=None、active PERF_KEEP空。本阶段没有新增代码级性能 KEEP。下方checkpoint160及更早状态按历史时点保留，不再作为上传未完成/旧服务现役事实。

checkpoint160持久化：代码/证据提交`5468f420`及补充证据`26e1c637`已同步166。API核实仓库public后，用户明确回复“允许”，授权本次研究代码、安装源码片段与现场证据公开上传；已成功push至研究分支，GitHub API核验`26e1c637`。本批材料不再继承旧待授权状态。两机下载cache覆盖182引用但`.mv`早于引用权重末次修改，不能证明当前上传终态；未解除模型加载gate。同步验收与补读来源见[checkpoint160](records/points/GLM-OPT-0002/research/pd_critical_path_20261006/CHECKPOINT160.md)。

2026-10-06 checkpoint160：新对话已开始实质执行，早前“先调整方案/上传期间仅资料”不作为当前仅文档限制。[Reset](records/points/GLM-OPT-0002/research/pd_critical_path_20261006/RESET.md)、[checkpoint](records/points/GLM-OPT-0002/research/pd_critical_path_20261006/CHECKPOINT160.md)、[独立Astra Review](records/points/GLM-OPT-0002/research/pd_critical_path_20261006/ASTRA_REVIEW.md)。两机全部182引用文件/177474tensor的header/index/offset/shape/dtype/精确长度与稳定stat核验通过，header/size及tokenizer/quant metadata一致；索引total_size差额恰等rot.weight额外payload，B警告原样保留/C另归约。上传工具终态、payload校验和及新模型加载/fit/完整PD仍unknown。亲读实际Ascend覆盖类确认rot合法忽略，不能由基类误判不支持；注册/源码不证明功能。现役仍旧GLM-5.2完整副本、32worker、三health200、running/waiting0；不继承PID/owner或旧队列。唯一H3问题为全rank/device-safe KV后的晚观测是否错过合法D机会，最大可删除Gap/time unknown；实现限域host诊断及reducer，Mac和166真实Python的8CPU检查通过，未接live进程。JobA480s超时INVALID/B、C、CPU均真实CLI/inner0并保留raw；这些不计性能成果。Current=None、active PERF_KEEP空，本阶段没有新增代码级性能 KEEP，完整PD E2E Gain=unknown。下一设备动作先补上传终态/可信transfer-complete证明与fresh所有权、5.3真实PD fit/功能，再冻结唯一短诊断的device/合法机会/有效输出关联；不进行猜测通知重写或扫描。

2026-10-06用户再次明确最终目标：**GLM-5.3专用、动态请求、现场可用的极致优化推理框架**。可删通用抽象/兼容层和成套重构控制/调度/PD执行链，保留实际模型依赖、正确算子、完整外部功能及动态请求；不限定为原生小patch或固定benchmark服务。既有SLA数值从验收源码恢复，已明确写入[MISSION](MISSION.md#性能与功能合同)：TTFT P50/P75/P90/P99 <4/8/12/30秒，TPOT P50/P90 <18/40毫秒。AGENTS、PLAN和新会话入口已同步目标澄清；这不构成新架构候选、性能Run或KEEP，Current=None不变。

2026-10-06 配置切换已完成：[GLM53-CONFIG Zcode Job](records/points/GLM-OPT-0002/jobs/GLM53-CONFIG-20261006/summary.md)真实CLI/inner退出0，bridge与独立validate均VALID。166/167共25个未来启动配置/资料已备份并改用5.3目标，Sol只读回验全部修改后hash一致；服务器仓库FF到43def762且原有8个dirty文件与index保留。没有操作权重上传、加载、服务或推理测试；5.3上传完成、fit、功能及完整PD性能仍待核验。

2026-10-06 用户开始上传GLM-5.3权重，明确后续全部研究/配置改用**GLM-5.3 W8A8标准PD**。两机只读确认目标是独立目录`/data/tiankuan/wio/GLM-5.3-w8a8`，新服务模型名`glm-53`；检查时尚缺config.json，不能据目录/分片出现宣布上传完成。本次仅切配置默认值、恢复导航与当前资料，不加载、不重启、不发生成请求或性能测试。当前resident及下方历史Run仍是当时实际模型，不能改标5.3；旧布局、源码机制和功能证据只作待重验线索。5.3 config/index/tokenizer/分片、安装版本支持、fit、正确性、完整PD E2E全部待核验，CurrentNone/active PERF_KEEP空。入口见[ENVIRONMENT_RECOVERY](ENVIRONMENT_RECOVERY.yaml#model)、[新模型资料](docs/research/GLM5_3_MODEL_AND_RUNTIME_ANALYSIS.md)、[PLAN](PLAN.md#当前执行方案标准pd优化)。已有容器/部署目录旧名称仅标识资源，不据此回退旧权重。

2026-10-06 用户要求“先调整方案”：当前阶段仅修订[标准PD执行方案](PLAN.md#当前执行方案标准pd优化)，本轮不推进部署、推理、profile、benchmark或候选实现。独立Astra方案复核已接受。参考为P/D各DP1TP16PP1；先用真实PD证据定位完整关键路径，限域研究基线不冒充完整产品验收。AsyncLLM只是条件接入点，不预选hook/API重构；DP2只在瓶颈证据支持时进入角色内对照，不做配置扫描。此段覆盖下方“唯一补证问题”的优先级；旧Reset/next_action不是执行队列。CurrentNone/active PERF_KEEP空不变。

2026-10-06 用户明确纠正目标：**标准P/D分离**，DP1/TP16或DP2/TP8等在P池、D池各自内部组织；不接受完整请求副本作为产品。Sol接受独立Astra纠偏Review，亲查实际原生proxy/connector及Run101原始helper、D body、外部KV命中metrics，见[PD_SCOPE_CORRECTION](records/points/GLM-OPT-0002/research/engine_commit_20261006/PD_SCOPE_CORRECTION.md)。撤销下段“优先PP1公平对照”及DP_PP_REVIEW中两独立TP16完整副本的当前研究优先级；保留原裁决历史。现役noKV/Full Replica与目标不符，旧PDv2/v3对合法请求绕过PD、DP1硬限制和export→dispatch取消回收未闭环是已定位合同缺口；DP2不支持、所有历史PD是假均未获证明。停止完整副本性能线与PP phase候选，不启动配置扫描/服务操作。唯一补证问题是原生标准PD全功能请求与KV生命周期的覆盖；最大可消除性能Gap及收益unknown，CurrentNone/active PERF_KEEP空不变。任何后续设备Run先做fresh现场核验及符合严格PD目标的新Reset/decision table；旧Reset与next_action不作为执行队列。

2026-10-06 用户追问DP/PP部署价值后，Sol亲读当前安装MoE分片源码与Run38原始启动log，完成独立Astra架构比较Review：[DP_PP_REVIEW](records/points/GLM-OPT-0002/research/engine_commit_20261006/DP_PP_REVIEW.md)。历史DP4TP8EP32/PP1实际fit且37完成，不能排除DP；DP2TP8EP16本机实际fit/完整正确性仍unknown。接受优先准备一个PP1公平对照的证据判断，不默认PP最优、不宣称DP已胜、不启动部署/Run。H1 scoped REJECT、H2 INCONCLUSIVE/PARKED、CurrentNone/active PERF_KEEP空不变；新device Run前须fresh Reset与decision table，下面checkpoint159及历史证据不作执行队列。

2026-10-06 checkpoint159：从远端研究分支b23845e4按GLM-RESEARCH-RULES-v2恢复；首次真实Zcode只读Job VALID，已进入实际安装源码、native控制与原始证据研究。Current=None，active PERF_KEEP为空。完整产品任务仍开放；本阶段没有新增代码级性能 KEEP。真实完整E2E Gain=unknown。最新[Reset](records/points/GLM-OPT-0002/research/engine_commit_20261006/POST_REVIEW_RESET.md)、[Sol设备裁决](records/points/GLM-OPT-0002/research/engine_commit_20261006/H2_DEVICE_DECISION.md)、[Run244 manifest](records/points/GLM-OPT-0002/runs/GLM-RUN-0244/manifest.json)；不继承下方checkpoint158的next_action。

## checkpoint159当前现场与裁决

现役仍是Full Replica / Complete-request Placement：original239D0插件V2TP8PP2DCP8/K2与211D1 V1TP4PP4DCP4/K1，public入口仍241冻结V14。Run242仅trace诊断后恢复原生代码，D0新域local-242R-166（epoch038ec789…），旧D0 STORE退休；D1原域/STORE保留。public配置已换成Run242/restored/service_config.json，不能重用241旧恢复配置。实际argv/所有roots/members由该目录native_engines_resident.json发现。最新只读证明在GLM-OPT-0002/jobs/FINAL-RESET-AUDIT-20261006/final_site.json（与Run244现场相同）：native32、三处health200、idle、public owner和D1 STORE不变。采集profiler已退出，无当前实验controller；身份observer仍服务驻留。下一次现场操作必须重验，不按这里的旧PID执行。

[H1裁决](records/points/GLM-OPT-0002/research/engine_commit_20261006/H1_DECISION.md)：Run242的28个有效输出reply producer在额外schedule前尚未开始，28个提前发布结果全为空；27个空commit前下一非空Decode已入队，末尾为清理。4请求各64 token的core commit与SSE完全一致。enqueue-first推迟ready useful commit在这个短窗口被否定，范围外unknown；未实施drain-first。六项CPU原生FIFO/future/异步消费者语义验证通过，代码只作为DIAGNOSTIC保留，未进产品stack。

Run242 host诊断exit0并完整恢复；device组件INVALID（PROFILING_MODE=dynamic禁用torch-npu API）。Run243 INVALID（动态模式duration/无stdin start，无设备raw），其专属profiler清理VALID。Run244采集phase因CLI wait超时FAILED，保留失败原始状态；真实16 rank start/stop/quit与device raw存在。随后只读audit、离线export/16-rank归约/跨rank广播联结VALID，不重跑负载。四请求各64输出只作diagnostic，不是标准完整E2E。bridge/schema/错误health端点失败也保留，不改写历史或以VALID数算性能成果。

最大可信可由代码消除的Gap/时间仍unknown。当前设备事实是PP阶段计算反相：1.193s共同有效输出窗口，两阶段计算重叠约3.57%；约33ms receiver broadcast等待对应后端必要target/MTP work。不能把它计成可消除33ms或50%设备空泡。独立Astra [Review](records/points/GLM-OPT-0002/research/engine_commit_20261006/H2_DEVICE_REVIEW.md)已完成并全接受；[Goal Review](records/points/GLM-OPT-0002/research/engine_commit_20261006/GOAL_REVIEW.md)触发后暂停惯性NPU Run。唯一条件候选H2 native admission phase-cohort为INCONCLUSIVE/PARKED：未证明AIV资源允许独立cohort并行，天然两请求终态尾部也不是匹配拆批成本。暂不实施性能patch/新Runtime路线/大型E2E；反证条件和最小验证只在Sol设备裁决维护。

以下checkpoint158段落仅保留历史原文，不能作为当前现场或执行队列。

2026-10-06T02:23:12.448562+00:00 checkpoint158：Zcode已实际恢复，CPU v5 Result→bridge VALID；唯一controller执行Run240与241完成，Zcode独立只读审计均VALID。parent157 24dc052a8cee8ab30cd8568fa746eb5af1603ea7。完整任务继续开放，CurrentNone/性能KEEP无/达标稳定容量和全可行域上界unknown。研究代码与raw在服务器、Mac只SSH；GPT关键研究代码裁决，真实Zcode执行/监控/归约；仅task资源/唯一controller，无旧队列/额外审批/新kernel。

## checkpoint158历史现场

public241/V14 HOST3212971/start299869850/boot6d9cf06f-a02d-49ef-b7fe-cc456a7b2025；config与HOST owner/observer证明在GLM-OPT-0002/runs/GLM-RUN-0241/restored。native239D0 HOST2064352/start295229107，epochf5085b8d6a9df34627f3845bc867c0eec70f83b6af363d3afae2c452bd32e0bc；native211D1 HOST380291/start293474771，epoch5551a30026d04dfc8be640232c9232d6d0c94a45cebb1997481144371a9e6f42。241终态两组healthy/idle/active0，native32/roots/epochs未变。240和241controllers已completed退出；仅public-lifetime HOST identity observer驻留，不称新Zcode后台任务。

D0 V2TP8PP2DCP8/K2/nativequeue资源3、empty queue admission2serial1；D1 V1TP4PP4DCP4/K1。8192t1024c1serial1/noEP/noKV/DP1；239target3n/draftprefill3n/draftdecode1..8；数学/precision/kernel/原生unsupportedguards保持。旧230D0 STORE已退休503，211D1 STORE保留；新增241base/schema/sse/background状态nativeD1，取消241cancel信用0。

## 最新代码与功能证据

V14 opt-in bundle在240/runtime_bundle（241继承）：新POST /v1/responses从V1域开始，避免未来structured链的固定nativeowner落在不兼容V2；真实结构化tool_choice required/named/jsonSchema/jsonobject等及thinking budget筛选V1，普通chat仍shape/load策略。thinking None与整数-1按原生no-op保留原路由。bound nativeowners固定/无迁移或STORE复制；排空/fault不转投不兼容native。body/native参数校验/wire/采样不改，兼容成本须真实E2E计入容量。

CPU v5实际Zcode11mockcalls VALID，ByteStream/唯一模拟ID夹具修正已验证；不是native性能证据。v3/v4及旧1850/v2原raw/claim失败保留，不再复用。

Run240 COMPLETED spec61934a84c2916e3ebe4e3ab715229b9ba508cd5ffcc0f869d931352f1690345c/source88：public239 exact cleanSDK0→public240/V14；native模型不重启。四fresh真实tools auto/none→D0、required/named→D1，4/100完成，D0 2/76/326prompt、D1 2/24/320prompt；nativeIDs/semantic-weatherShanghai/JSON+SSE/body-wire/usage/计数/lease/SDK0/native32/STORE211均VALID。audit240 reduction21349B SHA260529d5af7b3f8676af6e660d0e6403ef99f9843874d8f868af3cffe604842b。240public随后241 exact清理退休。

Run241 COMPLETED spec41ce1b15cb452658fd58abd8d22cba52e43d5b75686a263a3da2c1d5527008c4/source88：11/179完成，另1原生background8192跨public240 cleanSDK0重启至public241后同owner检索/取消，取消信用0。JSONschema/jsonobject city Shanghai语义通过；thinking0/8原生关闭token154842位置0/8，null/-1 D0，非法bool原生400；新Responses plain/store起始V1、previous schema、typed foreground/background SSE和starting_after回放、400404/旧retired503、同epochV1drain/readd保留状态/新无兼容member503前RPC通过。HTTPlease0并不代表nativejob0，实际后台运行已记录；模型和32worker/epoch不变。取消未计输出，原生未计完成的额外计数{"D0": {"uncredited_generation": 0.0, "uncredited_prompt": 0.0}, "D1": {"uncredited_generation": 843.0, "uncredited_prompt": 60.0}}。三个client SDKinit-final0；两公共入口SDKinit-final0/newresident init0。audit241 reduction21892B SHA45329b8f6be80d95bca6d8ca6c8d28795b85b9bbb829079254dbb05f30d2e680。有限功能证据，不是完整性能KEEP/稳定容量/全局界。

## checkpoint158历史下一研究问题

Zcode可用，执行阻塞已解除；不停止在CPU、搬文件或计划。复用139完整API与最新240/241功能证据，按风险补真实采样/可变到达等未覆盖分支；选择高价值调度/容量或nativeV2 structured原因区分性实验。public241 V14保持功能兼容，但新Responses都从V1域开始的资源/延迟成本未知；容量须统计同合同合法完整请求/输出，cancel与poll不能当完成。

native即时grammar候选尚未激活：原生EngineCore立即采样getmask缺deferred分支draft getter/filter，PP cooldown可能pendingFalse但scheduled drafts仍[-1,-1]。v2自洽词法CPU源码算例和5项guard/filter/rollback检查有效；236真实[455,68852,709] get_weatherge与旧bonus掩码相符，实际236 branch/placeholders与PP getter cohort覆盖/时效未记录，根因unknown。候选grammar_draft_control.py仅控制原生draft获取和filter时机，无kernel/math变更；缺身份覆盖fail-closed，真实首请求/交错/成本仍待，不盲热补丁。

## 历史适用与边界

231cap2 131.777 /234同epochcap3 94.249 /235repeatcap2 135.568 TPS均16/14208合法三SLOFAIL；236nativeV2required500/FSM失败，仅auto1/12有效，4失败输出不计；238V1D1工具4/41有效。239重建D0新epoch及V13首个plain2有效，旧230D0退休。Graph/draftcapture/MTP/queue机制证据和大量raw索引在checkpoint152–156；固定epoch重复token轨迹也可能不同，不能孤立GPUcause或声明到顶。DynamicK+DCP与PCP/PP native unsupportedguard不绕过。CurrentNone。

## 恢复入口

repo/data/tiankuan/wio/Inference-Foundry branch glm5-3-autonomous-20261001；site/data/tiankuan/wio/glm52-pd/deploy/glm52-single；Mac166/tmp/glm52-166.sock，167经166HOST ssh root@172.16.10.167。恢复AGENTS/START/HANDOFF/point，按需MISSION/PLAN/REUSE/RECORDING。核验fresh boot/start/argv/NSpid/NPU/locks/controller唯一性，不按旧PID盲操作。raw/spec/claimedJobs不改，服务器保存原日志；GitHub nonforce父CAS/其他dirty保留。
