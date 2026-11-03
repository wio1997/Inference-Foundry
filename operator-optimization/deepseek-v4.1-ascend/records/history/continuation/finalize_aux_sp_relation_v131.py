"""Seal F27/F32 conditional packet and representation analysis once."""
from copy import deepcopy
from pathlib import Path
from finalize_origin_coverage_v9 import N,O,read,write
from persist_profile_refinement_v90 import persist_refinement


def main():
    assert not (O/'REVIEW-CHECKPOINT-v131.json').exists()
    pre=read(O/'aux-sp-preflight-v131.json');assert pre['status']=='PASS' and pre['checks']>10000
    names=['aux-sp-inputs-v131.json','aux-sp-checkpoint-profile-v131.json','aux-sp-selector-profile-v131.json',
        'aux-sp-source-contract-v131.json','aux-sp-ordered-selection-models-v131.json','aux-sp-owner-balance-and-loss-witnesses-v131.json',
        'aux-sp-packet-work-and-capacity-models-v131.json','aux-sp-causality-and-variant-graph-v131.json','aux-sp-numeric-and-snapshot-gates-v131.json']
    detail=dict(evidence=names,checkpoint_aux_SP_graph_pairs=240,root_bound_pairs=192,phase_unbound_pairs=48,
        eager_aux_COMM_kept_unpaired=72,positive_prefix_ConcatD=16,positive_indexed_ConcatD=72,
        ordered_selection_models=7742,reference_cost_geometries=5,conditional_schedule_variants=5,
        new_scoped_pair_relations=1,new_pair_clauses=1,new_groups=0,new_main_candidates=0,
        actual_I_tensor_alias_or_native_numeric_bound=False,new_native_promotions=0,new_device_work=False,whole_requirement_closed=False)
    contract=(
        'Checkpointauxmean5120 isdistinctfromcurrentmHCweighted/inputnorm/projectedSWA512;F32carrierdoesnotremoveauxbyshape. '
        'Earlyaux+projectedSWApackkeepssamepayloadasindependentF32andretainedmainfeatureAG;2→1collectiveoperationspercheckpoint isnotprovenlaunch/latencygain. '
        'PostsampleexactIcannotgate thesamepacketrequiredbycurrentattention;earlyknownprefix/provedsupersetJandlateorderedowneraux aredifferentgraphs. '
        'Earlyfuturecontext/pre-RoPEKV rowfunctionneedscompatible3checkpointowners/fullmainW,allobserverclosure,N/Mwork/error/nativeM/norm/rounding/position/cache/bankcontracts. '
        'Feature-shardFP32continuationstillneedsallauxinputandnativeaccumulationorder/parity;diagonalblocksor3BF16partialoutputsareinsufficient. '
        'RetainmixedN640feature/n512SWArowdomains,packing/readerretirement/extraweights;contentandsnapshotmodelsarenotphysicalwire/HBM/allocatedVRAM/speedup.')
    rel=deepcopy(read(O/'candidate-relation-clauses-v130.json'));assert not any({r['a'],r['b']}=={'F27','F32'}for r in rel['clauses'])
    rel['clauses'].append(dict(id='REL131-01',a='F27',b='F32',relation='CHECKPOINT_AUX_SWA_PACKET_AND_EARLY_MAIN_REPRESENTATION',
        operator_claim=contract,evidence=names,applicable_domain='Selected3checkpointDSparkaux/currentSWApaths;earlyprefix,indexedlateI,ownerfutureandfeatureaccumvariants haveexplicitdistinctgates.',
        conditional_compatibility=True,all_runtime_gates_proved=False,speedup_additive=False,critical_path_gain_us=None,aux_SP_composition_v131=deepcopy(detail)))
    group=next(g for g in rel['higher_order_groups']if g['id']=='HG85-VERIFY-CONTEXT-DRAFT')
    assert {'F27','F32'}<=set(group['members']);group['aux_SP_composition_v131']=dict(**detail,contract=contract,whole_group_closed=False)
    rel.update(parent_relation_index='candidate-relation-clauses-v130.json',new_pair_clauses=1,new_higher_order_groups=0,
        clause_count=138,selected_pair_count=138,aux_SP_composition_v131=detail,overall_complete=False)
    write(O/'candidate-relation-clauses-v131.json',rel)
    cov=deepcopy(read(O/'candidate-relation-coverage-v130.json'));r=next(r for r in cov['pairs']if {r['a'],r['b']}=={'F27','F32'})
    r.update(scope_state='SCOPED_OPERATOR_RELATION_REVIEWED_CONDITIONAL',clauses=['REL131-01'],whole_pair_all_variants_closed=False,new_critical_path_gain_us=None,aux_SP_composition_v131=deepcopy(detail))
    cov.update(parent='candidate-relation-coverage-v130.json',parent_relation_index='candidate-relation-clauses-v131.json',scoped_conditional_pairs=138,remaining_pair_relations=457,aux_SP_composition_v131=detail,
        scope='138scopedconditionalpairs/26groups/35modelcandidates;457centralpairsremain. F27-F32 checkpointpacket/readiness/earlyrowfunctionreviewisconditional;allruntime/allvariantsopen.',overall_complete=False)
    write(O/'candidate-relation-coverage-v131.json',cov)
    nav=deepcopy(read(O/'candidate-pair-navigation-v130.json'));r=next(r for r in nav['pairs']if {r['a'],r['b']}=={'F27','F32'})
    r.update(conditional_clause_ids=['REL131-01'],semantic_pair_reconciliation_done_in_stated_scope=True,whole_pair_all_variants_closed=False,aux_SP_composition_v131=deepcopy(detail),evidence=names)
    nav.update(parent='candidate-pair-navigation-v130.json',parent_footprints='candidate-operator-footprints-v131.json',parent_relation_index='candidate-relation-clauses-v131.json',scoped_conditional_pairs=138,remaining_central_pair_reconciliation=457,aux_SP_composition_v131=detail)
    write(O/'candidate-pair-navigation-v131.json',nav)
    fp=deepcopy(read(O/'candidate-operator-footprints-v130.json'))
    for r in fp['candidates']:
        if r['id']in ['F27','F32']:r['aux_SP_composition_v131']=dict(evidence=names,no_new_native_operator_atoms=True,local_graph='aux-sp-causality-and-variant-graph-v131.json',all_runtime_or_native_parameters_bound=False)
    fp['parent']='candidate-operator-footprints-v130.json';write(O/'candidate-operator-footprints-v131.json',fp)
    op=deepcopy(read(O/'operator-analysis-coverage-ledger-v130.json'));op['aux_SP_composition_v131']=detail;write(O/'operator-analysis-coverage-ledger-v131.json',op)
    obligations=deepcopy(read(O/'coverage-current-obligations-v130.json'))
    for r in obligations['obligations']:
        if r['id']in ['G00','G01','G02','G05','G06','G07','G08','G09','G11','G13']:r['aux_SP_composition_v131']=deepcopy(detail)
        if r['id']=='G07':r['remaining']=(
            '138conditionalpairs/26groups,457centralpairsandallvariantsremainopen. F27-F32mustchooseearlyfullaux/provedearlyJ,lateexactowneraux,earlyfuturefullweightorfeatureaccumulationinonegraph. '
            'EarlycopackhaszeroextrapayloadsavingbeyondF32;lateIcannotgatethesamecurrentattentionpacket. Fixed-countlateauxdependsmaxuniqueownedrows,notM. '
            'PricefullWmainextra131.25MiB/card,N/Mearlywork,mixedrowpackIO,fullnorm/nativerounds/allreaders/capture/bankretirement;allnativephysicalspeedupsopen.')
        if r['id']=='G13':r['remaining']=(
            'v131targetedF27-F32conditionalcompositionuses240graphcheckpointpairs(192root/48phase-unbound),72eagerauxCOMMretainedwithoutTASK/SPpair,and88positiveConcatD(16prefix/72indexed). '
            'Actualindices/earlyconstancy/semanticextent,checkpointowner/communicatormembers/loadedweights/numericM/errorbehavior/nativeaccumulatorresume/oldreadersremainunbound. '
            'Do notmergeauxmeanwithprojectedSWAorclaimcomputedallphysicalrowsareallselected. Main5120norm/3KV512normsandpostselectposition/RoPE/cacheepochsremain. '
            'Nextprofile-firstreviewpendingF28-F32:embedding/HCinitialization,rowowner/paddingandlocalrawreadersversusproject-before-SP;reuseexistingF28/F32sourcecontractsandquery264/dummy48branches,withoutresurveyingframework. '
            'All9/1352nativefields,globalbranch/graph/physicalcapacity/MFU/lowestgatesremain. No newNPU/model/capture/replayauthorization;v120Padconsumed,nopendingapproval.')
    obligations.update(parent='coverage-current-obligations-v130.json',new_native_or_whole_closures=0,overall_complete=False);write(O/'coverage-current-obligations-v131.json',obligations)
    audit=deepcopy(read(O/'whole-scope-requirement-audit-v130.json'));audit.update(previous_goal_turn='PROGRESS:v130F29-F32 owner/carrier/order;postcheckPASS26265',input_checkpoint=130,followup_checkpoint=131,aux_SP_composition_v131=detail)
    for r in audit['requirements']:
        if r['id']=='CROSS_OPERATOR_LAYER_CARD_ITERATION':r.update(conditional_pair_relations=138,central_pairs_not_reconciled=457,higher_order_groups=26)
        if r['id']in ['ALL_MODULE_CONDITIONAL_BRANCHES','CROSS_OPERATOR_LAYER_CARD_ITERATION','THEORY_AND_MEASURED_COMPARISON','CRITICAL_EVIDENCE_AND_FINAL_CLOSURE']:r.setdefault('evidence',[]).extend(names)
    write(O/'whole-scope-requirement-audit-v131.json',audit)
    write(O/'PROFILE-CODE-QUESTION-GATE-v131.json',dict(status='EXACT_CHECKPOINT_COMM_AND_POSITIVE_SELECTOR_CONCAT_THEN_TARGETED_AUX_MAIN_SP_CONTRACTS',questions=read(O/names[0])['questions'],
        source_scope='Onlypinned3auxcheckpoint/mHCSP/mainprojection/fullnorm/orderedselector/earlyreturnsegments;noframeworksurvey,newSDK,remoteorNPUexecution.',
        no_model_operator_implementation=True,no_remote_Torch_CANN_NPU_model_or_replay=True,Pad_authorization='GRANTED_AND_CONSUMED',overall_complete=False))
    history=[
        'Predecessorv130revalidatedPASS26265/manifest102;previousgoalturncompletedF29-F32conditionalrelation,native/wholegoalstillopen.',
        'Reusev64checkpointTASKandv66COMMexactkeys:240aux/SPgraphpairs,192rootbound/48phase-unbound;all72eagerauxCOMMremainunpairedwithoutCOMPUTE_TASKjoin.',
        'Parent-onlyselectorstagehas88aten::catparentswithnoCID;refinementusesexactsame-threadboundednestedaclnnCatthenexactCIDmappingfor88ConcatD. Notnearest-timeTASKjoin. Bothstagesretained.',
        'Positiveprefix2069count16,indexed2089count72;shapesarenotactualIorconstancy. Raw32DB/cachednativepinsandallnativeparent/child/TASKdefinitionsverified.',
        'VerifierdevelopmentfirstwronglyrequiredTASKonexplicitlyunpairedeagerCOMM,thencomparedrawTEXTtimestampswithcachedintegers. Bothfailedscripts/receiptsretained;correctedschema-awarechecksretainmissingTASKandlosslessintegercomparison.',
        '7742finiteordered/duplicate/negativeindexrowfunctionmodels,ownerimbalance/loss/offdiagonal/BF16counterexamplesand5geometrypacket/work/snapshotmodelschecked;notnativeproof.',
        'Earlyaux+projectedSWAco-packis2→1collectiveops/checkpointwithnopayloadsavingbeyondF32. ExactlateIwaitcycleisscopedtocurrentattentionpacket;earlyprefixandsupersetneedseparatecontracts.',
        'Full-weightfuturecontext/3pre-RoPEKVandfeature-FP32continuationreusev34alternatives;newanalysiscouplesreadiness/currentcriticalpath/mixedrowdomains/N-M/weights/lifetimes,nottreatingtheseasnewalgorithmdiscoveries.',
        'Preflight'+str(pre['checks'])+'PASS;REL131-01addsF27-F32,HG85alreadyincludesboth.138pairs/26groups/457pending;358/26semanticand854/1346/48partialgraphunchanged. No runtime/native/wholeclosure.']
    write(O/'aux-sp-validation-history-v131.json',dict(events=history,overall_complete=False))
    gap=('NeedactualI/order/semanticextent,3checkpointrowowners,communicatormembers/rankorder,pointers/weights/gamma/eps/nativeM/norm/accumulationprecision,rawaux/contextobserversandcapture/reader/bankepochs. '
        'AllschedulevariantsmustincludeearlyN/Mwork,errorbehavior,fullmainW131.25MiBextraperCardormissingoffdiagonalinputcommunication,mixedfeature/SWApackcosts andretirement. '
        'No nativefusionparity,physicalwire/HBM/allocatedVRAM/latency/MFU/lowesttimeorwholepair/allbranchclosureestablished.')
    remaining=[r['id']+': '+r['title']+'. '+r['remaining']for r in obligations['obligations']]
    remaining.append('v131conditionalF27-F32relation:138pairs/26groups/457pending;240graphpairs+88nativeConcatD,7742rowmodels.358/26semantic854/1346/48partialgraph9/1352native849PMU16phase3136ambiguity0wholeunchanged.')
    action=(
        'ContinueoriginalGoal;v131PROGRESSisF27-F32checkpointaux/SWApacket/earlymainrepresentationcomposition,notnativeoptimizationorwholeclosure. '
        'Readaux-sp-summary/profile/checkpoint/selector/source/packet/causality/numeric/currentobligations-v131;verify_aux_sp_relation_v131mustPASS. '
        '240graphaux/SPpairs192root48phase-unbound;72eagerauxCOMMhavenoTASKandnopairedSP. Samecount/group-labelisnotalias/membershipproof.88exactnestedaclnnCat→CID→ConcatD:16earlyprefix2069and72indexed2089;actualIunbound. '
        'Auxmean5120differsfrommHCweighted/inputnorm/projectedSWA512. Earlyaux+projectedSWAco-packhaszeroextrapayloadsavingbeyondF32,2→1collectiveops/checkpointonlyreference;mainfeatureAGstays. Separatebufferadapterscanadd3*4(P+1)n(5120+512)logicalvalueIO. '
        'PostsampleIcannotgatethesamepacketrequiredbycurrentattention;lateexact3auxbatchseparate. N24M12balanced/skew/duplicateindicesgiveumax2/3/1;sameMdoesnotprovefixedcountreduction;preserveorder/duplicates/validnegativeextent. '
        'Earlyfuturefullmainprojectionrequiresall3compatiblelocalauxrows/fullWmain150MiBvs18.75MiBshard(extra131.25MiB/card),NphysicalratherthanMselectedwork;maycarrycontext5120/3pre-RoPEKV512/bothinlastSWApacketafterreadiness. Preservefullnorms/BF16rounds/postselectposition/RoPE/cache/error/allreaders;currenttargetpathcanbedelayed. '
        'Feature-shardFP32continuationstillneedsallauxinputs;A_pW_pdiagonalmissesoffdiagonalblocks;3BF16partialsnotexactfullsum(witness0vs2^-8). Nativeaccumulatorresume/order/parityNOT_ESTABLISHED. N640featureversusn512SWAdifferentrowdomains;outputcannotsharepacketdeliveringitsmissingthirdauxinput. SnapshotcontentisnotallocatedVRAM. '
        '7742finiteintegerrowmodels/5geometrycontentmodelsnotnativeproof;v34owner/featurealternativesarenotnewdiscoveries.138conditionalpairs26groups457pending;HG85alreadybothmembers.358/26semantic854/1346/48partialgraph2073/285598conditional9/1352pending849PMU16phase3136ambiguity0whole/noallmodulephysicalMFU/lowestunchanged. '
        'Nextprofile-firstpendingF28-F32embedding/HCinit/decoderrowownership,paddingandrawreadersversusproject-before-SP;reuseexistingF28/F32and264proposal48dummycontracts,onlytargetoperatorgaps. '
        'No remote/Torch/CANN/NPU/model/capture/replay/implementation;alllocaljobscomplete. Padv120consumed,nopendingnewapproval. Do noteditsealedv131/rerunfinalizer. VersionedpostcheckwithCURRENTmatchingRECOVERY. TaskACTIVE Loop004EVALUATING GoalACTIVE.')
    report='''本轮补齐 F27–F32：三份辅助状态、主上下文投影与当前 attention 的投影前移如何共同选择通信包及计算时机。全程使用已有 profile、固定源码和本地数学检查，没有运行服务器、模型、NPU、capture/replay 或优化实现。

原始键关联了 240 组 checkpoint 辅助 gather 与同一图、同一 block 的 attention SP gather，其中 192 组 root 已绑定、48 组 phase 仍未绑定。72 次 eager 辅助通信保留为未配对记录，它们没有 COMPUTE_TASK 节点。相同 count、dtype、P8 和 group label 不证明 tensor 相同、指针别名或 communicator 成员相同。

输入构造有两种正面记录：16 次 worker2069 前缀路径，72 次 worker2089 索引路径。aten::cat 父记录没有 connectionId；通过同线程内嵌 aclnnCat，再按精确 CID 关联到 88 次原生 ConcatD。没有用最近时间猜任务。前缀输出为 6400/6528×15360，索引输出为 12/24/6668×15360；形状不能恢复真实索引，也不能证明它是提前已知的恒等映射。这些构造记录不被强行绑定到上述 240 组图记录。

辅助状态是 checkpoint 的 HC 均值，当前 SWA 路径来自加权 mHC collapse 和 input norm。512 维投影不能无条件替代 5120 维辅助状态。有限反例说明相同归一化后投影可对应不同辅助均值；它否定通用代数替代，不证明训练轨迹里已经出现该状态。

提前把完整辅助平面与投影后的 SWA 平面合包，在行映射、成员顺序、ready、capture、bank 和读者合同满足时，可将每个 checkpoint 两个 collective 操作合成一个。相对于独立实施 F32 并保留辅助 gather，载荷完全相同，没有额外载荷节省；主 feature gather 仍在。三个 checkpoint 加主 feature 的操作数参考为 7→4，不能等同于 host launch 或延迟收益。若兼容原独立输入/输出缓冲区，pack/unpack 还会增加 3×4(P+1)n(5120+512) 字节的逻辑值读写；直接新 ABI 能否省掉适配，须核对布局和全部读者。

若精确 I 来自当前 target 的接受/拒绝结果，它不能门控同一个当前 attention 必需包：包→attention→采样→I→包会形成等待环。这只是所选包角色下的约束。提前已知前缀、经证明覆盖 I 的早期集合 J、晚到 I 后独立发送辅助状态，是不同方案。晚选辅助状态的固定 count 成本取决于 owner 上最多的唯一行数。N24、M12 时，平衡/偏斜/重复索引的 u_max 分别为2/3/1；即使 M 相同，偏斜方案也没有该固定 count 节省。顺序、重复以及按真实语义长度解释的合法负索引必须恢复。

另一条既有候选是：三个 checkpoint 的局部行在相同 owner 准备后，在最后一份当前包发送前提前计算未来 context，或三份 pre-RoPE KV。它需要完整 main 权重：150 MiB 对当前 18.75 MiB shard，每卡额外 131.25 MiB。提前处理物理 N 行而非选择后的 M 行，主投影工作比为 N/M；N24/M12 就是2。必须保留完整5120 norm、三份512 norm、BF16舍入、逐层权重，以及选择后的位置/RoPE/cache语义。还须关闭或保留所有原始 aux/context 读者。未来工作可能延迟当前 target 的关键路径，额外权重、padding、候选缓冲区与资源竞争不能当作免费重叠。

保留640维 weight shard 的 FP32累加表示也不是免费替代。每个 feature shard 仍需全部相关辅助输入；仅计算 A_p W_p 的对角块会缺少其它行与输出块。跨三个 checkpoint 继续原 FP32 累加递推，需要原顺序、cast一次和原 native M 策略的证明；三个独立 BF16 GEMM 后相加不等价，有限例得到0而原求和再舍入为2^-8。当前未建立 CANN 原语或 native 精度可行性。N×640 feature 与 n×512 当前 SWA 的 row domain 不同，需标记平面和 unpack；输出不能与运来它所缺第三份 aux 输入的同一个包互相等待。

7742 组有限整数行函数模型检查了选择前移、顺序、重复、合法负索引及 owner 重建；5组既有参考几何给出 sender 内容、N/M工作、权重和 snapshot 内容。它们不是新 invocation 绑定、网络 wire、HBM必要字节、已分配 VRAM、实测延迟或物理上限。完整主权重/feature shard 等替代来自 v34；本轮新增的是与 F32 的包角色、时机和全读者约束，不将既有算法重复记为发现。

新增 REL131-01，条件关系137→138、待核对458→457。HG85 已包含两候选，联合组仍26，全部子集及运行时合法性未关闭。候选35、历史框架记录7、语义358行/26模块、部分图854节点/1346边/48提示不变。9类/1352次原生字段、849个PMU缺口、16个phase未绑定图、3136项归属歧义均保留；整模块关闭0，整体目标继续。
'''
    inputs=[Path(r['path'])for r in read(O/'aux-sp-inputs-v131.json')['inputs']]
    refs=read(O/'aux-sp-selector-profile-v131.json')['databases'];inputs += [Path(r['path'])for r in refs]+[Path(r['cached_native']['path'])for r in refs]
    updates={cid:dict(review_v131=dict(evidence=names,pair=['F27','F32'],relation_clause='REL131-01',aux_not_projected_SWA=True,early_vs_late_selector_readiness_separated=True,
        ordered_selection_models=7742,full_weight_or_feature_accumulator_costs_explicit=True,actual_I_or_native_numeric_or_joint_fusion_proved=False,no_new_native_or_physical_gain=True))for cid in ['F27','F32']}
    persist_refinement(version=131,previous=130,title='Checkpoint auxiliary and SWA packet/main-representation composition',
        summary=dict(filename='aux-sp-summary-v131.json',previous_turn_classification='PROGRESS:v130F29-F32owner/carrier/order;PASS26265',**detail,
            refined_candidates=2,refined_modules=2,branch_records=358,semantic_modules=26,relation_clauses=138,higher_order_groups=26,pending_central_pair_reconciliation=457,
            partial_graph_node_rows=854,partial_graph_edge_rows=1346,structural_pair_hints=48,scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,
            pending_regime_reconciliation=9,pending_execution_reconciliation=1352,model_operator_implementation=False,native_test=False,actual_MFU_numeric_physical_ceiling_or_speedup_proved=False,
            Pad_authorization='GRANTED_AND_CONSUMED',remote_job_status='Alllocalcachedprofile/source/math/verifierworkcomplete;no remote/device/Torch/CANN/model/replay/serviceswork.'),
        changes=['Bind240checkpointaux/SPgraphpairsand88positiveConcatDviaexactnestedAPI/CID;preserve72eagerCOMMwithoutTASKand48phase-unboundpairs.',
            'ReconcileF27-F32earlycopackversuslateIwaitcycle,ownerimbalanceandfullweight/FP32featurefuturecontextvariantswithallreader/numeric/work/capacitygates.',
            'Addoneconditionalpair:138pairs/26groups/457pending;allnative/global/wholephysicalgatesremainopen.'],
        candidate_updates=updates,module_updates={
            'M18':dict(candidates=[],finding='3auxcheckpointmeanplanesretaindemanddespiteSWA512projection. Earlyfuturemain/contextvariantsneedcompatibleowner/fullWornumericFP32recurrence;preserve5120+3x512normsandpostselectsemantics.',evidence=names,gap=gap),
            'M23':dict(candidates=[],finding='Earlyaux/SWAco-packreducesreferencecollectivecountwithoutnewpayloadcreditbeyondF32;lateexactIisnotallowedtogatesamecurrentattentionpacket. Ownerimbalance,mixedrowplanes,packingandbanklifetimearecosts.',evidence=names,gap=gap)},
        remaining=remaining,action=action,jobs=[
            dict(kind='design-check',status='error',description='Retainedverifierdevelopmentfailures:unpairedCOMMwasincorrectlyrequiredtohaveTASK;rawTEXT/cachedinteger timestampsneedlosslessnormalization. Correctedwithnomissingidentityinvented.',artifacts=[N/'verify_aux_sp_relation_v131a.py',N/'verify_aux_sp_relation_v131b.py',O/'aux-sp-verifier-assumption-error-v131a.json',O/'aux-sp-verifier-schema-error-v131b.json']),
            dict(kind='profile',status='pass',description='240exact-keygraphcheckpointpairs,72unpairedeagerCOMM,16prefix+72indexednativeConcatD;raw32DB/source/CIDverified,no remote/devicework.',artifacts=[O/'aux-sp-checkpoint-profile-v131.json',O/'aux-sp-selector-profile-v131.json',O/'aux-sp-selector-parent-only-v131a.json']+inputs),
            dict(kind='design-check',status='pass',description='7742orderedrowmodels,5packetgeometry/capacityreferences,waitcycleandnumeric/losswitnesses;'+str(pre['checks'])+'preflightPASS. Conditionalrelationonly.',artifacts=[O/'aux-sp-preflight-v131.json',O/'aux-sp-validation-history-v131.json']+[O/n for n in names]+[O/'coverage-current-obligations-v131.json',O/'whole-scope-requirement-audit-v131.json'])],
        report=report,read_first=names+['coverage-current-obligations-v131.json'],publication_rows={},gate_updates=dict(overall_complete=False),
        scripts=[N/'analyze_aux_sp_profile_v131.py',N/'analyze_aux_sp_profile_parent_only_v131a.py',N/'aux_sp_models_v131.py',N/'analyze_aux_sp_relation_v131.py',N/'verify_aux_sp_relation_v131.py',N/'persist_profile_refinement_v90.py'],
        executor_prefix='aux-sp',main_script=Path(__file__),source_read=True)


if __name__=='__main__':main()
