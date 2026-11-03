"""Persist communication operator decisions, preserving unresolved full-goal obligations."""
from pathlib import Path
from copy import deepcopy
import hashlib
from finalize_origin_coverage_v9 import N,O,read,write
from persist_profile_refinement_v5 import persist_refinement

def main():
    assert not (O/'REVIEW-CHECKPOINT-v66.json').exists()
    pop=read(O/'communication-native-operator-population-v66.json')
    p=read(O/'communication-operator-regime-models-v66.json')
    cost=read(O/'communication-work-payload-models-v66.json')
    variants=read(O/'communication-fusion-variants-v66.json')
    inv=read(O/'M23-communication-and-cross-operator-branch-inventory-v66.json')
    cov=read(O/'operator-analysis-coverage-ledger-v66.json')
    contract=read(O/'communication-operator-contract-v66.json')
    assert pop['operation_count']==14720 and len(pop['operations'])==14720
    assert p['regime_count']==17 and p['captured_execution_count']==11528
    assert cost['count_group_regime_count']==61 and cost['type_dtype_rank_count_regime_count']==42
    assert cost['semantic_role_count_group_regimes']==76 and len(variants['variants'])==20
    assert cov['scoped_conditional_review_regimes']==1015 and cov['scoped_conditional_review_executions']==201431
    assert cov['remaining_regime_reconciliation_count']==1067 and cov['remaining_execution_reconciliation_count']==85519
    assert inv['M23_new_semantic_families']==18 and inv['new_cross_module_families']==4
    assert pop['noncompute_ops_to_schedule_one_to_one_join_proved'] is False
    old={m['id']:m for m in read(O/'module-phase-branch-matrix-v65.json')['modules']}
    changing={'M07','M11','M15','M18','M23'}
    assert len(old)==26 and old['M23'].get('branch_schema')is None
    assert all(old[mid].get('branch_schema')=='SEMANTIC_BRANCHES_V1'for mid in changing-{'M23'})
    for a in read(O/'checkpoint-manifest-v65.json')['artifacts']:
        q=Path(a['path']);b=q.read_bytes();assert len(b)==a['bytes'] and hashlib.sha256(b).hexdigest()==a['sha256'],q
    protected={mid:m for mid,m in old.items()if mid not in changing}
    assert len(protected)==len(old)-len(changing)==21
    write(O/'communication-protected-module-state-v66.json',dict(modules=protected,
        updated_modules=sorted(changing),M07_branches_unchanged=old['M07']['branches'],
        scope='Other21modulematrixrecordsidentical;M07reviewhistoryonly. M11/M15/M18retainalloldsemanticrowsandappend2/1/1;M23archives29chronologicalrecords.',overall_complete=False))
    e=['communication-native-operator-population-v66.json','communication-compute-regime-bindings-v66.json',
        'communication-graph-role-bindings-v66.json','communication-work-payload-models-v66.json',
        'communication-operator-contract-v66.json','communication-fusion-variants-v66.json',
        'communication-operator-regime-models-v66.json','operator-analysis-coverage-ledger-v66.json',
        'operator-regime-refinement-v66.json','M23-communication-and-cross-operator-branch-inventory-v66.json',
        'communication-local-validation-history-v66.json','communication-protected-module-state-v66.json']
    write(O/'PROFILE-CODE-QUESTION-GATE-v66.json',dict(
        status='COMMUNICATION_OPERATOR_COUNTS_NATIVE_IDENTITIES_AND_REPARTITION_COSTS',
        previous_turn_classification='PROGRESS:v65operator_head_Markov_sampler_snapshot_models',
        operator_questions=[
            'WhatdoCOMMcountandComputeInfocommunicationnodesmeanforeachcollectivetype?',
            'CanattentionoutputWA/WBmovebeforeTPheadrestoreandwhatnewweights/padding/precisioncost?',
            'Whenarecopygatherbatching/directdestinations/projectionorRSpipelinesvalidwithallreadersandreadytimes?',
            'Whichnetworkwork/traffic/resource/cutparametersareactuallyboundversusjustdeclaredalgorithmreference?'],
        allowed_source='Fiveprevious166SHA-boundlocaloperator/callerfiles;restoreTPheads/projectoutput/SPboundary/embeddingconsumeronly. PrimaryHCCLAPIarityreferencesversionqualified. No frameworkcatalog/HTTP/SSE/loader/lifecycleaudit.',
        new_remote_source_read=False,local_pinned_operator_clause_review=True,
        no_model_Torch_NPU_capture_health_process_kill_or_implementation=True,
        actual_physical_network_ceiling_proved=False,overall_complete=False))
    scopes={
        'F02':'EmbeddingAR/RSownershipcostincludingPceil8paddingandrequiredfull-readerAG;standardembeddingcomplete-first-layerconsumerretained.',
        'F03':'RSoutput→localHcPosttile/purecopycrosslayerbatch/readiness/generationconstraints;noequal-shapeCSE.',
        'F14':'Attention/inverseRoPE→TPsendlayoutdirectproducernewABIretainsbothBF16boundariesandalloutputreaders;AlltoAllperpeercountnotwholeplane.',
        'F15':'WAbeforeheadrestore:all8groups/tokenowner,4096→1024exchange,payload/4but+58.72MB/card/layerWA. ActualA/Mpadding/tiling/BF16guardandstreamedA2A-WAalternative.',
        'F16':'OwnerallWA/WBweights(+132.12MB/card/layer)canconditionalremoveoutputA2A/RSwithsameFonlyA=M,all8BF16partials/nativeSUMorder;WB-RSpanel/localHcPostandresourcepipelinecost.',
        'F24':'Engramgatherrole176callsandtypedcontentpayloadreconciledwithv62row-head/UVA/allreaderandgenerationmodels;notwholeparentextentwirevolume.',
        'F26':'104HCand104FP32premixgathers,mixed-bytebatch2→1launchwithunchangedpayload/P8pack-unpack18qcost;allrawsnapshot/normalizedreadersandreadytime.',
        'F27':'312threecheckpointauxgathers,conditionaldefer/retain/batch3→1launchwithunchangedpayload/layoutunpack/lostoverlap;3distinctgenerationsnotCSE.',
        'F29':'Existingv64contextfeatureRS-AG/slotmodelsreconciledwith61nativecount/groupcombinations,3norms/padding/fullcontextbankqueryreaderskept.',
        'F31':'13communicationrolecensus/61group-countmodels,AlltoAllperpeer/RSrecv/AGsendarity,3192nonComputeopsnoschedulebijection,copybanks/generation/raggedprotocol/physicalboundgaps.',
        'F32':'4472attentioninputSPgathers,streamedownedrowprojection/project-before-gatherrolemodelsandAIVVectorcontention/row-ready/fullcachepublicationconstraints;nofreequeryTPfactor.',
        'F35':'192vocablogitgatherscountmodelmatchesv65globalbestsummaryvsfullscore/probability/entropy/RNGconditionalprotocols;payloadratiosnotcriticalspeedup.'}
    updates={cid:dict(review_v66=dict(native_population=str(O/e[0]),operator_regime_models=str(O/e[6]),
        count_work_payload_models=str(O/e[3]),operator_contract=str(O/e[4]),variants=str(O/e[5]),
        candidate_specific_scope=s,conditional_operator_analysis_deliverable=True,implementation=False,native_test=False,new_performance_result=False),
        profile_review_state='COMM_OPERATOR_TYPED_COUNTS_REPARTITION_WEIGHTS_PRECISION_AND_RESOURCE_CONDITIONS_REVIEWED',
        conclusion_changing_gap='Actualvalidrows/CP-SPowner/aliases/allreaders/generations/newnativeprecision/HCCLreductionpolicy/loadedalgorithm/topology/injection/cut/latency/HBM-L2remainconditions;conditionalanalysisknownwithoutimplementation.',
        next_profile_or_contract_action='NextunreviewedM05/M06/M19/M04/M10embedding-HC-draft-vectorregimesandM21metadata/indexoperatorfamilies. Codeonlyforconcreteoperatorfusion/work/IO/dependency;noindependentframeworkresearch.')for cid,s in scopes.items()}
    defs={
        'M23':('All14720HCCLcallsreconstructedfrom32hash-verifiedrawDBs;11528exactM23Computejoinsin17regimes+3192otherCOMMops. 61group-label/countmodels42typedcombinations76semanticcombinations13roles,20operatorchoices/18M23semanticfamilies. NofabricatedSchedulebijection/physicalnetworkcapacity.',list(scopes)),
        'M11':('AlltoAllcountperpeerandWA-beforeexchange/ownerallWA-WBpartitionspriceextraweights/A-Mpadding/nativeBF16partials/SUMandresourcecost. Retain12semanticfamilies+2newones;existingG1/twoGEMMcontractsunchanged.',['F14','F15','F16','F03']),
        'M15':('HC/premix104+104mixedcopygathers:payloadunchanged,2→1launchconditionalnewbytepack/multioutABIand18qpack/unpackbudgetP8. Keepall7semanticfamilies+1new;rawHCsnapshot/FP32premix/nativeBF16normreaderslive.',['F26','F31']),
        'M18':('312auxresidualgatherscanconditionaldeferandcoalesceafterthirdcheckpointwithlocalsources/allreader/lostoverlap/layoutcost. Keep18semanticfamilies+1new;existingcontextfeaturepartition/padding/queryweightconsumersretained.',['F27','F29','F31']),
        'M07':('4472inputSPgathers/61countmodelsconnectexistingv61F32ownedrowprojection/streamedGEMMcontracts. SameblockKVcacheWAR/fullrow/normquant/readiness/AIVVectorresourceconditions;11semanticbranchesunchanged,reviewhistoryonly.',['F32'])}
    mods={mid:dict(finding=f,evidence=e[:10],candidates=cids,
        gap='Actualrow/alias/readergeneration/precision/physicalnetworkandresourceconstraints;allunobservedapplicablemodelbranchesremainopen.',
        profile_scope='AllsavedHCCLopidentitiesandtypedanalyticalchoices;MC2EP16/Gloo/otherprotocolsseparate,notmodel/frameworkclosure.')for mid,(f,cids)in defs.items()}
    remaining=[]
    for s in read(O/'module-coverage-gaps-v65.json')['remaining']:
        if s.startswith('v65fullobservedcoverage'):continue
        if s.startswith('Operatoruniverse stillincomplete:'):
            s='Operatoruniverse2082regimes/286950ComputeInfoexec/83types:1015regimes201431execreviewed,1067regimes85519pending. FullsavedHCCL14720callsinclude3192outsideComputeInfoandarenotaddedtotheComputeexecdenominator;allmodelbranches/fullscopeclosureseparate.'
        remaining.append(s)
    remaining.append('v66communication17regimes11528TASKreconciledagainstall14720COMMops,557688internalrecordsand61group-label/countmodels42typedcombinations76role/groupcombinations. 3192COMMopswithoutComputeInfoand3192Scheduledefinitionsretainedasdistinctsetswithoutbijectionclaim. 20choices;WA-preexchangeweights+58.72MB/card/layer,ownerWA/WB+132.12MB/card/layerwithA/MpadandBF16/HCCLsumconstraints;M23semantic18+crossmodule4newfamilies. 1015/201431reviewed,1067/85519pending. PendingbyfunctionalmoduleM21=828/76943,M05=113/3832,M05|M21=2/80,M06=23/520,M19=38/1560,M10=1/48,M04=59/1576,M12MemSet=3/960. Actualphysicalnetworktopology/injection/cut/latencyandtypedmemorycapacitiesremainexplicit;0wholemodulesclosed.')
    action=(
        '继续原Goal最新v66 PROGRESS。上一轮v65也是PROGRESS。用户要求代码只为算子能否融合和理论上限，不研究框架。profile优先，缺消费者/舍入/依赖/状态证据才定点读源码。'
        '先读operator-analysis-coverage-ledger-v66、communication-native-operator-population-v66、communication-operator-regime-models-v66、communication-work-payload-models-v66、communication-operator-contract-v66、communication-fusion-variants-v66及M23-communication-and-cross-operator-branch-inventory-v66。'
        '全2082/286950观察现在1015/201431显式条件review、1067/85519pending、0geometry-only。v66只promote17M23/11528，另把3192非ComputeCOMM也完整建模，不能把14720全加进Compute分母。'
        '32rawDB各SHA复核；14720COMM=11528connection+EXACT interval+operator ComputeJoin+3192无ComputeInfo。Graph同connection重放2次必须按interval区分。3192Scheduledefs数量相同但op134同conn只有Notify_Record/Wait，不可假定Schedule一一对应；错误preflight已停/留证。557688内部record不是消息数/bytes可直接累加。'
        '61rawgroup-label/countmodels=42(type,dtype,P,count)组合；76role/group/count组合、13语义角色。初次42误用于含group键已assert拒绝，保留61而非合并。4639directfullchain+1原queueflowproof+10080graph，80×126calls含64root/16unbound；来源schema不是tensor-edge或actualmetadata值证明。'
        '关键count语义：AG每卡sendCount，RS每卡recvCount/inputP倍，A2A每peer count，AR整卡array。M24 A2Acount12288 BF16只是24576B/peer，rankplane196608B，remote172032B；不是24576B总流量。declaredring/pairwisereference量不等于物理wire/单ring或universalceiling，RING-RING-NHR标签不够。NetworkactualBW/topology/injection/cut/latency/HBM-L2未知，不报微秒floor/speedup。'
        '新WA-beforeexchange把4096→1024，M=A24 A2A发送172032→43008B，output A2A+RSpayload2/3；需每卡完整8groupWA权重67.108864MB vs8.388608MB，extra58.720256MB/layer/40层2.34881024GB，F仅A=M相同，新G8localM vsG1globalM精度/tiling/冷权重。M20/A24、6668/6672padding需额外work或validrow-zero-fill/rowmap证明。'
        'Owner全8groupWA/WB可conditional免output A2A/RS，extra132.120576MB/card/layer，40层5.28482304GB；保留各WA BF16和WB BF16 partial及原nativeHCCL SUM政策/order，不能一大GEMM/自由FP32加。BF16[512,-512,1]二种order1vs0仅反例，不是实际HCCL精度实测。AIV占Vector，不是免费网络独立资源。'
        '20variants还包括直接attention/inverseRoPE sendlayout、A2A-WA/WB-RS/HcPost分块、embeddingAR→RS(全读者AG会加回通信)、HC BF16+premix FP32 2→1bytecopy AG(payload不变，普通pack/unpack18q额外读写)、3aux deferbatch3→1(保存3checkpoint/丢提前overlap/布局代价)、existingv65greedy/v64context/v62Engram协议。跨层同shape/地址CSE和无边信息BF16→INT8全值exact压缩排除。'
        'M23旧29chronology归约18semantic；M11旧12+2、M15旧7+1、M18旧18+1都保留，M07旧11只加history；其余21完整modulematrix记录保护。35模型候选+7legacy框架候选，0整模块完成；849missingPMU/16unboundgraphs/3136ambiguity/3MemSet960继续保留。'
        '下一步先剩余非M21算子：M05 113/3832 embedding/mask/cast/lookup，M06 23/520 HCinit，M19 38/1560 draft，M04 59/1576 norm小向量，M10 1/48 pad布局；再M21 828/76943输入metadata/index operator families及2M05|M21/80，具体profile分组后只定点补代码。3MemSet缺args/size证据仍pending，不能凭named kernel/报告关Goal。'
        '本轮全本地无服务器连接/无模型Torch/NPU/健康/进程/kill/实现，conditional分析交付不需要实现才算分析；所有真实未分析范围/上限条件必须保持，Goal active不得完成。')
    report='''本轮把通信当作算子与数据所有权问题分析，新增17类、11,528次计算表通信节点的条件分析，并检查全部14,720次HCCL调用。32份原始数据库均重新核对SHA；全程使用本地保存数据及5份已绑定原166版本的算子源码。

14,720次调用中，11,528次可以通过数据库内connection、完全相同的时间区间和算子类型对应到计算节点。其余3,192次保留为独立通信调用；通信调度定义也有3,192条，但数量相同不证明一一对应。557,688条内部记录同样不能当作消息数量或额外数据量累加。

通信已按13种语义用途、42种类型/dtype/卡数/count组合建模，加入原始group标签后为61种组合。AllGather的count是每卡发送元素数，ReduceScatter是每卡接收元素数，AlltoAll是每个peer的元素数，AllReduce才是每卡完整归约数组。固定24行图中，AlltoAll count=12288、BF16，每卡完整输入为196,608B，发送给其他7卡的内容为172,032B；不能把count×2=24,576B当成整卡通信量。[HCCL接口定义](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/80RC3alpha003/apiref/hcclapiref/CANN%208.0.RC3.alpha003%20%E9%9B%86%E5%90%88%E9%80%9A%E4%BF%A1%E6%8E%A5%E5%8F%A3%E5%8F%82%E8%80%83%2001.pdf)、[ReduceScatter定义](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/82RC1alpha001/API/hcclapiref/hcclcpp_07_0024.html)。接口参数含义由固定代码布局和profile count交叉验证；这些较早文档不证明当前9.2的算法或物理容量。

本轮较重要的重组是把第一段输出投影WA移到token所属卡，再交换结果。

| 条件方案 | 每卡每层代价与收益 |
|---|---|
| 当前先恢复TP heads，再WA→WB→ReduceScatter | WA权重8.389MB，WB权重10.486MB；24行时恢复heads远端发送172,032B |
| token所属卡先算全部8组WA，再恢复TP布局 | 交换维度4096→1024，发送43,008B；WA权重增至67.109MB，额外58.720MB |
| token所属卡计算全部8组WA/WB并完成等价归约 | 有条件免去输出AlltoAll与ReduceScatter；权重额外132.121MB，必须保留每组BF16中间结果和原归约语义 |

在交换行数与投影行数相同的条件下，前移WA不增加每卡常规矩阵乘操作数，输出AlltoAll＋ReduceScatter的逻辑发送量变为原来的2/3。但它把每组权重复制到所有卡，新G8小行数计算的tiling、冷权重读取和BF16精度均需保留为约束。40层推广会额外占用约2.349GB/卡；两级都前移约5.285GB/卡。若20行补齐到24行、6668补齐到6672行，直接在补齐行上计算还会增加工作，必须单独核算有效行裁剪、补零和行映射。

两级都前移也不能把分组结果随意用一次FP32求和替代当前HCCL归约。BF16值[512,-512,1]的两种成对归约顺序可以得到1与0；这说明需要确认原生累加和舍入政策，并不代表已测出当前HCCL的政策。

另有分块流水与调用合并机会：attention/逆RoPE直接写发送布局、AlltoAll到WA的面板流水、WB与ReduceScatter及本地HcPost交接、HC和FP32 premix的混合字节gather、以及三次aux checkpoint gather延后合并。它们保留原始内容、消费者、舍入和发布顺序。HC/premix从两次gather合成一次不会减少payload；P=8时，普通打包与拆包反而增加18q字节逻辑读写。三次aux合并还会丢失提前通信机会并需要保存三份局部值，不能只计算少两次调用。

embedding的AllReduce改成ReduceScatter，在对齐且所有读者都接受行分工时，ring参考发送量可以减半；当前标准embedding明确提供完整首层序列。任何仍需完整结果的读者若补回AllGather，总通信量就恢复原值，补齐时还可能更多。词表最优项摘要、上下文feature分片、Engram按行/头交换则与此前分析的概率、熵、RNG、完整bank和UVA读者约束一同保留。

所有通信字节与轮次均声明为ring或pairwise参考算法；记录中的RING-RING-NHR标签不证明实际只有一条flat ring。当前缺少绑定到这次执行的物理链路、注入带宽、cut容量、时延和HBM/L2流量，故本轮不编造通信微秒下界或端到端加速。AIV通信还使用Vector资源，会与norm、quant和RoPE竞争，计算与通信也不能默认免费重叠。TP8的HCCL记录不覆盖MC2内部EP16通信及CPU/Gloo元数据交换。

共记录20项条件方案或排除结论，建立18个通信语义分支，并向保留的输出投影、末端状态和aux上下文分支增加4项；其他21个模块矩阵记录保持一致。全观察台账现为1,015类、201,431次执行已有显式条件分析，尚有1,067类、85,519次待核对。真实物理容量、所有适用模型分支和3个MemSet待定项仍保持开放，整体Goal未完成。
'''
    jobs=[
        dict(kind='profile',status='error',description='Retain3resolvedlocalassertions:graphconnectionreplayneedsintervalqualification;3192nonComputeCOMMnotprovenSchedulebijection;42typedkeysbecome61withgroup-label. Allfailedbeforecurrentartifact/persistenceandcorrectedwithoutdroppingcalls.',artifacts=[O/e[10]]),
        dict(kind='profile',status='pass',description='32rawDBSHAverified;14720COMM/557688internal/11528exactComputejoins/3192otherops;4639direct+1queue+10080graph,80graphs64root/16unbound,13rolesconserved.',artifacts=[O/e[0],O/e[1],O/e[2],N/'analyze_communication_operators_v66.py']),
        dict(kind='review',status='pass',description='5old166SHA-boundlocaloperatorclausespinTPheadlayout/projection/padding/embeddingownership;primaryAPIarityreferencesversionqualified. No remote/frameworkaudit.',artifacts=[O/e[4]]+[Path(x['path'])for x in contract['source_files']]),
        dict(kind='design-check',status='pass',description='61group-count/42typed/76role-combinationsmodelall14720calls;20choiceswithWA-beforeexchange/ownerweight-A-Mpadding/HCpack18q/embeddingfullreadertradeoffsandBF16sumcounterexample. No actualnetworkpeak/nativebenchmark.',artifacts=[O/e[3],O/e[5],N/'model_communication_operators_v66.py']),
        dict(kind='review',status='pass',description='1015review201431+1067pending85519conserve2082/286950. M23semantic18+crossmodule4newbranches,M07historyonly,21otherrecordsprotected. Globaluncertainties/0wholemoduleclosurepreserved.',artifacts=[O/e[6],O/e[7],O/e[8],O/e[9],O/e[11],N/'consolidate_communication_operators_v66.py'])]
    persist_refinement(version=66,previous=65,title='Communication operator count semantics, cross-card projection partitions and cost constraints',
        summary=dict(filename='communication-operator-ceiling-summary-v66.json',previous_turn_classification='PROGRESS:v65operator_head_Markov_sampler_snapshot_models',
            reviewed_regimes=17,reviewed_executions=11528,all_HCCL_operations=14720,nonComputeInfo_HCCL_operations=3192,
            communication_internal_records=557688,group_label_count_models=61,type_dtype_P_count_models=42,semantic_role_group_count_models=76,semantic_roles=13,
            operator_decisions=20,M23_new_semantic_families=18,new_cross_module_semantic_families=4,protected_other_modules=21,
            observed_regimes=2082,scoped_reviewed_regimes=1015,scoped_reviewed_executions=201431,
            geometry_only_regimes=0,geometry_only_executions=0,pending_regime_reconciliation=1067,pending_execution_reconciliation=85519,
            MemSet_explicit_pending_regimes=3,MemSet_explicit_pending_executions=960,
            noncompute_ops_to_schedule_one_to_one_join_proved=False,new_remote_source_read=False,local_pinned_operator_clause_review=True,
            implementation=False,native_test=False,new_performance_result=False,all_actual_physical_numeric_ceilings_proved=False,
            remote_job_status='No new remote job/connection;all local immutableDB/pinnedoperatorclause/math. No model/Torch/NPU/capture/health/process/kill/implementation/backgroundanalysis.'),
        changes=['17M23regimes11528TASKscopedconditionalreviewagainstall14720HCCLops/3192nonComputeops,32rawDBSHAverified;61group-label-countmodels42typedcombinationsand13roles.',
            '20communicationoperatorchoicesquantifyWA-beforeexchangeandownerweights/padding/precision,streamedproducer-consumerhandoffs/copybatching/embeddingfullreaderpenalty;noinventedphysicalnetworkfloor.',
            '1015regimes201431execreviewed,1067pending85519;M23semantic18and4crossmodulebranchesnew,M07historyonly/21othersprotected;globaluncertainties0wholemodulesunchanged.'],
        candidate_updates=updates,framework_updates={},module_updates=mods,semantic_branches=inv['modules'],remaining=remaining,action=action,
        jobs=jobs,report=report,read_first=[e[7],e[6],e[0],e[3],e[4],e[5],e[9]],publication_rows={},gate_updates=dict(overall_complete=False),
        scripts=[N/x for x in ['analyze_communication_operators_v66.py','model_communication_operators_v66.py','consolidate_communication_operators_v66.py','verify_communication_operators_v66.py']],
        executor_prefix='zcode-communication-operators',main_script=Path(__file__),source_read=False)

if __name__=='__main__':main()
