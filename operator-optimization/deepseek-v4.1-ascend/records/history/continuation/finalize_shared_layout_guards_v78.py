"""Whole-scope audit and shared-MLP cross-card semantic coverage checkpoint."""
from copy import deepcopy
from pathlib import Path
from finalize_origin_coverage_v9 import N,O,read,write
from persist_profile_refinement_v5 import persist_refinement


def main():
    assert not(O/'REVIEW-CHECKPOINT-v78.json').exists()
    names=['all-module-fixed-case-domain-audit-v78.json','shared-layout-guard-cases-v78.json',
        'shared-shard-precision-proof-v78.json','shared-layout-work-communication-models-v78.json',
        'shared-layout-source-index-v78.json','shared-collective-API-evidence-v78.json',
        'candidate-relation-clauses-v78.json','candidate-relation-coverage-v78.json',
        'operator-analysis-coverage-ledger-v78.json','shared-layout-branch-inventory-v78.json','next-operator-proof-questions-v78.json']
    for name in names[:6]:assert(O/name).exists()
    old={m['id']:m for m in read(O/'module-phase-branch-matrix-v77.json')['modules']}
    rel=deepcopy(read(O/'candidate-relation-clauses-v77.json'))
    r=next(r for r in rel['clauses']if r['id']=='REL73-09')
    r['refinement_v78']=dict(operator_claim='TP-sharding shared weights is a joint quantization/reduction/output-layout rewrite. Current SP constructors replicate linears;generic modeflag is insufficient. Full-I amax and rawINT32 reduction before Down dequant preserve the stated contract;naive per-shard quant andBF16partial RS have counterexamples.',
        work='Balanced m localrows:GEMM6mHI staysunchanged;new allgather/globalamax/INT32RS and new owner BF16merge join must be priced. GenericONEADD/MergePost/DownAdd still count mergeonce.',
        evidence=names[:6],all_runtime_gates_proved=False,critical_path_gain_us=None)
    r['evidence']+=names[1:6]
    hg=next(g for g in rel['higher_order_groups']if g['id']=='HG73-MOE')
    hg['shared_layout_refinement_v78']=dict(contract='Choose consistent parameter layout plus fullrow quantizer/rawINT32 reduction and rawoutput ABI before overlap/merge pricing. Kernel-HCCL multi-layer submission-domain count constraints apply only in that declared API domain,not general modeldepth/host capture.',evidence=names[1:6],all_subsets_analyzed=False)
    write(O/names[6],rel)
    cg=deepcopy(read(O/'candidate-relation-coverage-v77.json'));cg.update(parent_coverage='candidate-relation-coverage-v77.json',shared_layout_refinement_v78='REL73-09/HG73-MOE',overall_complete=False);write(O/names[7],cg)
    cov=deepcopy(read(O/'operator-analysis-coverage-ledger-v77.json'));cov.update(parent_coverage_ledger='operator-analysis-coverage-ledger-v77.json',
        new_pending_regimes_reconciled=0,new_pending_executions_reconciled=0,
        shared_layout_guard_review_v78=dict(existing_shared_regimes=42,existing_shared_executions=22360,source_quant_selector_cases=56,source_layout_cases=8,
            new_regime_promotions=0,precision_witness='shared-shard-precision-proof-v78.json'),overall_complete=False);write(O/names[8],cov)
    # Reconcile legacy chronologicalM13rows,archived in full by persist_refinement.
    definitions=[
        ('Current typed sharedchain and BF16 input fanout','Observed42regimes22360executions/4472eachstage;BF16input→INT8+rowscale→INT8weightG1 INT32→DSQ→INT8weightDown BF16→merge. Shared input is rounded normalizedBF16,not routerFLOAT/pre-round values.','Currentnativeattributes/loadedquantizer and allinputgeneration/alias/readers still conditional.'),
        ('Physical sharedTP group and fixedmodel SP parameter construction','Controller uses physicaltp_group.world_size,not EP-rewritten logicaltp_size. FixedDeepseekV41MLP sets disable_tp=SP in bothlinears;current observed fullH×2I/I×H weights agree with replicatedSPlayout.','No flag-only weightsharding. Newpartitionedconstruction/packing/checkpoint/scaleownership and actualrankrowmaps required.'),
        ('Generic TP/SP/SEDP controller layouts and rowplan helpers','Eight sourcebooleanmodecases partitioned:TP5,SP+SEDP1,SP-only1,nonSPsharedDP1. SP-only AG/unpad/RS and nonSP DPpad/shard/gather keep fulloutput contracts;controller labeldoesnotproveweightobjects.','TP1/otherSP/DP branches are conditional alternatives,not observedruntimecasehits;unbalancedN/P must be bound.'),
        ('A8INT/MXFP/linear-wrapper quantization selector','56sourcecases:2A8INT,1A8MXFP,53wrapper acrossLoRA/two-scale-presence/sevenquantclasses. A8INT permitsW8A8/W4A8,but observed nativeweightsareINT8;LoRA/missing-scale useswrappers.','No substitution ofunobservedINT4/MXFP/LoRAnumerics;actualadapter/helpernativebehavior remainsperapplicabledomain.'),
        ('ClampedSwiGLU/Situ/MXFP/wrapper activation domains','Fixedmodel declaresSiluAndMulWithClamp10;observedDSQpath usesINT32→FP32/mode1/clamp10. Situ andMXFP branches areseparate. InstalledstandardGMM-SwiGLU API lacksprovedcurrentclampcontract.','Constructorclass isnotloadedinheritance/objectproof;preserve allglu/scale/NaN/activation contracts andspecificwrapper readers.'),
        ('InputDynamicQuant plus GateUp fusion and producer reuse','FullHrow scale precedes finalINT8 values. Quantbeforegather canconditionalshareoneoriginalrowquantization;newINT8+FLOATscaleAG preservesBF16readers androwepochs.','Loadedexactquantizer/shapeindependence/rowstrides plus new gatherABI/packing/events required;nofree duplicatequantwork.'),
        ('GateUp→activation→rowquant and globalI amax','ExistingF22UP/SPLIT fusion/split models retained. AcrossIshards,localamaxchangesglobalquantizer;8shardwitness[1×7,64] excludesdroppingglobalamaxforcurrentcontract.','GlobalFLOATMAXreduce/finiteNaNzero semantics,newbuffers/latency and actualnative dequant order required.'),
        ('Down integer partials, dequant and BF16 rounding','ValidINT8partialdots257/−256:roundBF16afterfullsum=1,roundpartialsthenBF16sum=0. BiasfreeK2304integerpartial/reductionmagnitude≤37748736fitsINT32.','RawINT32SUM RS beforefloatdequant is a newconditionalABI,not currentBF16Down+RSdropin;native dequant ordering still unproved.'),
        ('Shared weightsharding work/memory/communication tradeoff','BalancedP8rowcapacity model:currentreplicatedtoken-shard andnewTPweight-shard each6mHI GEMMops/rank. Weightpayload35,389,440→4,423,680B;correctquantAG+amax+INT32RS ringreference537852B/rank at m3.','No8xcomputebenefit or actualVRAM/HBM/speedupclaim;validN/paddedrows,tileutilization/networklatency/resources must bepriced.'),
        ('Current MC2serial path and counterfactual overlap milestones','SavedMC2=1 plus pinnedvalidation disables current sharedoverlap;currentserialevidence retained. Alternate earlyDQ/GateUp/activation/Down requires router/stage/combine/output events.','Cannotadd idealbranch-overlapcap to fusioninterfacebudget;finiteCube/Vector/BW/communication contention and readinessremain.'),
        ('Tensor output/legacytuple/wrapper and helper consumers','Fixedsharedcontroller returnsTensor;genericwrapperpart2 may have Qwenexpert_gate but fixedMLPctor doesnotcreateit. SP/nonSP/localDP finalization and defer_output_wait retaincallbacks/banklifetimes.','Do not assume allgeneric helpers absentglobally;exactempty/None/pad/alias/observers retainedwhentheyaffectoutputs.'),
        ('DownBF16→shared+routedBF16 merge boundary','CurrentDownround precedesmergeBF16round beforeHCwiden;newDownAdd must preservebothrounds andwaitmatchingroutedrows,includingpostshardownerresults.','Allmergealpha/scaling/tuple/shape/reader/event/rowownerconditionsremain.'),
        ('SingletonAdd specialization and merge→HC alternatives','ONEADD,DownAdd andMergePost shareoneexisting4472ForeachAddList stage;all3200graphMerge→HCsourcepairs andnativeHC arithmetic retained.','Mutuallyoverlapping interfacebudgets countedonce;allproducer/HCresidual/nextPre/auxreaders retained.'),
        ('Shared/routed common quantized input and padding generations','F21/F22QCOMMON requiresidenticalBF16input/rowmap/mask/quantizer. CurrentMC2BF16input andreceiverexpertrows neednewABI;sharedN800vsMC2P816 cannot beassumedidenticalplane.','Keepvalidprefix/tail/fullEP receiver work,newquantizeddispatchscales/ownership andnativequantizeridentity.'),
        ('Crosslayer/iteration capacity, API domain and state generations','Currentkernel-HCCLdocA3totalPrepare+InterSync≤63 is conditionalAPIconstraint:3calls/layer×40=120 cannotshareoneunchangeddomainunderthatlimit. Parameteraddress isnotnewinput/cachehitproof.','Notalimitonmodeldepth/hostcapture;CANN9.2loadedcompatibility,allwriters/shape/stream/rankepochs andactualtypedcapacity/lowestutilizationranking stillopen.')]
    branches={'M13':[],'M23':deepcopy(old['M23']['branches'])}
    for i,(branch,finding,gap) in enumerate(definitions,1):
        branches['M13'].append(dict(id=f'M13-OP-{i:02d}',branch=branch,state='SCOPED_OPERATOR_GUARDS_FUSION_PRECISION_AND_REPARTITION_ANALYZED_CONDITIONAL;WHOLE_MODULE_OPEN',
            evidence=names[:8]+['shared-expert-shape-ceiling-v57.json','shared-expert-fusion-variants-and-interactions-v57.json','shared-overlap-activation-contract-v21.json','merge-to-HC-contract-v22.json','HC-operator-numerical-and-dependency-contract-v59.json'],
            finding=finding,gap=gap,scope='Exact frozen V4.1 target/draft sharedMLPcurrenttypedpaths and statedcounterfactualoperatorlayouts;not framework research/nativeimplementation.'))
    branches['M23'].append(dict(id='M23-OP-19',branch='SharedMLP replicatedSP→TPweight-shard:full-I amax plus rawINT32SUM RS before dequant;quantizedinputAG alternative and new output-owner merge.',
        state='CONDITIONAL_NEW_TYPED_COLLECTIVE_REPARTITION_DOMAIN;NOT_OBSERVED_OR_NATIVE_VALIDATED',evidence=names[:8],
        gap='Currentprofile hasBFP16RS only;publicdocsupportsINT32/FLOATSUM/MAX forA3 butnotinstalledversionproof. Newweights/rawaccumulator/quantizer/capture/unevenrowprotocol andactualnetworkcapacitiesneeded.',
        scope='Balancedcapacityformula/newoperatorgraphonly;allactualTP/DP/EP/CPgroups/receiverwork remainseparate.'))
    mapping={1:[1,4,5,6,7,8],2:[11,12],3:[1,10],4:[15],5:[2,3,10],6:[12,13],7:[5,15],8:[1,14],9:[9,10],10:[6,7,8,9,13,15],11:[12,13]}
    write(O/names[9],dict(modules=branches,previous_M13_chronological_records=11,new_M13_semantic_families=15,
        old_M13_record_to_new_families={str(k):[f'M13-OP-{n:02d}'for n in ns]for k,ns in mapping.items()},
        M23_existing18_families_preserved=True,new_M23_semantic_family='M23-OP-19',protected_other_modules=24,
        input_global_records=335,result_global_records=340,input_semantic_modules=22,result_semantic_modules=23,
        remaining_chronological_modules=['M00','M02','M03'],whole_modules_closed=0,overall_complete=False))
    write(O/names[10],dict(completed_bounded_questions=[dict(id='M13-SELECTOR-CONSTRUCTION-AND-EXACT-TP-SHARDING',state='CONDITIONAL_OPERATOR_ANALYSIS_REVIEWED',evidence=names[:6])],
        questions=read(O/names[0])['next_work'],
        whole_model_guard_audit_state='26modules/335inputrecordsretrievalandscopeauditcomplete;onlyM13guarddomainsnewlyexaminedthisturn. Otheractualbranchproofsremaininexistingfamilies;allguardclosureNOTclaimed.',
        unchanged_pending='9regimes1352executions=7MemSet1272+2BOOL80;849PMU/16phaseunbound/3136ambiguous',overall_complete=False))
    write(O/'PROFILE-CODE-QUESTION-GATE-v78.json',dict(status='ALL_MODULE_SCOPE_CROSSWALK_AND_SHARED_MLP_OPERATOR_LAYOUT_PRECISION',
        previous_turn_classification='PROGRESS:v77constantpaddingrawoutputreuseandcostproof',
        questions=['Which of26moduleinventories are semanticversuschronological?','DoesgenericSP-onlymode prove fixedMLPweights areTPsharded?','Canper-shardquant/BF16partialRS preserve currentfull-I INT8quant+Downround?','Whatwork/communication/kernelAPIlimitschange for a legalconditionalrepartition?'],
        allowed_source='Allcurrentmodulefamilyrecords andsavedtypedledgers;threealreadyfrozenMLP/model/configclauses;two public primarykernel-HCCLAPIpages. No broadframeworkcode ornewserver/model/devicework.',
        new_remote_source_read=False,new_source_files_obtained=0,new_public_document_snapshots=2,local_operator_clause_read=True,overall_complete=False))
    gap='NewsharedTPweight-sharding comparison requiresconsistentnewparameterownership/full-Iamax/rawINT32beforedequant/newowneroutput andnativeprecision/capturecontracts. PublickernelAPItypes aredocumentedbutnotinstalledCANN9.2attested;no performance/capacitygainproved. Othermodelbranch/composition/physicalcapacity andoperand9/1352gapsremain;conditionalanalysisdoesnotrequireimplementation.'
    updates={cid:dict(review_v78=dict(whole_scope_audit=str(O/names[0]),source_guards=str(O/names[1]),precision=str(O/names[2]),costs=str(O/names[3]),
        source_index=str(O/names[4]),primary_collective_docs=str(O/names[5]),conditional_operator_analysis_deliverable=True,implementation=False,native_test=False,new_performance_result=False))for cid in ['F22','F23']}
    updates['F22'].update(profile_review_state='SHARED_SELECTOR_CONSTRUCTION_AND_CROSS_CARD_QUANTIZATION_REPARTITION_REVIEWED_CONDITIONAL',conclusion_changing_gap=gap,
        next_profile_or_contract_action='RetainF22currentfusion/splitmodels plusconditionalTPweight-shardgraph;currentSPreplicatesweights,compute6mHIunchanged. Globalamax/rawINT32RS/nativequantizer arejointgates. AdvancewholemodelM22operator-lifetime/statecrosswalkandtypedceilings,notframeworkinternals.')
    updates['F23'].update(next_profile_or_contract_action='Afteranysharedrepartition,matchrawINT32RS→originaldequant→oneBF16DownroundbeforeBF16merge/HC;retainroutedrowownerjoinandoneAddbudget. NoBF16partialRSdropinorindependentoverlapgain.')
    mods={mid:dict(finding='M13legacy11chronologicalrecords reconciledto15semanticoperatorfamilies;56quant/8parallelcases+fixedSPconstructor;42existingtypedregimesunchanged. NewsharedexactTPshard domain usesglobalIamax/rawINT32RS andsame6mHI GEMMops/rank,withpricedprotocol/coldweightreferences.',
        evidence=names,candidates=['F22','F23'],gap=gap,profile_scope='SharedquantizedMLP,physicalparameter/rowownership,fullrowquantreduction,Downround/merge and specificcollectiveAPIconstraints only.')for mid in ['M13','M23']}
    remaining=list(read(O/'module-coverage-gaps-v77.json')['remaining'])
    remaining += ['v78all26modules/335inputrecordsauditedforretrievalandoperator-boundaryscope;22semantic4chronological. M13now15semanticfamiliesfromlegacy11archivedrecords,M23addsOP19;340currentrecords/23semanticmodules/3chronological(M00M02M03). Thisisnotproofallfixedcaseguardsnewlyclosed;onlyM13operatorguarddomainnewlyexamined.',
        'v78sharedTPsharding:fixedSPconstructorfullreplicatedlinears,genericcontrollerflagnotweightlayoutproof;56quantcases8modecasescompleteinfinitepartitionedsourceinputdomainnotruntimehits. Current42regimes22360alreadyreviewed,nopromotions. Full-Iamax/INT32-beforedequantreduce plusunchangedBF16merge required;BF16partialwitness1versus0. BalancedperrankGEMM6mHIunchanged,no8xcomputeclaim. NewrawINT32/nativequant/dequant/capture/allrankrowmapping andactualnetworkcapacityconditionsremain.',
        'v78currentkernel-HCCLprimarydocsA3typesINT32/FLOATSUM/MAXand63Prepare+InterSync/domainconstraintareversion-limitedreferences;notinstalledCANN9.2proof,notmodeldepth/ordinaryhostcapturelimit. Single3call/layer×40domain120>63conditionalcountproof;splitdomains/resets/otherAPIneedcontract. Current14720HCCLtypedcensusnoINT32RS.45relations10groups/550centralpending andglobal9/1352+849/16/3136+0wholeclosuresunchanged.']
    action=('继续原Goal,v78 PROGRESS(前v77有效)。用户指定profile优先，只围绕算子融合/拆分重组/理论上限，不研究框架。先读all-module-fixed-case-domain-audit-v78/shared-layout-guard-cases/shared-shard-precision-proof/shared-layout-work-communication-models/shared-layout-source-index/shared-collective-API-evidence-v78和next-operator-proof-questions-v78。'
        '全26模块335旧记录检索/作用域交叉表已建，不等于全335分支新证明。22semantic+4chronological；M13旧11归一为15families并完整archive，M23新增OP19，现340记录23semantic，M00M02M03仍legacy。24其余modulematrix字节结构完全不变。'
        'M13固定SPconstructor在GateUp和Down都disable_tp=True，创建完整复制weights；genericSP-onlycontroller mode不能证明固定对象TP分片，不能flipflag作合法重组。56selectorcase(2INT/1MXFP/53wrapper)与8parallelcase(5TP/1SPSEDP/1SPonly/1DPonly)是源码分区，不是runtimehit；现42regimes22360sharednative仍4472每stage，不增operatorcount。'
        '新的跨卡条件图：一致新TPweights/packingownership，fullH输入DQ一次可AG_INT8+scale，pairedlocalG1/原FP32clampedactivation，globalIamax后INT8量化，DownrawINT32partial SUM RS后原scale/dequant/BF16round再与routedBF16merge。每分片自己amax改变原量化；257/-256INT8dot抽象反例证明BF16partialsum0≠fullsumround1。INT32各partial<=K*16384不溢出只在声明conventionaldot域，不证明nativefloat次序。'
        '平衡capacityN=P*m模型每rankGEMM6mHI和当前SP+rep都相同，weightpayload35.38944MB→4.42368MB但非actualVRAM/HBM。m3新quantAG+MAX+INT32RS普通ringservicebytes537852B/rank，3collectives；naiveBF16AG+RS430080B不保精度不能作为等价成本。原shape/unpad/不均匀actualN另证。coldweightref22.1184→2.7648us只published1.6TB参考，不是speedup，M800densecompute上限不因分片8x。'
        '两份publickernel-HCCLdocs快照SHA已存：A3支持INT32/FLOAT及SUM/MAX为documentedreference，未绑定installed9.2/nativecompatibility；现14720HCCLcensus只有BFP16RS无INT32RS。docA3单通信域Prepare+InterSync总≤63，仅若同kernelHCCLdomain跨layerunroll，3*40=120超限(无其他call最多21layer)；不泛化为模型深度或ordinaryhostcapture上限。'
        'REL73-09/HG73MOE补强但不增边/组，45clauses10groups550central未汇入。下步M22S06-09/S12-13只对照现算子/状态schema去重publication/lifetime义务，查真正缺失值代际/读者，不扩框架；M02按物理state/allocator/rollback重组旧记录；并独立推进typedcapacity/模块同口径ranking和595pair旧证据汇入。条件分析不要求实现native验证，真正未分析范围仍open。'
        '2073regimes285598conditionalreview/9regimes1352pending(7MemSet1272+2BOOL80)、849missingPMU16phaseunbound3136ambiguous、35model7legacyframework、0wholemodules闭环保护。无remote/model/Torch/NPU/newcapture/服务/进程/实现，3PASS Run，无本轮ERROR。verify_shared_layout_guards_v78应PASS。Goalactive，不因台账或报告成稿complete。')
    report='''本轮完成了全部26模块的覆盖记录交叉核对，并补齐共享专家跨卡重组的算子约束。没有修改模型、框架或设备环境。

原335条记录中，22个模块已有语义分支，4个仍混有按分析轮次追加的记录。共享专家已有42类、22,360次原生执行的条件分析，本轮将其11条旧记录归一为15个算子分支，并保留完整历史；通信模块增加1个共享专家重组分支。当前为340条记录、23个语义模块。这个整理不代表全部固定负载分支已被新证明。

固定模型在SP模式下，Gate-Up和Down构造都设置disable_tp=True，即权重复制。通用共享专家控制器中的“SP-only”标签，不能证明固定模型对象已持有TP分片权重。因此，权重分片需要连同构造、参数打包与所有权一起改变，不能只比较配置标志。

源码量化选择器按LoRA、两组scale存在性和量化类别划分为56个案例；并行模式选择器为8个案例。这是有限条件分区，不能当成实际运行的分支命中次数。现有profile仍是每个阶段4,472次：输入量化、INT8权重Gate-Up得到INT32、激活再量化、INT8权重Down得到BF16、共享/路由合并。

跨卡重组新增两处不能跳过的精度约束。第一，当前激活量化依赖完整I=2304行的最大值，每卡分别量化会改变结果。第二，Down的整数部分和应先归约，再执行原有dequant和一次BF16舍入。抽象合法INT8点积部分和257与−256给出反例：完整和舍入得到1，分别舍入再相加得到0。这不是一次原生设备测量，而是排除不受条件限制的等价声明。

在平衡的P=8、每卡m行比较域内，当前复制权重配合token分片，与新权重分片的每卡GEMM工作量都为6mHI，不能再算八分之一。权重有效载荷从35.39MB/卡减为4.42MB/卡，但保留精度的重组还需输入gather、全行最大值归约及INT32 ReduceScatter。在m=3的普通ring服务字节参考中，量化后gather加两种归约合计537,852B/卡；它不是物理网络下界，也不是预测提速。

当前公开[ReduceScatter](https://asc.gitcode.com/api/SIMD-API/adv_api/HCCL_communication/HCCL_Kernel/ReduceScatter.html)与[AllReduce](https://asc.gitcode.com/api/SIMD-API/adv_api/HCCL_communication/HCCL_Kernel/AllReduce.html)文档列出了A3的INT32/FLOAT及SUM/MAX能力，但未绑定本机CANN9.2。文档还限制单个A3 kernel-HCCL通信域中的Prepare和相关同步调用总数不超过63：仅在这个适用域内，3调用/层乘40层的单域融合会超限，需要合法分域、重置或其他协议。这不是对模型深度或普通host图捕获的限制。现有14,720次HCCL记录中，没有本候选所需的INT32 ReduceScatter执行证据。

现有45条候选对关系和10组组合约束已补充上述条件，没有虚增新主候选或重复计算收益。当前仍保留2,073类、285,598次执行的条件分析，9类、1,352次关键操作数证据待补；全模型分支、组合、实际物理容量和同口径模块利用率结论仍未闭环。下一步只围绕实际算子的状态、发布与读者约束整理M22/M02记录，并继续跨候选组合与理论上限取证。Goal保持进行中。
'''
    src=read(O/names[4]);docs=read(O/names[5])
    jobs=[dict(kind='profile',status='pass',description='All26module335inputrecords crosswalkedwithoutguardclosureclaim;current42M13regimes22360exec/4472each typedstage and14720HCCL typedcensus retained. No reassignments/promotions.',artifacts=[O/names[0],O/names[1],O/names[5],N/'audit_all_module_domains_v78.py']),
        dict(kind='design-check',status='pass',description='Threepinnedmodel/MLP/configclauses;56quantselector/8layoutcases,fixedSPfullweights andfullIamax/BF16partialcounterexamples;conditionalrawINT32RSgraph andbalancedcompute/communication/coldweightmodels. TwoAPI snapshots carryversion/domainlimits;no nativeexperiment.',artifacts=[O/names[i]for i in[1,2,3,4,5]]+[Path(r['path'])for r in src['source_files']]+[Path(r['snapshot_path'])for r in docs['primary_sources']]),
        dict(kind='review',status='pass',description='M13legacy11archived→15semanticfamilies;M23old18preserved+OP19;24othermodulesidentical. 45relations10groups/550centralpending,2073/285598review9/1352pending/global84916/3136/0closuresprotected.',artifacts=[O/names[i]for i in[6,7,8,9,10]])]
    persist_refinement(version=78,previous=77,title='All-module operator-scope audit and shared-MLP precision-preserving cross-card repartition',
        summary=dict(filename='shared-layout-scope-ceiling-summary-v78.json',previous_turn_classification='PROGRESS:v77constantpaddingrawoutputreuseandcostproof',
            audited_modules=26,input_records=335,result_records=340,input_semantic_modules=22,result_semantic_modules=23,
            quantization_selector_cases=56,parallel_controller_cases=8,observed_shared_regimes=42,observed_shared_executions=22360,
            new_M13_semantic_families=15,new_M23_family='M23-OP-19',protected_other_modules=24,balanced_shard_GEMM_compute_ratio=1,
            M3_precision_graph_ring_reference_bytes=537852,source_files_reused=3,new_source_files_obtained=0,new_public_document_snapshots=2,new_remote_source_read=False,
            relation_clauses=45,higher_order_groups=10,pending_central_pair_reconciliation=550,scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,
            pending_regime_reconciliation=9,pending_execution_reconciliation=1352,implementation=False,native_test=False,new_performance_result=False,all_actual_physical_numeric_ceilings_proved=False,
            remote_job_status='No remote/server/model/Torch/NPU/capture/service/process work;onlylocalfrozenoperators,existingprofile,offlinearithmetics andtwo publicAPIpages.'),
        changes=['Crosswalk26module335records;M13legacy11→15semanticfamilies/M23newOP19,notallguardproof.','FixedSPreplicatedparameterconstruction+56quant/8modecases;globalIamaxandBF16partialcounterexamplesrequirejointnewownership/rawINT32RSgraph;6mHIcomputerankunchanged.','Version-limitedprimarykernel-HCCLtype/callbudgetreferences;45relations10groups/profileglobalgaps/0wholeclosurespreserved.'],
        candidate_updates=updates,framework_updates={},module_updates=mods,semantic_branches=branches,remaining=remaining,action=action,jobs=jobs,report=report,
        read_first=names,publication_rows={},gate_updates=dict(overall_complete=False),scripts=[N/'analyze_shared_layout_guards_v78.py',N/'capture_shared_collective_docs_v78.py',N/'audit_all_module_domains_v78.py',N/'verify_shared_layout_guards_v78.py'],
        executor_prefix='shared-layout-guards',main_script=Path(__file__),source_read=False)


if __name__=='__main__':main()
