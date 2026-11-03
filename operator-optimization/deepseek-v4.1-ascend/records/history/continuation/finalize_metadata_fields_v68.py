"""Persist profile-bound metadata field analysis; native planner/full scope remain open."""
from pathlib import Path
from copy import deepcopy
import hashlib
from finalize_origin_coverage_v9 import N,O,read,write
from persist_profile_refinement_v5 import persist_refinement

def main():
    assert not (O/'REVIEW-CHECKPOINT-v68.json').exists()
    p=read(O/'metadata-field-regime-models-v68.json');g=read(O/'metadata-field-native-chains-v68.json')
    costs=read(O/'metadata-field-fusion-costs-v68.json');variants=read(O/'metadata-field-fusion-variants-v68.json')
    contract=read(O/'metadata-field-operator-contract-v68.json');cov=read(O/'operator-analysis-coverage-ledger-v68.json')
    inv=read(O/'M21-typed-field-branch-inventory-v68.json');ref=read(O/'operator-regime-refinement-v68.json')
    assert (p['regime_count'],p['captured_execution_count'])==(586,67631)
    assert [len(g[k])for k in ['CP_length_groups','C2_cos_lookup_groups','C2_sin_lookup_groups','SWA_recount_groups','C2_length_quotient_residual_groups','slot_coordinate_core_groups']]==[2792,104,104,69,208,2601]
    assert len(variants['variants'])==22 and len(contract['source_files'])==3
    assert (cov['scoped_conditional_review_regimes'],cov['scoped_conditional_review_executions'])==(1831,276286)
    assert (cov['remaining_regime_reconciliation_count'],cov['remaining_execution_reconciliation_count'])==(251,10664)
    assert ref['remaining_M21']==[242,9312]and ref['remaining_opaque_metadata']==[18,529]and ref['remaining_opaque_family_clear']==[3,79]
    assert contract['existing_C2_fused_builder']['already_implemented']is True
    for a in read(O/'checkpoint-manifest-v67.json')['artifacts']:
        q=Path(a['path']);b=q.read_bytes();assert len(b)==a['bytes']and hashlib.sha256(b).hexdigest()==a['sha256'],q
    old={m['id']:m for m in read(O/'module-phase-branch-matrix-v67.json')['modules']}
    assert len(old)==26 and old['M21']['branch_schema']=='SEMANTIC_BRANCHES_V1'and len(old['M21']['branches'])==18
    assert len(inv['modules']['M21'])==18 and inv['extended_family_count']==8
    protected={mid:m for mid,m in old.items()if mid!='M21'};assert len(protected)==25
    write(O/'metadata-field-protected-module-state-v68.json',dict(modules=protected,
        previous_M21_branches=old['M21']['branches'],extended_M21_family_ids=inv['extended_families'],
        scope='All25othermodulematrixrecordsidentical;M21all18familiesretained,8enrichedwithtypedmodels. No moduleclosure.',overall_complete=False))
    e=['metadata-field-regime-models-v68.json','metadata-field-native-chains-v68.json','metadata-field-fusion-costs-v68.json',
        'metadata-field-operator-contract-v68.json','metadata-field-fusion-variants-v68.json','metadata-C2-header-native-check-v68.json',
        'metadata-fixed-parameter-reconciliation-v68.json','metadata-existing-builder-reconciliation-v68.json',
        'operator-analysis-coverage-ledger-v68.json','operator-regime-refinement-v68.json','M21-typed-field-branch-inventory-v68.json',
        'metadata-field-local-validation-history-v68.json','metadata-field-protected-module-state-v68.json']
    write(O/'PROFILE-CODE-QUESTION-GATE-v68.json',dict(status='TYPED_SWA_CP_C2_FIELD_OPERATOR_WORK_AND_FUSION_BOUNDS',
        previous_turn_classification='PROGRESS:v67embedding_initialization_selected_ID_position_models',
        questions=['Whichper-groupSWAcoordinates/INT64flatoutputsandCP/C2lengthsarefusablewithoutstrongermasking?',
            'CanC2expandedindicesbecomerowlookupsunderexactnativeconversion/error/strides?',
            'Howdoescurrentfallbackcomparewithalreadyimplementedv39all-fieldC2kernel?',
            'WhencanSWAglobalrawlengthreplacelocalC256recountwithclamp/slicing/lifetime?',
            'Whichsourcecopy/field/PMU/opaqueplanobligationsremainoutsidethis586regimescope?'],
        allowed_source='Twoalready166pinnedoperatorcallerclausesplusoldv39fusedkernelhash/equivalencereused. ConfigSHA-bound0/1/2ratios/window128. No framework/API/lifecycleaudit.',
        new_remote_source_read=False,local_pinned_operator_clause_review=True,
        no_model_Torch_NPU_capture_health_process_kill_or_implementation=True,
        opaque_metadata_operators_pending=[18,529],plan_zero_regimes_pending=[3,79],overall_complete=False))
    updates={
        'F31':dict(review_v68=dict(regime_models=str(O/e[0]),native_chains=str(O/e[1]),typed_field_costs=str(O/e[2]),
            operator_contract=str(O/e[3]),variants=str(O/e[4]),candidate_specific_scope='586regimes67631TASKs/2792CP5TASK/2601coordinate14TASKcore/208q-r/69recount. INT32pair+INT64flat/SWArawcountclamp/C2indexexpansion/nativewidthandallreaderfrontiers.',
            conditional_operator_analysis_deliverable=True,implementation=False,native_test=False,new_performance_result=False),
            profile_review_state='SWA_CP_C2_FIELD_TYPED_WORK_IO_CONSUMER_BOUNDS_REVIEWED'),
        'F34':dict(review_v68=dict(regime_models=str(O/e[0]),typed_field_costs=str(O/e[2]),operator_contract=str(O/e[3]),
            header_native_check=str(O/e[5]),existing_builder_reconciliation=str(O/e[7]),
            candidate_specific_scope='104C2header14TASKcores116B+8+source5copies40B;C2complete/sourcepos/C-Sbanksandoldring/index/longreaders. Reuseexistingall-fieldv39kernelconditionallyoutsidepackedguard;notnewmissingimplementation.',
            conditional_operator_analysis_deliverable=True,implementation=False,native_test=False,new_performance_result=False),
            profile_review_state='C2_METADATA_EXISTING_FUSED_BUILDER_AND_SOURCE_COPY_COSTS_RECONCILED')}
    for u in updates.values():
        u.update(conclusion_changing_gap='Completefieldvalue/row/window/group/alias/lastreader-generation/nativeconversion/HCCL-or-C2bankfrontier/physicalVector-index-memorycapacityremainconditions. Opaque18planregimes529and3plan-zero79notcovered;missingimplementationaloneisnotunfinishedconditionalanalysis.',
            next_profile_or_contract_action='Nextremaining242M21regimes9312exec:DSpark/fused-slot/SWA producer/nextIDs/RoPE/accessors/forward-context/fillsand18opaqueplannerops529+3planzeros79. Reuseoldv37-v54contracts;codeonlyspecificoperatorbound/consumerquestions.')
    mods={'M21':dict(finding='586regimes67631TASKhavetypedfield/consumerconditionalmodels. 2792CPgroups/2601coordinatecores/208q-r/104cos+104sin/69recount/104C2headerchains.22choicespriceexistingfusedbuilderreuse/rowlookup/INT64flat/lengthclamp/resourceandbankconstraints;18semanticfamiliesretained8extended.242M21regimes9312remain.',
        evidence=e[:11],candidates=['F31','F34'],gap='OtherM21operatorfamiliesincluding18opaqueplannerregimes529+3planzero79,actualvalues/strides/generations/allreaders/nativeprecision/physicalcapacityandunobservedapplicablemodelbranchesremainopen.',
        profile_scope='OnlySWA/CP/C2/physical-slot/batchfields. SOURCEcopyvaluationnotinventedtracejoin;other25modulesunchanged.')}
    remaining=[]
    for s in read(O/'module-coverage-gaps-v67.json')['remaining']:
        if s.startswith('v67selected'):continue
        s=s.replace('compression0/4/32','compression0/1/2')
        if s.startswith('Observedoperatorfrontier'):
            s='Observedoperatorfrontier2082regimes/286950ComputeInfoexec/83types:1831regimes276286reviewed,251regimes10664pending=242M21/9312+7MemSet/1272+2BOOL/80. HCCL14720populationseparate. Notallmodelbranches/physicalcapacityorwhole-goalcompletion.'
        if s.startswith('M21remaininginput/'):
            s='M21remaining242regimes9312execneedoperatorwork/selectedIO/consumerconditions:18opaqueSMLA/QLIops529+3plan-clear79andDSpark/fused-slot/SWAproducer/nextID/RoPE/accessor/forward-context/fillfields. Reusev37-v54answers;notbroadframeworkresearch. 586fieldregimes67631nowreviewedwith22typedvariants.'
        remaining.append(s)
    remaining.append('v68new586/67631typedmetadatareview. Coordinate14TASKcore140N+3scalarFill1known12B→20NnewINT32pairs+INT64flat. CP5TASK38B+4→20B;C2q/r56B→12B;joint94B+4→28B. C2cos/sin6TASK4112N→1044NrowCast+2gathersor1032Njoint(M6400budget19.712MB),bothFLOATbanks/cast-indexerrors/strideskept. SWA69recounts7684N→orderedglobal-lengthsliceonlyclamp(V,0,C)orcurrentW128Q<=5C256bound/epoch/fences. 104headercore116B+8+5SOURCEcopyvaluation40B=156B+8→36B,notinventednativeMemcpytraffic. FullC2builderalreadyv39underpackedguard;reuselegacyrequiresstride/domain/INT32/bank/frontierproof. ActualcompressRatios0/1/2correctsparentgapwording4/32,notunderlyingprioroperatorcosts.251/10664pendingand0wholemodulesremain;implementationabsencealoneisnotunfinishedconditionalanalysis.')
    action=(
        '继续原Goal最新v68 PROGRESS；v67上一轮也是PROGRESS。用户约束：只围绕算子是否可融合、必要计算/访存/通信/依赖和理论上限，充分profile，缺具体证据再定点代码，不研究框架。'
        '先读operator-analysis-coverage-ledger-v68、metadata-field-regime-models-v68、metadata-field-native-chains-v68、metadata-field-fusion-costs-v68、metadata-field-operator-contract-v68、metadata-field-fusion-variants-v68、metadata-C2-header-native-check-v68、metadata-fixed-parameter-reconciliation-v68、metadata-existing-builder-reconciliation-v68及M21-typed-field-branch-inventory-v68。'
        '本轮586regimes67631exec：SWA_FIELDS308/47642、CP_LENGTHS23/13960、C2_FIELDS118/3256、COMPRESSED_SLOT123/2276、BATCH_COUNTS14/497。全2082/286950现在1831/276286scopedreview、251/10664pending、0geometry。这不是Goal完成百分比。'
        'Nativegroups2792CP5TASK、2601coordinate14TASKcore(3scalarFillseparatelyall2880/line)、208q/r5TASK、104cos3TASK+104sin3TASK、69recount4TASK。独立核104C2header14TASK known116B+8，source5plane.copy_valuation另40B，总156B+8但不伪造MemcpyTASK或HBM。'
        'CanonicalconfigSHA43d3..：compress_ratios0/1/2(40target+3draft),window128。v67gap一句0/4/32是错误scope概括，已receipt修正当前口径，勿当真实压缩比；4/32其他维度分别命名。'
        'Legacy坐标pairINT32[N,2]8N+flatINT64[N]8N，输入rawslotINT32 4N，新fullwrite20N不是16N。14TASK core140N，加3Fill1共12B(非NFill)，条件knownbudget120N+12。ViewCopy16384base是容量，实际Ncolumn。r1/uncompressed不加actualEndmask；r2物理completion+skip+actualEnd+optionalposParity都保留。现packedhelper更强mask/要求positions/ABI，不可无条件替代。r2INT32(s+1)wrap模2等价odd-bit；只有physical=page*Blogical+pos%Blogical且B偶等同代际才合logicalParity，(s5,pos4)反例别漏。B来自getStorageBlockSize，未记录值不得假定power2。'
        'CPGPU_L-suffix→clip→where(nonempty) known38B+4→20B；B2/4不是tokenN。GPUlength拒绝后权威，CPU镜像上界不可替代。已cachehit的临时localL可条件死，但首次miss/global-local/window不同都保，G/实际cachehit数量未伪造。C2q/r原56B→12B，已有lengths:c2缓存不是新收益；联合L/q/r原94B+4→28B需完整fieldgeneration/ready。'
        'C2CS source FLOAT全表1048576×1×1×64每张268MB只是extent，原每cos BroadcastINT64N64→CastINT32N64→Gather，双表known4112N→rowCast+2gather1044N或joint1032N，M6400jointbudget19712000B。保原cast/error边界和两FLOATbanks，不改M18BF16C-S，不直接用advancedIndex负值wrap，也不把row0算成1/0。'
        '完整C2 header/complete/sourcepos/C-S fused kernel其实v39已存在(build_c2_ring_metadata),SHAf288..位于W/sampler-source-supplement/.../ops/triton/c2_ring_metadata.py，当前legacy callback走fallback；本轮只是新native成本对齐和条件reuse outsidepackedguard，不能说缺一个新kernel。旧equivalence条件：INT32界/非负合法position、cos/sin TABLE与OUT共stride(否则改独立stride)、全bank别名/lastreader、COMPRESSOR/input-ready/reuse/zero-queryfrontiers。Header-only/rowlookup是单独粒度，预算不叠加。'
        'SWA global producer本已有rawV长度，CP只传indices未传length，69本地4TASK重算C256known7684N可新sliceview或8Nbankcopy。必须clamp(V,0,C)或证明V<=C；W128,Q<=5给V<=133，原raw400vscount256反例。严格rowmap/epoch/frontier/banklastreader、logicalindex/physicalfallback区分，opaqueSMLA/QLI计划未借此替代。'
        '22variants，M21保留18semanticfamilies只扩展8，其他25整条matrix记录保护。M21剩242regimes9312：18opaque metadata ops529 +3opaque-family ZerosLike79、其余DSpark/fused-slot/SWA producer/nextID/RoPE/accessors/forwardcontext/fills。初次consolidation误把21/608 plan-related family当全opaque操作已assert停止，实际18/529+3/79，局部未checkpoint coverage数值正确已留backup重建，ERROR审计保留。'
        '下一步先读operator-regime-refinement-v68剩余分类，逐family做必要work/selectedIO/消费者/融合/拆分/代际reuse；旧v37-v54已经回答slotfused13ptr数组、nextIDs/selector、C2helper、CP/cache/QLI消费等勿重做。原生metadata18/529缺PMU的TASK类型/opaque ABI/参数/输出读者是具体后续证据问题，不能用通用公式冒充完成。'
        '35model+7legacyframeworkcandidates，0wholemodules关闭；7MemSet1272+2BOOL80仍pending，849missingPMU/16phase-unbound/3136functionalambiguity全保。2caller+1existingkernel本地旧166pins复用，本轮无remote/model/Torch/NPU/capture/健康/进程/kill/实现。Conditional分析不用等实现，但真实未分析范围不得因报告结束而关Goal。')
    report='''本轮按具体数据字段分析SWA、CP和C2元数据，补齐586类、67,631次执行的条件工作量、读写和融合模型。使用本地保存profile、两份已绑定原166版本的调用代码，以及此前已核对的C2融合kernel；没有连接服务器或执行模型。

原生任务已对齐为2,792组CP长度计算、2,601组坐标生成核心、208组C2商/余数、104组cos及104组sin查询、69组SWA有效索引计数。另对104组C2 header逐项核对：14个设备任务的已知值读写为116B+8；源码中五个INT32平面的复制另按40B估值。这些复制没有被伪装成已绑定的Memcpy轨迹。

| 条件方案 | 已知逻辑值读写模型 | 必须保留的内容 |
|---|---|---|
| 一次写入slot坐标与flat bank | 140N+12 → 20N | 两列INT32坐标和INT64 flat都要写；负slot、物理完成、位置奇偶及实际范围按原分支处理 |
| CP长度计算合并 | 38B+4 → 20B | 拒绝处理后的GPU权威长度、正确suffix和空query判断 |
| C2商/余数合并 | 56B → 12B | 原INT32输出与批次代际；现有lengths:c2缓存已是基线 |
| C2 cos/sin按行查询 | 4112N → 1044N或1032N | 原转换和越界语义、两份FLOAT输出、表及输出stride、生命周期 |
| SWA计数转发 | 原计数7684N；新增bank复制时为8N | 同一批索引、顺序CP行切片、容量截断和完整发布协议 |

N表示该字段的token行数，B表示请求数；它们不能混用。原Fill是写一个标量，不能按N个元素估算。flat bank为INT64，每行8B，另有INT32坐标对8B，因此新接口仅必要输出就有16N字节。ViewCopy描述符里的16384是底层bank容量，实际复制仍是N个列元素。

C2查询目前把每行索引扩展到64个维度，再转换为INT32并逐元素gather。按行转换后查询，或使用保留转换/错误策略的联合行查询，可避免扩展索引。在N=6400的案例中，联合方案的逻辑接口预算为19.712MB；整张268MB的cos或sin表只是存储范围，不能算作每次读取量。

完整C2融合已有实现：build_c2_ring_metadata一次生成五个header平面、complete、source position和cos/sin，目前legacy回调仍走逐项构造。复用它是一个已有机制的适用性选择，需要确认合法整数域、cos/sin共用stride假设，或调整为独立stride，并保留固定输出bank、旧ring读者及COMPRESSOR发布/复用前沿。单独的header或行查询方案是其他粒度，不能把全部收益相加。

SWA全局生产者已经输出长度。把它传给局部CP消费者，可以研究避免四个任务的重算，但原长度是raw visible length，非负索引计数是clamp(V,0,C)。当前W=128、Q≤5、C=256的条件足以保证相同；脱离该条件，raw V=400而C=256就是反例。直接借用view还需确认bank不会过早重写；必要时复制到新的固定bank并计入8N读写。

本轮也修正了配置口径：固定配置的compress_ratios为0/1/2，sliding_window为128。上一版未完成项中写的“0/4/32压缩比”不是这份配置的值；当前模型和范围已以固定SHA的配置为准。

共记录22个条件方案或排除结论。1,831类、276,286次执行已有显式条件分析，251类、10,664次仍待核对，包括其余242类M21算子、7类MemSet和2类布尔字段。M21待核对项中，原生SMLA/QLI计划生成是18类、529次，另有3类、79次计划bank清零。实际物理容量、所有适用模型分支和整体Goal仍未完成；本轮的逻辑模型不能直接换算成加速比例。
'''
    jobs=[
        dict(kind='review',status='error',description='Retainresolvedconsolidationpreflight:21/608plan-relatedfamilyincludes3/79ZerosLike;actualcustommetadata18/529. Uncheckpointedcoveragewasnumericallycorrect/archived,allrecreatedaftercorrectscopeassertionbeforewrites;no priorauthoritymutation.',artifacts=[O/e[11],O/'metadata-coverage-preflight-before-refinement-v68.json']),
        dict(kind='profile',status='pass',description='586regimes67631uniqueTASK/PMUcounts;2792CP5TASK/2601coord14core/208q-r/104cos+104sin/69recounttypedgroupsand104header14TASKcrosscheckconserved.',artifacts=[O/e[0],O/e[1],O/e[5],N/'analyze_metadata_fields_v68.py',N/'review_metadata_headers_v68.py']),
        dict(kind='review',status='pass',description='2old166callerSHA+existingv39fusedC2kernelSHA/equivalencereused;actualconfig0/1/2/window128correctsparentwording. Existingfullfusionnotnewmissingkernel;stride/integer/bank/frontierdomainskept.',artifacts=[O/e[3],O/e[6],O/e[7]]+[Path(x['path'])for x in contract['source_files']]),
        dict(kind='design-check',status='pass',description='22typedfieldchoiceswithflatINT64/pairINT32,scalarFill1,CP/q-r/header/sourcecopy/CSrowindex/SWAcountclampmodels. Allnativegroupvaluations/parity-wrap/packedmask/rawcountcounterexamplesPASS. No physicalthroughput/nativeimplementationclaim.',artifacts=[O/e[2],O/e[4],N/'model_metadata_fields_v68.py',N/'reconcile_existing_metadata_builder_v68.py']),
        dict(kind='review',status='pass',description='1831review276286+251pending10664=2082/286950. M21remaining242/9312includesopaque18/529+planzeros3/79;7MemSet1272+2BOOL80/globaluncertaintyretained.',artifacts=[O/e[8],O/e[9],N/'consolidate_metadata_fields_v68.py']),
        dict(kind='review',status='pass',description='M21all18semanticfamiliesretained8extended;other25matrixrecordsprotected,35model/7legacyframeworkcandidates0wholemodulesclosed. Scopeonlyoperatorwork/consumerbounds.',artifacts=[O/e[10],O/e[12]])]
    persist_refinement(version=68,previous=67,title='SWA, CP and C2 metadata field operator work and fusion bounds',
        summary=dict(filename='metadata-field-ceiling-summary-v68.json',previous_turn_classification='PROGRESS:v67embedding_initialization_selected_ID_position_models',
            reviewed_regimes=586,reviewed_executions=67631,operator_decisions=22,M21_retained_semantic_families=18,M21_extended_families=8,protected_other_modules=25,
            CP_length_groups=2792,coordinate_core_groups=2601,C2_length_groups=208,C2_cos_groups=104,C2_sin_groups=104,SWA_count_groups=69,C2_header_groups=104,
            source_pins_reused=3,actual_compression_ratios=[0,1,2],sliding_window=128,existing_C2_fused_kernel_recognized=True,
            observed_regimes=2082,scoped_reviewed_regimes=1831,scoped_reviewed_executions=276286,geometry_only_regimes=0,
            pending_regime_reconciliation=251,pending_execution_reconciliation=10664,remaining_M21_regimes=242,remaining_M21_executions=9312,
            opaque_metadata_regimes_pending=18,opaque_metadata_executions_pending=529,plan_zero_regimes_pending=3,plan_zero_executions_pending=79,
            MemSet_explicit_pending_regimes=7,MemSet_explicit_pending_executions=1272,BOOL_regimes_pending=2,BOOL_executions_pending=80,
            new_remote_source_read=False,local_pinned_operator_clause_review=True,implementation=False,native_test=False,new_performance_result=False,
            all_actual_physical_numeric_ceilings_proved=False,
            remote_job_status='No new remote job/connection;all local saved-profile/pinnedoperator/mathonly. No model/Torch/NPU/capture/health/process/kill/implementation/backgroundanalysis.'),
        changes=['586M21regimes67631TASKtypedfieldreviewwith2792CP/2601coord/208q-r/104cos+104sin/69recount/104headergroups.22conditionaloperatorchoicesandallnativeidentitycountsconserved.',
            'TypedINT64flat/INT32pair,scalarFill1/sourcecopyvaluations,CSrowlookup/SWArawcount-clamp/legacy-packedmaskandexistingv39fusedbuilderconstraintsquantified;config0/1/2scopewordingcorrected.',
            '1831/276286reviewed,251/10664pendingincluding242M21/9312+7MemSet/1272+2BOOL/80. M21all18familiesretained8extended,other25unchanged;globaluncertainty/0wholemoduleclosurepreserved.'],
        candidate_updates=updates,framework_updates={},module_updates=mods,semantic_branches=inv['modules'],remaining=remaining,action=action,
        jobs=jobs,report=report,read_first=[e[8],e[0],e[1],e[2],e[3],e[4],e[5],e[6],e[7],e[10]],publication_rows={},gate_updates=dict(overall_complete=False),
        scripts=[N/x for x in ['analyze_metadata_fields_v68.py','model_metadata_fields_v68.py','review_metadata_headers_v68.py','reconcile_existing_metadata_builder_v68.py','consolidate_metadata_fields_v68.py','verify_metadata_fields_v68.py']],
        executor_prefix='zcode-metadata-fields',main_script=Path(__file__),source_read=False)

if __name__=='__main__':main()
