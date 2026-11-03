"""Seal HC/index-publication and HC/partial-attention relations once."""
from pathlib import Path
from copy import deepcopy
from finalize_origin_coverage_v9 import N,O,read,write
from persist_profile_refinement_v90 import persist_refinement


def main():
    assert not(O/'REVIEW-CHECKPOINT-v138.json').exists()
    pre=read(O/'hc-sparse-preflight-v138.json');assert pre['status']=='PASS'and pre['checks']>42000 and pre['source_positive_shared_TopK_views']==720
    names=['hc-sparse-inputs-v138.json','hc-sparse-eager-native-frontiers-v138.json','hc-sparse-shared-topk-views-v138.json','hc-sparse-source-and-join-contract-v138.json',
        'hc-sparse-fork-and-join-DAG-models-v138.json','hc-sparse-partial-state-information-witnesses-v138.json','hc-sparse-publication-and-work-cuts-v138.json','hc-sparse-composition-clauses-v138.json']
    comp=read(O/'hc-sparse-composition-clauses-v138.json');detail=dict(evidence=names,eager_HC_sparse_frontiers=960,CPU_forward_roots=24,source_bound_consumer_TASKs=3480,
        QLI_prepare_query_quant_chains=192,candidate_publish_TASKs=24,source_positive_shared_TopK_view_APIs=720,source_topk_groups=192,source20_candidate_to_later_MODE2_frontiers=96,
        ideal_DAG_cases=1296,repriced_DAG_cases=7776,real_reference_partial_state_cases=96,new_scoped_pair_relations=2,new_pair_clauses=2,new_groups=0,new_main_candidates=0,new_candidate_module_links=0,
        prior_v60_v83_v93_v124_answers_reused=True,actual_selected_ID_payload_bank_pointer_epoch_crossrank_partial_ABI_new_native_precision_or_capacity_bound=False,
        new_native_promotions=0,new_device_work=False,whole_requirement_closed=False)
    def fork(n):return deepcopy(read(O/(n+'-v137.json')))
    def save(n,d):write(O/(n+'-v138.json'),d)
    rel=fork('candidate-relation-clauses')
    for c in comp['clauses']:
        assert not any({r['a'],r['b']}=={c['a'],c['b']}for r in rel['clauses'])
        rel['clauses'].append(dict(id=c['id'],a=c['a'],b=c['b'],relation='HC_QUERY_VS_SHARED_SPARSE_PUBLICATION_READINESS'if c['b']=='F12'else'HC_COEFFICIENT_DEADLINE_AND_PARTIAL_ATTENTION_HEAD_STATE',
            operator_claim=c['contract'],evidence=names,applicable_domain='Savedtargeteager960HC/SP/attentionfrontiersandnormalfallbackSOURCE20candidate/readbankpaths;earlySWAandnewpartialABIareconditionalvariants,notexistingnativebehavior.',
            conditional_compatibility=True,all_runtime_gates_proved=False,speedup_additive=False,critical_path_gain_us=None,HC_sparse_composition_v138=deepcopy(detail)))
    for gid,member in comp['group_extensions']:
        g=next(g for g in rel['higher_order_groups']if g['id']==gid);assert member not in g['members'];g['members'].append(member);g.update(all_subsets_analyzed=False,legal_joint_fusion_proved=False);g['HC_sparse_composition_v138']=dict(**detail,whole_group_closed=False)
    rel.update(parent_relation_index='candidate-relation-clauses-v137.json',new_pair_clauses=2,new_higher_order_groups=0,clause_count=148,selected_pair_count=148,HC_sparse_composition_v138=detail,overall_complete=False);save('candidate-relation-clauses',rel)
    cv=fork('candidate-relation-coverage');nav=fork('candidate-pair-navigation')
    for c in comp['clauses']:
        for d in [cv,nav]:
            r=next(r for r in d['pairs']if {r['a'],r['b']}=={c['a'],c['b']});r.update(whole_pair_all_variants_closed=False,new_critical_path_gain_us=None,HC_sparse_composition_v138=deepcopy(detail))
            if d is cv:r.update(scope_state='SCOPED_OPERATOR_RELATION_REVIEWED_CONDITIONAL',clauses=[c['id']])
            else:r.update(conditional_clause_ids=[c['id']],semantic_pair_reconciliation_done_in_stated_scope=True,evidence=names)
    cv.update(parent='candidate-relation-coverage-v137.json',parent_relation_index='candidate-relation-clauses-v138.json',scoped_conditional_pairs=148,remaining_pair_relations=447,HC_sparse_composition_v138=detail,
        scope='148scopedconditionalpairs/26groups/447pending;HCquery/candidatebankandpartialattentionjointstate/deadlineconstraintsreviewed;allruntime/allvariants/wholemodelremainopen.',overall_complete=False);save('candidate-relation-coverage',cv)
    nav.update(parent='candidate-pair-navigation-v137.json',parent_footprints='candidate-operator-footprints-v138.json',parent_relation_index='candidate-relation-clauses-v138.json',scoped_conditional_pairs=148,remaining_central_pair_reconciliation=447,HC_sparse_composition_v138=detail);save('candidate-pair-navigation',nav)
    touched=['F01','F12','F13','F14','F15','F31','F32'];fp=fork('candidate-operator-footprints')
    for r in fp['candidates']:
        if r['id']in touched:r['HC_sparse_composition_v138']=dict(evidence=names,no_new_native_operator_atoms=True,local_graph='hc-sparse-composition-clauses-v138.json',all_runtime_or_all_native_partial_ABI_gates_bound=False)
    fp['parent']='candidate-operator-footprints-v137.json';save('candidate-operator-footprints',fp)
    op=fork('operator-analysis-coverage-ledger');op['HC_sparse_composition_v138']=detail;save('operator-analysis-coverage-ledger',op)
    obligations=fork('coverage-current-obligations')
    for r in obligations['obligations']:
        if r['id']in ['G00','G01','G02','G05','G06','G07','G08','G09','G10','G11','G13']:r['HC_sparse_composition_v138']=deepcopy(detail)
        if r['id']=='G07':r['remaining']='148conditionalpairs/26groups/447centralpairsandallvariantsopen. HCearlyY/QLI/partialSWA-CMP/Postmustuseonesharedgraph. Candidate membershipcanhoistafterexactsource20epochbutcurrentqueryscoresstillneeded;720source-positiveTopKviewskeep192producerbanklifetimes. Splitattentionneedscompatibleheadnormalizers/numerators,sinkonce,occurrencemultiplicity/order/nativeBF16exp/PV/output/ropeandheadmixboundaries. Statecontents/HOL/contention/livenessnotfree;HCcoeff/Post/pre4/rawXremain.'
        if r['id']=='G10':r['remaining']='148conditionalrelations/26groups/447pending. HG93addsF12/F13butallsubsets/nativelegalityopen. 1296idealjointDAGsshow64synergy/22overlappingcredits/1210case-specificadditivity;notindependentspeedups. 7776repricedhypotheticalcasesincludebothwinsandlosses. No physical wholemodelbound.'
        if r['id']=='G13':r['remaining']=(
            'v138F01-F12/F13bind960eagerHC/SPtoSMLA/inverseRoPE/layout,192QLI/prepare/queryquantand24candidatecopies=3480TASKs;720CP206sharedTopKsliceAPIsin192producergroupspositivelybindlaterreusebranches. '
            '96source20candidate→consumer24/28/32/36rolejoinspreserveepoch/queryrowmap/visibility;currentconsumerQLIcannotreusepastqueryscores. ProducerpreparefusionstillpublishesalllaterTopKreaders;API-onlyselfcopiesnotzero-costproof. '
            'EarlySWAneedsperheadpartialstate:normalizedBF16outputs+disabledLSEinsufficient;projectedheadmixedpartialsloseinformationevenwithperheadmasses. PreserveBF16tile/nativeorder/occurrences/sinkonceandactualinverseRoPE/headexchange/Oprojection/RS/HCcoeff/Post/pre4. '
            '1296ideal+7776repricedDAGs/96realreferencecasesarenotnativeprecision/gain. Partialstateformatcontentscanbelarge;actualworkspace/lifetime/reuseandfiniteprogressunknown,persistentwaitmayrequireverifiedresourcepartition/yield. '
            'Next pending F01-F09/F10:current index query/weight/FP16scale producers and auxiliary-stream quantization versus HCcoeff/Y;reuse192quantpositivechainsandv60/v83typednumericfacts. Do notinferpriority/resource-overlapfromhostenqueueorder. '
            '4472savedSPjoinsretained;358semantic26modules/854nodes1346edges48hints/2073regimes285598executionsconditional;9/1352native/849PMU/16phase/3136assignment/0whole/physicalMFU-lowestunknown unchanged. No newNPU/model/capture/replayauthorization;Padv120consumed,nopendingapproval.')
    obligations.update(parent='coverage-current-obligations-v137.json',new_native_or_whole_closures=0,overall_complete=False);save('coverage-current-obligations',obligations)
    audit=fork('whole-scope-requirement-audit');audit.update(previous_goal_turn='PROGRESS:v137HC-poolnorm/state-row-union/sharedbudget;postcheckPASS74826',input_checkpoint=137,followup_checkpoint=138,HC_sparse_composition_v138=detail)
    for r in audit['requirements']:
        if r['id']=='CROSS_OPERATOR_LAYER_CARD_ITERATION':r.update(conditional_pair_relations=148,central_pairs_not_reconciled=447,higher_order_groups=26)
        if r['id']in ['ALL_MODULE_CONDITIONAL_BRANCHES','CROSS_OPERATOR_LAYER_CARD_ITERATION','THEORY_AND_MEASURED_COMPARISON','CRITICAL_EVIDENCE_AND_FINAL_CLOSURE']:r.setdefault('evidence',[]).extend(names)
    save('whole-scope-requirement-audit',audit)
    write(O/'PROFILE-CODE-QUESTION-GATE-v138.json',dict(status='PROFILE_BOUND_HC_SMLA_QLI_PUBLISH_REUSE_THEN_DIRECT_PARTIAL_HEAD_STATE_AND_DEADLINE_CLAUSES',questions=read(O/'hc-sparse-inputs-v138.json')['questions'],
        source_scope='OnlypinnedHC/directDSA/indexer/CPheadrestoreclausesandpreviouslycollectedSMLAnativeadapter/onlineSoftmax/roundingpieces. v60/v83/v93/v124answersreused;no framework/loader/scheduler/SDKsurvey,remote/sourcecollection/deviceexecution.',
        no_model_operator_implementation=True,no_remote_Torch_CANN_NPU_model_or_replay=True,Pad_authorization='GRANTED_AND_CONSUMED',overall_complete=False))
    write(O/'hc-sparse-validation-history-v138.json',dict(events=[
        'Predecessorv137manifest108digestsmatched/CURRENTPASS74826andRECOVERY137. PreviousGoalturnclassifiedPROGRESS:twoHC-C2relations/directnormwrapper/rowunion/sharedbudget.',
        'All960eagerHC/SPparentsnowhavesource/CIDnativeSMLA,inverseRoPEandTranspose;192QLI/prepare/queryquantand24source20candidateTensorMove=3480uniqueTASKs. Priorv60rawopswerenotnewrawdeviceobservations;newHC/SPsourcejoinsaretheprogress.',
        '192aclnnInplaceCopyTopKsourceAPIsrecordedwithexactCIDTASKlists;absenceofTASKnotlatency/aliasproof. CandidatecopyhaspositiveTensorMove.',
        'InitialpreflightPASS39314forcorebindings/DAGs. Added720source-positiveCP206sharedTopKviewAPIsand192producer-readergroupsbeforesealingtoanswerlater-readerlifetimegap;extendedpreflightPASS'+str(pre['checks'])+'.',
        'Normal-sourcecandidate20bankcanpreparemembershipbeforelaterconsumerHCunderrow/epoch/resourcegates;720laterbankviewskeepQLIpublicationneeded. ActualIDs/pointers/generation/crossrankepisodesunbound.',
        '1296independentweightedDAGchecks:64synergy,22overlappingcredits,1210case-specificadditivity. 7776repricedcases3716jointfaster/3464slower/596ties;arbitraryhypotheticalservicesandPostINPUTreadyonly,notnativegainorfullmodeltime.',
        '96finiteexact-realpartialstatisticcasesretainconsistentduplicate-IDvalues,sinkonceandemptybranches. Headmixinformationloss/sinkduplication/occurrencededup/resourcewaitcyclewitnessesareconditionalmodels,notactualdevicefaultsornativeBF16parity.',
        'REL138-01/02F01-F12/F13;HG93addsF12/F13.148pairs/26groups/447pending. Broadcandidategaps/native/global/wholestatusretained.'],overall_complete=False))
    gap=(
        'NeedactualselectedIDmultiset/physicalemittedorder/queryrow/candidateTopKbankepoch/pointers/currentmetadataandalllaterreaders,source20logicalmembershipfiniteness/visibility/offset/filler/tiecontracts. '
        'NewpartialSMLAneedscompatiblehead-resolvednormalizers/numerators/sinkonce/emptysemantics/nativeBF16exp/PV/onlinetile/reduction/output/rotary/Oprojectionandallalias/capturecontracts;ordinarymixedprojectionisnotalosslessjoin. '
        'NeedactualCube/Vector/SFU/GM/linkresourcecapacity,partialhandoffworkspace/lifetimes/transportandfiniteprogress,HCcoeff/Y/newpre4/Postdeadlines/currentrawXretirement. '
        'No transformednativeprecision/actualoverlap/physicalMFU/lowest/speedup/allvariants/wholeclosure. Existing9/1352native/849PMU/16phase/3136ambiguityandbroadercandidategapsremain.')
    remaining=[r['id']+': '+r['title']+'. '+r['remaining']for r in obligations['obligations']]
    remaining.append('v138F01-F12/F13:148pairs/26groups/447pending;960HC-sparsefrontiers/3480TASKs/720sharedviews/1296ideal7776repriced96referencecases.4472SPjoins;358/26semantic854/1346/48partialgraph9/1352native849PMU16phase3136ambiguity0wholeunchanged.')
    action=(
        'ContinueoriginalGoal;v138PROGRESSisF01-F12sparsepublicationandF01-F13partialattention/HCdeadlinecomposition. Readhc-sparse-summary/native-frontiers/shared-topk-views/source/DAGs/partial-witnesses/workcuts/composition/currentobligations-v138;verify_hc_sparse_relation_v138mustPASS. '
        '960eagerHC/SPparents→960SMLA+inverseRoPE+Transpose,192QLI/prepare/queryquant,24source20candidateTensorMove=3480TASKs.192TopKselfcopyAPIswithexactCIDTASKlists;noTASKnotzero-costclaim.720CP206sliceAPIsbindlaterTopKbankreusein192producergroups;sourceSMLAcompletiondoesnotretirelaterbankreaders. '
        'Source20candidatebank96laterMODE2QLIsmaypreparepuremembershipbeforeconsumerHCYunderexactrow/epoch/universe/resourcegates. Currentconsumerquery/score/TopKstillneeded;QLIINT8scoresnotSMLABF16scores. Preservechronologicalvalidprefix,-1suffix,duplicateoccurrences,filler/ties/offsets/allstablebankreaders. '
        'EarlySWAversusCMPisnewconditionalpartialstateABI. NativeintegratedSMLAreturnsBF16outputandemptyLSE;normalizedoutputslosemass. Perhead(m,l,a)chosenFP32referenceformatnotnativeparity;keeporiginalBF16exp/PV/update/order/rounds/sinkonce/emptysemantics/physicalgatheroccurrences. HeadmixedOpartialslosenormalizationinformationevenwithperheadmasses:mergebeforeheadmixorcarrypricedhead-resolvedcontributions. Headexchangepermutesonlyunderrow/head/epochandnewdatatypecollectiveABI. '
        '1296idealDAGcases64synergy22overlappingcredit1210case-specificadditivity;7776repriced3716faster3464slower596ties. Hypotheticalservice,HCPostINPUTreadyendpoint,commonPostserviceomittedandlaterpre4/FFNnotpriced;noactualgain. Currentfull4Hcoeff/postcomb/pre4/rawXPostremain. Sharedresources/stateinterface/finiteprogressnotfree;fulloccupancywaitonQLImaycycleunlessverifiedyield/reservation/preemption/jointscheduler. '
        'PartialstateFP32(D512+2)*64*Mreferenceoneplane;notmandatorynewallocation/HBM,oldGMworkspacereuse/lifetimeunbound. OnechosenHC/QLI/SMLA/communicationgraphandsharedcapacity/warmgrant;noaddingoldlaunch/interface/bankcredits. '
        'REL138-01/02F01-F12/F13;HG93addsF12/F13.148conditionalpairs26groups447pending;358/26semantic854/1346/48partialgraph2073/285598conditional9/1352pending849PMU16phase3136ambiguity0whole/noallmodulephysicalMFU/lowestunchanged. Broadercandidategapsretainedwithscopedreview.4472SPjoinsremain. '
        'NextpendingF01-F09/F10currentindexquery/weight/FP16scaleproducerandauxstreamquantizationversusHCcoeff/Y;reuse192positivequantfrontiersandv60/v83typednumericfacts. Hostenqueueordernotpriorityoroverlapproof. '
        'Alllocaljobscomplete,no remote/Torch/CANN/NPU/model/capture/replay/implementation. Padv120consumed,nopendingapproval. Do noteditsealedv138/rerunfinalizer. Versionedpostcheck/CURRENTmustmatchRECOVERY;TaskACTIVE Loop004EVALUATING GoalACTIVE.')
    report='''本轮补齐 F01–F12、F01–F13：HC 分支与稀疏索引发布、SWA/压缩注意力拆分后的共同调度。固定profile优先，随后只核对直接算子调用和已保存的native语义源码。整体目标仍未完成。

960组eager HC/SP前沿现在关联到SMLA、逆RoPE和head发送布局转换；另有192组QLI→索引整理及query量化、24次source20候选块TensorMove发布，共3480个原生TASK。这是将既有原始profile中的TASK精确关联到HC/SP前沿，不是新增设备测量或原生字段关闭。192次TopK self-copy源码API保留精确CID TASK列表，没有TASK不能据此认定成本为零。

为核对融合后的索引生命周期，又补了720次CP第206行shared.topk_indices切片API，归入192个生产者组。source2/8/14的TopK分别供后续5层复用，source20/24/28/32/36分别供后续3层复用。因此，在生产层把QLI、整理和当前SMLA合成一个kernel，也不能删除后续层仍要读取的稳定TopK发布。这里是源码分支和原生消费者的关联，实际bank地址、ID内容、世代和跨rank episode仍需证据。

source20还发布候选块，供source24、28、32、36的MODE2 QLI使用，共96个后续前沿。候选membership只依赖对应source20结果与逻辑行/可见域/epoch，可在资源允许时提前构造；它不必等待未来consumer HC生成新的Y。但未来consumer的query、QLI分数和位置TopK仍依赖当前层输入，不能被这个membership替代。QLI的INT8/DEQF16/ReLU/headreduce分数也不是SMLA的BF16 attention分数。QLI epilogue融合整理时，仍要保留有效前缀、时间序、-1后缀、重复出现次数、fillers/ties和所有候选/TopK读者。

SWA数据路径本身不依赖压缩索引，可以研究在QLI或压缩分支就绪前先执行。但当前SMLA已经在一个kernel中处理SWA+compressed KV、online softmax、sink与PV，调用方关闭LSE输出。把它拆成两份普通BF16 normalized输出后相加或拼接，不能恢复共同归一化。必须提供兼容的归一化质量和分子，例如声明域内的per-head(m,l,a)状态，并核对空分支、sink只计一次、原始出现多重集合、mask与实际gather顺序。

96个实数参考用例核对了状态合并，保留同一plane内重复ID对应的相同值；另有sink重复计算和错误去重的反例。这些证明信息与数学边界，不是native BF16等价性。原kernel的BF16指数/PV输入转换、online更新、tile顺序和最终BF16舍入仍需保留或另立数值合同。仅启用LSE或返回BF16 partial output，也不能自动恢复原先未舍入的信息。

合并点还受head维度限制：不同head具有不同归一化质量。反例中，两种partial状态有相同的O投影结果、相同的per-head质量，正确最终投影却分别为0.5和0.75，说明普通混合后的O投影partial已经丢失head贡献。通常应在head混合前合并，或继续携带足够的head级投影贡献并计价。head恢复通信只是特定行/head映射下的置换；若提前交换partial状态，还需新的dtype、布局、rank/epoch和collective/capture合同。原逆RoPE、O投影数值与归约边界不能因实数线性公式而自动移动。

HC与SWA提前执行必须按同一张图估算。1296组理想同服务模型中，64组出现协同、22组收益重叠、1210组在该具体用例中可相加。例如，单独提前HC的Y可能仍被外部QLI期限挡住；与SWA先执行组合后，之前没有显示的重叠才有价值。反过来，当前HC系数未就绪时，更早结束attention也可能不缩短Post入口等待。模型终点是HCPost输入就绪，省略共同Post服务，后续pre4/FFN没有计价。

再加入假设的state/launch/merge服务及HC系数竞争后，7776组模型同时出现较快、较慢和相同结果。这些数字仅说明独立收益不能相加、额外成本能改变结论，不是实际负载胜率或硬件最优时间。真实Cube/Vector/SFU/访存/链路容量、队列和资源竞争尚未绑定。当前完整4H系数、post/comb、下一FFN所需pre4与Post原始X残差仍然存在。

partial状态交付也有代价。在声明的FP32(m,l,a512)格式中，一个head有514个FLOAT值；M=800、64 heads时，一份值内容为105267200B，相比原BF16输出52428800B更大。它可能复用原online GM workspace，也可能需要新增bank或写读，不能把所有内容直接当作额外分配或HBM流量，更不能视为免费重叠。持久SWA kernel若一边占满QLI需要的资源、一边等待QLI发布，还会在无抢占的条件模型中形成等待环；必须通过已验证的资源保留、yield/抢占、launch边界或联合调度保证有限进展。这里没有断言当前设备发生死锁。

新增REL138-01/02，条件关系146→148、待核对449→447；HG93加入F12、F13，联合组仍26，全部子集与联合native合法性保持开放。35主候选、7历史框架记录、358语义行/26模块、4472次固定SP前沿关联和854节点/1346边/48提示的部分图保持。9类/1352次原生证据、849个PMU缺口、16个phase未绑定图、3136项归属歧义仍未补齐，整模块关闭0，全部模块物理MFU及最低时间仍未建立。
'''
    updates={cid:dict(review_v138=dict(evidence=names,pairs=comp['new_pairs'],source_bound_TASKs=3480,source_positive_shared_TopK_views=720,
        scoped_conclusion_changing_gap=gap,existing_broad_conclusion_gap_preserved=True,one_HC_sparse_baseline=True,no_native_fusion_or_physical_gain_proved=True))for cid in touched}
    inputs=[Path(r['path'])for r in read(O/'hc-sparse-inputs-v138.json')['inputs']];refs=read(O/'hc-sparse-eager-native-frontiers-v138.json')['databases'];inputs += [Path(r['raw_database']['path'])for r in refs]+[Path(r['cached_native']['path'])for r in refs]
    persist_refinement(version=138,previous=137,title='HC sparse-index publication and partial attention head-state/deadline composition',
        summary=dict(filename='hc-sparse-summary-v138.json',previous_turn_classification='PROGRESS:v137HCpoolnorm/state-row-union/sharedbudget;PASS74826',**detail,
            refined_candidates=7,refined_modules=8,branch_records=358,semantic_modules=26,relation_clauses=148,higher_order_groups=26,pending_central_pair_reconciliation=447,
            partial_graph_node_rows=854,partial_graph_edge_rows=1346,structural_pair_hints=48,scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,pending_regime_reconciliation=9,pending_execution_reconciliation=1352,
            model_operator_implementation=False,native_test=False,actual_MFU_numeric_physical_ceiling_or_speedup_proved=False,Pad_authorization='GRANTED_AND_CONSUMED',remote_job_status='All local saved-profile/source/partial-state/DAG work complete;no remote/device/Torch/CANN/model/replay/service work.'),
        changes=['Bind960HC/SPattentionfrontiers/3480TASKsand720positiveCPsharedTopKviewAPIs;retain192producerbankreadergroupsand24candidatepublishes.',
            'ReconcileHCfork/QLIpublication/partialSWA-CMPwithper-headnormalizationinformation,sink/multiset/rounding/headmix/resource/livenessandcurrentcoeff/Postdeadlines.',
            'AddREL138-01/02F01-F12/F13;148pairs/26groups/447pending;allnative/global/wholeuncertaintyandbroadercandidategapspreserved.'],candidate_updates=updates,
        module_updates={mid:dict(candidates=[],finding=f,evidence=names,gap=gap)for mid,f in {
            'M06':'HCcoeff/YforkandPost/pre4/rawXdeadlinescomposewithQLI/partialattentioninonegraph;idealindependentgainsshowbothsynergyandoverlap.',
            'M07':'192queryquantnativefrontiersbindcurrentquerydependency;candidate membershipdoesnotremovequeryproductionorQLIscoremath.',
            'M09':'192QLI/prepareand24candidateTensorMovesplus720sharedTopKviewskeepcurrent/laterstablebankpublicationneeded;puremembershipcanhoistunderexactepochgates.',
            'M10':'960SMLAfrontiersgroundearlySWA/CMPpartialstatealternatives;normalizedBF16andemptyLSEinsufficient,originalsink/multiplicity/order/roundingremain.',
            'M11':'960inverseRoPE/layoutfrontierskeeptailround/headexchange/Oprojection/RScontinuation;headmixedpartialsloseperheadnormalizationinformation.',
            'M21':'Selectedcandidate/logicalrow/visibleuniverse/epochpreparationcanprecedelaterconsumerHCY;currentqueryTopKandactualbank/offsetmetadataremain.',
            'M22':'Partialpublicationrequiresverifiedper-rowrelease/acquire/capturelayout/ownershipandfiniteprogress;fullresourcepersistentwaitmayneednewreservation/yieldcontract.',
            'M23':'Perheadstatecrossrankmovement/newdtype/layoutandonejointCube/Vector/linkbudgetpriced;logicalpartialstatecontentsnotHBM/allocation/wirelowerbound.'}.items()},
        remaining=remaining,action=action,jobs=[
            dict(kind='profile',status='pass',description='960HC/SP→3480source/CIDnativeTASKs,192TopKcopyAPIs,720CP206sharedviews/192readergroups;rawsource/API/TASK/Compute/order/dtypefieldschecked.',artifacts=[O/'hc-sparse-eager-native-frontiers-v138.json',O/'hc-sparse-shared-topk-views-v138.json']+inputs),
            dict(kind='design-check',status='pass',description='1296ideal/7776repricedweightedDAGs,96realpartialstatecases/headmix/sink/multiplicity/livenesswitnessesand'+str(pre['checks'])+'extendedpreflightchecksPASS;twoconditionalHC-sparserelations.',artifacts=[O/'hc-sparse-preflight-v138.json',O/'hc-sparse-validation-history-v138.json']+[O/n for n in names]+[O/'coverage-current-obligations-v138.json',O/'whole-scope-requirement-audit-v138.json'])],
        report=report,read_first=names+['coverage-current-obligations-v138.json'],publication_rows={},gate_updates=dict(overall_complete=False),
        scripts=[N/'analyze_hc_sparse_native_v138.py',N/'analyze_hc_sparse_shared_views_v138.py',N/'analyze_hc_sparse_relation_v138.py',N/'verify_hc_sparse_relation_v138.py',N/'persist_profile_refinement_v90.py'],executor_prefix='hc-sparse',main_script=Path(__file__),source_read=True)


if __name__=='__main__':main()
