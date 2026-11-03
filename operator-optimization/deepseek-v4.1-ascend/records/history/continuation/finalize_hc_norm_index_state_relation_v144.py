"""Seal scoped input-normalizer/Indexer/C2 pair/state composition once."""
from pathlib import Path
from copy import deepcopy
from finalize_origin_coverage_v9 import N,O,read,write
from persist_profile_refinement_v90 import persist_refinement


def main():
    assert not(O/'REVIEW-CHECKPOINT-v144.json').exists()
    pre=read(O/'hc-norm-index-state-preflight-v144.json');assert pre['status']=='PASS'and pre['checks']>50000
    names=['hc-norm-index-state-inputs-v144.json','hc-norm-index-state-native-joins-v144.json','hc-norm-index-state-source-contract-v144.json','hc-norm-index-state-observed-v144.json','hc-norm-index-state-pool-and-precision-witnesses-v144.json','hc-norm-index-state-pair-algebra-v144.json','hc-norm-index-state-readiness-DAGs-v144.json','hc-norm-index-state-work-and-lifetimes-v144.json','hc-norm-index-state-composition-v144.json'];comp=read(O/'hc-norm-index-state-composition-v144.json')
    detail=dict(evidence=names,source_join_frontiers=192,reused_native_TASK_union=5064,C2_frontiers=72,C1_frontiers=24,Index_query_weight_only_frontiers=96,pair_reference_cases=784,complement_changed_cases=640,reference_DAG_cases=6144,chosen_future_deadline_violations=855,nominal_faster_but_future_violation_cases=665,new_scoped_pair_relations=6,new_pair_clauses=6,new_groups=0,new_main_candidates=0,new_candidate_module_links=0,new_native_promotions=0,new_device_work=False,whole_requirement_closed=False)
    def fork(n):return deepcopy(read(O/(n+'-v143.json')))
    def save(n,d):write(O/(n+'-v144.json'),d)
    rel=fork('candidate-relation-clauses')
    relations={'F09':'INPUT_RMS_TO_TYPED_QR_INDEX_QUERY','F10':'INPUT_RMS_QA_WEIGHT_PANEL_AND_PRODUCT','F11':'INPUT_RMS_LATENT_INDEX_KEY_DISTINCT_NORMS','F17':'INPUT_RMS_BF16_TO_C2_FLOAT_JOINT_PROJECTION','F18':'INPUT_RMS_C2_PAIR_SOFTMAX_NORM_PRECISION','F19':'INPUT_PROJECTED_LATENT_RING_DISTINCT_RETIREMENT'}
    for c in comp['clauses']:
        assert not any({r['a'],r['b']}=={c['a'],c['b']}for r in rel['clauses'])
        rel['clauses'].append(dict(id=c['id'],a=c['a'],b=c['b'],relation=relations[c['b']],operator_claim=c['contract'],evidence=names,applicable_domain='192savedeagertypedIndex/C1/C2sourcekeyjoins5064priorTASKs:72C2/24C1/96query-weightonly. Newmixedproducer/pair-specialization/statehandoff andallnativeprecision/epochs/readersremainconditional.',conditional_compatibility=True,all_runtime_gates_proved=False,speedup_additive=False,critical_path_gain_us=None,HC_norm_index_state_v144=deepcopy(detail)))
    for gid,m in comp['group_extensions']:
        g=next(g for g in rel['higher_order_groups']if g['id']==gid);assert m not in g['members'];g['members'].append(m);g.update(all_subsets_analyzed=False,legal_joint_fusion_proved=False);g['HC_norm_index_state_v144']=dict(**detail,whole_group_closed=False)
    rel.update(parent_relation_index='candidate-relation-clauses-v143.json',new_pair_clauses=6,new_higher_order_groups=0,clause_count=173,selected_pair_count=173,HC_norm_index_state_v144=detail,overall_complete=False);save('candidate-relation-clauses',rel)
    cv=fork('candidate-relation-coverage');nav=fork('candidate-pair-navigation')
    for c in comp['clauses']:
        for d in [cv,nav]:
            r=next(r for r in d['pairs']if {r['a'],r['b']}=={c['a'],c['b']});r.update(whole_pair_all_variants_closed=False,new_critical_path_gain_us=None,HC_norm_index_state_v144=deepcopy(detail))
            if d is cv:r.update(scope_state='SCOPED_OPERATOR_RELATION_REVIEWED_CONDITIONAL',clauses=[c['id']])
            else:r.update(conditional_clause_ids=[c['id']],semantic_pair_reconciliation_done_in_stated_scope=True,evidence=names)
    cv.update(parent='candidate-relation-coverage-v143.json',parent_relation_index='candidate-relation-clauses-v144.json',scoped_conditional_pairs=173,remaining_pair_relations=422,HC_norm_index_state_v144=detail,scope='173scopedconditionalpairs/26groups/422pending. InputRMS/Indextypedconsumers,C2pairalgebraanddistinctZ/projected/ringretirementreviewed;native/wholepair/allsubsetclosureopen.',overall_complete=False);save('candidate-relation-coverage',cv)
    nav.update(parent='candidate-pair-navigation-v143.json',parent_footprints='candidate-operator-footprints-v144.json',parent_relation_index='candidate-relation-clauses-v144.json',scoped_conditional_pairs=173,remaining_central_pair_reconciliation=422,HC_norm_index_state_v144=detail);save('candidate-pair-navigation',nav)
    touched=['F01','F02','F05','F09','F10','F11','F17','F18','F19','F34'];fp=fork('candidate-operator-footprints')
    for r in fp['candidates']:
        if r['id']in touched:r['HC_norm_index_state_v144']=dict(evidence=names,no_new_native_operator_atoms=True,local_graph='hc-norm-index-state-composition-v144.json',all_runtime_or_new_native_fusion_gates_bound=False)
    fp['parent']='candidate-operator-footprints-v143.json';save('candidate-operator-footprints',fp)
    op=fork('operator-analysis-coverage-ledger');op['HC_norm_index_state_v144']=detail;save('operator-analysis-coverage-ledger',op)
    gap=('NeedactualZ/QR/latentinputrows,gamma/beta/epsilon/weightgenerations,nativeBF16/FLOAT/HF32/quant/FP16productpolicies,andmixedBF16/FLOATproducerABI/strides/VectorCubecapacity. '
         'C2RATIO2oneexp/twoprobabilitiesisafinitererefencevariant,notattestednativeSFUgain;needcompiledexp0folding/opcodes/latencyandexp0/FTZ/ties/NaNInf/reduction/nativeBF16poolparity. Neveruse1-pmaxunconditionally. '
         'KeepfullTzero/noncompletion/norm/bias/rstd/IndexKeyLongslot/CSoldvalueandfutureFP32ringobligations. Z/FLOATinput/projectedKVScorehaveseparatedirectlastreaders;actualaliases/capture/queuedwriters/memoryfitunbound. '
         'EarlyC2/normforkrequiresmetadata/old-ringgenerationandstatecommitbeforeactualnextmatchingreader;6144DAGs855futureviolationsarehypothetical,notactualiterationbinding. '
         'Cast/jointmatrix/inputnorm/panel/statecreditoneunion,notadditive. NoactualE/B/U/newnativeprecision/physicalcapacity/MFU/lowest/actualgain/allvariantsclosure;9/1352native849PMU16phase3136assignmentsandbroadercandidate gaps retained.')
    obligations=fork('coverage-current-obligations')
    for r in obligations['obligations']:
        if r['id']in ['G00','G01','G02','G05','G06','G07','G08','G09','G10','G11','G13']:r['HC_norm_index_state_v144']=deepcopy(detail)
        if r['id']=='G07':r['remaining']='173conditionalpairs/26groups/422pending.F02-F09/F10/F11/F17/F18/F19typedInputRMS/Index/C2poolandstatecompositionreviewed. '+gap
        if r['id']=='G10':r['remaining']='173conditionalpairs/26groups/422pending;HG73C2addsF02/F09/F10/F11,HG87commitaddsF09/F10/F17.6144arbitraryDAGs4128faster1792slower224ties;855futuredeadlineviolationsincluding665nominalfaster,notadmissiblefixed-futuregain. Allsubsets/nativelegalityopen.'
        if r['id']=='G13':r['remaining']=('v144192typedsourcejoins5064priorTASKs(72C2/24C1/96Indexonly),no newnativebindingsorclasspromotions. IndexQusesQR1280,WeightusesZ5120;Key128andQuery128quantdifferent. '
            'C2commoninputscalarchangessoftmaxtemperature/direction;downstreamRMSnotcancellation.784chosenfiniteFLOATpaircasesoneexppreservingtwoprobabilitiesmatchesreference,640complementchanges;nativecompiler/exp0/FTZ/NaNInf/tie/opcodeproofopen. '
            'LogicalFLOATcastplane12THold;jointexistingplanesaves4THandremaining8TH,cast+jointfullunion12THnotplus4TH. MixedQAWeight1312BF16/SWA512BF16/C21024FLOATneedsnewtypedkernel/rowmap,notdowngradedweight. '
            'Zlastdirectreader/FLOATtwoGEMMreads/projectedpoolANDringread/latentoldIndexreadandfuturegenerationdifferent. Earlycurrentendpointdoesnotretireprojectedtail. Fullvalidcurrentprojectionrowunionandringcapacityremain. '
            'NextpendingF02-F12/F13/F14/F15/F16inputnormalizer-to-indexselection/sparseattention/output/RS composition;reusev138/v140/rankfrontiers,actualhead/row/score/numerical/cache deadlines,profilefirst. '
            '358semantic26modules854nodes1346edges48hints2073regimes285598conditional9classes1352native849PMU16phase3136assignments0whole/physicalMFU-lowestunknown. No newNPU/model/capture/replayauthorization;Padv120consumed,nopendingapproval.')
    obligations.update(parent='coverage-current-obligations-v143.json',new_native_or_whole_closures=0,overall_complete=False);save('coverage-current-obligations',obligations)
    audit=fork('whole-scope-requirement-audit');audit.update(previous_goal_turn='PROGRESS:v143inputRMS-QKV5relations/10824TASKs;postcheckPASS83640',input_checkpoint=143,followup_checkpoint=144,HC_norm_index_state_v144=detail)
    for r in audit['requirements']:
        if r['id']=='CROSS_OPERATOR_LAYER_CARD_ITERATION':r.update(conditional_pair_relations=173,central_pairs_not_reconciled=422,higher_order_groups=26)
        if r['id']in ['ALL_MODULE_CONDITIONAL_BRANCHES','CROSS_OPERATOR_LAYER_CARD_ITERATION','THEORY_AND_MEASURED_COMPARISON','CRITICAL_EVIDENCE_AND_FINAL_CLOSURE']:r.setdefault('evidence',[]).extend(names)
    save('whole-scope-requirement-audit',audit)
    write(O/'PROFILE-CODE-QUESTION-GATE-v144.json',dict(status='PROFILE_TYPED_INPUT_INDEX_C2_JOIN_THEN_PAIR_STATE_RETIREMENT',questions=read(O/'hc-norm-index-state-inputs-v144.json')['questions'],source_scope='OnlydirectHC/inputNorm/IndexQ-Weight-Key/C2callerandRATIO2pool/update clauses;reusepriorv106/v135-v143answers. No framework/SDKsurvey/source download/targetchange.',no_model_operator_implementation=True,no_remote_Torch_CANN_NPU_model_or_replay=True,Pad_authorization='GRANTED_AND_CONSUMED',overall_complete=False))
    write(O/'hc-norm-index-state-validation-history-v144.json',dict(events=[
        'v143authority/CURRENT-PASS83640/RECOVERY143and114manifestfilesmatched;previousGoalturnPROGRESS.',
        '192priornativefrontierunion5064uniqueTASKsrawAPI/source/CID/Computechecked;72C2/24C1/96Indexonly. C2all72Query/IndexRoPEbeforecast andringbeforepoolNorm;66IndexQuant-C2KVoverlap areobservationsnotsemanticbarriers/gain.',
        'InputRMSZBF16,QR1280nativeINT8/FLOAT,IndexWeightBF16/FLOAT/FP16andC2ZFLOAT/twoFLOATweightsremainseparate. KeynativeFLOATscalevsQueryFP16-beforeINT8retained.',
        'C2scalar/pooldirection/oldhistoryandFLOATweightdowngradecounterexamples. Ratio2oneexpfiniteFLOAT784casesmatchesreference;640caseschangedbycomplement. SmallpmaxroundingcounterexampleBF16 .2060546875vs0.',
        '6144sufficientDAGs4128faster1792slower224ties;855fixedfutureviolationsincluding665nominalfaster. Z/FLOAT/projectedplane/ringretirementdifferent,notactualiterationdeadlineclaim.',
        'FLOATinputold12TH=4THwrite+8THreads;jointreadcredit4TH/remaining8TH/fullcastjointunion12TH;noadditivecreditsorphysicalHBMclaims. Fullprojectionrowunion/futureFP32tailsstillrequired.',
        'PreflightPASS'+str(pre['checks'])+';REL144-01..06,173pairs26groups422pending. Allbroadercandidate/native/global gapsretained.'],overall_complete=False))
    remaining=[r['id']+': '+r['title']+'. '+r['remaining']for r in obligations['obligations']]
    remaining.append('v144173conditionalpairs/26groups/422pending;192joins5064priorTASK/784pairrefs640complementchanges/6144DAG855futureviolations.4472SPjoins358/26semantic854/1346/48partialgraph9/1352native849PMU16phase3136ambiguity0whole unchanged.')
    action=('ContinueoriginalGoal;v144PROGRESSF02-F09/F10/F11/F17/F18/F19.Readhc-norm-index-state-summary/native-joins/source/observed/pool-precision-witnesses/pair-algebra/DAGs/worklifetimes/composition/currentobligations-v144;verify_hc_norm_index_state_relation_v144mustPASS. '
        '192priorIndex/C1C2sourcekeyjoins5064uniqueTASKsrawverified:72C2/24C1/96IndexqueryWeightonly,no newnativebindings/classes. All72QueryandIndexRoPEbeforeC2cast/ringupdatebeforepoolnorm,66IndexQuantintersectsC2KV;observednotrequiredbarriers/gain. '
        'InputRMSZroundedBF16H5120;QA/WeightuseZ,IndexQusesQR1280norm/nativeDQ;C2castsZtoFLOATandusesFLOATKV/gateweights. NewQAWeight1312BF16/SWA512BF16/C21024FLOATpanelsneedmixedmode/row/stride/handoff,notoneordinaryBF16matrix. Keepallnativequant/scale/round/gamma/beta/readers. '
        'C2perfeaturesoftmaxscoreschangedbyinputscalar;followingRMScannotrestoredirection,old-ring/currentfactorsseparate. Ratio2oneexp(min-max)withtwoseparateprobsandoriginalroworderisfiniteFLOATreferenceonly:784matches,640complementchanges. pmax1/pmin2.061e-9 givesBF16 .2060546875vs1-pmax0. Needactualcompilerexp0folding/instructions/SFU/nativeFTZ/NaNInf/ties/roundpolicy;notclaimactualspeedup. '
        'Zdirectlastread,FLOATtwoGEMMinputread,projectedKVScorepoolANDringread,latentIndexWkoldtailread/futureFP32statearedifferentfrontiers. Earlynorm/Index/Longneedsretainprojectedtailorinlineexactstate/newrecomputecontract;statecommitstillbeforematchingfutureoldreader. FullTcompletion-zero/norm/bias/rstd/slot/CS andv137allvalidprojectionrowunionretained. HCrawXcoeff/Post/pre4separate. '
        '6144arbitraryC2DAGs4128faster1792slower224ties,855fixedfutureviolationsincluding665nominalfaster. Notadmissiblefixedfuturegains/nativeiterationevidence/physicalbound. FLOATcastplane12THold;jointexistingplanesaves4TH,remaining8TH;fullcastjointunion12THnotplus4TH.7744Btaggedallpanelrowexcludesmetadata/headers/halosanddoesnotprovenewABI/fit;logicalnotHBM/allocation/time. '
        'REL144-01..06;HG73C2addsF02/F09/F10/F11,HG87commitaddsF09/F10/F17.173conditionalpairs26groups422pending;F01star34scopednotwhole.358/26semantic854/1346/48partialgraph2073/285598conditional9/1352native849PMU16phase3136ambiguity0whole/noallmodulephysicalMFU-lowest unchanged;broadercandidate gapsretained. '
        'NextpendingF02-F12/F13/F14/F15/F16inputnormalizer-indexselection/sparseattention/output/RS;reuseexistingv138/v140/rankfrontierswithactualhead/row/score/numeric/cachedeadlines,profilefirst. Allotherpendingpairsremain. '
        'Alllocalworkcomplete,no remote/device/Torch/CANN/model/capture/replay/implementation.Padv120grantconsumed,nopendingapproval.Do noteditsealedv144/rerunfinalizer;versionedpostcheck/CURRENTmustmatchRECOVERY.TaskACTIVE Loop004EVALUATING GoalACTIVE.')
    report='''本轮补齐 F02–F09/F10/F11/F17/F18/F19：输入归一化与 Indexer、C2 双投影、池化、latent 分叉及未来 ring 状态的条件组合。整体目标仍未完成。

192组已有source-key前沿合并后，复核5064个唯一TASK：72组C2、24组C1、96组仅Index query/Weight。这是既有原生证据的组合核对，没有新增原生归属或类别关闭。当前72组都在Query/Index RoPE之后才开始C2输入cast，并在ring update之后做pool norm；66组Index量化与C2 KV TASK区间相交。这些是当前调度观察，不是不可消除的依赖或可恢复时延。

输入不能混为一谈：QA/IndexWeight读取5120维的归一化BF16 Z；IndexQ读取另外归一化过的1280维QR，经native量化后进入INT8矩阵乘；C2先把已经BF16舍入的Z扩展成FLOAT，再用两个FLOAT权重计算KV与score。QA1280+Weight32的BF16面板可以登记与输入RMS组合，但与C2连接时必须保留混合精度和不同的行映射；把FLOAT权重降成BF16不是等价实现。IndexKey的native FLOAT scale→FP16 cache，也不能用Query先FP16 scale再量化的路径替换。

C2按feature对两个原始请求partner的score做softmax，再加权KV。输入统一乘以标量会同时改变score温度和KV；构造例的输出方向比从e变成e²，后面的标量RMS无法恢复原方向。因此不能仅凭末端还有norm，就消掉前端input RMS。若一份值来自旧ring、另一份来自当前段，不同世代的缩放更不能混合。

发现一个值得继续核对的池化细化机会：在所选有限FLOAT参考域，ratio-2可用一次exp(min−max)，分别计算1/(1+e)和e/(1+e)，再恢复原始行的乘加顺序。784个参考Case与原表达式一致。不过，编译器是否已消除exp(0)、实际SFU指令与吞吐、native exp/FTZ/NaN/Inf/tie及BF16输出精度都未证明，不能声称实际提速。

较小概率不能直接写成1−较大概率：640个参考Case发生变化。一个反例中，p_max在FLOAT中已等于1，而p_min仍是约2.061×10⁻⁹；它乘以较大KV后，原BF16输出为0.2060546875，使用补数则变成0。融合或专用pair kernel必须保留这类数值边界，以及原始request pairing、旧ring读取、pool BF16舍入、全部T行初始化/norm、可选bias及key/long缓存语义。

生命周期也分开了。Z可在QA/SWA/Weight/C1/C2 cast等直接读者结束后逻辑退休；C2 FLOAT输入还需保留到两次GEMM读完；投影后的KV/score要保留到pool与任何延后ring update都读完。它们不必全部等待当前attention结束，也不能全部在pool结束时删除。实际存储别名、额外读者、capture和排队写者仍需补证，TASK结束只是一个充分的读取完成界。

若让pool norm、IndexKey或long-KV提前于ring commit执行，必须保留FP32投影tail，或通过已证明的新producer直接写入精确state；未来相同page/generation的旧状态读取仍须等待commit。6144个假设服务的DAG中，4128组当前终点较快、1792组较慢、224组相同；其中855组违反所选后续ring-read期限，包含665组表面上较快的情况。它们不能当成满足固定未来期限的收益，更不是实际跨迭代时间证据。

FLOAT输入接口也要去重：所选基线的cast输出写入4TH B、两个GEMM读取8TH B，共12TH B。两个投影在现有FLOAT plane上合并，可减少其中4TH B输入读取，剩8TH B；若新cast+joint producer能合法消去整个plane，联合参考仍是12TH B，不能再叠加前面的4TH。这里是逻辑接口计数，不是HBM流量、分配量或节省时间。C2的全部有效当前投影行、未来FP32 tail与既有ring容量不会因pool/norm融合自动消失。

条件关系167→173，待核对428→422，联合组仍26。35主候选、7历史框架记录、358语义行/26模块、4472次SP关联和854节点/1346边/48提示的部分图保留。9类/1352次原生证据、849个PMU缺口、16个phase未绑定图、3136项归属歧义仍待补齐；整模块关闭0，全模块物理MFU与最低时间尚未建立。没有新增服务器或NPU运行。
'''
    updates={cid:dict(review_v144=dict(evidence=names,pairs=comp['new_pairs'],reused_native_TASK_union=5064,pair_reference_cases=784,scoped_conclusion_changing_gap=gap,existing_broad_conclusion_gap_preserved=True,no_native_fusion_or_physical_gain_proved=True))for cid in touched}
    inputs=[Path(x['path'])for x in read(O/'hc-norm-index-state-inputs-v144.json')['inputs']];refs=read(O/'hc-norm-index-state-native-joins-v144.json')['databases'];inputs +=[Path(x['raw_database']['path'])for x in refs]+[Path(x['cached_native']['path'])for x in refs]
    findings={'M06':'InputRMSZ,QR,latent andHCrawXcoeff/Post/pre4 distinct;earlyconsumer/stateeventsnotglobalrawXretirement.','M07':'BF16Z->FLOATC2 andQRnativequant/QueryFP16scale paths requiretypedmixedproducer;notoneH5120fold.','M08':'PerfeatureC2softmaxtemperatureblocksinputscalarcancellation;ratio2oneexp/twoprobvariantneedsnative/compilerproof;fullprojection/tailstateretained.','M09':'IndexQ/Weight/Keyinputs/scalesdiffer;keyoldlatent/longWARandCS/slotgenerationsremain.','M10':'EarlyC2/cachepublication/normstateforkrequiresactualrangesandfuturestateprogress,notcurrentTASKintervalgains.','M21':'Originalrequestpairs/pages/generations/metadata andzeroQ/stateeffectsretained;SPboundariesnotnewhistorysegments.','M22':'Z/FLOAT/projectedKVScore/bankretirementsdistinct;lastreaderANDqueuedwriter/capture/aliascapacitygatesopen.','M23':'Mixedprecisionpanel/newhandoff andownerpartner/statehalosmustbepriced;cast/joint/projectioncreditsoneunion.','M24':'CurrentendpointcanimprovewhilechosenfutureRingreadviolatesdeadline;665hypotheticalfastercasesinvalidunderthatfixedfuturecontract.'}
    persist_refinement(version=144,previous=143,title='Input-RMS typed Index/C2 pair arithmetic and distinct value/state retirement',summary=dict(filename='hc-norm-index-state-summary-v144.json',previous_turn_classification='PROGRESS:v143inputRMS-QKV5relations/10824TASKs;PASS83640',**detail,refined_candidates=10,refined_modules=9,branch_records=358,semantic_modules=26,relation_clauses=173,higher_order_groups=26,pending_central_pair_reconciliation=422,partial_graph_node_rows=854,partial_graph_edge_rows=1346,structural_pair_hints=48,scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,pending_regime_reconciliation=9,pending_execution_reconciliation=1352,model_operator_implementation=False,native_test=False,actual_MFU_numeric_physical_ceiling_or_speedup_proved=False,Pad_authorization='GRANTED_AND_CONSUMED',remote_job_status='All local saved-profile/typedIndex-C2/pairalgebra/retirement/DAG work complete;no remote/device/Torch/CANN/model/replay/service work.'),changes=['Reconcile192Index/C1C2sourcekeyjoins5064priorTASKs;typedZ/QR/latent/quant/key/stateboundaries retained.', 'DeriveC2temperature/smallprobability/mixedweightcounterexamples,784finitepairreferences and6144DAGfuture/retirementchecks.', 'AddREL144-01..06;173pairs26groups422pending;allbroadercandidate/native/global gapsunchanged.'],candidate_updates=updates,module_updates={mid:dict(candidates=[],finding=f,evidence=names,gap=gap)for mid,f in findings.items()},remaining=remaining,action=action,jobs=[
        dict(kind='profile',status='pass',description='192exactsourcekeyunion5064priorTASKsrawAPI/CID/Computeverified;72C2/24C1/96Indexonly/currentordersandtypedquant/weightschecked.',artifacts=[O/'hc-norm-index-state-native-joins-v144.json',O/'hc-norm-index-state-observed-v144.json']+inputs),
        dict(kind='design-check',status='pass',description='784pairreferences/640complementchanges,6144DAGs/855chosenfutureviolations,typedpool/weight/lifetimecounterexamples and'+str(pre['checks'])+'preflightchecksPASS;sixconditionalrelations.',artifacts=[O/'hc-norm-index-state-preflight-v144.json',O/'hc-norm-index-state-validation-history-v144.json']+[O/n for n in names]+[O/'coverage-current-obligations-v144.json',O/'whole-scope-requirement-audit-v144.json'])],report=report,read_first=names+['coverage-current-obligations-v144.json'],publication_rows={},gate_updates=dict(overall_complete=False),scripts=[N/'analyze_hc_norm_index_state_native_v144.py',N/'analyze_hc_norm_index_state_relation_v144.py',N/'verify_hc_norm_index_state_relation_v144.py',N/'persist_profile_refinement_v90.py'],executor_prefix='hc-norm-index-state',main_script=Path(__file__),source_read=True)


if __name__=='__main__':main()
