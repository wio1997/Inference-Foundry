"""Persist cross-operator HC/attention contracts and a full-scope incompleteness audit."""
from copy import deepcopy
from pathlib import Path
import hashlib
from finalize_origin_coverage_v9 import N,O,R,read,write
from persist_profile_refinement_v90 import persist_refinement


def main():
    assert not(O/'REVIEW-CHECKPOINT-v93.json').exists()
    assert read(O/'HC-attention-fork-analysis-preflight-v93.json')['status']=='PASS'
    names=['HC-attention-fork-source-contract-v93.json','HC-attention-fork-native-frontiers-v93.json','HC-attention-shared-resource-models-v93.json',
      'HC-attention-deadline-and-precision-models-v93.json','HC-attention-fork-variants-v93.json','HC-attention-composition-clauses-v93.json',
      'HC-attention-fork-math-validation-v93.json','operator-graph-G5-v93.json','operator-graph-node-coverage-v93.json','operator-graph-edge-coverage-v93.json',
      'operator-graph-candidate-bindings-v93.json','operator-graph-coverage-audit-v93.json']
    packet=read(O/names[5]);members=packet['higher_order_groups'][0]['members']
    rel=deepcopy(read(O/'candidate-relation-clauses-v92.json'));rel['clauses']+=deepcopy(packet['clauses']);rel['higher_order_groups']+=deepcopy(packet['higher_order_groups'])
    rel.update(clause_count=122,selected_pair_count=122,parent_relation_index='candidate-relation-clauses-v92.json',no_new_maincandidate=True,overall_complete=False);write(O/'candidate-relation-clauses-v93.json',rel)
    cov=deepcopy(read(O/'candidate-relation-coverage-v92.json'))
    for c in packet['clauses']:
        p=next(p for p in cov['pairs']if{p['a'],p['b']}=={c['a'],c['b']});assert not p['clauses']
        p.update(scope_state='SCOPED_OPERATOR_RELATION_REVIEWED_CONDITIONAL',clauses=[c['id']],whole_pair_all_variants_closed=False,new_critical_path_gain_us=None)
    cov.update(scoped_conditional_pairs=122,remaining_pair_relations=473,higher_order_groups=25,parent='candidate-relation-coverage-v92.json',
      parent_relation_index='candidate-relation-clauses-v93.json',HC_attention_refinement_v93=dict(new_pairs=7,new_groups=1,evidence=names),overall_complete=False)
    write(O/'candidate-relation-coverage-v93.json',cov)
    fps=deepcopy(read(O/'candidate-operator-footprints-v92.json'));index={r['id']:r for r in read(O/'candidate-record-index-v92.json')['model_candidates']}
    binding={r['id']:r for r in read(O/names[10])['candidates']}
    for row in fps['candidates']:
        q=R/'candidates'/(row['id']+'.json');actual=hashlib.sha256(q.read_bytes()).hexdigest();assert actual==index[row['id']]['master_sha256'];row['captured_master_sha256']=actual
        if row['id']in ['F01','F02']:row['operator_atom_union']+=['mhc.attnPre.currentCoeff','mhc.attnPre.delayedY']
        if row['id']=='F03':row['operator_atom_union']+=['mhc.attnPost.nextFfnPre']
        if row['id']in members:row['HC_attention_source_fork_v93']=deepcopy(binding[row['id']]['HC_fork_source_supplement_v93'])
    fps.update(parent='candidate-operator-footprints-v92.json',overall_complete=False);write(O/'candidate-operator-footprints-v93.json',fps)
    nav=deepcopy(read(O/'candidate-pair-navigation-v92.json'));fpmap={r['id']:r for r in fps['candidates']}
    for pair in nav['pairs']:
        pair['conditional_clause_ids']=[r['id']for r in rel['clauses']if{r['a'],r['b']}=={pair['a'],pair['b']}]
        pair['semantic_pair_reconciliation_done_in_stated_scope']=bool(pair['conditional_clause_ids'])
        pair['shared_operator_atom_union']=sorted(set(fpmap[pair['a']]['operator_atom_union'])&set(fpmap[pair['b']]['operator_atom_union']))
    nav.update(parent='candidate-pair-navigation-v92.json',scoped_conditional_pairs=122,remaining_central_pair_reconciliation=473,overall_complete=False);write(O/'candidate-pair-navigation-v93.json',nav)
    op=deepcopy(read(O/'operator-analysis-coverage-ledger-v92.json'));op['HC_attention_fork_refinement_v93']=dict(evidence=names,frontiers=3200,root_bound=2560,phase_unbound=640,
      strict_unique_SWA=3120,remaining_ambiguous_SWA_C1=80,reused_native_tasks=44800,new_regime_or_execution_promotions=0,new_pairs=7,new_groups=1,
      current_coeff_Y_split_prior_answer_version=58,actual_internal_C_Y_service_or_freeslack_proved=False)
    write(O/'operator-analysis-coverage-ledger-v93.json',op)
    matrix=read(O/'module-phase-branch-matrix-v92.json');m06=deepcopy(next(m['branches']for m in matrix['modules']if m['id']=='M06'))
    gap=('Currentcoefficient andYsplit ispriorv58 sourcefact,notnativehandoff. ActualC/Yservice,Postrowread deadlines,newbank/reader/rowepochs,nativeHF32/Gram/Vectorprecision andsharedresourcecosts areunbound. '
      'WeightcutsrequiredeclaredL1path/uncompressed distinctplanes/oneinitialgrant/fulloutputdomain;GMdirect/pruning/compression/parameterdedup remainalternative domains. '
      '3200savedgraphfrontiers preserve80KVroleambiguities and16phase-unboundgraphs;observedspansarenotfreeslackorrequestgain. NoactualphysicalMFU/rank/speedupproof.')
    for row in m06:
        if row['id']in ['M06-OP-06','M06-OP-07','M06-OP-08','M06-OP-09','M06-OP-12']:
            row['evidence']+=names;row['HC_attention_review_v93']=dict(native_frontier_binding=3200,one_current_coeff_work=True,global_resource_credit_not_per_candidate=True,all_readers_closed=False);row['gap']=gap
    assert 'M06-OP-15'not in {r['id']for r in m06}
    m06.append(dict(id='M06-OP-15',branch='Currentattentioncoefficient/externalYfork throughQA/Qnorm/DQ/QB,SWA/attention/output/RS toPostandfollowingFFNPre;sharedweightcut/deadlines and20+4columnprioritysplit.',
      state='SCOPED_SHARED_WORK_DEADLINE_AND_RESOURCE_CONTRACT_REVIEWED;WHOLE_MODULE_OPEN',evidence=names,gap=gap,
      scope='Samecurrentblock,source/currenttypedM3N24graphcohort anddeclaredresource/precisiondomains;notallcaller/output/runtimeclosure.'))
    mids=['M06','M07','M10','M11','M12','M22','M23','M24']
    mods={mid:dict(finding='CurrentHCcoefficient/externalYattentionfork has3200nativeTASK-linkedfrontiers andexplicitPost/nextPre deadlines. Joint3/4weightplanes useoneinitialwarmgrant;candidateanchors/rowowners remain distinct. No newnativeassignmentoractualgain.',
      evidence=names,candidates=[cid for cid in members if cid in next(m['candidate_links']for m in matrix['modules']if m['id']==mid)],gap=gap,
      profile_scope='Cross-operator/layer/owner/state contracts andtypedresourcecuts;noframeworksurvey.')for mid in mids}
    updates={cid:dict(review_v93=dict(evidence=names,HC_attention_fork_native_frontiers=3200,conditional_union_resource_cuts=8,
      current_HC_source_split_prior_version=58,existing_split_not_counted_new=True,
      new_relation_ids=[r['id']for r in packet['clauses']if cid in[r['a'],r['b']]],new_joint_contract='HG93-HC-ATTENTION-FORK-DEADLINE',
      one_baseline_not_independent_speedups=True,native_alias_handoff_or_precision_closed=False,implementation=False,native_test=False),conclusion_changing_gap=gap)for cid in members}
    updates['F01']['next_profile_or_contract_action']='F01isalreadypresentHCboundaryanchor,nosecondHCwork. Preservecurrentpost/comb-beforePost andpre-beforeFFNPre. Newweightcut/headrow/phasejoinisconditional;do nottreat281usobservedspanasfreeoverlap orcollapsefull4HcoeffintoY.'
    write(O/'PROFILE-CODE-QUESTION-GATE-v93.json',dict(status='WHOLE_SCOPE_GAP_REVIEW_THEN_NATIVE_OPERATOR_DEADLINE_JOIN',
      questions=['Whichpendingrelationsalreadyhaveoperatorfactsbutno centralcontract?',
        'WherearecurrentHCoutputsconsumedintheattention/FFNpath?',
        'CanallHC/projectionweightwork bepricedwithindependentwarmgrants? No;useoneunioncut.',
        'Canfull4HcoefficientprojectionuseonlycollapsedY? Genericinformationcounterexample.'],
      cached_pinpoint_model_source_files=1,source_split_fact_prior_version=58,new_target_source_files=0,new_public_document_read=False,new_remote_source_read=False,
      no_framework_wide_research=True,no_Torch_model_NPU_capture_or_server_work=True,overall_complete=False))
    write(O/'HC-attention-validation-history-v93.json',dict(corrected_model_assumptions=[dict(topic='Q_Binputweightshape',initial_assumption='Ordinary2Dweightdescriptor',
      saved_evidence='INT8[1024,80,16,32]withDT_BF16[32768]/FLOAT[3]sideinputs',correction='LogicalK/Nfromactivation/output andelementproduct;noinferredmemoryformat/weightorientation. Scales remainseparateservice,notzero.')],
      deadline_domain_refinement=dict(previous_overstrong_endpoint='Full24-column weight work charged before current attention Post',
        correction='Current Post requires20post/comb columns;thelastpre4group may be omitted from this deadline. FollowingFFNPre requiresall24;earlierlayers pre4 groups remain mandatory in a multilayer window.',
        resource_cut_cases=8,weight_bytes_omitted_only_at_last_Post_deadline=327680),
      phase_and_SWA_ambiguities_retained=True,all_local_jobs_finished=True,new_device_work=False,overall_complete=False))
    write(O/'HC-attention-question-status-v93.json',dict(newly_completed=[
      '3200existingnativegraphblocksjoinHCPre/norm/query/KV/output/HCPost/nextFFNboundary;44800previousTASKs,notnewexecutions.',
      '7previouslycentral-unreconciledrelationsand1currentattentionforkcontractnowexplicit.',
      'HC/QA/SWA/QBweightunionhasoneinitialwarmgrant,notonepercandidate;typedinputs/rowowners/padding/scale sideinputs explicit.'],
      reused_prior_answers=['ExternalpremixY/currentCoeffsplitv58/v59','HC/HF32native-sourceprecisionv58/v89','QA/SWArowownershipv61','NoSort/MFU/clockscalingclaim'],
      unresolved=[gap,'Whole-scopeunreviewedrelations/modulebranchschemas/criticalruntimeevidencelistedinseparatefullaudit.'],whole_modules_closed=0,overall_complete=False))
    # Requirements come from the full user objective and task AGENTS;structural coverage is kept separate.
    pending=[r for r in op['regimes']if r['state']!='SCOPED_OPERATOR_CONDITIONAL_FUSION_AND_WORK_IO_REVIEWED']
    audit=dict(previous_goal_turn='PROGRESS:v92authoritativemanifestandoperatorcontractsverified',input_checkpoint=92,followup_checkpoint=93,
      source_scope='Specifiedmodel/version/hardware/load;codequestionsonlyforoperatorfusion/split/reorg/theory. Hostmoduleboundariesneedoperatorselection/readinesscontracts,noframeworksurvey.',
      requirements=[
        dict(id='MODEL_VERSION_HARDWARE_LOAD',state='PARTIAL',evidence=['HC-attention-fork-source-contract-v93.json','typed-capacity-primary-contract-v89.json'],gap='Source/casepinned;allinstalledbinary/mode/physicalcapacity bindingsnotclosed.'),
        dict(id='ALL_MODULE_CONDITIONAL_BRANCHES',state='PARTIAL',modules=26,semantic_modules=24,branch_records=355,chronological_only=['M00','M03'],gap='Reconcileonlyoperatorselection/readinesssemanticsforthesehostboundaries;allapplicableoperatorbranchesnotexhausted.'),
        dict(id='CURRENT_FINITE_GRAPH_LEDGER',state='SCOPED_STRUCTURAL_PASS_NOT_GLOBAL_SEMANTICS',evidence=['operator-graph-coverage-audit-v93.json'],rows=[743,1209],gap='Partialgraphs/connectedsets2..4 are notall primitive/backend/runtime/higher-order combinations.'),
        dict(id='CROSS_OPERATOR_LAYER_CARD_ITERATION',state='PARTIAL',model_candidates=35,legacy_framework_records=7,conditional_pair_relations=122,higher_order_groups=25,central_pairs_not_reconciled=473,gap='Emptycentralentries cancontainpriorlocalanswers;theydo notproveindependence/noop. Allpairvariantsandjointsemantics remainopen.'),
        dict(id='NATIVE_OPERATOR_REGIMES',state='PARTIAL_EVIDENCE',conditional_reviewed=[2073,285598],pending=[9,1352],gap='Pendingoperatorsarealreadyidentified/questionsaudited;missingoperand/fieldidentityisnotzerowork.'),
        dict(id='THEORY_AND_MEASURED_COMPARISON',state='PARTIAL',evidence=['A3-matrix-typed-capacity-cases-v89.json','HC-attention-shared-resource-models-v93.json'],gap='Scopedtyped/resourcecutsnotallalgorithmoractualphysicalwallclockceilings;memorytiers/mixedissue/actualfrequency/readinessremain.'),
        dict(id='LOWEST_MODULE_UTILIZATION',state='NOT_ESTABLISHED',gap='Noall-modulevalidcommonphysicalMFUdenominator. Zero-primary/unboundmodulesnotrankedzero;busyfractions,exposure,peakreferencefractionsandMFUnotinterchangeable.'),
        dict(id='CAUSES_AND_CRITICAL_PATH',state='PARTIAL',global_missing_PMU=849,phase_unbound_graphs=16,ambiguous_assignments=3136,gap='Profiles/sourceprovidecompetitors,notuniversalcausalrequestcriticalpath. TASK/COMM sumsorobservedspansnotrecoverablelatency.'),
        dict(id='CRITICAL_EVIDENCE_AND_FINAL_CLOSURE',state='NOT_ACHIEVED',gap='Allreader/alias/parameter/row/stateepochsandcriticalnativefieldsnotclosed;report/manifest/mathcheckcompletioncannotcloseGoal.')],
      pending_native_rows=[dict(regime=r['regime_id'],operator=r['operator'],executions=r['executions'],state=r['state'],
        unresolved=('Destination/length/read-writeoverlapandnativehelperscope'if r['operator']=='MemSet'else'BOOLfieldidentityandallconsumers'))for r in pending],
      no_whole_optimization_percentage_claimed=True,whole_modules_closed=0,overall_complete=False)
    assert len(pending)==9 and sum(r['executions']for r in pending)==1352
    write(O/'whole-scope-requirement-audit-v93.json',audit)
    remaining=list(read(O/'module-coverage-gaps-v92.json')['remaining'])
    remaining.append('v93:FullrequirementauditstillPARTIAL,24/26semantic355records(M00/M03chronologicalonlyneedoperatorboundaries),no validallmodulelowestMFU.3200HC-attentionnativefrontiers2560root640unbound,3120strictSWA/80layer20ambiguity,44800oldTASKreused.7centralclauses+1HG=>122pairs25groups473pending,not7independentgains. ExistingY/coeffsplitv58notnew. JointHCFLOAT1.966MB+QA13.107MB+SWA5.243MB fixedB/L1onewarmgrant followingFFNPreall24deadline6102cycles;withQBINT841.943MB19755cycles. EarlierPostonly20guaranteed5995/19648,omitlastpre4weight327680B once;40layerprevious39pre4groups remainmandatory. 8cutscoverallchargedservice/nocomputedpartials/notwholeinputbarrier.Nativescalevectors/4DQBdescriptorretained. Full4HcoeffcannotgenericallyfactorY,local3/global24 rowcosts;20+4independentpaddingadds50%reference andidealPostnormalizationcanloseoverlap. NewpartialG5child19/22=>743/1209,45structpairhints notsemantics.2073/285598review9/1352pending849/16/3136 unchanged0wholeclosed.')
    report='''本轮先按完整目标复核覆盖，再把mHC与注意力主路径的跨候选关系汇入台账。整体仍未完成：26个模块中24个已有语义分支台账，M00/M03仍需仅整理与算子选择/就绪有关的边界；原生关键字段、完整候选组合和统一模块利用率分母也未闭合。详见whole-scope-requirement-audit-v93.json。没有把“已有条件结论”当作尚未研究，也没有把报告成稿当作全目标完成。

从已有profile按database、graph、block及原生TASK号，连通3200组注意力HcPre→norm→投影/输出→HcPost→FFN HcPre/norm边界，复用44800条既有TASK。其中2560组有阶段归属、640组仍来自16张阶段未知图；3120组有唯一SWA投影候选，80组layer20的SWA/C1角色歧义保持原样。HcPre完成到HcPost开始的观测跨度中位数约281.6微秒，它包含现有注意力路径，不能当作可免费重叠或可回收的时间。

关键数据关系由已有冻结源码确认：当前Y使用上一个pre_mix，当前post/comb在注意力HcPost消费，当前pre在随后FFN HcPre消费。随后FFN Pre还依赖HcPost更新后的残差，F04 FFN归一化不能与原注意力归一化按同一输入合并。Y/系数拆分方向在v58/v59已经分析，本轮新增的是原生块级消费边界、联合资源计价及七条中央关系合同，而非再次宣称发现这项拆分。

HC系数投影读完整4H原残差，Q_A读RMS后的BF16 Y局部3行，SWA读全局24行，输入、dtype和owner均不同，不能直接横向拼成一个共享A的宽GEMM。一个通用反例中，外部premix=[1,0,0,0]，两组HC输入的第0通道相同而第1通道不同，折叠Y及其归一化相同，但选择第1通道的合法系数权重得到不同结果。它否定无条件因式化，不代表已检查当前checkpoint是否具有特殊结构。

联合资源模型保留同一份HC工作，并区分两个截止点。到随后FFN Pre所需输入就绪时，当前HC的24列系数全部必要：HC系数FLOAT权重为1966080B，Q_A BF16权重13107200B，SWA BF16权重5242880B，三者共20316160B。在声明的24-AIC、未压缩互异权重、固定B且必须经过L1→L0B、仅一次初始L0B驻留额度的域中，联合时间下界为6102周期；加入Q_B的41943040B INT8权重后为19755周期。

到当前attention Post所需输出就绪这一较早截止点，只保证post/comb的20列必要，故从保证工作量中扣去最后pre4列的327680B，相应较弱下界为5995/19648周期。跨40层时只有最后一层的pre4列可这样扣除，前39层pre已供后续FFN使用，不能逐层扣除。三/四权重平面、一/四十层、两个截止点共8种条件cut。窗口须覆盖全部计价服务且初态无已计算的部分结果；没有把某个原生TASK开始时间指定为窗口起点，也没有强制整张输入就绪才能流式处理。每个窗口只能扣一份初始warm额度，不能按候选或层重复扣。允许GM直达L0、压缩、权重去重或删减输出的实现另建模型。这些不是实际耗时、HBM流量或已证明瓶颈。

核对Q_B时，原生权重描述实际为[1024,80,16,32]，不是普通二维矩阵；逻辑K/N由激活、输出和元素总数确定，没有凭形状推断内存格式。另有BF16[32768]通道scale与FLOAT[3]行scale，它们的服务和精度保留，未塞进不适用的L1→L0B权重cut，也未计为零成本。四种精度的独立计算容量界取max；没有凭独立峰值拼出混合执行能力。

局部行/全局行也不能混用。若为了拼投影而把完整HC原残差搬到全局24行，系数有效工作会变为8倍，HC4原始payload的ring参考为860160B/卡，而已有归一化H payload为215040B/卡。这是条件载荷计价，不是4倍通信耗时，更没有消除当前权重服务。

对系数输出的20列post/comb与4列pre，可分析不同消费截止点，但独立M/N按16对齐的Mmad参考中，紧急20列与原24列同样占32列；再拆4列会使该参考服务量增加50%。它不证明所有实现都更慢，只说明有用FLOPs少了并不意味着这个分块方案的服务量也下降。

将系数末端归一化移进HcPost也不是必然收益。固定相同服务时长、无限资源的依赖模型中，提前完成当前系数与Y/注意力分叉，比等注意力结果后再做系数归一化更早或相等；后移可能失去原先重叠。实际若改变launch、Vector竞争或GM接口，则必须重新计价；HcPost按D切分时，还要避免把39轮归一化在多个D tile中重复。48400组模型检查只验证这条依赖关系，没有把实测HcPre耗时拆成未知的C/Y服务。

新登记七条条件关系，覆盖HC当前系数与FFN norm、Qnorm/DQ/QB、QA/SWA、KV状态、注意力输出、ReduceScatter和输出投影的耦合。一个联合合同把这些边界放在同一图里，防止候选之间重复计算同一HcPre、重记接口收益或把共享资源当免费。仍须新Y可见性、全部原始/归一化/缓存读者、Post和下一Pre的参数/行/状态epoch、原生数值及真实资源成本证据。

当前部分G5子图新增19行节点、22行边，版本绑定的总结构台账743/1209行，导航提示45对。这些不是唯一算子或全覆盖率。累计355条分支记录、24个语义模块、122条条件关系和25组联合合同；中央仍有473对待汇入或复核。2073个regime/285598次执行已作条件分析，9个regime/1352次执行仍有关键证据缺口；849条PMU缺失、16张阶段未知图、3136条模块归属歧义不变。没有整模块关闭，目标继续。
'''
    action=('继续原Goal，v93PROGRESS，v92亦PROGRESS。先读whole-scope-requirement-audit-v93和HC-attention-summary/PROFILE-FINDINGS/source/native/shared-resource/deadline-precision/variants/composition/mathvalidation及G5v93/coverage/bindings。'
      '整体仍26modules24semantic355records，M00/M03历史记录只需整理operatorselection/readiness边界，不做framework通读；122pairs25groups473centralpending不等于473肯定完全没局部研究。9regimes1352calls已识别问题但MemSetdest/len/overlap或BOOL字段/allreader缺证，notzero；lowestallmoduleMFU仍未能同分母定案，busy/exposure/peakfractions不是MFU。'
      '本轮新native联接3200HC-attnfrontier2560root640unbound，44800oldTASKs，SWAunique3120+layer20ambig80保留。旧Y/coeffsplitv58/v59不是新发现；post/comb在currentattentionPost用，pre在followingFFNPre用，FFNPre还依赖updatedPostresidual。281.6us观测span不能当freeslack/gain。'
      '7newcentralcontracts F01 vsF04/F05/F06/F08/F14/F15/F16 +HG93-HC-ATTENTION-FORK-DEADLINE，F01 baselinealreadyimplemented不是再造一份HC。Full4H/FLOAT系数vsBF16normY/local3 QA,QB/global24SWA不同input/type/owner，不能onewideGEMM。Generic同Y异coeffcounter非actualcheckpointstructureproof。'
      'JointHC/QA/SWAfull24weights20,316,160B，24AIC/fixedB/L1only/uncompresseddistinct/oneinitialL0warmgrantdomain，到followingFFNPreinputready为6102cycles；加QBINT841,943,040B为19755。较早currentPost只保证20post/comb必要，减lastpre4的327680B后5995/19648；40layerdistinctwindow只减最后pre4，前39组仍必需。共8cuts，每window一份grant且覆盖全部chargedservice/无computedpartials，不强制wholeinputbarrier/原生tasktimeorigin，GMdirect/compression/pruning/dedup另域。NativeQBblocked4D[1024,80,16,32]product等logicalKN但不猜format；BF16channel32768/FLOATrow3 scales另服务不是0。Independenttypedpeaks只max，不拼mixedissuecapacity。'
      '20urgent+4deferredcols的独立M/N16paddingreferenceurgent20同original24、合计+50%，非nativeallfusionsbound。把fullHC搬global24可多8x有效work、ringraw860160vsnorm215040B，不是latencyratio。Fixedsame-serviceidealDAG lateCoeffNormPost≥earlyCfork，48400cases，不推实际时间或省launch必快。'
      'NewG5HCfork19/22 currentpartial743/1209;currentstructhints45 notsemanticcoverage。355/24,122/25/473,2073/285598review9/1352pending849PMU16phase3136ambiguity0wholeclosed。'
      '下一步从fullaudit优先补真正未整理的operator边界/尚未汇总的有证据跨候选关系或影响理论的直接operator分支；不要继续复读Y/coeff分叉/采样pivot，也不要盲增Cartesianpairs或扩大framework研究。现有profile优先，缺定案证据才定点代码。无采集授权只准备有限计划，保持offline/frozen/math/必要primary，无model/Torch/NPU/capture/service/upload。GoalACTIVE/TaskACTIVE/LoopEVALUATING；verify_HC_attention_fork_v93须PASS，不重跑finalizer不提前complete。')
    oldgraphs=[Path(p)for k,p in read(O/names[-1])['graph_catalog'].items()if k!='G5']
    jobs=[dict(kind='profile',status='pass',description='Join3200savedHC-attentionblocks44800existingTASKs toPost/FFNdeadlines;uniqueSWA3120/ambiguous80 andphasecounts retained,noalias/slack/nativepromotion.',
      artifacts=[O/names[0],O/names[1],Path(read(O/names[0])['path'])]),
      dict(kind='design-check',status='pass',description='8jointweightcuts withseparatePost/FFNPredeadlines,lastpre4omittedonlyonce,logicalKNvsblockedQB/scaleinputs,independent20+4paddingreference and48400idealDAGcases.7relations1group not7gains,fullscopeauditremainsincomplete.',
        artifacts=[O/n for n in names[2:]]+[O/'HC-attention-fork-analysis-preflight-v93.json',O/'whole-scope-requirement-audit-v93.json',O/'HC-attention-validation-history-v93.json',O/'typed-capacity-primary-contract-v89.json',R/'fusion-scan.json']+oldgraphs)]
    persist_refinement(version=93,previous=92,title='Whole-scope audit and HC attention fork deadlines with shared resource cuts',
      summary=dict(filename='HC-attention-summary-v93.json',previous_turn_classification='PROGRESS:v92pivotoperatorcontracts',cached_pinpoint_model_source_files=1,new_target_source_files=0,new_public_primary_documents=0,
        exact_native_frontiers=3200,root_bound_frontiers=2560,phase_unbound_frontiers=640,strict_unique_SWA_frontiers=3120,ambiguous_SWA_C1_frontiers=80,reused_native_tasks=44800,
        shared_weight_cuts=8,coefficient_deadlines=['LAST_ATTENTION_POST_REQUIRED_OUTPUTS_READY','FOLLOWING_FFN_PRE_REQUIRED_INPUTS_READY'],ideal_equal_service_DAG_cases=48400,current_HC_split_prior_answer_version=58,old_split_counted_new=False,
        new_graph_node_rows=19,new_graph_edge_rows=22,finite_graph_node_rows=743,finite_graph_edge_rows=1209,current_structural_pair_hints=45,
        refined_candidates=len(members),refined_modules=8,new_inventory_families=1,branch_records=355,semantic_modules=24,
        new_pairs=7,new_groups=1,relation_clauses=122,higher_order_groups=25,pending_central_pair_reconciliation=473,
        scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,pending_regime_reconciliation=9,pending_execution_reconciliation=1352,new_regimes_or_executions_promoted=0,
        all_requirements_or_lowest_MFU_or_physical_speedup_proved=False,implementation=False,native_test=False,remote_job_status='Alllocal profile/source/math/audit work finished;no remote/model/Torch/NPU/capture/servicework.'),
      changes=['Reauditfullobjective;explicitcritical-evidence versuscentralreview/gapstatus,notreport-completion.',
        'Join3200existingnativeHC-attentionfrontiers andretainSWA/phaseambiguities;sourcecoefficientdeadlinesexplicit.',
        'Jointtypedweightservicecuts/onewarmgrant,owner/input/padding/blockedQB/scale/DAGfences.',
        '7newcentralrelations1jointcontract=>122/25/473;355records24semantic,globalnativeevidencegaps/GoalACTIVE unchanged.'],
      candidate_updates=updates,module_updates=mods,semantic_branches={'M06':m06},remaining=remaining,action=action,jobs=jobs,report=report,
      read_first=['whole-scope-requirement-audit-v93.json']+names+['HC-attention-question-status-v93.json'],
      publication_rows={'F01':'| F01 | 已有注意力 mHC Pre 复合边界及其系数/外部Y分叉 | 已实现边界／条件重组 | 当前post/comb必须供本层Post，pre供下一FFNPre。Y/系数拆分是既有方向；本轮联接3200原生块并补共享权重/截止点合同。不能把同一HC工作、warm额度或观测跨度重复记为收益。 |'},
      gate_updates=dict(overall_complete=False),scripts=[N/'analyze_HC_attention_fork_v93.py',N/'model_HC_attention_composition_v93.py',N/'verify_HC_attention_fork_v93.py',N/'persist_profile_refinement_v90.py'],
      executor_prefix='HC-attention',main_script=Path(__file__),source_read=False,additional_shared_writes={R/'fusion-scan.json':read(O/'fusion-scan-proposed-v93.json')})


if __name__=='__main__':main()
