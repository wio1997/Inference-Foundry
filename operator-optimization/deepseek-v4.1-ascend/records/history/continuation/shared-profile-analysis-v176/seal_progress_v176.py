"""Persist positive public-operator evidence and the completed source rule round."""
import json
import sys
from pathlib import Path
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H));sys.path.insert(1,str(H.parent))
from finalize_origin_coverage_v9 import N,O,R,read,write
from analyze_evidence_batches_v155 import pin
from persist_operator_review_delta import persist_delta

old=read(O/'REVIEW-CHECKPOINT-v174.json');prefix='public-profile-graph-and-rule-round'
receipt=read(H/'final-terminal-and-service-receipt.json')
facts=read(H/'operator-conclusions-from-profile.json')
rule=read(H/'declared-source-template-rule-review.json')
checks={
    'positive_operator_review':read(H/'operator-evidence-review-checks.json')['status']=='PASS',
    'native_intact_node_prefix':facts['intact_native_nodes_EP0']==821146,
    'partial_case_explicit':receipt['frozen_case_status']['valid'] is False,
    'all_exact_handles_terminal':receipt['all_submitted_handles_terminal'],
    'original_service_HTTP200':receipt['service']['health_http']==200 and receipt['service']['running'],
    'original_script_SHA':receipt['service']['serve_sha256']=='c297201185b5676254af4d7693279f89bfa60f9c76eb340058a78ecf730b7bba',
    'source_rule_round_27templates':len(rule['capsules'])==27,
    'source_rule_nodes_edges':len(rule['node_coverage'])==854 and len(rule['edge_coverage'])==1346,
    'no_new_source_family_from_rule_round':rule['new_non_equivalent_candidate_families']==[],
}
assert all(checks.values()),checks
authorities={}
for key,stem in [('candidate_ledger','candidate-profile-ledger'),('module_ledger','module-profile-ledger'),('branch_matrix','module-phase-branch-matrix')]:
    p=Path(old[key]);x=read(p);x.update(parent_review_authority=pin(p),current_review_checkpoint=176)
    x['current_public_profile_binding_delta']={
        'facts':str(H/'operator-conclusions-from-profile.json'),
        'scope':'joint176 actual startup/capture and incompleteC8 runtime prefix;no oldcohort backfill',
        'source_rule_round':str(H/'declared-source-template-rule-review.json'),
        'all16_observed_greedy_sampler_calls':1104,
        'complete_current_runtime_graph_or_all_future_epochs_proved':False}
    if key=='candidate_ledger':
        affected={cid for f in facts['facts'] for cid in f['candidates']}
        for c in x['candidates']:
            if c['id'] in affected:c['public_operator_evidence_v176']=[f['id'] for f in facts['facts'] if c['id'] in f['candidates']]
    if key=='module_ledger':
        for m in x['modules']:
            fs=[f['id'] for f in facts['facts'] if m['id'] in f['modules']]
            if fs:m['public_operator_evidence_v176']=fs
    q=O/(stem+'-v176.json');write(q,x);authorities[key]=str(q)

changes=[
 'Combinedjoint176diagnostic loaded/captured bothmodels andexecuted apartialfrozenC8 prefix,beforeperworker768MiBETthresholdstoppedcollection.FullCasefailed;all16workerFINALabsent.No fullcase/outputparity/performance successclaim.',
 'Independentoriginalruntime restorationconfirmedHTTP200,exactoriginalserveSHA,currentStartedAt2026-11-03T01:17:47.463945252Z;controllerandbothCPUparserexacthandles terminal.No furtherNPU probe or modelreload neededfor this offlineanalysis.',
 'CPUstreaming recovered821146intactEP0nativeETnodes/235public-runtimeoperatornames,and5362of5366bindingmarkers;last4unmatchedsample_tokenscallbacks atsteps71/72 explicitlyretained.Unclosedtop-levelET notsynthetically repaired.',
 '16workers eachreport69completedAscendRejectionSampler.forwardentry/returnpairs,1104calls:allGreedyTrue/allRandomFalse,noPenaltiesTrue,max_num_logprobsNone,syntheticFalse,entropyFalse.Actualobservedguardsnowbound;allcase/scoredeadnessnotinferred.',
 'Target/draft topk_indices_buffer confirmedsame rawStorageImpl/address inbothcapture/formalstartcatalogues:INT32[8192,512],16MiB.Sharedwriter/reader/rollbackepochs mustbe retained,notindependent banksor repeatedallocations.',
 'Four target/draft embedding/headBF16[16160,5120]params have distinctrawstorageobjects/addresses.Numericalequalitywasnotread;noCSEfromshape alone.SharedinputmultioutputGEMM remainsconditionalwithallrequiredweights/work.',
 '73namedC2maskviews bindone8192Bbackingwithobservedlengths24/1704/3200/8176;146slice/as_strided and73copy occurrences.Nofixed24Ballocation/constantgeneration orwrite-once inference.GlobalETstorageIDjoinsremainoverapproximations,diagnosticRFreadsnotbaselineconsumers.',
 'Oneconditional semanticrewrite-rule roundcovers27declaredsource templates,854nodes,1346edges,7rules,nonew non-equivalentfamily.Fork/rounding/duplicateindex/RNG/outputepoch/communication/lifetimeconstraints retained;not yetwhole-scopeP1/P3completionproof.',
 'Previouslycompleted1101staticoperator-effectfrontier,all42compositioncontracts and94P5comparisons reusedwithoutmodel-source rereads,Cartesianpairsor freshcapacityqueries.49actualmoduleclasses/3108identityrecords havezero registeredmodulehooks inEP0catalogues;notuniversalobserverabsence.'
]
write(O/(prefix+'-preflight-v176.json'),{'status':'PASS','preflight':True,'checks':checks})
inputs=[p for p in H.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
write(O/(prefix+'-inputs-v176.json'),{'inputs':[pin(p) for p in sorted(inputs)],'overall_complete':False})
write(O/(prefix+'-summary-v176.json'),{
    'status':'PROGRESS','previous_goal_turn':'PROGRESS','overall_complete':False,
    'new_native_operand_or_phase_promotions':0,'new_whole_module_closures':0,
    'intact_native_ET_nodes_EP0':821146,'matched_public_binding_markers_EP0':5362,
    'public_runtime_operator_names_EP0':235,'actual_sampler_calls_all16':1104,
    'positive_operator_facts':4,'conditional_source_rule_templates':27,
    'conditional_source_rule_nodes':854,'conditional_source_rule_edges':1346,
    'all_submitted_handles_terminal':True,'original_service_HTTP':200,
    'current_full_C8_success':False,'performance_comparison_source_unchanged':True})

action=(
 '继续原完整目标，checkpoint176。先读operator-conclusions-from-profile、declared-source-template-rule-review、final-terminal-and-service-receipt。joint176因768MiB/worker阈值在部分C8后停止，已恢复原服务health200，所有控制器/CPU分析handle终止；不得因收尾需要再默认重跑完整Case。'
 '已取得821146完整原生节点、235op名称、5362marker匹配；最后4个sample_tokens回调在step71/72未匹配，不能虚补。1104实际sampler调用的greedy/noPenalty/noLogprob/noEntropy/noSynthetic已绑定，只限观察人群。准备态greedy/random双True不代表实际随机需求。'
 '目标/草稿16MiB索引bank真实同StorageImpl/地址，不能当独立；4个embedding/head storage不同但未读数值，不凭shape或指针判断数值相同/不同。C2mask73个视图指同8KiB backing，长度24/1704/3200/8176，保留writer/reader/epoch；全局ET ID联接只是过近似，RF诊断读不是原负载消费者。'
 '900/91/110合1101原静态frontier已审，42候选组合已审，94P5显式比较已成；27源码模板854node/1346edge的7规则条件语义完整一轮无新非等价候选族。这些不要重读/重配对/重算缺失fmax。'
 '仅余一次完整范围闭合审计：用现有26模块360语义行、source census/调用帧、49实际moduleclasses、235profile op目录与27模板账本，逐项列出仍没有算子值/分支/消费者/别名/跨迭代条件答案的具体ID，核对G0和effectivecallee完整性。不能仅凭模板计数或partialET宣称全范围完成；也不能把已分析的条件未知、历史epoch、物理fmax、私有MemSetABI或未实现提速继续冒充未分析范围。'
 '如果发现具体未分析的调用/图端口，只对那一个算子问题用现有profile/已缓存合同补齐，必要时定点取证，不广读框架或收集全量Tensor值。没有新NPU采集的默认计划。若核对后只剩已明确回答的条件/数值不确定性，按协议给出相应结论并完成实际达到的阶段；不得被旧helper硬编码overallfalse/whole_modules_closed0无限拖延，也不得在真实scope缺口仍在时标Goal完成。')
queue={'version':176,'parent':pin(O/'coverage-work-queue-v174.json'),
    'scope_preserved_by':old['coverage_obligations'],
    'work':[{'id':'Q176-FINAL-EXACT-UNANALYZED-SCOPE-AUDIT',
             'state':'CURRENT_G0_EFFECTIVE_CALLEE_AND_SCOPE_RECONCILIATION',
             'action':action,'evidence':[str(H/'operator-conclusions-from-profile.json'),str(H/'declared-source-template-rule-review.json'),str(N/'shared-profile-analysis-v174/current-conclusion-changing-evidence-policy.json')],
             'acceptance':'Every genuinelyunanalysedmodule/guard/graphport has explicitID and operator question,or positive scope closure. Conditional unknowns are differentiated from missing analysis;no pseudo completion or artificial universalnative/performancegate.'}],
    'completed_or_resolved_previous_work':[
        {'id':'Q174-CURRENT-G0-BINDING','state':'SUBSTANTIVE_PARTIAL_NATIVE_PUBLIC_BINDINGS_RECOVERED_NOT_FULL_G0_PROOF'},
        {'id':'Q174-FULL-RULE-PASS','state':'ALL27_DECLARED_SOURCE_TEMPLATES_CONDITIONAL_RULE_ROUND_COMPLETE;FULL_SCOPE_AUDIT_REMAINS'}],
    'no_default_new_NPU_acquisition':True,'overall_complete':False}
gate={'version':176,'parent':pin(Path(old['source_gate'])),
    'policy':'EXISTING_PROFILE_AND_CONTRACTS_FIRST;ONLY_AN_EXACT_UNANALYZED_OPERATOR_QUESTION_CAN_OPEN_SOURCE',
    'completed_frontier_and_rule_round':str(H/'declared-source-template-rule-review.json'),
    'current_facts':str(H/'operator-conclusions-from-profile.json'),
    'no_whole_framework_reread_or_cartesian_pairs':True,
    'no_default_model_probe_or_fullCase_repeat':True,
    'no_independence_from_absent_evidence':True,'overall_complete':False}
overlay={'version':176,'changes':changes,'authorities':authorities,'facts':str(H/'operator-conclusions-from-profile.json'),
    'conditional_source_rule_round':str(H/'declared-source-template-rule-review.json'),
    'restoration_and_terminal_receipt':str(H/'final-terminal-and-service-receipt.json'),
    'new_whole_module_closures':0,'overall_complete':False}
base=read(Path(old['coverage_obligations']))['obligations']
recs={r['id']:{'current_dispatch_policy':'Oneexactunanalysedscopeaudit,notrepeatprofiles/source/rules. Differentiateconditionaluncertainty frommissinganalysis.',
    'public_profile_facts':str(H/'operator-conclusions-from-profile.json'),
    'conditional_source_rule_round':str(H/'declared-source-template-rule-review.json'),
    'remaining_work':queue['work'][0]['id'],'whole_domain_closed_by_this_increment':False,
    'complete_original_scope_preserved':True} for r in base}
report='''整体继续 PROGRESS，未将全部目标标完成。原服务已恢复，health=200，启动脚本SHA一致；启动时间为2026-11-03T01:17:47.463945252Z。全部提交的控制器和CPU分析handle均已终止。

本次原生Execution Trace合并采集完成模型加载/图创建，并执行部分冻结C8迭代，随后触发单worker768MiB停止阈值。完整C8校验失败，没有16份FINAL，也没有完整输出对照；不把诊断时间或部分结果当性能基线。此前有效profile继续承担性能证据。版本复制时的诊断hash错误、一次额外我方容器启动重载和原服务恢复均有独立记录。第一版CPU提取在写出完整邻域后因64MiB检查失败，保留产物并仅用CPU完成汇总，不重跑NPU。

从未闭合顶层的ET文件中逐节点恢复EP0的821146个完整JSON节点、235种公开/运行时op名称，5362个绑定marker与侧录匹配；最后4个sample_tokens回调位于step71/72，保持未匹配。没有人工补文件尾，没有由ET ID推出物理分配或retirement顺序。全局storage-ID邻域只是过近似，诊断RF引用也不是基线必须保留的消费者。

关键算子证据有四项。第一，16worker各69次完整拒绝采样入口/返回，共1104调用，实际参数为all_greedy=True、all_random=False、no_penalties=True、max_num_logprobs=None、synthetic=False、entropy=False。支持这批调用的greedy专门化前提；仍保留score其他读者、argmax ties、correction、输出单元和RNG契约。准备态双True flags不能当实际随机需求，观察窗口也不外推为全Case所有分支。

第二，目标和草稿的topk_indices_buffer在capture/formal_start两次目录中指向相同原始StorageImpl与地址，INT32[8192,512]共16MiB。它们是共享状态，不能按独立bank安排写者，也不能重复计两份显存。是否可省写仍取决于发布、读取、回滚和代际，分配尺寸不是每次HBM流量。

第三，目标/草稿embedding和head四个BF16[16160,5120]参数分别绑定不同storage对象/地址。未读取数值，因此不声称数值不同；同shape同样不能支持值CSE。共享输入、多输出投影是另一种组合，仍需计全部必要权重与计算。

第四，C2mask有73个命名视图绑定同一个8192B持久backing，出现长度24、1704、3200、8176，并有73次source-level copy。BOOL[24]不是固定24B allocation，也不代表恒定值或代际。可以在公开语义边界融合最终写者，但不能因指针稳定而宣布write-once或删除所有写入。

已对27个声明源码模板、854节点、1346边完成7条融合/拆分/共享/跨模块/多消费者规则的一轮条件语义检查，无新增非等价候选族。保留各模板特有的数值舍入、HC fork、QR/QS区别、SMLA重复index、多输出、共享bank/RNG/通信/迭代状态约束。该轮覆盖声明图，尚不能仅凭数量或partialET宣布完整模型G0及有效callee全集已闭合。

此前1101静态点的算子效应、42候选共同组合约束、94条P5比较均直接复用，未重复读模型源码、做笛卡尔配对或追查物理fmax。当前EP0参数目录有49种module class、3108身份记录，注册module hooks均为0；这不等于所有native/其他观察者都不存在。

只保留一个收尾范围审计：把26模块360语义行、源码/调用帧清单、实际class/op目录及27模板逐项对齐，列出仍真正缺算子条件/消费者/别名/跨迭代答案的具体ID，核对G0和effectivecallee完整性。已有明确条件答案、历史epoch、物理上限未知或未实现提速不继续当通用阻塞项；确实未分析的范围也不能因报告成稿而关闭。没有默认的新NPU采集计划。
'''
updates={str(R/'module-analysis.json'):{'latest_public_operator_profile_facts':str(H/'operator-conclusions-from-profile.json')},
    str(R/'fusion-scan.json'):{'current_conditional_source_rule_round':str(H/'declared-source-template-rule-review.json'),
        'current_public_graph_binding_facts':str(H/'operator-conclusions-from-profile.json'),
        'current_remaining_full_scope_work':queue['work'],'whole_model_exhaustion_proven':False},
    str(R/'records.json'):{'latest_public_operator_evidence':str(H/'operator-conclusions-from-profile.json'),
        'latest_original_service_and_job_state':str(H/'final-terminal-and-service-receipt.json')}}
jobs=[
 {'kind':'review','status':'error','description':'joint176partialC8 stoppedat768MiBworkerETlimit;no fullcase/FINAL orbenchmarksuccess.CPUfirstextractoutputsizeassert failedafterneighborhoodwrite;retaineddatareviewedoffline.Originalservicerestored.','artifacts':[H/'acquisition-failure-and-artifact-shapes.json',H/'restoration-and-CPU-observation-006.json',H/'final-terminal-and-service-receipt.json']},
 {'kind':'review','status':'pass','description':'CPUstreamrecover821146intactnodes/235opnames/5362bindingmarkers,unclosedfileandlast4callbacks explicitlyretained;all16sidecarindexandrawbankfacts,no NPUrerun.','artifacts':[H/'native-ET-prefix-summary.json',H/'all16-sidecar-evidence-index.json',H/'public-bank-epoch-and-role-review.json',H/'CPU-recovery-final-observation.json']},
 {'kind':'review','status':'pass','description':'1104actualsamplercallsbindgreedyguards;target/draft16MiBindexbankalias,c2dynamicviews/8KiBbacking,fourseparateparameterbanks boundwithoutnumerical-equalityorindependenceassumptions.','artifacts':[H/'operator-conclusions-from-profile.json',H/'operator-evidence-review-checks.json',H/'actual-module-classes-and-hook-registry.json']},
 {'kind':'review','status':'pass','description':'Conditional source7rule roundcovers27templates/854nodes/1346edges;nonew non-equivalentfamily,no whole-scopeG0completionpromotion.','artifacts':[H/'declared-source-template-rule-review.json']},
]
persist_delta(version=176,previous=174,prefix=prefix,changes=changes,report=report,action=action,
    remaining_delta='Oneexactunanalysedscope/G0/effectivecalleeclosureaudit remains;conditional source round complete.',
    source_gate=gate,operator_overlay=overlay,queue=queue,jobs=jobs,scripts=list(H.glob('*.py')),
    obligation_reconciliations=recs,authority_overrides=authorities,shared_json_updates=updates,
    pending_full_scope_domains=[queue['work'][0]['id']+':'+queue['work'][0]['acceptance']],
    remote_status='joint176failedafterpartialfrozenC8onworkerETlimit,originalruntime/scriptrestoredHTTP200,all exactcontroller/CPUhandles terminal. No defaultnewNPUacquisition.')
m=read(O/'checkpoint-manifest-v176.json');assert all(pin(r['path'])==r for r in m['artifacts'])
post={'status':'PASS','checkpoint':176,'manifest_artifacts':len(m['artifacts']),'all_pins_verified':True,
    'original_service_HTTP':200,'all_exact_handles_terminal':True,'overall_complete':False}
write(O/(prefix+'-postcheck-result-v176.json'),post);write(N/'CURRENT-POSTCHECK.json',post)
write(N/'LIVE-SHARED-COHORT.json',{'effective_checkpoint':176,'status':'JOINT176_AND_CPU_ANALYSIS_TERMINAL_ORIGINAL_SERVICE_RESTORED',
    'receipt':str(H/'final-terminal-and-service-receipt.json'),'all_exact_handles_terminal':True,
    'current_full_case_valid':False,'original_service_HTTP':200,'no_default_new_NPU_acquisition':True,'overall_complete':False})
write(N/'NEXT-EVIDENCE-PENDING.json',{'effective_checkpoint':176,'status':'ONE_EXACT_UNANALYZED_SCOPE_CLOSURE_AUDIT',
    'next_action':action,'work_queue':str(O/'coverage-work-queue-v176.json'),'overall_complete':False})
print(json.dumps(post))
