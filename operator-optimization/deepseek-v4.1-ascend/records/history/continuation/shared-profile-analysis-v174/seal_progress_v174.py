"""Seal one substantive frontier/composition/P5 advance;do not repeat old runs."""
import json
import sys
from copy import deepcopy
from pathlib import Path

H=Path(__file__).resolve().parent
sys.path.insert(0,str(H));sys.path.insert(1,str(H.parent))
from finalize_origin_coverage_v9 import N,O,R,read,write
from analyze_evidence_batches_v155 import pin
from persist_operator_review_delta import persist_delta

old=read(O/'REVIEW-CHECKPOINT-v173.json')
prefix='operator-effect-and-composition'
checks={
    'remaining900_effect_checks':read(H/'remaining900-review-checks.json')['status']=='PASS',
    'all42_composition_checks':read(H/'all42-composition-checks.json')['status']=='PASS',
    'all1101_static_bijection':read(H/'all1101-static-operator-frontier-reconciliation.json')['original_unreviewed_static_frontier_remaining']==0,
    'P5_explicit94_comparisons':read(H/'P5-contract-checks.json')['comparison_count']==94,
    'v175_original_service_restored':read(N/'shared-profile-analysis-v175/restoration-observation-004.json')['health']==200,
    'v176_started_once_with_recorded_pin_correction':read(N/'shared-profile-analysis-v176/submission.json')['pid']==1361847,
}
assert all(checks.values()),checks

authorities={}
for key,stem in [('candidate_ledger','candidate-profile-ledger'),('module_ledger','module-profile-ledger'),('branch_matrix','module-phase-branch-matrix')]:
    src=Path(old[key]);x=read(src)
    x['parent_review_authority']=pin(src)
    x['current_review_checkpoint']=174
    x['current_operator_effect_and_composition_review']={
        'static_frontier':str(H/'all1101-static-operator-frontier-reconciliation.json'),
        'composition':str(H/'all42-composition-coverage.json'),
        'P5':str(H/'module-analysis.json'),
        'evidence_policy':str(H/'current-conclusion-changing-evidence-policy.json'),
        'original900_navigation_frontier_has_operator_effect_answers':True,
        'all_registered35_plus7_have_composition_contracts':True,
        'native_or_all_possible_candidate_exhaustiveness_not_promoted':True,
    }
    dst=O/(stem+'-v174.json');write(dst,x);authorities[key]=str(dst)

changes=[
    'Previously900unreviewed navigation sites matched exactASTnodes/80owners and reviewed in35operator-effectdomains,including full early-return continuations;no900individual framework tasks or repeated source reads.',
    'Exact original1101site inventory reconciled as disjoint110prior localguard reviews+91v173lifecycle reviews+900v174effect reviews;original static frontier0remaining. This is finite source-conditional coverage,not live alias/branch proof.',
    'All35model+7framework registeredfamilies bound to22source/hosttemplates and12global compositionconstraints. Remaining23families reviewed withoutCartesianpairs;original35atomunions and12G5sourcetemplates preserved.',
    'Currentservice API/metrics prove twoengine cacheconfigs agree on kv_sharing_fast_prefill=False and mamba_cache_mode=none;9guardexpressions partially evaluated,unresolvedfields keptunknown.',
    'Canonical module-analysis now carries94explicitP5records:26symbolicwholemodule contracts,42unimplementedconditionalcompositions and26matchedpartial dense-mode populations. Each retains Q,T,L,Q/T,U,L/T,T/L,metric,dominance and evidence/status fields;unknown notzero,1800MHz ratedreference notphysicalfmax.',
    'v175 NPU interface selftestpassed16distinctdevices(nonowningstorage/result/alias),but diagnosticstartup exited beforefullcase. Controllerrestored originalscript/runtime;independenthealth200 andexactserveSHAverified. No fullcase or modelalias conclusion promoted.',
    'v176 combinedET/bank/scalaracquisition live under exactPID1361847/ticks541381757;no repeatedNPUselftest,no processwidefilesizelimit,independent diagnosticlog. Copieddiagnosticpinerror repaired withone recorded extraownedstartupreload beforefullcase;allsourcehashes exact,modelcode unchanged.',
]
write(O/(prefix+'-preflight-v174.json'),{'status':'PASS','preflight':True,'checks':checks})
inputs=[p for p in H.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
inputs += [N/'shared-profile-analysis-v175/all16-NPU-interface-selftest-results.json',
           N/'shared-profile-analysis-v175/restoration-observation-004.json',
           N/'shared-profile-analysis-v176/submission.json',N/'shared-profile-analysis-v176/pin-correction.json',
           N/'shared-profile-analysis-v176/observation-004.json']
write(O/(prefix+'-inputs-v174.json'),{'inputs':[pin(p) for p in sorted(set(inputs))],'overall_complete':False})
write(O/(prefix+'-summary-v174.json'),{
    'status':'PROGRESS','previous_goal_turn':'PROGRESS','overall_complete':False,
    'new_native_operand_or_phase_promotions':0,'new_whole_module_closures':0,
    'source_sites_newly_effect_reviewed':900,'source_effect_domains':35,
    'original_static_frontier_sites_reconciled':1101,'original_unreviewed_static_frontier_remaining':0,
    'registered_families_composed':42,'semantic_rows_retained':360,
    'P5_contracts':94,'new_model_optimization_implementations':0,
    'live_collector':'joint176','live_exact_handle':read(N/'shared-profile-analysis-v176/submission.json')})

action=(
    '继续原完整目标，checkpoint174。900/80owner/35effectdomains与全部1101原静态点已审，不再把旧900导航关联当未审或重复读；42候选组合已按22模板/12全局约束审完登记范围，不做配对穷举。'
    '26模块+42候选+26matched partial-mode的94条P5比较已在shared-profile-analysis-v174/module-analysis.json；缺值符号化，1800仅参考，不再查fmax/要求全局物理MFU或实现提速。'
    '优先观察已运行joint176的准确PID1361847/start_ticks541381757，不因一次工具超时或旧日志错误重启。远端analysis/joint176/STATUS.json与独立diagnostic-service.log；当前已通过更正pin后的导入并有worker_load_model_return。v175仅16NPU接口自检成功、fullcase0且原服务恢复health200，不当模型证据。'
    'v176保留PIN-CORRECTION记录的一次额外诊断启动重载；控制器原model_reloads字段不含这次，不漏报。等自动恢复原服务后核验health/serveSHA/workerFINAL/冻结C8与8texts，再CPU解析ET和bank/scalar元数据，不用诊断耗时替代v164性能。'
    '真正剩余两项：一份当前一致的公开tensor/bank/control G0补充及消费者/epoch绑定；按当前声明G1/G5/G6/G7/G8/G9与补充图做一次完整节点/边/guard/重写规则审计，记录新非等价机会并闭合全轮。'
    '13旧partial多数已有明确公开语义条件答案；只对选中改写依赖的运行前提补证。不同ETstorageID不自动独立，诊断RF marker不是基线必需消费者，privatehelper删除不等于公开函数替换。保留全部模块/条件/op/layer/card/iteration范围，真正未分析前不标Goal完成。')
queue={'version':174,'parent':pin(O/'coverage-work-queue-v173.json'),'scope_preserved_by':str(old['coverage_obligations']),
       'work':[
           {'id':'Q174-CURRENT-G0-BINDING','state':'JOINT176_LIVE_OBSERVE_EXACT_HANDLE','action':action,'evidence':[str(N/'shared-profile-analysis-v176/submission.json'),str(H/'current-conclusion-changing-evidence-policy.json')]},
           {'id':'Q174-FULL-RULE-PASS','state':'FINITE_SCOPE_FINAL_RULE_AUDIT_NEXT','action':'Reuse854source-node/1346edge coverage and all42composed contracts;integrate currentpublicgraph supplement,review alldeclaredrewrite rules once,record any new non-equivalentopportunity. NoCartesianpairs.','evidence':[str(H/'all42-composition-coverage.json'),str(H/'all1101-static-operator-frontier-reconciliation.json')]}],
       'completed_frontiers':['Q173-REMAINING-OPERATOR-EFFECT-FRONTIER','Q173-ALL42-COMPOSITION-COVERAGE'],
       'overall_complete':False}
gate={'version':174,'parent':pin(Path(old['source_gate'])),'policy':'PROFILE_FIRST_THEN_OPERATOR_SPECIFIC_CONTRACT_ONLY',
      'no_reopening_reviewed900_91_or110_sites':True,'no_independence_from_blank_evidence':True,
      'no_new_source_read_unless_profile_or_full_rule_audit_names_missing_operator_fact':True,
      'operator_evidence_policy':str(H/'current-conclusion-changing-evidence-policy.json'),'overall_complete':False}
overlay={'version':174,'changes':changes,'authorities':authorities,
         'static_frontier':str(H/'all1101-static-operator-frontier-reconciliation.json'),
         'P5':str(H/'module-analysis.json'),'all42_composition':str(H/'all42-composition-coverage.json'),
         'new_whole_module_closures':0,'overall_complete':False}
obligations=read(Path(old['coverage_obligations']))['obligations']
recs={r['id']:{'current_evidence_policy':str(H/'current-conclusion-changing-evidence-policy.json'),
    'completed_original_static_frontier':str(H/'all1101-static-operator-frontier-reconciliation.json'),
    'registered42_composition_complete':str(H/'all42-composition-coverage.json'),
    'P5_explicit_contracts':str(H/'module-analysis.json'),
    'current_work':'Current coherent public graph binding and final declared full-rule pass. OldprivateABI/physicalMFU/attainablegain notuniversalanalysisgates.',
    'whole_domain_closed_by_this_increment':False,'complete_original_scope_preserved':True} for r in obligations}
report='''本次进度为 PROGRESS，完整目标仍在推进。

剩余900处静态条件点已沿80个函数归并为35类算子效应，检查输入/输出、精度、所有权、状态、RNG、同步与 early return 后续效应。与既有110处局部检查、上轮91处生命周期检查形成1101个原始静态点的精确、不重叠覆盖；原静态 frontier 已无未审点。这个结论限于声明的源码清单，不等于所有运行时别名与分支已被观测。

全部35个模型候选和7个框架候选现已映射到22个共同模板及12项全局资源、舍入、状态和存活期约束。剩余23族已补组合审查，保留原始节点与G5模板，避免笛卡尔配对、现有F21融合重复计功，以及独立理想提速相乘。

理论比较已落成94条规范记录：26个完整功能模块的条件/符号模型，42个未实现候选的组合模型，26组工作量与累计kernel时间相匹配的局部直接稠密模式比较。Q、T、L、Q/T、U、L/T、T/L、活动指标、主导约束和证据状态均显式列出。未知频率/必要流量/模块时间不填0；1800MHz仅额定参考。局部累计时间不冒充请求walltime，候选也不借用基线性能作为自身测量。

当前两路engine缓存指标一致，kv_sharing_fast_prefill=False、mamba_cache_mode=none，9处条件可据此部分求值；其他字段保持未知。未再重复检查旧profile缺失字段或重新阅读相同算子代码。

合并取证使用原生PyTorch Execution Trace与公开tensor/bank/scalar绑定。v175在16个不同NPU上验证接口、view/alias、结果及不额外持有storage成功，但诊断启动随后退出，冻结负载未执行。控制器恢复原服务，独立health=200且原脚本SHA一致；启动失败原日志被恢复覆盖，未虚构根因。

v176已改为独立日志、不设全进程文件大小限制、不重复NPU小自检。复制版本号时误改诊断器中的一个hash，已修正并单独记录一次额外的我方容器启动重载；目标框架/模型源码未改。当前同一控制器PID1361847、start_ticks541381757继续加载，已观察到worker_load_model_return。继续观察该handle，采集结束自动恢复原服务；不因旧错误或观察超时重启。

剩余工作集中为两项：把一个当前一致的公开tensor、bank、控制选择和消费者图补齐；利用已有G1/G5/G6/G7/G8/G9节点/边账本与当前补充图，完成一轮声明规则的整体审计，判断是否还产生新非等价机会。已明确回答的条件语义不因私有helper、历史epoch或物理MFU缺口而重新打开；确实影响选中改写的未知值继续补证。仍有未分析范围，未把报告成稿当作目标完成。
'''
updates={
    str(R/'module-analysis.json'):{'current_whole_framework_analysis_path':str(H/'module-analysis.json'),'current_functional_module_count':26,'current_registered_candidate_count':42,'current_P5_comparison_count':94},
    str(R/'fusion-scan.json'):{'current_operator_effect_frontier':str(H/'all1101-static-operator-frontier-reconciliation.json'),'current_registered_composition_coverage':str(H/'all42-composition-coverage.json'),'current_remaining_full_scope_work':queue['work'],'whole_model_exhaustion_proven':False},
    str(R/'records.json'):{'current_operator_evidence_policy':str(H/'current-conclusion-changing-evidence-policy.json'),'current_module_comparison_authority':str(H/'module-analysis.json')},
    str(R/'bounds-input.json'):{'current_functional_module_and_candidate_P5_contracts':str(H/'module-analysis.json'),'rated1800_not_physical_fmax':True},
}
jobs=[{'kind':'review','status':'pass','description':'900exactsites/80owners reviewedin35operator-effectdomains;original1101sitebijection reconciled,no remaining oldstaticfrontier.','artifacts':[H/'remaining900-operator-dataflow-review.json',H/'remaining900-review-checks.json',H/'all1101-static-operator-frontier-reconciliation.json']},
      {'kind':'review','status':'pass','description':'All35+7registeredfamilies composed in22source/hosttemplates with12globalconstraints;noCartesianpairs,no universalexhaustivenessclaim.','artifacts':[H/'all42-composition-coverage.json',H/'remaining23-composed-family-review.json',H/'all42-composition-checks.json']},
      {'kind':'review','status':'pass','description':'94canonicalP5records cover26modulecontracts,42conditionalcompositions,26matchedpartialmodepopulations;exactarithmetic andsymbolicunknownskept.','artifacts':[H/'module-analysis.json',H/'P5-contract-checks.json']},
      {'kind':'review','status':'error','description':'v175all16NPUinterfaceselftests passedbutdiagnosticstartupexitedbeforefullcase;originalservicerestoredHTTP200,scriptSHAexact.No modelclosurepromoted.','artifacts':[N/'shared-profile-analysis-v175/all16-NPU-interface-selftest-results.json',N/'shared-profile-analysis-v175/restoration-observation-004.json']},
      {'kind':'review','status':'pass','description':'v176collectorpreflight/sourcepins passed;submittedone exacthandleandrecordeddiagnosticpinfix/extraownedstartupreload.Collectionstilllive,notdeclaredsuccessful.','artifacts':[N/'shared-profile-analysis-v176/submission.json',N/'shared-profile-analysis-v176/pin-correction.json',N/'shared-profile-analysis-v176/observation-004.json']}]
persist_delta(version=174,previous=173,prefix=prefix,changes=changes,report=report,action=action,
    remaining_delta='Current observed public graph binding and final full-rule pass remain.',
    source_gate=gate,operator_overlay=overlay,queue=queue,jobs=jobs,scripts=list(H.glob('*.py')),
    obligation_reconciliations=recs,authority_overrides=authorities,shared_json_updates=updates,
    pending_full_scope_domains=[r['id']+':'+r['action'] for r in queue['work']],
    remote_status='joint175 terminalfailedbeforefullcase/originalrestoredHTTP200; joint176livePID1361847 ticks541381757,extraownedstartupreload1 recorded in PIN-CORRECTION,automaticoriginalrestorepending.')
manifest=read(O/'checkpoint-manifest-v174.json');assert all(pin(r['path'])==r for r in manifest['artifacts'])
post={'status':'PASS','checkpoint':174,'manifest_artifacts':len(manifest['artifacts']),'all_pins_verified':True,'overall_complete':False}
write(O/(prefix+'-postcheck-result-v174.json'),post);write(N/'CURRENT-POSTCHECK.json',post)
write(N/'LIVE-SHARED-COHORT.json',{'effective_checkpoint':174,'status':'JOINT176_NATIVE_ET_COLLECTION_LIVE',
    'handle':read(N/'shared-profile-analysis-v176/submission.json'),'last_observation':str(N/'shared-profile-analysis-v176/observation-004.json'),
    'extra_owned_diagnostic_startup_reload':1,'original_service_restore_pending':True,'overall_complete':False})
write(N/'NEXT-EVIDENCE-PENDING.json',{'effective_checkpoint':174,'status':'JOINT176_AND_FINAL_RULE_PASS','work_queue':str(O/'coverage-work-queue-v174.json'),'next_action':action,'overall_complete':False})
print(json.dumps(post))
