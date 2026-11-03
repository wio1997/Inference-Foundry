"""Publish the dependency corrections without turning structural checks into scope proof."""
import json
import sys
from copy import deepcopy
from pathlib import Path
H=Path(__file__).resolve().parent
N=H.parent;O=N/'profile-first-20261031';H177=N/'shared-profile-analysis-v177'
sys.path.insert(0,str(H177));sys.path.insert(1,str(N))
from finalize_origin_coverage_v9 import R,read,write
from analyze_evidence_batches_v155 import pin
from persist_operator_review_delta import persist_delta

prefix='public-dependency-and-state-ports';v=178
old=read(O/'REVIEW-CHECKPOINT-v177.json')
manifest=read(O/'checkpoint-manifest-v177.json')
bad=[r['path'] for r in manifest['artifacts'] if pin(r['path'])!=r]
assert not bad,bad
write(H/'predecessor-pin-check.json',{'status':'PASS','checkpoint':177,'pins_verified':len(manifest['artifacts']),'mismatches':bad})
for name in ['guard-review-checks.json','public-dependency-overlay-checks.json','boundary-state-port-checks.json','assembled-public-G0-checks.json','independent-overlay-checks.json']:
    assert read(H/name)['status']=='PASS',name

action=('核验27条件效应域和15状态端口合同对360语义行的实质覆盖：只检查会改变融合/拆分/重排结论的消费者、别名与状态效应；'
        '复用现有profile和合同，不重复读取整批代码或枚举候选配对。'
        '区分真正缺失的公共效应与可保持原接口的私有未知；仅前者或足以改变候选结论的条件才开定点源码/探针。')
queue={'version':v,'parent':pin(O/'coverage-work-queue-v177.json'),'work':[
 {'id':'Q178-SEMANTIC-PORT-SUFFICIENCY','state':'CONCLUSION_CHANGING_PUBLIC_EFFECT_ACCEPTANCE',
  'action':action,'evidence':[str(H/'exact-guard-operator-contract-review.json'),str(H/'hierarchical-G0-with-boundary-and-state-ports.json'),
     str(H/'host-score-and-state-port-contracts.json'),str(H/'delta-rule-and-evidence-sufficiency-review.json')],
  'remaining':'Exact reference coverage and graph construction are done;independent semantic adequacy of their public consumer/state envelopes is not yet certified.',
  'not_required':'No Cartesian pairs,no universal private-alias/MFU/opcode proof,no new fullC8 acquisition by default.'}],
 'completed_previous_queue_outputs':{
   'Q177-EXACT-GUARD-CONTRACT-COVERAGE':str(H/'exact-guard-operator-contract-review.json'),
   'Q177-HIERARCHICAL-G0-PORT-CLOSURE':str(H/'hierarchical-G0-with-boundary-and-state-ports.json')},
 'structural_outputs_are_not_full_scope_semantic_acceptance':True,
 'container_process_and_probe_authorization':'User explicitly permits owned-container inference/framework process operations and necessary probes;do not ask again.',
 'no_default_new_NPU_acquisition':True,'overall_complete':False}

audit=deepcopy(read(H177/'current-requirement-audit.json'))
status_changes={
 'ALL_MODULE_CONDITIONAL_BRANCHES':('EXACT1867_REFERENCE_TO27_OPERATOR_DOMAIN_ANSWERS_PRESENT;SEMANTIC_ADEQUACY_ACCEPTANCE_PENDING',
    'Original1101reviewed static guards retained. New1867exactrefs each assigned one operator effect-domain answer. Broad-domain assignment is not by itself proof that every conclusion-changing continuation/consumer is accounted for.'),
 'CURRENT_FINITE_GRAPH_LEDGER':('HIERARCHICAL_PUBLIC_G0_AND_ALIAS_STATE_PORTS_BUILT;SEMANTIC_SUFFICIENCY_PENDING',
    '2163semantic/effect nodes and2965edges include40exact target attention roles,122materialized model templates,10host/score boundary templates,15state envelopes and all42family refs. Removed obsolete DSpark context->block0 edge;corrected Engram selected host-UVA/overlapFalse case. Counts/endpoints do not prove all-consumer adequacy.'),
 'CROSS_OPERATOR_LAYER_CARD_ITERATION':('REGISTERED42_AND7_CHANGED_DEPENDENCY_RULES_ANSWERED;WHOLE_SCOPE_ACCEPTANCE_PENDING',
    '8TopK source epochs and38reader layers have range RAW/WAR/WAW including next-iteration reuse. Candidate and state port bindings retain rank/group/slot/generation/RNG readers. No Cartesian pair debt or new unconditional native promotion.'),
 'CRITICAL_EVIDENCE_AND_FINAL_CLOSURE':('NOT_ACHIEVED_PUBLIC_EFFECT_CONTRACT_SUFFICIENCY_REVIEW_REMAINS',
    'One finite acceptance step remains:judge semantic coverage of the guarded domains and state/consumer envelopes against existing360rows and profile/source evidence. Do not substitute counts,report completion or global private-ABI proof for this review.')}
for r in audit['requirements']:
    if r['id'] in status_changes:
        r['state'],r['finding']=status_changes[r['id']]
        r['completion_proved_for_analysis_contract']=False
        r['evidence']=[str(H/'exact-guard-operator-contract-review.json'),str(H/'hierarchical-G0-with-boundary-and-state-ports.json'),str(H/'delta-rule-and-evidence-sufficiency-review.json')]
audit.update(status='CURRENT_REQUIREMENTS_WITH_PUBLIC_DEPENDENCY_PROGRESS',previous_goal_turn='PROGRESS',overall_complete=False)
write(H/'current-requirement-audit.json',audit)

bindings=read(H/'candidate-to-public-hierarchy-bindings.json')['rows'];bm={r['candidate']:r for r in bindings}
states=read(H/'host-score-and-state-port-contracts.json')['state_envelopes']
authorities={}
for key,stem in [('candidate_ledger','candidate-profile-ledger'),('module_ledger','module-profile-ledger'),('branch_matrix','module-phase-branch-matrix')]:
    p=Path(old[key]);x=deepcopy(read(p));x.update(parent_review_authority=pin(p),current_review_checkpoint=v)
    x['current_public_dependency_and_state_review']={
      'guarded_operator_answers':str(H/'exact-guard-operator-contract-review.json'),
      'hierarchical_G0':str(H/'hierarchical-G0-with-boundary-and-state-ports.json'),
      'candidate_bindings':str(H/'candidate-to-public-hierarchy-bindings.json'),
      'state_ports':str(H/'host-score-and-state-port-contracts.json'),
      'delta_rule_review':str(H/'delta-rule-and-evidence-sufficiency-review.json'),
      'public_effect_semantic_acceptance_pending':True,'native_eligibility_or_module_completion_promoted':False}
    if key=='candidate_ledger':
        for c in x['candidates']:c['current_public_dependency_binding_v178']=bm[c['id']]
    else:
        for m in x['modules']:m['current_state_envelopes_v178']=[s['id'] for s in states if m['id'] in s['modules']]
    q=O/(stem+'-v178.json');write(q,x);authorities[key]=str(q)

changes=[
 'Previous goal turn PROGRESS independently revalidated:all1692v177manifest pins match;no prior profile/source acquisition repeated.',
 '1867exact guards mapped one-to-one into27operator effect-domain answers,reusing original1101static reviews and360semantic rows. This is conditional answer coverage,not runtime branch/native legality proof.',
 'Pinned config and two targeted build_layer_plan functions instantiate40target roles:2SWA,3C2source,15C2consumer,1C1source,4C1indexsource,15C1consumer. Draft40/41/42 and loaded selector snapshots retained.',
 'Built2163node2965edge public conditional hierarchy:122materialized model-template instances,10host/score boundary templates,15state/alias envelopes,all26module ports and42candidate bindings. Macro/refinement/case accounting kept separate.',
 'Corrected two conclusion-relevant historical graph errors:Engram lookup has positive NPU host-UVA gather/dequant and overlapFalse snapshot;DSpark context projection feeds shared KV,not draft block0query hidden.',
 'Added8shared TopK value epochs and38target attention readers with per-range RAW/WAR/WAW and next-iteration retirement. Source24index/source20KV example separates source namespaces. Version-buffer cost R_live*512*4 is conditional,not universal16MiB traffic/allocation/gain.',
 'Seven declared rules reviewed only changed dependencies;no new non-equivalent family,no Cartesian pair enumeration,no new performance or unconditional native eligibility claim.',
 'Only offline CPU evidence work this turn;no server login,NPU context,model call or restart. Last service verification belongs to v177 and is not presented as a new health check.',
 'Whole goal remains open for one finite semantic sufficiency review. Exact reference/node counts alone do not prove full effect/consumer coverage;unchanged opaque native interfaces need not become universal blockers.'
]
write(H/'verified-changes.json',{'status':'PROGRESS','changes':changes,'overall_complete':False})

report=(H177/'REPORT.md').read_text()
report=report.replace('已有的1101处精确静态点完成算子效应审查；全部42个登记族有组合约束，27份源码模板的854节点/1346边完成一轮7规则条件审查。完整模型范围仍需把精确条件集合与这些答案对齐，并核齐分层G0的消费者、别名、状态与跨owner端口。',
 '已有1101处精确静态点的算子效应审查继续复用；另外1867处精确条件已接到27个算子效应域。当前分层公共图包含2163个语义/状态节点、2965条依赖或效应边，绑定40层目标网络、3层草稿边界、26模块及42候选族。引用和端点检查通过，但条件域/状态端口的实质覆盖仍待独立核验，尚不构成整目标完成证明。')
for r in audit['requirements']:
    prior=next(a for a in read(H177/'current-requirement-audit.json')['requirements'] if a['id']==r['id'])
    report=report.replace('|'+r['id']+'|'+prior['state']+'|','|'+r['id']+'|'+r['state']+'|')
old_next='下一步只做两项范围检查：精确条件到已有算子合同的覆盖对应；分层G0的调用、消费者、别名和共享状态端口对应。不得把已回答的条件未知、历史私有ABI、物理fmax或未实施提速重新当作通用阻塞；也不得在真实未分析范围仍在时因本页成稿标完成。'
report=report.replace(old_next,'本轮修正了三个会影响融合判断的依赖：①Engram基线为已观测的NPU host-UVA lookup，overlap标志为False；②DSpark上下文经共享KV供3个query块使用，不能和query embedding合并成一条输入；③TopK共享16MiB backing包含8个源层值代际、38层读者，重排必须保护重叠范围的最后读者和下一轮覆写。缓冲地址相同不代表索引值相同。\n\n下一步只有一项验收：检查27个条件效应域与15个状态端口合同是否实质覆盖360条语义行中所有会改变融合结论的公共效应。只对明确漏项定点取证；保持原接口的私有未知、物理fmax或未实施性能不自动变成全局阻塞。未完成该核验前，整体目标继续保持未完成。')
report+='\n\nv178公共依赖图：[hierarchical-G0-with-boundary-and-state-ports.json]('+str(H/'hierarchical-G0-with-boundary-and-state-ports.json')+')；候选状态绑定：[candidate-to-public-hierarchy-bindings.json]('+str(H/'candidate-to-public-hierarchy-bindings.json')+')；本轮规则审查：[delta-rule-and-evidence-sufficiency-review.json]('+str(H/'delta-rule-and-evidence-sufficiency-review.json')+')。本轮仅离线处理已有证据，没有新建NPU上下文、执行推理或重启服务。\n'
report=report.replace(str(H177/'current-requirement-audit.json'),str(H/'current-requirement-audit.json'))
(H/'REPORT.md').write_text(report)

gate={'version':v,'parent':pin(Path(old['source_gate'])),
 'policy':'PROFILE_AND_EXISTING_OPERATOR_CONTRACT_FIRST;ONLY_NAMED_CONCLUSION_CHANGING_EFFECT_JUSTIFIES_TARGET_BODY_OR_PROBE',
 'new_targeted_source_read':['Pinned model.py:_latest_source/build_layer_plan only for exact layer-role instantiation'],
 'original1101_and72file_census_not_reexecuted':True,'user_container_process_probe_authorization_already_present':True,
 'no_cartesian_pair_work':True,'no_default_full_NPU_acquisition':True,'overall_complete':False}
overlay={'version':v,'changes':changes,'authorities':authorities,'public_G0':str(H/'hierarchical-G0-with-boundary-and-state-ports.json'),
 'guard_answers':str(H/'exact-guard-operator-contract-review.json'),'state_envelopes':str(H/'host-score-and-state-port-contracts.json'),
 'remaining_semantic_acceptance':queue['work'],'new_whole_module_closures':0,'overall_complete':False}
summary={'status':'PROGRESS','previous_goal_turn':'PROGRESS','overall_complete':False,
 'new_native_operand_or_phase_promotions':0,'new_whole_module_closures':0,'current_requirements':9,
 'current_answered_analysis_requirements':sum(r['completion_proved_for_analysis_contract'] for r in audit['requirements']),
 'guard_refs_with_conditional_domain_answers':1867,'guard_effect_domains':27,'public_G0_nodes':2163,'public_G0_edges':2965,
 'target_layer_roles':40,'draft_layer_roles':3,'state_envelopes':15,'candidate_bindings':42,'changed_dependency_rules':7,
 'new_NPU_context_model_or_restart_calls':0,'fresh_service_health_check':False,
 'source_scope_read':'Only cached pinned build_layer_plan/_latest_source targeted for source-role binding;no repeated source census.'}
write(O/(prefix+'-summary-v178.json'),summary)
inputs=[Path(r['path']) for r in read(H/'hierarchical-G0-public-dependency-overlay.json')['pins']]
inputs+=list(H.glob('*.json'))+list(H.glob('*.py'))+[H/'REPORT.md']
write(O/(prefix+'-inputs-v178.json'),{'inputs':[pin(p) for p in sorted(set(inputs))],'overall_complete':False})
write(O/(prefix+'-preflight-v178.json'),{'status':'PASS','preflight':True,
 'checks':{'previous1692pins_match':not bad,'all_independent_overlay_checks_pass':read(H/'independent-overlay-checks.json')['status']=='PASS',
           'no_native_or_whole_module_completion_claim':True}})

prior_obligations=read(Path(old['coverage_obligations']))['obligations']
remaining={r['id']:r['remaining'] for r in prior_obligations}
remaining.update({
 'G00':'26modules/360rows/42families retained. Exact1867guard answers and hierarchicalG0built;Q178semantic public-effect sufficiency acceptance remains.',
 'G02':'Frozen request/source/1104observed greedy calls retained;27conditional guard-domain answers nowpresent. Check their semantic effect/consumer adequacy without requiring unobserved branches to run.',
 'G05':'Shared TopK8sourceepochs/38readers/rangeWAR/WAW and15state envelopes explicit. Semantic sufficiency of preserved public consumer/alias/rollback ports remains;stable addresses alone still do not attest retirement.',
 'G07':'All42family bindings andstate envelopes explicit;7changed-dependency rules answered. Currentpendingwork is Q178semantic sufficiency,not old212/383paircounts.',
 'G09':'Hierarchical2163node2965edge publicgraph nowconstructed with40targetroles. Exact references/endpoints do not prove all effects covered;check existing semantic contracts instead of collecting another inventory.',
 'G10':'35model+7hostfamilies composed withsharedresource/state constraints;whole-scopecertificate awaits Q178semantic sufficiency,not extra pair enumeration.',
 'G13':'Alloriginal360rows/1101staticreviews preserved and1867newrefs routed to27effectanswers. Independentlycheck public-effectadequacy;do not reopen the same source set.',
 'G14':'10concretehost/scoreboundarytemplates and15stateenvelopes nowretaincopy/control/publication/lifetime. NativeDMAprivatefields remain conditional forprivateedits;no blanket probe gate.'})
ob_answers={r['id']:{'remaining':remaining[r['id']],'current_evidence_authority':str(H/'current-requirement-audit.json'),
 'current_work_queue':str(O/'coverage-work-queue-v178.json'),'legacy_wording_is_historical_after_current_reconciliation':True} for r in prior_obligations}
updates={
 str(R/'STATE.json'):{'fusion_scan_status':'PUBLIC_HIERARCHY_AND_CHANGED_RULE_ROUND_DONE_SEMANTIC_PORT_SUFFICIENCY_PENDING',
   'current_requirement_audit':str(O/'whole-scope-requirement-audit-v178.json'),'current_public_operator_graph':str(H/'hierarchical-G0-with-boundary-and-state-ports.json')},
 str(R/'fusion-scan.json'):{'current_requirement_audit':str(O/'whole-scope-requirement-audit-v178.json'),
   'current_public_operator_graph':str(H/'hierarchical-G0-with-boundary-and-state-ports.json'),
   'current_changed_dependency_rule_round':str(H/'delta-rule-and-evidence-sufficiency-review.json'),
   'current_remaining_full_scope_work':queue['work'],'whole_model_exhaustion_proven':False},
 str(R/'records.json'):{'current_hierarchical_public_G0':str(H/'hierarchical-G0-with-boundary-and-state-ports.json'),
   'current_candidate_state_bindings':str(H/'candidate-to-public-hierarchy-bindings.json'),
   'current_guard_operator_answers':str(H/'exact-guard-operator-contract-review.json')},
 str(R/'module-analysis.json'):{'current_public_dependency_review':str(H/'hierarchical-G0-with-boundary-and-state-ports.json'),
   'current_state_port_contracts':str(H/'host-score-and-state-port-contracts.json'),'comparison_values_unchanged_v178':True}}
jobs=[
 {'kind':'review','status':'pass','description':'Reuse validated1692pins;1867exactguardrefs have27conditional operator-domain answers,not runtime legality proof.',
  'artifacts':[H/'predecessor-pin-check.json',H/'exact-guard-operator-contract-review.json',H/'guard-review-checks.json']},
 {'kind':'review','status':'pass','description':'Exact40layerroles,2163node2965edge publichierarchy,42bindings,15stateenvelopes;correct Engram/DSpark ancestry and TopKepoch range/iteration constraints. No NPU/model/restart.',
  'artifacts':[H/'layer-role-and-selector-bindings.json',H/'hierarchical-G0-with-boundary-and-state-ports.json',H/'host-score-and-state-port-contracts.json',H/'candidate-to-public-hierarchy-bindings.json',H/'independent-overlay-checks.json']},
 {'kind':'review','status':'pass','description':'Sevenchanged-dependency ruleanswers;no newequivalent-family task/pairenumeration/nativepromotion. Fullsemantic public-effectacceptance remains explicit.',
  'artifacts':[H/'delta-rule-and-evidence-sufficiency-review.json',H/'current-requirement-audit.json',H/'REPORT.md']}]
persist_delta(version=v,previous=177,prefix=prefix,changes=changes,report=report,action=action,
 remaining_delta=queue['work'][0]['remaining'],source_gate=gate,operator_overlay=overlay,queue=queue,jobs=jobs,
 scripts=list(H.glob('*.py')),authority_overrides=authorities,shared_json_updates=updates,
 shared_text_updates={str(R/'REPORT.md'):report},requirement_audit_data=audit,obligation_current_answers=ob_answers,
 pending_full_scope_domains=[queue['work'][0]['id']+':'+queue['work'][0]['remaining']],
 remote_status='OfflineCPUonly;no remotecommands,NPUcontext,inference orrestart thisturn.Lastservicehealth200 belongs tov177,not a freshcheck.')
new=read(O/'checkpoint-manifest-v178.json');assert all(pin(r['path'])==r for r in new['artifacts'])
post={'status':'PASS','checkpoint':v,'manifest_artifacts':len(new['artifacts']),'all_pins_verified':True,
      'public_G0_and_rule_delta_published':True,'no_new_NPU_model_restart':True,'overall_complete':False}
write(O/(prefix+'-postcheck-result-v178.json'),post);write(N/'CURRENT-POSTCHECK.json',post)
write(N/'LIVE-SHARED-COHORT.json',{'effective_checkpoint':v,'status':'OFFLINE_REVIEW_NO_REMOTE_ACTION',
 'last_service_receipt':str(H177/'final-CPU-and-service-receipt.json'),'new_NPU_model_restart':0,
 'no_default_new_NPU_acquisition':True,'overall_complete':False})
write(N/'NEXT-EVIDENCE-PENDING.json',{'effective_checkpoint':v,'status':'SEMANTIC_PUBLIC_EFFECT_SUFFICIENCY_REVIEW',
 'next_action':action,'work_queue':str(O/'coverage-work-queue-v178.json'),'overall_complete':False})
print(json.dumps(post))
