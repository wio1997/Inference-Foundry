import copy
import json
import sys
from pathlib import Path
H=Path(__file__).resolve().parent;N=H.parent;O=N/'profile-first-20261031';P=N/'shared-profile-analysis-v178'
sys.path.insert(0,str(N/'shared-profile-analysis-v177'));sys.path.insert(1,str(N))
from finalize_origin_coverage_v9 import R,read,write
from analyze_evidence_batches_v155 import pin
from persist_operator_review_delta import persist_delta
prefix='semantic-consumer-and-alias-review'
old=read(O/'REVIEW-CHECKPOINT-v178.json')
assert read(H/'predecessor-pin-check.json')['status']=='PASS'
assert read(H/'semantic-graph-refinement-checks.json')['status']=='PASS'
assert read(H/'semantic-review-checks.json')['status']=='PASS'
review=read(H/'semantic-contract-sufficiency-review.json')
graph=read(H/'hierarchical-G0-with-boundary-and-state-ports.json')
assert len(graph['nodes'])==2203 and len(graph['edges'])==3023

action=('按原目标及P0–P7逐项完成当前状态验收：固定模型/源码/CANN/硬件/负载、26模块/360语义行及精确条件域、'
        '修正后的G0/42候选/跨层跨卡跨迭代规则、94条P5对照、分组最低模块与原因、资料及证据引用。'
        '只对真正缺失或太弱的结论证据继续工作；条件化私有ABI/原始物理速率不自动变成全局阻塞，也不得只凭引用/计数或报告成稿标完成。')
queue={'version':179,'parent':pin(O/'coverage-work-queue-v178.json'),'work':[
 {'id':'Q179-ORIGINAL-OBJECTIVE-COMPLETION-AUDIT','state':'REQUIREMENT_BY_REQUIREMENT_CURRENT_EVIDENCE_AUDIT',
  'action':action,'evidence':[str(H/'semantic-contract-sufficiency-review.json'),str(H/'hierarchical-G0-with-boundary-and-state-ports.json'),
      str(H/'semantic-refinement-rule-round.json'),str(H/'current-requirement-audit.json')],
  'remaining':'Final audit must prove every original requirement against the current artifacts and distinguish conditional eligibility from an unanalyzed range;completion remains unproven.',
  'not_required':'No rerun of the360row review,source census,Cartesian pairs or fullNPUcapture unless a named evidence deficiency changes a conclusion.'}],
 'previous_queue_completed_by':str(H/'semantic-contract-sufficiency-review.json'),
 'user_process_probe_authorization_already_present':True,'no_default_new_NPU_acquisition':True,'overall_complete':False}

audit=copy.deepcopy(read(P/'current-requirement-audit.json'))
changed={
 'ALL_MODULE_CONDITIONAL_BRANCHES':('FIXED_CASE360_ROWS_AND27_GUARDED_EFFECT_DOMAINS_SEMANTICALLY_REVIEWED_WITH_CONDITIONS',
  'All360row answers and27domain answers were read,not inferred from reference counts.18state envelopes nowretain3layer-specific DSparkKV,freshcandidatepublication,cross-rowgathers/orderedcross-planealiases and separateSPmask. Original1101static and1867exactrefs retained;native branch execution remains conditional.'),
 'CURRENT_FINITE_GRAPH_LEDGER':('PUBLIC_CONDITIONAL_G0_RECONCILED_WITH_NAMED_CONSUMER_ALIAS_STATE_CORRECTIONS',
  'Current2203node3023edge hierarchy has40targetroles,3draftlayers,124materializedmodeltemplates,10boundarytemplates and18stateenvelopes.3distinctKVoutputs/caches,512position versus2048candidatebanks and3gather/2orderedwrite semantics are explicit. Opaque native effects remain within preserved publicinterfaces,not assumed absent.'),
 'CROSS_OPERATOR_LAYER_CARD_ITERATION':('ALL42_CONDITIONAL_COMPOSITIONS_AND_SEMANTIC_REFINEMENT_RULE_ROUND_REVIEWED',
  'Prior27template/7rule round and42jointresource/lifetimecontracts reused. All7rules applied to corrected dependency effects,with no newnonequivalentfamily. Per-layer/group/rank/valueepoch/candidate/TopK/queuedwriter constraints remain;no Cartesian pair enumeration or unconditional feasibility/gain claim.')}
for r in audit['requirements']:
    if r['id'] in changed:
        r['state'],r['finding']=changed[r['id']]
        r['completion_proved_for_analysis_contract']=True
        r['evidence']=[str(H/'semantic-contract-sufficiency-review.json'),str(H/'hierarchical-G0-with-boundary-and-state-ports.json'),str(H/'semantic-refinement-rule-round.json')]
    if r['id']=='CRITICAL_EVIDENCE_AND_FINAL_CLOSURE':
        r.update(state='FINAL_ORIGINAL_OBJECTIVE_AND_P0_P7_CURRENT_STATE_AUDIT_NOT_YET_PERFORMED',
          finding='Semantic review completed and four publiceffect refinements applied. Complete only after current source/scope/metrics/graph/ledger/report/records requirements are independently verified,not because this report or structuralcheck passes.',
          completion_proved_for_analysis_contract=False,evidence=[str(H/'semantic-contract-sufficiency-review.json'),str(O/'coverage-work-queue-v179.json')])
audit.update(status='SEMANTIC_REVIEW_COMPLETE_FINAL_COMPLETION_UNPROVEN',previous_goal_turn='PROGRESS',overall_complete=False)
write(H/'current-requirement-audit.json',audit)

limits=[
 {'objects':['M18','F27','F29','F31'],'qualification':'3layer-specificKVwork/norm/cache remains. Jointinput1536projection may reuse loads,not compute one512output and reuse it. Currentwork/PMUnumbers already model3paths and are unchanged.'},
 {'objects':['M09','F09','F10','F12','F13','F19','F31'],'qualification':'candidate2048output is fresh then copied;outABIchange separate fromprepare-indexselfalias. Bitmap8192B->544Blogical/query is conditionalrepresentation,not actualallocation/traffic or measuredgain.'},
 {'objects':['M04','M21','M24','F20','F31','H02','H06','H07'],'qualification':'Batch40N/175Nreferenceinterfacebudget uses an equivalent snapshot protocol.3gathers precedecomputedwrite;acceptedELSE followscomputedwrite. Allsourcehoist needs disjointplanes or exact emulation of that alias effect;preserve2INT32destinations,promotedwhere/overflow/error.'},
 {'objects':['M05','M19','M21','F15','F28','H06','H07'],'qualification':'ProvidedSPmaskTrueprefix andTruepad suffix retain their input values;SPmask andC2mask identities cannot be deduced fromBOOLshape.'}]
write(H/'bound-and-eligibility-qualifications.json',{'status':'CONDITIONAL_DOMAIN_QUALIFICATIONS_NO_NEW_NUMERIC_CLAIM',
 'entries':limits,'94P5_values_and_measured_cohorts_unchanged':True,'overall_complete':False})

binding={b['candidate']:b for b in read(H/'candidate-to-public-hierarchy-bindings.json')['rows']}
mr={m['module']:m for m in review['module_reviews']};rr={r['semantic_row']:r for r in review['semantic_row_reviews']}
authorities={}
for key,stem in [('candidate_ledger','candidate-profile-ledger'),('module_ledger','module-profile-ledger'),('branch_matrix','module-phase-branch-matrix')]:
    p=Path(old[key]);x=copy.deepcopy(read(p));x.update(parent_review_authority=pin(p),current_review_checkpoint=179)
    x['current_semantic_contract_sufficiency_v179']={
      'review':str(H/'semantic-contract-sufficiency-review.json'),'current_G0':str(H/'hierarchical-G0-with-boundary-and-state-ports.json'),
      'current_guard_answers':str(H/'exact-guard-operator-contract-review.json'),'current_state_envelopes':str(H/'host-score-and-state-port-contracts.json'),
      'eligibility_and_bound_qualifications':str(H/'bound-and-eligibility-qualifications.json'),
      'final_goal_completion_audit_pending':True,'conditional_analysis_does_not_prove_native_implementation':True}
    if key=='candidate_ledger':
        for c in x['candidates']:c['current_semantic_dependency_binding_v179']=binding[c['id']]
    else:
        for m in x['modules']:
            m['current_semantic_consumer_review_v179']=mr[m['id']]
            if key=='branch_matrix':
                for r in m['branches']:r['current_semantic_review_v179']=rr[r['id']]
    q=O/(stem+'-v179.json');write(q,x);authorities[key]=str(q)

changes=[
 'PreviousgoalturnPROGRESS independentlyverified:all1744v178pinsmatch. No oldsourcecensus orprofileacquisitionrepeated.',
 'Read all360semanticrow answers across26modules andall27guarded effect-domain answers;manual modulefindings distinguish namedpubliceffects from conditionalnative/privatelegality. Exact1867reference inventory and original1101staticreviews reused.',
 'Semantic audit found and repaired4coarsepubliceffectrelations:3layer-ownedDSparkKV instead ofequalKVfanout,separatecandidate2048freshout/bank/filterreaders,batchcross-rowgatheredsnapshot+orderedcross-planealiasread,andSPmask separate fromC2mask.',
 'Existingmatchedsource/profilefacts justifycorrections:18sourcecontracthashbindings matchcurrentcensus. Only26linecachedupdate_num_computed_tokens_for_batch_changebody readtoresolve3gathers/2writes andpostcomputedwriteacceptedELSEread;no genericframeworkstudy.',
 'Replayed retainedfinitecounterexample:old[10,20],map[1,0],valid[1,1] needs[21,11],unsafeper-rowinplace gives[21,22]. Allsourcesnapshot isconditionalonpreservingcomputed->acceptedcross-planealiaseffects.',
 'Currenthierarchy2203nodes3023edges,124materializedmodeltemplates,18stateenvelopes,42familybindings. Logicalhierarchycountsarenotkernelcounts/performanceproof. Sourcealreadyhad3KVworkpaths;94P5numericvalues unchanged,eligibilityconditions tightened.',
 'Sevenchanged-semantic-effectrulesreviewed;nonewnonequivalentfamily,noCartesianpairs,newnativepromotion0,newperformanceclaim0.',
 'OnlyofflineCPUwork;servercommands,NPUcontexts,inference/restart0. Broadownedcontainerauthorization retained;no redundantpermissionrequest.',
 'Finaloriginalobjective/P0-P7completionauditremains.8of9analysisrequirements nowhavecurrentcontractevidence;overallgoalcompletionnotclaimed.'
]
write(H/'verified-changes.json',{'status':'PROGRESS','changes':changes,'overall_complete':False})

report=(P/'REPORT.md').read_text()
report=report.replace('当前分层公共图包含2163个语义/状态节点、2965条依赖或效应边','当前分层公共图包含2203个语义/状态节点、3023条依赖或效应边')
report=report.replace('引用和端点检查通过，但条件域/状态端口的实质覆盖仍待独立核验，尚不构成整目标完成证明。',
 '本轮已阅读360条语义行和27个条件效应域，并据此修正四处公共状态关系。条件化算子分析已有实质覆盖；整体完成仍需最后按原目标与P0–P7逐项核验。')
for r in audit['requirements']:
    prior=next(a for a in read(P/'current-requirement-audit.json')['requirements'] if a['id']==r['id'])
    report=report.replace('|'+r['id']+'|'+prior['state']+'|','|'+r['id']+'|'+r['state']+'|')
old_next='下一步只有一项验收：检查27个条件效应域与15个状态端口合同是否实质覆盖360条语义行中所有会改变融合结论的公共效应。只对明确漏项定点取证；保持原接口的私有未知、物理fmax或未实施性能不自动变成全局阻塞。未完成该核验前，整体目标继续保持未完成。'
new_next=('本轮实质核验补回四项约束：\n\n'
 '- DSpark公共上下文相同，但3层使用各自WKV、norm/RoPE、cache/group/slot；联合1536投影可共享输入，不能复用一份512维结果。\n'
 '- source20的2048候选块输出是fresh output并另行发布，和512位置TopK/prepare别名关系不同；保留24/28/32/36四个过滤读者、filler/tie/latest-visible语义及代际。\n'
 '- 批状态修正的3次gather必须在computed写入前完成；accepted的else值在computed写入后读取。融合需要保留跨行旧值和潜在跨平面别名顺序。保留反例：逐行原地计算得到[21,22]，正确值为[21,11]。\n'
 '- SP提供mask的True前缀和True padding尾部必须保留；不能凭BOOL shape把它和C2 mask视为同一状态。\n\n'
 '现在只有最后的完整验收：依据原目标及P0–P7检查当前模型/环境/负载、范围台账、G0/候选/规则、理论与实测、最低模块及原因、证据与交付是否全部成立。不是再次逐行读代码或增加配对。条件化私有未知不自动成为全局阻塞；确有范围或证据缺口时仍须继续，不能仅因本报告成稿标完成。')
report=report.replace(old_next,new_next)
report=report.replace(str(P/'current-requirement-audit.json'),str(H/'current-requirement-audit.json'))
report+='\n\nv179语义核验：[semantic-contract-sufficiency-review.json]('+str(H/'semantic-contract-sufficiency-review.json')+')；修正后公共图：[hierarchical-G0-with-boundary-and-state-ports.json]('+str(H/'hierarchical-G0-with-boundary-and-state-ports.json')+')；条件上限限定：[bound-and-eligibility-qualifications.json]('+str(H/'bound-and-eligibility-qualifications.json')+')。原94条P5数值和采集cohort保持不变，没有新增推理或重启。\n'
(H/'REPORT.md').write_text(report)

summary={'status':'PROGRESS','previous_goal_turn':'PROGRESS','overall_complete':False,'new_native_operand_or_phase_promotions':0,
 'new_whole_module_closures':0,'current_requirements':9,'current_answered_analysis_requirements':8,
 'semantic_rows_reviewed':360,'module_reviews':26,'guard_domain_answers_reviewed':27,'state_envelopes':18,
 'public_G0_nodes':2203,'public_G0_edges':3023,'candidate_bindings':42,'changed_effect_rules':7,
 'new_NPU_context_model_or_restart_calls':0,'targeted_cached_source_body_lines':26,'fresh_service_health_check':False}
write(O/(prefix+'-summary-v179.json'),summary)
inputs=list(H.glob('*.json'))+list(H.glob('*.py'))+[H/'REPORT.md']+[Path(r['path']) for r in review['pins']]
write(O/(prefix+'-inputs-v179.json'),{'inputs':[pin(p) for p in sorted(set(inputs))],'overall_complete':False})
write(O/(prefix+'-preflight-v179.json'),{'status':'PASS','preflight':True,
 'checks':{'predecessor1744pinsverified':True,'semantic_refinement_checks_pass':True,'no_final_completion_claim':True}})
gate={'version':179,'parent':pin(Path(old['source_gate'])),
 'policy':'PROFILE_AND_PRIOR_OPERATOR_CONTRACT_FIRST;ONLY_NAMED_CONCLUSION_CHANGING_BODY_OR_PROBE',
 'targeted_source_receipt':str(H/'targeted-batch-source-order-review.json'),
 'new_remote_source_read':False,'source_census_reexecuted':False,'cartesian_pairs':False,
 'user_owned_container_process_probe_authorization_already_present':True,'overall_complete':False}
overlay={'version':179,'changes':changes,'authorities':authorities,
 'semantic_review':str(H/'semantic-contract-sufficiency-review.json'),'current_G0':str(H/'hierarchical-G0-with-boundary-and-state-ports.json'),
 'current_eligibility_and_bound_qualifications':str(H/'bound-and-eligibility-qualifications.json'),
 'new_whole_module_closures':0,'overall_complete':False}
obs=read(Path(old['coverage_obligations']))['obligations']
answers={r['id']:{'remaining':r['remaining'],'current_evidence_authority':str(H/'current-requirement-audit.json'),
 'current_work_queue':str(O/'coverage-work-queue-v179.json'),'legacy_wording_is_historical_after_current_reconciliation':True} for r in obs}
for key in ['G00','G02','G05','G07','G09','G10','G13','G14']:
    answers[key]['remaining']='Semantic360row/27domain/18statecontract review and4publicgraphrefinements complete;final original-objective/P0-P7 audit remains. Specificnative/mode/value/retirement conditions remain typedeligibility premises,not automaticnewanalysisrange.'
updates={
 str(R/'STATE.json'):{'current_requirement_audit':str(O/'whole-scope-requirement-audit-v179.json'),
  'current_public_operator_graph':str(H/'hierarchical-G0-with-boundary-and-state-ports.json'),
  'fusion_scan_status':'SEMANTIC_PUBLIC_SCOPE_REVIEW_DONE_FINAL_ORIGINAL_OBJECTIVE_AUDIT_PENDING'},
 str(R/'fusion-scan.json'):{'current_requirement_audit':str(O/'whole-scope-requirement-audit-v179.json'),
  'current_public_operator_graph':str(H/'hierarchical-G0-with-boundary-and-state-ports.json'),
  'current_semantic_consumer_review':str(H/'semantic-contract-sufficiency-review.json'),
  'current_changed_dependency_rule_round':str(H/'semantic-refinement-rule-round.json'),
  'current_remaining_full_scope_work':queue['work'],'whole_model_exhaustion_proven':False},
 str(R/'records.json'):{'current_hierarchical_public_G0':str(H/'hierarchical-G0-with-boundary-and-state-ports.json'),
  'current_semantic_contract_review':str(H/'semantic-contract-sufficiency-review.json'),
  'current_candidate_state_bindings':str(H/'candidate-to-public-hierarchy-bindings.json')},
 str(R/'module-analysis.json'):{'current_public_dependency_review':str(H/'hierarchical-G0-with-boundary-and-state-ports.json'),
  'current_state_port_contracts':str(H/'host-score-and-state-port-contracts.json'),
  'current_semantic_module_review':str(H/'semantic-contract-sufficiency-review.json'),
  'conditional_fusion_upper_bound_qualifications_v179':limits,'94P5numeric_values_unchanged_v179':True},
 str(R/'bounds-input.json'):{'current_conditional_domain_qualifications':str(H/'bound-and-eligibility-qualifications.json')}
}
jobs=[
 {'kind':'review','status':'pass','description':'PreviousPROGRESS1744pinsverified;read360semanticrows/27guarddomains andrecord26moduleconsumer/statefindings.18reusedcontractsourcehashesmatchcurrentcensus.',
  'artifacts':[H/'predecessor-pin-check.json',H/'semantic-contract-sufficiency-review.json',H/'semantic-review-checks.json']},
 {'kind':'review','status':'pass','description':'Repair3layer-ownedDSparkKV,candidatefreshpublication,batch3gather/2orderedwritealiascontract andseparateSPmask.2203node3023edge publichierarchy,124modelinstances18stateenvelopes.26linecachedtargetedfunctiononly.',
  'artifacts':[H/'hierarchical-G0-with-boundary-and-state-ports.json',H/'host-score-and-state-port-contracts.json',H/'candidate-to-public-hierarchy-bindings.json',H/'targeted-batch-source-order-review.json',H/'semantic-graph-refinement-checks.json']},
 {'kind':'review','status':'pass','description':'Sevenrefinementrules:nonewnonequivalentfamily/nativeeligibility/performanceclaims.NoNPU/model/restart.94P5valuesretainedwithsharperalias/ownershipconditions;finaloriginal-objectiveauditpending.',
  'artifacts':[H/'semantic-refinement-rule-round.json',H/'bound-and-eligibility-qualifications.json',H/'current-requirement-audit.json',H/'REPORT.md']}
]
persist_delta(version=179,previous=178,prefix=prefix,changes=changes,report=report,action=action,
 remaining_delta=queue['work'][0]['remaining'],source_gate=gate,operator_overlay=overlay,queue=queue,jobs=jobs,
 scripts=list(H.glob('*.py')),authority_overrides=authorities,shared_json_updates=updates,shared_text_updates={str(R/'REPORT.md'):report},
 requirement_audit_data=audit,obligation_current_answers=answers,
 pending_full_scope_domains=[queue['work'][0]['id']+':'+queue['work'][0]['remaining']],
 remote_status='OfflineCPUonly;no servercommands,NPUcontext,inferenceorrestart.Freshhealthnotchecked;v177receiptretainedaslastverifiedhealth.')
m=read(O/'checkpoint-manifest-v179.json');assert all(pin(r['path'])==r for r in m['artifacts'])
post={'status':'PASS','checkpoint':179,'manifest_artifacts':len(m['artifacts']),'all_pins_verified':True,
 'semantic_contract_review_published':True,'final_completion_audit_pending':True,'no_new_NPU_model_restart':True,'overall_complete':False}
write(O/(prefix+'-postcheck-result-v179.json'),post);write(N/'CURRENT-POSTCHECK.json',post)
write(N/'LIVE-SHARED-COHORT.json',{'effective_checkpoint':179,'status':'OFFLINE_SEMANTIC_REVIEW_NO_REMOTE_ACTION',
 'last_service_receipt':str(N/'shared-profile-analysis-v177/final-CPU-and-service-receipt.json'),'new_NPU_model_restart':0,'overall_complete':False})
write(N/'NEXT-EVIDENCE-PENDING.json',{'effective_checkpoint':179,'status':'FINAL_ORIGINAL_OBJECTIVE_COMPLETION_AUDIT',
 'next_action':action,'work_queue':str(O/'coverage-work-queue-v179.json'),'overall_complete':False})
print(json.dumps(post))
