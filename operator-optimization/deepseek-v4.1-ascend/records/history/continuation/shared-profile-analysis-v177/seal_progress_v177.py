"""Publish corrected current requirements and the exact source coverage reference set."""
import json
import sys
from pathlib import Path
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H));sys.path.insert(1,str(H.parent))
from finalize_origin_coverage_v9 import N,O,R,read,write
from analyze_evidence_batches_v155 import pin
from persist_operator_review_delta import persist_delta

old=read(O/'REVIEW-CHECKPOINT-v176.json');prefix='exact-scope-and-current-outcomes'
census=read(H/'profile-selected-operator-guard-census.json')
receipt=read(H/'final-CPU-and-service-receipt.json')
checks={**read(H/'coverage-audit-checks.json')['checks'],
    'predecessor1619pins_unchanged':read(H/'predecessor-pin-check.json')['status']=='PASS',
    'profile_source_wrapper100answers':len(read(H/'source-wrapper-operator-scope-reconciliation.json')['wrapper_rows'])==100,
    'current94_comparisons_and26modules':read(H/'current-delivery-checks.json')['status']=='PASS',
    'source_census72files_no_missing_mismatch':len(census['files'])==72 and not census['missing_or_mismatched'],
    'all_selected_profile_symbols_present':not any(f['missing_profile_symbols'] for f in census['files']),
    'CPU_parser_terminal':not receipt['CPU_parser_same_handle_live'],
    'service_HTTP200_script_unmodified':receipt['health_http']==200 and receipt['serve_sha256']=='c297201185b5676254af4d7693279f89bfa60f9c76eb340058a78ecf730b7bba',
    'new_NPU_or_restart0':receipt['new_NPU_context_model_calls_restart_calls_this_turn']==0}
assert all(v for v in checks.values()),checks
authorities={}
for key,stem in [('candidate_ledger','candidate-profile-ledger'),('module_ledger','module-profile-ledger'),('branch_matrix','module-phase-branch-matrix')]:
    p=Path(old[key]);x=read(p);x.update(parent_review_authority=pin(p),current_review_checkpoint=177)
    x['current_exact_scope_audit']={
        'current_requirements':str(H/'current-requirement-audit.json'),
        'exact_source_guard_reference':str(H/'profile-selected-operator-guard-census.json'),
        'source_wrapper_scope_answers':str(H/'source-wrapper-operator-scope-reconciliation.json'),
        'duplicate_definition_operator_review':str(H/'duplicate-definition-operator-review.json'),
        'exact_guard_to_existing_semantic_contract_bijection_still_required':True,
        'no_new_candidate_family_or_runtime_applicability_promotion':True}
    q=O/(stem+'-v177.json');write(q,x);authorities[key]=str(q)

changes=[
 'PreviousgoalturnPROGRESS independentlyrevalidated:all1619checkpoint176artifactpins match;no oldproof/body reexecution.',
 'Replacedstale blanketrequirementstateswithcurrent9requirementaudit.P4groupedmeasured minima,P5explicit94conditional/symboliccomparisons andP6supportedlocation/competingcauses are answeredfor their protocolscope;globalphysicalMFUnotrequired. Wholefusiongoalremainsunachieved.',
 '821146intactEP0nativeETnodes resolvedinto10434exactparent/publicoperator/scopegroups with136schemas,covering235public/runtimeopnames.Native nesting isnotadditionalarithmeticor deviceexecutionorder;unemittedtoproot retained.',
 'All100generic supportframes formerlyunassigned byexclusivePMUowner classifier now haveexplicitoperator-scopeanswers;505retainedframeindexincludes492actualusedand13extras,not505newcurrentexecutions.PMUownership/rankingsnotchanged.',
 'Profile/loaded-class-selectedCPUASTcensus fixes72sourcefiles,573selectedfunctions,1867guardreferencesoutsidealreadyreviewedworker/acl_graphsources.All67retainedprofilefilehashesmatch;5loadedclassfilesnewlybound;0missingfiles/missingprofilesymbols/hashmismatches.',
 'Censuskeepssource-position-qualifiedduplicates,getter/setters,nestedlexicalhelper/callablereferences andfullfunctionlexicalcontinuations.Samequalname cannot silentlyoverwritedifferentbranches. Fourpreviouslycollidingdefinitions/fiveguards haveexplicitF12/F20/F21/F31answers,includinggraph-awareemptyQLI,versionedexpertmap,zero-IDaliasandEPLBrecording.',
 'Exactguardsetisnot1867newunanalysedoptimizationtasks.Semanticcoveragebijectionagainstexisting360families/1101staticanswersandcompletehierarchicalG0interface/alias/stateportreconciliationarestillactualscopegaps;no completionfromcounts.',
 'Currentmodule-analysisandREPORT preparedwith26functionalmodules,94currentcomparisons,42registeredfamilies andM12/M23measuredgroupresults;old41layer/76comparisonanalysispreservedasimmutablesnapshot.',
 'OnlyCPUparsing/read-onlySSHidentityandhealththisturn.NoNPUcontext/modelcalls/restart.SSHControlPersist3600expiredandwasreopenedusingpreviouslyauthorizedcredentialswithoutpersistingpassword;health200/originalserveSHAconfirmed,CPUhandles terminal.'
]
write(O/(prefix+'-preflight-v177.json'),{'status':'PASS','preflight':True,'checks':checks})
write(O/(prefix+'-inputs-v177.json'),{'inputs':[pin(p) for p in sorted(H.rglob('*')) if p.is_file() and '__pycache__' not in p.parts],'overall_complete':False})
write(O/(prefix+'-summary-v177.json'),{'status':'PROGRESS','previous_goal_turn':'PROGRESS','overall_complete':False,
    'new_native_operand_or_phase_promotions':0,'new_whole_module_closures':0,
    'current_requirements':9,'current_answered_analysis_requirements':5,
    'source_files':72,'matching_retained_profile_source_files':67,'new_loaded_class_source_files':5,
    'selected_functions':573,'exact_guard_references':1867,'native_parent_scope_groups':10434,
    'generic_support_frames_answered':100,'current_module_metric_objects':26,'current_performance_comparisons':94,
    'new_NPU_context_model_or_restart_calls':0,'service_HTTP':200})

action=(
 '继续完整目标，checkpoint177。先读REPORT、current-requirement-audit、exact-source-guard-coverage-frontier。当前9项验收已按证据改正：P4是12MAC模块M12最低(合并2.41549%，两窗2.55250/1.17269)，20Vector模块M23算术活动最低，不是物理MFU；P5有94规范条件/符号比较，P6有F21内部位置与竞争原因。不要再沿用“最低未建立/必须全局物理MFU/查fmax”的旧门槛。'
 '235op父调用已从821146完整ET节点重建成10434组，136schema；100通用wrapper都有算子范围答案，不改PMU排他归属。505旧索引仅492actualused、13extras。'
 '新的精确覆盖参照是72文件/573函数/1867guard，67profileSHA全部匹配+5actualclass源文件新绑定、0missing/hash差异。worker/acl_graph原1101点不重读。条件引用不是1867项新任务：优先按算子效应/已有合同归并，做exactguard→已有360语义族/合同或明确保持原控制流的覆盖表；只展开真正不匹配、会改变算子值/精度/行/归属/状态/就绪的条件。不要广读框架，不做笛卡尔配对。'
 'Census按位置区分同名条件定义、property getter/setter/overload，保留nested helper及被partial/executor传递的callable，保存整函数lexicalcalls/writes/returns以避免early-return后续效应遗漏。JSON是词法过近似，不是路径纯度或全部动态callee证明。旧按qualname单key的中间census已留档；最新文件才是权威。'
 '四个同名定义/五个guard已有明确答案：QLI只有actualtokens0且非graph才zero-return；expert_map getter按runner版本取不同对象；EPLB empty返回原topk_IDs并保留alias，非空map/load-record/validrows另有状态；wrapper/fallback不是独立新算术阶段。'
 '随后将G1/G5/G6/G7/G8/G9及26模块/42候选/现有host与metadata合同拼成可审计的hierarchical G0端口账本，显式全部消费者/alias/inplace/state/communication/iteration边与条件；目标/草稿16MiB索引bank同rawStorageImpl，C2mask为8KiBbacking不同长度view，不能从空白推独立。'
 '只有这两处实际范围证据闭合后才能按真实目标验收；若出现新operator问题继续补，不能缩到已有模板。已分析但运行值/物理capacity/历史privateABI或未实现gain未知不作为通用未分析项。currentmodule-analysis与报告已同步到R根目录，旧76比较/41层记录在本轮snapshot不混用。'
 '本轮无新NPU/context/inference/restart，服务health200原serveSHA一致，CPU全部终止。没有默认新NPU采集。下轮直接用现有精确census与合同推进，不要再次收集同一profile或重复问授权。')
queue={'version':177,'parent':pin(O/'coverage-work-queue-v176.json'),'scope_preserved_by':old['coverage_obligations'],
    'work':[
        {'id':'Q177-EXACT-GUARD-CONTRACT-COVERAGE','state':'FINITE_REFERENCE_TO_SEMANTIC_ANSWERS_RECONCILIATION',
         'action':'Use72file/573function/1867guardreference and existing360semanticfamily contracts;group by operator effects,retain all source cases and callable alternatives,read only a genuinelyunmatched effect. Preserve original1101reviewedguards and fullfunctioncontinuations.','evidence':[str(H/'profile-selected-operator-guard-census.json'),str(H/'exact-source-guard-coverage-frontier.json'),str(H/'duplicate-definition-operator-review.json')],
         'remaining':'Exactguard-to-existing-answer coverage not yetproved;counts alone are not completion.'},
        {'id':'Q177-HIERARCHICAL-G0-PORT-CLOSURE','state':'EXPLICIT_PUBLIC_INTERFACE_ALIAS_STATE_AND_CASE_RECONCILIATION',
         'action':'Reconcile all26modules/49loadedclasses/235profileopnames with sourcegraphsG1/G5/G6/G7/G8/G9 and host/metadata boundarycontracts. FreezehierarchicalG0,withallconsumers,alias/inplace,communication,iteration/stateports andguardedalternatives;retainopaqueprivateAPIasconditionalboundaries,neverinventindependence.','evidence':[str(H/'operator-parent-scope-census.json'),str(H/'loaded-module-instance-census.json'),str(H/'source-wrapper-operator-scope-reconciliation.json'),str(N/'shared-profile-analysis-v176/operator-conclusions-from-profile.json')],
         'remaining':'Completepublicgraph/effectivecallee/port/aliasuniverse notproven bypartialET orsource-templatecounts.'}],
    'completed_previous_audit_outputs':str(H/'current-requirement-audit.json'),'no_default_new_NPU_acquisition':True,'overall_complete':False}
gate={'version':177,'parent':pin(Path(old['source_gate'])),
    'policy':'CURRENT_PROFILE_CALLER_AND_EXACT_GUARD_REFERENCE_FIRST;ONLY_UNMATCHED_OPERATOR_EFFECT_CAN_OPEN_TARGET_BODY',
    'exact_reference':str(H/'profile-selected-operator-guard-census.json'),'source_wrapper_answers':str(H/'source-wrapper-operator-scope-reconciliation.json'),
    'already_reviewed1101_guards_not_reopened':True,'no_cartesian_pairs_or_generic_framework_research':True,
    'no_default_device_probe':True,'overall_complete':False}
overlay={'version':177,'changes':changes,'authorities':authorities,
    'current_audit':str(H/'current-requirement-audit.json'),'exact_guard_reference':str(H/'profile-selected-operator-guard-census.json'),
    'current_module_analysis':str(H/'module-analysis.json'),'current_report':str(H/'REPORT.md'),
    'new_whole_module_closures':0,'overall_complete':False}

remaining_by_id={
 'G00':'Exactsourceguard-to-contractcoverage andhierarchicalpublicG0scope/effectivecalleeclosure remain. Original26modules/360rows/42families allretained.',
 'G01':'Current67profilehashesmatch and5loadedclasssourcesnewlybound. Actualnativeinternalmode/valueidentity remains a conditionalpremise for affectedclaims;no universalbinary/opcodegate for allpublicoperator analysis.',
 'G02':'Frozenrequest/source-domain and1104observedgreedysampler calls bound. Exactapplicableguard/case coverage stillrequiresQ177-GUARD;unobservedvalues are not all newanalysis tasks.',
 'G03':'Originalhistorical9privateoperandregimes remainunsupported for privatehelper/task deletion. PublicPad replacement has separate semanticcontract;no pendingPadprobeapproval or newblanketnativepacket task.',
 'G04':'Receiver-work/source/owner reviews retained inmatchingcohorts. Newnativeworkspace/write/retirementclaims remainconditional;do not joinv168receiverwork tov164PMUtime or count allocated128rows asMACrows.',
 'G05':'HierarchicalG0mustretain sharedindex/C2banks plusallsource reader/writer/rollback/publicationports. Exactaddressesalone do not closegenerationorretirement.',
 'G06':'Numericalsourcecontracts preserveoriginalrounding/quantizer/RNG/TopK andallowdeclaredconditionalnewdomains. Newnativechanged-kernelparity remainsunproved;implementation is notanalysisacceptance.',
 'G07':'All42families havejointwork/resource/lifetimecontracts;completegraphport/guardcoverage stillrequiresQ177G0/GUARD. Old212pair/383pendingcounts are provenance,notworkqueue.',
 'G08':'94P5records retainnecessarywork/capacity domains andsymbolicphysicalrateuppers. Physicalfmax/Vector/HBM/networkuppersunknown where documented;no repeatcapacitysearch or unconditionalMFU/speedup claim.',
 'G09':'27sourcegraph rule round answered.72file/573function/1867guard reference and235opparentcensus identify exactscopebasis;completehierarchicalG0/guardbijection stillopen.',
 'G10':'All35model+7hostfamilies composed under12globalconstraints withoutCartesianpairs. Whole-scopecertificate stilldependsG0/guardclosure,notmorepairenumeration.',
 'G11':'Time/counter/cohortuncertaintyandunmeasuredrequestcriticalpatheffects explicitlyretained. No causalrequestgain asserted;P6containsupportedlocation andcompetinghypotheses ratherthan fabricated uniquecause.',
 'G12':'Currentcomparablegroupsestablished:M12lowest12MACmodules inbothwindows/pooled;M23lowest20Vectorarithmeticgroups. Missing/unrecorded/ambiguoushostphysicalMFUremainunknown,notzero. GlobalphysicalMFUisnotadditionalP4gate.',
 'G13':'Alloriginalsemanticrows/1101staticanswers and42compositionsretained. Nextonlyexactsourceguardcoverage+G0portreconciliation;knownconditionalunknownsarenotunreadcode.',
 'G14':'Copy/control/publicationinterfaces mustbeincluded inG0ports/alias/lifetimeledger. OriginalhistoricalDMAprivatefieldsremainunknown forprivate edits;logicalcopyextentsare notphysicalDMAtraffic ordefaultnewprobe.'}
base=read(Path(old['coverage_obligations']))['obligations']
answers={r['id']:{'remaining':remaining_by_id[r['id']],
    'current_evidence_authority':str(H/'current-requirement-audit.json'),
    'current_work_queue':str(O/'coverage-work-queue-v177.json'),
    'legacy_wording_is_historical_after_current_reconciliation':True} for r in base}
recs={r['id']:{'current_requirement_audit':str(H/'current-requirement-audit.json'),
    'current_dispatch_policy':'ExactguardcontractandhierarchicalG0closureonly;no universalMFU/privateABI/attainablegain gate.',
    'whole_domain_closed_by_this_increment':False,'complete_original_scope_preserved':True} for r in base}

updates={
 str(R/'module-analysis.json'):read(H/'module-analysis.json'),
 str(R/'STATE.json'):{'module_diagnosis_status':'COMPLETE_FOR_DECLARED_RECORDED_COMPARABLE_GROUPS_WITH_UNKNOWNS',
    'lowest_module_ids':['M12'],'lowest_grouped_modules':{'AIC_MAC':'M12','AIV_Vector':'M23'},
    'performance_comparison_status':'94_CURRENT_CONDITIONAL_SYMBOLIC_MATCHED_RECORDS',
    'theoretical_bound_status':'CONDITIONAL_OR_SYMBOLIC_WITH_EXPLICIT_MISSING_PHYSICAL_INPUTS',
    'fusion_scan_status':'DECLARED_SOURCE_RULE_ROUND_DONE_FULL_SCOPE_GUARD_G0_CLOSURE_PENDING',
    'current_requirement_audit':str(O/('whole-scope-requirement-audit-v177.json'))},
 str(R/'fusion-scan.json'):{'current_requirement_audit':str(O/'whole-scope-requirement-audit-v177.json'),
    'current_exact_guard_reference':str(H/'profile-selected-operator-guard-census.json'),
    'current_remaining_full_scope_work':queue['work'],'whole_model_exhaustion_proven':False},
 str(R/'records.json'):{'current_requirement_audit':str(O/'whole-scope-requirement-audit-v177.json'),
    'current_public_call_hierarchy':str(H/'operator-parent-scope-census.json'),
    'current_support_frame_scope_answers':str(H/'source-wrapper-operator-scope-reconciliation.json')},
 str(R/'bounds-input.json'):{'current_functional_module_and_candidate_P5_contracts':str(H/'module-analysis.json'),
    'legacy_comparisons_preserved_separately':str(H/'module-analysis-before-v177.json')}}
jobs=[
 {'kind':'review','status':'pass','description':'PreviousPROGRESS1619pinsverified;current9requirementaudit replacesstaleP4/P5/P6status,genuineG0/guardclosuregapsretained.','artifacts':[H/'predecessor-pin-check.json',H/'current-requirement-audit.json',H/'coverage-audit-checks.json']},
 {'kind':'review','status':'pass','description':'CPU821146nodeparentcensus235opnames/10434groups/136schemas;100supportframeshaveoperator-scopeanswerswithoutchangingPMUownership.','artifacts':[H/'operator-parent-scope-census.json',H/'source-wrapper-operator-scope-reconciliation.json',H/'loaded-module-instance-census.json']},
 {'kind':'review','status':'pass','description':'Profile/class-selected72files573functions1867guardrefs,67profilehashesmatch5newclassfilesbound,no missingfile/symbol/mismatch.Duplicate/nested/callabledefinitionskept,sourcecensusnotsemanticcompletion.','artifacts':[H/'profile-selected-operator-guard-census.json',H/'exact-source-guard-coverage-frontier.json',H/'duplicate-definition-operator-review.json',H/'guard-census-execution.json']},
 {'kind':'review','status':'error','description':'PreserveCPUcensusconstructionerrors(hostcontainerpathnamespace,comprehensionposition)andintermediatequalname/callableclosureindices;correctedlatestreferencesonly.No modelreplay/probeused.','artifacts':[H/'guard-census-execution-host-path-error.json',H/'guard-census-comprehension-error.json',H/'profile-selected-operator-guard-census-before-definition-disambiguation.json',H/'profile-selected-operator-guard-census-before-callable-reference-closure.json']},
 {'kind':'review','status':'pass','description':'Currentreport26functionalmodulemetrics/94P5records/42familiesprepared;historical41layer/76comparisonsnapshotretained.Servicehealth200/originalscript,CPUterminal,newNPU/restart0.','artifacts':[H/'REPORT.md',H/'module-analysis.json',H/'module-analysis-before-v177.json',H/'current-delivery-checks.json',H/'final-CPU-and-service-receipt.json']},
]
persist_delta(version=177,previous=176,prefix=prefix,changes=changes,report=(H/'REPORT.md').read_text(),action=action,
    remaining_delta='Exact sourceguard-to-existingcontract coverage and completehierarchicalpublicG0port/alias/state/case reconciliation remain.',
    source_gate=gate,operator_overlay=overlay,queue=queue,jobs=jobs,scripts=list(H.glob('*.py')),
    obligation_reconciliations=recs,authority_overrides=authorities,shared_json_updates=updates,
    shared_text_updates={str(R/'REPORT.md'):(H/'REPORT.md').read_text()},
    requirement_audit_data=read(H/'current-requirement-audit.json'),obligation_current_answers=answers,
    pending_full_scope_domains=[r['id']+':'+r['remaining'] for r in queue['work']],
    remote_status='OnlyCPUprofile/sourcecensus/read-onlySSHthisturn;allparserprocessesterminal,health200andoriginalserveSHA.NewNPU/model/restart0.No defaultnewdeviceacquisition.')
m=read(O/'checkpoint-manifest-v177.json');assert all(pin(r['path'])==r for r in m['artifacts'])
published=read(O/'whole-scope-requirement-audit-v177.json')
assert next(r for r in published['requirements'] if r['id']=='LOWEST_MODULE_UTILIZATION')['state'].startswith('PROTOCOL_P4_')
assert len(read(R/'module-analysis.json')['performance_comparisons'])==94
post={'status':'PASS','checkpoint':177,'manifest_artifacts':len(m['artifacts']),'all_pins_verified':True,
    'current_requirement_audit_replaces_stale_states':True,'current_root_report_and_module_analysis_published':True,
    'original_service_HTTP':200,'no_new_NPU_model_restart':True,'overall_complete':False}
write(O/(prefix+'-postcheck-result-v177.json'),post);write(N/'CURRENT-POSTCHECK.json',post)
write(N/'LIVE-SHARED-COHORT.json',{'effective_checkpoint':177,'status':'CPU_CENSUSES_TERMINAL_SERVICE_UNCHANGED',
    'receipt':str(H/'final-CPU-and-service-receipt.json'),'all_exact_handles_terminal':True,
    'original_service_HTTP':200,'new_NPU_model_restart':0,'no_default_new_NPU_acquisition':True,'overall_complete':False})
write(N/'NEXT-EVIDENCE-PENDING.json',{'effective_checkpoint':177,'status':'EXACT_GUARD_CONTRACT_AND_HIERARCHICAL_G0_CLOSURE',
    'next_action':action,'work_queue':str(O/'coverage-work-queue-v177.json'),'overall_complete':False})
print(json.dumps(post))
