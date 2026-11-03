"""Current-state self review; graph/hash checks supplement,not replace semantic arguments."""
import collections
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
H=Path(__file__).resolve().parent;N=H.parent;O=N/'profile-first-20261031';P=N/'shared-profile-analysis-v179'
R=Path('/Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41');K=R.parent.parent;M=Path('/Users/wio/work/171-vllm-migration-20261102')
def read(p):return json.loads(p.read_text())
def pin(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def save(n,x):(H/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def F(x):return Fraction(x['exact_fraction'])
g=read(H/'G0-current-v180.json');rec=read(H/'records.json');a=read(H/'module-analysis.json');s=read(H/'profile-summary.json')
scan=read(H/'fusion-scan.json');contract=read(H/'contract.json');p5=a['performance_comparisons']
semantic=read(P/'semantic-contract-sufficiency-review.json');branch=read(O/'module-phase-branch-matrix-v179.json')
source=read(N/'shared-profile-analysis-v177/profile-selected-operator-guard-census.json')
guard=read(P/'exact-guard-operator-contract-review.json')
state=read(P/'host-score-and-state-port-contracts.json')
baseline=read(O/'new171-full-case-baseline-reference-v164.json');completion=read(O/'shared-builtin-window-completion-v164b.json')
fixed=read(N/'shared-profile-analysis-v172/fixed-case-branch-contract-reuse.json')
causes=read(N/'shared-profile-analysis-v172/lowest-module-cause-review.json')

checks={}
def check(name,value):
 checks[name]=bool(value)
 if not value:raise AssertionError(name)
check('migration_full_file_SHA_pass',read(M/'postflight-target.json')['final']['full_file_sha256_comparison']=='PASS')
check('CANN92_primary_snapshot',any(x.get('rc')==0 and 'version=9.2.0' in x.get('text','') for x in read(M/'source-CANN-version-files.json')))
check('current_fixed_text_request_hashes',fixed['input_hashes_match_registered_case'] and fixed['client_sha_matches_execution_pin'])
check('valid_unprofiled_reference_8x32K1K',baseline['case_valid'] and baseline['responses']==8 and baseline['all_prompt32768_output1024'])
check('complete_profile_case_all16footers',completion['baseline_valid'] and completion['profile_valid'] and completion['all16_NPU_worker_footers_present'])
check('successful_profile_text_match',completion['exact_response_text_matched_count']==8)
check('profile_source_frontier_matches',len(source['files'])==72 and not source['missing_or_mismatched'] and not any(f['missing_profile_symbols'] for f in source['files']))
check('guard_refs_answered_with_domain_review',{x['id'] for x in source['guards']}=={x['guard_id'] for x in guard['guard_coverage']} and all('semantic_review_v179' in d for d in guard['domains']))
check('360semantic_rows_reviewed',len(semantic['semantic_row_reviews'])==360 and {r['semantic_row'] for r in semantic['semantic_row_reviews']}=={r['id'] for m in branch['modules'] for r in m['branches']})
check('all26module_review_findings_present',len(semantic['module_reviews'])==26 and all(m['reviewed_answer_and_consumer_effects'] and not m['unanalysed_public_effects_in_these_rows_after_corrections'] for m in semantic['module_reviews']))
check('case_roles_not_inferred_from_counts',read(N/'shared-profile-analysis-v178/layer-role-and-selector-bindings.json')['role_counts']=={
 'ATTN_SWA':2,'ATTN_C2_SOURCE':3,'ATTN_C2_CONSUMER':15,'ATTN_C1_SOURCE':1,'ATTN_C1_INDEX_SOURCE':4,'ATTN_C1_CONSUMER':15})
ids={n['id'] for n in g['nodes']}
check('graph_endpoint_integrity',len(ids)==len(g['nodes']) and all(e[k] in ids for e in g['edges'] for k in ['producer','consumer']))
check('3layer_owned_KV_paths',all(f'draft.shared_kv.layer{i}/persistent_swa_state' in ids for i in [40,41,42]) and not any(i.startswith('draft.shared_kv/') for i in ids))
check('candidate_bank_distinct_and4readers','state.candidate_block_bank' in ids and sum(e['producer']=='state.candidate_block_bank' and e['dependency_kind']=='CANDIDATE_EPOCH_RAW' for e in g['edges'])==4)
check('batch3gather_beforewrite_and_ordered_aliasread',sum(e['dependency_kind']=='SNAPSHOT_READSET' for e in g['edges'])==3 and any(e['dependency_kind']=='ORDERED_POTENTIAL_ALIAS_READ_AFTER_WRITE' for e in g['edges']))
check('SPmask_and_C2mask_separate',{'SP_VALIDITY_MASK','C2_BOOL_MASK_BANK'}<={e['id'] for e in state['state_envelopes']})
check('TopK8valueepochs_and_next_iterationWAR',len(g['shared_topk_epochs'])==8 and sum(len(e['reader_layers']) for e in g['shared_topk_epochs'])==38 and any(e['dependency_kind']=='BANK_RANGE_WAR' and e.get('iteration_offset')==1 for e in g['edges']))

definition_ids={'definition:'+t['id']+'/'+n['id'] for t in g['template_definitions'] for n in t['nodes']}
allc={r['id'] for r in rec['fusion_candidates']}
check('all35model_plus7host_records',allc=={f'F{i:02}' for i in range(1,36)}|{f'H{i:02}' for i in range(1,8)})
for c in rec['fusion_candidates']:
 cp=H/'candidates/whole-scope-v180'/Path(c['candidate_graph_path']).name;cg=read(cp)
 check(c['id']+'_graph_current_G0_and_mapping',cg['original_graph_sha256']==pin(H/'G0-current-v180.json')['sha256'] and c['candidate_graph_sha256']==pin(cp)['sha256'] and set(c['original_node_ids'])<=definition_ids)
 check(c['id']+'_conditional_consumers_and_no_measurement',c['status']=='CONDITIONAL' and c['other_consumers'] and c['semantic_conditions'] and c['measured_candidate_T'] is None and not c['performance_gain_measured'])
check('all7declared_rules_have_refinement_answers',len(read(P/'semantic-refinement-rule-round.json')['rules'])==7 and read(P/'semantic-refinement-rule-round.json')['new_candidate_family_count']==0)
check('compositions_mutual_exclusion_and_resource_constraints_retained',len(read(N/'shared-profile-analysis-v174/all42-composition-coverage.json')['global_higher_order_constraints'])==12)
check('full_rule_round_prior_template_scope',len(read(N/'shared-profile-analysis-v176/declared-source-template-rule-review.json')['capsules'])==27)

check('all26modules_withmetrics_or_unknown',len(a['modules'])==26 and all(m['primary_status'] and m['P5_status'] for m in a['modules']))
check('fulloperator_population_not_topN',sum(r['count'] for r in read(N/'shared-profile-analysis-v165/ALL32-RANKING.json')['operator_ranking'])==288846)
check('raw3690file32profilemanifest',read(N/'shared-profile-analysis-v165/INPUT-MANIFEST.json')['profiles']==32 and len(read(N/'shared-profile-analysis-v165/INPUT-MANIFEST.json')['files'])==3690)
rank={x['metric_name']:x for x in a['rankings']}
check('MAC_minimum_M12_in12comparable',len(rank['AIC_MAC']['ordered_modules'])==12 and rank['AIC_MAC']['ordered_modules'][0]['module']=='M12')
check('Vector_minimum_M23_in20comparable',len(rank['AIV_Vector']['ordered_modules'])==20 and rank['AIV_Vector']['ordered_modules'][0]['module']=='M23')
check('no_false_global_minimum',causes['no_trustworthy_global_minimum_across_host_Cube_Vector_communication'])
check('M12min_robust_in_both_windows',len(causes['measured_minima'])==2 and all(x['measured_lowest_Cube_module']=='M12' and x['recorded_ambiguity_robust'] for x in causes['measured_minima']))
check('cause_hypotheses_not_asserted_causally',any(c['state']=='HYPOTHESIS' for c in causes['claims']) and any(c['state']=='NOT_SUPPORTED_BY_THIS_METRIC' for c in causes['claims']))

required=['effective_work_Q','measured_time_T','time_lower_L','measured_throughput','throughput_upper_U','measured_over_ceiling_ratio','time_gap_ratio','utilization_metric_reference','dominant_constraint_ids','bound_status','source_ids']
check('all94P5records_complete_or_explicitunknown',len(p5)==94 and all(all(k in r for k in required) for r in p5))
check('P5all360semanticrows_linked',{x for r in p5 for x in r.get('branch_ids',[])}=={r['id'] for m in branch['modules'] for r in m['branches']})
check('all42candidateP5Tunknown',len([r for r in p5 if r.get('candidate_only')])==42 and all(r['measured_time_T'] is None and r['measured_throughput'] is None for r in p5 if r.get('candidate_only')))
recalcs=[]
for r in p5:
 if r['object_kind']!='MATCHED_PARTIAL_INVOCATION_POPULATION':continue
 rate={'BF16':4096,'INT8':8192,'FLOAT_HF32_OFF':1024,'FLOAT_HF32_ON':2048}[r['implementation_domain']['mode']]
 q=Fraction(r['effective_work_Q']);t=F(r['measured_time_T']);cycle=q/(24*rate);refL=cycle/1800000000;ref=r['reference1800MHz_only']
 check(r['object_id']+'_exact_arithmetic',F(r['measured_throughput'])==q/t and F(r['time_lower_L']['cycle_equivalent'])==cycle and F(ref['time_floor_seconds'])==refL and F(ref['L_over_T'])==refL/t and F(ref['T_over_L'])==t/refL)
 check(r['object_id']+'_rated_not_physical',ref['rated_frequency_is_not_physical_maximum'] and r['time_lower_L']['numeric_value'] is None and 'f_upper' in r['time_lower_L']['expression'])
 recalcs.append({'object_id':r['object_id'],'Q_MAC':str(q),'T_second':str(t),'MAC_per_second':str(q/t),'cycles_at_24cores':str(cycle),
   'rated1800_reference_L_seconds':str(refL),'rated_reference_L_over_T':str(refL/t),'rated_reference_T_over_L':str(t/refL),
   'physical_fmax_bound_proved':False,'status':'EXACT_RATIONAL_RECALC_MATCH;CONDITIONAL_DENSE_MODE_ONLY'})
check('26matched_partial_recalculations',len(recalcs)==26)
save('exact-rational-P5-recalculation.json',{'review':'SELF_REVIEW','records':recalcs,'no_new_physical_capacity_claim':True})
check('bounded_profile_summary',(H/'profile-summary.json').stat().st_size<=32768)
check('current_contract_host_and_authorization',contract['hardware']['host']=='172.16.10.171' and contract['device_measurement_authorized'] and not contract['implementation_changes_authorized'])
check('no_optimization_implementation',rec['implemented_optimization'] is False and rec['performance_gain_measured'] is False)

arguments=[
 ('MODEL_VERSION_HARDWARE_LOAD','Migration fullSHA/model/image snapshot,successful171generation and CANN9.2 installrecord bind environment. Frozen8textrequests+2prefixseeds/source/client hashes,unprofiled8x32768/1024reference and complete16workerformalprofile bind theload;driver/firmware differences useraccepted. Singlebaseline is not stabilityproof.',
  [str(M/'postflight-target.json'),str(M/'RUNTIME-READY-171-20261102.json'),str(M/'source-CANN-version-files.json'),str(N/'shared-profile-analysis-v172/fixed-case-branch-contract-reuse.json'),str(O/'shared-builtin-window-completion-v164b.json')]),
 ('ALL_MODULE_CONDITIONAL_BRANCHES','26manualmodulefindings reviewed all360typedsemanticanswers and27guarddomains. Retained1101staticreviews and1867exactguardrefs include continuations/returnaliases/conditionalhelpers. Inactivevision follows frozenplain-textinputcontract,notmissingtrace. New4publiceffectrepairs close concrete coarsening errors;native execution/applicability remains conditional without introducing unreviewedpublicwork.',
  [str(P/'semantic-contract-sufficiency-review.json'),str(P/'exact-guard-operator-contract-review.json'),str(P/'targeted-batch-source-order-review.json')]),
 ('CURRENT_FINITE_GRAPH_LEDGER','Current fixedsource hierarchicalG0 retains all40targetroles,3draftlayers,macro/refinementdistinction and documentedtyped alternatives. Per-layerKV/candidate/TopK/SP/C2/batch planes and originalorderedalias effects are explicit. Opaque nativecalls retain every original publiceffect,so no hiddenreader is declaredabsent;strongerprivateedits remain conditional/unsupported. Candidate42graphs map to currentG0definitions and preserve consumercontracts.',
  [str(H/'G0-current-v180.json'),str(H/'candidate-graph-index.json'),str(P/'host-score-and-state-port-contracts.json')]),
 ('CROSS_OPERATOR_LAYER_CARD_ITERATION','Complete27source-template rulecapsules and7changed-effectrules cover pair/triple/fork/multi/split/sharedinput/crossmodule/multiconsumer cases.42families retain12globalresource/choice constraints. Current8TopKepochs/38readers,candidate20->24/28/32/36,3KVowners,batchsnapshot/orderedalias and perrange nextiteration retirement refine alreadyregisteredvariants;last semantic round no newnonequivalentfamily. This is declared finite-rule convergence,not allpossiblealgorithm exhaustion.',
  [str(N/'shared-profile-analysis-v176/declared-source-template-rule-review.json'),str(P/'semantic-refinement-rule-round.json'),str(N/'shared-profile-analysis-v174/all42-composition-coverage.json'),str(H/'current-fusion-coverage.json')]),
 ('NATIVE_OPERATOR_REGIMES','Originalsource/typednative regimes and observedprofile attribution retained. ExistingF21fusion not recounted. UnsupportedprivateMemSet/taskpacket edits remain unsupported;wholepublicPad/newABI alternatives use their own contract.3KVparameter namespaces andfreshcandidatecopycannot be elided by same shape. No newnativekernel/equivalence/feasibility claim.',
  [str(P/'bound-and-eligibility-qualifications.json'),str(N/'shared-profile-analysis-v176/operator-conclusions-from-profile.json'),str(H/'records.json')]),
 ('THEORY_AND_MEASURED_COMPARISON','94records cover26wholefunctionalmodules,42candidatefamilies and26matchedpartialdensemodepopulations;all360semanticcases linked. RequiredT/L/Qthroughput,ratio,metric,dominantconstraint/evidence/status fields are present or explicitUNKNOWN. Exactrational recomputation validates26matchedQ/T/cycle/ratedreference ratios. MACalgorithm/mode/core/pool/time boundaries are stated;physicalfmax unknown stayssymbolic. Missing necessarywork/capacity orcandidateT doesnot become a finite numericceiling or speedup promise.',
  [str(H/'module-analysis.json'),str(H/'bounds-input.json'),str(H/'exact-rational-P5-recalculation.json'),str(P/'bound-and-eligibility-qualifications.json')]),
 ('LOWEST_MODULE_UTILIZATION','All32formalprofiles/84nativeoperatorgroups/288846executions aggregated before display. Exclusiveownership201351 andambiguous87495 separate;missing3510PMU notzero. Same reportedpositivecycle-weightedgroups rankM12lowest12MACmodules andM23lowest20Vectorarithmeticmodules,including bothwindowM12robustness. No globalhost/Cube/Vector/comm minimum or MFU claim.',
  [str(N/'shared-profile-analysis-v172/module-activity-review.json'),str(H/'profile-summary.json'),str(H/'module_metrics.csv'),str(H/'operator_metrics.csv')]),
 ('CAUSES_AND_CRITICAL_PATH','MeasuredM12minimum located inalreadyfusedF21,owning98.76/99.19%M12cumulativenativetime and99.72/99.93%reportedAICcycles. Smallreceivertiles/rankimbalance andpeerarrival/polling retained as competinghypotheses withsource/cohort limits;HBMsaturation-alone notsupported. Kernelcumsumsnotrequestcriticalpath;no reachablegain/uniquecause fabricated. ProtocolP6 explicitly permits unresolvedhypotheses.',
  [str(N/'shared-profile-analysis-v172/lowest-module-cause-review.json'),str(N/'shared-profile-analysis-v172/module-window-diagnosis.json')]),
 ('CRITICAL_EVIDENCE_AND_FINAL_CLOSURE','Stagedcanonicalcontract/G0/42candidategraphs/scan/94comparisons/records/profileindex+boundedsummary/fullaggregates/bounds/rawrationaloutput/profilingplan are coherent. Old169host/31record/falseauthorization/root-summary state preserved as history and replaced. Completion still requires publishing these exact bytes and verifying root entrypoints plus Task/Loop state;this audit doesnotdeclare currentroot complete before publication.',
  [str(H/'prepared-current-delivery-checks.json'),str(H/'records.json'),str(H/'contract.json')])]
save('prepublication-requirement-audit.json',{
 'status':'EVIDENCE_AND_STAGED_DELIVERY_PASS;PUBLICATION_POSTCHECK_REQUIRED',
 'review_type':'SELF_REVIEW;not an independent reviewer or formal proof',
 'requirements':[{'id':i,'state':'PROVED_FOR_ANALYSIS_CONTRACT' if i!='CRITICAL_EVIDENCE_AND_FINAL_CLOSURE' else 'STAGED_PASS_NOT_YET_PUBLISHED',
   'argument':arg,'evidence':[pin(Path(p)) for p in refs]} for i,arg,refs in arguments],
 'checks':checks,
 'nonblocking_conditional_unknowns':rec['unknowns'],
 'unanalysed_fixed_case_public_scope_items_identified':[],
 'missing_critical_evidence_for_claimed_conclusions':[],
 'limits':'Discovery/diagnosis only,no optimization implementation,no unconditional numericphysicalceiling or candidate speedup.No scope narrowing to a singleoperator/rank/window;conditionalcases explicit.',
 'overall_complete':False})
print(json.dumps({'status':'STAGED_PASS','checks':len(checks),'requirements':len(arguments),'P5_exact_recalculations':len(recalcs),'publication_postcheck_required':True}))
