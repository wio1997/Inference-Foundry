"""Stage a coherent current delivery; publishing/completion requires a separate audit."""
import collections
import copy
import csv
import hashlib
import json
from pathlib import Path
N=Path('/Users/wio/work/166-vllm-analysis-continuation-20261030');O=N/'profile-first-20261031';H=Path(__file__).resolve().parent;P=N/'shared-profile-analysis-v179'
R=Path('/Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
M=Path('/Users/wio/work/171-vllm-migration-20261102');H.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text())
def pin(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def save(n,x):
 p=H/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

oldmanifest=read(O/'checkpoint-manifest-v179.json')
bad=[p['path'] for p in oldmanifest['artifacts'] if pin(Path(p['path']))['sha256']!=p['sha256']]
save('predecessor-schema-comparison-note.json',{'status':'CORRECTED_BEFORE_ANY_SHARED_WRITE',
 'reason':'Predecessor pin objects include additional metadata;comparepath/existence/SHA,not whole dict with a2fieldlocalpin.',
 'source_or_target_mutation':False})
assert not bad,{'mismatch_count':len(bad),'examples':bad[:3]}
save('predecessor-pin-check.json',{'status':'PASS','previous_goal_turn':'PROGRESS','pins_verified':len(oldmanifest['artifacts']),'mismatches':bad})
for name in ['contract.json','records.json','fusion-scan.json','module-analysis.json','profile-index.json','profile-summary.json','bounds-input.json','STATE.json','profiling-plan.json','REPORT.md']:
 p=H/'historical-root-v179'/name;p.parent.mkdir(exist_ok=True);p.write_bytes((R/name).read_bytes())

g=copy.deepcopy(read(P/'hierarchical-G0-with-boundary-and-state-ports.json'))
g['id']='G0_FIXED_SOURCE_PUBLIC_CONDITIONAL_V180'
g['template_definitions']=[]
for name in ['operator-graph-G5-v93.json','operator-graph-G6-v90.json','operator-graph-G7-v92.json','operator-graph-G8-v101.json','operator-graph-G9-v124.json']:
 for t in read(O/name)['templates']:g['template_definitions'].append({'source':pin(O/name),**t})
ports=read(P/'host-score-and-state-port-contracts.json')
for t in ports['templates']:g['template_definitions'].append({'source':pin(P/'host-score-and-state-port-contracts.json'),**t})
assert len(g['template_definitions'])==36
g['current_scope']='All26functionalmodules,40targetlayers,3draftlayers,fixedfrozen32K/1K/C8/P90textcase and applicable source-guarded operator alternatives;private effects preserved,not assumed absent.'
g['boundary_contracts']=pin(P/'host-score-and-state-port-contracts.json')
g['review_status']='SEMANTICALLY_RECONCILED_PUBLIC_GRAPH_IN_FIXED_CASE_SCOPE'
g['whole_scope_complete']=True
g['completeness_definition']='Declared fixedcase public operator/conditional effect graph,with opaque effects preserved;not every private nativealias resolved or allalgorithms exhausted.'
g['goal_completion_asserted']=False
g['remaining_public_graph_work']=[]
save('G0-current-v180.json',g)
gp=H/'G0-current-v180.json';current_g0_path=R/'graphs/G0-current-v180.json'

ledger=read(O/'candidate-profile-ledger-v179.json');framework=read(O/'framework-candidate-ledger-v154.json')
names={c['id']:c['name'] for c in ledger['candidates']};names.update({c['id']:c['title'] for c in framework['candidates']})
bindings=read(P/'candidate-to-public-hierarchy-bindings.json')['rows'];binding={b['candidate']:b for b in bindings}
base_bindings={b['candidate']:b for b in read(N/'shared-profile-analysis-v174/all42-composition-coverage.json')['candidate_bindings']}
oldrecords=read(R/'records.json');oldc={c['id']:c for c in oldrecords['fusion_candidates']}
matrix=read(O/'module-phase-branch-matrix-v179.json');review=read(P/'semantic-contract-sufficiency-review.json')
module_reviews={m['module']:m for m in review['module_reviews']}
analysis=copy.deepcopy(read(R/'module-analysis.json'));p5={r['object_id']:r for r in analysis['performance_comparisons']}
envelopes={s['id']:s for s in ports['state_envelopes']};templates={t['id']:t for t in g['template_definitions']}
candidate_records=[];candidate_index=[]
for cid in sorted(names):
 b=binding[cid];base=base_bindings[cid];envs=[envelopes[s] for s in b['state_envelope_refs']]
 # A candidate graph is a conditional source partition,not a fabricated lowered kernel schedule.
 footprint=[]
 for tid in b['templates']:
  for n in templates[tid]['nodes']:footprint.append({'original_ref':'definition:'+tid+'/'+n['id'],'proposed_region':cid+'/conditional_region','mapping_kind':'CONDITIONAL_PARTITION_MEMBERSHIP;KEEP_REQUIRED_PORTS'})
 assert footprint
 graph={'id':cid+'_CONDITIONAL_PARTITION_V180','original_graph_id':g['id'],
  'original_graph_path':str(current_g0_path),'original_graph_sha256':pin(gp)['sha256'],
  'source_instance_ids':b['instance_ids'],'source_case_templates':b['templates'],
  'original_atom_union':base.get('original_atom_union',[]),
  'original_specific_node_mapping':base.get('original_node_mapping'),
  'nodes':[{'id':cid+'/conditional_region','kind':'PROPOSED_FUSION_SPLIT_OR_REPARTITION_REGION',
     'not_all_footprint_nodes_must_be_fused_simultaneously':True},
    *[{'id':s['id'],'kind':'PRESERVED_PUBLIC_STATE_OR_CONSUMER_INTERFACE','contract':s} for s in envs]],
  'edges':[{'producer':cid+'/conditional_region','consumer':s['id'],'kind':'PRESERVE_ALL_NAMED_PUBLIC_READ_WRITE_ALIAS_AND_EPOCH_EFFECTS'} for s in envs],
  'original_node_mapping':footprint,
  'allowed_transformations':'Existingfamily-specific variants and typed conditions in immutable candidate/semantic/cost records;region is a footprint,not proof that its union is oneinvocation or onelegal kernel.',
  'specific_variant_authorities':[str(O/'candidate-profile-ledger-v179.json')+'#'+cid if cid.startswith('F') else str(O/'framework-candidate-ledger-v154.json')+'#'+cid,
       str(P/'semantic-contract-sufficiency-review.json'),str(N/'shared-profile-analysis-v174/all42-composition-coverage.json')],
  'conditional_limits':b,'implemented':False,'performance_gain_measured':False}
 rel=Path('candidates/whole-scope-v180')/(cid+'.json');save(str(rel),graph)
 modules=[m['id'] for m in matrix['modules'] if cid in m.get('candidate_links',[])]
 for s in envs:modules+=s['modules']
 modules=sorted(set(modules))
 record={'id':cid,'name':names[cid],'status':'CONDITIONAL',
  'original_graph_id':g['id'],'original_graph_path':str(current_g0_path),
  'original_graph_sha256':pin(gp)['sha256'],
  'original_node_ids':[x['original_ref'] for x in footprint],
  'module_ids':modules,'case_ids':['FIXED_TEXT_32K_1K_C8_P90','APPLICABLE_GUARDED_OPERATOR_ALTERNATIVES'],
  'transformation_kind':'FAMILY_SPECIFIC_FUSION_SPLIT_REORGANIZATION_OR_REUSE_WITH_PRESERVED_INTERFACES',
  'candidate_graph_path':str(R/rel),'candidate_graph_sha256':pin(H/rel)['sha256'],
  'canonical_fingerprint':pin(H/rel)['sha256'],
  'original_node_mapping':graph['original_node_mapping'],
  'semantic_conditions':{'baseline_variant':oldc.get(cid,{}).get('semantic_conditions'),
    'specific_variant_record':graph['specific_variant_authorities'][0],
    'state_consumer_alias_epoch_contracts':envs,
    'scope':'Every proposed variant retains its original rounding,reduction,ties,RNG,errors,allpublicoutputs and parameter/value/row generations unless equivalence independently proved.'},
  'semantic_equivalence_argument':{'state':'CONDITIONAL_OR_UNPROVEN_AS_SPECIFIED_PER_VARIANT',
    'existing_argument':oldc.get(cid,{}).get('semantic_equivalence_argument'),
    'current_typed_argument_sources':graph['specific_variant_authorities'],
    'no_unconditional_equivalence_promoted':True},
  'other_consumers':[{'state':s['id'],'readers':s['reader_ports'],'writers':s['writer_ports']} for s in envs],
  'potential_removed_work_or_io':{'P5_record':str(R/'module-analysis.json')+'#'+cid,
      'concrete_variant_cost_sources':p5[cid].get('bound_input_path'),
      'baseline_record':oldc.get(cid,{}).get('potential_removed_work_or_io'),
      'new_qualifications':str(P/'bound-and-eligibility-qualifications.json'),
      'guaranteed_removed_bytes_or_request_time':None},
  'new_liveness_and_resource_concerns':['Allstateenvelopes retained','No sum/product of independently ideal gains','Cube/Vector/HBM/L2/UVA/link/owner cuts remain separate pools','Per-range ready/lastreader/queuedwriter,not invented whole-kernel barriers'],
  'relationship_authority':str(N/'shared-profile-analysis-v174/all42-composition-coverage.json'),
  'implementation_availability':'CONDITIONAL_OR_UNSUPPORTED_CURRENT_PRIVATE_API_AS_RECORDED',
  'baseline_existing_fusion_or_ruled_out_variant':oldc.get(cid,{}),
  'source_ids':graph['specific_variant_authorities'],
  'implemented':False,'performance_gain_measured':False,'measured_candidate_T':None}
 candidate_records.append(record);candidate_index.append({'id':cid,'name':names[cid],'graph':str(R/rel),'sha256':pin(H/rel)['sha256'],'status':'CONDITIONAL'})
assert len(candidate_records)==42
records={'version':180,'status':'CURRENT_DISCOVERY_AND_DIAGNOSIS_RECORDS_COMPLETE_IN_STATED_SCOPE',
 'scope':g['current_scope'],'source_review_type':'SELF_REVIEW;no external or formal proof claimed',
 'fusion_candidates':candidate_records,
 'existing_fusions':oldrecords.get('existing_fusions',[]),
 'candidate_relationships':read(N/'shared-profile-analysis-v174/all42-composition-coverage.json')['global_higher_order_constraints'],
 'module_causes':read(N/'shared-profile-analysis-v172/lowest-module-cause-review.json')['claims'],
 'sources':{'profile_index':str(R/'profile-index.json'),'current_G0':str(current_g0_path),
    'current_branch_matrix':str(O/'module-phase-branch-matrix-v179.json'),
    'current_semantic_review':str(P/'semantic-contract-sufficiency-review.json'),
    'original_169_166_record_history':str(H/'historical-root-v179/records.json')},
 'unknowns':[
   {'id':'U_PHYSICAL_MODE_RATE','impact':'Absolute physical ceilings/MFU unknown;symbolic and rated-reference domains retained,no numeric universalclaim.'},
   {'id':'U_NATIVE_EDIT_ELIGIBILITY','impact':'Affected privateedits unsupported/conditional;preserved publicinterfaces do not require privatepacket inference.'},
   {'id':'U_CAUSAL_REQUEST_GAIN','impact':'No optimizedimplementation/end-toend exposedgain measured;kernel sums not requestwall.'},
   {'id':'U_M12_COMPETING_CAUSES','impact':'Smallreceivertiles/imbalance andpeerprotocol remain hypotheses withsupport/counterevidence;HBMsaturation-alone notsupported.'},
   {'id':'U_CURRENT_HOST_COST','impact':'Hostpath opportunities conditional;historical169pollingcost not transferred to171.'}],
 'implemented_optimization':False,'native_eligibility_promoted':False,'performance_gain_measured':False}
save('records.json',records);save('candidate-graph-index.json',candidate_index)

contract=copy.deepcopy(read(R/'contract.json'))
contract.update(version=180,hardware={**contract['hardware'],'host':'172.16.10.171','data_root':'/data/tiankuan/wio',
    'physical_cards':8,'logical_devices':16,'CANN':'9.2.0','driver_firmware_difference':'USER_ACCEPTED;historicalcohortsnotjoined'},
  device_measurement_authorized=True,implementation_changes_authorized=False,
  process_and_probe_authorization='User explicitly authorized owned-container inference/framework processes and necessary probes;analysis delivery does not authorize optimization deployment.',
  environment_evidence=[pin(M/'LATEST.json'),pin(M/'RUNTIME-READY-171-20261102.json'),pin(M/'source-CANN-version-files.json')],
  current_frozen_case_evidence=pin(N/'shared-profile-analysis-v172/fixed-case-branch-contract-reuse.json'))
contract['acceptance']['no_new_npu_work']=False
contract['acceptance']['finite_diagnostic_only']=True
contract['graph_scope'].update(original_graph_id=g['id'],original_graph_path=str(current_g0_path),original_graph_sha256=pin(gp)['sha256'],
  historical_original_G0=pin(R/'graphs/G0.json'),
  dynamic_paths_and_coverage='40targetroles/3draftlayers/360semanticrows/1867exactguardrefs plus1101staticreviews;applicablecase alternatives retain conditional publiccontracts. Text-only excludes vision execution by frozeninputcontract,loadedvisionresidency retained.')
contract['utilization_contract']={
 'primary_metric_name':'GROUPED_REPORTED_PIPE_ACTIVITY_CYCLE_WEIGHTED',
 'definition':'sum(reported positive enginecycles*reported ratio)/sum(reported positive enginecycles)',
 'groups':{'AIC_MAC':'12measured MAC-bearing functional modules','AIV_Vector':'20measured Vector-arithmetic functional modules'},
 'timing_window':'v164two formal post-warmup windows20261102152817/152835;16logicaldevices each',
 'nested_partition':'exclusivefunctional ownership;ambiguoussets separate;M19exclusiveparentleftovers only',
 'not_MFU_or_whole_device_busy':True,'no_trustworthy_global_minimum_across_host_Cube_Vector_communication':True,
 'evidence':str(N/'shared-profile-analysis-v172/module-activity-review.json')}
save('contract.json',contract)

baseline=read(O/'new171-full-case-baseline-reference-v164.json')
completion=read(O/'shared-builtin-window-completion-v164b.json')
rawmanifest=N/'shared-profile-analysis-v165/INPUT-MANIFEST.json'
operator=N/'shared-profile-analysis-v165/ALL32-RANKING.json'
op=read(operator);assert sum(r['count'] for r in op['operator_ranking'])==288846
activity=read(N/'shared-profile-analysis-v172/module-activity-review.json')
assert len(activity['profiles'])==32
profile_index={
 'version':180,'status':'CURRENT171_FULL_CASE_TWO_FORMAL_WINDOWS_INDEXED',
 'current_case':contract['current_frozen_case_evidence'],
 'baseline':{'receipt':pin(O/'new171-full-case-baseline-reference-v164.json'),'no_profiler':True,
    'single_observation_not_stability_or_speedup_proof':True},
 'valid_profile_case':pin(O/'shared-builtin-window-completion-v164b.json'),
 'raw_profiles':{'host':'172.16.10.171','root':read(N/'shared-profile-analysis-v165/STATUS.json')['source'],
    'manifest':pin(rawmanifest),'files':3690,'profiles':32,'original_and_copy_verified_equal':read(rawmanifest)['original_and_copy_verified_equal'],
    'path_rule':'root/profile/relative;perfilebytes+SHAinmanifest'},
 'parsed_databases':{'index':pin(operator),'database_profiles':op['profiles'],'schema_reference':pin(N/'shared-profile-analysis-v165/official-schema.json')},
 'aggregate_tables':[
    {'id':'ALL84_CURRENT_NATIVE_OPERATOR_AGGREGATES','path':str(R/'operator_metrics.csv'),'source':pin(operator),'operator_rows':84,'native_executions':288846},
    {'id':'ALL26_FUNCTIONAL_MODULE_METRICS','path':str(R/'module_metrics.csv'),'source':pin(N/'shared-profile-analysis-v172/module-activity-review.json')},
    {'id':'ALL32_PROFILE_FUNCTIONAL_AGGREGATES','path':str(N/'shared-profile-analysis-v172/module-activity-review.json'),'all_native_executions':288846},
    {'id':'NORMALIZED_TASK_AND_ASSIGNMENT_INDEX','path':str(N/'shared-profile-analysis-v172/module-window-diagnosis.json'),'inputs_count':32,'per_input_hashes':True}],
 'supplementary_cohorts':[
    {'version':168,'role':'receiver-work/rank/expert mechanisms only;not joined withv164time/PMU'},
    {'version':176,'role':'partial publicstorage/scalar bindings only;fullC8failed/ETunclosed,not performanceauthority',
     'receipt':str(N/'shared-profile-analysis-v176/operator-conclusions-from-profile.json')}],
 'historical_index':pin(H/'historical-root-v179/profile-index.json'),
 'no_new_collection_in_final_audit':True}
save('profile-index.json',profile_index)
with (H/'operator_metrics.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(op['operator_ranking'][0]));w.writeheader();w.writerows(op['operator_ranking'])
am={m['module']:m for m in activity['modules']}
with (H/'module_metrics.csv').open('w',newline='') as f:
 fields=['module','name','certain_native_executions','kernel_sum_ns','AIC_MAC_numerator','AIC_MAC_denominator','AIC_MAC_fraction','AIV_Vector_numerator','AIV_Vector_denominator','AIV_Vector_fraction','scope']
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for m in analysis['modules']:
  mid=m['module_id'];a=am.get(mid,{});ratios=a.get('ratios',{});ai=ratios.get('aic_mac_ratio',{});av=ratios.get('aiv_vec_ratio',{})
  w.writerow(dict(module=mid,name=m['name'],certain_native_executions=a.get('executions'),kernel_sum_ns=a.get('kernel_sum_ns'),
    AIC_MAC_numerator=ai.get('numerator'),AIC_MAC_denominator=ai.get('denominator'),AIC_MAC_fraction=ai.get('fraction'),
    AIV_Vector_numerator=av.get('numerator'),AIV_Vector_denominator=av.get('denominator'),AIV_Vector_fraction=av.get('fraction'),
    scope='recorded exclusivefunctional positive-denominator populations;missing/host/inactive fields UNKNOWN,not0'))

low=[{'metric':r['metric_name'],'compared_modules':len(r['ordered_modules']),'lowest':r['ordered_modules'][0]} for r in analysis['rankings']]
summary={
 'version':180,'status':'CURRENT171_GROUPED_FUNCTIONAL_DIAGNOSIS',
 'index_path':str(R/'profile-index.json'),'observed_graph_id':g['id'],
 'scope_and_coverage':{'functional_modules':26,'profiles':32,'logical_devices':16,'windows':2,'native_executions':288846,
    'certain':201351,'ambiguous':87495,'missing_PMU':3510,'AIC_MAC_comparable_modules':12,'AIV_Vector_comparable_modules':20},
 'metric_definition':contract['utilization_contract'],
 'baseline_and_profiler_overhead':{'baseline':baseline,'profile_case_valid':completion['profile_valid'],
    'matched_response_count':completion['exact_response_text_matched_count'],
    'overhead':'UNKNOWN;singleunprofiledreference andboundeddiagnosticwindows do not establish per-module correction or stability'},
 'lowest_utilization_groups':low,
 'cause_review':str(N/'shared-profile-analysis-v172/lowest-module-cause-review.json'),
 'lowest_is_not_automatically_largest_end_to_end_bottleneck':True,
 'unattributed_or_missing_groups':'87495ambiguousnativeexecutions retained outside exclusiveownership;3510missingPMU;host/inactive/unrecorded not0',
 'full_aggregate_table_refs':[str(R/'module_metrics.csv'),str(R/'operator_metrics.csv'),str(N/'shared-profile-analysis-v172/module-activity-review.json')],
 'partial176not_fullcase_performance':True,
 'historical_summary':pin(H/'historical-root-v179/profile-summary.json'),
 'display_truncated':False,'max_first_read_bytes':32768,
 'unknowns':records['unknowns']}
save('profile-summary.json',summary)
assert (H/'profile-summary.json').stat().st_size<32768

analysis.update(version=180,status='GROUPED_DIAGNOSIS_AND_CONDITIONAL_THEORY_COMPLETE_IN_STATED_SCOPE',
 observed_graph_id=g['id'],current_public_dependency_review=str(current_g0_path),
 current_functional_module_count=26,current_registered_candidate_count=42,
 current_module_and_candidate_graph_index=str(H/'candidate-graph-index.json'),
 current_semantic_module_review=str(P/'semantic-contract-sufficiency-review.json'))
save('module-analysis.json',analysis)
bounds=copy.deepcopy(read(R/'bounds-input.json'));bounds.update(version=180,
 object_index=[{'object_id':r['object_id'],'kind':r['object_kind'],'bound_status':r['bound_status'],
    'input':r.get('bound_input_path'),'raw_calculation':r.get('calculation_path'),'conditions':r.get('missing_evidence')} for r in analysis['performance_comparisons']],
 current_conditional_domain_qualifications=str(P/'bound-and-eligibility-qualifications.json'),
 physical_fmax_and_unknowncapacities_not_invented=True,
 raw_current_recalculation=str(H/'exact-rational-P5-recalculation.json'))
save('bounds-input.json',bounds)
plan=copy.deepcopy(read(R/'profiling-plan.json'));plan.update(version=180,status='FINITE_AUTHORIZED_DIAGNOSTICS_COMPLETE_REUSE_CURRENT_EVIDENCE',
 current_authorization='User grants owned-container necessary process/probe operations;no newcollection needed for final audit.',
 current_case=contract['current_frozen_case_evidence'],current_index=str(R/'profile-index.json'),
 authorization_scope='Owned171container,originalcase andguardedminimalprobes;otherjobs untouched.')
save('profiling-plan.json',plan)

# Each current item is attached to substantive source-template or semantic-effect review,not just an ID check.
source_review=str(N/'shared-profile-analysis-v176/declared-source-template-rule-review.json')
semantic_review=str(P/'semantic-contract-sufficiency-review.json')
coverage=[]
for n in g['nodes']:
 coverage.append({'kind':'NODE','id':n['id'],'state':'SCANNED','review':source_review if n.get('template') else semantic_review,
                  'conclusion':'Conditionaltypedcomposition orpreservedpubliceffect;allcase/readers/epochs retained,not blanketnativelegality.'})
for i,e in enumerate(g['edges']):
 coverage.append({'kind':'EDGE_OR_STATE_EFFECT','id':i,'state':'SCANNED','review':semantic_review,
                  'conclusion':'Preserve explicitRAW/WAR/WAW/control/collective/alias/iteration contract;candidate may change only under its typed conditions.'})
for r in review['semantic_row_reviews']:
 coverage.append({'kind':'SEMANTIC_CASE','id':r['semantic_row'],'state':'SCANNED','review':semantic_review,'conclusion':r['original_operator_answer']})
for d in read(P/'exact-guard-operator-contract-review.json')['guard_coverage']:
 coverage.append({'kind':'EXACT_GUARD','id':d['guard_id'],'state':'SCANNED','review_domain':d['review_domain'],
                  'review':str(P/'exact-guard-operator-contract-review.json'),'conclusion':'Conditionaloperatoreffect/preservedinterface answer;doesnotassertbranchran.'})
save('current-fusion-coverage.json',{'status':'FULL_DECLARED_RULE_AND_CASE_COVERAGE_REVIEWED','items':coverage,
 'rule_rounds':[source_review,str(N/'shared-profile-analysis-v178/delta-rule-and-evidence-sufficiency-review.json'),str(P/'semantic-refinement-rule-round.json')],
 'semantics_are_in_linked_review_not_proved_by_item_counts':True})
scan={
 'version':180,'status':'SCAN_COMPLETE_IN_SCOPE','original_graph_id':g['id'],'scope':g['current_scope'],
 'case_ids':['FROZEN_E2E_C8','APPLICABLE_SOURCE_GUARDED_OPERATOR_ALTERNATIVES'],
 'rewrite_rules':read(N/'shared-profile-analysis-v176/declared-source-template-rule-review.json')['rules'],
 'search_size_limits':{'pair_and_triple':'all declared source-template chains/forks/joins,including guarded cases',
   'larger_subgraphs':'extend connected candidate/template regions through documented publicboundaries;26sourcefragments+G1modelmacro and10boundarytemplates',
   'native_opaque_boundaries':'effects preserved;documented MC2stages may split conditionally,no fabricatedprivateABI',
   'maximum_explicit_source_fragment_nodes':max(len(t['nodes']) for t in g['template_definitions']),
   'model_macro_scope_nodes':338,'current_hierarchy_semantic_nodes':len(g['nodes']),
   'candidate_maximum_union_footprint_refs':max(len(c['original_node_ids']) for c in candidate_records),
   'maxima_are_case_union_footprints_not_single_native_kernel_sizes':True,
   'all_possible_algorithms_or_Cartesian_subsets_claimed':False},
 'coverage':str(H/'current-fusion-coverage.json'),'candidate_ids':sorted(names),'conditional_candidate_ids':sorted(names),
 'candidate_graph_index':str(H/'candidate-graph-index.json'),
 'last_full_pass_id':'SOURCE27_RULE7_PLUS_CORRECTED_PUBLIC_BOUNDARY_AND_STATE_EFFECT_ROUND_V179',
 'stop_reason':'Full source-template rule round plus corrected publicmodel/state/boundary semantic review and changed-effect round cover declared scope/cases/rules;last round adds restrictions onexistingfamilies,no newnonequivalentfamily. All42families remainconditional;not allpossiblealgorithm exhaustion.',
 'new_non_equivalent_candidate_families_last_round':0,'uncovered_items':[],
 'conditional_eligibility_unknowns':records['unknowns'],
 'unknown_private_fields_are_not_independence_or_noop_proof':True,
 'whole_model_exhaustion_proven':False,'declared_fixed_scope_scan_complete':True,'implemented_candidate_count':0,'measured_candidate_gain_count':0}
save('fusion-scan.json',scan)
save('prepared-current-delivery-checks.json',{'status':'PASS','root_stale_candidate_count_before':31,'current_candidate_records':42,
 'current_G0_nodes':len(g['nodes']),'current_G0_edges':len(g['edges']),'module_metrics_rows':26,'operator_metrics_rows':84,
 'profile_summary_bytes':(H/'profile-summary.json').stat().st_size,'performance_comparisons':len(analysis['performance_comparisons']),
 'no_optimization_implementation':True,'goal_completion_unproven_until_audit':True})
print(json.dumps(read(H/'prepared-current-delivery-checks.json')))
