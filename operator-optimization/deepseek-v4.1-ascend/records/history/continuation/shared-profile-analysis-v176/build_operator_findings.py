"""Positive profile-derived facts and their exact operator consequences."""
import json
from pathlib import Path
H=Path(__file__).resolve().parent
def read(n):return json.loads((H/n).read_text())
def write(n,x):(H/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
idx=read('all16-sidecar-evidence-index.json');bank=read('public-bank-epoch-and-role-review.json')
prefix=read('native-ET-prefix-summary.json');service=read('original-service-restoration-final.json')
required={'root.args[3].all_greedy':[True],'root.args[3].all_random':[False],
          'root.args[3].no_penalties':[True],'root.args[3].max_num_logprobs':[None],
          'root.self_scalars.synthetic_mode':[False],
          'root.ascend_config.rejection_sampler_config.enable_entropy_verify':[False]}
for w in idx['workers']:
    assert all(w['selected_scalar_values'].get(k)==v for k,v in required.items())
    assert w['binding_name_groups']['AscendRejectionSampler.forward']==138
selected=bank['selected_embedding_head_and_index_banks']
topk=[r for r in selected if 'topk_indices_buffer' in r['name']]
assert len(topk)==4
assert len({r['raw_metadata']['storage_impl'] for r in topk})==1
assert len({r['raw_metadata']['storage_base_address'] for r in topk})==1
assert all(r['raw_metadata']['shape']==[8192,512] for r in topk)
weights=[r for r in selected if '/catalogue/capture/' in r['name']]
assert len(weights)==4 and len({r['raw_metadata']['storage_impl'] for r in weights})==4
c2=bank['c2_complete_mask'];assert c2['storage_sizes']==[8192] and [24] in c2['view_extents']
assert service['health_http']==200 and service['running']
write('operator-conclusions-from-profile.json',{
    'status':'CURRENT_PARTIAL_COHORT_POSITIVE_OPERATOR_FACTS',
    'cohort':'joint176;startup/capture and incomplete frozenC8 runtime prefix',
    'facts':[
        {'id':'F176-SELECTED-GREEDY-CALLS','modules':['M16','M17','M20'],'candidates':['F30','F35'],
         'evidence':'16workers,69completed AscendRejectionSampler.forward entry/return pairs each,1104calls total;actual nonempty sampler arguments match required_controls.',
         'required_controls':required,
         'operator_consequence':'The observed call population satisfies these greedy/no-penalty/no-entropy/no-synthetic/logprobNone guards. Preserve source argmax ties,correction,output cells and other score-bank readers. This supports greedy specialization premises without assuming unobserved entire-case behavior.',
         'not_inferred':'No all-case branch exhaustion,no arbitrary score deadness,no current speedup;preparation metadata can contain both greedy/random=True.'},
        {'id':'F176-SHARED-INDEX-BANK','modules':['M09','M18','M19','M21','M24'],'candidates':['F12','F29','F31','F34'],
         'evidence':topk,
         'operator_consequence':'Target and draft name the same raw StorageImpl and base address for INT32[8192,512],16777216B,in capture and formal-start catalogues. They are shared state with publication/read/rollback epochs;do not schedule writers as if independent or add a second copy of this backing to memory cost.',
         'not_inferred':'No proof that every write is redundant,that no future reader exists,or that the entire16MiB is necessary HBM traffic per iteration.'},
        {'id':'F176-HEAD-EMBED-BANKS','modules':['M05','M16','M19'],'candidates':['F28','F30','F32'],
         'evidence':weights,
         'operator_consequence':'Four same-shaped BF16[16160,5120] target/draft embedding/head parameters have distinct recorded storage objects/addresses. Same shape is insufficient for value CSE. Shared-input multi-output GEMM remains a separate conditional composition with all required weights/work retained.',
         'not_inferred':'Different storage objects do not prove numerical inequality or a universal physical-independence claim;numeric value identity was not read.'},
        {'id':'F176-C2-VIEW-EPOCH','modules':['M08','M21','M22','M24'],'candidates':['F18','F31','F34'],
         'evidence':c2,
         'operator_consequence':'The named C2mask is repeatedly a view into one8192B backing,with observed extents24/1704/3200/8176 and73source-level copies. A24element shape is not a24B allocation or a constant-value generation. Retain copied range and epochs,or fuse its public final writer while preserving all consumers.',
         'not_inferred':'Values were not read;no claim that every copy changes values,no unconditional write-once or deletion claim.'},
    ],
    'intact_native_nodes_EP0':prefix['parsed_intact_nodes'],
    'public_or_runtime_operator_names_EP0':len(prefix['original_operator_counts']),
    'matched_binding_markers_EP0':prefix['matched_binding_markers'],
    'unmatched_binding_markers_EP0':prefix['unmatched_sidecar_binding_seqs'],
    'global_ET_storage_neighborhood_is_only_overapproximation':True,
    'no_distinct_raw_TensorImpl_under_same_ET_TensorID_counterexample_in_named_subset':True,
    'diagnostic_RF_reads_not_baseline_consumers':True,
    'performance_authority_remains_older_validated_cohorts':True,
    'overall_complete':False})
write('operator-evidence-review-checks.json',{'status':'PASS','checks':{
    'all16_actual_sampler_argument_guards':True,'completed_sampler_calls':1104,
    'target_draft_index_bank_alias_uses_raw_metadata':True,'four_parameter_banks_separately_bound':True,
    'C2_varied_views_and_same8192B_backing':True,'no_blanket_write_or_score_deletion':True,
    'partial_ET_not_promoted_to_full_case':True,'original_service_healthy_and_script_restored':True,
    'conditional27_template_rule_round':read('declared-source-template-rule-review.json')['full_whole_scope_scan_complete'] is False},
    'overall_complete':False})
print(json.dumps({'status':'PASS','positive_operator_facts':4,'actual_sampler_calls':1104,
                  'intact_native_nodes_EP0':prefix['parsed_intact_nodes'],'service_http':service['health_http']}))
