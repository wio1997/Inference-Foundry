"""Verify scoped evidence arithmetic, provenance and unchanged global gaps."""
from collections import Counter
from itertools import combinations
from pathlib import Path
import hashlib
from finalize_origin_coverage_v9 import N,O,R,W,T as TASK_DIR,L,read


def main():
    cp=read(O/'REVIEW-CHECKPOINT-v77.json')
    assert cp['status']=='PROGRESS'and not cp['overall_complete']and cp['whole_modules_closed']==0
    mf=read(O/'checkpoint-manifest-v77.json')['artifacts']
    for r in mf:
        b=Path(r['path']).read_bytes();assert len(b)==r['bytes']and hashlib.sha256(b).hexdigest()==r['sha256'],r['path']
    masters=read(O/'candidate-record-index-v77.json')['model_candidates'];assert len(masters)==35
    for r in masters:assert hashlib.sha256(Path(r['master_record']).read_bytes()).hexdigest()==r['master_sha256']
    source=read(O/'padding-route-source-index-v77.json');assert len(source['source_files'])==6
    for r in source['source_files']:assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256']
    assert not source['new_remote_source_read']and source['new_source_files_obtained']==0
    doc=source['documented_sort_reference'];assert not doc['used_as_numeric_capacity_or_loaded_binary_proof']
    if 'snapshot_path'in doc:assert hashlib.sha256(Path(doc['snapshot_path']).read_bytes()).hexdigest()==doc['snapshot_sha256']
    d=read(O/'padding-route-domain-v77.json');assert d['known_MC2_router_pad_calls']==320 and d['M816_TopK_population']==640
    assert not d['all_M816_calls_assumed_padded']and not d['cache_hits_or_request_time_fraction_proved']and not d['functional_assignments_rewritten']
    assert len(d['groups'])==8 and all(r['pad_owners']==r['TopK_owners']==40 and not r['repeated_same_layer_cache_uses_proved']for r in d['groups'])
    pairs=d['source_order_correspondences'];assert len(pairs)==len({(r['database_index'],r['TopK_TASK_rowid'])for r in pairs})==320
    assert not any(r['native_input_pointer_or_layer_parameter_ID_attested']for r in pairs)
    assert d['explicit_padding_score_elements']==1966080 and d['all_observed_TopK_score_elements']==306816000
    sem=read(O/'padding-route-semantic-proof-v77.json');w=sem['reference_FP32_witness']
    assert w['stage_rounded_weight']!=w['algebraic_one_over_k']
    assert sem['ranking_witness']['rounded_phi_plus_bias'][0]==sem['ranking_witness']['rounded_phi_plus_bias'][1]
    assert sem['ranking_witness']['bias'][0]<sem['ranking_witness']['bias'][1]
    assert not sem['consumer_frontier']['all_global_hooks_or_aliases_closed']
    c=read(O/'padding-route-cost-models-v77.json');n,p,e,k=800,816,384,6
    original=4*p*e+8*p+8*p*k+8*e
    integrated=4*n*e+8*n+8*p*k+8*e+8*k
    assembly=4*n*e+8*n+2*8*n*k+8*k+8*p*k+8*e
    assert(c['original_TopK_reference_bytes'],c['integrated_prefix_plus_tail_reference_bytes'],c['native_prefix_plus_separate_assembly_reference_bytes'])==(original,integrated,assembly)==(1302144,1277488,1354288)
    assert c['integrated_reference_budget_bytes']==24656 and c['separate_assembly_extra_bytes']==52144
    assert c['joint_reference_budget_bytes']==2506832 and c['joint_already_counted_direct_producer_valid_prefix_copy_budget_bytes']==2457600 and c['joint_incremental_reference_over_that_budget_bytes']==49232
    assert c['router_GEMM_padded_row_FLOPs_saved']==0 and c['output_template_bytes']==48 and c['forty_layer_template_payload_bytes_per_rank']==1920
    assert c['fixed_row_loop_model']['longest_row_count_before']==c['fixed_row_loop_model']['longest_row_count_after']==17
    assert c['partition_original']['last_program_rows']==17 and c['partition_prefix']['last_program_rows']==1
    assert c['theory']['state']=='SYMBOLIC_CEILING'and not c['theory']['reference_bytes_are_proven_necessary_HBM']
    assert c['theory']['request_time_lower_bound_ns']is None and c['actual_measured_speedup']is None
    v=read(O/'padding-route-variants-v77.json');assert len(v['variants'])==v['variant_count']==5
    assert v['conditional_operator_analysis_complete_for_this_question']and not v['whole_candidate_all_variants_closed']
    assert all(not r['new_main_candidate']and not r['native_validated']and r['actual_measured_speedup']is None for r in v['variants'])
    assert not v['repeated_layer_hits_proved']and v['all_EP_receiver_work_preserved']
    assert v['build_options'][1]['strict_positive_rowwork_break_even_uses']==52 and v['build_options'][2]['reference_copy_bytes']==96
    history=read(O/'padding-route-validation-history-v77.json');assert len(history['errors'])==1
    err=history['errors'][0];assert err['partial_artifact_writes']==err['checkpoint_or_shared_state_mutations']==0
    assert hashlib.sha256(Path(err['saved_pre_fix_script']).read_bytes()).hexdigest()==err['sha256']
    old=read(O/'candidate-relation-clauses-v76.json');rel=read(O/'candidate-relation-clauses-v77.json')
    assert len(rel['clauses'])==45 and len(rel['higher_order_groups'])==10 and old['higher_order_groups']==rel['higher_order_groups']
    for a,b in zip(old['clauses'],rel['clauses']):
        if a['id']!='REL73-07':assert a==b
        else:
            assert b['evidence'][:len(a['evidence'])]==a['evidence']
            assert not b['refinement_v77']['all_runtime_gates_proved']
    cg=read(O/'candidate-relation-coverage-v77.json');nodes={r['id']for r in cg['nodes']}
    assert len(nodes)==35 and len(cg['pairs'])==595 and cg['scoped_conditional_pairs']==45 and cg['remaining_pair_relations']==550
    assert {(r['a'],r['b'])for r in cg['pairs']}==set(combinations(sorted(nodes),2))
    assert all(not r['whole_pair_all_variants_closed']for r in cg['pairs'])
    before={r['id']:r for r in read(O/'module-phase-branch-matrix-v76.json')['modules']};after={r['id']:r for r in read(O/'module-phase-branch-matrix-v77.json')['modules']}
    inv=read(O/'padding-route-branch-inventory-v77.json');assert inv['retained_families']=={'M12':15}and len(inv['extended_families'])==6
    for mid in before:
        if mid!='M12':assert before[mid]==after[mid]
        else:
            assert after[mid]['branches']==inv['modules'][mid]and read(Path(after[mid]['legacy_branch_history']))['module']==before[mid]
            assert[r['id']for r in before[mid]['branches']]==[r['id']for r in after[mid]['branches']]
            for a,b in zip(before[mid]['branches'],after[mid]['branches']):
                assert b['evidence'][:len(a['evidence'])]==a['evidence']
                if a['id']not in inv['extended_families']:assert a==b
    cov=read(O/'operator-analysis-coverage-ledger-v77.json');assert cov['regimes']==read(O/'operator-analysis-coverage-ledger-v76.json')['regimes']
    reviewed=[r for r in cov['regimes']if r['state'].startswith('SCOPED_')];pending=[r for r in cov['regimes']if r['state'].startswith('REGIME_')]
    assert(len(reviewed),sum(r['executions']for r in reviewed))==(2073,285598)
    assert(len(pending),sum(r['executions']for r in pending))==(9,1352)
    assert cov['unchanged_global_uncertainty']=={'ambiguous_module_assignment_executions':3136,'missing_PMU':849,'phase_unbound_graphs':16}
    for pth in[N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json']:assert read(pth)['latest_profile_first_review']==cp
    assert read(TASK_DIR/'index.json')['status']=='ACTIVE'and read(L/'state.json')['status']=='EVALUATING'
    runs=read(O/'task-run-registration-v77.json')['runs'];assert len(runs)==4 and Counter(r['status']for r in runs)=={'pass':3,'error':1}
    print(dict(status='PROGRESS',manifest_artifacts_verified=len(mf),explicit_pad_calls=320,source_order_correspondences=320,native_pointer_bindings=0,
        byte_references=[original,integrated,assembly],longest_row_count=[17,17],conditional_question_reviewed=True,
        retained_M12_families=15,enriched=6,other25modules_identical=True,relations=45,central_pending=550,higher_order_groups=10,
        profile_reviewed=[2073,285598],profile_pending=[9,1352],whole_modules_closed=0,overall_complete=False))


if __name__=='__main__':main()
