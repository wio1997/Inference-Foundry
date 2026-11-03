"""Verify bounded predicate/source/coverage claims,not full-goal completion."""
from collections import Counter
from itertools import combinations
from pathlib import Path
import hashlib
from finalize_origin_coverage_v9 import N,O,R,W,T as TASK_DIR,L,read


def main():
    cp=read(O/'REVIEW-CHECKPOINT-v76.json');assert cp['status']=='PROGRESS'and not cp['overall_complete']and cp['whole_modules_closed']==0
    mf=read(O/'checkpoint-manifest-v76.json')['artifacts']
    for row in mf:
        b=Path(row['path']).read_bytes();assert len(b)==row['bytes']and hashlib.sha256(b).hexdigest()==row['sha256'],row['path']
    masters=read(O/'candidate-record-index-v76.json')['model_candidates'];assert len(masters)==35
    for row in masters:assert hashlib.sha256(Path(row['master_record']).read_bytes()).hexdigest()==row['master_sha256']
    source=read(O/'router-predicate-source-index-v76.json');assert len(source['source_files'])==10 and not source['new_remote_source_read']
    for row in source['source_files']:assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256']
    guards=read(O/'router-predicate-guard-models-v76.json');assert(guards['regime_count'],guards['native_executions'])==(7,4472)
    assert guards['expert_populations']=={'384':4160,'128':312}and guards['vision_config_num_hidden_layers']==32
    assert guards['router_sentinel_interval']==[129257,129262]and guards['engram_image_IDs']==[129264,129265]
    keys=[]
    for r in guards['regimes']:
        assert r['described_vision_bias_present']and r['described_text_bias_present']and r['tid2eid_DT_UNDEFINED']
        assert r['source_hashFlag_from_optional_input_presence']==0 and r['source_addBiasVl_from_optional_input_presence']==1
        assert not r['actual_token_values_or_vision_row_population_known']and not r['loaded_native_tiling_blob_attested']
        keys +=[tuple(k)for k in r['native_TASK_keys']]
    assert len(keys)==len(set(keys))==4472
    proof=read(O/'router-ID-predicate-source-proof-v76.json');i=proof['existing_INT32_route']
    assert i['current_I64_numeric_key']==6 and i['source_I32_numeric_key']==4 and i['host_device_suffix_names_are_different_but_numeric_keys_match']
    assert not i['loaded_key4_or_bridge_variant_attested']and not i['extra_I32_kernel_source_codegen_required']
    cases=i['complete_signed32_interval_cases'];assert len(cases)==5 and sum(r['hi']-r['lo']+1 for r in cases)==i['domain_cardinality']==1<<32
    assert cases[0]['lo']==-(1<<31)and cases[-1]['hi']==(1<<31)-1
    for a,b in zip(cases,cases[1:]):assert a['hi']+1==b['lo']
    assert not proof['formal_solver_or_native_test']and proof['raw_ID_bank_and_Engram_lookback_minus1_semantics_unchanged']
    witnesses={r['token_ID']:r for r in proof['cross_field_witnesses']}
    assert witnesses[129257]['router_image_row']and not witnesses[129257]['engram_image_dead']
    assert not witnesses[129264]['router_image_row']and witnesses[129264]['engram_image_dead']
    assert proof['padding_not_function_of_ID_alone']=={'raw_ID':0,'real_prefix_padding':False,'zero_filled_alignment_tail_padding':True}
    costs=read(O/'router-predicate-fusion-costs-v76.json');first=costs['models'][0]
    assert(first['M'],first['layers'])==(3,40)
    assert first['source_INT64_prep_and_TopK_ID_reference_bytes']==5840
    assert first['same_INT64_native_no_caller_sanitizer_reference_bytes']==2400 and first['same_INT64_sanitizer_budget_bytes']==3440
    assert first['same_INT64_cast_once_plus_native_consumers_reference_bytes']==996 and first['existing_v63_sanitize_once_plus_native_consumers_reference_bytes']==1082
    assert first['native_I32_ID_reference_bytes']==480 and first['native_I32_full_ID_prep_budget_bytes']==5360
    assert first['new_BOOL_predicate_once_reference_bytes']==135
    assert costs['graph_native_cleanup_groups']==3200 and costs['graph_native_Equal_Fill_Select_records']==9600
    assert len(costs['deleted_kernel_keys_if_variant_legal'])==len(set(map(tuple,costs['deleted_kernel_keys_if_variant_legal'])))==9600
    assert costs['task_sums_are_not_request_critical_path_or_host_launch_counts']and costs['actual_measured_speedup']is None
    for m in costs['aligned_ID_plane_models']:
        n,p,l=m['raw_ID_rows'],m['aligned_TopK_rows'],m['layers']
        assert m['same_I64_cleanup_budget_bytes']==l*(26*p+8)
        assert m['source_Cast_pad_cleanup_consumer_reference_bytes']-m['same_I64_no_sanitizer_reference_bytes']==m['same_I64_cleanup_budget_bytes']
        assert not m['actual_TopK_paired_invocation_attested']
    assert[(r['raw_ID_rows'],r['aligned_TopK_rows'],r['layers'])for r in costs['aligned_ID_plane_models']]==[(800,816,40),(816,816,40),(3,3,40),(2,2,3),(3,3,3)]
    variants=read(O/'router-predicate-fusion-variants-v76.json');assert variants['variant_count']==len(variants['variants'])==5
    assert all(not r['new_main_candidate']and not r['loaded_native_variant_validated']and r['actual_measured_speedup']is None for r in variants['variants'])
    bools=read(O/'BOOL-prefix-copy-frontier-v76.json');assert(bools['regimes'],bools['executions'],bools['total_known_BOOL_value_write_bytes'])==(2,80,1920)
    assert bools['copy_count_histogram']=={'4':80}and len(bools['records'])==80
    assert bools['pending_regimes_promoted']==bools['pending_executions_promoted']==0
    assert all(not r['field_or_reader_identity_proved']and not r['copy_roles_addresses_and_lengths_bound']for r in bools['records'])
    assert not any(bools['profile_memory_flags'])and len(bools['profile_memory_flags'])==32
    assert not bools['server_capture_cache_file_existence_rechecked']
    history=read(O/'router-predicate-local-validation-history-v76.json')
    for r in history['partial_artifacts']:
        path=Path(r['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256']
        current=O/path.name.replace('-before-extractor-fix','');assert read(current)==read(path)
    assert 'resume_guard_error'in history
    oldrel=read(O/'candidate-relation-clauses-v75.json');rel=read(O/'candidate-relation-clauses-v76.json')
    assert rel['clauses'][:44]==oldrel['clauses']and rel['higher_order_groups'][:9]==oldrel['higher_order_groups']
    assert len(rel['clauses'])==45 and len(rel['higher_order_groups'])==10
    assert rel['clauses'][-1]['id']=='REL76-01'and(rel['clauses'][-1]['a'],rel['clauses'][-1]['b'])==('F20','F24')
    cg=read(O/'candidate-relation-coverage-v76.json');nodes={n['id']for n in cg['nodes']}
    assert len(nodes)==35 and len(cg['pairs'])==595 and cg['scoped_conditional_pairs']==45 and cg['remaining_pair_relations']==550
    assert{(r['a'],r['b'])for r in cg['pairs']}==set(combinations(sorted(nodes),2))and all(not r['whole_pair_all_variants_closed']for r in cg['pairs'])
    before={m['id']:m for m in read(O/'module-phase-branch-matrix-v75.json')['modules']};after={m['id']:m for m in read(O/'module-phase-branch-matrix-v76.json')['modules']}
    inv=read(O/'router-predicate-branch-inventory-v76.json');assert len(before)==len(after)==26
    for mid in before:
        if mid not in inv['modules']:assert before[mid]==after[mid]
        else:
            assert after[mid]['branches']==inv['modules'][mid]and read(Path(after[mid]['legacy_branch_history']))['module']==before[mid]
            assert[r['id']for r in before[mid]['branches']]==[r['id']for r in after[mid]['branches']]
            for a,b in zip(before[mid]['branches'],after[mid]['branches']):
                assert b['evidence'][:len(a['evidence'])]==a['evidence']
                if a['id']not in inv['extended_families']:assert a==b
    assert inv['retained_families']=={'M05':10,'M12':15,'M14':19}and len(inv['extended_families'])==8
    cov=read(O/'operator-analysis-coverage-ledger-v76.json');prev=read(O/'operator-analysis-coverage-ledger-v75.json');assert cov['regimes']==prev['regimes']
    reviewed=[r for r in cov['regimes']if r['state'].startswith('SCOPED_')];pending=[r for r in cov['regimes']if r['state'].startswith('REGIME_')]
    assert(len(reviewed),sum(r['executions']for r in reviewed))==(2073,285598)and(len(pending),sum(r['executions']for r in pending))==(9,1352)
    assert cov['unchanged_global_uncertainty']=={'ambiguous_module_assignment_executions':3136,'missing_PMU':849,'phase_unbound_graphs':16}
    assert cov['new_pending_regimes_reconciled']==cov['new_pending_executions_reconciled']==0
    for path in[N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json']:assert read(path)['latest_profile_first_review']==cp
    assert read(TASK_DIR/'index.json')['status']=='ACTIVE'and read(L/'state.json')['status']=='EVALUATING'
    runs=read(O/'task-run-registration-v76.json')['runs'];assert len(runs)==5 and Counter(r['status']for r in runs)=={'pass':3,'error':2}
    print(dict(manifest_artifacts_verified=len(mf),candidate_masters=35,TopK_guard_regimes=7,TopK_guard_executions=4472,source_INT32_key=4,current_INT64_key=6,source_same_I64_cleanup_route=True,graph_cleanup_native_records=9600,BOOL80_still_pending=True,retained_semantic_families=44,enriched=8,other23modules_identical=True,relations=45,central_pending=550,higher_order_groups=10,profile_reviewed=[2073,285598],profile_pending=[9,1352],whole_modules_closed=0,status='PROGRESS',overall_complete=False))


if __name__=='__main__':main()
