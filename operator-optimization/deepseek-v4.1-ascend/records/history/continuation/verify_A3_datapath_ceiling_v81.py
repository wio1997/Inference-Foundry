"""Verify conditionalhardware/domain/counterclaims,not fullgoaloractualMFU."""
from pathlib import Path
from itertools import combinations
from math import isclose
from collections import Counter
import hashlib
from finalize_origin_coverage_v9 import N,O,R,W,T as TASK_DIR,L,read


def main(version=81):
    cp=read(O/(f'REVIEW-CHECKPOINT-v{version}.json'));assert cp['status']=='PROGRESS'and not cp['overall_complete']and cp['whole_modules_closed']==0
    mf=read(O/(f'checkpoint-manifest-v{version}.json'))['artifacts']
    for r in mf:
        b=Path(r['path']).read_bytes();assert len(b)==r['bytes']and hashlib.sha256(b).hexdigest()==r['sha256'],r['path']
    masters=read(O/(f'candidate-record-index-v{version}.json'))['model_candidates'];assert len(masters)==35
    for r in masters:assert hashlib.sha256(Path(r['master_record']).read_bytes()).hexdigest()==r['master_sha256']
    doc=read(O/'A3-datapath-primary-evidence-v80.json');assert len(doc['primary_sources'])==6 and all(r['status']=='READ'for r in doc['primary_sources'])
    for r in doc['primary_sources']:
        b=Path(r['snapshot_path']).read_bytes();assert len(b)==r['snapshot_bytes']and hashlib.sha256(b).hexdigest()==r['snapshot_sha256']
    assert any(r['id']=='NO-L1-UB-BYPASS'for r in doc['facts'])
    assert doc['captured_frequency_MHz']==1800 and not doc['captured_frequency_is_proven_hardware_maximum']
    assert doc['typed_Vector_Exp_Ln_Sqrt_Sort_INT64_numeric_capacity']is None and doc['actual_HBM_L2_GM_bandwidth_upper']is None
    d=read(O/'A3-typed-ingress-regime-models-v80.json');assert len(d['regimes'])==d['regime_count']==97
    assert sum(r['executions']for r in d['regimes'])==d['native_executions']==30030
    assert sum(r['native_PMU_records']for r in d['regimes'])==d['native_PMU_records']==30030 and d['missing_PMU_records']==0
    assert d['input_dtype_regime_counts']=={'DT_BF16':66,'INT8':31}
    for r in d['regimes']:
        weight_bytes=r['K']*r['N']*(1 if r['weight_dtype']=='INT8' else 2)
        assert weight_bytes==r['weight_payload_bytes']and r['conventional_dense_ops_2_per_MAC']==2*r['M']*r['N']*r['K']
        B=max(0,weight_bytes-24*65536)/(24*128);AB=max(0,weight_bytes-48*65536)/(24*(256+128))
        assert isclose(r['L1_only_B_min_elapsed_core_clock_cycles'],B)and isclose(r['L1_only_either_input_min_elapsed_core_clock_cycles'],AB)
        assert isclose(r['fixed_1800MHz_B_time_reference_us'],B/1800)and isclose(r['fixed_1800MHz_either_time_reference_us'],AB/1800)
        assert not r['actual_native_orientation_L1_ingress_or_no_compression_proved']and not r['physical_numeric_actual_time_ceiling_proved']
        assert r['B_model_exceeds_recorded_MTE1_counter']==r['either_model_exceeds_recorded_MTE1_counter']==0
    assert not d['global_profile_ambiguity_or_missing_PMU_rewritten']and not d['actual_physical_numeric_time_ceiling_proved']
    m=read(O/'Markov-warm-ingress-frontier-v80.json');assert len(m['records'])==m['executions']==520 and m['regimes']==3
    assert len({(r['database_index'],r['TASK_rowid'])for r in m['records']})==520
    q=m['five_step_recompute_domain'];assert q['weight_payload_bytes']==66191360 and q['B_orientation_required_weight_load_bytes']==323092480
    assert isclose(q['B_orientation_min_elapsed_cycles'],105173.33333333333)and isclose(q['B_orientation_fixed1800MHz_time_reference_us'],58.42962962962963)
    assert m['B_budget_exceedances']==m['either_budget_exceedances']==0 and not m['effective_load_throughput_or_memory_bottleneck_proved']
    for r in m['records']:
        assert isclose(r['aic_mte1_counter_equivalent_cycles'],r['aic_total_cycles']*r['aic_mte1_ratio'])
        assert not r['loaded_native_path_or_MAC_MFU_proved']
    j=read(O/'A3-joint-ingress-cut-proof-v80.json');assert sum(r['bytes']for r in j['planes'])==j['total_distinct_uncompressed_weight_bytes']==77332480
    assert j['one_initial_L0B_grant_bytes']==24*65536 and isclose(j['joint_min_elapsed_cycles'],24661.333333333332)
    assert j['joint_min_elapsed_cycles']>j['independent_per_kernel_grants_sum_cycles']and not j['universal_algorithm_lower_bound']and j['critical_path_gain']is None
    f=read(O/'A3-fusion-interface-frontiers-v80.json');assert len(f['candidates'])==f['candidate_count']==35
    assert {r['candidate_id']for r in f['candidates']}=={f'F{i:02d}'for i in range(1,36)}
    assert all(not r['whole_candidate_closed']and not r['actual_native_or_new_fused_path_validated']for r in f['candidates'])
    shared=read(O/'shared-CV-interface-revaluation-v80.json');assert len(shared['models'])==5
    for r in shared['models']:
        M,H,I=r['M'],r['H'],r['I'];expected=6*M*H+18*M*I
        assert sum(r['full_value_store_plus_read_GM_bridges_in_CV_ownership_domain'].values())==r['GM_bridge_reference_sum']==expected
        assert r['prior_v57_ideal_all_interface_value_budget_bytes']==expected+16*M
        split=r['split_activation_interface'];assert split['combined_internal_plus_external_reference_bytes']==24*M*I and split['added_reference_bytes_if_both_materialized']==8*M*I
        assert not r['HBM_traffic_necessary_from_GM_bridges_proved']and not r['GM_bandwidth_numeric_upper_known']and r['actual_latency_gain']is None
    assert next(r for r in shared['models']if r['M']==3)['GM_bridge_reference_sum']==216576
    cap=read(O/'A3-local-capacity-and-ceiling-scope-v80.json');v=cap['specific_plan_check']
    assert v['QB_result_bytes']==v['physical_UB']==196608 and v['C_and_S_bytes']==1536 and v['total']>v['physical_UB']and not v['fits']
    assert not cap['whole_model_lowest_MFU_module_established']and not cap['all_actual_physical_numeric_time_ceilings_complete']
    routers=read(O/'router-CV-interface-domains-v80.json');assert len(routers['models'])==3 and routers['models'][0]['same_width_two_GM_bridges_reference_bytes']==132096
    assert routers['live_BF16_expert_shared_plane_preserved']and routers['FP32_HF32_compute_numeric_capacity_not_inferred']and routers['actual_GM_or_HBM_savings_or_speedup']is None
    before={r['id']:r for r in read(O/'module-phase-branch-matrix-v79.json')['modules']};after={r['id']:r for r in read(O/(f'module-phase-branch-matrix-v{version}.json'))['modules']}
    inv=read(O/'A3-datapath-branch-inventory-v80.json');assert len(inv['modules'])==9 and inv['protected_other_modules']==17
    assert sum(len(r['branches'])for r in after.values())==342 and sum(r.get('branch_schema')=='SEMANTIC_BRANCHES_V1'for r in after.values())==24
    for mid in before:
        if mid not in inv['modules']:assert before[mid]==after[mid]
        else:
            assert read(Path(after[mid]['legacy_branch_history']))['module']==before[mid]and after[mid]['branches']==inv['modules'][mid]
            assert [r['id']for r in before[mid]['branches']]==[r['id']for r in after[mid]['branches']]
            for a,b in zip(before[mid]['branches'],after[mid]['branches']):
                assert b['evidence'][:len(a['evidence'])]==a['evidence']
                if a['id']not in inv['extended_existing_families']:assert a==b
    assert after['M07']['branches'][-1]['state']=='EXCLUDED_FROM_FROZEN_V41_QUERY_PATH;LEGACY_REFERENCE_ONLY'
    oldrel=read(O/'candidate-relation-clauses-v79.json');rel=read(O/(f'candidate-relation-clauses-v{version}.json'))
    assert rel['clauses']==oldrel['clauses']and len(rel['clauses'])==46
    assert rel['higher_order_groups'][:10]==oldrel['higher_order_groups']and len(rel['higher_order_groups'])==11
    assert rel['higher_order_groups'][-1]['id']=='HG80-A3-LOCAL-PATH'and len(rel['higher_order_groups'][-1]['members'])==13 and not rel['higher_order_groups'][-1]['all_subsets_analyzed']
    cg=read(O/(f'candidate-relation-coverage-v{version}.json'));nodes={r['id']for r in cg['nodes']}
    assert len(nodes)==35 and len(cg['pairs'])==595 and cg['scoped_conditional_pairs']==46 and cg['remaining_pair_relations']==549 and cg['higher_order_groups']==11
    assert {(r['a'],r['b'])for r in cg['pairs']}==set(combinations(sorted(nodes),2))and all(not r['whole_pair_all_variants_closed']for r in cg['pairs'])
    cov=read(O/(f'operator-analysis-coverage-ledger-v{version}.json'));assert cov['regimes']==read(O/'operator-analysis-coverage-ledger-v79.json')['regimes']
    reviewed=[r for r in cov['regimes']if r['state'].startswith('SCOPED_')];pending=[r for r in cov['regimes']if r['state'].startswith('REGIME_')]
    assert(len(reviewed),sum(r['executions']for r in reviewed))==(2073,285598)and(len(pending),sum(r['executions']for r in pending))==(9,1352)
    assert cov['unchanged_global_uncertainty']=={'ambiguous_module_assignment_executions':3136,'missing_PMU':849,'phase_unbound_graphs':16}
    for p in[N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json']:assert read(p)['latest_profile_first_review']==cp
    assert read(TASK_DIR/'index.json')['status']=='ACTIVE'and read(L/'state.json')['status']=='EVALUATING'
    history=read(O/(f'A3-datapath-validation-history-v{version}.json'));assert len(history['errors'])==(1 if version==80 else 2)
    for err in history['errors']:
        assert err['state']=='RESOLVED'and err['shared_state_writes_before_failure']==0
        assert hashlib.sha256(Path(err['saved_pre_fix_verifier']).read_bytes()).hexdigest()==err['sha256']
    runs=read(O/(f'task-run-registration-v{version}.json'))['runs'];assert Counter(r['status']for r in runs)==({'pass':3,'error':1} if version==80 else {'pass':2,'error':1})
    result=dict(checkpoint_version=version,status='PROGRESS',manifest_artifacts_verified=len(mf),primary_docs=6,typed_regimes=97,typed_executions=30030,typed_PMU=30030,
        counter_budget_exceedances=0,Markov=[3,520],five_step_conditional_cycles=q['B_orientation_min_elapsed_cycles'],candidate_frontiers=35,refined_module_surfaces=9,
        unchanged_other_modules=17,records=342,semantic_modules=24,relations=46,central_pending=549,higher_order_groups=11,
        actual_max_frequency_proved=False,actual_lowest_module_MFU_proved=False,profile_reviewed=[2073,285598],profile_pending=[9,1352],whole_modules_closed=0,overall_complete=False)
    print(result)
    return result


if __name__=='__main__':main()
