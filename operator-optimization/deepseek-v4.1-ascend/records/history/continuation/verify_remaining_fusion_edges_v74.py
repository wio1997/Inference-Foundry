"""Bounded relation/source/value audit; no assertion of whole-goal completion."""
from collections import Counter
from itertools import combinations
from pathlib import Path
import hashlib
from finalize_origin_coverage_v9 import N,O,R,W,T as TASK_DIR,L,read


def main():
    cp=read(O/'REVIEW-CHECKPOINT-v74.json');assert cp['status']=='PROGRESS'and not cp['overall_complete']and cp['whole_modules_closed']==0
    manifest=read(O/'checkpoint-manifest-v74.json')['artifacts']
    for row in manifest:
        b=Path(row['path']).read_bytes();assert len(b)==row['bytes']and hashlib.sha256(b).hexdigest()==row['sha256'],row['path']
    masters=read(O/'candidate-record-index-v74.json')['model_candidates'];assert len(masters)==35
    for row in masters:assert hashlib.sha256(Path(row['master_record']).read_bytes()).hexdigest()==row['master_sha256']
    old=read(O/'candidate-relation-clauses-v73.json');rel=read(O/'candidate-relation-clauses-v74.json')
    assert rel['clauses'][:28]==old['clauses']and rel['higher_order_groups'][:5]==old['higher_order_groups']
    assert len(rel['clauses'])==43 and len(rel['higher_order_groups'])==8
    pairs={(r['a'],r['b'])for r in rel['clauses']};assert len(pairs)==43
    assert len(rel['touched_candidates'])==35
    graph=read(O/'candidate-relation-coverage-v74.json');nodes={n['id']for n in graph['nodes']}
    assert len(nodes)==35 and len(graph['pairs'])==595 and {(r['a'],r['b'])for r in graph['pairs']}==set(combinations(sorted(nodes),2))
    assert graph['scoped_conditional_pairs']==43 and graph['remaining_pair_relations']==552 and not graph['untouched_candidate_nodes']
    assert all(not r['whole_pair_all_variants_closed']for r in graph['pairs'])and not graph['all_higher_order_combinations_analyzed']
    for r in graph['pairs']:
        assert bool(r['clauses'])==((r['a'],r['b'])in pairs)
    src=read(O/'remaining-fusion-edge-source-index-v74.json');assert len(src['saved_contracts'])==18 and len(src['local_operator_files'])==4
    for row in src['saved_contracts']+src['local_operator_files']:assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256']
    q=read(O/'QB-head-RMS-RoPE-applicability-v74.json')
    assert q['source_requires_full512_for_head_RMS']and q['QR_norm1280_is_a_distinct_earlier_activation']and not q['historical_loaded_call_path_attested']
    assert not q['native_standalone_head_RMS_visibility_known']and q['retained_alias_readers_rstd_generation_and_capture_conditions']
    vis=read(O/'QB-to-query-RoPE-native-visibility-v74.json');assert vis['sequence_count']==len(vis['graph_sequences'])==3200
    assert vis['visible_middle_ComputeInfo_ops']=={}and vis['visible_standalone_RmsNorm512']==0
    assert len({(r['database_index'],r['Q_B_TASK_rowid'])for r in vis['graph_sequences']})==3200
    assert len({(r['database_index'],r['query_RoPE_TASK_rowid'])for r in vis['graph_sequences']})==3200
    assert not vis['native_alias_or_tensor_identity_proved']and not vis['missing_PMU_or_phase_binding_resolved']
    costs=read(O/'remaining-fusion-edge-costs-v74.json')
    assert costs['theory_status']=='CONDITIONAL_CEILING'and not costs['physical_numeric_full_joint_ceiling_proved']and costs['actual_runtime_gain']is None
    assert [r['M']for r in costs['QB_and_output_materialization_models']]==[2,3,800,816,830,834]
    for row,work in zip(costs['QB_and_output_materialization_models'],costs['head_RMS_source_direct_work_models']):
        m=row['M'];f=m*64*512;r=m*64*64
        assert row['QB_head_RMS_RoPE_reference_bytes']==6*f+4*r
        assert row['separate_QB_fused_head_RMS_RoPE_reference_bytes']==6*f
        assert row['joint_QB_head_RMS_RoPE_reference_bytes']==2*f
        assert row['head_RMS_RoPE_consumer_budget_bytes']==4*r and row['all_three_joint_budget_bytes']==4*f+4*r
        assert row['standalone_inverse_RoPE_pack_consumer_reference_bytes']==6*f and row['attention_inverse_RoPE_direct_pack_reference_bytes']==2*f
        assert work['head_rows']==m*64 and work['head_values']==f and work['sum_reduction_adds']==m*64*511
        assert work['QB_INT8_direct_GEMM_ops']==2*m*1280*32768 and work['actual_Vector_special_function_capacity']is None
    assert costs['QB_and_output_materialization_models'][1]['all_three_joint_budget_bytes']==442368
    for row in costs['C1_joint_projection_models']:
        assert row['two_distinct_BF16_weight_bytes']==10485760
        assert row['separate_materialized_reference_bytes']-row['joint_materialized_reference_bytes']==2*row['M']*5120
        assert row['separate_512_norms_and_preRoPE_fork_retained']
    assert costs['C1_joint_projection_models'][1]['separate_reference_us']==costs['C1_joint_projection_models'][1]['joint_reference_us']
    math=read(O/'remaining-fusion-edge-math-v74.json');assert math['ordered_three_source_selection_cases']==48 and not math['native_tests']
    witness=math['BF16_stage_order_counterexample']
    assert witness['original_norm_then_RoPE_last_pair']==[-2.703125,2.953125]and witness['reordered_RoPE_then_norm_last_pair']==[-2.71875,2.9375]
    gap=read(O/'candidate-stale-gap-reconciliation-v74.json');assert set(gap['candidates'])=={'F27','F34'}
    candidates={c['id']:c for c in read(O/'candidate-profile-ledger-v74.json')['candidates']}
    for cid,x in gap['candidates'].items():
        assert candidates[cid]['conclusion_changing_gap']==x['current_gap']and candidates[cid]['next_profile_or_contract_action']==x['current_action']
        assert x['prior_gap']!=x['current_gap']and not x['candidate_closed']
    assert candidates['F07']['profile_review_state']=='FULL_HEAD_RMS_PATH_CONTRACT_CORRECTED;HISTORICAL_GRAPH_APPLICABILITY_OPEN'
    before={m['id']:m for m in read(O/'module-phase-branch-matrix-v73.json')['modules']}
    after={m['id']:m for m in read(O/'module-phase-branch-matrix-v74.json')['modules']}
    inv=read(O/'remaining-fusion-branch-inventory-v74.json');assert len(before)==len(after)==26
    for mid in before:
        if mid!='M07':assert before[mid]==after[mid]
    assert after['M07']['branches']==inv['modules']['M07']and len(after['M07']['branches'])==12
    assert [b['id']for b in after['M07']['branches'][:11]]==[b['id']for b in before['M07']['branches']]
    assert after['M07']['branches'][-1]['id']=='M07-OP-12'
    assert read(Path(after['M07']['legacy_branch_history']))['module']==before['M07']
    cov=read(O/'operator-analysis-coverage-ledger-v74.json');prev=read(O/'operator-analysis-coverage-ledger-v73.json');assert cov['regimes']==prev['regimes']
    reviewed=[r for r in cov['regimes']if r['state'].startswith('SCOPED_')];pending=[r for r in cov['regimes']if r['state'].startswith('REGIME_')]
    assert(len(reviewed),sum(r['executions']for r in reviewed))==(2073,285598)and(len(pending),sum(r['executions']for r in pending))==(9,1352)
    assert cov['unchanged_global_uncertainty']=={'ambiguous_module_assignment_executions':3136,'missing_PMU':849,'phase_unbound_graphs':16}
    assert cov['new_pending_regimes_reconciled']==cov['new_pending_executions_reconciled']==0
    for path in [N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json']:assert read(path)['latest_profile_first_review']==cp
    assert read(TASK_DIR/'index.json')['status']=='ACTIVE'and read(L/'state.json')['status']=='EVALUATING'
    runs=read(O/'task-run-registration-v74.json')['runs'];assert len(runs)==3 and Counter(r['status']for r in runs)=={'pass':3}
    print(dict(manifest_artifacts_verified=len(manifest),candidate_master_hashes=35,relation_pairs=43,central_pending=552,all35_candidate_nodes_touched=True,higher_order_groups=8,graph_visibility_sequences=3200,head_RMS_semantic_family_added=True,M07_semantic_families=12,other25modules_identical=True,stale_plan_support_gaps_corrected=['F27','F34'],profile_reviewed=[2073,285598],profile_pending=[9,1352],whole_modules_closed=0,status='PROGRESS',overall_complete=False))


if __name__=='__main__':main()
