"""Validate the bounded relation ledger and its accounting, not goal completion."""
from collections import Counter
from itertools import combinations
from pathlib import Path
import hashlib
from finalize_origin_coverage_v9 import N,O,R,W,T as TASK_DIR,L,read


def main():
    cp=read(O/'REVIEW-CHECKPOINT-v73.json')
    assert cp['status']=='PROGRESS' and cp['whole_modules_closed']==0 and not cp['overall_complete']
    mf=read(O/'checkpoint-manifest-v73.json')['artifacts']
    for a in mf:
        b=Path(a['path']).read_bytes()
        assert len(b)==a['bytes'] and hashlib.sha256(b).hexdigest()==a['sha256'],a['path']
    registry=read(O/'candidate-record-index-v73.json')['model_candidates']
    assert len(registry)==35
    for row in registry:
        assert hashlib.sha256(Path(row['master_record']).read_bytes()).hexdigest()==row['master_sha256']
    graph=read(O/'candidate-relation-coverage-v73.json')
    clauses=read(O/'candidate-relation-clauses-v73.json')
    nodes={n['id'] for n in graph['nodes']}
    assert len(nodes)==35 and graph['pair_universe']==len(graph['pairs'])==595
    assert {(r['a'],r['b']) for r in graph['pairs']}==set(combinations(sorted(nodes),2))
    relations={(r['a'],r['b']):r for r in clauses['clauses']}
    assert len(relations)==clauses['clause_count']==28 and len(clauses['higher_order_groups'])==5
    selected=0
    for row in graph['pairs']:
        key=(row['a'],row['b'])
        assert not row['whole_pair_all_variants_closed'] and row['new_critical_path_gain_us'] is None
        if key in relations:
            assert row['clauses']==[relations[key]['id']] and row['scope_state']=='SCOPED_OPERATOR_RELATION_REVIEWED_CONDITIONAL'
            selected+=1
        else:
            assert not row['clauses'] and row['scope_state'].startswith('NOT_YET_RECONCILED_HERE')
    assert selected==graph['scoped_conditional_pairs']==28 and graph['remaining_pair_relations']==567
    assert graph['untouched_candidate_nodes']==['F01','F07','F27','F29','F33']
    touched={x for p in relations for x in p}
    assert len(touched)==30 and touched==set(clauses['touched_candidates'])
    for row in clauses['clauses']:
        assert row['evidence'] and not row['all_runtime_gates_proved'] and not row['speedup_additive']
        for fn in row['evidence']:
            assert (O/fn).exists()
    for row in clauses['higher_order_groups']:
        assert len(row['members'])>=4 and set(row['members'])<=nodes
        assert not row['all_subsets_analyzed'] and not row['legal_joint_fusion_proved']
    source=read(O/'candidate-relation-source-index-v73.json')
    assert len(source['files'])==11 and not source['new_remote_source_read']
    for row in source['files']:
        assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256']
    work=read(O/'joint-candidate-work-models-v73.json')
    lookup=work['Engram_owner_dedup']
    assert lookup['baseline_bytes']==27648 and lookup['joint_bytes']==1440
    assert lookup['joint_saved']==26208==sum(lookup['valid_ordered_decomposition'])
    assert lookup['invalid_sum_of_standalone_saved']==40320>lookup['baseline_bytes']
    assert not lookup['physical_host_or_UVA_traffic_proved'] and not lookup['actual_A_or_U_histogram_proved']
    rows=work['head_row_observer_union']
    assert rows['invalid_standalone_sum_saved']==30>24 and rows['all_observer_union_rows']==[0,1,2,3,4,5,10,20]
    assert rows['need_reopen_first_row_filter'] and rows['native_head_already_selected']
    for r in work['head_cold_dense_reference_models']:
        m=r['rows']
        assert r['weight_bytes']==165478400 and r['dense_direct_FLOPs']==2*m*5120*16160
        assert r['cold_materialized_reference_bytes']==165478400+2*m*(5120+16160)
        assert r['joint_reference_us']==max(r['Cube_376TF_reference_us'],r['HBM_1p6TB_reference_us'])
        assert r['HBM_1p6TB_reference_us']>r['Cube_376TF_reference_us'] and not r['measured']
    plan=work['plan_reuse_not_attention_value_reuse']
    assert (plan['old_attention'],plan['new_attention'])==(0,0.5)
    assert work['joint_resource_model']['status']=='SYMBOLIC_CEILING' and work['actual_runtime_speedup'] is None
    math=read(O/'candidate-composition-math-validation-v73.json')
    assert math['set_union_observer_cases']==87381 and math['no_Torch_or_native_execution']
    assert not math['actual_alias_reader_or_new_implementation_proof']
    prev=read(O/'candidate-profile-ledger-v72.json')
    current=read(O/'candidate-profile-ledger-v73.json')
    for a,b in zip(prev['candidates'],current['candidates']):
        assert a['id']==b['id']
        if a['id'] in touched:
            assert b['review_v73']['relation_clause_ids']
            assert not b['candidate_closed'] and not b['legal_fusion_proved'] and b['critical_path_gain_us'] is None
        else:
            assert a==b
    assert read(O/'module-phase-branch-matrix-v72.json')['modules']==read(O/'module-phase-branch-matrix-v73.json')['modules']
    oldcov=read(O/'operator-analysis-coverage-ledger-v72.json')
    cov=read(O/'operator-analysis-coverage-ledger-v73.json')
    assert oldcov['regimes']==cov['regimes'] and cov['new_pending_regimes_reconciled']==cov['new_pending_executions_reconciled']==0
    reviewed=[r for r in cov['regimes'] if r['state'].startswith('SCOPED_')]
    pending=[r for r in cov['regimes'] if r['state'].startswith('REGIME_')]
    assert (len(reviewed),sum(r['executions'] for r in reviewed))==(2073,285598)
    assert (len(pending),sum(r['executions'] for r in pending))==(9,1352)
    assert cov['unchanged_global_uncertainty']=={'ambiguous_module_assignment_executions':3136,'missing_PMU':849,'phase_unbound_graphs':16}
    for path in [N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json']:
        assert read(path)['latest_profile_first_review']==cp
    assert read(TASK_DIR/'index.json')['status']=='ACTIVE' and read(L/'state.json')['status']=='EVALUATING'
    runs=read(O/'task-run-registration-v73.json')['runs']
    assert len(runs)==3 and Counter(r['status'] for r in runs)=={'pass':3}
    assert len(read(O/'framework-candidate-ledger-v73.json')['candidates'])==7
    print(dict(manifest_artifacts_verified=len(mf),candidate_masters_verified=35,pairs_indexed=595,scoped_pairs=28,central_reconciliation_pending=567,candidates_touched=30,higher_order_groups=5,all26module_records_unchanged=True,profile_reviewed=[2073,285598],profile_pending=[9,1352],whole_modules_closed=0,checkpoint_status='PROGRESS',overall_complete=False))


if __name__=='__main__':
    main()
