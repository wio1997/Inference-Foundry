"""Verify evidence identities, joint models and ledger preservation; no target execution."""
from pathlib import Path
from itertools import combinations
import hashlib,sys
from finalize_origin_coverage_v9 import N,O,R,W,T as TASK_DIR,L,read


def verify_receipt(item):
    blob=Path(item['path']).read_bytes()
    assert len(blob)==item['bytes'] and hashlib.sha256(blob).hexdigest()==item['sha256'],item['path']


def main(analysis_only=False):
    inputs=read(O/'operator-composition-inputs-v83.json')
    for item in inputs['files']:verify_receipt(item)
    assert inputs['missing_optional_names']==['C2-physical-ring-storage-models-v79.json']
    supplement=read(O/'ring-persistence-input-supplement-v83.json')
    assert len(supplement['files'])==3
    for item in supplement['files']:verify_receipt(item)
    assert supplement['facts']['ring_objects']==3 and supplement['facts']['standalone_even_compaction_global_pool_divisor_change']==0
    graph=read(R/'graphs/source-templates-G5.json');overlay=read(O/'source-template-applicability-overlay-v83.json')
    assert (len(graph['templates']),sum(len(t['nodes'])for t in graph['templates']),sum(len(t['edges'])for t in graph['templates']))==(12,287,344)
    assert (overlay['templates'],overlay['nodes'],overlay['edges'])==(12,287,344) and len(overlay['corrections'])==6
    assert not overlay['original_graph_modified'] and not overlay['full_primitive_or_runtime_graph_proved']
    footprints=read(O/'candidate-operator-footprints-v83.json')['candidates']
    assert len(footprints)==35 and {r['id']for r in footprints}=={f'F{i:02d}'for i in range(1,36)}
    for row in footprints:
        assert row['operator_atom_union'] and row['modules']
        assert not row['simultaneous_variant_or_runtime_binding_proved']
        snapshot=O/(row['id']+'-master-before-v83.json')
        if analysis_only or not snapshot.exists():snapshot=R/'candidates'/(row['id']+'.json')
        assert hashlib.sha256(snapshot.read_bytes()).hexdigest()==row['captured_master_sha256']
    prior=read(O/'candidate-relation-clauses-v82.json')
    additions=read(O/'attention-composition-clauses-v83.json')
    assert len(additions['clauses'])==22 and len(additions['higher_order_groups'])==3
    expected_clauses=prior['clauses']+additions['clauses']
    expected_pairs={tuple(sorted([c['a'],c['b']]))for c in expected_clauses}
    assert len(expected_pairs)==69
    assert len({c['id']for c in expected_clauses})==69
    assert all(c['evidence'] and c['applicable_domain'] and not c['all_runtime_gates_proved'] and not c['speedup_additive'] for c in additions['clauses'])
    navigation=read(O/'candidate-pair-navigation-v83.json')
    assert len(navigation['pairs'])==595 and navigation['scoped_conditional_pairs']==69 and navigation['remaining_central_pair_reconciliation']==526
    allpairs={tuple(sorted([p['a'],p['b']]))for p in navigation['pairs']}
    assert allpairs==set(combinations(sorted(r['id']for r in footprints),2))
    assert {tuple(sorted([p['a'],p['b']]))for p in navigation['pairs']if p['conditional_clause_ids']}==expected_pairs
    assert all(not p['whole_pair_all_variants_closed']for p in navigation['pairs'])
    models=read(O/'attention-joint-interface-models-v83.json')
    for row in models['scale_models']:
        count=row['M'];assert row['both_private_FP16_plane_union_reference_bytes']==256*count
        assert row['retained_INT8_query_VC_GM_reference_bytes']==8192*count
        assert row['one_native_M16_broadcast_product_GM_plane_store_read_reference_bytes']==2048*count
    for row in models['output_models']:
        count=row['M'];assert row['distinct_transfer_union_reference_bytes']==(4*64*512+4*64*64)*count
        assert row['final_packed_BF16_output_write_bytes']==65536*count and row['actual_gain'] is None
    for row in models['dirty_persistent_publication_models']:
        expected=130*row['dirty_live_unique_index_rows']+1024*(row['dirty_live_unique_cmp_rows']+row['dirty_live_unique_SWA_rows'])
        assert row['distinct_bank_publication_total_bytes']==expected
    witness=models['split_normalized_output_information_loss_witness']
    assert witness['first_merged_output']==(1*0+1*1)/(1+1)==0.5
    assert witness['second_merged_output']==(3*0+1*1)/(3+1)==0.25
    assert models['crosscard_repartition_identity']['current_per_rank_query_key_pairs']==models['crosscard_repartition_identity']['repartition_per_rank_query_key_pairs']==3072
    first=models['completed_pair_projection_demand_examples'][0]
    assert first['start']==0 and first['length']==6
    assert first['current_pair_projection_rows']==[0,1,2,3,4,5] and first['even_only_future_ring_rows']==[0,2,4]
    assert not set(first['current_pair_projection_rows'])<=set(first['even_only_future_ring_rows'])
    for row in models['completed_pair_projection_demand_examples']:
        for a,b in row['completed_pairs']:assert a+1==b and a%2==0 and b%2==1
    resource=read(O/'joint-operator-resource-cut-proof-v83.json')
    assert resource['proof']['physical_numeric_beta_for_actual_A3_INT8_plus_BF16_or_Vector'] is None
    assert resource['proof']['published_752TOP_and_376TF_joint_simplex_not_proved']
    example=resource['illustrative_joint_window']
    assert [example[k]for k in ['QLI_INT8_direct_ops','QLI_useful_FP16_headreduce_ops','SMLA_BF16_direct_QK_PV_ops']]==[427819008,3342336,251658240]
    assert example['QLI_scalar_score_GM_interface_bytes']+example['SMLA_score_and_exponential_GM_interfaces_bytes']==1892352
    assert example['private_query_scale_plus_weight_interface_budget_bytes']==768
    credits=read(O/'attention-transfer-credit-ledger-v83.json')
    assert len(credits['transfers'])==25 and len(credits['compute_credit_atoms'])==4
    assert len({r['id']for r in credits['transfers']})==25
    for row in credits['examples']:
        count=row['M'];assert row['correct_unique_output_transfer_union_bytes']==147456*count
        assert row['wrong_sum_of_two_output_reports_bytes']==2*row['correct_unique_output_transfer_union_bytes']
        assert row['joint_scalar_plus_output_union_bytes']==147712*count
    if analysis_only:
        print(dict(pre_checkpoint_analysis='PASS',footprints=35,navigation=595,new_clauses=22,total_pairs=69,
            transfer_atoms=25,compute_atoms=4,joint_capacity_numeric=False,new_native_work=False))
        return
    manifest=read(O/'checkpoint-manifest-v83.json')
    for item in manifest['artifacts']:verify_receipt(item)
    current=read(O/'candidate-relation-clauses-v83.json')
    assert current['clauses'][:47]==prior['clauses'] and len(current['clauses'])==69
    assert len(current['higher_order_groups'])==14
    coverage=read(O/'candidate-relation-coverage-v83.json')
    assert (coverage['scoped_conditional_pairs'],coverage['remaining_pair_relations'],coverage['higher_order_groups'])==(69,526,14)
    assert sum(bool(p['clauses'])for p in coverage['pairs'])==69
    before={m['id']:m for m in read(O/'module-phase-branch-matrix-v82.json')['modules']}
    after={m['id']:m for m in read(O/'module-phase-branch-matrix-v83.json')['modules']}
    assert len(after)==26 and sum(len(m['branches'])for m in after.values())==342
    assert sum(m.get('branch_schema')=='SEMANTIC_BRANCHES_V1'for m in after.values())==24
    changed={'M07','M08','M09','M10','M11','M22'}
    for mid,mod in after.items():
        assert not mod.get('closed',False)
        assert [r.get('id')for r in mod['branches']]==[r.get('id')for r in before[mid]['branches']]
        if mid not in changed:assert mod['branches']==before[mid]['branches']
    candidates=read(O/'candidate-profile-ledger-v83.json')['candidates']
    assert len(candidates)==35 and all(not c['candidate_closed'] and not c['legal_fusion_proved']for c in candidates)
    assert len(read(O/'framework-candidate-ledger-v83.json')['candidates'])==7
    oldops=read(O/'operator-analysis-coverage-ledger-v82.json');ops=read(O/'operator-analysis-coverage-ledger-v83.json')
    assert ops['regimes']==oldops['regimes'] and ops['unchanged_global_uncertainty']==oldops['unchanged_global_uncertainty']
    assert (ops['scoped_conditional_review_regimes'],ops['scoped_conditional_review_executions'])==(2073,285598)
    assert (ops['remaining_regime_reconciliation_count'],ops['remaining_execution_reconciliation_count'])==(9,1352)
    cp=read(O/'REVIEW-CHECKPOINT-v83.json');assert cp['status']=='PROGRESS' and cp['whole_modules_closed']==0 and not cp['overall_complete']
    for path in [N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json']:
        assert read(path)['latest_profile_first_review']['candidate_ledger']==str(O/'candidate-profile-ledger-v83.json')
    assert read(TASK_DIR/'index.json')['status']=='ACTIVE' and read(L/'state.json')['status']=='EVALUATING'
    print(dict(checkpoint=83,status='PROGRESS',manifest_artifacts_verified=len(manifest['artifacts']),operator_footprints=35,
        pair_navigation=595,new_conditional_clauses=22,pairs=69,higher_order_groups=14,central_pending=526,
        transfer_atoms=25,compute_atoms=4,refined_modules=6,records=342,semantic_modules=24,
        profile_reviewed=[2073,285598],profile_pending=[9,1352],actual_joint_capacity_bound=False,
        whole_modules_closed=0,overall_complete=False))


if __name__=='__main__':main('--analysis-only'in sys.argv)
