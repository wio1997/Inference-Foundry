"""Audit corrected V4.1 operator entry, native source bindings and bounded costs."""
from collections import Counter
from itertools import combinations
from pathlib import Path
import hashlib
from finalize_origin_coverage_v9 import N,O,R,W,T as TASK_DIR,L,read


def main():
    cp=read(O/'REVIEW-CHECKPOINT-v75.json');assert cp['status']=='PROGRESS'and cp['whole_modules_closed']==0 and not cp['overall_complete']
    mf=read(O/'checkpoint-manifest-v75.json')['artifacts']
    for row in mf:
        b=Path(row['path']).read_bytes();assert len(b)==row['bytes']and hashlib.sha256(b).hexdigest()==row['sha256'],row['path']
    masters=read(O/'candidate-record-index-v75.json')['model_candidates'];assert len(masters)==35
    for row in masters:assert hashlib.sha256(Path(row['master_record']).read_bytes()).hexdigest()==row['master_sha256']
    c=read(O/'V41-query-operator-path-contract-v75.json');assert len(c['source_files'])==5
    for row in c['source_files']:assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256']
    assert c['V41_CP_MRO']==['AscendDSAV41CPImpl','_ReplicatedCacheImpl','AscendDSAV41Impl']
    assert c['legacy_CP_forward_is_not_in_V41_query_chain']and c['both_V41_preprocess_branches_have_no_post_QB_head_RMS']and not c['post_QB_head_RMS_mandatory_in_frozen_V41_case']and not c['loaded_native_ELF_attested']
    b=read(O/'V41-QB-native-source-binding-v75.json');assert(b['native_QB_total'],b['direct_native_QB'],b['graph_native_QB'],b['bound_direct_TASKs'])==(4367,1167,3200,1167)
    assert not b['unbound_direct_TASKs']and len(b['records'])==1167 and len(b['graph_records'])==3200
    assert b['direct_contexts']=={'CALLER_TARGET_MODEL_OR_ENGRAM':960,'CALLER_DRAFT_OR_DRAFT_METADATA':207}and b['DSpark_frame_population']=={'with_DSpark_frame':207}
    assert len({(r['database_index'],r['TASK_rowid'])for r in b['records']+b['graph_records']})==4367
    assert b['direct_source_frame_counts']=={'/vllm-workspace/vllm-ascend/vllm_ascend/attention/dsa_v41.py(491): multistream_preprocess':1167,'/vllm-workspace/vllm-ascend/vllm_ascend/attention/dsa_v41.py(359): _preprocess':1167}
    for r in b['records']:
        assert r['native_API_table']=='PYTORCH_API'and r['native_API_name']=='aclnnQuantMatmulWeightNz'
        assert r['owning_CPU_quant_API_name']=='npu::npu_quant_matmul'and r['V41_query_path_proved']and not r['legacy_query_forward_in_chain']and not r['source_program_post_QB_head_RMS_required']
    oldgraph=read(O/'QB-to-query-RoPE-native-visibility-v74.json')['graph_sequences']
    assert {(r['database_index'],r['TASK_rowid'])for r in b['graph_records']}=={(r['database_index'],r['Q_B_TASK_rowid'])for r in oldgraph}
    work=read(O/'V41-QB-RoPE-work-and-bound-models-v75.json');assert len(work['models'])==6 and work['active_V41_head_norm_512_work']==0
    for r in work['models']:
        m=r['M'];rot=m*64*64
        assert r['QB_INT8_direct_dense_ops']==2*m*1280*32768 and r['INT8_weight_bytes']==41943040
        assert r['separate_QB_RoPE_reference_bytes']-r['joint_QB_RoPE_reference_bytes']==r['tail_interface_reference_budget_bytes']==4*rot
        assert r['head_RMS512_work_in_active_V41_source']==0 and r['actual_HBM_or_request_gain']is None
        assert r['joint_two_resource_reference_us']==max(r['Cube_INT8_752TOP_reference_us'],r['joint_HBM_1p6TB_reference_us'])
    m800=next(r for r in work['models']if r['M']==800);assert m800['tail_interface_reference_budget_bytes']==13107200 and m800['joint_two_resource_reference_us']==m800['separate_two_resource_reference_us']
    assert len(work['producer_tile_partner_checks'])==6 and all(r['split_rotary_pairs']==0 and r['rotary_pairs']==2048 for r in work['producer_tile_partner_checks'])
    assert work['shared_resource_model']['status']=='CONDITIONAL_CEILING'and work['shared_resource_model']['actual_request_numeric_ceiling']is None
    comp=read(O/'V41-QB-reference-to-native-comparisons-v75.json');assert comp['regimes']==8 and comp['native_QB_executions']==4367
    assert sum(r['executions']for r in comp['comparisons'])==4367 and all(r['ratio_is_not_MFU_or_measured_physical_efficiency']for r in comp['comparisons'])
    frontier=read(O/'V41-QB-query-RoPE-frontier-v75.json');assert frontier['graph_sequences']==frontier['runtime_QB_carriers_bound']==len(frontier['records'])==3200
    assert not frontier['runtime_QB_carriers_unbound']and frontier['middle_native_TASK_types']=={'EVENT WAIT':3200}and frontier['fixed_native_rope_fits_each_matched_carrier_gap']
    assert frontier['conservative_carrier_gap_ns']['median']==11480 and frontier['fixed_duration_ideal_shift_budget_ns']['median']==4040
    for r in frontier['records']:
        assert r['runtime_QB_carrier_proved']and r['carrier_and_ComputeInfo_are_not_independent_work']
        assert r['fixed_duration_ideal_shift_budget_ns']==min(r['conservative_carrier_end_to_RoPE_start_ns'],r['query_RoPE_duration_ns'])
        assert r['actual_critical_path_gain_ns']is None
    correction=read(O/'V41-head-RMS-scope-correction-v75.json');assert len(correction['relations'])==3 and len(correction['groups'])==1
    old=read(O/'candidate-relation-clauses-v74.json');new=read(O/'candidate-relation-clauses-v75.json');assert len(new['clauses'])==44 and len(new['higher_order_groups'])==9
    changes={r['id']for r in correction['relations']}
    for a,z in zip(old['clauses'],new['clauses'][:43]):
        assert a['id']==z['id']
        if a['id']not in changes:assert a==z
        else:assert z['supersedes_current_case_head_RMS_claim_v74']
    assert new['clauses'][-1]['id']=='REL75-01'and(new['clauses'][-1]['a'],new['clauses'][-1]['b'])==('F07','F08')
    graph=read(O/'candidate-relation-coverage-v75.json');nodes={r['id']for r in graph['nodes']}
    assert len(nodes)==35 and len(graph['pairs'])==595 and{(r['a'],r['b'])for r in graph['pairs']}==set(combinations(sorted(nodes),2))
    assert graph['scoped_conditional_pairs']==44 and graph['remaining_pair_relations']==551 and graph['higher_order_groups']==9 and not graph['untouched_candidate_nodes']
    assert all(not r['whole_pair_all_variants_closed']for r in graph['pairs'])
    before={r['id']:r for r in read(O/'module-phase-branch-matrix-v74.json')['modules']};after={r['id']:r for r in read(O/'module-phase-branch-matrix-v75.json')['modules']}
    for mid in before:
        if mid!='M07':assert before[mid]==after[mid]
    assert len(after['M07']['branches'])==12 and [r['id']for r in before['M07']['branches']]==[r['id']for r in after['M07']['branches']]
    legacy=next(r for r in after['M07']['branches']if r['id']=='M07-OP-12');assert legacy['applicable_to_fixed_case']is False and legacy['state']=='EXCLUDED_FROM_FROZEN_V41_QUERY_PATH;LEGACY_REFERENCE_ONLY'
    assert read(Path(after['M07']['legacy_branch_history']))['module']==before['M07']
    cand={r['id']:r for r in read(O/'candidate-profile-ledger-v75.json')['candidates']}
    assert cand['F07']['profile_review_state']=='V41_QB_ROPE_PATH_BOUND;LEGACY_HEAD_RMS_OUTSIDE_FROZEN_CASE'
    cov=read(O/'operator-analysis-coverage-ledger-v75.json');prev=read(O/'operator-analysis-coverage-ledger-v74.json');assert cov['regimes']==prev['regimes']
    scoped=[r for r in cov['regimes']if r['state'].startswith('SCOPED_')];pending=[r for r in cov['regimes']if r['state'].startswith('REGIME_')]
    assert(len(scoped),sum(r['executions']for r in scoped))==(2073,285598)and(len(pending),sum(r['executions']for r in pending))==(9,1352)
    assert cov['unchanged_global_uncertainty']=={'ambiguous_module_assignment_executions':3136,'missing_PMU':849,'phase_unbound_graphs':16}
    assert cov['new_pending_regimes_reconciled']==cov['new_pending_executions_reconciled']==0
    for path in[N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json']:assert read(path)['latest_profile_first_review']==cp
    assert read(TASK_DIR/'index.json')['status']=='ACTIVE'and read(L/'state.json')['status']=='EVALUATING'
    runs=read(O/'task-run-registration-v75.json')['runs'];assert len(runs)==6 and Counter(r['status']for r in runs)=={'pass':3,'error':3}
    print(dict(manifest_artifacts_verified=len(mf),candidate_masters_verified=35,direct_QB_V41_bound=1167,target_draft=[960,207],graph_role_only=3200,active_headRMS_work=0,runtime_carriers_bound=3200,relations=44,central_reconciliation_pending=551,higher_order_groups=9,M07_current_families=11,M07_excluded_legacy_reference=1,other25modules_identical=True,profile_reviewed=[2073,285598],profile_pending=[9,1352],whole_modules_closed=0,status='PROGRESS',overall_complete=False))


if __name__=='__main__':main()
