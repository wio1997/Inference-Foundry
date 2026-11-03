"""Read-only v66 evidence/persistence verification; no full-goal completion claim."""
from pathlib import Path
from collections import Counter
import hashlib,json
from finalize_origin_coverage_v9 import N,O,R,W,T,L,read

def main():
    cp=read(O/'REVIEW-CHECKPOINT-v66.json')
    assert cp['status']=='PROGRESS' and cp['overall_complete'] is False and cp['whole_modules_closed']==0
    manifest=read(O/'checkpoint-manifest-v66.json')['artifacts']
    for a in manifest:
        p=Path(a['path']);b=p.read_bytes();assert len(b)==a['bytes']and hashlib.sha256(b).hexdigest()==a['sha256'],p
    masters=read(O/'candidate-record-index-v66.json')['model_candidates'];assert len(masters)==35
    for r in masters:assert hashlib.sha256(Path(r['master_record']).read_bytes()).hexdigest()==r['master_sha256']
    pop=read(O/'communication-native-operator-population-v66.json');ops=pop['operations']
    assert len(ops)==14720 and len({(r['database_index'],r['COMM_OP_rowid'])for r in ops})==14720
    assert sum(r['internal_records']for r in ops)==557688
    assert Counter(r['node_namespace']for r in ops)=={'COMPUTE':11528,'NO_COMPUTEINFO_NODE':3192}
    assert len(pop['semantic_role_counts'])==13 and sum(pop['semantic_role_counts'].values())==14720
    assert pop['source_grades']=={'GRAPH_CONTROL_AND_ORDER_SCHEMA_NOT_TENSOR_EDGE':10080,'DIRECT_NATIVE_FULLCHAIN':4639,'REUSED_EXPLICIT_QUEUE_FLOW_PROOF':1}
    assert pop['communication_schedule_definitions']==3192 and pop['noncompute_ops_to_schedule_one_to_one_join_proved'] is False
    assert len(pop['raw_database_files'])==32
    for r in pop['raw_database_files']:assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256']
    p=read(O/'communication-operator-regime-models-v66.json');assert len(p['regimes'])==17
    keys=[tuple(k)for r in p['regimes']for k in r['native_TASK_keys']]
    ckeys=[tuple(k)for r in p['regimes']for k in r['exact_COMM_OP_keys']]
    assert len(keys)==len(set(keys))==11528 and len(ckeys)==len(set(ckeys))==11528
    assert set(ckeys)=={(r['database_index'],r['COMM_OP_rowid'])for r in ops if r['node_namespace']=='COMPUTE'}
    cost=read(O/'communication-work-payload-models-v66.json')
    assert len(cost['count_group_regimes'])==61 and cost['type_dtype_rank_count_regime_count']==42
    assert cost['semantic_role_count_group_regimes']==76
    modelkeys=[tuple(k)for r in cost['count_group_regimes']for k in r['COMM_OP_keys']]
    assert len(modelkeys)==len(set(modelkeys))==14720
    assert set(modelkeys)=={(r['database_index'],r['COMM_OP_rowid'])for r in ops}
    for r in cost['count_group_regimes']:
        m=r['model'];q=r['count']*{'BFP16':2,'FP32':4}[r['dtype']];p8=r['P'];assert p8==8
        if r['operator']=='hcom_allGather_':assert m['input_bytes']==q and m['output_bytes']==p8*q and m['declared_ring_send_bytes']==7*q
        if r['operator']=='hcom_reduceScatter_':assert m['input_bytes']==p8*q and m['output_bytes']==q and m['declared_ring_send_bytes']==7*q
        if r['operator']=='hcom_alltoall_':assert m['input_bytes']==m['output_bytes']==p8*q and m['declared_pairwise_send_bytes']==7*q
        if r['operator']=='hcom_allReduce_':assert m['input_bytes']==m['output_bytes']==q and m['declared_ring_send_bytes']==1.75*q
    a=cost['operator_repartition_models'][0]
    assert (a['M_projection'],a['A_CP_uniform_exchange_rows'])==(24,24)
    assert a['baseline_restore_AlltoAll_send_bytes_per_rank']==172032 and a['pre_exchange_WA_rank1024_AlltoAll_send_bytes_per_rank']==43008
    assert a['pre_exchange_WA_extra_bytes_per_rank']==58720256 and a['query_owner_extra_weight_bytes_per_rank']==132120576
    assert a['pre_exchange_output_payload_ratio']==2/3
    for r in cost['HC_premix_mixed_dtype_gather_batch_models']:
        q=r['HC_local_bytes']+r['premix_local_bytes'];assert r['pack_plus_unpack_additional_value_bytes']==18*q
    assert len(read(O/'communication-fusion-variants-v66.json')['variants'])==20
    for r in read(O/'communication-operator-contract-v66.json')['source_files']:
        assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256']
    contract=read(O/'communication-operator-contract-v66.json')
    assert hashlib.sha256(Path(contract['config_path']).read_bytes()).hexdigest()==contract['config_sha256']
    cov=read(O/'operator-analysis-coverage-ledger-v66.json');assert len(cov['regimes'])==2082
    reviewed=[r for r in cov['regimes']if r['state'].startswith('SCOPED_')];pending=[r for r in cov['regimes']if r['state'].startswith('REGIME_')]
    assert len(reviewed)==1015 and sum(r['executions']for r in reviewed)==201431
    assert len(pending)==1067 and sum(r['executions']for r in pending)==85519
    assert len(reviewed)+len(pending)==2082 and sum(r['executions']for r in cov['regimes'])==286950
    assert cov['geometry_only_other_role_regimes']==0 and cov['explicit_profile_insufficient_executions']==960
    assert cov['unchanged_global_uncertainty']=={'ambiguous_module_assignment_executions':3136,'missing_PMU':849,'phase_unbound_graphs':16}
    old={r['id']:r for r in read(O/'module-phase-branch-matrix-v65.json')['modules']}
    new={r['id']:r for r in read(O/'module-phase-branch-matrix-v66.json')['modules']}
    inv=read(O/'M23-communication-and-cross-operator-branch-inventory-v66.json')['modules']
    for mid,rows in inv.items():
        assert new[mid]['branches']==rows and new[mid]['branch_schema']=='SEMANTIC_BRANCHES_V1'
        assert read(Path(new[mid]['legacy_branch_history']))['module']==old[mid]
    for mid,n in [('M11',2),('M15',1),('M18',1)]:
        assert new[mid]['branches'][:-n]==old[mid]['branches'] and len(new[mid]['branches'])==len(old[mid]['branches'])+n
    assert len(new['M23']['branches'])==18 and len(old['M23']['branches'])==29
    assert new['M07']['branches']==old['M07']['branches']
    assert len(new['M07']['review_history'])==len(old['M07']['review_history'])+1
    protected=read(O/'communication-protected-module-state-v66.json')['modules']
    assert len(protected)==21 and all(new[mid]==before for mid,before in protected.items())
    assert read(O/'framework-candidate-ledger-v65.json')['candidates']==read(O/'framework-candidate-ledger-v66.json')['candidates']
    for q in [N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json']:
        d=read(q);assert d['latest_profile_first_review']==cp and d['overall_complete']is False
    for q in [L/'state.json',T/'index.json']:
        d=read(q);assert d['status']in ['EVALUATING','ACTIVE']and d['next_action']==cp['next_action']
    runs=read(O/'task-run-registration-v66.json')['runs'];assert len(runs)==5 and sum(r['status']=='error'for r in runs)==1
    print(json.dumps(dict(manifest_artifacts_verified=len(manifest),master_hashes_verified=35,raw_database_hashes_verified=32,
        COMM_operators=14720,exact_ComputeInfo_join=11528,nonCompute_COMM=3192,group_count_models=61,typed_count_combinations=42,
        semantic_group_count_combinations=76,operator_choices=20,M23_semantic_families=18,new_cross_module_branches=4,
        protected_other_modules=21,M07_existing_branches_unchanged=11,reviewed=[1015,201431],pending=[1067,85519],
        checkpoint_status='PROGRESS',whole_modules_closed=0,overall_complete=False),ensure_ascii=False))

if __name__=='__main__':main()
