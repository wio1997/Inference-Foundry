"""Scoped audit of selector/precision/repartition evidence and protected gaps."""
from pathlib import Path
from itertools import combinations
from collections import Counter
import hashlib
from finalize_origin_coverage_v9 import N,O,R,W,T as TASK_DIR,L,read


def main():
    cp=read(O/'REVIEW-CHECKPOINT-v78.json');assert cp['status']=='PROGRESS'and not cp['overall_complete']and cp['whole_modules_closed']==0
    mf=read(O/'checkpoint-manifest-v78.json')['artifacts']
    for r in mf:
        b=Path(r['path']).read_bytes();assert len(b)==r['bytes']and hashlib.sha256(b).hexdigest()==r['sha256'],r['path']
    masters=read(O/'candidate-record-index-v78.json')['model_candidates'];assert len(masters)==35
    for r in masters:assert hashlib.sha256(Path(r['master_record']).read_bytes()).hexdigest()==r['master_sha256']
    audit=read(O/'all-module-fixed-case-domain-audit-v78.json');assert len(audit['modules'])==26 and audit['input_family_record_count']==335 and audit['input_semantic_modules']==22
    assert sum(r['input_family_record_count']for r in audit['modules'])==335
    assert audit['legacy_chronological_modules']==['M00','M02','M03','M13']and audit['newly_rechecked_operator_guard_modules']==['M13']
    assert not audit['all_model_guard_conditions_newly_proved']and audit['all_existing_records_preserved']
    assert hashlib.sha256((O/audit['source_matrix']).read_bytes()).hexdigest()==audit['source_matrix_sha256']
    assert audit['fixed_case']['attributes']['CANN']=='9.2.0'and audit['fixed_case']['shape']==[8,32768]
    for row in audit['modules']:
        assert(O/row['primary_existing_evidence']).exists()and not row['observed_all_branches']and not row['whole_module_closed']
        assert all(not b['this_audit_is_new_per_branch_semantics_proof']for b in row['existing_families'])
    source=read(O/'shared-layout-source-index-v78.json');assert len(source['source_files'])==3 and not source['new_remote_source_read']
    for r in source['source_files']:assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256']
    g=read(O/'shared-layout-guard-cases-v78.json');assert g['finite_source_case_counts']=={'quantization':56,'parallel':8}
    q=g['quantization_selector_cases'];assert len(q)==len({(r['lora'],r['up_scale_present'],r['down_scale_present'],r['quant_type'])for r in q})==56
    for r in q:
        eligible=not r['lora']and r['up_scale_present']and r['down_scale_present']
        expected='A8_INT_FUSED'if eligible and r['quant_type']in ['W8A8','W4A8']else'A8_MXFP_FUSED'if eligible and r['quant_type']=='W4A8MXFP'else'LINEAR_WRAPPER'
        assert r['path']==expected
    assert Counter(r['path']for r in q)=={'A8_INT_FUSED':2,'A8_MXFP_FUSED':1,'LINEAR_WRAPPER':53}
    assert len(g['parallel_controller_cases'])==8 and all(not r['controller_mode_alone_proves_parameter_layout']for r in g['parallel_controller_cases'])
    assert Counter(r['controller_mode']for r in g['parallel_controller_cases'])=={'TENSOR_PARALLEL':5,'SEQUENCE_PARALLEL_SEDP':1,'SEQUENCE_PARALLEL_ONLY':1,'SHARED_EXPERT_DATA_PARALLEL_ONLY':1}
    assert g['fixed_model']['SP_disables_TP_in_both_linears']and g['fixed_model']['SP_only_weight_sharding_requires_new_construction_and_ownership']
    assert len(g['typed_regimes'])==42 and sum(r['executions']for r in g['typed_regimes'])==g['typed_executions']==22360
    assert len(g['typed_stage_executions'])==5 and set(g['typed_stage_executions'].values())=={4472}
    proof=read(O/'shared-shard-precision-proof-v78.json');w=proof['early_BF16_partial_witness'];assert w['full_sum_then_BF16']==1 and w['BF16_partials_then_BF16_sum']==0
    qw=proof['fullrow_scale_witness'];assert qw['global_quantized_values']==[2]*7+[127]and qw['independently_quantized_shard_values']==[127]*8
    assert proof['conventional_INT32_domain']['full_down_bound']==2304*16384<2**31-1 and proof['conventional_INT32_domain']['full_up_bound']==5120*16384<2**31-1
    assert len(proof['precision_preserving_graph'])==5 and not proof['implementation']and not proof['native_test']
    costs=read(O/'shared-layout-work-communication-models-v78.json');assert len(costs['models'])==5
    for r in costs['models']:
        m,p,h,i=r['local_materialized_rows'],r['P'],r['H'],r['I']
        assert r['replicated_GEMM_INT8_ops_per_rank']==r['sharded_GEMM_INT8_ops_per_rank']==6*m*h*i
        assert r['replicated_INT8_weight_payload_bytes_per_rank']==35389440 and r['sharded_INT8_weight_payload_bytes_per_rank']==4423680
        assert r['additional_precision_graph_quant_AG_MAX_INT32_RS_ring_send_bytes']==(p-1)*m*(5*h+12)
        assert r['additional_naive_BF16_AG_plus_BF16_RS_ring_send_bytes']==(p-1)*m*4*h
        assert r['raw_INT32_RS_input_elements_per_rank']==p*r['raw_INT32_RS_output_elements_per_rank']
        assert r['balanced_model_is_not_actual_unpadded_N_or_all_ranks_layout_proof']
    assert costs['models'][1]['additional_precision_graph_quant_AG_MAX_INT32_RS_ring_send_bytes']==537852
    assert costs['theory']['state']=='SYMBOLIC_CEILING'and costs['theory']['actual_physical_numeric_ceiling']is None and costs['theory']['actual_measured_speedup']is None
    doc=read(O/'shared-collective-API-evidence-v78.json');assert len(doc['primary_sources'])==2
    for r in doc['primary_sources']:
        b=Path(r['snapshot_path']).read_bytes();assert len(b)==r['snapshot_bytes']and hashlib.sha256(b).hexdigest()==r['snapshot_sha256']
        assert not r['establishes_throughput_or_installed_compatibility']
    assert sum(r['calls']for r in doc['saved_actual_collective_type_counts'])==14720 and not doc['proposed_INT32_RS_observed_in_this_profile']
    bound=doc['crosslayer_constraint'];assert bound['forty_layer_calls']==120>bound['limit']==63 and bound['max_layers_without_any_other_calls']==21
    assert bound['not_a_limit_on_model_depth_or_ordinary_host_graph_capture']and not bound['installed_version_applicability_proved']
    before={r['id']:r for r in read(O/'module-phase-branch-matrix-v77.json')['modules']};after={r['id']:r for r in read(O/'module-phase-branch-matrix-v78.json')['modules']}
    inv=read(O/'shared-layout-branch-inventory-v78.json');assert sum(len(r['branches'])for r in after.values())==340
    assert sum(r.get('branch_schema')=='SEMANTIC_BRANCHES_V1'for r in after.values())==23
    assert len(after['M13']['branches'])==15 and len(after['M23']['branches'])==19
    assert after['M23']['branches'][:18]==before['M23']['branches']and after['M23']['branches'][-1]['id']=='M23-OP-19'
    assert set(inv['old_M13_record_to_new_families'])==set(map(str,range(1,12)))
    assert {r['id']for r in after['M13']['branches']}=={f'M13-OP-{i:02d}'for i in range(1,16)}
    for mid in before:
        if mid not in ['M13','M23']:assert before[mid]==after[mid]
        else:assert read(Path(after[mid]['legacy_branch_history']))['module']==before[mid]
    oldrel=read(O/'candidate-relation-clauses-v77.json');rel=read(O/'candidate-relation-clauses-v78.json')
    assert len(rel['clauses'])==45 and len(rel['higher_order_groups'])==10
    for a,b in zip(oldrel['clauses'],rel['clauses']):
        if a['id']!='REL73-09':assert a==b
        else:assert b['evidence'][:len(a['evidence'])]==a['evidence']and not b['refinement_v78']['all_runtime_gates_proved']
    for a,b in zip(oldrel['higher_order_groups'],rel['higher_order_groups']):
        if a['id']!='HG73-MOE':assert a==b
        else:assert not b['shared_layout_refinement_v78']['all_subsets_analyzed']
    cg=read(O/'candidate-relation-coverage-v78.json');nodes={n['id']for n in cg['nodes']}
    assert len(nodes)==35 and len(cg['pairs'])==595 and cg['scoped_conditional_pairs']==45 and cg['remaining_pair_relations']==550
    assert {(r['a'],r['b'])for r in cg['pairs']}==set(combinations(sorted(nodes),2))and all(not r['whole_pair_all_variants_closed']for r in cg['pairs'])
    cov=read(O/'operator-analysis-coverage-ledger-v78.json');assert cov['regimes']==read(O/'operator-analysis-coverage-ledger-v77.json')['regimes']
    reviewed=[r for r in cov['regimes']if r['state'].startswith('SCOPED_')];pending=[r for r in cov['regimes']if r['state'].startswith('REGIME_')]
    assert(len(reviewed),sum(r['executions']for r in reviewed))==(2073,285598)and(len(pending),sum(r['executions']for r in pending))==(9,1352)
    assert cov['unchanged_global_uncertainty']=={'ambiguous_module_assignment_executions':3136,'missing_PMU':849,'phase_unbound_graphs':16}
    for p in[N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json']:assert read(p)['latest_profile_first_review']==cp
    assert read(TASK_DIR/'index.json')['status']=='ACTIVE'and read(L/'state.json')['status']=='EVALUATING'
    runs=read(O/'task-run-registration-v78.json')['runs'];assert len(runs)==3 and all(r['status']=='pass'for r in runs)
    print(dict(status='PROGRESS',manifest_artifacts_verified=len(mf),module_audit=26,prior_records=335,current_records=340,semantic_modules=23,
        new_M13_families=15,new_M23_family='M23-OP-19',selector_cases=56,mode_cases=8,shared_profile=[42,22360],precision_witness=[1,0],
        balanced_GEMM_ratio=1,M3_ring_reference=537852,relations=45,central_pending=550,higher_order_groups=10,
        profile_reviewed=[2073,285598],profile_pending=[9,1352],whole_modules_closed=0,overall_complete=False))


if __name__=='__main__':main()
