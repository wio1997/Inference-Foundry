"""Verify exact saved identities and conditional mathematical domains, not device legality."""
from pathlib import Path
from collections import Counter,defaultdict
from itertools import product
import ast,hashlib,sys,zipfile
from finalize_origin_coverage_v9 import N,O,R,W,T as TASK_DIR,L,read,write
from mirror_saved_profile_core_v53 import CACHE
from analyze_state_engram_v62 import dimensions

def receipt(item):
    data=Path(item['path']).read_bytes()
    assert hashlib.sha256(data).hexdigest()==item['sha256'],item['path']
    if 'bytes'in item:assert len(data)==item['bytes']

def main(analysis_only=False):
    prior=read(O/'terminal-layer-prior-evidence-v86.json')
    for item in prior['input_receipts']:receipt(item)
    assert not prior['new_full_reader_or_native_fusion_closure']and len(prior['already_established'])==4
    source=read(O/'terminal-layer-operator-source-v86.json');assert len(source['source_files'])==3
    receipt(source['original_archive'])
    with zipfile.ZipFile(source['original_archive']['path'])as archive:
        for item in source['source_files']:
            receipt(item);data=Path(item['path']).read_bytes();assert data==archive.read(item['archive_member'])
            lines=data.decode().splitlines()
            for clause in item['clauses']:assert lines[clause['first']-1:clause['last']]==clause['source']
    oldhashes=read(O/'M15-normalized-content-consumer-and-precision-contract-v31.json')['source_files']
    for item in source['source_files']:
        key='/vllm-workspace/'+item['archive_member'];assert oldhashes[key]==item['sha256']
    assert source['normalized_hidden_identifier_load_inventory']['_bookkeeping_sync']==[3132]
    assert not source['raw_and_normalized_all_readers_closed']and not source['new_remote_source_read']
    native=read(O/'terminal-layer-native-frontier-v86.json')
    for item in native['input_receipts']:receipt(item)
    assert(native['graphs_count'],native['root_bound'],native['phase_unbound'],native['reused_ComputeInfo_executions'])==(80,64,16,2800)
    assert native['identical_operator_type_multiplicity_schema_count']==1
    assert native['per_case_shapes_and_dtypes_kept_without_cross_case_equality_claim']
    oldfront={(r['database_index'],r['graph_ordinal']):r for r in read(O/'DSpark-tail-native-boundaries-v85.json')['graph_boundaries']}
    bydb=defaultdict(list);seen=set()
    for graph in native['graphs']:
        key=(graph['database_index'],graph['graph_ordinal']);assert key not in seen;seen.add(key)
        old=oldfront[key];assert graph['root_bound']==old['root_bound']
        stages=graph['stages'];assert len(stages)==35 and Counter(s['operator']for s in stages)==native['schema']
        assert stages[-1]['TASK_rowid']==old['last_FFN_HcPost_TASK']
        assert not graph['actual_selected_row_values_or_owner_plan_attested']and not graph['all_state_writers_or_external_observers_closed']
        bydb[graph['database_index']].append(graph)
    assert seen==oldfront.keys()and sum(g['root_bound']for g in native['graphs'])==64
    assignments=defaultdict(dict)
    for a in read(CACHE/'analysis/native-PMU-complete-v18/resource-functional-attribution-and-uncertainty-v20-v3.json')['execution_assignments']:
        assignments[a[0]][a[1]]=a
    for db,graphs in bydb.items():
        raw=read(CACHE/f'analysis/native-execution-identity-v14/db-{db:02d}.json')
        tasks={t['task_record_rowid']:t for t in raw['TASK_records']};defs={d['globalTaskId']:d for d in raw['COMPUTE_definitions']}
        for graph in graphs:
            ids={a[1]for a in assignments[db].values()if a[2]=='GRAPH'and a[8]==graph['graph_ordinal']and a[9]==39}
            assert ids=={s['TASK_rowid']for s in graph['stages']}
            for s in graph['stages']:
                t=tasks[s['TASK_rowid']];d=defs[t['globalTaskId']]
                assert s['globalTaskId']==t['globalTaskId']and(s['startNs'],s['endNs'])==(t['startNs'],t['endNs'])
                assert(s['operator'],s['input_shapes'],s['output_shapes'],s['input_types'],s['output_types'])==(d['opType_text'],d['inputShapes_text'],d['outputShapes_text'],d['inputDataTypes_text'],d['outputDataTypes_text'])
            stages=graph['stages']
            for target,shape in [('MatMulV2',[[3,5120],[1280,5120]]),('MatMulV2',[[24,5120],[512,5120]]),
                ('MatMulV2',[[24,1024],[5120,1024]]),('MatMulV3',[[3,5120],[384,5120]])]:
                row=next(s for s in stages if s['operator']==target and dimensions(s['input_shapes'])==shape)
                assert row['input_types']==('FLOAT;FLOAT'if target=='MatMulV3'else 'DT_BF16;DT_BF16')
            wa=next(s for s in stages if s['operator']=='TransposeBatchMatMul');assert dimensions(wa['input_shapes'])==[[24,1,4096],[1,4096,1024]]
            qb=next(s for s in stages if s['operator']=='QuantBatchMatmulV3'and dimensions(s['input_shapes'])[0]==[3,1280])
            assert dimensions(qb['input_shapes'])[1]==[1024,80,16,32]and qb['input_types'].startswith('INT8;INT8;')and qb['output_types']=='DT_BF16'
            gateup=next(s for s in stages if s['operator']=='QuantBatchMatmulV3'and dimensions(s['input_shapes'])[0]==[3,5120])
            assert dimensions(gateup['input_shapes'])[1]==[144,320,16,32]and gateup['output_types']=='INT32'
            mc2=next(s for s in stages if s['operator']=='DispatchFFNCombine')
            assert dimensions(mc2['input_shapes'])[1:4]==[[24,5120,4608],[24,2304,5120],[3,6]]
    demand=read(O/'terminal-layer-demand-cut-v86.json');assert len(demand['stages'])==12
    assert not demand['previous_layer_pruning_or_future_cache_write_deletion_authorized']and not demand['native_test']
    model=read(O/'terminal-layer-owner-and-work-models-v86.json');assert len(model['models'])==5
    assert sum(m['recorded_root_count']for m in model['models'])==88
    oldroots=read(O/'M16-current-head-ownership-and-step-census-v32.json')['root_and_current_step_audit']['target_root_regimes']
    assert native['target_head_row_regimes']==oldroots
    for m,old in zip(model['models'],oldroots):
        n,u,p,hid=m['N'],m['U'],m['P'],5120;cap=n//p
        assert(n,m['head_K'],m['recorded_root_count'])==(old['normalized_rows'],old['head_rows'],old['roots'])and u==m['head_K']
        lo=(u+p-1)//p;hi=min(u,cap)
        assert(m['minimum_feasible_owner_max'],m['maximum_feasible_owner_max'])==(lo,hi)
        assert m['equal_count_published_row_slots_min']==p*lo and m['equal_count_published_row_slots_max']==p*hi
        assert m['per_rank_ring_send_value_bytes_min']==2*(p-1)*lo*hid
        assert m['per_rank_ring_send_value_bytes_max']==2*(p-1)*hi*hid
        assert m['ideal_variable_count_group_total_send_value_bytes']==2*(p-1)*u*hid
        assert m['padded_equal_count_group_total_send_value_bytes_max']==2*p*(p-1)*hi*hid
        assert m['group_original_collapse_row_evaluations']==p*n
        assert m['group_owner_selected_collapse_row_evaluations']==u
        assert m['group_receiver_selected_Norm_row_evaluations']==p*u and m['group_owner_selected_Norm_row_evaluations']==u
        assert m['original_head_GEMM_rows_per_rank_kept']==m['head_K']
        assert m['source_group_dense_BF16_KV_projection_ops_kept']==2*p*n*hid*512
        assert m['selected_source_group_Q_A_BF16_ops']==2*u*hid*1280
        assert m['selected_source_group_Q_B_INT8_ops']==2*u*1280*32768
        assert m['selected_TP_group_WO_A_BF16_ops']==2*p*u*4096*1024
        assert m['selected_TP_group_WO_B_BF16_ops']==2*p*u*1024*hid
        assert m['selected_source_group_router_FLOAT_ops']==2*u*hid*384
        assert m['selected_source_group_shared_INT8_dense_ops']==2*u*hid*(4608+2304)
        assert m['selected_source_group_routed_INT8_dense_ops_reference']==2*6*u*hid*(4608+2304)
        assert m['actual_MC2_receiver_rows_or_active_experts_or_startup_or_gain']is None
    # Independent exhaustive finite census of all4^8 legal owner count vectors.
    extrema=defaultdict(list)
    for owners in product(range(4),repeat=8):extrema[sum(owners)].append(max(owners))
    for u,values in extrema.items():assert(min(values),max(values))==((u+7)//8,min(u,3))
    concrete=next(m for m in model['models']if(m['N'],m['U'])==(24,12))
    assert(concrete['equal_count_published_row_slots_min'],concrete['equal_count_published_row_slots_max'])==(16,24)
    weights=read(O/'terminal-layer-weight-and-dependency-proof-v86.json');w=weights['weight_planes']
    assert w['Q_A_BF16_bytes']==13107200 and w['Q_B_INT8_raw_bytes']==41943040
    assert w['WO_A_local_G1_BF16_bytes']==8388608 and w['WO_B_BF16_bytes']==10485760
    assert w['router_FLOAT_bytes']==7864320 and w['shared_gate_up_INT8_raw_bytes']==23592960 and w['shared_down_INT8_raw_bytes']==11796480
    additions=read(O/'terminal-layer-composition-clauses-v86.json');assert len(additions['clauses'])==10 and len(additions['higher_order_groups'])==2
    previous=read(O/'candidate-relation-clauses-v85.json')
    assert len({frozenset((c['a'],c['b']))for c in previous['clauses']+additions['clauses']})==105
    assert all(not c['all_runtime_gates_proved']and not c['speedup_additive']and c['critical_path_gain_us']is None for c in additions['clauses'])
    if analysis_only:
        assert not(O/'REVIEW-CHECKPOINT-v86.json').exists()
        write(O/'terminal-layer-analysis-preflight-v86.json',dict(status='PASS',source_files=3,prior_selector=31,prior_quota=65,
          graphs=[80,64,16],saved_ComputeInfo_executions=2800,root_shape_models=5,root_shapes_total=88,
          independent_exhaustive_owner_vectors_checked=4**8,new_pairs=10,new_groups=2,new_device_execution=False,all_readers_closed=False,overall_complete=False))
        print(dict(preflight='PASS',graphs=[80,64,16,2800],models=5,owner_vectors=4**8,new_pairs=10,all_readers_closed=False));return
    manifest=read(O/'checkpoint-manifest-v86.json')
    for item in manifest['artifacts']:receipt(item)
    rel=read(O/'candidate-relation-clauses-v86.json');cov=read(O/'candidate-relation-coverage-v86.json')
    assert len(rel['clauses'])==105 and len(rel['higher_order_groups'])==21
    assert(cov['scoped_conditional_pairs'],cov['remaining_pair_relations'],cov['higher_order_groups'])==(105,490,21)
    for c in previous['clauses']:
        now=next(r for r in rel['clauses']if r['id']==c['id']).copy();now.pop('terminal_demand_refinement_v86',None);assert now==c
    nav=read(O/'candidate-pair-navigation-v86.json')['pairs'];assert len(nav)==595 and sum(bool(p['conditional_clause_ids'])for p in nav)==105
    footprints=read(O/'candidate-operator-footprints-v86.json')['candidates'];assert len(footprints)==35
    for row in footprints:
        p=O/(row['id']+'-master-before-v86.json')
        if not p.exists():p=R/'candidates'/(row['id']+'.json')
        assert hashlib.sha256(p.read_bytes()).hexdigest()==row['captured_master_sha256']
    before={m['id']:m for m in read(O/'module-phase-branch-matrix-v85.json')['modules']}
    after={m['id']:m for m in read(O/'module-phase-branch-matrix-v86.json')['modules']}
    changed={'M04','M06','M07','M10','M11','M12','M13','M15','M16','M21','M22','M23','M24'}
    assert len(after)==26 and sum(len(m['branches'])for m in after.values())==347
    assert sum(m.get('branch_schema')=='SEMANTIC_BRANCHES_V1'for m in after.values())==24
    for mid,mod in after.items():
        assert not mod.get('closed',False)
        oldids=[b.get('id')for b in before[mid]['branches']];assert[b.get('id')for b in mod['branches']][:len(oldids)]==oldids
        if mid not in changed:assert mod['branches']==before[mid]['branches']
    assert after['M10']['branches'][-1]['id']=='M10-OP-12'and after['M12']['branches'][-1]['id']=='M12-OP-16'
    candidates=read(O/'candidate-profile-ledger-v86.json')['candidates'];assert len(candidates)==35
    assert all(not c['candidate_closed']and not c['legal_fusion_proved']for c in candidates)
    assert len(read(O/'framework-candidate-ledger-v86.json')['candidates'])==7
    oldops=read(O/'operator-analysis-coverage-ledger-v85.json');ops=read(O/'operator-analysis-coverage-ledger-v86.json')
    assert ops['regimes']==oldops['regimes']and ops['unchanged_global_uncertainty']==oldops['unchanged_global_uncertainty']
    assert(ops['scoped_conditional_review_regimes'],ops['scoped_conditional_review_executions'])==(2073,285598)
    assert(ops['remaining_regime_reconciliation_count'],ops['remaining_execution_reconciliation_count'])==(9,1352)
    cp=read(O/'REVIEW-CHECKPOINT-v86.json');assert cp['status']=='PROGRESS'and not cp['overall_complete']and cp['whole_modules_closed']==0
    for path in[N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json']:
        assert read(path)['latest_profile_first_review']['candidate_ledger']==str(O/'candidate-profile-ledger-v86.json')
    assert read(TASK_DIR/'index.json')['status']=='ACTIVE'and read(L/'state.json')['status']=='EVALUATING'
    print(dict(checkpoint=86,status='PASS',manifest_artifacts=len(manifest['artifacts']),source_files=3,last_layer_graphs=[80,64,16],
      saved_ComputeInfo_executions=2800,relations=[105,21,490],branches=[347,24],all_readers_closed=False,whole_modules_closed=0,overall_complete=False))

if __name__=='__main__':main('--analysis-only'in sys.argv)
