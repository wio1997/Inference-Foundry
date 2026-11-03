"""Verify finite graph migration and conditional mixed-projection math,not native legality."""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,math,sys,zipfile
from finalize_origin_coverage_v9 import N,O,R,W,T as TASK_DIR,L,read,write
from mirror_saved_profile_core_v53 import CACHE
from analyze_state_engram_v62 import dimensions
from reconcile_operator_graph_coverage_v88 import connected_sets
from model_mixed_SWA_C2_v88 import f32,bf16

def receipt(item):
    b=Path(item['path']).read_bytes();assert hashlib.sha256(b).hexdigest()==item['sha256'],item['path']
    if 'bytes'in item:assert len(b)==item['bytes']

def main(analysis_only=False):
    audit=read(O/'operator-graph-coverage-audit-v88.json')
    for r in audit['input_receipts']:receipt(r)
    assert audit['legacy_counts']==dict(nodes=623,edges=1029)
    assert all(r['legacy']!=r['current']for r in audit['legacy_graph_hash_mismatches'].values())
    g1=read(O/'operator-graph-G1-v88.json');g5=read(O/'operator-graph-G5-v88.json')
    assert(len(g1['nodes']),len(g1['edges']),len(g5['templates']))==(338,690,12)
    assert sum(len(t['nodes'])for t in g5['templates'])==294 and sum(len(t['edges'])for t in g5['templates'])==360
    assert not g1['complete']and not g5['whole_model_complete']
    t=next(t for t in g5['templates']if t['id']=='DSPARK_BOUNDARIES');pairs={(e['producer'],e['consumer'])for e in t['edges']}
    assert ('sp_gather_slice','last_hidden_dynamic_input')in pairs and('last_hidden_dynamic_input','confidence')in pairs
    assert('ordered_markov_inputs','confidence')in pairs and('confidence','dynamic_scheduler')in pairs
    assert('sample_4','ordered_markov_inputs')not in pairs
    assert all((f'sample_{i}','ordered_markov_inputs')in pairs for i in range(4))
    assert('draft_token_output','dynamic_scheduler')not in pairs
    assert not t['source_reconciliation_v88']['new_semantic_discovery']and t['source_reconciliation_v88']['prior_answer_version']==36
    nodes=read(O/'operator-graph-node-coverage-v88.json')['rows'];edges=read(O/'operator-graph-edge-coverage-v88.json')['rows']
    assert len(nodes)==632 and len(edges)==1050
    assert all(r['semantic_coverage']=='NOT_ESTABLISHED_BY_ENUMERATION'for r in nodes+edges)
    for graph,tid,g in [('G1','',g1)]+[('G5',t['id'],t)for t in g5['templates']]:
        ids={n['id']for n in g['nodes']};nr=[r for r in nodes if(r['graph'],r['template'])==(graph,tid)]
        er=[r for r in edges if(r['graph'],r['template'])==(graph,tid)]
        assert{r['node_id']for r in nr}==ids and len(er)==len(g['edges'])
        assert all(r['edge']==g['edges'][r['edge_index']]for r in er)
        counts=next(r for r in audit['enumeration_counts']if(r['graph'],r['template'])==(graph,tid))
        for size in[2,3,4]:
            assert sum(r['connected_set_counts'].get(str(size),0)for r in nr)==size*counts['connected_subsets'][str(size)]
        assert all(e['producer']in ids and e['consumer']in ids for e in g['edges'])
    # Four-node star:all six pairs? only3connected edges;all3parent+two-child triples andone4set.
    star=connected_sets({'x','a','b','c'},[dict(producer='x',consumer=c)for c in ['a','b','c']])
    assert{n:len(s)for n,s in star.items()}=={2:3,3:3,4:1}
    assert('a','b','x')in star[3]
    bindings=read(O/'operator-graph-candidate-bindings-v88.json');assert len(bindings['candidates'])==35 and len(bindings['structural_pair_hints'])==38
    assert all(not c['all_variants_mapped']for c in bindings['candidates'])
    assert next(c for c in bindings['candidates']if c['id']=='F31')['prior_v36_source_supplement']['scope'].startswith('Only')
    source=read(O/'mixed-SWA-C2-source-contract-v88.json');assert len(source['source_files'])==5
    receipt(source['original_archive'])
    with zipfile.ZipFile(source['original_archive']['path'])as z:
        for item in source['source_files']:
            receipt(item);data=Path(item['path']).read_bytes();assert data==z.read(item['archive_member']);lines=data.decode().splitlines()
            for clause in item['clauses']:assert lines[clause['first']-1:clause['last']]==clause['source']
    for r in source['prior_input_receipts']:receipt(r)
    for r in source['saved_primary_sources']:
        if r.get('status')=='READ':
            receipt(dict(path=r['snapshot_path'],sha256=r['snapshot_sha256'],bytes=r['snapshot_bytes']))
    profile=read(O/'mixed-SWA-C2-native-frontiers-v88.json')
    for r in profile['input_receipts']:receipt(r)
    assert(profile['count'],profile['root_bound'],profile['phase_unbound'],profile['distinct_graphs'])==(240,192,48,80)
    bydb=defaultdict(list)
    for row in profile['frontiers']:bydb[row['database_index']].append(row)
    old={(r['database_index'],r['graph_ordinal'],r['block_ordinal']):r for r in read(O/'state-engram-graph-operator-chains-v62.json')['C2_chains']}
    seen=set()
    for db,rows in bydb.items():
        raw=read(CACHE/f'analysis/native-execution-identity-v14/db-{db:02d}.json');tasks={r['task_record_rowid']:r for r in raw['TASK_records']};defs={d['globalTaskId']:d for d in raw['COMPUTE_definitions']}
        for row in rows:
            key=(db,row['graph_ordinal'],row['layer']);assert key not in seen;seen.add(key)
            assert row['root_bound']==old[key]['root_bound']and row['C2_pool_TASK']==old[key]['pool']['TASK_rowid']
            parts=[row['SWA_BF16']]+row['C2_FLOAT_projections']+[row['C2_input_cast']]
            assert len({p['TASK_rowid']for p in parts})==4
            for p in parts:
                d=defs[tasks[p['TASK_rowid']]['globalTaskId']]
                assert(p['globalTaskId'],p['operator'],p['input_shapes'],p['output_shapes'],p['input_types'],p['output_types'])==(d['globalTaskId'],d['opType_text'],d['inputShapes_text'],d['outputShapes_text'],d['inputDataTypes_text'],d['outputDataTypes_text'])
            assert row['SWA_BF16']['input_types']=='DT_BF16;DT_BF16'
            assert all(p['input_types']=='FLOAT;FLOAT'and p['output_types']=='FLOAT'for p in row['C2_FLOAT_projections'])
            assert dimensions(row['SWA_BF16']['input_shapes'])==[[24,5120],[512,5120]]
            assert not row['actual_same_storage_address_or_loaded_FLOAT_HF32_mode_or_weight_values_attested']
    assert seen==old.keys()
    cost=read(O/'mixed-SWA-C2-cost-models-v88.json');assert len(cost['models'])==5 and len(cost['weight_ingress_cuts'])==4
    assert cost['theory_status']=='CONDITIONAL_CEILING'and not cost['universal_algorithm_lower_bound']
    for r in cost['models']:
        n,H,D=r['N'],r['H'],r['D'];assert H==5120 and D==512
        assert r['source_input_plane_GM_reference_bytes']==16*n*H
        assert r['FLOAT_pair_joined_plus_SWA_input_plane_GM_reference_bytes']==12*n*H
        assert r['already_F17_FLOAT_pair_join_credit_bytes']==4*n*H
        assert r['original_SWA_BF16_dense_ops']==2*n*H*D and r['C2_two_FLOAT_dense_ops']==4*n*H*D
        assert r['retained_SWA_BF16_Cube_to_Vector_bridge_reference_bytes']==4*n*D
        assert r['retained_two_C2_FLOAT_Cube_to_Vector_bridge_reference_bytes']==16*n*D
        assert r['one_all_FLOAT_output_plus_UB_BF16round_Norm_extra_vs_segmented_output_reference_bytes']==4*n*D
        assert r['one_all_FLOAT_output_then_separate_BF16_cast_plane_extra_vs_segmented_output_reference_bytes']==8*n*D
        assert r['actual_HBM_or_latency_gain']is None
    for cut in cost['weight_ingress_cuts']:
        w=cost['weights']if cut['representation']=='original_typed_planes'else cost['conditional_reencoded_weights']
        q=cut['C2_source_windows']*sum(w.values());assert q==cut['distinct_uncompressed_weight_bytes']
        for prefix,grant,rate in [('fixed_B','fixed_B_initial_L0B_capacity_grant_bytes','fixed_B_L1_to_L0B_max_bytes_per_aggregate_cycle'),
            ('flexible_orientation_relaxed','flexible_orientation_initial_L0A_L0B_grant_bytes','flexible_orientation_relaxed_max_bytes_per_aggregate_cycle')]:
            cycles=cut[prefix+'_min_aggregate_elapsed_cycles'];needed=max(0,q-cut[grant]);assert(cycles-1)*cut[rate]<needed<=cycles*cut[rate]
        assert cut['actual_wallclock_lower_bound_us']is None
    orig=[r for r in cost['weight_ingress_cuts']if r['representation']=='original_typed_planes']
    assert [r['fixed_B_min_aggregate_elapsed_cycles']for r in orig]==[8022,25088]
    plan=cost['capacity_specific_plan'];assert plan['full_dual_input_L1_bytes']>plan['L1_per_AIC']and plan['full_three_FP32_accumulator_bytes']>plan['L0C_per_AIC']
    comm=read(O/'mixed-SWA-C2-project-before-gather-v88.json')
    assert(comm['old_full_hidden_BF16_per_rank_ring_send_reference_bytes'],comm['projected_joint_typed_send_reference_bytes'])==(215040,107520)
    assert comm['projected_SWA_BF16_send_reference_bytes']+comm['projected_two_C2_FLOAT_send_reference_bytes']==107520 and comm['joint_send_ratio']==.5
    witness=read(O/'mixed-SWA-C2-precision-witnesses-v88.json');r=witness['weight_reencoding']
    assert bf16(r['FLOAT_weight'])==r['BF16_weight']==1 and f32(r['FLOAT_weight']*r['BF16_input'])==r['original_FLOAT_result']==256.5
    assert r['reencoded_result']==256 and witness['exact_BF16_operands_different_reduction_order']['left_result']==0
    assert len(read(O/'mixed-SWA-C2-variants-v88.json')['variants'])==5
    additions=read(O/'mixed-SWA-C2-composition-clauses-v88.json');assert len(additions['clauses'])==3 and len(additions['higher_order_groups'])==1
    prev=read(O/'candidate-relation-clauses-v87.json');assert len({frozenset((c['a'],c['b']))for c in prev['clauses']+additions['clauses']})==115
    if analysis_only:
        assert not(O/'REVIEW-CHECKPOINT-v88.json').exists()
        write(O/'graph-mixed-analysis-preflight-v88.json',dict(status='PASS',graph_rows=[632,1050],prior_confidence_answer=36,
          mixed_frontiers=[240,192,48],typed_models=5,variants=5,conditional_weight_cuts=4,new_pairs=3,new_groups=1,
          new_device_execution=False,all_primitive_or_reader_closure=False,overall_complete=False))
        print(dict(preflight='PASS',graph_rows=[632,1050],mixed=[240,192,48],weight_cuts=4,new_pairs=3));return
    manifest=read(O/'checkpoint-manifest-v88.json')
    for item in manifest['artifacts']:receipt(item)
    rel=read(O/'candidate-relation-clauses-v88.json');cov=read(O/'candidate-relation-coverage-v88.json')
    assert len(rel['clauses'])==115 and len(rel['higher_order_groups'])==24
    assert(cov['scoped_conditional_pairs'],cov['remaining_pair_relations'],cov['higher_order_groups'])==(115,480,24)
    for c in prev['clauses']:
        now=next(r for r in rel['clauses']if r['id']==c['id']).copy();now.pop('mixed_projection_refinement_v88',None);assert now==c
    nav=read(O/'candidate-pair-navigation-v88.json')['pairs'];assert len(nav)==595 and sum(bool(p['conditional_clause_ids'])for p in nav)==115
    for row in read(O/'candidate-operator-footprints-v88.json')['candidates']:
        p=O/(row['id']+'-master-before-v88.json')
        if not p.exists():p=R/'candidates'/(row['id']+'.json')
        assert hashlib.sha256(p.read_bytes()).hexdigest()==row['captured_master_sha256']
    before={m['id']:m for m in read(O/'module-phase-branch-matrix-v87.json')['modules']};after={m['id']:m for m in read(O/'module-phase-branch-matrix-v88.json')['modules']}
    changed={'M07','M08','M09','M10','M16','M19','M20','M21','M22','M23','M24'}
    assert len(after)==26 and sum(len(m['branches'])for m in after.values())==351 and sum(m.get('branch_schema')=='SEMANTIC_BRANCHES_V1'for m in after.values())==24
    for mid,m in after.items():
        oldids=[b.get('id')for b in before[mid]['branches']];assert [b.get('id')for b in m['branches']][:len(oldids)]==oldids and not m.get('closed',False)
        if mid not in changed:assert m['branches']==before[mid]['branches']
    assert after['M07']['branches'][-1]['id']=='M07-OP-13'and after['M08']['branches'][-1]['id']=='M08-OP-14'
    cs=read(O/'candidate-profile-ledger-v88.json')['candidates'];assert len(cs)==35 and all(not c['candidate_closed']and not c['legal_fusion_proved']for c in cs)
    assert len(read(O/'framework-candidate-ledger-v88.json')['candidates'])==7
    ops=read(O/'operator-analysis-coverage-ledger-v88.json');oldops=read(O/'operator-analysis-coverage-ledger-v87.json')
    assert ops['regimes']==oldops['regimes']and ops['unchanged_global_uncertainty']==oldops['unchanged_global_uncertainty']
    assert(ops['scoped_conditional_review_regimes'],ops['scoped_conditional_review_executions'])==(2073,285598)
    assert(ops['remaining_regime_reconciliation_count'],ops['remaining_execution_reconciliation_count'])==(9,1352)
    scan=read(R/'fusion-scan.json');assert scan['per_node_edge_coverage_ledger']['summary']==str(O/'operator-graph-coverage-audit-v88.json')
    cp=read(O/'REVIEW-CHECKPOINT-v88.json');assert cp['status']=='PROGRESS'and not cp['overall_complete']and cp['whole_modules_closed']==0
    for path in[N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json']:
        assert read(path)['latest_profile_first_review']['candidate_ledger']==str(O/'candidate-profile-ledger-v88.json')
    assert read(TASK_DIR/'index.json')['status']=='ACTIVE'and read(L/'state.json')['status']=='EVALUATING'
    print(dict(checkpoint=88,status='PASS',manifest_artifacts=len(manifest['artifacts']),graph_rows=[632,1050],mixed_frontiers=[240,192,48],
      relations=[115,24,480],branches=[351,24],all_readers_closed=False,whole_modules_closed=0,overall_complete=False))

if __name__=='__main__':main('--analysis-only'in sys.argv)
