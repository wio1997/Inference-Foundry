"""Verify saved identities, prefix/dependency witnesses and explicit resource domains."""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,zipfile,sys
from finalize_origin_coverage_v9 import N,O,R,W,T as TASK_DIR,L,read,write
from mirror_saved_profile_core_v53 import CACHE
from analyze_state_engram_v62 import dimensions

def receipt(item):
    data=Path(item['path']).read_bytes();assert hashlib.sha256(data).hexdigest()==item['sha256'],item['path']
    if 'bytes'in item:assert len(data)==item['bytes']

def acyclic(edges):
    nodes={v for e in edges for v in e};remaining=set(nodes)
    while remaining:
        ready={v for v in remaining if not any(a in remaining and b==v for a,b in edges)}
        if not ready:return False
        remaining-=ready
    return True

def main(analysis_only=False):
    prior=read(O/'KV-publication-prior-evidence-v87.json')
    for item in prior['input_receipts']:receipt(item)
    assert not prior['new_all_reader_or_loaded_binary_proof']
    source=read(O/'KV-publication-source-contract-v87.json');assert len(source['source_files'])==5
    receipt(source['original_archive'])
    with zipfile.ZipFile(source['original_archive']['path'])as archive:
        for item in source['source_files']:
            receipt(item);path=Path(item['path'])
            if 'archive_member'in item:assert path.read_bytes()==archive.read(item['archive_member'])
            lines=path.read_text().splitlines()
            for clause in item.get('clauses',[]):
                assert 1<=clause['first']<=clause['last']<=len(lines)
                assert lines[clause['first']-1:clause['last']]==clause['source']
            if 'selected_json_fields'in item:
                cfg=read(path)['text_config'];assert item['selected_json_fields']=={k:cfg[k]for k in item['selected_json_fields']}
    assert not source['all_KV_or_external_readers_closed']and not source['current_loaded_native_binary_identity_proved']
    topology=read(O/'KV-publication-shared-reader-topology-v87.json')
    for item in topology['input_receipts']:receipt(item)
    assert(topology['target_layers'],topology['final_layer'],topology['final_compress_ratio'],topology['compressed_KV_source'],topology['index_key_scale_source'],topology['latest_index_query_selection_source'])==(40,39,1,20,20,36)
    cfg=next(s['selected_json_fields']for s in source['source_files']if 'selected_json_fields'in s)
    assert all(cfg['compress_ratios'][i]==1 and max(s for s in cfg['kv_source_layer_ids']if s<=i)==20 for i in topology['source20_longKV_consumer_layers'])
    profile=read(O/'KV-publication-native-scope-v87.json')
    for item in profile['input_receipts']:receipt(item)
    strict=profile['strict_graph_SWA_projection_rows'];amb=profile['existing_layer20_SWA_C1_ambiguous_projection_rows']
    assert len(strict)==3120 and len(amb)==160 and profile['new_regimes_or_executions_promoted']==0
    assert sum(r['root_bound']for r in strict)==2496 and sum(r['root_bound']for r in amb)==128
    bydb=defaultdict(list);groups=defaultdict(set)
    for row in strict+amb:
        bydb[row['database_index']].append(row)
        if row['existing_role']=='SWA_KV_PROJECTION':groups[(row['database_index'],row['graph_ordinal'])].add(row['block_ordinal'])
        else:assert row['block_ordinal']==20 and row['existing_modules']==['M07','M08']
    assert len(groups)==80 and all(v==set(range(40))-{20}for v in groups.values())
    assignments=defaultdict(dict)
    for a in read(CACHE/'analysis/native-PMU-complete-v18/resource-functional-attribution-and-uncertainty-v20-v3.json')['execution_assignments']:assignments[a[0]][a[1]]=a
    for db,rows in bydb.items():
        data=read(CACHE/f'analysis/native-execution-identity-v14/db-{db:02d}.json')
        tasks={t['task_record_rowid']:t for t in data['TASK_records']};defs={d['globalTaskId']:d for d in data['COMPUTE_definitions']}
        for row in rows:
            t=tasks[row['TASK_rowid']];d=defs[t['globalTaskId']];a=assignments[db][row['TASK_rowid']]
            assert(row['globalTaskId'],row['graph_ordinal'],row['block_ordinal'],row['existing_role'],row['existing_modules'])==(t['globalTaskId'],a[8],a[9],a[5],a[3])
            assert(row['input_shapes'],row['output_shapes'],row['input_types'],row['output_types'])==(d['inputShapes_text'],d['outputShapes_text'],d['inputDataTypes_text'],d['outputDataTypes_text'])
            assert dimensions(row['input_shapes'])==[[24,5120],[512,5120]]and dimensions(row['output_shapes'])==[[24,512]]
    oldfront={(g['database_index'],g['graph_ordinal']):g for g in read(O/'terminal-layer-native-frontier-v86.json')['graphs']}
    fronts=profile['terminal_scatter_attention_frontiers'];assert len(fronts)==80 and sum(r['root_bound']for r in fronts)==64
    for row in fronts:
        old=oldfront[(row['database_index'],row['graph_ordinal'])]
        scatter=next(s for s in old['stages']if s['operator']=='ScatterNdUpdateSk');attn=next(s for s in old['stages']if s['operator']=='SparseFlashMla')
        assert(row['cache_scatter_TASK'],row['attention_TASK'])==(scatter['TASK_rowid'],attn['TASK_rowid'])
        assert dimensions(scatter['input_shapes'])[0][-2:]==[128,512]and dimensions(scatter['input_shapes'])[2]==[24,512]
        assert not row['actual_valid_slots_duplicate_order_pointer_alias_or_acceptance_values_attested']
    proof=read(O/'KV-publication-prefix-and-dependency-proof-v87.json')
    assert [(r['q'],r['d'],r['c'],r['conditional_retained_input_prefix_rows'])for r in proof['prefix_cases']]==[(10,0,0,10),(8,5,2,4),(6,5,2,2),(6,5,6,6)]
    prefix_checks=0
    for q in range(1,33):
        for d in range(min(5,q-1)+1):
            for c in range(d+2):
                rejected=d+1-c if d else 0;kept=q-rejected
                assert 0<=kept<=q
                if d:assert kept==(q-d-1)+c
                else:assert kept==q
                prefix_checks+=1
    identity=proof['shifted_identity_witness'];assert identity['retained_input_ids']==identity['current_input_ids'][:identity['c']]
    assert identity['retained_input_ids']!=identity['sampled_output_ids']
    assert proof['stale_current_read_witness']['single_current_key_attention_expected']!=proof['stale_current_read_witness']['omit_scatter_without_overlay_result']
    assert not acyclic([('verify','accept'),('accept','write'),('write','verify')])
    assert acyclic([('KVready','verify'),('verify','accept'),('accept','commit'),('KVready','commit'),('commit','future_read')])
    assert not proof['all_readers_or_native_rewrite_legal_proved']
    variants=read(O/'KV-publication-variants-v87.json');assert len(variants['variants'])==6
    assert variants['variants'][0]['decision']=='EXCLUDED_AS_UNCONDITIONAL_REWRITE'
    assert all(not v['implementation']and not v['native_test']and v['critical_path_gain_us']is None for v in variants['variants'])
    model=read(O/'KV-publication-value-work-models-v87.json');assert len(model['models'])==7
    for row in model['models']:
        r,a,b=row['R_valid_distinct_rows'],row['A_required_retained_rows'],row['BF16_row_bytes'];assert b==1024
        assert row['source_producer_bank_plus_scatter_value_bytes']==3*r*b
        assert row['retained_bank_overlay_plus_commit_value_bytes']==(r+2*a)*b
        assert row['copied_bank_overlay_plus_commit_value_bytes']==(3*r+2*a)*b
        assert row['fused_direct_producer_to_cache_value_bytes']==r*b
        assert row['retained_overlay_credit_vs_source_value_bytes']==2*(r-a)*b
        assert row['retained_overlay_extra_vs_fused_direct_value_bytes']==row['copied_overlay_extra_vs_source_value_bytes']==2*a*b
        assert row['recompute_accepted_KV_BF16_dense_ops']==2*a*5120*512
        assert row['actual_traffic_capacity_or_speedup']is None
    # Independent integer reference comparison,not a native/performance test.
    byte_checks=0
    for r in range(1,129):
        for a in range(r+1):
            source_bytes=r+r+r;retained=r+a+a;copied=r+r+r+a+a;direct=r
            assert direct<=retained<=source_bytes<=copied
            assert retained-direct==copied-source_bytes==a+a
            byte_checks+=1
    memory=read(O/'KV-publication-memory-page-models-v87.json');assert len(memory['row_bank_models'])==4 and len(memory['page_examples'])==5
    for row in memory['row_bank_models']:
        cap,lay=row['N_physical_bank_capacity'],row['retained_layers']
        assert row['unaliased_KV_BF16_bank_reference_bytes']==cap*lay*512*2
        assert row['normalized_hidden_BF16_input_bank_reference_bytes']==10*row['unaliased_KV_BF16_bank_reference_bytes']
        assert row['input_HC4_BF16_bank_reference_bytes']==40*row['unaliased_KV_BF16_bank_reference_bytes']
        assert row['actual_incremental_allocation_or_usable_budget_cost']is None
    for row in memory['page_examples']:
        off=row['start_offset_in_128row_page'];q=row['scheduled_current_rows']
        assert row['touched_128row_pages']==len({(off+i)//128 for i in range(q)})
        assert row['raw_page_payload_bytes']==row['touched_128row_pages']*128*1024
    credit=memory['capacity_credit_example'];assert credit['raw_snapshot_requested_bytes']==credit['forty_layer_8192x512_BF16_row_bank_reference_bytes']==335544320
    assert not credit['actual_free_memory_or_usable_pool_credit_proved']
    additions=read(O/'KV-publication-composition-clauses-v87.json');assert len(additions['clauses'])==7 and len(additions['higher_order_groups'])==2
    previous=read(O/'candidate-relation-clauses-v86.json')
    assert len({frozenset((c['a'],c['b']))for c in previous['clauses']+additions['clauses']})==112
    assert all(not c['all_runtime_gates_proved']and not c['speedup_additive']for c in additions['clauses'])
    if analysis_only:
        assert not(O/'REVIEW-CHECKPOINT-v87.json').exists()
        write(O/'KV-publication-analysis-preflight-v87.json',dict(status='PASS',pinned_files=5,strict_SWA=3120,retained_ambiguities=160,
          terminal_frontiers=[80,64,16],prefix_integer_cases=prefix_checks,byte_integer_cases=byte_checks,variants=6,
          value_models=7,row_bank_models=4,page_examples=5,new_pairs=7,new_groups=2,new_device_execution=False,all_readers_closed=False,overall_complete=False))
        print(dict(preflight='PASS',profile=[3120,160,80,64,16],variants=6,prefix_checks=prefix_checks,byte_checks=byte_checks));return
    manifest=read(O/'checkpoint-manifest-v87.json')
    for item in manifest['artifacts']:receipt(item)
    rel=read(O/'candidate-relation-clauses-v87.json');cov=read(O/'candidate-relation-coverage-v87.json')
    assert len(rel['clauses'])==112 and len(rel['higher_order_groups'])==23
    assert(cov['scoped_conditional_pairs'],cov['remaining_pair_relations'],cov['higher_order_groups'])==(112,483,23)
    for c in previous['clauses']:
        now=next(r for r in rel['clauses']if r['id']==c['id']).copy();now.pop('KV_publication_refinement_v87',None);assert now==c
    nav=read(O/'candidate-pair-navigation-v87.json')['pairs'];assert len(nav)==595 and sum(bool(p['conditional_clause_ids'])for p in nav)==112
    fp=read(O/'candidate-operator-footprints-v87.json')['candidates'];assert len(fp)==35
    for row in fp:
        path=O/(row['id']+'-master-before-v87.json')
        if not path.exists():path=R/'candidates'/(row['id']+'.json')
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['captured_master_sha256']
    before={m['id']:m for m in read(O/'module-phase-branch-matrix-v86.json')['modules']};after={m['id']:m for m in read(O/'module-phase-branch-matrix-v87.json')['modules']}
    changed={'M02','M07','M08','M09','M10','M14','M15','M21','M22','M23','M24'}
    assert len(after)==26 and sum(len(m['branches'])for m in after.values())==349
    assert sum(m.get('branch_schema')=='SEMANTIC_BRANCHES_V1'for m in after.values())==24
    for mid,m in after.items():
        oldids=[b.get('id')for b in before[mid]['branches']];assert [b.get('id')for b in m['branches']][:len(oldids)]==oldids
        assert not m.get('closed',False)
        if mid not in changed:assert m['branches']==before[mid]['branches']
    assert after['M10']['branches'][-1]['id']=='M10-OP-13'and after['M24']['branches'][-1]['id']=='M24-OP-13'
    cs=read(O/'candidate-profile-ledger-v87.json')['candidates'];assert len(cs)==35 and all(not c['legal_fusion_proved']and not c['candidate_closed']for c in cs)
    assert len(read(O/'framework-candidate-ledger-v87.json')['candidates'])==7
    oldops=read(O/'operator-analysis-coverage-ledger-v86.json');ops=read(O/'operator-analysis-coverage-ledger-v87.json')
    assert ops['regimes']==oldops['regimes']and ops['unchanged_global_uncertainty']==oldops['unchanged_global_uncertainty']
    assert(ops['scoped_conditional_review_regimes'],ops['scoped_conditional_review_executions'])==(2073,285598)
    assert(ops['remaining_regime_reconciliation_count'],ops['remaining_execution_reconciliation_count'])==(9,1352)
    cp=read(O/'REVIEW-CHECKPOINT-v87.json');assert cp['status']=='PROGRESS'and not cp['overall_complete']and cp['whole_modules_closed']==0
    for path in[N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json']:
        assert read(path)['latest_profile_first_review']['candidate_ledger']==str(O/'candidate-profile-ledger-v87.json')
    assert read(TASK_DIR/'index.json')['status']=='ACTIVE'and read(L/'state.json')['status']=='EVALUATING'
    print(dict(checkpoint=87,status='PASS',manifest_artifacts=len(manifest['artifacts']),profile=[3120,160,80,64,16],
      publication_variants=6,relations=[112,23,483],branches=[349,24],all_readers_closed=False,whole_modules_closed=0,overall_complete=False))

if __name__=='__main__':main('--analysis-only'in sys.argv)
