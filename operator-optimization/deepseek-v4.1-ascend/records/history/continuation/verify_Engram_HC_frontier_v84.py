"""Verify saved native identities, precision/storage math and versioned coverage scope."""
from pathlib import Path
from collections import defaultdict,Counter
import hashlib,sys
from finalize_origin_coverage_v9 import N,O,R,W,T as TASK_DIR,L,read,write
from mirror_saved_profile_core_v53 import CACHE
from analyze_state_engram_v62 import dimensions


def receipt(item,path_key='path',size_key='bytes',hash_key='sha256'):
    blob=Path(item[path_key]).read_bytes()
    assert len(blob)==item[size_key]and hashlib.sha256(blob).hexdigest()==item[hash_key],item[path_key]


def main(analysis_only=False):
    source=read(O/'Engram-HC-operator-source-contract-v84.json')
    assert len(source['source_files'])==3 and source['new_source_files_obtained']==0
    for item in source['source_files']:
        receipt(item);lines=Path(item['path']).read_text().splitlines()
        for clause in item['clauses']:assert lines[clause['first']-1:clause['last']]==clause['source']
    assert source['config']['target_layers']==[1,14] and not source['config']['draft_Engram_constructed']
    doc=read(O/'torch-split-primary-evidence-v84.json')
    receipt(doc,'snapshot_path','snapshot_bytes','snapshot_sha256')
    assert '/2.8/'in doc['url']and not doc['actual_NPU_addresses_or_allocator_liveness_attested']
    profile=read(O/'Engram-HC-native-frontiers-v84.json')
    assert(profile['graph_frontier_count'],profile['root_bound_frontiers'],profile['phase_unbound_frontiers'])==(160,128,32)
    assert profile['intermediate_TASK_type_counts']=={'MEMCPY_ASYNC':160,'AI_CORE':160}
    assert len(profile['WKV_observed_regimes'])==5 and sum(r['executions']for r in profile['WKV_observed_regimes'])==208
    assert{r['M']for r in profile['WKV_observed_regimes']}=={3,800,816,834}
    old_bodies={(b['database_index'],b['graph_ordinal'],b['layer']):b for b in read(O/'state-engram-graph-operator-chains-v62.json')['Engram_bodies']}
    by_db=defaultdict(list);seen=set()
    for row in profile['frontiers']:
        key=(row['database_index'],row['graph_ordinal'],row['target_layer']);assert key not in seen;seen.add(key)
        old=old_bodies[key];assert row['root_bound']==old['root_bound']
        assert row['gate_BF16_tail_TASK']==old['stages'][-1]['TASK_rowid']and row['WKV_TASK']==old['WKV']['TASK_rowid']
        assert row['actual_lookup_ready_time_or_hoist_window_ns']is None and not row['MEMCPY_address_length_or_defuse_attested']
        by_db[row['database_index']].append(row)
    assert len(seen)==160 and len({(r['database_index'],r['graph_ordinal'])for r in profile['frontiers']})==80
    for db,rows in by_db.items():
        native=read(CACHE/f'analysis/native-execution-identity-v14/db-{db:02d}.json')
        tasks={t['task_record_rowid']:t for t in native['TASK_records']};defs={d['globalTaskId']:d for d in native['COMPUTE_definitions']}
        for row in rows:
            tail=tasks[row['gate_BF16_tail_TASK']];pre=tasks[row['first_attn_HcPre_TASK']];wkv=tasks[row['WKV_TASK']]
            assert defs[tail['globalTaskId']]['opType_text']=='Cast'and defs[pre['globalTaskId']]['opType_text']=='HcPre'
            assert dimensions(defs[tail['globalTaskId']]['outputShapes_text'])[0]==dimensions(defs[pre['globalTaskId']]['inputShapes_text'])[0]==[3,4,5120]
            assert defs[tasks[row['previous_FFN_HcPost_TASK']]['globalTaskId']]['opType_text']=='HcPost'
            assert defs[tasks[row['input_RMS_TASK']]['globalTaskId']]['opType_text']=='RmsNorm'
            assert row['WKV_duration_ns']==wkv['endNs']-wkv['startNs']
            assert row['tail_to_HcPre_start_gap_ns']==pre['startNs']-tail['endNs']
            between=[t for t in native['TASK_records']if t['globalPid']==tail['globalPid']and t['streamId']==tail['streamId']and tail['endNs']<=t['startNs']<pre['startNs']]
            assert Counter(t['task_type_name']for t in between)==row['intervening_TASK_types']=={'MEMCPY_ASYNC':1,'AI_CORE':1}
            assert not any(t['globalTaskId']in defs for t in between)
    variants=read(O/'Engram-WKV-hoist-and-memo-variants-v84.json');assert len(variants['variants'])==5
    scalar=next(v for v in variants['variants']if v['id']=='F25-KEY-RSTD-EARLY')
    assert any('Ordinary common-gate branch only'in c for c in scalar['conditions'])
    assert variants['actual_hit_rate']is None and variants['lookup_ready_time_or_free_hoist_window']is None
    assert not variants['implementation']and not variants['native_test']
    witness=read(O/'Engram-HC-rounding-witnesses-v84.json')
    assert(witness['gate_before_HC_collapse']['original_HC_Y_BF16'],witness['gate_before_HC_collapse']['skip_gate_BF16_round_HC_Y'])==(5,4)
    assert witness['gate_before_HC_collapse']['original_BF16_gate_outputs']==[258,-255,1,1]
    assert witness['preweight_key_reassociation']['original_left_associated']=='Inf'
    assert witness['preweight_key_reassociation']['preweighted_key_result']==2**100
    model=read(O/'Engram-WKV-storage-and-work-models-v84.json')
    assert len(model['models'])==6 and len(model['memo_reference_models'])==11
    for row in model['models']:
        count=row['M'];assert row['WKV_dense_BF16_ops']==count*314572800 and row['WKV_weight_payload_bytes']==314572800
        assert row['BF16_WKV_result_bank_bytes']==51200*count
        assert row['raw_KV_plus_key_rstd_bytes']==51216*count
        assert row['FP32_key_plus_value_view_retaining_full_KV_bytes']==133136*count
        assert row['FP32_key_plus_independent_value_and_rstd_bytes']==92176*count
        assert row['value_copy_to_retire_raw_KV_reference_read_write_bytes']==20480*count
        assert row['HC_coefficient_FLOAT_cast_GM_bridge_reference_bytes']==163840*count
    for row in model['memo_reference_models']:
        count=row['M'];miss=row['distinct_misses']
        assert row['baseline_cold_value_reference_bytes']==314572800+63488*count
        assert row['declared_missbuf_assembly_cache_reference_bytes']==(314572800 if miss else 0)+12288*miss+102400*miss+102400*count+96*count
        assert row['full_uncompressed_weight_plane_required_in_miss_GEMM']==bool(miss)
        assert row['miss_conventional_ops']==314572800*miss
        assert not row['actual_hit_or_performance_observed']
    for row in model['cache_capacity_reference_models']:
        budget=row['assumed_cache_budget_bytes'];assert row['maximum_full_key_raw_BF16_entries_before_extra_metadata']==budget//51296
        assert row['maximum_entries_with_key_rstd_before_extra_metadata']==budget//51312
    additions=read(O/'HC-Engram-composition-clauses-v84.json');assert len(additions['clauses'])==15 and len(additions['higher_order_groups'])==2
    previous=read(O/'candidate-relation-clauses-v83.json')
    assert len({tuple(sorted([c['a'],c['b']]))for c in previous['clauses']+additions['clauses']})==84
    if analysis_only:
        assert not(O/'REVIEW-CHECKPOINT-v84.json').exists()
        write(O/'Engram-HC-analysis-preflight-v84.json',dict(status='PASS',source_files=3,profile_frontiers=160,
            root_bound=128,phase_unbound=32,WKV_executions=208,variants=5,models=6,memo_cases=11,native_execution=False,overall_complete=False))
        print(dict(pre_checkpoint_analysis='PASS',profile=[160,128,32],WKV=[5,208],variants=5,storage_models=6,memo_cases=11))
        return
    for item in read(O/'checkpoint-manifest-v84.json')['artifacts']:receipt(item)
    relations=read(O/'candidate-relation-clauses-v84.json');coverage=read(O/'candidate-relation-coverage-v84.json')
    assert len(relations['clauses'])==84 and len(relations['higher_order_groups'])==16
    for old in previous['clauses']:
        now=next(c for c in relations['clauses']if c['id']==old['id']).copy();now.pop('pure_WKV_refinement_v84',None);assert now==old
    assert(coverage['scoped_conditional_pairs'],coverage['remaining_pair_relations'],coverage['higher_order_groups'])==(84,511,16)
    navigation=read(O/'candidate-pair-navigation-v84.json');assert len(navigation['pairs'])==595 and sum(bool(p['conditional_clause_ids'])for p in navigation['pairs'])==84
    footprints=read(O/'candidate-operator-footprints-v84.json')['candidates'];assert len(footprints)==35
    for row in footprints:
        old=O/(row['id']+'-master-before-v84.json')
        if not old.exists():old=R/'candidates'/(row['id']+'.json')
        assert hashlib.sha256(old.read_bytes()).hexdigest()==row['captured_master_sha256']
    before={m['id']:m for m in read(O/'module-phase-branch-matrix-v83.json')['modules']}
    after={m['id']:m for m in read(O/'module-phase-branch-matrix-v84.json')['modules']}
    assert len(after)==26 and sum(len(m['branches'])for m in after.values())==344
    assert sum(m.get('branch_schema')=='SEMANTIC_BRANCHES_V1'for m in after.values())==24
    changed={'M06','M12','M13','M14','M22','M23','M24'}
    for mid,mod in after.items():
        oldids=[b.get('id')for b in before[mid]['branches']];newids=[b.get('id')for b in mod['branches']]
        assert newids[:len(oldids)]==oldids and not mod.get('closed',False)
        if mid not in changed:assert mod['branches']==before[mid]['branches']
    assert len(after['M14']['branches'])==20 and len(after['M24']['branches'])==12
    assert after['M14']['branches'][-1]['id']=='M14-OP-20'and after['M24']['branches'][-1]['id']=='M24-OP-12'
    candidates=read(O/'candidate-profile-ledger-v84.json')['candidates'];assert len(candidates)==35
    assert all(not c['candidate_closed']and not c['legal_fusion_proved']for c in candidates)
    assert len(read(O/'framework-candidate-ledger-v84.json')['candidates'])==7
    oldops=read(O/'operator-analysis-coverage-ledger-v83.json');ops=read(O/'operator-analysis-coverage-ledger-v84.json')
    assert ops['regimes']==oldops['regimes']and ops['unchanged_global_uncertainty']==oldops['unchanged_global_uncertainty']
    assert(ops['scoped_conditional_review_regimes'],ops['scoped_conditional_review_executions'])==(2073,285598)
    assert(ops['remaining_regime_reconciliation_count'],ops['remaining_execution_reconciliation_count'])==(9,1352)
    for row in read(O/'Engram-HC-transfer-credit-v84.json')['models']:
        count=row['M'];extent=count*4*5120
        assert row['conditional_direct_store_unique_credit_bytes']==4*extent
        assert row['updated_hidden_destination_write_retained_bytes']==2*extent
        assert row['direct_store_plus_collapse_norm_distinct_transfer_union_bytes']==6*extent+4*count*5120
        assert row['HC_coefficient_FLOAT_store_read_bridge_retained_bytes']==8*extent
    assert len(read(O/'Engram-HC-validation-history-v84.json')['errors'])==2
    assert all(e['resolved']for e in read(O/'Engram-HC-validation-history-v84.json')['errors'])
    checkpoint=read(O/'REVIEW-CHECKPOINT-v84.json');assert checkpoint['status']=='PROGRESS'and not checkpoint['overall_complete']and checkpoint['whole_modules_closed']==0
    for path in[N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json']:
        assert read(path)['latest_profile_first_review']['candidate_ledger']==str(O/'candidate-profile-ledger-v84.json')
    assert read(TASK_DIR/'index.json')['status']=='ACTIVE'and read(L/'state.json')['status']=='EVALUATING'
    print(dict(checkpoint=84,status='PROGRESS',manifest_artifacts_verified=len(read(O/'checkpoint-manifest-v84.json')['artifacts']),
        source_files=3,public_API_pages=1,graph_frontiers=[160,128,32],WKV=[5,208],variants=5,memo_cases=11,
        candidate_pairs=84,higher_order_groups=16,central_pending=511,branch_records=344,semantic_modules=24,
        profile_reviewed=[2073,285598],profile_pending=[9,1352],actual_cache_hits_or_hoist_gain_proved=False,whole_modules_closed=0,overall_complete=False))


if __name__=='__main__':main('--analysis-only'in sys.argv)
