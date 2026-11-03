"""Independent evidence/provenance and mathematical scope audit; no target execution."""
from pathlib import Path
from collections import Counter
import hashlib,sys
from finalize_origin_coverage_v9 import N,O,R,W,T as TASK_DIR,L,read


def main(analysis_only=False):
    primary=read(O/'quant-epilogue-public-source-index-v82.json')
    details=read(O/'quant-epilogue-primary-details-v82.json')
    assert primary['commit']==details['repo_commit']=='a61e016a71f422cb280af0c024975f13629ca100'
    files=primary['files']+details['files'];assert len(files)==5
    for item in files:
        blob=Path(item['local_path']).read_bytes()
        assert len(blob)==item['bytes'] and hashlib.sha256(blob).hexdigest()==item['sha256']
        assert hashlib.sha1(b'blob '+str(len(blob)).encode()+b'\0'+blob).hexdigest()==item['git_blob_sha1']
    for doc in details['primary_documents']:
        blob=Path(doc['snapshot_path']).read_bytes()
        assert len(blob)==doc['snapshot_bytes'] and hashlib.sha256(blob).hexdigest()==doc['snapshot_sha256']
    text=(O/'FixpipeL0CtoGM-primary-text-v82.txt').read_text()
    table=text.split('针对Atlas A3系列产品，Fixpipe接口支持的数据类型组合如下：',1)[1].split('针对Atlas A2',1)[0]
    assert 'bfloat16_t' in table.split('int32_t',1)[0] and 'bfloat16_t' not in table.split('int32_t',1)[1]
    assert 'half' in table.split('int32_t',1)[1]
    atomic=(O/'SetAtomicAdd-primary-text-v82.txt').read_text()
    assert all(s in atomic for s in ['Atlas A3','bfloat16_t','不会自动清零GM数据','DisableDmaAtomic','bf16-saturation-mode'])
    profile=read(O/'quant-vector-owner-profile-v82.json')
    assert (profile['regime_count'],profile['executions'],profile['missing_PMU_records'])==(31,14143,0)
    native_seen=set();execs=Counter();regimes=Counter()
    for row in profile['regimes']:
        records=row['records'];assert len(records)==row['executions']==row['native_PMU_records']
        cohort=row['cohort'];execs[cohort]+=len(records);regimes[cohort]+=1
        for ob in records:
            key=(ob['database_index'],ob['TASK_rowid']);assert key not in native_seen;native_seen.add(key)
            positive=cohort!='shared_GateUp_control'
            assert (ob['aiv_total_cycles']>0)==positive
            assert (ob['aiv_vec_time']>0)==positive
            assert (ob['mixBlockNum']>0)==positive
        if cohort=='shared_GateUp_control': assert row['signature']['outputDataTypes_text']=='INT32'
        else: assert row['signature']['outputDataTypes_text']=='DT_BF16'
    assert execs==dict(shared_Down=4472,Q_B=4367,index_Q=832,shared_GateUp_control=4472)
    assert regimes==dict(shared_Down=8,Q_B=8,index_Q=6,shared_GateUp_control=9)
    source=read(O/'quant-epilogue-source-contract-v82.json')
    assert len(source['source_facts'])==5 and not source['loaded_CANN_9p2_binary_or_tiling_key_attested']
    variants=read(O/'DownAdd-epilogue-variants-v82.json')
    assert len(variants['variants'])==5 and not variants['implementation'] and not variants['native_benchmark']
    assert all(v.get('critical_path_gain_us') is None for v in variants['variants'])
    witness=read(O/'epilogue-rounding-witnesses-v82.json')
    assert (witness['pre_cast_bias_counterexample']['original'],witness['pre_cast_bias_counterexample']['pre_cast_bias_replacement'])==(0,1)
    assert (witness['K_split_BF16_atomic_counterexample']['original_full_K_then_BF16'],witness['K_split_BF16_atomic_counterexample']['BF16_partial_atomics_reference'])==(1,0)
    bounds=read(O/'quant-epilogue-boundary-models-v82.json')
    assert len(bounds['models'])==5 and bounds['read_write_or_aggregate_capacity_upper_values'] is None
    for row in bounds['models']:
        elems=row['M']*5120;assert row['K']==2304 and row['elements']==elems
        assert row['base_total_value_bytes_including_raw_bridge']==16*elems
        assert row['VectorFresh_total_value_bytes_including_raw_bridge']==12*elems
        assert row['base_GM_read_bytes']==row['base_GM_write_bytes']==8*elems
        assert row['VectorFresh_GM_read_bytes']==row['VectorFresh_GM_write_bytes']==6*elems
        assert row['removable_S_BF16_interface_reference_bytes']==4*elems
        assert row['fresh_copy_then_atomic_value_reference_bytes']==8*elems
        assert row['actual_HBM_traffic_or_time_gain'] is None
    extension=read(O/'Q_B-index-epilogue-extension-v82.json')
    assert extension['no_post_QB_head_RMS_512_in_active_case']
    for row in extension['Q_B_models']:
        assert row['RoPE_tail_interface_removable_value_bytes']==row['original_v75_tail_budget']==4*row['M']*64*64
        assert row['conditional_common_INT32_CV_bridge_value_bytes']==8*row['M']*32768
    if analysis_only:
        print(dict(pre_checkpoint_analysis='PASS',public_files=5,API_pages=2,regimes=31,executions=14143,native_execution=False))
        return
    manifest=read(O/'checkpoint-manifest-v82.json')
    for item in manifest['artifacts']:
        blob=Path(item['path']).read_bytes()
        assert len(blob)==item['bytes'] and hashlib.sha256(blob).hexdigest()==item['sha256'],item['path']
    shared=read(O/'shared-quant-owner-CV-interfaces-v82.json')
    for row in shared['models']:
        rows=row['M'];assert row['four_bridge_total_value_bytes']==10*rows*5120+18*rows*2304
        assert row['four_bridge_total_value_bytes']-row['old_Cube_owned_BF16_Down_domain_four_bridge_bytes']==4*rows*5120
    before={m['id']:m for m in read(O/'module-phase-branch-matrix-v81.json')['modules']}
    after={m['id']:m for m in read(O/'module-phase-branch-matrix-v82.json')['modules']}
    assert len(after)==26 and sum(len(m['branches'])for m in after.values())==342
    assert sum(m.get('branch_schema')=='SEMANTIC_BRANCHES_V1'for m in after.values())==24
    changed={'M07','M09','M13','M22'}
    for mid,mod in after.items():
        assert not mod.get('closed',False)
        assert [b.get('id')for b in mod['branches']]==[b.get('id')for b in before[mid]['branches']]
        if mid not in changed:assert mod['branches']==before[mid]['branches']
    candidates=read(O/'candidate-profile-ledger-v82.json')['candidates']
    assert len(candidates)==35 and all(not c['candidate_closed'] and not c['legal_fusion_proved'] for c in candidates)
    assert len(read(O/'framework-candidate-ledger-v82.json')['candidates'])==7
    clauses=read(O/'candidate-relation-clauses-v82.json')
    assert len(clauses['clauses'])==47 and len(clauses['higher_order_groups'])==11
    assert len({tuple(sorted((c['a'],c['b'])))for c in clauses['clauses']})==47
    relation=read(O/'candidate-relation-coverage-v82.json')
    assert len(relation['pairs'])==595 and relation['scoped_conditional_pairs']==47 and relation['remaining_pair_relations']==548
    assert sum(bool(p['clauses'])for p in relation['pairs'])==47
    assert all(not p['whole_pair_all_variants_closed']for p in relation['pairs'])
    operators=read(O/'operator-analysis-coverage-ledger-v82.json')
    assert (operators['scoped_conditional_review_regimes'],operators['scoped_conditional_review_executions'])==(2073,285598)
    assert (operators['remaining_regime_reconciliation_count'],operators['remaining_execution_reconciliation_count'])==(9,1352)
    assert operators['regimes']==read(O/'operator-analysis-coverage-ledger-v81.json')['regimes']
    checkpoint=read(O/'REVIEW-CHECKPOINT-v82.json')
    assert checkpoint['status']=='PROGRESS' and not checkpoint['overall_complete'] and checkpoint['whole_modules_closed']==0
    for path in [N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json']:
        obj=read(path);assert obj['latest_profile_first_review']['candidate_ledger']==str(O/'candidate-profile-ledger-v82.json')
    assert read(TASK_DIR/'index.json')['status']=='ACTIVE' and read(L/'state.json')['status']=='EVALUATING'
    history=read(O/'quant-epilogue-validation-history-v82.json');assert len(history['errors'])==1 and history['errors'][0]['resolved']
    print(dict(checkpoint=82,status='PROGRESS',manifest_artifacts_verified=len(manifest['artifacts']),
        public_files=5,API_pages=2,profile=[31,14143],BF16_AIV_positive=9671,INT32_control_AIV_zero=4472,
        DownAdd_variants=5,rounding_counterexamples=2,records=342,semantic_modules=24,
        relation_clauses=47,higher_order_groups=11,central_pending=548,profile_reviewed=[2073,285598],
        profile_pending=[9,1352],native_source_attested=False,new_native_gain=False,whole_modules_closed=0,overall_complete=False))


if __name__=='__main__': main('--analysis-only' in sys.argv)
