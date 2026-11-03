"""v71 bounded operator/source/coverage audit, not a full-goal completion check."""
from collections import Counter
from pathlib import Path
import hashlib,json
from finalize_origin_coverage_v9 import N,O,R,W,T as TASK_DIR,L,read

def main():
    cp=read(O/'REVIEW-CHECKPOINT-v71.json');assert cp['status']=='PROGRESS'and cp['whole_modules_closed']==0 and cp['overall_complete']is False
    mf=read(O/'checkpoint-manifest-v71.json')['artifacts']
    for a in mf:
        path=Path(a['path']);b=path.read_bytes();assert len(b)==a['bytes']and hashlib.sha256(b).hexdigest()==a['sha256'],path
    masters=read(O/'candidate-record-index-v71.json')['model_candidates'];assert len(masters)==35
    for a in masters:assert hashlib.sha256(Path(a['master_record']).read_bytes()).hexdigest()==a['master_sha256']
    p=read(O/'planner-regime-models-v71.json');assert(p['regime_count'],p['captured_execution_count'])==(21,608)
    keys=[tuple(k)for r in p['regimes']for k in r['native_TASK_keys']];assert len(keys)==len(set(keys))==608
    assert sum(r['missing_PMU']for r in p['regimes'])==529
    ops=Counter()
    for r in p['regimes']:
        ops[r['operator']]+=r['executions'];work=r['work_model']
        if r['operator']=='ZerosLike':assert r['missing_PMU']==0 and work['known_value_write_bytes']==4096 and work['exact_value_stores']==1024
        else:
            assert r['native_TASK_types']=={'KERNEL_AICPU':r['executions']}and r['missing_PMU']==r['executions']
            expected=3600 if r['operator']=='SparseFlashMlaMetadata'else 3456
            assert work['defined_prefix_bytes']==expected and work['unwritten_suffix_bytes']==4096-expected
            assert work['defined_plan_fields_contain_tensor_addresses']is False and work['actual_input_values_and_hitrate_known']is False
            assert work['source_fixed_bank_copy_reference_bytes']==8192 and work['copy_TRACE_TASK_identity_proved']is False
    assert ops=={'SparseFlashMlaMetadata':321,'VllmAscendQuantLightningIndexerV2Metadata':208,'ZerosLike':79}
    old_audit=read(O/'opaque-plan-TASK-type-audit-v69.json');oldkeys={(r['database_index'],r['TASK_rowid'])for r in old_audit['native_tasks']};currentkeys={tuple(k)for r in p['regimes']if r['operator']!='ZerosLike'for k in r['native_TASK_keys']};assert oldkeys==currentkeys and len(oldkeys)==529
    c=read(O/'planner-source-algorithm-contract-v71.json');assert len(c['source_files'])==13
    for row in c['source_files']:assert hashlib.sha256(Path(row['local_path']).read_bytes()).hexdigest()==row['sha256']
    assert sum(r.get('public_source_bytes_match',False)for r in c['source_files'])==6
    locators=read(O/'planner-public-source-locators-v71.json')['files']+read(O/'planner-public-api-locators-v71.json')['files'];assert len(locators)==6
    for row in locators:
        b=Path(row['path']).read_bytes();assert hashlib.sha1(b'blob '+str(len(b)).encode()+bytes([0])+b).hexdigest()==row['blob_sha']
        original=next(r for r in c['source_files']if r['path']=='/vllm-workspace/vllm-ascend/'+row['repo_path']);assert Path(original['local_path']).read_bytes()==b
    support=read(O/'prior-support-file-binding-reconciliation-v71.json');assert len(support['files'])==3 and support['all3_exact_original_container_bytes_match_frozen']
    for row in support['files']:assert row['prior_frozen_SHA_match']and hashlib.sha256(Path(row['prior_frozen_path']).read_bytes()).hexdigest()==row['sha256']
    assert len(c['SMLA_effective_AICPU_attrs'])==22 and len(c['QLI_effective_AICPU_attrs'])==14 and c['fresh_reset_state_condition']and c['loaded_native_binary_attested']is False
    assert 'is_batch_consistency'in c['SMLA_effective_AICPU_attrs']and 'quant_mode'not in c['QLI_effective_AICPU_attrs']
    for name,prefix,suffix in [('SMLA',3600,496),('QLI',3456,640)]:assert c['defined_schema'][name]['defined_prefix_bytes']==prefix and c['defined_schema'][name]['unwritten_suffix_bytes']==suffix
    costs=read(O/'planner-fusion-costs-v71.json');assert len(costs['publication_models'])==21 and len(costs['store_models'])==1176
    assert costs['source_bank_copy_reference_bytes_total']==4333568 and costs['caller_zero_value_write_bytes_total']==323584
    for x in costs['store_models']:
        u,v=x['U'],x['V'];assert x['SMLA_source_value_store_bytes']==3888+28*u+24*v and x['compact_A3_nonSplitG_endpoints_reference_bytes']==4*(2+4*u+6*v)
        if v==0:assert x['QLI_source_value_store_bytes']==3744+28*u
        else:assert x['QLI_source_value_store_bytes']is None
    assert [x['positive_plan_families']for x in costs['conditional_SP8_SWA_families']]==[3,7]
    for x in costs['conditional_SP8_SWA_families']:assert sum(r['T']for r in x['rank_plans'])==x['global_query_rows']
    math=read(O/'planner-math-validation-v71.json');assert math['sparse_original_saturation_cases']==174085 and math['A3_store_UV_cases']==1176 and math['QLI_small_query_histogram_cases']==2450
    assert math['QLI_same2048bucket_cost_counterexample']['cost']==[58,68]and math['callerzero_vs_nativeempty_core_enable']==[0,1]and math['actual_69_19_population_is_not_hitrate_proof']
    variants=read(O/'planner-fusion-variants-v71.json');assert variants['variant_count']==len(variants['variants'])==16
    assert sum(v['decision'].startswith('EXCLUDED_')for v in variants['variants'])==3 and all(v['measured_speedup']is None for v in variants['variants'])
    cov=read(O/'operator-analysis-coverage-ledger-v71.json');prev=read(O/'operator-analysis-coverage-ledger-v70.json');reviewed=[r for r in cov['regimes']if r['state'].startswith('SCOPED_')];pending=[r for r in cov['regimes']if r['state'].startswith('REGIME_')]
    assert(len(reviewed),sum(r['executions']for r in reviewed))==(2073,285598)and(len(pending),sum(r['executions']for r in pending))==(9,1352)
    assert len(cov['regimes'])==2082 and sum(r['executions']for r in cov['regimes'])==286950
    assert Counter('|'.join(r['module_candidates'])for r in pending)=={'M19':4,'M12':3,'M05|M21':2}
    selected={r['regime_id']for r in p['regimes']}
    for a,b in zip(prev['regimes'],cov['regimes']):
        assert a['regime_id']==b['regime_id']
        if a['regime_id']not in selected:assert a==b
        else:assert a['state'].startswith('REGIME_')and b['state'].startswith('SCOPED_')
    m21=[r for r in cov['regimes']if r['module_candidates']==['M21']];assert len(m21)==828 and sum(r['executions']for r in m21)==76943 and all(r['state'].startswith('SCOPED_')for r in m21)
    assert(cov['explicit_profile_insufficient_regimes'],cov['explicit_profile_insufficient_executions'])==(7,1272)
    assert(cov['explicit_unresolved_BOOL_field_regimes'],cov['explicit_unresolved_BOOL_field_executions'])==(2,80)
    assert cov['unchanged_global_uncertainty']=={'ambiguous_module_assignment_executions':3136,'missing_PMU':849,'phase_unbound_graphs':16}
    before={m['id']:m for m in read(O/'module-phase-branch-matrix-v70.json')['modules']};after={m['id']:m for m in read(O/'module-phase-branch-matrix-v71.json')['modules']};inv=read(O/'M21-planner-branch-inventory-v71.json')
    assert len(before)==len(after)==26 and after['M21']['branches']==inv['modules']['M21']and len(after['M21']['branches'])==18 and len(inv['extended_families'])==9
    assert read(Path(after['M21']['legacy_branch_history']))['module']==before['M21']
    for mid in before:
        if mid!='M21':assert before[mid]==after[mid]
    for a,b in zip(before['M21']['branches'],after['M21']['branches']):
        assert a['id']==b['id']and b['evidence'][:len(a['evidence'])]==a['evidence']
        if a['id']not in inv['extended_families']:assert a==b
    for path in [N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json']:assert read(path)['latest_profile_first_review']==cp
    assert read(TASK_DIR/'index.json')['status']=='ACTIVE'and read(L/'state.json')['status']=='EVALUATING'
    runs=read(O/'task-run-registration-v71.json')['runs'];assert len(runs)==6 and Counter(r['status']for r in runs)=={'pass':4,'error':2}
    assert len(read(O/'framework-candidate-ledger-v71.json')['candidates'])==7
    print(dict(manifest_artifacts_verified=len(mf),candidate_master_hashes_verified=35,new_regimes=21,new_unique_TASKs=608,source_files=13,official_immutable_blob_matches=6,prior_support_SHA_matches=3,planner_variants=16,all_unambiguous_M21_conditional_reviewed=[828,76943],reviewed=[2073,285598],pending=[9,1352],M21_families_retained=18,enriched=9,other_modules_identical=25,missing_PMU_849_preserved=True,checkpoint_status='PROGRESS',whole_modules_closed=0,overall_complete=False))

if __name__=='__main__':main()
