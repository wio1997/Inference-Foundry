"""Read-only v68 evidence/count/cost/persistence audit, not a full-goal verifier."""
from pathlib import Path
from collections import Counter
import hashlib,json
from finalize_origin_coverage_v9 import N,O,R,W,T,L,read

def main():
    cp=read(O/'REVIEW-CHECKPOINT-v68.json');assert cp['status']=='PROGRESS'and cp['whole_modules_closed']==0 and cp['overall_complete']is False
    mf=read(O/'checkpoint-manifest-v68.json')['artifacts']
    for a in mf:
        p=Path(a['path']);b=p.read_bytes();assert len(b)==a['bytes']and hashlib.sha256(b).hexdigest()==a['sha256'],p
    masters=read(O/'candidate-record-index-v68.json')['model_candidates'];assert len(masters)==35
    for a in masters:assert hashlib.sha256(Path(a['master_record']).read_bytes()).hexdigest()==a['master_sha256']
    p=read(O/'metadata-field-regime-models-v68.json');assert len(p['regimes'])==586
    keys=[tuple(k)for r in p['regimes']for k in r['native_TASK_keys']]
    assert len(keys)==len(set(keys))==67631 and sum(r['missing_PMU']for r in p['regimes'])==0
    bykey={tuple(k):r for r in p['regimes']for k in r['native_TASK_keys']}
    g=read(O/'metadata-field-native-chains-v68.json')
    sizes={'CP_length_groups':(2792,5),'C2_cos_lookup_groups':(104,3),'C2_sin_lookup_groups':(104,3),
        'SWA_recount_groups':(69,4),'C2_length_quotient_residual_groups':(208,5),'slot_coordinate_core_groups':(2601,14)}
    for name,(count,n)in sizes.items():
        assert len(g[name])==count and all(len(x['stages'])==n for x in g[name])
        ids=[(x['database_index'],s['TASK_rowid'])for x in g[name]for s in x['stages']]
        assert len(ids)==len(set(ids))==count*n and set(ids)<=set(keys)
        for x in g[name]:
            value=sum(bykey[(x['database_index'],s['TASK_rowid'])]['known_logical_value_bytes_excluding_unknown_control_scalars']for s in x['stages'])
            nr=x['rows'];expected={'CP_length_groups':38*nr+4,'C2_cos_lookup_groups':2056*nr,'C2_sin_lookup_groups':2056*nr,
                'SWA_recount_groups':7684*nr,'C2_length_quotient_residual_groups':56*nr,'slot_coordinate_core_groups':140*nr}[name]
            assert value==expected
    header=read(O/'metadata-C2-header-native-check-v68.json');assert len(header['header_groups'])==104
    for x in header['header_groups']:
        assert x['known_14TASK_value_bytes']==116*x['B']+8 and x['additional_five_source_copy_reference_bytes']==40*x['B']
        assert x['source_copy_TRACE_identity_proved']is False
    cost=read(O/'metadata-field-fusion-costs-v68.json')
    for x in cost['slot_coordinate_models']:
        assert x['required_coordinate_output_bytes']==8*x['N']and x['required_flat_output_bytes']==8*x['N']
        assert x['fused_raw_INT32_read_and_INT32_pairs_INT64_flat_write_bytes']==20*x['N']
        assert x['interface_budget_excluding_unknown_control_scalars']==120*x['N']+12
    for x in cost['C2_cos_sin_row_lookup_models']:
        assert x['source_two_3TASK_Broadcast64_Cast32_Gather_value_bytes']==4112*x['N']
        assert x['row_cast_then_two_gather_interface_budget_bytes']==3068*x['N']and x['joint_row_lookup_interface_budget_bytes']==3080*x['N']
    assert next(x for x in cost['C2_cos_sin_row_lookup_models']if x['N']==6400)['joint_row_lookup_interface_budget_bytes']==19712000
    contract=read(O/'metadata-field-operator-contract-v68.json');assert len(contract['source_files'])==3
    for r in contract['source_files']:assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256']
    assert hashlib.sha256(Path(contract['config_path']).read_bytes()).hexdigest()==contract['config_sha256']
    param=read(O/'metadata-fixed-parameter-reconciliation-v68.json')
    assert param['actual_distinct_ratios']==[0,1,2]and set(param['actual_compress_ratios'])=={0,1,2}and param['sliding_window']==128
    assert contract['existing_C2_fused_builder']['already_implemented']is True
    variants=read(O/'metadata-field-fusion-variants-v68.json')['variants'];assert len(variants)==22
    assert 'existingv39'in next(v['choice']for v in variants if v['id']=='META-C2-TOKEN-CS-COMPOSE')
    cov=read(O/'operator-analysis-coverage-ledger-v68.json');assert len(cov['regimes'])==2082
    rev=[r for r in cov['regimes']if r['state'].startswith('SCOPED_')];pending=[r for r in cov['regimes']if r['state'].startswith('REGIME_')]
    assert (len(rev),sum(r['executions']for r in rev))==(1831,276286)
    assert (len(pending),sum(r['executions']for r in pending))==(251,10664)
    assert len(rev)+len(pending)==2082 and sum(r['executions']for r in cov['regimes'])==286950
    assert Counter('|'.join(r['module_candidates'])for r in pending)=={'M21':242,'M19':4,'M12':3,'M05|M21':2}
    opaque=[r for r in pending if r['operator']in ['SparseFlashMlaMetadata','VllmAscendQuantLightningIndexerV2Metadata']]
    clears=[r for r in pending if r['operator']=='ZerosLike'and '(1174)'in r['role']]
    assert (len(opaque),sum(r['executions']for r in opaque))==(18,529)
    assert (len(clears),sum(r['executions']for r in clears))==(3,79)
    assert (cov['explicit_profile_insufficient_regimes'],cov['explicit_profile_insufficient_executions'])==(7,1272)
    assert (cov['explicit_unresolved_BOOL_field_regimes'],cov['explicit_unresolved_BOOL_field_executions'])==(2,80)
    assert cov['unchanged_global_uncertainty']=={'ambiguous_module_assignment_executions':3136,'missing_PMU':849,'phase_unbound_graphs':16}
    new={m['id']:m for m in read(O/'module-phase-branch-matrix-v68.json')['modules']}
    old={m['id']:m for m in read(O/'module-phase-branch-matrix-v67.json')['modules']}
    inv=read(O/'M21-typed-field-branch-inventory-v68.json')
    assert new['M21']['branches']==inv['modules']['M21']and len(new['M21']['branches'])==18
    assert read(Path(new['M21']['legacy_branch_history']))['module']==old['M21']
    ids=set(inv['extended_families']);assert len(ids)==8
    for before,after in zip(old['M21']['branches'],new['M21']['branches']):
        assert (before['id'],before['branch'])==(after['id'],after['branch'])
        if before['id']not in ids:assert before==after
        else:assert after['evidence'][:len(before['evidence'])]==before['evidence']and after['typed_operator_review_v68']
    protected=read(O/'metadata-field-protected-module-state-v68.json')['modules']
    assert len(protected)==25 and all(new[mid]==before for mid,before in protected.items())
    assert read(O/'framework-candidate-ledger-v67.json')['candidates']==read(O/'framework-candidate-ledger-v68.json')['candidates']
    for q in [N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json']:
        d=read(q);assert d['latest_profile_first_review']==cp and d['overall_complete']is False
    for q in [L/'state.json',T/'index.json']:
        d=read(q);assert d['status']in ['EVALUATING','ACTIVE']and d['next_action']==cp['next_action']
    runs=read(O/'task-run-registration-v68.json')['runs'];assert len(runs)==6 and sum(r['status']=='error'for r in runs)==1
    print(json.dumps(dict(manifest_artifacts_verified=len(mf),candidate_master_hashes_verified=35,source_hashes_verified=3,
        new_native_regimes=586,new_unique_TASKs=67631,CP_groups=2792,coordinate_groups=2601,C2_header_groups=104,
        C2_CS_groups=[104,104],SWA_recount_groups=69,variants=22,existing_C2_kernel_recognized=True,actual_compression_ratios=[0,1,2],
        M21_families_retained=18,enriched=8,other_modules_identical=25,reviewed=[1831,276286],pending=[251,10664],
        opaque_ops_pending=[18,529],plan_zero_pending=[3,79],checkpoint_status='PROGRESS',whole_modules_closed=0,overall_complete=False),ensure_ascii=False))

if __name__=='__main__':main()
