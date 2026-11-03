"""Bounded v69 evidence and persistence check, never a full-model completion check."""
from collections import Counter
from pathlib import Path
import hashlib
from finalize_origin_coverage_v9 import N,O,R,W,T,L,read

def main():
    cp=read(O/'REVIEW-CHECKPOINT-v69.json');assert cp['status']=='PROGRESS'and cp['whole_modules_closed']==0 and cp['overall_complete']is False
    mf=read(O/'checkpoint-manifest-v69.json')['artifacts']
    for a in mf:
        q=Path(a['path']);b=q.read_bytes();assert len(b)==a['bytes']and hashlib.sha256(b).hexdigest()==a['sha256'],q
    masters=read(O/'candidate-record-index-v69.json')['model_candidates'];assert len(masters)==35
    for a in masters:assert hashlib.sha256(Path(a['master_record']).read_bytes()).hexdigest()==a['master_sha256']
    p=read(O/'remaining-field-regime-models-v69.json');assert(p['regime_count'],p['captured_execution_count'])==(161,6680)
    keys=[tuple(k)for r in p['regimes']for k in r['native_TASK_keys']];assert len(keys)==len(set(keys))==6680 and sum(r['missing_PMU']for r in p['regimes'])==0
    bykey={tuple(k):r for r in p['regimes']for k in r['native_TASK_keys']}
    g=read(O/'remaining-field-native-chains-v69.json')['chains'];sizes={'next_ID_13TASK':(88,13),'rope_173_3TASK':(288,3),'rope_174_3TASK':(288,3),'rope_185_2TASK':(96,2),'rope_186_2TASK':(96,2)}
    for name,(count,num)in sizes.items():
        assert len(g[name])==count and all(len(x['stages'])==num for x in g[name])
        ks=[(x['database_index'],s['TASK_rowid'])for x in g[name]for s in x['stages']];assert len(ks)==len(set(ks))==count*num and set(ks)<=set(keys)
        if name.startswith('next'):
            for x in g[name]:
                B,K=x['shape'];assert sum(bykey[(x['database_index'],s['TASK_rowid'])]['known_logical_value_bytes_excluding_control']for s in x['stages'])==38*B*K+94*B
    costs=read(O/'remaining-field-fusion-costs-v69.json');assert len(costs['next_ID_models'])==88 and len(costs['rope_lookup_models'])==384
    for x in costs['next_ID_models']:
        B,K=x['B'],x['K'];assert x['fused_keep_row_values_and_read_all_backup_reference_bytes']==4*B*K+16*B
        assert x['logical_interface_budget_bytes']==34*B*K+78*B
        if K==6:assert x['fused_plus_padded_inputs_reference_bytes']==4*B*K+36*B-4
    assert Counter((x['B'],x['K'])for x in costs['next_ID_models'])=={(2,1):16,(2,6):16,(4,6):56}
    for x in costs['rope_lookup_models']:
        assert x['source_known_value_bytes']==(4112 if x['kind']=='cached'else 1056)*x['N']
        assert x['joint_row_lookup_reference_bytes']==1032*x['N']and x['required_two_FLOAT_outputs_bytes']==512*x['N']
    for x in costs['SWA_models']:
        assert x['C']==256 and x['N']==5*x['B']
        assert x['same_layout_required_output_bytes']==5*x['compact_required_output_bytes']
    for x in costs['group_slot_models']:assert x['capacity_C_unbound']and x['G']==13
    variants=read(O/'remaining-field-fusion-variants-v69.json');assert variants['variant_count']==len(variants['variants'])==20
    assert sum(x['decision']=='EXCLUDED_WITH_COUNTEREXAMPLE'for x in variants['variants'])==2
    math=read(O/'remaining-field-math-validation-v69.json');assert [math[k]for k in ['next_ID_exact_cases','full_slot_suffix_cases','whole_BOOL_threshold_cases']]==[39060,5524,5524]
    assert math['hole_counterexample']['baseline']==-1 and math['negative_counterexample']['baseline_valid']is True and len(math['source_query_reference_correction'])==5
    for r in p['regimes']:
        if r['operator']=='copy_and_expand_dflash_and_dspark_inputs_kernel':assert r['work_model']['query_source_value_reference_bytes']=='(28+4*HAS_REJECTED)*B*Q'
    opaque=read(O/'opaque-plan-TASK-type-audit-v69.json');assert len(opaque['native_tasks'])==529 and opaque['task_type_counts']=={'KERNEL_AICPU':529}and opaque['missing_PMU']==529
    assert opaque['operator_counts']=={'SparseFlashMlaMetadata':321,'VllmAscendQuantLightningIndexerV2Metadata':208}
    source=read(O/'remaining-field-source-binding-v69.json');assert len(source['files'])==10 and sum(x['prior_historical_file_pin_matches']for x in source['files'])==9
    for x in source['files']:assert hashlib.sha256(Path(x['path']).read_bytes()).hexdigest()==x['sha256']
    assert source['forward_context_current_file_same_revision_is_condition']
    mask=read(O/'MC2-mask-typed-valuation-v69.json');assert [mask[k]for k in ['source_known_value_bytes','known_materialized_BOOL_bytes','logical_interface_budget_bytes']]==[1461368,161984,1299384]
    c=read(O/'operator-analysis-coverage-ledger-v69.json');prev=read(O/'operator-analysis-coverage-ledger-v68.json');assert len(c['regimes'])==2082
    done=[r for r in c['regimes']if r['state'].startswith('SCOPED_')];pending=[r for r in c['regimes']if r['state'].startswith('REGIME_')]
    assert(len(done),sum(r['executions']for r in done))==(1992,282966)and(len(pending),sum(r['executions']for r in pending))==(90,3984)
    assert Counter('|'.join(r['module_candidates'])for r in pending)=={'M21':81,'M12':3,'M19':4,'M05|M21':2}
    selected={r['regime_id']for r in p['regimes']}
    for a,b in zip(prev['regimes'],c['regimes']):
        assert a['regime_id']==b['regime_id']
        if a['regime_id']not in selected:assert a==b
        else:assert a['state'].startswith('REGIME_')and b['state'].startswith('SCOPED_')
    assert(c['explicit_profile_insufficient_regimes'],c['explicit_profile_insufficient_executions'])==(7,1272)
    assert(c['explicit_unresolved_BOOL_field_regimes'],c['explicit_unresolved_BOOL_field_executions'])==(2,80)
    assert c['unchanged_global_uncertainty']=={'ambiguous_module_assignment_executions':3136,'missing_PMU':849,'phase_unbound_graphs':16}
    assert sum(r['executions']for r in c['regimes'])==286950
    old={m['id']:m for m in read(O/'module-phase-branch-matrix-v68.json')['modules']};new={m['id']:m for m in read(O/'module-phase-branch-matrix-v69.json')['modules']};inv=read(O/'M21-remaining-field-branch-inventory-v69.json')
    assert len(old)==len(new)==26 and new['M21']['branches']==inv['modules']['M21']and len(new['M21']['branches'])==18
    assert read(Path(new['M21']['legacy_branch_history']))['module']==old['M21']
    for mid in old:
        if mid!='M21':assert new[mid]==old[mid]
    for a,b in zip(old['M21']['branches'],new['M21']['branches']):
        assert a['id']==b['id']and b['evidence'][:len(a['evidence'])]==a['evidence']
        if a['id']not in inv['extended_families']:assert a==b
    assert len(inv['extended_families'])==9
    for q in [N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json']:assert read(q)['latest_profile_first_review']==cp
    assert read(T/'index.json')['status']=='ACTIVE'and read(L/'state.json')['status']=='EVALUATING'
    assert read(O/'candidate-profile-ledger-v69.json')['whole_modules_closed']==0 and len(read(O/'framework-candidate-ledger-v69.json')['candidates'])==7
    print(dict(manifest_artifacts_verified=len(mf),candidate_masters=35,sources=10,prior_historical_SHA_matches=9,new_regimes=161,new_unique_TASKs=6680,next_ID_groups=88,rope_pairs=384,variants=20,opaque_AICPU_TASKs=529,M21_families_retained=18,enriched=9,other_modules_identical=25,reviewed=[1992,282966],pending=[90,3984],checkpoint_status='PROGRESS',whole_modules_closed=0,overall_complete=False))

if __name__=='__main__':main()
