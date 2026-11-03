"""v70 bounded operator/evidence check; not a whole-goal verifier."""
from collections import Counter
from pathlib import Path
import hashlib
from finalize_origin_coverage_v9 import N,O,R,W,T as TASK_DIR,L,read

def main():
    cp=read(O/'REVIEW-CHECKPOINT-v70.json');assert cp['status']=='PROGRESS'and cp['whole_modules_closed']==0 and cp['overall_complete']is False
    mf=read(O/'checkpoint-manifest-v70.json')['artifacts']
    for a in mf:
        q=Path(a['path']);b=q.read_bytes();assert len(b)==a['bytes']and hashlib.sha256(b).hexdigest()==a['sha256'],q
    masters=read(O/'candidate-record-index-v70.json')['model_candidates'];assert len(masters)==35
    for a in masters:assert hashlib.sha256(Path(a['master_record']).read_bytes()).hexdigest()==a['master_sha256']
    p=read(O/'selected-input-regime-models-v70.json');assert(p['regime_count'],p['captured_execution_count'])==(60,2024)
    keys=[tuple(k)for r in p['regimes']for k in r['native_TASK_keys']];assert len(keys)==len(set(keys))==2024 and sum(r['missing_PMU']for r in p['regimes'])==0
    bykey={tuple(k):r for r in p['regimes']for k in r['native_TASK_keys']}
    g=read(O/'selected-input-native-chains-v70.json')['chains'];sizes={'COMPOSED_DRAFT_ID_GATHER':(72,5),'AUX_GATHER_CONCAT':(72,7),'DRAFT_FIXED_BANK_TAILS':(88,4),'MARKOV_SEED_COLUMN':(104,1),'AUX_PREFIX_CONCAT':(16,1),'TARGET_INPUT_ID_GATHER':(72,2),'TARGET_POSITION_GATHER':(72,2),'NEXT_STEP_POSITIONS':(88,2),'SAMPLE_HIDDEN_ROWS':(104,2),'INITIAL_SELECTOR_TEMP':(16,1)}
    allgroup=[]
    for f,(count,num)in sizes.items():
        assert len(g[f])==count and all(len(x['stages'])==num for x in g[f])
        ks=[(x['database_index'],s['TASK_rowid'])for x in g[f]for s in x['stages']];assert len(ks)==len(set(ks))==count*num and set(ks)<=set(keys);allgroup+=ks
    assert len(allgroup)==len(set(allgroup))==2024
    costs=read(O/'selected-input-fusion-costs-v70.json')
    for name,f,field in [('composed_ID_models','COMPOSED_DRAFT_ID_GATHER','source_5TASK_known_value_bytes'),('aux_gather_concat_models','AUX_GATHER_CONCAT','source_7TASK_known_value_bytes'),('aux_prefix_concat_models','AUX_PREFIX_CONCAT','source_concat_known_value_bytes'),('draft_tail_models','DRAFT_FIXED_BANK_TAILS','source_4Fill_write_bytes')]:
        assert len(costs[name])==len(g[f])
        for a,b in zip(costs[name],g[f]):assert a[field]==sum(bykey[(b['database_index'],s['TASK_rowid'])]['known_logical_value_bytes_excluding_control']for s in b['stages'])
    for x in costs['composed_ID_models']:
        S,D=x['S'],x['D'];assert x['source_5TASK_known_value_bytes']==24*S+24*D and x['full_L_validation_with_local_index_reuse_reference_bytes']==8*S+12*D and x['full_L_validation_interface_budget_bytes']==16*S+12*D
    for x in costs['aux_gather_concat_models']:
        T,H=x['T'],x['H'];assert H==5120 and x['joint_gather_concat_reference_bytes']==4*T+12*T*H and x['joint_interface_budget_bytes']==20*T+12*T*H
    assert max(x['joint_interface_budget_bytes']for x in costs['aux_gather_concat_models'])==409815280
    assert next(x for x in costs['aux_prefix_concat_models']if x['T']==6400)['source_concat_known_value_bytes']==393216000
    assert sum(x['source_4Fill_write_bytes']for x in costs['draft_tail_models'])==11531424
    assert Counter(x['B']for x in costs['draft_tail_models'])=={2:32,4:56}
    for x in costs['draft_tail_models']:assert x['target_G13_slot_capacity_not_bound_by_this']and x['first_three_before_metadata_builder']and x['last_sample_tail_later']
    assert len(costs['Markov_seed_copy_models'])==104
    for x in costs['Markov_seed_copy_models']:assert x['source_selected_copy_bytes']==16*x['B']and x['source_base_capacity_bytes']==192
    assert len(costs['simple_gather_models'])==336
    for x in costs['simple_gather_models']:assert x['source_2TASK_known_value_bytes']==(8+2*x['row_bytes'])*x['T']
    sample=Counter()
    for r in p['regimes']:
        if r['family']=='SAMPLE_HIDDEN_ROWS'and r['operator']=='Index':sample[r['context']]+=r['executions']
    assert sample=={'CALLER_DUMMY':16,'CALLER_DRAFT_OR_DRAFT_METADATA':88}
    c=read(O/'selected-input-operator-contract-v70.json');assert len(c['source_files'])==8 and sum(r['prior_historical_SHA_match']for r in c['source_files'])==6
    for r in c['source_files']:assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256']
    assert c['same166revision_condition_for_two_support_files']
    proof=read(O/'selected-input-def-use-and-guard-proof-v70.json');assert proof['used_update_positions_load_lines']==[1312,1332]and proof['no_incoming_selector_load_before_overwrite']
    assert proof['all_loads_under_should_update_next_steps']and proof['draft_bank_capacity_binding']['profile_all88four_fill_groups_match']
    b=proof['draft_bank_capacity_binding'];assert hashlib.sha256(Path(b['serve']).read_bytes()).hexdigest()==b['serve_sha256']
    assert [b[k]for k in ['slot','seq','QSL','sample']]==[8208,8200,8200,8192]
    math=read(O/'selected-input-math-validation-v70.json');assert math['composed_negative_repeat_valid_cases']==8132 and math['aux_selection_concat_cases']==780 and math['dirty_tail_write_bytes']==[0,0,96,0,0,96]
    vs=read(O/'selected-input-fusion-variants-v70.json');assert vs['variant_count']==len(vs['variants'])==16
    assert sum(v['decision']=='EXCLUDED_WITH_COUNTEREXAMPLE'for v in vs['variants'])==2 and sum(v['decision']=='EXCLUDED_WITH_UNCLOSED_CONSUMER'for v in vs['variants'])==1
    cov=read(O/'operator-analysis-coverage-ledger-v70.json');prev=read(O/'operator-analysis-coverage-ledger-v69.json');done=[r for r in cov['regimes']if r['state'].startswith('SCOPED_')];pending=[r for r in cov['regimes']if r['state'].startswith('REGIME_')]
    assert(len(done),sum(r['executions']for r in done))==(2052,284990)and(len(pending),sum(r['executions']for r in pending))==(30,1960)
    assert len(cov['regimes'])==2082 and sum(r['executions']for r in cov['regimes'])==286950
    selected={r['regime_id']for r in p['regimes']}
    for a,b in zip(prev['regimes'],cov['regimes']):
        assert a['regime_id']==b['regime_id']
        if a['regime_id']not in selected:assert a==b
        else:assert a['state'].startswith('REGIME_')and b['state'].startswith('SCOPED_')
    assert Counter('|'.join(r['module_candidates'])for r in pending)=={'M21':21,'M19':4,'M12':3,'M05|M21':2}
    assert(cov['explicit_profile_insufficient_regimes'],cov['explicit_profile_insufficient_executions'])==(7,1272)
    assert(cov['explicit_unresolved_BOOL_field_regimes'],cov['explicit_unresolved_BOOL_field_executions'])==(2,80)
    assert cov['unchanged_global_uncertainty']=={'ambiguous_module_assignment_executions':3136,'missing_PMU':849,'phase_unbound_graphs':16}
    old={m['id']:m for m in read(O/'module-phase-branch-matrix-v69.json')['modules']};new={m['id']:m for m in read(O/'module-phase-branch-matrix-v70.json')['modules']};inv=read(O/'M21-selected-input-branch-inventory-v70.json')
    assert len(old)==len(new)==26 and new['M21']['branches']==inv['modules']['M21']and len(new['M21']['branches'])==18
    assert read(Path(new['M21']['legacy_branch_history']))['module']==old['M21']
    assert len(inv['extended_families'])==6
    for mid in old:
        if mid!='M21':assert new[mid]==old[mid]
    for a,b in zip(old['M21']['branches'],new['M21']['branches']):
        assert a['id']==b['id']and b['evidence'][:len(a['evidence'])]==a['evidence']
        if a['id']not in inv['extended_families']:assert a==b
    assert 'F27'in new['M21']['candidate_links']and 'F31'in new['M21']['candidate_links']
    for q in [N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json']:assert read(q)['latest_profile_first_review']==cp
    assert read(TASK_DIR/'index.json')['status']=='ACTIVE'and read(L/'state.json')['status']=='EVALUATING'
    runs=read(O/'task-run-registration-v70.json')['runs'];assert len(runs)in [5,6] and sum(r['status']=='error'for r in runs)==1 and sum(r['status']=='pass'for r in runs)==len(runs)-1
    assert len(read(O/'framework-candidate-ledger-v70.json')['candidates'])==7
    print(dict(manifest_artifacts_verified=len(mf),candidate_masters=35,new_regimes=60,new_unique_TASKs=2024,composed_groups=72,aux_groups=72,tail_groups=88,seed_copies=104,variants=16,source_files=8,prior_historical_SHA_matches=6,M21_families_retained=18,enriched=6,other_modules_identical=25,reviewed=[2052,284990],pending=[30,1960],checkpoint_status='PROGRESS',whole_modules_closed=0,overall_complete=False))

if __name__=='__main__':main()
