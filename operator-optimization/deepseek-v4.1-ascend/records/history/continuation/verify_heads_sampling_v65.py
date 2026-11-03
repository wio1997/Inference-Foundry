"""Read-only verification of v65 persistence, operator identity and protected history."""
from pathlib import Path
from collections import Counter
import hashlib,json
from finalize_origin_coverage_v9 import N,O,R,W,T,L,read

def main():
    cp=read(O/'REVIEW-CHECKPOINT-v65.json')
    assert cp['status']=='PROGRESS' and cp['overall_complete'] is False and cp['whole_modules_closed']==0
    artifacts=read(O/'checkpoint-manifest-v65.json')['artifacts']
    for a in artifacts:
        p=Path(a['path']);b=p.read_bytes()
        assert len(b)==a['bytes'] and hashlib.sha256(b).hexdigest()==a['sha256'],p
    index=read(O/'candidate-record-index-v65.json')['model_candidates']
    assert len(index)==35
    for row in index:
        assert hashlib.sha256(Path(row['master_record']).read_bytes()).hexdigest()==row['master_sha256'],row['id']
    cov=read(O/'operator-analysis-coverage-ledger-v65.json')
    assert len(cov['regimes'])==2082 and sum(r['executions']for r in cov['regimes'])==286950
    reviewed=[r for r in cov['regimes']if r['state'].startswith('SCOPED_')]
    pending=[r for r in cov['regimes']if r['state'].startswith('REGIME_')]
    assert len(reviewed)==998 and sum(r['executions']for r in reviewed)==189903
    assert len(pending)==1084 and sum(r['executions']for r in pending)==97047
    assert len(reviewed)+len(pending)==2082 and cov['geometry_only_other_role_regimes']==0
    assert cov['unchanged_global_uncertainty']=={'ambiguous_module_assignment_executions':3136,'missing_PMU':849,'phase_unbound_graphs':16}
    assert cov['explicit_profile_insufficient_regimes']==3 and cov['explicit_profile_insufficient_executions']==960
    p=read(O/'heads-sampling-regime-models-v65.json')
    keys=[tuple(k)for r in p['regimes']for k in r['native_TASK_keys']]
    assert len(keys)==len(set(keys))==9792
    g=read(O/'heads-sampling-native-chains-v65.json')
    M15keys={(x['database_index'],t['TASK_rowid']) for x in g['target_tails'] for t in x['stages']}
    assert len(M15keys)==624
    m15all={tuple(k)for r in p['regimes']if r['module_candidates']==['M15']for k in r['native_TASK_keys']}
    assert M15keys<=m15all and len(m15all-M15keys)==416
    m16all={tuple(k)for r in p['regimes']if r['module_candidates']==['M16']for k in r['native_TASK_keys']}
    m16keys={(x['database_index'],x[k]['TASK_rowid'])for x in g['base_heads']for k in ['GEMM','gather_transpose']}
    m16keys|={(x['database_index'],x['MarkovGEMM']['TASK_rowid'])for x in g['Markov_steps']}
    assert m16all==m16keys and len(m16all)==904
    markovkeys=set()
    for x in g['Markov_steps']:
        assert Counter(t['operator']for t in x['other_stages'])=={'GatherV2':1,'Slice':3,'Add':1,'ViewCopy':2,'ArgMaxV2':1,'Cast':1}
        markovkeys|={(x['database_index'],t['TASK_rowid'])for t in x['other_stages']}
        markovkeys.add((x['database_index'],x['MarkovGEMM']['TASK_rowid']))
    assert len(markovkeys)==5200
    statekeys={(x['database_index'],t['TASK_rowid'])for x in g['state_corrections']for t in x['stages']}
    stateall={tuple(k)for r in p['regimes']if r['module_candidates']==['M24']for k in r['native_TASK_keys']}
    assert statekeys==stateall and len(statekeys)==1224
    inv=read(O/'M15-M16-M17-M20-M24-operator-branch-inventory-v65.json')
    new={m['id']:m for m in read(O/'module-phase-branch-matrix-v65.json')['modules']}
    for mid,rows in inv['modules'].items():
        assert new[mid]['branches']==rows and new[mid]['branch_schema']=='SEMANTIC_BRANCHES_V1'
        assert Path(new[mid]['legacy_branch_history']).exists() and new[mid]['closed'] is False
        hist=read(Path(new[mid]['legacy_branch_history']))
        old=next(m for m in read(O/'module-phase-branch-matrix-v64.json')['modules']if m['id']==mid)
        assert hist['module']==old
    protected=read(O/'heads-sampling-protected-module-state-v65.json')['modules']
    assert len(protected)==21 and all(new[mid]==old for mid,old in protected.items())
    fw64=read(O/'framework-candidate-ledger-v64.json')['candidates']
    fw65=read(O/'framework-candidate-ledger-v65.json')['candidates']
    assert fw64==fw65 and len(fw65)==7
    for r in read(O/'heads-sampling-operator-contract-v65.json')['source_files']:
        assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256']
    models=read(O/'heads-sampling-fusion-costs-v65.json')
    for r in models['target_tail_models']:
        assert r['source_post_snapshot_value_bytes']-r['fused_collapse_round_norm_value_bytes']==76*r['N']*5120
    for r in models['state_correction_models']:
        assert sum(r['stages_value_bytes_per_row'].values())==215
        assert r['interface_budget_bytes']==175*r['N']
    assert sum(r['groups']for r in models['separate_draft_helper_collapse_models'])==104
    assert models['Markov_memoization_and_certified_pruning']['full_BF16_transition_table_bytes']==33426636800
    assert len(read(O/'heads-sampling-fusion-variants-v65.json')['variants'])==28
    for q in [N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json']:
        d=read(q);assert d['latest_profile_first_review']==cp and d['overall_complete'] is False
    for q in [L/'state.json',T/'index.json']:
        d=read(q);assert d['status']in ['EVALUATING','ACTIVE'] and d['next_action']==cp['next_action']
    reg=read(O/'task-run-registration-v65.json')
    assert len(reg['runs'])==5 and sum(r['status']=='error'for r in reg['runs'])==1
    assert all(Path(r['run_dir']).exists()for r in reg['runs'])
    print(json.dumps(dict(manifest_artifacts_verified=len(artifacts),master_candidate_hashes_verified=35,
        unique_reviewed_TASKs=9792,Markov_10TASK_chain_identities=5200,state_identities=1224,
        semantic_families=49,other_modules_identical=21,framework_candidates_unchanged=7,
        reviewed_total=[998,189903],pending_total=[1084,97047],runs=5,
        checkpoint_status='PROGRESS',whole_modules_closed=0,overall_complete=False),ensure_ascii=False))

if __name__=='__main__':main()
