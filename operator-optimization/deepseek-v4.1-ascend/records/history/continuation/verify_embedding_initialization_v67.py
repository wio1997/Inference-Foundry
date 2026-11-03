"""Read-only v67 identity, logical-cost, gap-retention and persistence verification."""
from pathlib import Path
from collections import Counter
import hashlib,json
from finalize_origin_coverage_v9 import N,O,R,W,T,L,read

def main():
    cp=read(O/'REVIEW-CHECKPOINT-v67.json');assert cp['status']=='PROGRESS'and cp['overall_complete']is False and cp['whole_modules_closed']==0
    manifest=read(O/'checkpoint-manifest-v67.json')['artifacts']
    for a in manifest:
        p=Path(a['path']);b=p.read_bytes();assert len(b)==a['bytes']and hashlib.sha256(b).hexdigest()==a['sha256'],p
    masters=read(O/'candidate-record-index-v67.json')['model_candidates'];assert len(masters)==35
    for r in masters:assert hashlib.sha256(Path(r['master_record']).read_bytes()).hexdigest()==r['master_sha256']
    p=read(O/'embedding-initialization-regime-models-v67.json');assert len(p['regimes'])==236
    allkeys=[tuple(k)for r in p['regimes']for k in r['native_TASK_keys']]
    assert len(allkeys)==len(set(allkeys))==7616
    done=[r for r in p['regimes']if r['consumer_fusion_review_complete_in_this_turn']]
    unknown=[r for r in p['regimes']if not r['consumer_fusion_review_complete_in_this_turn']]
    assert (len(done),sum(r['executions']for r in done))==(230,7224)
    assert (len(unknown),sum(r['executions']for r in unknown))==(6,392)
    g=read(O/'embedding-initialization-native-chains-v67.json')
    expected={'embedding_groups':(272,14),'target_HC_initializations':(104,5),'draft_initializations':(104,15),
        'sampled_ID_groups':(72,4),'draft_ID_groups':(72,6),'position_groups':(88,4)}
    for k,(groups,stages)in expected.items():
        assert len(g[k])==groups and all(len(x['stages'])==stages for x in g[k])
        keys=[(x['database_index'],t['TASK_rowid'])for x in g[k]for t in x['stages']]
        assert len(keys)==len(set(keys))==groups*stages and set(keys)<=set(allkeys)
    modulekeys=lambda mid:{tuple(k)for r in p['regimes']if r['module_candidates']==[mid]for k in r['native_TASK_keys']}
    for mid,key in [('M06','target_HC_initializations'),('M19','draft_initializations')]:
        assert modulekeys(mid)=={(x['database_index'],t['TASK_rowid'])for x in g[key]for t in x['stages']}
    embedkeys={(x['database_index'],t['TASK_rowid'])for x in g['embedding_groups']for t in x['stages']}
    assert len(modulekeys('M05')-embedkeys)==24 and len(modulekeys('M05'))==3832
    cost=read(O/'embedding-initialization-fusion-costs-v67.json')
    for r in cost['draft_initialization_models']:
        n,a,l=r['N'],r['A_padded'],r['local_rows'];assert a==l*8
        assert r['current_global_Tile_plus_HC_Pad_value_bytes']==(18*n+8*a)*5120
        assert r['delayed_Tile_after_2D_Pad_SP_value_bytes']==(2*n+2*a+10*l)*5120
        assert sum(x['valid_owned_rows']for x in r['rank_models'])==n
    assert next(r for r in cost['draft_initialization_models']if r['N']==20)['delayed_Tile_interface_budget_bytes']==2222080
    for r in cost['embedding_models']:
        n=r['N'];assert all(x['source_14TASK_value_bytes']==8*n*5120+150*n for x in r['cases'])
        assert [x['fixed_AG_to_AR_payload_ratio']for x in r['owner_protocol_cases']]==[.5,4.]
        assert r['extra_unique_invalid_weight_rows_at_most']==1 and r['extra_unique_invalid_weight_bytes_at_most']==10240
    for r in cost['position_gather_cast_add_models']:assert r['interface_budget_bytes']==32*r['N']
    for r in cost['initial_premix_and_virtual_HC_models']:assert r['materialization_interface_budget_bytes']==12*r['N_local']+8
    variants=read(O/'embedding-initialization-fusion-variants-v67.json')['variants']
    assert len(variants)==22 and sum(r['decision']=='UNRESOLVED_EVIDENCE_RETAINED'for r in variants)==1
    for r in read(O/'embedding-initialization-operator-contract-v67.json')['source_files']:
        assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256']
    assert read(O/'BOOL-prefix-candidate-review-v67.json')['excluded_candidate'].startswith('shared_attention_state.reset()')
    cov=read(O/'operator-analysis-coverage-ledger-v67.json');assert len(cov['regimes'])==2082
    rev=[r for r in cov['regimes']if r['state'].startswith('SCOPED_')];pending=[r for r in cov['regimes']if r['state'].startswith('REGIME_')]
    assert (len(rev),sum(r['executions']for r in rev))==(1245,208655)
    assert (len(pending),sum(r['executions']for r in pending))==(837,78295)
    assert len(rev)+len(pending)==2082 and sum(r['executions']for r in cov['regimes'])==286950
    assert Counter('|'.join(r['module_candidates'])for r in pending)=={'M21':828,'M19':4,'M12':3,'M05|M21':2}
    mem=[r for r in pending if r['operator']=='MemSet'];assert (len(mem),sum(r['executions']for r in mem))==(7,1272)
    bo=[r for r in pending if r['module_candidates']==['M05','M21']];assert (len(bo),sum(r['executions']for r in bo))==(2,80)
    assert cov['explicit_profile_insufficient_regimes']==7 and cov['explicit_profile_insufficient_executions']==1272
    assert cov['unchanged_global_uncertainty']=={'ambiguous_module_assignment_executions':3136,'missing_PMU':849,'phase_unbound_graphs':16}
    new={m['id']:m for m in read(O/'module-phase-branch-matrix-v67.json')['modules']}
    old={m['id']:m for m in read(O/'module-phase-branch-matrix-v66.json')['modules']}
    inv=read(O/'M04-M05-M06-M19-M10-operator-branch-inventory-v67.json')['modules']
    assert sum(map(len,inv.values()))==61
    for mid,rows in inv.items():
        assert new[mid]['branches']==rows and new[mid]['branch_schema']=='SEMANTIC_BRANCHES_V1'
        assert read(Path(new[mid]['legacy_branch_history']))['module']==old[mid]
    assert new['M10']['branches'][:-2]==old['M10']['branches']and len(new['M10']['branches'])==11
    assert sum('PENDING'in r['state']for r in new['M05']['branches'])==1
    assert sum('PENDING'in r['state']for r in new['M19']['branches'])==1
    protected=read(O/'initialization-protected-module-state-v67.json')['modules']
    assert len(protected)==21 and all(new[mid]==before for mid,before in protected.items())
    assert read(O/'framework-candidate-ledger-v66.json')['candidates']==read(O/'framework-candidate-ledger-v67.json')['candidates']
    for q in [N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json']:
        d=read(q);assert d['latest_profile_first_review']==cp and d['overall_complete']is False
    for q in [L/'state.json',T/'index.json']:
        d=read(q);assert d['status']in ['EVALUATING','ACTIVE']and d['next_action']==cp['next_action']
    runs=read(O/'task-run-registration-v67.json')['runs'];assert len(runs)==5 and all(r['status']=='pass'for r in runs)
    print(json.dumps(dict(manifest_artifacts_verified=len(manifest),candidate_master_hashes_verified=35,source_hashes_verified=7,
        unique_selected_TASKs=7616,new_review=[230,7224],new_pending_retained=[6,392],embedding_groups=272,target_and_draft_init=[104,104],
        ID_position_groups=[72,72,88],variant_records=22,semantic_families=61,other_modules_identical=21,
        reviewed=[1245,208655],pending=[837,78295],total_MemSet_pending=[7,1272],BOOL_pending=[2,80],
        checkpoint_status='PROGRESS',whole_modules_closed=0,overall_complete=False),ensure_ascii=False))

if __name__=='__main__':main()
