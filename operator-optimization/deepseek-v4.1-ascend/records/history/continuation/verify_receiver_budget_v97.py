"""Capacity/clipping identities,workspace max and shared-memory invariants;no device tests."""
from pathlib import Path
import argparse,hashlib,json,math
from finalize_origin_coverage_v9 import O,R,read,write

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');args=ap.parse_args();checks=[]
    def check(label,ok):assert ok,label;checks.append(label)
    src=read(O/'receiver-budget-operator-source-contract-v97.json')
    for row in src['parts']:
        p=Path(row['path']);ls=p.read_text().splitlines();check('source:'+p.name,hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'])
        for s in row['segments']:check('lines:'+p.name+':'+str(s['first_line']),ls[s['first_line']-1:s['last_line']]==s['source'])
    check('genericpointerunusednotallGMremoved',len(src['generic_full_load_workspace_mentions'])==2 and src['source_default_not_current_native_value'])
    ev=read(O/'receiver-capacity-evidence-status-v97.json');check('4472cachedMC2tasks32DBs',len(ev['current_export_projection'])==32 and ev['current_MC2_TASKs']==4472)
    for row in ev['current_export_projection']:check('cachednative:'+str(row['database_index']),hashlib.sha256(Path(row['cache_path']).read_bytes()).hexdigest()==row['cache_sha256'])
    check('currentCandsamecallboundsstillunknown',ev['actual_current_C']is None and ev['actual_same_call_all_rank_n_k_or_receiver_A']is None)
    for row in ev['historical']:check('historicalsha:'+Path(row['path']).name,hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256']and 'Historical169only'in row['scope'])
    count=0
    for B in range(129):
        for old in range(1,129):
            new=min(old,max(1,B));guard=min(old,((max(1,B)+254)//128)*128)
            for A in range(B+1):
                assert min(A,old)==min(A,new)==min(A,guard);count+=1
    check('capacitytransformallAupToB',count==1073280)
    geometry=0
    for ranks in range(1,9):
        for E in range(1,9):
            for k in range(1,ranks*E+1):
                chosen=list(range(min(k,E)))+list(range(E,E+max(0,k-E)))
                assert len(chosen)==k and len(set(chosen))==k and max(chosen)<ranks*E
                assert sum(e<E for e in chosen)==min(k,E);geometry+=1
    check('tightindexdomaingeometryconstructivecases',geometry==1296)
    models=read(O/'receiver-capacity-and-workspace-models-v97.json');check('fiveprofilegeometrycases',len(models['cases'])==5)
    Bexpected={(3,6):288,(3,3):144,(2,3):96,(816,6):78336,(834,6):80064}
    variants=0
    for row in models['cases']:
        M,k,E=row['M'],row['k'],row['local_experts'];B=Bexpected[(M,k)]
        assert row['receiver_assignment_geometry_upper']==B and row['assumed_same_call_all_sources_n_le_M']
        assert row['C_logic']==min(65536,max(1,B))and row['C_guard_reference']==min(65536,((B+254)//128)*128)
        for q in row['variants']:
            C=q['receiver_capacity'];constant=4*k*((M+255)//256*256)+8*16*16*E+64*(E+32)
            coc=constant+(15360+8)*C;init=16*1024**2+32*M*k+320
            if q['discarded_generic_init_scratch']:assert row['full_load_source_case'];init=0
            assert q['coc_workspace_bytes']==coc and q['requested_workspace_bytes']==16*1024**2+max(coc,init)
            assert not q['native_allocation_live_or_reserved_bytes_attested'];variants+=1
    check('24workspacevariantsonealiasbankeach',variants==24)
    j=read(O/'receiver-workspace-shared-live-budget-v97.json');s=j['max_workspace_synergy']
    check('defaultguardandjointrequestvalues',(s['old'],s['capacity_only'],s['unused_init_only'],s['both'])==(1023993344,33555328,1023993344,24704512))
    check('maxmakesrealjointsynergy',s['old_minus_init']==0 and s['extra_saving_from_joint']==8850816 and s['old_minus_both']==999288832)
    p=j['persistent_bank_value_reference'];check('WKVandC2payloadreferences',p['WKV_result_and_ID_value_bytes_per_entry']==51296 and p['total']==64*51296+393216)
    x=j['synthetic_liveness_counterexample'];check('decodecreditnotpeakheadroom',x['new_phase_peak']==x['old_phase_peak']and x['peak_credit']==0 and x['feasible_only_decode']and not x['feasible_across_both_phases'])
    packet=read(O/'receiver-budget-composition-clauses-v97.json');oldpairs={frozenset([r['a'],r['b']])for r in read(O/'candidate-relation-clauses-v96.json')['clauses']}
    check('twoexistingblankpairsnowconcretecontracts',len(packet['clauses'])==2 and all(frozenset([r['a'],r['b']])not in oldpairs and not r['all_runtime_gates_proved']for r in packet['clauses']))
    if args.preflight:
        for row in read(O/'checkpoint-manifest-v96.json')['artifacts']:
            p=Path(row['path']);check('pretransition:'+p.name,p.stat().st_size==row['bytes']and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'])
    else:
        cp=read(O/'REVIEW-CHECKPOINT-v97.json');check('PROGRESSnotgoalcomplete',cp['status']=='PROGRESS'and not cp['overall_complete'])
        rc=read(O/'candidate-relation-coverage-v97.json');check('124/25/471',rc['scoped_conditional_pairs']==124 and rc['higher_order_groups']==25 and rc['remaining_pair_relations']==471)
        rel=read(O/'candidate-relation-clauses-v97.json');g=next(g for g in rel['higher_order_groups']if g['id']=='HG85-SHARED-MEMORY-BUDGET');check('existingbudgetgroupextended',set(g['members'])=={'F21','F25','F26','F31','F34'})
        bm=read(O/'module-phase-branch-matrix-v97.json');check('359records24semantic',sum(len(m['branches'])for m in bm['modules'])==359 and sum(m.get('branch_schema')=='SEMANTIC_BRANCHES_V1'for m in bm['modules'])==24)
        op=read(O/'operator-analysis-coverage-ledger-v97.json');prev=read(O/'operator-analysis-coverage-ledger-v96.json')
        for f in ['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']:check('nativeunchanged:'+f,op[f]==prev[f])
        req=read(O/'whole-scope-requirement-audit-v97.json');check('all9requirementsopen',len(req['requirements'])==9 and not req['overall_complete']and req['whole_modules_closed']==0)
        check('noallmodulelowestMFUclaim',next(r for r in req['requirements']if r['id']=='LOWEST_MODULE_UTILIZATION')['state']=='NOT_ESTABLISHED')
        for r in read(O/'candidate-record-index-v97.json')['model_candidates']:check('master:'+r['id'],hashlib.sha256(Path(r['master_record']).read_bytes()).hexdigest()==r['master_sha256'])
        for row in read(O/'checkpoint-manifest-v97.json')['artifacts']:
            p=Path(row['path']);check('manifest:'+p.name,p.stat().st_size==row['bytes']and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'])
    result=dict(status='PASS',checks=len(checks),preflight=args.preflight,capacity_identity_models=count,geometry_models=geometry,workspace_models=variants,
      no_native_Torch_or_model_execution=True,overall_complete=False)
    if args.preflight:write(O/'receiver-budget-analysis-preflight-v97.json',result)
    print(json.dumps(result))

if __name__=='__main__':main()
