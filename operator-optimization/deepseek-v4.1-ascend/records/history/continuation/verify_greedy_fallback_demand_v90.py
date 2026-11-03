"""Independent fixed-input output/demand proofs and source/ledger/manifest checks;noTorch."""
from itertools import product
from copy import deepcopy
from pathlib import Path
from collections import Counter
import hashlib,argparse,json
from finalize_origin_coverage_v9 import O,N,R,read,write


def source_general(out,n,draft,target,bonus,S,greedy,synthetic,U,rates):
    """Flatten/map/scatter/min/masked-gather semantics of the pinned general body."""
    out=deepcopy(out);B=len(n);g=[True]*B if greedy is None else greedy
    starts=[];cur=0
    for nr in n:starts.append(cur);cur+=nr
    req=[r for r,nr in enumerate(n)for _ in range(nr)];pos=[i-starts[r]for i,r in enumerate(req)]
    accept=[(U[i]<rates[pos[i]]and draft[i]>=0)if synthetic else draft[i]==target[i]for i in range(cur)]
    if S==0:first=[0]*B
    else:
        pm=[[-1]*S for _ in n];mm=[[False]*S for _ in n]
        for i,(r,j)in enumerate(zip(req,pos)):pm[r][j]=j;mm[r][j]=not accept[i]
        mp=[[pm[r][j]if mm[r][j]else 2*S for j in range(S)]for r in range(B)]
        first=[min(row)for row in mp];first=[n[r]if j==2*S else j for r,j in enumerate(first)]
    copylen=[min(j+1,nr)for j,nr in zip(first,n)]
    cp=[(draft[i]if accept[i]else target[i])if synthetic else target[i]for i in range(cur)]
    for r in range(B):
        for j in range(S+1):
            if j<copylen[r]and g[r]:out[r][j]=cp[starts[r]+j]
    for r in range(B):
        if g[r]and first[r]>=n[r]:out[r][n[r]]=bonus[r]
    return out


def source_special(out,draft,target,bonus,synthetic,U,rates):
    """Literal columns0/1 helper,including oldcol1 retention and untouched suffix."""
    out=deepcopy(out)
    for r in range(len(out)):
        accept=(U[r]<rates[0]and draft[r]>=0)if synthetic else draft[r]==target[r]
        out[r][0]=(draft[r]if accept else target[r])if synthetic else target[r]
        out[r][1]=bonus[r]if accept else out[r][1]
    return out


def demand_scan(out,n,draft,target,bonus,S,greedy,synthetic,U,rates):
    """Per-request stream;target argmax is requested lazily via an explicit provider."""
    out=deepcopy(out);queries=[];g=[True]*len(n)if greedy is None else greedy;start=0
    def q(i):queries.append(i);return target[i]
    for r,nr in enumerate(n):
        if g[r]:
            rejected=False
            for j in range(nr):
                i=start+j
                if synthetic:
                    accepted=U[i]<rates[j]and draft[i]>=0
                    if accepted:out[r][j]=draft[i]
                    else:out[r][j]=q(i);rejected=True;break
                else:
                    value=q(i);out[r][j]=value
                    if draft[i]!=value:rejected=True;break
            if not rejected:out[r][nr]=bonus[r]
        start+=nr
    return out,queries


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');args=ap.parse_args();checks=[]
    def check(s,c):assert c,s;checks.append(s)
    contract=read(O/'greedy-fallback-source-contract-v90.json');p=Path(contract['path'])
    check('frozen body identity',hashlib.sha256(p.read_bytes()).hexdigest()==contract['sha256'])
    for f in contract['functions']:check('function slice:'+f['name'],p.read_text().splitlines()[f['first_line']-1:f['last_line']]==f['source'])
    check('2bodies6ifs12arms',len(contract['functions'])==2 and contract['local_if_sites']==6 and contract['local_if_arms']==12)
    check('actualcount specialization doesnot proveS1',not contract['caller_specialization']['requires_output_capacity_equals_one']and 'max_spec_len'not in contract['caller_specialization']['test'])
    totals=Counter();examples=[]
    for S in range(6):
        counts=list(product(range(min(S,4)+1),repeat=2))
        if S==5:counts +=[(5,0),(0,5),(5,1),(1,5)]
        for ns in counts:
            B=len(ns);Nt=sum(ns)
            for bits in product([False,True],repeat=Nt):
                for synthetic in [False,True]:
                    for invalid in [False,True]:
                        draft=[-1 if invalid and i%3==0 else 100+i for i in range(Nt)]
                        target=[(100+i if bits[i]else 200+i)if not synthetic else 200+i for i in range(Nt)]
                        U=[.25 if bit else .75 for bit in bits];rates=[.5]*max(1,S);bonus=[900+r for r in range(B)]
                        for greedy in [None,*map(list,product([False,True],repeat=B))]:
                            for fill in [-1,99]:
                                old=[[fill]*(S+1)for _ in ns]
                                ref=source_general(old,ns,draft,target,bonus,S,greedy,synthetic,U,rates)
                                new,queries=demand_scan(old,ns,draft,target,bonus,S,greedy,synthetic,U,rates)
                                assert ref==new,(S,ns,bits,synthetic,invalid,greedy,fill,ref,new)
                                if synthetic:assert len(queries)<=sum(bool(n)and(greedy is None or greedy[r])for r,n in enumerate(ns))
                                else:assert len(queries)<=sum(n for r,n in enumerate(ns)if greedy is None or greedy[r])
                                assert len(queries)==len(set(queries))
                                totals['general_vs_demand_cases']+=1
                                if all(n==1 for n in ns)and greedy is None:
                                    assert source_special(old,draft,target,bonus,synthetic,U,rates)==ref
                                    totals['special_vs_general_cases']+=1
    check('independent stream proof exercised',totals['general_vs_demand_cases']>10000 and totals['special_vs_general_cases']>0)
    case=dict(S=5,n=[1],draft=[7],target=[7],bonus=[9],greedy=None,synthetic=False,U=[.25],rates=[.5]*5)
    correct=source_general([[-1]*6],**case);wrong=source_special([[99]*6],case['draft'],case['target'],case['bonus'],False,case['U'],case['rates'])
    check('S5 actualone suffix counterexample',correct==[[7,9,-1,-1,-1,-1]]and wrong==[[7,9,99,99,99,99]])
    examples.append(dict(kind='FUNCTION_NAME_NOT_CAPACITY_GUARD',counts=[1],S=5,correct=correct,wrong_drop_fill=wrong,
      runtime_occurrence_attested=False,changed_N_shape_branch_not_current_profile_case=True))
    synthetic_case=dict(n=[2],draft=[3,4],target=[8,9],bonus=[7],S=2,greedy=None,synthetic=True,U=[.25,.75],rates=[.5,.5])
    result,qs=demand_scan([[-1]*3],**synthetic_case)
    check('synthetic accepted draft vs target',result==[[3,9,-1]]and qs==[1])
    examples.append(dict(kind='SYNTHETIC_ACCEPTED_DRAFT_NOT_TARGET',output=result,argmax_query_rows=qs,wrong_target_prefix=[8,9,-1]))
    allaccept=deepcopy(synthetic_case);allaccept['U']=[.25,.25]
    result,qs=demand_scan([[-1]*3],**allaccept)
    check('syntheticallaccept zero argmax rows',result==[[3,4,7]]and qs==[])
    examples.append(dict(kind='SYNTHETIC_ALL_ACCEPT',output=result,argmax_query_rows=qs,baseline_argmax_rows=2,all_original_rng_draws_retained=True))
    poison=source_general([[99,99,99]],**synthetic_case)
    placeholder=source_general([[-1,-1,-1]],**synthetic_case)
    check('generic old bank cannot forceplaceholder',poison==[[3,9,99]]and poison!=placeholder)
    examples.append(dict(kind='GENERIC_HELPER_RETAINS_OLD_BANK',oldbank_result=poison,caller_placeholder_result=placeholder))
    models=read(O/'greedy-fallback-demand-cost-models-v90.json')
    check('18costmodels3geometries',len(models['models'])==18 and models['profile_capacity_geometries']==3)
    for m in models['models']:
        B,S,V=m['B'],m['S'],m['V'];Nvalid=m['N_valid_scenario']
        check('scorework:'+m['scenario']+str(B)+str(m['T_capacity']),m['baseline_FLOAT_argmax_score_reference_bytes']==4*Nvalid*V and m['synthetic_argmax_score_reference_bytes']==4*m['synthetic_demand_argmax_rows']*V)
        check('fullwritecredit:'+m['scenario']+str(B)+str(m['T_capacity']),m['original_fill_plus_helper_store_reference_bytes']-m['transformed_fullwriter_store_reference_bytes']==m['fullwrite_output_store_credit_bytes'])
        check('capacity vs changed input shape:'+m['scenario']+str(B)+str(m['T_capacity']),m['scenario_input_N_matches_recorded_sampler_capacity']==(Nvalid==m['T_capacity'])and m['changed_N_scenarios_are_unobserved_shape_branches']==(Nvalid!=m['T_capacity']))
        if all(m['greedy_scenario']):
            check('joint target baseline counted once:'+m['scenario']+str(B)+str(m['T_capacity']),
              m['guarded_Index_Cast_clone_argmax_22NV_reference_bytes']-m['pure_ID_joint_synthetic_2AV_reference_bytes']==m['pure_ID_joint_synthetic_credit_reference_bytes'])
        else:check('mixed rows do not inherit pureIDglobal budget',m['pure_ID_joint_synthetic_credit_reference_bytes']is None)
        if m['specialized_actual_one_count_not_S1']:check('specialized tails:'+str(B)+str(m['T_capacity']),m['source_specialized_untouched_suffix_cells']==B*(S-1)and m['original_helper_store_reference_bytes']==8*B)
    check('oneAIV whole-row exclusions only',models['FLOAT_full_vocab_row_bytes']==517120>models['Vector_UB_per_AIV']and models['BF16_full_vocab_row_bytes']==258560>models['Vector_UB_per_AIV'])
    check('UB capacity source',models['Vector_UB_per_AIV']==read(O/models['capacity_reference'])['documented_physical_bytes']['UB_per_AIV'])
    ex=next(m for m in models['models']if m['B']==2 and m['T_capacity']==10 and m['scenario']=='full_capacity_all_accept')
    wrong_budget=20*10*129280+ex['baseline_FLOAT_argmax_score_reference_bytes']
    check('oldfusionplusargmax credit overcount witness',wrong_budget>ex['guarded_Index_Cast_clone_argmax_22NV_reference_bytes'])
    examples.append(dict(kind='ONE_JOINT_BASELINE_NO_ADDED_CREDITS',wrong_added_credit=wrong_budget,
      entire_guarded_target_baseline=ex['guarded_Index_Cast_clone_argmax_22NV_reference_bytes'],
      correct_joint_reference_credit=ex['pure_ID_joint_synthetic_credit_reference_bytes'],physical_HBM_or_speedup_proof=False))
    graph=read(O/'operator-graph-G6-v90.json');gn=ge=0
    for t in graph['templates']:
        ids={n['id']for n in t['nodes']};check('graph node identities:'+t['id'],len(ids)==len(t['nodes']))
        check('graph endpoints:'+t['id'],all(e['producer']in ids and e['consumer']in ids for e in t['edges']))
        gn+=len(ids);ge+=len(t['edges'])
        if t['id']=='SAMPLER_GREEDY_GENERAL_BODY':
            check('synthetic mismatch independent oftargetargmax',all(e['condition']=='not synthetic'for e in t['edges']if e['producer']=='target_argmax'and e['consumer']=='mismatch'))
        if t['id']=='SAMPLER_GREEDY_ACTUAL_N1_BODY':
            check('synthetic specialized acceptance independent oftargetargmax',all(e['condition']=='not synthetic'for e in t['edges']if e['producer']=='target_argmax'and e['consumer']=='accept'))
    check('47newsource nodes69edges',gn==47 and ge==69)
    audit=read(O/'operator-graph-coverage-audit-v90.json');check('finite ledger679/1119',audit['node_rows']==679 and audit['edge_rows']==1119 and not audit['F35_full_variant_mapping_complete'])
    for k,path in audit['graph_catalog'].items():check('graphdigest:'+k,hashlib.sha256(Path(path).read_bytes()).hexdigest()==audit['graph_sha256'][k])
    oldn=read(O/'operator-graph-node-coverage-v88.json')['rows'];olde=read(O/'operator-graph-edge-coverage-v88.json')['rows']
    check('oldG1G5rows unchanged',read(O/'operator-graph-node-coverage-v90.json')['rows'][:len(oldn)]==oldn and read(O/'operator-graph-edge-coverage-v90.json')['rows'][:len(olde)]==olde)
    if args.preflight:
        trans=read(O/'persistence-transition-contract-v90.json')
        check('transition helper source identity',hashlib.sha256(Path(trans['parent']).read_bytes()).hexdigest()==trans['parent_sha256']and hashlib.sha256(Path(trans['derived']).read_bytes()).hexdigest()==trans['derived_sha256'])
        code=Path(trans['derived']).read_text();check('shared pointer mutation after predecessor validation',code.index('for a in read(O/')<code.index('for path, payload in (additional_shared_writes or {}).items():')<code.index('model=deepcopy(read(O/'))
        for a in read(O/'checkpoint-manifest-v89.json')['artifacts']:
            q=Path(a['path']);check('pre-transitionv89digest:'+q.name,q.stat().st_size==a['bytes']and hashlib.sha256(q.read_bytes()).hexdigest()==a['sha256'])
        write(O/'greedy-fallback-pointwise-validation-v90.json',dict(fixed_input_case_counts=dict(totals),examples=examples,
          source_algebra_only=True,native_Torch_or_NPU_test=False,new_source_bodies_analyzed=2,
          all_comparators_native_rounding_and_aliases_proved=False,overall_complete=False))
    else:
        cp=read(O/'REVIEW-CHECKPOINT-v90.json');check('checkpoint progress',cp['status']=='PROGRESS'and not cp['overall_complete'])
        op=read(O/'operator-analysis-coverage-ledger-v90.json');check('profilepopulations unchanged',(op['scoped_conditional_review_regimes'],op['scoped_conditional_review_executions'],op['remaining_regime_reconciliation_count'],op['remaining_execution_reconciliation_count'])==(2073,285598,9,1352))
        bm=read(O/'module-phase-branch-matrix-v90.json');check('352branch24semantic',sum(len(m['branches'])for m in bm['modules'])==352 and sum(m.get('branch_schema')=='SEMANTIC_BRANCHES_V1'for m in bm['modules'])==24)
        check('same global uncertainty',op['unchanged_global_uncertainty']==read(O/'operator-analysis-coverage-ledger-v89.json')['unchanged_global_uncertainty'])
        rc=read(O/'candidate-relation-coverage-v90.json');check('same115/24/480',rc['scoped_conditional_pairs']==115 and rc['higher_order_groups']==24 and rc['remaining_pair_relations']==480)
        check('canonical graph pointer successor matches proposal',read(R/'fusion-scan.json')==read(O/'fusion-scan-proposed-v90.json'))
        for row in read(O/'candidate-record-index-v90.json')['model_candidates']:
            check('master index:'+row['id'],hashlib.sha256(Path(row['master_record']).read_bytes()).hexdigest()==row['master_sha256'])
        for row in read(O/'candidate-profile-ledger-v90.json')['candidates']:
            check('conditional candidate:'+row['id'],not row['candidate_closed']and not row['legal_fusion_proved'])
        for a in read(O/'checkpoint-manifest-v90.json')['artifacts']:
            q=Path(a['path']);check('manifest:'+q.name,q.stat().st_size==a['bytes']and hashlib.sha256(q.read_bytes()).hexdigest()==a['sha256'])
    result=dict(status='PASS',checks=len(checks),fixed_input_case_counts=dict(totals),preflight=args.preflight,new_server_model_Torch_NPU_or_native_work=False,overall_complete=False)
    if args.preflight:write(O/'greedy-fallback-analysis-preflight-v90.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
