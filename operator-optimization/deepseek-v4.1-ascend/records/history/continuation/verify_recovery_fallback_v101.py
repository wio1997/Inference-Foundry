"""Independent finite arithmetic, branch identity, rowplan and graph coverage checks;no Torch/device work."""
from pathlib import Path
from fractions import Fraction as F
from itertools import product,combinations
from collections import Counter
import argparse,hashlib,json,struct
from finalize_origin_coverage_v9 import N,O,R,read,write

def f32(x):return struct.unpack('>f',struct.pack('>f',x))[0]
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');a=parser.parse_args();checks=[]
    def check(name,value):
        assert value,name
        checks.append(name)
    for r in read(O/'recovery-fallback-inputs-v101.json')['inputs']:
        p=Path(r['path']);check('input:'+p.name,p.stat().st_size==r['bytes']and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'])
    src=read(O/'recovery-fallback-source-and-applicability-v101.json');check('conditionalnotnativepresence',src['observed_greedy_chains']==72 and src['exact_saved_name_audit']['sample_recovered_tokens']==0 and src['exact_saved_name_audit']['rejection_random_sample']==0)
    for r in src['parts']:
        p=Path(r['path']);lines=p.read_text().splitlines();check('source:'+p.name+str(len(lines)),hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'])
        for s in r['segments']:check('segment:'+str(s['first_line']),lines[s['first_line']-1:s['last_line']]==s['source'])
    p=Path(src['parts'][0]['path']).read_text();t=Path(src['parts'][1]['path']).read_text();uniform=Path(src['parts'][2]['path']).read_text()
    check('actualbackendformulas','pi = min(pi * target_prob / draft_prob, 1.0)'in t and'pi = pi.clamp(max=1.0)'in p and'pi = torch.cumprod(pi, dim=-1)'in p)
    check('notupstreamrecoveryborrow','q_ptr'in t and'new_p = prob / q'in t and'dtype=torch.float64'in uniform)
    w=read(O/'recovery-backend-nonidentity-v101.json');b=w['block_clip_counterexample'];ps=list(map(F,b['target_selected']));ds=list(map(F,b['draft_selected']));us=list(map(F,b['uniforms']))
    tp=F(1);fp=F(1);u=F(1);lt=lf=-1;seqT=[];seqF=[]
    for i,(P,D,U)in enumerate(zip(ps,ds,us)):
        tp=min(tp*P/D,1);fp*=min(P/D,1);u*=U;seqT.append(str(tp));seqF.append(str(fp))
        if tp>=u:lt=i
        if fp>=u:lf=i
    check('blockclipcounter',(lt,lf)==(b['Triton_last_accepted'],b['PyTorch_last_accepted'])==(2,-1)and seqT==b['Triton_recursive_pi']and seqF==b['PyTorch_ratio_capped_pi'])
    # Ratiofactors0.5,2,1remainexactwithFLOAT32operandmodel;thresholdsfarfromboundaries.
    tp32=fp32=1.0;u32=1.0;u64=1.0;lt32=lf32=-1
    for i,(P,D,U)in enumerate(zip(ps,ds,us)):
        pf,qf,uf=map(lambda x:f32(float(x)),[P,D,U]);tp32=min(f32(f32(tp32*pf)/qf),1.0);fp32=f32(fp32*min(f32(pf/qf),1.0));u32=f32(u32*uf);u64*=float(U)
        if tp32>=u32:lt32=i
        if fp32>=u64:lf32=i
    check('clipcounterrobusttooperandrounds',(lt32,lf32)==(2,-1))
    ud=w['uniform_dtype_counterexample'];check('ordinaryuniformprecisioncounter',f32(ud['uniform_FLOAT64'])==0.5 and 0.5<ud['uniform_FLOAT64']and ud['Triton_accept']and not ud['PyTorch_accept'])
    opt=w['optional_block_recovery_counterexample'];P=list(map(F,opt['target']));D=list(map(F,opt['draft']));pi=F(opt['prefix_pi']);arg=lambda x:max(range(len(x)),key=lambda i:x[i])
    check('optionalweightedresidualnotordinary',arg([max(x-y,0)for x,y in zip(P,D)])==0 and arg([max(pi*x-y,0)for x,y in zip(P,D)])==1 and opt['current_root_does_not_pass_use_block_verify'])
    mapping=read(O/'recovery-owner-plan-and-state-reuse-v101.json');cases=0;positions=0
    for B in range(1,6):
        for ns in product(range(4),repeat=B):
            cu=[];s=0
            for n in ns:s+=n;cu.append(s)
            for tail in [-1,0,1,3]:
                T=max(0,s+tail);source=[];search=[];fill=[0]*T;start=0
                for i,end in enumerate(cu):
                    for tok in range(start,min(end,T)):fill[tok]=i
                    start=end
                for tok in range(T):
                    matches=[i for i in range(B)if(0 if i==0 else cu[i-1])<=tok<cu[i]];source.append(matches[0]if matches else 0)
                    # Independentlinearupper-boundreference,includingB→request0tail.
                    found=next((i for i,end in enumerate(cu)if end>tok),B);search.append(found if found<B else 0)
                assert source==search==fill;cases+=1;positions+=T
    check('5456rowmapcases',cases==mapping['exhaustive_cases']==5456 and positions==mapping['checked_token_positions'])
    check('rightbiaszeroandtail',mapping['zero_draft_witness']['source_owner']==1 and mapping['boundary_and_tail_witness']['source_owners_for_T5']==[0,2,2,0,0])
    check('nonmonotoneguardneeded',mapping['nonmonotone_counterexample']['interval_reference']!=mapping['nonmonotone_counterexample']['binary_search_not_legal'])
    scan=read(O/'recovery-prefix-scan-and-demand-v101.json');assoc=0
    for rs in product([F(0),F(1,4),F(1),F(2),F(4)],repeat=3):
        for x in [F(0),F(1,4),F(1,2),F(1)]:
            seq=x
            for ratio in rs:seq=min(seq*ratio,1)
            aa=rs[2]*rs[1]*rs[0];bb=min(F(1),rs[2],rs[2]*rs[1]);assert min(aa*x,bb)==seq
        assoc+=1
    check('125exactrealscanmodels',assoc==scan['associativity_cases']==125)
    fp=scan['FP32_reassociation_counterexample'];x=1.0;seq=[]
    for P,D in zip(fp['P'],fp['D']):x=min(f32(f32(x*P)/D),1.0);seq.append(x)
    check('IEEEunderflowfalsifiesdropinscan',seq==fp['sequential_pi']and seq[-1]==0 and fp['reassociated_last_pi']==2**-100)
    u=1.0;la=-1
    for i,U in enumerate(fp['uniforms']):
        u=f32(u*f32(U))
        if seq[i]>=u:la=i
    check('underflowchangesacceptance',la==fp['sequential_last_accepted']==-1 and fp['reassociated_last_pi']>=u and fp['reassociated_last_accepted']==2)
    cost=read(O/'recovery-fallback-work-and-interface-models-v101.json');check('80referencegraphs',len(cost['models'])==80)
    for r in cost['models']:
        B,S,T,V,K=r['B'],r['S'],r['T_capacity_reference'],r['V'],r['selected_recovery_rows_reference'];assert T==B*S
        old=48*T*V;new=0 if r['private_plane_fusion']else 48*K*V
        assert old==r['standalone_six_FLOAT_private_plane_write_read_reference_bytes']and old-new==r['conditional_removed_private_interfaces_bytes']
        assert r['original_Q_BxV_FLOAT_RNG_bank_bytes']==4*B*V and r['original_all_uniform_and_seeded_Q_draws_preserved']
        assert r['P_and_D_and_Q_consumer_load_value_reference_bytes']==(8 if r['NO_DRAFT_PROBS']else 12)*K*V
    check('interfaceunionnotadditive',not cost['actual_runtime_flag_or_native_speedup_or_numeric_physical_ceiling_proved'])
    tail=cost['padded_tail_counterexample'];check('invalidtailmustbe-neginf',max(range(4),key=lambda i:([-1e10]*3+[0])[i])==3 and max(range(4),key=lambda i:([-1e10]*3+[float('-inf')])[i])==0 and tail['source_argmax']==0)
    graph=read(O/'operator-graph-G8-v101.json');audit=read(O/'operator-graph-coverage-audit-v101.json');check('5G8templates',len(graph['templates'])==5 and not graph['complete'])
    nrows=[r for r in read(O/'operator-graph-node-coverage-v101.json')['rows']if r['graph']=='G8'];erows=[r for r in read(O/'operator-graph-edge-coverage-v101.json')['rows']if r['graph']=='G8'];check('70nodes88edges',len(nrows)==70 and len(erows)==88)
    for template in graph['templates']:
        ids=[n['id']for n in template['nodes']];check('uniqueIDs:'+template['id'],len(set(ids))==len(ids))
        neigh={i:set()for i in ids}
        for edge in template['edges']:
            assert edge['producer']in neigh and edge['consumer']in neigh;neigh[edge['producer']].add(edge['consumer']);neigh[edge['consumer']].add(edge['producer'])
        nc={i:Counter()for i in ids};ec={i:Counter()for i in range(len(template['edges']))}
        for k in [2,3,4]:
            cnt=0
            for ss in combinations(ids,k):
                chosen=set(ss);seen={ss[0]};front=[ss[0]]
                while front:
                    for v in neigh[front.pop()]&chosen-seen:seen.add(v);front.append(v)
                if seen!=chosen:continue
                cnt+=1
                for v in ss:nc[v][str(k)]+=1
                for i,e in enumerate(template['edges']):
                    if e['producer']in chosen and e['consumer']in chosen:ec[i][str(k)]+=1
            assert cnt==audit['G8_connected_sets'][template['id']][str(k)]
        for row in nrows:
            if row['template']==template['id']:assert dict(nc[row['node_id']])==row['connected_set_counts']
        for row in erows:
            if row['template']==template['id']:assert dict(ec[row['edge_index']])==row['connected_set_counts']
        check('bruteconnectedsets:'+template['id'],True)
    check('813/1297partialsourcecount',(audit['node_rows'],audit['edge_rows'],audit['current_structural_pair_hints'])==(813,1297,46)and not audit['overall_complete'])
    for key,path in audit['graph_catalog'].items():check('catalog:'+key,hashlib.sha256(Path(path).read_bytes()).hexdigest()==audit['graph_sha256'][key])
    if a.preflight:manifest=read(O/'checkpoint-manifest-v100.json')
    else:
        cp=read(O/'REVIEW-CHECKPOINT-v101.json');check('progressnotgoalcomplete',cp['status']=='PROGRESS'and not cp['overall_complete'])
        rel=read(O/'candidate-relation-coverage-v101.json');check('131/25/464unchanged',(rel['scoped_conditional_pairs'],rel['higher_order_groups'],rel['remaining_pair_relations'])==(131,25,464))
        bm=read(O/'module-phase-branch-matrix-v101.json');check('362branches24semantic',sum(len(m['branches'])for m in bm['modules'])==362 and sum(m.get('branch_schema')=='SEMANTIC_BRANCHES_V1'for m in bm['modules'])==24)
        op=read(O/'operator-analysis-coverage-ledger-v101.json');prev=read(O/'operator-analysis-coverage-ledger-v100.json')
        for key in ['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']:check('nativeunchanged:'+key,op[key]==prev[key])
        req=read(O/'whole-scope-requirement-audit-v101.json');check('9requirementsremainopen',len(req['requirements'])==9 and not req['overall_complete'])
        check('allmoduleMFUstillnotestablished',next(r for r in req['requirements']if r['id']=='LOWEST_MODULE_UTILIZATION')['state']=='NOT_ESTABLISHED')
        for row in read(O/'candidate-record-index-v101.json')['model_candidates']:check('master:'+row['id'],hashlib.sha256(Path(row['master_record']).read_bytes()).hexdigest()==row['master_sha256'])
        manifest=read(O/'checkpoint-manifest-v101.json')
    for row in manifest['artifacts']:
        p=Path(row['path']);check('manifest:'+p.name,p.stat().st_size==row['bytes']and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'])
    result=dict(status='PASS',checks=len(checks),preflight=a.preflight,map_cases=cases,monoid_cases=assoc,G8_nodes=70,G8_edges=88,new_runtime_executions=0,overall_complete=False)
    if a.preflight:write(O/'recovery-fallback-analysis-preflight-v101.json',result)
    print(json.dumps(result))

if __name__=='__main__':main()
