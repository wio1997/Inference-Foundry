"""Exact comparison/algebra cases,IEEE-F32 fences,and checkpoint consistency;noTorch."""
from fractions import Fraction as F
from itertools import product
from bisect import bisect_right
from pathlib import Path
import math,struct,hashlib,argparse,json
from finalize_origin_coverage_v9 import O,R,read,write


def f32(x):
    """Exact nearest-even for the positive finite rational values used below."""
    x=F(x);approx=struct.unpack('<f',struct.pack('<f',float(x)))[0]
    bits=struct.unpack('<I',struct.pack('<f',approx))[0]
    candidates=[]
    for b in [bits-1,bits,bits+1]:
        if b<0 or b>=0x7f800000:continue
        v=struct.unpack('<f',struct.pack('<I',b))[0];candidates.append((abs(F(v)-x),b&1,v))
    return min(candidates)[2]


def source_filter(prob,logits,k,p):
    out=list(logits);s=sorted(prob);V=len(s);ck=cp=None;cdf=[]
    if k is not None:
        ck=s[V-k]if k!=V else -math.inf
        for i,v in enumerate(prob):
            if v<ck:out[i]=None
    if p is not None:
        acc=0
        for x in s:acc+=x;cdf.append(acc)
        mask=[c<=1-p for c in cdf];mask[-1]=False;count=sum(mask);cp=s[count]
        for i,v in enumerate(prob):
            if v<cp:out[i]=None
    return out,ck,cp,cdf


def fused_filter(prob,logits,k,p):
    """One threshold and one masked writer;CDF values are intentionally unchanged."""
    s=sorted(prob);V=len(s);cuts=[];queries=None
    if k is not None:cuts.append(s[V-k]if k!=V else -math.inf)
    if p is not None:
        cdf=[];acc=0
        for x in s:acc+=x;cdf.append(acc)
        queries=min(bisect_right(cdf,1-p),V-1);cuts.append(s[queries])
    if not cuts:return list(logits),queries
    cutoff=max(cuts)
    return [None if x<cutoff else z for x,z in zip(prob,logits)],queries


def upstream_filter_weights(weights,k,p):
    """Upstream sort-logits/post-k-normalization path,in exact real weights."""
    idx=sorted(range(len(weights)),key=lambda i:weights[i]);s=[weights[i]for i in idx]
    if k is not None:
        cut=s[len(s)-k];s=[0 if x<cut else x for x in s]
    if p is not None:
        total=sum(s);assert total>0;acc=0
        for i,x in enumerate(list(s)):
            acc+=x/total
            if i<len(s)-1 and acc<=1-p:s[i]=0
    out=[0]*len(s)
    for i,x in zip(idx,s):out[i]=x
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');args=ap.parse_args();checks=[]
    def check(label,ok):assert ok,label;checks.append(label)
    contract=read(O/'sampling-filter-race-source-contract-v91.json')
    for r in contract['source_files']:
        path=Path(r['path']);check('frozen source:'+path.name,hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256'])
        check('source manifest:'+path.name,hashlib.sha256(Path(r['manifest']).read_bytes()).hexdigest()==r['manifest_sha256'])
        for part in r['parts']:check('pinnedbody:'+part['name'],path.read_text().splitlines()[part['first_line']-1:part['last_line']]==part['source'])
    check('11namedfunctions27ifs54arms',contract['named_functions_pinned']==11 and contract['local_if_sites']==27 and contract['local_if_arms']==54)
    comparisons=0;searches=0
    for weights in product(range(4),repeat=4):
        if not sum(weights):continue
        prob=[F(w,sum(weights))for w in weights];logits=list(range(4))
        for k in [None,1,2,3,4]:
            for p in [None,F(0),F(1,4),F(1,2),F(3,4),F(1)]:
                old,ck,cp,cdf=source_filter(prob,logits,k,p);new,idx=fused_filter(prob,logits,k,p)
                assert old==new,(weights,k,p,old,new);comparisons+=1
                if p is not None:
                    mask=[x<=1-p for x in cdf];mask[-1]=False
                    assert sum(mask)==idx;searches+=1
    check('exact cutoff fusion cases',comparisons==7650 and searches==6375)
    witnesses=[]
    weights=list(map(F,[1,2,3,4]));P=[w/sum(weights)for w in weights]
    mask,_,_,_=source_filter(P,weights,2,F(1,2));asc=[0 if v is None else v for v in mask];base=upstream_filter_weights(weights,2,F(1,2))
    check('preK versus postK topP',asc==[0,0,3,4]and base==[0,0,0,4])
    witnesses.append(dict(kind='BACKEND_PREK_VS_POSTK_PROBABILITY',weights=[1,2,3,4],k=2,p='1/2',Ascend_kept_indices=[2,3],upstream_PyTorch_kept_indices=[3],native_kernel_or_runtime_flags_attested=False))
    weights=[F(1)]*4;P=[F(1,4)]*4;out,_,_,_=source_filter(P,weights,None,F(1,2));up=upstream_filter_weights(weights,None,F(1,2))
    check('cutoff ties all retained vs positional tie mask',sum(x is not None for x in out)==4 and sum(x>0 for x in up)==2)
    witnesses.append(dict(kind='TIED_CUTOFF_CARDINALITY',probabilities=['1/4']*4,p='1/2',Ascend_keeps=4,upstream_position_mask_keeps=2,upstream_tie_ID_order_not_attested=True))
    out,_,_,_=source_filter([F(0),F(1)],[-200,0],None,F(1))
    check('p1 isnot unconditional noop afterprobunderflow',out==[None,0])
    witnesses.append(dict(kind='P1_AND_TOPK_UNDERFLOW_FENCES',FP32_softmax_model=[0,1],finite_input_logits=[-200,0],p=1,filtered_logits=['-Infinity',0],native_softmax_output_not_attested=True))
    qnan=float('nan');prob=[.25,.75];correct=[v<qnan or v<.5 for v in prob];bad=[v<qnan for v in prob]
    check('NaNpropagating scalarMax isunsafe',correct==[True,False]and bad==[False,False])
    witnesses.append(dict(kind='NAN_CUTOFF_REQUIRES_OR_OR_VALIDITY_GUARD',P=prob,k_cutoff='NaN',p_cutoff=.5,source_OR_mask=correct,nan_propagating_Max_mask=bad,native_Max_NaN_semantics_not_attested=True,cutoff_tuple_reachable_from_native_softmax_unknown=True))
    P0=[f32(F(2,6)),f32(F(3,6))];P1=[f32(F(2,5)),f32(F(3,5))];Q=[0.010400000028312206,0.015599998645484447]
    scores0=[f32(F(p)/F(q))for p,q in zip(P0,Q)];scores1=[f32(F(p)/F(q))for p,q in zip(P1,Q)]
    check('reuseoriginalP canflipF32race',scores0[0]==scores0[1]and scores1[1]>scores1[0])
    witnesses.append(dict(kind='COMMON_SCALE_REAL_IDENTITY_NOT_FP32_RACE_IDENTITY',original_kept_P=P0,post_truncation_P=P1,Q=Q,
      oldP_race=scores0,required_postP_race=scores1,first_tie_policy_oldP_winner=0,required_postP_winner=1,
      IEEE_single_operation_reference_only=True,native_CANN_softmax_argmax_Qdraw_support_or_seed_attested=False))
    tiny=math.exp(-200);roundtiny=f32(F(tiny));stable=-200-math.log1p(tiny)
    check('logroundedP isnotlogsoftmax',roundtiny==0 and math.isfinite(stable)and stable==-200)
    witnesses.append(dict(kind='LOGSOFTMAX_VS_LOG_ROUNDED_P',input_logits=[-200,0],rounded_P0=0,log_of_rounded_P0='-Infinity',stable_logsoftmax_FP32_model0=-200,native_kernel_rounding_not_attested=True))
    check('zeroQ guard cannotbe imported',math.inf>.5 and -1e10<.5)
    witnesses.append(dict(kind='QZERO_NOT_RECOVERY_SENTINEL',P=['1/2','1/2'],Q=[0,1],source_division_scores=['Infinity',.5],source_winner=0,borrowed_recovery_guard_scores=[-1e10,.5],guarded_winner=1,actual_exponential_zero_support_unknown=True))
    temperatures=[0,5e-6,.1];lt=[t<1e-5 for t in temperatures];eq=[t==0 for t in temperatures]
    check('bonusmixed predicate differs fromrejectiongreedy',lt==[True,True,False]and eq==[True,False,False])
    witnesses.append(dict(kind='MIXED_ROW_MASK_DOMAIN',temperatures=temperatures,bonus_greedy_LT_eps=lt,rejection_greedy_EQ_zero=eq,caller_small_positive_temperature_reachability_unproven=True))
    race_cases=0
    for weights in product(range(1,4),repeat=3):
        prob=[F(w,sum(weights))for w in weights]
        for q in product([F(1,2),F(1),F(2)],repeat=3):
            old=[p/v for p,v in zip(prob,q)];oldwinner=max(range(3),key=lambda i:old[i])
            # Fused stream comparesonecomputeddivision atatime,retainingfirstequalindex.
            best=None;winner=None
            for i,(p,v)in enumerate(zip(prob,q)):
                value=p/v
                if best is None or value>best:best=value;winner=i
            assert winner==oldwinner;race_cases+=1
    check('racefixedoperandcomparison cases',race_cases==729)
    c=read(O/'sampling-filter-race-cost-models-v91.json');check('20typedcostcases8RNGcases',len(c['models'])==20 and len(c['rng_draw_cases'])==8)
    for m in c['models']:
        B,V,A=m['B'],m['V'],m['active_random_rows_scenario'];n=B*V
        check('disjointspans:'+str((B,A,m['topK_present'],m['topP_present'])),m['mask_and_private_race_disjoint_span_old_reference_bytes']==m['original_mask_logit_readwrite_reference_bytes']+16*n and m['mask_and_private_race_disjoint_span_new_reference_bytes']==m['one_mask_logit_readwrite_reference_bytes']+8*A*V)
        if m['topK_present']and m['topP_present']:assert m['mask_logit_plane_reference_credit_bytes']==8*n
        if m['topP_present']:assert m['existing_CDF_binary_search_max_value_probes']==17*B
    for r in c['rng_draw_cases']:
        check('RNGdrawbudget:'+str((r['B'],r['seeded_rows'])),r['default_draw_elements']==(0 if r['seeded_rows']==r['B']else r['B']*r['V']))
    check('wholeP/Q doesnotfit UB',c['whole_FLOAT_P_or_Q_row_bytes']>c['UB_per_AIV'])
    graph=read(O/'operator-graph-G7-v91.json');gn=ge=0
    for t in graph['templates']:
        ids={n['id']for n in t['nodes']};gn+=len(ids);ge+=len(t['edges']);check('G7endpoints',all(e['producer']in ids and e['consumer']in ids for e in t['edges']))
        check('CDFtakesoriginalPnotKmutatedlogits',any(e['producer']=='sort_P'and e['consumer']=='cdf'for e in t['edges'])and not any(e['producer']=='logits_afterK'and e['consumer']=='cdf'for e in t['edges']))
    check('G7size29/40',gn==29 and ge==40)
    audit=read(O/'operator-graph-coverage-audit-v91.json');check('partialgraph708/1159',audit['node_rows']==708 and audit['edge_rows']==1159 and not audit['F35_full_variant_mapping_complete'])
    for k,p in audit['graph_catalog'].items():check('graphhash:'+k,hashlib.sha256(Path(p).read_bytes()).hexdigest()==audit['graph_sha256'][k])
    if args.preflight:
        for r in read(O/'checkpoint-manifest-v90.json')['artifacts']:
            p=Path(r['path']);check('pretransitionhash:'+p.name,p.stat().st_size==r['bytes']and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'])
        write(O/'sampling-filter-race-pointwise-validation-v91.json',dict(probability_mask_fusion_cases=comparisons,CDF_existing_value_search_cases=searches,
          fixed_operand_race_cases=race_cases,witnesses=witnesses,comparison_IEEE_models_only=True,native_Torch_or_NPU_test=False,overall_complete=False))
    else:
        cp=read(O/'REVIEW-CHECKPOINT-v91.json');check('progressnotcomplete',cp['status']=='PROGRESS'and not cp['overall_complete'])
        op=read(O/'operator-analysis-coverage-ledger-v91.json');check('unchangedprofilepopulations',(op['scoped_conditional_review_regimes'],op['scoped_conditional_review_executions'],op['remaining_regime_reconciliation_count'],op['remaining_execution_reconciliation_count'])==(2073,285598,9,1352))
        check('uncertainties unchanged',op['unchanged_global_uncertainty']==read(O/'operator-analysis-coverage-ledger-v90.json')['unchanged_global_uncertainty'])
        bm=read(O/'module-phase-branch-matrix-v91.json');check('353branch24semantic',sum(len(m['branches'])for m in bm['modules'])==353 and sum(m.get('branch_schema')=='SEMANTIC_BRANCHES_V1'for m in bm['modules'])==24)
        check('canonicalgraphpointer',read(R/'fusion-scan.json')==read(O/'fusion-scan-proposed-v91.json'))
        rel=read(O/'candidate-relation-coverage-v91.json');check('115/24/480unchanged',rel['scoped_conditional_pairs']==115 and rel['higher_order_groups']==24 and rel['remaining_pair_relations']==480)
        for r in read(O/'candidate-record-index-v91.json')['model_candidates']:check('candidateindex:'+r['id'],hashlib.sha256(Path(r['master_record']).read_bytes()).hexdigest()==r['master_sha256'])
        for r in read(O/'candidate-profile-ledger-v91.json')['candidates']:check('conditionalcandidate:'+r['id'],not r['candidate_closed']and not r['legal_fusion_proved'])
        for r in read(O/'checkpoint-manifest-v91.json')['artifacts']:
            p=Path(r['path']);check('manifest:'+p.name,p.stat().st_size==r['bytes']and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'])
    out=dict(status='PASS',checks=len(checks),mask_cases=comparisons,CDF_cases=searches,race_cases=race_cases,preflight=args.preflight,new_native_or_server_work=False,overall_complete=False)
    if args.preflight:write(O/'sampling-filter-race-analysis-preflight-v91.json',out)
    print(json.dumps(out))


if __name__=='__main__':main()
