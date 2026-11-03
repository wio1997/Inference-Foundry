"""Ordered helper/virtual-mask/compaction proofs and conditional-source numerical fences."""
from fractions import Fraction as F
from itertools import product,permutations
from pathlib import Path
from copy import deepcopy
import math,argparse,hashlib,json
from finalize_origin_coverage_v9 import O,R,read,write

EPS=F(1,10**9)


def tile_summary(values):
    m=min(values);return m,sum(abs(x-m)<EPS for x in values)


def update(state,tile):
    m,c=state;t,n=tile
    if t<m:return t,n
    return min(m,t),c+(n if abs(t-m)<EPS else 0)


def original_mask(values,pivot,dup,quota,block,active=True,quota_active=True,fill=None):
    if not active:return list(values)
    out=[];kept=0
    for start in range(0,len(values),block):
        tile=values[start:start+block];local=0;tile_kept=0
        for x in tile:
            g=abs(x-dup)<EPS
            local+=g;keepdup=g and local+kept<=quota;tile_kept+=keepdup
            keep=(x>pivot)and(not quota_active or not(g and not keepdup))
            out.append(x if keep else fill)
        if quota_active:kept+=tile_kept
    return out


def virtual_mask(values,pivot,dup,quota,active=True,quota_active=True,fill=None):
    if not active:return list(values)
    out=[];prefix=0
    for x in values:
        g=abs(x-dup)<EPS;prefix+=g
        keep=x>pivot and(not quota_active or not(g and prefix>quota))
        out.append(x if keep else fill)
    return out


def scalar_K_source_model(V,k,one_high=True):
    """Exact-value compressed all0withone1 full-vocab branch,orconstant0 row.

    Samplefirst8192 is0;outlierO=1or0<=k;sourcefull-tertiary search.
    No native approximate Exp or reducer is executed.
    """
    lo=0.;hi=1. if one_high else 0.;k_pivots_num=0;min_larger=math.inf;num_min=0
    I=0
    def stats(pivot):
        pairs=[(0.,V-1),(1.,1)]if one_high else[(0.,V)]
        above=[(v,n)for v,n in pairs if v>pivot];count=sum(n for _,n in above)
        if not above:return count,math.inf,0
        m=min(v for v,_ in above);return count,m,sum(n for v,n in above if abs(v-m)<1e-9)
    while True:
        p0=(hi-lo)/3+lo;p1=(hi-lo)*2/3+lo
        c0,m0,n0=stats(p0);c1,m1,n1=stats(p1);found=False
        if c0>=k and c0-n0<k:pivot=p0;k_pivots_num=c0;min_larger=m0;num_min=n0;found=True
        if c1>=k and c1-n1<k:pivot=p1;k_pivots_num=c1;min_larger=m1;num_min=n1;found=True
        if c1>k:lo=p1
        elif c0>k:lo=p0
        if c0<k:hi=p0
        elif c1<k:hi=p1
        I+=1
        if I>=18 or abs(lo-hi)<1e-9:pivot=(hi+lo)/2;min_larger=m0;num_min=n0;found=True
        if found:break
    maxlog=1. if one_high else 0.
    active=pivot<maxlog and pivot!=-math.inf
    return dict(V=V,k=k,iterations=I,pivot=pivot,active_final_mask=active,kept_values=(1 if active and one_high else V),
      stale_k_count_if_iteration_fallback=k_pivots_num,num_keep_mod_uint32=(num_min-(k_pivots_num-k))%(2**32))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');args=ap.parse_args();checks=[];w=[]
    def check(s,c):assert c,s;checks.append(s)
    d=read(O/'pivot-filter-source-contract-v92.json');p=Path(d['path']);check('sourcehash',hashlib.sha256(p.read_bytes()).hexdigest()==d['sha256'])
    check('manifesthash',hashlib.sha256(Path(d['manifest']).read_bytes()).hexdigest()==d['manifest_sha256'])
    for f in d['functions']:check('body:'+f['name'],p.read_text().splitlines()[f['first_line']-1:f['last_line']]==f['source'])
    check('51if102arms5while15for',d['source_if_sites']==51 and d['source_if_arms']==102 and d['while_loops']==5 and d['for_loops']==15)
    platform=d['NPU_route'];pp=Path(platform['path']);check('platformroutingclausehash',hashlib.sha256(pp.read_bytes()).hexdigest()==platform['sha256'])
    check('platformmanifesthash',hashlib.sha256(Path(platform['manifest']).read_bytes()).hexdigest()==platform['manifest_sha256'])
    for r in platform['parts']:check('platformslice:'+str(r['first_line']),pp.read_text().splitlines()[r['first_line']-1:r['last_line']]==r['source'])
    A=(F(0),1);B=(EPS/2,1);C=(EPS/5,1)
    ab=update(A,B);ba=update(B,A);left=update(update(A,B),C);right=update(A,update(B,C))
    check('fuzzymerge noncommutative',ab[1]==2 and ba[1]==1)
    check('fuzzymerge nonassociative preservingorder',left[1]==3 and right[1]==2)
    w.append(dict(kind='FUZZY_MIN_COUNT_NOT_A_MONOID',ordered_values=['0','eps/2','eps/5'],epsilon='1e-9',AB_count=2,BA_count=1,left_group_count=3,right_group_count=2,native_FP32_probe=False))
    values=[EPS/2,F(0)];whole=tile_summary(values);split=update(tile_summary(values[:1]),tile_summary(values[1:]))
    check('tileboundary changes fuzzycount',whole[1]==2 and split[1]==1)
    w.append(dict(kind='TILE_BOUNDARY_FENCE',input=['eps/2','0'],one_tile_min_count=2,two_tile_ordered_count=1,changing8192to4096_not_unconditionally_equivalent=True))
    ordered_cases=0;separated_cases=0
    for values in product([F(0),EPS/5,EPS/2,2*EPS],repeat=6):
        tiles=[values[i:i+2]for i in range(0,6,2)]
        # Direct original tile-value updates versus stagedtupleproduction and orderedmaster merge.
        direct=(math.inf,0)
        for tile in tiles:
            m=min(tile);n=sum(abs(v-m)<EPS for v in tile)
            direct=update(direct,(m,n))
        partials=[tile_summary(tile)for tile in tiles];master=(math.inf,0)
        for partial in partials:master=update(master,partial)
        assert direct==master;ordered_cases+=1
    for vals in product([F(0),2*EPS,4*EPS],repeat=4):
        expected=tile_summary(vals)
        for perm in permutations(range(4)):
            state=(math.inf,0)
            for i in perm:state=update(state,(vals[i],1))
            assert state==expected;separated_cases+=1
    check('ordered partial preserving originaltiles',ordered_cases==4096)
    check('wellseparated exactties monoid domain',separated_cases==1944)
    virtual_cases=0
    for vals in product([F(-1)*EPS/2,F(0),EPS/2,2*EPS],repeat=4):
        for quota in range(5):
            for block in [1,2,4]:
                for active in [False,True]:
                    for qa in [False,True]:
                        a=original_mask(vals,F(0),F(0),quota,block,active,qa,fill=7)
                        b=virtual_mask(vals,F(0),F(0),quota,active,qa,fill=7)
                        assert a==b;virtual_cases+=1
    check('virtual descriptor quota prefix cases',virtual_cases==15360)
    check('nop doesnot applymaskedvalue tooriginal-inf',virtual_mask([-math.inf,F(1)],F(0),F(0),0,active=False,fill=7)==[-math.inf,F(1)])
    w.append(dict(kind='VIRTUAL_MASK_NEEDS_ACTIVE_FLAG',old_values=['-Infinity',1],fill=7,inactive_result=['-Infinity',1],wrong_blind_inf_pivot_result=[7,1]))
    compaction_cases=0
    for mask in product([False,True],repeat=8):
        for block in [1,2,4,8]:
            old=list(range(1,9));bank=list(old);out=0
            for start in range(0,8,block):
                loaded=bank[start:start+block]
                for off,value in enumerate(loaded):
                    if mask[start+off]:bank[out]=F(value,3);out+=1
            assert bank[:out]==[F(v,3)for v,m in zip(old,mask)if m];compaction_cases+=1
    check('serial inplace Expbank compaction',compaction_cases==1024)
    w.append(dict(kind='PARALLEL_COMPACTION_READ_BEFORE_WRITE',old_E=[10,20,30,40],later_tile_compacts_to_indices=[0,1],earlier_unread_tile_can_be_overwritten=True,requires_barrier_separatebank_or_serialorder=True,actual_race_observed=False))
    model=scalar_K_source_model(129280,2,True);constant=scalar_K_source_model(129280,2,False)
    check('source18cap tiedboundary isnot exactK',model['iterations']==18 and model['active_final_mask']and model['kept_values']==1 and model['pivot']>0)
    check('constantrow finalguard keepsall',not constant['active_final_mask']and constant['kept_values']==129280)
    w.append(dict(kind='SOURCE_PIVOT_FALLBACK_CARDINALITY',single1_after8192zeros_model=model,constant0_row_model=constant,
      source_scalar_count_reference_not_native_fullkernel=True,doesnot_attest_installed_A3_execution=True))
    check('sample heuristic notstability certificate',1000>math.log((2-2**-23)*2**127))
    w.append(dict(kind='STANDALONE_SAMPLE_SHIFT_CAN_OVERFLOW',first8192_logits=0,later_logit=1000,heuristic_max_sample=0,exp_1000_not_finite_FP32=True,
      fullmax_is_computed_but_not_used_as_this_shift=True,native_Exp_reducer_cast_final_output_not_attested=True))
    w.append(dict(kind='POSTK_EXP_INPUTS_NOT_IDENTICAL',third_pass='exp(where(selectedK,L,-inf)-max)',fourth_pass='exp(original_outlier_L-max)',unconditional_maskedExp_reuse_ruledout=True))
    costs=read(O/'pivot-filter-work-and-resource-models-v92.json');check('68workmodels',len(costs['models'])==68)
    check('source8192tiles16',costs['NUM_TILES']==16 and costs['padded_tile_lanes']==131072)
    for m in costs['models']:
        if m['case']=='K_ONLY':
            expect=m['O_scenario']if m['O_scenario']>m['k']else m['V'];assert m['search_values_per_iteration']==expect
            assert m['total_requested_reference_bytes_if_final_mask_active']==m['sample_load_reference_bytes']+m['initial_full_load_reference_bytes']+m['outlier_store_reference_bytes']+m['search_load_reference_bytes']+m['active_final_mask_readwrite_reference_bytes']
        else:
            check('O0strictoutlierbranch excluded:'+str((m['B'],m['full_buffer_fallback'],m['iterations_scenario'])),m['O_scenario']!=0 or m['full_buffer_fallback'])
    prefix=read(O/'pivot-filter-workspace-prefix-and-effects-v92.json');check('sixworkspaceprefixdomains',len(prefix['rows'])==6 and prefix['source_case_current_prefix_written_before_read'])
    graph=read(O/'operator-graph-G7-v92.json');check('expandedG7has2templates',len(graph['templates'])==2)
    t=graph['templates'][1];ids={n['id']for n in t['nodes']};check('child16nodes28edges',len(ids)==16 and len(t['edges'])==28 and all(e['producer']in ids and e['consumer']in ids for e in t['edges']))
    audit=read(O/'operator-graph-coverage-audit-v92.json');check('partial724/1187',audit['node_rows']==724 and audit['edge_rows']==1187 and not audit['F35_full_variant_mapping_complete'])
    for k,path in audit['graph_catalog'].items():check('graphhash:'+k,hashlib.sha256(Path(path).read_bytes()).hexdigest()==audit['graph_sha256'][k])
    if args.preflight:
        for r in read(O/'checkpoint-manifest-v91.json')['artifacts']:
            path=Path(r['path']);check('pretransition:'+path.name,path.stat().st_size==r['bytes']and hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256'])
        write(O/'pivot-filter-math-validation-v92.json',dict(ordered_tile_summary_cases=ordered_cases,separated_exacttie_cases=separated_cases,
          virtual_mask_cases=virtual_cases,serial_compaction_cases=compaction_cases,witnesses=w,native_Torch_NPU_execution=False,overall_complete=False))
    else:
        cp=read(O/'REVIEW-CHECKPOINT-v92.json');check('checkpointprogress',cp['status']=='PROGRESS'and not cp['overall_complete'])
        op=read(O/'operator-analysis-coverage-ledger-v92.json');check('profileunchanged',(op['scoped_conditional_review_regimes'],op['scoped_conditional_review_executions'],op['remaining_regime_reconciliation_count'],op['remaining_execution_reconciliation_count'])==(2073,285598,9,1352))
        check('uncertaintyunchanged',op['unchanged_global_uncertainty']==read(O/'operator-analysis-coverage-ledger-v91.json')['unchanged_global_uncertainty'])
        bm=read(O/'module-phase-branch-matrix-v92.json');check('354branches24semantic',sum(len(m['branches'])for m in bm['modules'])==354 and sum(m.get('branch_schema')=='SEMANTIC_BRANCHES_V1'for m in bm['modules'])==24)
        rc=read(O/'candidate-relation-coverage-v92.json');check('115/24/480unchanged',rc['scoped_conditional_pairs']==115 and rc['higher_order_groups']==24 and rc['remaining_pair_relations']==480)
        check('canonicalgraphpointer',read(R/'fusion-scan.json')==read(O/'fusion-scan-proposed-v92.json'))
        for r in read(O/'candidate-record-index-v92.json')['model_candidates']:check('masterindex:'+r['id'],hashlib.sha256(Path(r['master_record']).read_bytes()).hexdigest()==r['master_sha256'])
        for r in read(O/'candidate-profile-ledger-v92.json')['candidates']:check('candidateconditional:'+r['id'],not r['candidate_closed']and not r['legal_fusion_proved'])
        for r in read(O/'checkpoint-manifest-v92.json')['artifacts']:
            path=Path(r['path']);check('manifest:'+path.name,path.stat().st_size==r['bytes']and hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256'])
    out=dict(status='PASS',checks=len(checks),ordered_cases=ordered_cases,separated_cases=separated_cases,virtual_mask_cases=virtual_cases,compaction_cases=compaction_cases,preflight=args.preflight,new_native_server_or_model_work=False,overall_complete=False)
    if args.preflight:write(O/'pivot-filter-analysis-preflight-v92.json',out)
    print(json.dumps(out))


if __name__=='__main__':main()
