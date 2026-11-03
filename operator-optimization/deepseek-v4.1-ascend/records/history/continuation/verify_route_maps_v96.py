"""Independent permutation, source work and state/manifest checks for route-map analysis."""
from pathlib import Path
from itertools import permutations
import argparse,hashlib,json,math,random
from finalize_origin_coverage_v9 import O,R,read,write

def direct_inverse(sigma):
    out=[None]*len(sigma)
    for dst,src in enumerate(sigma):assert out[src]is None;out[src]=dst
    return out

def block_touch(lo,hi,offset):
    return len({(offset+i)//32 for i in range(4*lo,4*hi)})*32

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');args=ap.parse_args();checks=[]
    def check(label,ok):assert ok,label;checks.append(label)
    src=read(O/'route-map-operator-source-contract-v96.json')
    for row in src['parts']:
        p=Path(row['path']);ls=p.read_text().splitlines();check('source:'+p.name,hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'])
        for seg in row['segments']:check('lines:'+p.name+':'+str(seg['first_line']),ls[seg['first_line']-1:seg['last_line']]==seg['source'])
    check('noELForgainclaim',not src['loaded_native_binary_or_new_scatter_performance_attested'])
    doc=read(O/'DataCopyPad-primary-evidence-v96.json');p=Path(doc['path'])
    check('primaryAPIcontextbodyandhash',doc['usable_document_body']and hashlib.sha256(p.read_bytes()).hexdigest()==doc['sha256']and b'DataCopyParams'in p.read_bytes()and b'blockLen'in p.read_bytes())
    count=0
    for n in range(9):
        for sigma in permutations(range(n)):
            # Reference the unique-key second sort,not another inversion implementation.
            ref=[dst for key,dst in sorted((src,dst)for dst,src in enumerate(sigma))]
            inv=direct_inverse(sigma);assert inv==ref and all(inv[sigma[j]]==j for j in range(n));count+=1
    check('46234allpermutationsT0to8',count==46234)
    rng=random.Random(9601);partial=0
    for n in [6,9,18]:
        for _ in range(1000):
            sigma=list(range(n));rng.shuffle(sigma);inv=direct_inverse(sigma)
            for budget in [1,2,3,6,18,40,48]:
                p=(n+budget-1)//budget;segments=[(a,min(a+p,n))for a in range(0,n,p)]
                for c,(lo,hi)in enumerate(segments):
                    positions=range(n)if c==0 else range(lo,hi)
                    local={src:dst for dst,src in enumerate(sigma)if src in positions}
                    assert all(local[x]==inv[x]for x in positions)
                partial+=1
    check('21000partialconsumerprefixmodels',partial==21000)
    m=read(O/'route-map-source-work-and-interfaces-v96.json')
    check('3512full960excluded',m['full_load_executions']==3512 and m['excluded_non_full_load_executions']==960)
    check('120240sourceSortAPItotal60120inverseonly',m['current_source_weighted_Sort_call_reference']==120240 and m['inverse_only_weighted_Sort_call_reference']==60120)
    by={(r['M'],r['topK']):r for r in m['cases']}
    for r in m['cases']:
        T,C=r['T'],r['current_partitions'];assert T<32 and r['high_level_full_load_Sort_calls_reference']==2*C
        assert r['conditional_shared_mapping_GM_value_difference']==4*T*(C-2)
        assert r['consumer_demand_local_inverse_UB_writes']==4*(2*T-1)
    check('exactsmallcohortworkandvalueformulas',True)
    old=read(O/'route-quant-tiling-and-work-models-v95.json');sigs={(r['M'],r['topK']):r for r in old['cases']}
    for r in m['block_exposure_cases']:
        prior=sigs[(r['M'],r['topK'])];T=r['M']*r['topK'];C=prior['used_quant_AIV_partitions'];ao=r['ID_base_offset_mod32'];bo=r['map_base_offset_mod32']
        ids=block_touch(0,T,ao);out=block_touch(0,T,bo)
        a=C*ids+out;b=ids+out+sum(block_touch(lo,hi,bo)for lo,hi in prior['core_assignment_intervals'])
        assert r['baseline_copy_block_exposure_bytes']==a and r['shared_copy_block_exposure_bytes']==b and r['difference']==a-b
    check('192blockphasecases',len(m['block_exposure_cases'])==192)
    small=next(r for r in m['block_exposure_cases']if(r['M'],r['topK'],r['ID_base_offset_mod32'],r['map_base_offset_mod32'])==(2,3,0,0))
    check('M2logicalsavebut32Btouchincrease',by[(2,3)]['conditional_shared_mapping_GM_value_difference']==96 and small['difference']==-32)
    check('24jointcasesnotadditive',len(m['joint_partition_and_map_cases'])==24)
    def joint(**kw):return next(r for r in m['joint_partition_and_map_cases']if all(r[k]==v for k,v in kw.items()))
    baseline=joint(M=3,topK=6,token_aligned=False,shared_first_permutation=False,local_inverse_direct=False)['high_level_Sort_calls_reference']
    row=joint(M=3,topK=6,token_aligned=True,shared_first_permutation=False,local_inverse_direct=False)['high_level_Sort_calls_reference']
    inv=joint(M=3,topK=6,token_aligned=False,shared_first_permutation=False,local_inverse_direct=True)['high_level_Sort_calls_reference']
    both=joint(M=3,topK=6,token_aligned=True,shared_first_permutation=False,local_inverse_direct=True)['high_level_Sort_calls_reference']
    check('36to6or18or3donotsumcredits',(baseline,row,inv,both)==(36,6,18,3))
    sem=read(O/'route-map-semantic-and-ceiling-contract-v96.json');w=sem['mask_witness']
    check('postmaskchangesinverse',w['inverse_before_mask']==[0,1]and w['inverse_after_mask']==[1,0])
    cap=sem['capacity_witness'];check('clippingmakesarbitrarytiesobservable',cap['outputs_after_inverse_first']!=cap['outputs_after_inverse_reordered'])
    eq=sem['equal_histogram_not_equal_inverse_witness'];check('countscacheisinversemapscope',sorted(eq['effective_ids_a'])==sorted(eq['effective_ids_b'])and eq['inverse_a']!=eq['inverse_b'])
    hist_cases=0
    for n in range(9):
        for tail in range(5):
            for k in range(1,7):
                for E in [2,3,384]:
                    ids=[(j*13+n)%E for j in range(n*k)];effective=ids+[E]*(tail*k)
                    assert all(effective.count(e)==ids.count(e)+(tail*k if e==E else 0)for e in set(effective)|set(ids)|{E})
                    hist_cases+=1
    check('810knowninactivehistogramprefixcases',hist_cases==810)
    check('symbolicmapcutnotuniversalphysical',sem['fixed_representation_symbolic_cut']['rate_upper']is None and not sem['whole_MC2_or_native_legality_closed'])
    nf=read(O/'route-map-native-frontiers-v96.json');fs=nf['frontiers'];check('3200/2560/9600oldTASKs',len(fs)==3200 and nf['root_bound']==2560 and nf['reused_unique_native_TASKs']==9600)
    for r in fs:assert r['topK_output_shapes']==[[3,6],[3,6],[3,384]]and r['topK_output_types']=='FLOAT;INT32;FLOAT'and not r['inner_sort_timing_or_postmask_values_or_native_permutation_proved']
    check('allTopKdescriptorplanesretained',True)
    if args.preflight:
        for row in read(O/'checkpoint-manifest-v95.json')['artifacts']:
            p=Path(row['path']);check('pretransition:'+p.name,p.stat().st_size==row['bytes']and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'])
    else:
        cp=read(O/'REVIEW-CHECKPOINT-v96.json');check('PROGRESSnotcompletion',cp['status']=='PROGRESS'and not cp['overall_complete'])
        cov=read(O/'candidate-relation-coverage-v96.json');check('122/25/473notinflated',cov['scoped_conditional_pairs']==122 and cov['higher_order_groups']==25 and cov['remaining_pair_relations']==473)
        bm=read(O/'module-phase-branch-matrix-v96.json');check('358records24semantic',sum(len(x['branches'])for x in bm['modules'])==358 and sum(x.get('branch_schema')=='SEMANTIC_BRANCHES_V1'for x in bm['modules'])==24)
        op=read(O/'operator-analysis-coverage-ledger-v96.json');prev=read(O/'operator-analysis-coverage-ledger-v95.json')
        for f in ['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']:check('nativeunchanged:'+f,op[f]==prev[f])
        req=read(O/'whole-scope-requirement-audit-v96.json');check('all9requirementsstillopen',len(req['requirements'])==9 and not req['overall_complete']and req['whole_modules_closed']==0)
        check('allmodulelowestMFUnotclaimed',next(r for r in req['requirements']if r['id']=='LOWEST_MODULE_UTILIZATION')['state']=='NOT_ESTABLISHED')
        for row in read(O/'candidate-record-index-v96.json')['model_candidates']:check('master:'+row['id'],hashlib.sha256(Path(row['master_record']).read_bytes()).hexdigest()==row['master_sha256'])
        for row in read(O/'checkpoint-manifest-v96.json')['artifacts']:
            p=Path(row['path']);check('manifest:'+p.name,p.stat().st_size==row['bytes']and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'])
    result=dict(status='PASS',checks=len(checks),preflight=args.preflight,inverse_permutation_cases=count,partial_reader_models=partial,
      no_native_model_or_Torch_execution=True,overall_complete=False)
    if args.preflight:write(O/'route-map-analysis-preflight-v96.json',result)
    print(json.dumps(result))

if __name__=='__main__':main()
