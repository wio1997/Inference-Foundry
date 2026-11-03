"""Independent rank/summary/reference-work and saved-frontier checks, without target code execution."""
from pathlib import Path
from itertools import permutations,product
from collections import Counter
from fractions import Fraction as F
import argparse,hashlib,json,struct
from finalize_origin_coverage_v9 import N,O,R,read,write

def f32(x):return struct.unpack('f',struct.pack('f',float(x)))[0]
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');a=parser.parse_args();checks=[]
    def check(name,value):
        assert value,name
        checks.append(name)
    def pinned(r):
        p=Path(r['path']);return p.stat().st_size==r['bytes']and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
    inp=read(O/'selection-attention-inputs-v103.json')
    for r in inp['inputs']:check('input:'+r['path'],pinned(r))
    src=read(O/'selection-attention-source-contract-v103.json')
    for r in src['parts']:
        check('source:'+r['path'],pinned(r));ls=Path(r['path']).read_text().splitlines()
        for s in r['segments']:check('segment:'+str(s['first_line'])+Path(r['path']).name,ls[s['first_line']-1:s['last_line']]==s['source'])
    text={Path(r['path']).name:Path(r['path']).read_text()for r in src['parts']}
    prep=text['prepare_indexer_indices.py'];csa=text['sparse_flash_mla_csa_block_vector.h']
    check('ordered_valid_prefix_contract','valid = (selected >= 0) & (selected < visible[:, None])'in prep and 'descending=False'in prep and 'selected == SENTINEL, -1'in prep)
    check('native_probability_round_and_previous_PV_rescale','RoundMode::CAST_ROUND, computeSize'in csa and 'RowMuls(bmm2ResPreUb, bmm2ResPreUb, expUb'in csa and 'Add(bmm2ResUb, bmm2ResUb, bmm2ResPreUb'in csa)
    check('native_previous_stat_input','inSumTensor = softmaxSumUb[inIdx'in csa and 'inMaxTensor = softmaxMaxUb[inIdx'in csa)
    front=read(O/'selection-attention-native-frontiers-v103.json');old=read(O/'attention-output-owner-frontiers-v100.json')
    old_by={(r['database_index'],r['graph_ordinal'],r['block_ordinal']):r for r in old['frontiers']if r['index_selection']}
    unique=set();modes=Counter();roots=Counter()
    for r in front['frontiers']:
        b=old_by[(r['database_index'],r['graph_ordinal'],r['block_ordinal'])];s=r['stages']
        assert s['QLI']==b['index_selection']['QLI']and s['prepare']==b['index_selection']['prepare']and s['SMLA']==b['stages']['SMLA']
        assert s['QLI']['endNs']<=s['prepare']['startNs']<=s['prepare']['endNs']<=s['SMLA']['startNs']
        assert len({x['streamId']for x in s.values()})==1
        assert r['observed_QLI_start_to_SMLA_end_ns']==s['SMLA']['endNs']-s['QLI']['startNs']
        for x in s.values():unique.add((r['database_index'],x['task_record_rowid']))
        modes[r['mode']]+=1;roots[r['root_bound']]+=1
    check('640existing_frontiers_1920tasks',len(old_by)==len(front['frontiers'])==640 and len(unique)==front['reused_unique_TASKs']==1920)
    check('512root128unbound',roots=={True:512,False:128})
    check('source_candidate_consumer_mode_counts',modes==front['modes']=={'MODE3_UNFILTERED':240,'MODE1_CANDIDATE_SOURCE':80,'MODE2_CONSUMER':320})
    proof=read(O/'selection-winner-readiness-and-cover-v103.json');cases=0
    # Compute rank by counting strict total-order predecessors, rather than using the analyzer's sort.
    def chosen(ids,values,k):
        return {i for i in ids if sum(values[j]>values[i]or(values[j]==values[i]and j<i)for j in ids)<k}
    for n in range(2,7):
        for v in permutations(range(n)):
            for k in range(1,min(3,n-1)+1):
                goal=chosen(list(range(n)),v,k)
                for p in range(1,n+1):
                    groups=[list(range(j*n//p,(j+1)*n//p))for j in range(p)]
                    union=set().union(*(chosen(g,v,k)for g in groups))
                    assert chosen(sorted(union),v,k)==goal
                    assert len(union)==sum(min(k,len(g))for g in groups);cases+=1
    check('15088rank_partition_cases',cases==proof['finite_partition_and_rank_cases']==15088)
    ties=0
    for v in product([-1,0,1],repeat=5):
        goal=chosen(list(range(5)),v,2)
        for p in range(1,6):
            groups=[list(range(j*5//p,(j+1)*5//p))for j in range(p)]
            union=set().union(*(chosen(g,v,2)for g in groups))
            assert chosen(sorted(union),v,2)==goal;ties+=1
    check('1215tie_totalorder_reference_cases',ties==1215)
    witness=proof['unseen_score_witness']
    check('unseen_completion_changes_winner',chosen([0,1,2],witness['unseen_low_completion'],1)=={0}and chosen([0,1,2],witness['unseen_high_completion'],1)=={2})
    check('no_forced_global_kernel_barrier',proof['no_universal_whole_operator_barrier'])
    ret=read(O/'selection-attention-retraction-and-rounding-v103.json');real=0
    for masses in product([F(1,8),F(1,2),F(1)],repeat=3):
        for vals in product([F(-2),F(0),F(1)],repeat=3):
            for i in range(3):
                l=sum(masses);num=sum(m*v for m,v in zip(masses,vals));rest=[j for j in range(3)if j!=i]
                assert (num-masses[i]*vals[i])/(l-masses[i])==sum(masses[j]*vals[j]for j in rest)/sum(masses[j]for j in rest);real+=1
    check('2187real_retraction_cases',real==ret['exact_real_reference']['positive_mass_finite_cases']==2187)
    c=ret['rounded_state_collision'];eps=2**-25
    check('rounded_summary_same_despite_distinct_survivor',all(h['m']==0 and h['l']==f32(1+eps)==1 and h['a']==f32(1+eps*h['remaining_value'])==1 for h in c['histories']))
    check('same_removed_item_cannot_recover_outputs',c['after_correct_remaining_outputs']==[0,1]and c['removed_item']=={'mass':1,'value':1}and c['rounded_subtraction_l']==c['rounded_subtraction_a']==0)
    r=ret['provisional_BF16_rebase']
    # Check the exact neighboring BF16 lattice values without reusing a bit-round helper.
    check('BF16old_exp_nearest',abs(F(5019,10000)-F(1,2))<abs(F(5019,10000)-F(129,256))and r['rounded_old_exp']==.5)
    check('BF16new_exp_nearest',abs(F(5019,10000)*F(3,4)-F(193,512))<abs(F(5019,10000)*F(3,4)-F(3,8))and r['correct_new_tile_exp']==float(F(193,512)))
    q1=F.from_float(f32(float(F(193,512)/F.from_float(r['denominator_reference']))))
    q2=F.from_float(f32(float(F(3,8)/F.from_float(r['denominator_reference']))))
    lattice=[F(i,512)for i in range(128,145)]
    check('reference_final_output_difference',r['rescaled_old_exp']==.375 and
        min(lattice,key=lambda x:abs(x-q1))==F.from_float(r['fresh_final_tile_output'])==F(140,512)and
        min(lattice,key=lambda x:abs(x-q2))==F.from_float(r['speculative_rebased_output'])==F(139,512))
    check('append_not_replace',ret['append_only_eviction_witness']['append_only_output']==.5 and ret['append_only_eviction_witness']['correct_K1_replacement_output']==1)
    cost=read(O/'selection-attention-speculation-cost-models-v103.json');check('36work_cases',len(cost['models'])==36)
    for r in cost['models']:
        n,p,k=r['N'],r['partitions'],r['K_valid_reference'];lengths=[(i+1)*n//p-i*n//p for i in range(p)]
        A=sum(min(k,x)for x in lengths);base=4*64*512*k;new=2*64*512*(A+k)
        assert r['local_topk_union_attention_QK_rows']==A and k<=A<=min(n,p*k)
        assert base==r['baseline_compressed_classical_ops']and new==r['speculative_compressed_classical_ops']
        assert F(r['compressed_compute_work_ratio'])==F(new,base)
        assert F(r['all_with_SWA_work_ratio'])==F(new+4*64*512*128,base+4*64*512*128)
        assert r['partition_union_FLOAT_attention_logit_values_bytes']==256*A and r['resident_heap_FLOAT_attention_logit_values_bytes']==256*k
    check('case_formulas_not_speedups',not cost['actual_valid_lengths_winner_sequences_native_precision_or_physical_speedup_known']and cost['theory']['status']=='SYMBOLIC_CEILING')
    ex=next(r for r in cost['models']if r['N']==34816 and r['partitions']==8)
    check('34816key_8partition_example',ex['local_topk_union_attention_QK_rows']==4096 and F(ex['compressed_compute_work_ratio'])==F(9,2)and F(ex['all_with_SWA_work_ratio'])==F(19,5)and ex['partition_union_FLOAT_attention_logit_values_bytes']==1048576 and ex['resident_heap_FLOAT_attention_logit_values_bytes']==131072)
    variants=read(O/'selection-attention-variants-v103.json');check('sixchoices_existing_edge',len(variants['variants'])==6 and variants['existing_relation']=='REL83-08'and not variants['new_candidate_or_pair'])
    if a.preflight:manifest=read(O/'checkpoint-manifest-v102.json')
    else:
        cp=read(O/'REVIEW-CHECKPOINT-v103.json');check('PROGRESSnotwhole',cp['status']=='PROGRESS'and not cp['overall_complete'])
        bm=read(O/'module-phase-branch-matrix-v103.json');check('364rows24semantic',sum(len(m['branches'])for m in bm['modules'])==364 and sum(m.get('branch_schema')=='SEMANTIC_BRANCHES_V1'for m in bm['modules'])==24)
        rel=read(O/'candidate-relation-coverage-v103.json');check('131_25_464same', (rel['scoped_conditional_pairs'],rel['higher_order_groups'],rel['remaining_pair_relations'])==(131,25,464))
        op=read(O/'operator-analysis-coverage-ledger-v103.json');prev=read(O/'operator-analysis-coverage-ledger-v102.json')
        for k in ['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']:check('nativeunchanged:'+k,op[k]==prev[k])
        req=read(O/'whole-scope-requirement-audit-v103.json');check('MFUstillnotestablished',next(r for r in req['requirements']if r['id']=='LOWEST_MODULE_UTILIZATION')['state']=='NOT_ESTABLISHED')
        for r in read(O/'candidate-record-index-v103.json')['model_candidates']:check('master:'+r['id'],hashlib.sha256(Path(r['master_record']).read_bytes()).hexdigest()==r['master_sha256'])
        manifest=read(O/'checkpoint-manifest-v103.json')
    for r in manifest['artifacts']:check('manifest:'+r['path'],pinned(r))
    result=dict(status='PASS',checks=len(checks),preflight=a.preflight,rank_partition_cases=cases,tie_cases=ties,real_retraction_cases=real,frontiers=640,new_native_promotions=0,overall_complete=False)
    if a.preflight:write(O/'selection-attention-analysis-preflight-v103.json',result)
    print(json.dumps(result))

if __name__=='__main__':main()
