"""Check one shared memory remainder, capacity frontier and cache trace counterexamples."""
from pathlib import Path
import argparse,hashlib,json
from finalize_origin_coverage_v9 import N,O,T,L,read,write

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');a=ap.parse_args();checks=[]
    def ck(n,v):assert v,n;checks.append(n)
    def pinned(r):
        b=Path(r['path']).read_bytes();return len(b)==r['bytes']and hashlib.sha256(b).hexdigest()==r['sha256']
    for r in read(O/'row-cache-budget-inputs-v121.json')['inputs']:ck('input:'+r['path'],pinned(r))
    rep=read(O/'row-cache-representation-contract-v121.json');f=rep['formats']
    ck('source_payload_formats',{k:v['encoded_payload_bytes']for k,v in f.items()}==dict(BF16=512,INT8_GROUP32=288,MXFP8_GROUP32=264))
    ck('chosen_aligned_slots',{k:v['illustrative_slot_bytes']for k,v in f.items()}==dict(BF16=576,INT8_GROUP32=352,MXFP8_GROUP32=352))
    ck('format_not_runtime_flag',not rep['current_runtime_format_constexpr_attested'])
    b=read(O/'row-cache-common-budget-models-v121.json');C=b['C'];ck('source_poolC',C==540928)
    ck('144_36_models',len(b['models'])==144 and len(b['no_old_free_ID_loss_capacity_references'])==36)
    for i,r in enumerate(b['models']):
        D=r['ring_extra_bytes']+r['row_cache_slots']*r['selected_row_cache_slot_bytes']+r['WKV_memo_entries']*r['WKV_entry_result_and_ID_key_bytes']
        ck('combined:'+str(i),D==r['combined_extra_bytes']and r['actual_raw_budget_credit_assumed']==0)
        # Directfloor comparison at both remainder endpoints and every threshold side.
        points={0,C-1,D%C,max(0,D%C-1)}
        deltas=[r['released_common_IDs']+((10000*C+rem-D)//C)-10000 for rem in points]
        ck('single_remainder:'+str(i),(min(deltas),max(deltas))==(r['net_free_global_ID_delta_min'],r['net_free_global_ID_delta_max']))
        ck('no_actual_credit:'+str(i),not r['actual_release_or_capacity_gain_proved'])
    for i,r in enumerate(b['no_old_free_ID_loss_capacity_references']):
        for rem,label in [(0,'remainder0'),(C-1,'remainderCminus1')]:
            K=r['row_capacity_with_no_old_free_ID_loss_at_'+label];q=r['selected_slot_bytes']
            fixed=r['ring_extra_bytes']+r['WKV_memo_entries']*51296
            if K<0:ck('infeasibleK0:'+str(i)+label,fixed>r['released_common_IDs']*C+rem)
            else:ck('sharpcapacity:'+str(i)+label,fixed+K*q<=r['released_common_IDs']*C+rem<fixed+(K+1)*q)
    ex=b['example_private_even_ring_and4096_INT8_slots'];ck('not_automatic_capacity_gain',ex['combined_extra_bytes']==2228224 and ex['net_free_global_ID_delta_min']==-1 and ex['net_free_global_ID_delta_max']==0)
    # Small exhaustive joint slot frontier: independently count feasible K and J for each common remainder.
    cases=0
    for c in range(3,18):
        for released in range(4):
            for remainder in range(c):
                for fixed in [0,2,7]:
                    feasible={(k,j)for k in range(30)for j in range(30)if 2*k+3*j+fixed<=released*c+remainder}
                    direct={(k,j)for k in range(30)for j in range(30)if released+(100*c+remainder-2*k-3*j-fixed)//c-100>=0}
                    assert feasible==direct;cases+=1
    ck('1800_joint_frontier_cases',cases==1800)
    traces=read(O/'row-cache-miss-traffic-and-memo-demand-v121.json')['synthetic_trace_models']
    expected={('interleaved',0):6,('interleaved',1):6,('interleaved',2):6,('interleaved',3):3,
        ('grouped',0):6,('grouped',1):3,('grouped',2):3,('grouped',3):3}
    for r in traces:
        ck('trace:'+r['name']+str(r['K']),r['cold_LRU_misses']==expected[(r['name'],r['K'])]and r['INT8_logical_host_payload']==288*r['cold_LRU_misses'])
        ck('disabled_not_fill_then_read:'+r['name']+str(r['K']),(r['INT8_fill_then_read_cache_reads']is None)==(r['K']==0))
    race=read(O/'row-cache-retirement-and-numerical-contract-v121.json')['mixed_epoch_witness']
    ck('mixed_epoch_values_neither_valid',2*3==race['old_value']==6 and 4*5==race['new_value']==20 and 2*5==10 and 4*3==12 and not set(race['mixed_values'])&{6,20})
    ck('not_hardware_capacity_proof',not b['actual_DeltaB_r_K_allocation_or_hardware_speedup_proved'])
    if a.preflight:manifest=read(O/'checkpoint-manifest-v120.json')
    else:
        cp=read(O/'REVIEW-CHECKPOINT-v121.json');ck('progress_open',cp['status']=='PROGRESS'and not cp['overall_complete'])
        rel=read(O/'candidate-relation-clauses-v121.json');pairs={tuple(sorted((r['a'],r['b'])))for r in rel['clauses']}
        ck('135_unique_pairs',len(pairs)==len(rel['clauses'])==135 and ('F24','F34')in pairs)
        g=next(g for g in rel['higher_order_groups']if g['id']=='HG85-SHARED-MEMORY-BUDGET');ck('existing_group_extended',len(rel['higher_order_groups'])==26 and 'F24'in g['members'])
        cov=read(O/'candidate-relation-coverage-v121.json');ck('135_26_460',(cov['scoped_conditional_pairs'],cov['higher_order_groups'],cov['remaining_pair_relations'])==(135,26,460))
        nav=read(O/'candidate-pair-navigation-v121.json');p=next(r for r in nav['pairs']if{r['a'],r['b']}=={'F24','F34'});ck('pair_reconciled_in_scope',p['conditional_clause_ids']==['REL121-01']and p['semantic_pair_reconciliation_done_in_stated_scope']and not p['whole_pair_all_variants_closed'])
        matrix=read(O/'module-phase-branch-matrix-v121.json');ck('356_26_unchanged',len(matrix['modules'])==26 and sum(len(m['branches'])for m in matrix['modules'])==356)
        old=read(O/'operator-analysis-coverage-ledger-v120.json');new=read(O/'operator-analysis-coverage-ledger-v121.json')
        for k in ['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']:ck('native_unchanged:'+k,old[k]==new[k])
        gs={g['id']:g for g in read(O/'coverage-current-obligations-v121.json')['obligations']};ck('current_text135_26_460','135 conditional relations/26 groups leave460'in gs['G10']['remaining'])
        for r in read(O/'candidate-record-index-v121.json')['model_candidates']:ck('candidate:'+r['id'],hashlib.sha256(Path(r['master_record']).read_bytes()).hexdigest()==r['master_sha256'])
        ck('Task_active_Loop_evaluating',read(T/'index.json')['status']=='ACTIVE'and read(L/'state.json')['status']=='EVALUATING')
        manifest=read(O/'checkpoint-manifest-v121.json')
    for r in manifest['artifacts']:ck('manifest:'+r['path'],pinned(r))
    result=dict(status='PASS',checks=len(checks),preflight=a.preflight,independent_joint_frontier_cases=cases,
        budget_models=144,capacity_reference_models=36,new_device_work=False,new_native_promotions=0,overall_complete=False)
    if a.preflight:write(O/'row-cache-budget-preflight-v121.json',result)
    print(json.dumps(result))

if __name__=='__main__':main()
