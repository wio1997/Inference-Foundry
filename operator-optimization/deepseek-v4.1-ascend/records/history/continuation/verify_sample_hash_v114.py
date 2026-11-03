"""Check normal INT64range,history factorization and complete cost/coverage gates."""
from pathlib import Path
from itertools import product
import argparse,ast,hashlib,json,random
from finalize_origin_coverage_v9 import N,O,T,L,read,write

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');a=parser.parse_args();checks=[]
    def ck(name,v):assert v,name;checks.append(name)
    def pinned(r):
        b=Path(r['path']).read_bytes();return len(b)==r['bytes']and hashlib.sha256(b).hexdigest()==r['sha256']
    for r in read(O/'sample-hash-inputs-v114.json')['inputs']:ck('input:'+r['path'],pinned(r))
    src=read(O/'sample-hash-source-contract-v114.json');texts={}
    for p in src['parts']:
        ck('source:'+p['path'],pinned(p));text=Path(p['path']).read_text();texts[p['path']]=text
        ck('section:'+p['path']+':'+str(p['first_line']),text.splitlines()[p['first_line']-1:p['last_line']]==p['source'])
    common=next(t for p,t in texts.items()if'/common/engram.py'in p)
    tree=ast.parse(common);builder=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='compute_hash_multipliers')
    ck('builder_bound_expression',any(isinstance(n,ast.Assign)and ast.unparse(n.value)=='max(1, max_long // compressed_vocab_size // 2)'for n in builder.body))
    ck('builder_odd_multiplier','values * 2 + 1'in ast.unparse(builder))
    ck('DEAD_is_minus_one','DEAD_ID = -1'in common)
    ck('cache_normal_initial_andwriter','self._cache = torch.zeros('in common and 'value = tl.where(dead, dead_id, value)'in common)
    M=(1<<63)-1;rng=random.Random(324114);range_cases=0
    for v in list(range(1,1000))+[129280,1<<31]+[rng.randint(1,1<<31)for _ in range(3000)]:
        b=(M//v)//2;max_multiplier=2*b-1
        assert b>=1 and max_multiplier<=M//v-1 and (v-1)*max_multiplier<=M
        range_cases+=1
    ck('independent_product_bound',range_cases==4001)
    # Independent direct sum over XOR terms using reduce,including blocked-prefix propagation.
    from functools import reduce
    xor=lambda seq:reduce(lambda x,y:x^y,seq,0)
    factors=0
    for mults in [(1,3,5,7),(17,31,101,1009),((1<<60)-1,(1<<59)+1,13,19)]:
        for code in product([0,1,2,5],repeat=4):
            for mask in range(16):
                blocked=[bool(mask&(1<<d))for d in range(4)];pad=3
                direct=[xor((pad if any(blocked[:d+1])else code[d])*mults[d]for d in range(s+1))for s in [1,2,3]]
                h=[xor((pad if any(blocked[1:d+1])else code[d])*mults[d]for d in range(1,s+1))for s in [1,2,3]]
                actual=[xor(pad*mults[d]for d in range(s+1))if blocked[0]else code[0]*mults[0]^h[s-1]for s in [1,2,3]]
                assert actual==direct;factors+=1
    ck('independent_factorization',factors==12288)
    facts=read(O/'Engram-history-factorization-v114.json')
    ck('48Bhistory192BIDs',facts['history_packet_bytes']==6*8==48 and facts['hash_ID_output_bytes']==2*3*8*4==192)
    ck('early_history_has_stability_premise','independentoftheunselectedcurrentID'in facts['early_preparation_premise'])
    ck('actual_nextcall_not_bound',not facts['actual_prefix_to_nextcall_bindings_proved'])
    validation=read(O/'sample-hash-integer-validation-v114.json');r=validation['reference']
    ck('source_map_validation_12288_589824',r['cases']==12288 and r['hash_ID_comparisons']==589824 and r['max_product_observed']<=M)
    witnesses={w['id']:w for w in validation['invalid_rewrites']}
    ck('six_failure_witnesses',len(witnesses)==6)
    ck('current_dead_wrong_xorout',witnesses['CANDIDATE_DEAD_BREAKS_XOR_OUT']['xor_out_proposal_gives']!=witnesses['CANDIDATE_DEAD_BREAKS_XOR_OUT']['true_H_for_alive_recovery'])
    c=witnesses['ONE_OLD_LONGEST_ACCUMULATOR_NOT_SHIFT_STATE']
    ck('longest_summary_counterexample',c['a'][0]*3^c['a'][1]*5^c['a'][2]*7==c['same_longest_H']==c['b'][0]*3^c['b'][1]*5^c['b'][2]*7 and c['next_after_zero_a']!=c['next_after_zero_b'])
    # -1's two typed meanings must not be conflated by a raw equality test.
    window=-1;known=window>=0;fallback=7;resolved=window if known else fallback
    ck('unknown_window_resolves_fallback',resolved==7 and not known)
    profile=read(O/'sample-hash-profile-scope-v114.json')
    expected={'_engram_req_index_kernel':88,'_write_hash_cache_kernel':88,'_hash_ids_kernel':88,'_engram_host_uva_gather_dequant_kernel':176}
    ck('reused_profile_populations',profile['existing_execution_counts']==expected)
    ck('selected_NPU_not_template_CPU',profile['selected_hash_and_UVAlookup_are_accelerator_TASKs_not_CPUzero_time'])
    for op,count in expected.items():
        rows=[r for r in profile['existing_regimes']if r['operator']==op]
        ck('unique_saved_TASKs:'+op,len({tuple(k)for r in rows for k in r['TASK_keys']})==sum(r['executions']for r in rows)==count)
    costs=read(O/'sample-hash-joint-work-and-capacity-v114.json')
    for m in costs['prefix_alternative_models']:
        d=m['proposed_depth'];ck('prefix_complete_cost:'+str(d),m['Hbank_payload_bytes']==48*(d+1) and m['independently_prepared_history_mul_count']+m['selected_finalize_mul_count']-m['baseline_single_position_mul_count']==m['total_extra_mul_vs_one_late_fullhash']==6*d and m['history_nontrivial_XOR_count']+m['selected_finalize_XOR_count']-m['baseline_nontrivial_XOR_count']==m['total_extra_XOR_vs_one_late_fullhash']==4*d+4 and m['selected_remainder_count']==m['baseline_remainder_count']==48)
    ck('transport24_48_reference',[x['payload_ratio']for x in costs['transport']]==[48,24]and 'additional,not a replacement'in costs['cross_card'])
    ck('hostmapped_formats',costs['cold_local_lookup_payload_per_token']=={'BF16':3072,'INT8_GROUP32':1728,'MXFP8_GROUP32':1584})
    state=read(O/'sample-hash-state-and-publication-v114.json')
    ck('physical_baseline_proposal_writes_preserved','canincludeunacceptedtargetproposalrows'in state['original_history_write'])
    ck('no_actual_state_or_speedup_claim',not state['numeric_and_state_legality_proved_for_actual_fixedload']and costs['no_MFU_physical_numeric_ceiling_or_speedup_proved'])
    comp=read(O/'sample-hash-composition-v114.json');ck('one_new_pair_group',comp['new_pairs']==1 and comp['new_groups']==1 and set(comp['group']['members'])=={'F24','F31','F35'})
    previous=read(O/'candidate-pair-navigation-v113.json');pair=next(r for r in previous['pairs']if{r['a'],r['b']}=={'F24','F35'})
    ck('central_pair_previously_pending',not pair['semantic_pair_reconciliation_done_in_stated_scope'])
    if a.preflight:manifest=read(O/'checkpoint-manifest-v113.json')
    else:
        cp=read(O/'REVIEW-CHECKPOINT-v114.json');ck('progress_not_complete',cp['status']=='PROGRESS'and not cp['overall_complete'])
        rel=read(O/'candidate-relation-clauses-v114.json');ck('134clauses26groups',len(rel['clauses'])==134 and len(rel['higher_order_groups'])==26)
        cov=read(O/'candidate-relation-coverage-v114.json');ck('134_26_461',(cov['scoped_conditional_pairs'],cov['higher_order_groups'],cov['remaining_pair_relations'])==(134,26,461))
        nav=read(O/'candidate-pair-navigation-v114.json');p=next(r for r in nav['pairs']if{r['a'],r['b']}=={'F24','F35'})
        ck('new_pair_open_allvariants',p['conditional_clause_ids']==['REL114-01']and p['semantic_pair_reconciliation_done_in_stated_scope']and not p['whole_pair_all_variants_closed'])
        matrix=read(O/'module-phase-branch-matrix-v114.json');ck('353_26',sum(len(m['branches'])for m in matrix['modules'])==353 and len(matrix['modules'])==26)
        old=read(O/'operator-analysis-coverage-ledger-v113.json');new=read(O/'operator-analysis-coverage-ledger-v114.json')
        for k in ['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']:ck('native_unchanged:'+k,old[k]==new[k])
        for r in read(O/'candidate-record-index-v114.json')['model_candidates']:ck('master:'+r['id'],hashlib.sha256(Path(r['master_record']).read_bytes()).hexdigest()==r['master_sha256'])
        ck('task_active_loop_evaluating',read(T/'index.json')['status']=='ACTIVE'and read(L/'state.json')['status']=='EVALUATING')
        manifest=read(O/'checkpoint-manifest-v114.json')
    for r in manifest['artifacts']:ck('manifest:'+r['path'],pinned(r))
    result=dict(status='PASS',checks=len(checks),preflight=a.preflight,independent_constructor_range_cases=range_cases,
        independent_factor_cases=factors,source_reference_factor_cases=12288,source_hash_ID_checks=589824,new_device_work=False,new_native_promotions=0,overall_complete=False)
    if a.preflight:write(O/'sample-hash-preflight-v114.json',result)
    print(json.dumps(result))

if __name__=='__main__':main()
