"""Independent interval/cache-state/cost and persisted coverage checks; CPU only."""
from pathlib import Path
from itertools import permutations
import argparse,ast,hashlib,json
from finalize_origin_coverage_v9 import N,O,T,L,read,write

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');a=parser.parse_args();checks=[]
    def ck(name,value):assert value,name;checks.append(name)
    def pinned(r):
        b=Path(r['path']).read_bytes();return len(b)==r['bytes']and hashlib.sha256(b).hexdigest()==r['sha256']
    for r in read(O/'draft-cache-join-inputs-v113.json')['inputs']:ck('input:'+r['path'],pinned(r))
    src=read(O/'draft-cache-join-source-contract-v113.json');source={}
    for p in src['parts']:
        ck('pinned_source:'+p['path'],pinned(p));ls=Path(p['path']).read_text().splitlines()
        ck('section:'+p['path']+':'+str(p['first_line']),ls[p['first_line']-1:p['last_line']]==p['source']);source[Path(p['path']).name]=Path(p['path']).read_text()
    ut=source['utils.py'];fn=next(n for n in ast.parse(ut).body if isinstance(n,ast.FunctionDef)and n.name=='copy_and_expand_dflash_and_dspark_inputs_kernel')
    names=[n.arg for n in fn.args.args]
    ck('metadata_has_no_hidden_input','target_hidden_states'not in names and 'context_states'not in names)
    first=ut.split('# --- Part 1: context positions / slot_mapping copy ---')[1].split('# --- Part 2: query block expand ---')[0]
    ck('full_context_copy_not_rejection_filtered','num_rejected'not in first and 'total_input_tokens'in first and 'context_slot_mapping_ptr + offs'in first)
    ck('query_linear_positions','query_kv_slot_pos = effective_seq_len + q_idx'in ut)
    ck('DCP_inverse_fields','virtual_block_size = CP_INTERLEAVE_SIZE * DCP_SIZE'in ut)
    ck('not_RoPE_slot','query_pos = last_pos + 1 + q_idx'in ut and 'block_num_q = local_kv_slot_pos // block_size'in ut)
    ds_fn=next(n for n in ast.walk(ast.parse(source['dspark.py']))if isinstance(n,ast.FunctionDef)and n.name=='_store_standard_swa_kv')
    cache_alias=any(isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='cache'for t in n.targets)and ast.unparse(n.value)=='attn.dsa_attn.swa_cache_layer'for n in ds_fn.body)
    cache_arg=any(isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='scatter_cache_sk'and ast.unparse(n.args[0])=='cache.kv_cache[0]'for n in ast.walk(ds_fn))
    ck('same_source_bank_expression',cache_alias and cache_arg and 'attn.dsa_attn.swa_cache_layer.kv_cache[0]'in source['dsa_v41.py'])
    ck('constructor_query_branches','self.num_query_per_req = self.num_speculative_tokens'in source['dspark_proposer.py']and 'self.num_query_per_req = 1 + self.num_speculative_tokens'in source['dspark_proposer.py'])
    f=read(O/'draft-cache-join-native-frontiers-v113.json');rows=f['frontiers']
    ck('104_frontiers_312_cycles',len(rows)==104 and sum(len(r['context_cycles'])for r in rows)==312)
    ck('88_main16_dummy',sum(r['prior_main']is not None for r in rows)==88 and f['contexts']=={'CALLER_DUMMY':16,'CALLER_DRAFT_OR_DRAFT_METADATA':88})
    keys=set()
    for i,r in enumerate(rows):
        cs=r['context_cycles'];ck('ordered_context_repeat:'+str(i),all(cs[j]['endNs']<=cs[j+1]['startNs']for j in range(2))and cs[-1]['endNs']<=r['draft_repeat']['startNs'])
        ck('no_runtime_alias_claim:'+str(i),r['physical_alias_or_counterfactual_critical_path_proved']is False)
        for c in cs:
            key=(r['database_index'],c['first_TASK']);ck('unique_context:'+str(key),key not in keys);keys.add(key)
    algebra=read(O/'draft-cache-slot-conflict-algebra-v113.json');ck('conditional_actual_overlap_unknown',algebra['status']=='CONDITIONAL_CEILING'and algebra['runtime_overlap_count']is None)
    # Independent direct finite intervals,including larger tails/querycounts than analyzer fixtures.
    interval_cases=0
    for n in range(1,25):
        for r in range(n+1):
            for q in [1,2,5,9,31]:
                for prefix in [0,11]:
                    end=prefix+n
                    c=set(range(prefix,end));z=set(range(end-r,end-r+q))
                    assert len(c&z)==min(r,q)
                    interval_cases+=1
    ck('independent_interval_cardinality',interval_cases==3240)
    # Count each residue-owner independently rather than invoking analyzer's map function.
    owner_cases=0
    for d in [1,2,3,4,8]:
        for width in [1,2,3,7]:
            for p in range(300):
                group,offset=divmod(p,d*width);rank,within=divmod(offset,width)
                local=group*width+within
                assert(local//width)*(d*width)+rank*width+local%width==p
                owner_cases+=1
    ck('independent_compact_inverse',owner_cases==6000)
    order=read(O/'draft-cache-order-and-witnesses-v113.json');ck('finite_mapping_cases',order['CPU_reference']['cases']==25344)
    ck('one_correct_serial_schedule',len(order['schedules'])==3 and sum(s['source_equivalent']for s in order['schedules'])==1)
    ck('read_and_final_state_gate',any(s['observed']=='query'and s['final']=='context'for s in order['schedules']))
    witness=next(w for w in order['witnesses']if w['id']=='REJECTED_TAIL_OVERLAP')
    ck('collision_witness',set(witness['context_logical'])&set(witness['query_logical'])==set(witness['overlap'])=={8,9})
    ck('five_invalid_inference_witnesses',len(order['witnesses'])==5)
    # The all-layer union,not the intersection,defines producer demand.
    dl=[{0,1},{1,2},{2,3}];other={4}
    live=set().union(*dl,other);ck('upstream_demand_union',live==set(range(5))and set.intersection(*dl)==set())
    demand=read(O/'draft-cache-demand-and-readiness-v113.json');ck('allreader_not_attested',demand['actual_allreader_or_epoch_closure_proved']is False)
    work=read(O/'draft-cache-joint-work-and-ceiling-v113.json');ck('symbolic_no_speedup',work['status']=='SYMBOLIC_CEILING'and not work['actual_MFU_physical_numeric_floor_or_speedup_proved'])
    for i,e in enumerate(work['examples']):
        overlap=sum(min(r,q)for r,q in zip(e['rejected'],e['query_per_request']));union=e['context_rows']+e['query_rows']-overlap
        ck('example_overlap:'+str(i),overlap==e['conditional_overwrite_rows'])
        ck('example_write_budget:'+str(i),e['source_context_plus_query_cache_row_write_bytes']-e['conditional_final_union_row_write_bytes']==1024*overlap and e['conditional_final_union_row_write_bytes']==union*1024)
        ck('example_WKV_cost:'+str(i),e['full_ordinary_WKV_reference_OPS']-e['after_private_dead_context_row_pruning_WKV_OPS']==2*overlap*5120*512==e['conditional_WKV_OPS_difference_per_layer'])
        ck('example_main_alllayer_gate:'+str(i),e['if_same_dead_rows_atall3layers_andno_other_main_readers_main_OPS_difference']==2*overlap*15360*5120 and not e['actual_profile_rejected_counts'])
    comp=read(O/'draft-cache-composition-v113.json');ck('new_pair_not_prior_repeat',comp['new_pairs']==1 and comp['new_groups']==0 and comp['old_rowpack_andlazy_perlayer_opportunities_not_new'])
    nav=read(O/'candidate-pair-navigation-v112.json');oldpair=next(r for r in nav['pairs']if(r['a'],r['b'])==('F27','F28'))
    ck('old_central_pair_pending',not oldpair['semantic_pair_reconciliation_done_in_stated_scope'])
    if a.preflight:manifest=read(O/'checkpoint-manifest-v112.json')
    else:
        cp=read(O/'REVIEW-CHECKPOINT-v113.json');ck('progress_not_complete',cp['status']=='PROGRESS'and not cp['overall_complete'])
        clauses=read(O/'candidate-relation-clauses-v113.json');ck('133_clauses',len(clauses['clauses'])==133 and len(clauses['higher_order_groups'])==25)
        cov=read(O/'candidate-relation-coverage-v113.json');ck('133_25_462',(cov['scoped_conditional_pairs'],cov['higher_order_groups'],cov['remaining_pair_relations'])==(133,25,462))
        nav=read(O/'candidate-pair-navigation-v113.json');pair=next(r for r in nav['pairs']if(r['a'],r['b'])==('F27','F28'))
        ck('one_pair_added',pair['conditional_clause_ids']==['REL113-01']and pair['semantic_pair_reconciliation_done_in_stated_scope']and not pair['whole_pair_all_variants_closed'])
        matrix=read(O/'module-phase-branch-matrix-v113.json');ck('353_26',len(matrix['modules'])==26 and sum(len(m['branches'])for m in matrix['modules'])==353)
        old=read(O/'operator-analysis-coverage-ledger-v112.json');new=read(O/'operator-analysis-coverage-ledger-v113.json')
        for k in ['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']:ck('native_unchanged:'+k,old[k]==new[k])
        for r in read(O/'candidate-record-index-v113.json')['model_candidates']:ck('master:'+r['id'],hashlib.sha256(Path(r['master_record']).read_bytes()).hexdigest()==r['master_sha256'])
        ck('task_loop_active',read(T/'index.json')['status']=='ACTIVE'and read(L/'state.json')['status']=='EVALUATING')
        manifest=read(O/'checkpoint-manifest-v113.json')
    for r in manifest['artifacts']:ck('manifest:'+r['path'],pinned(r))
    result=dict(status='PASS',checks=len(checks),preflight=a.preflight,source_order_frontiers=104,
        independent_interval_cases=interval_cases,independent_compact_inverse_cases=owner_cases,CPU_source_map_cases=25344,
        new_device_work=False,new_native_promotions=0,overall_complete=False)
    if a.preflight:write(O/'draft-cache-join-preflight-v113.json',result)
    print(json.dumps(result))

if __name__=='__main__':main()
