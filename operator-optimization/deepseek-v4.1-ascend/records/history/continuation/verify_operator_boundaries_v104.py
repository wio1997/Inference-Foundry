"""Check retained scope, zero-scheduled source obligations and saved scalar census."""
from pathlib import Path
from collections import Counter
import argparse,ast,hashlib,json,statistics
from finalize_origin_coverage_v9 import N,O,R,read,write

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');a=parser.parse_args();checks=[]
    def check(name,value):
        assert value,name
        checks.append(name)
    def pinned(r):
        p=Path(r['path']);return p.stat().st_size==r['bytes']and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
    for r in read(O/'operator-boundary-inputs-v104.json')['inputs']:check('input:'+r['path'],pinned(r))
    src=read(O/'operator-boundary-source-contract-v104.json')
    for r in src['parts']:
        check('source:'+r['path'],pinned(r));ls=Path(r['path']).read_text().splitlines()
        for s in r['segments']:check('segment:'+str(s['first_line']),ls[s['first_line']-1:s['last_line']]==s['source'])
    runner=Path(src['parts'][0]['path']).read_text();tree=ast.parse(runner)
    method=next(n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name=='execute_model')
    empty=next(n for n in ast.walk(method)if isinstance(n,ast.If)and ast.unparse(n.test)=='not num_scheduled_tokens')
    updates=[n for n in ast.walk(method)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='_update_states']
    check('state_update_before_zero_guard',len(updates)==1 and updates[0].lineno<empty.lineno)
    guard=empty.body[0];check('external_DP_guard',isinstance(guard,ast.If)and 'distributed_executor_backend' in ast.unparse(guard.test)and "'external_launcher'"in ast.unparse(guard.test)and 'data_parallel_size > 1'in ast.unparse(guard.test))
    call=next(n for n in ast.walk(guard)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='_dummy_run')
    check('dummy_one_token_skip_GDN',ast.literal_eval(call.args[0])==1 and any(k.arg=='skip_gdn_state_update'and ast.literal_eval(k.value)is True for k in call.keywords))
    check('KV_preemptions_before_state_update',runner.index('get_kv_transfer_group().handle_preemptions(kv_connector_metadata)')<runner.index('deferred_state_corrections_fn = self._update_states'))
    check('empty_vs_KV_output_source','return EMPTY_MODEL_RUNNER_OUTPUT'in ast.unparse(empty)and 'return self.kv_connector_no_forward(scheduler_output, self.vllm_config)'in ast.unparse(empty))
    saved=read(O/'operator-boundary-saved-work-domain-v104.json');records={};hist=Counter();rc=Counter();zero=[];installed=set();stream_counts=Counter();resume=0
    for r in saved['metadata_files']:
        check('metadata:'+r['path'],pinned(r));es=[json.loads(x)for x in Path(r['path']).read_text().splitlines()]
        out={e['call_id']:e for e in es if e['event']=='execute_exit'}
        for e in es:
            if e['event']=='installed':installed.add(e['source_sha256'])
            if e['event']!='execute_enter':continue
            key=(e['data_parallel_rank'],e['rank'],e['pid'],e['call_id']);records[key]=e
            assert e['total_scheduled_tokens']==sum(e['scheduled_tokens'].values())
            hist[e['total_scheduled_tokens']]+=1;rc[len(e['scheduled_tokens'])]+=1;resume+=bool(e['cached_request_state']['resumed_req_ids'])
            stream_counts[key[:3]]+=1
            if e['total_scheduled_tokens']==0:
                assert e['scheduled_tokens']=={}and e['cached_request_state']['req_ids']==[]and e['new_request_state']==[]
                x=out[e['call_id']];assert x['error_type']is None
                zero.append((key,e,x))
    check('3712_entries_16_streams',len(records)==3712 and len(stream_counts)==16)
    check('233_231_entries_per_DP_worker',all(n==(233 if k[0]==0 else 231)for k,n in stream_counts.items()))
    check('64_empty_not_64_device_zero',len(zero)==64 and saved['zero_scheduled_entries']==64 and saved['positive_scheduled_entries']==3648 and saved['no_native_phase_or_zero_device_work_promotion'])
    check('empty_four_per_worker',Counter(k[:3]for k,e,x in zero)=={k:4 for k in stream_counts})
    check('histograms',dict(hist)=={int(k):v for k,v in saved['scheduled_total_token_histogram'].items()}and dict(rc)=={int(k):v for k,v in saved['scheduled_request_count_histogram'].items()})
    check('missing_flags_not_false',all(not r['source_branch_actual_executor_KV_EC_predicates_attested']for r in saved['zero_scheduled_calls']))
    check('no_resumption_surface_not_global_impossible',resume==saved['resumed_ids_present_entries']==0)
    check('recorded_source_SHA_matches',installed=={saved['source_model_runner_sha256']}=={src['parts'][0]['sha256']})
    indexed={(r['dp_rank'],r['local_tp_rank'],r['pid'],r['call_id']):r for r in saved['zero_scheduled_calls']}
    for key,e,x in zero:
        r=indexed[key];assert r['label']==e['label']and r['enter_monotonic_ns']==e['monotonic_ns']and r['exit_monotonic_ns']==x['monotonic_ns']
        assert r['wrapper_wall_ns']==x['wrapper_wall_ns']and r['wrapper_thread_cpu_ns']==x['wrapper_thread_cpu_ns']
    for field,label in [('wrapper_wall_ns','zero_call_wrapper_wall_ns'),('wrapper_thread_cpu_ns','zero_call_wrapper_thread_cpu_ns')]:
        v=sorted(x[field]for k,e,x in zero);z=saved[label]
        check('instrumented_stats:'+field,z==dict(n=64,minimum=v[0],median=statistics.median(v),p90=v[int(63*.9)],maximum=v[-1]))
    inv=read(O/'operator-boundary-semantic-inventory-v104.json');cross=read(O/'operator-boundary-history-crosswalk-v104.json');prev=read(O/'module-phase-branch-matrix-v103.json')
    check('15_semantic_rows',len(inv['modules']['M00'])==8 and len(inv['modules']['M03'])==7 and not inv['exhaustive_runtime_branch_claim'])
    mids={m['id']for m in prev['modules']};known={r['id']for rows in inv['modules'].values()for r in rows}
    for mid,rows in inv['modules'].items():
        for r in rows:
            check('scoped_evidence:'+r['id'],r['evidence']and all((O/f).exists()for f in r['evidence'])and set(r['operator_owner_modules'])<=mids and bool(r['gap']))
    old_keys={(mid,i)for mid in ['M00','M03']for i,r in enumerate(next(m['branches']for m in prev['modules']if m['id']==mid))}
    check('all_26_old_rows_retained',len(cross['crosswalk'])==26 and {(r['module'],r['old_ordinal'])for r in cross['crosswalk']}==old_keys and cross['dropped_old_records']==0)
    used=set()
    for r in cross['crosswalk']:
        old=next(m['branches']for m in prev['modules']if m['id']==r['module'])[r['old_ordinal']]
        assert r['old_record']==old and not r['old_record_closed']and set(r['semantic_rows'])<=known and r['semantic_rows'];used.update(r['semantic_rows'])
    check('all_15_rows_have_history',used==known)
    check('353_count_not_completion_percent',364-26+15==cross['new_total_rows']==353 and cross['scope_was_not_redefined_as_complete'])
    demand=read(O/'operator-boundary-demand-and-theory-contract-v104.json')
    for r in demand['zero_local_LM_source_cases']:
        check('zero_source_case:'+str((r['external_launcher'],r['DP_more_than_one'],r['has_KV_transfer'])),
            r['prebranch_update_states_retained']and r['source_requires_DP_dummy_before_return']==(r['external_launcher']and r['DP_more_than_one'])and
            r['source_return']==('KV_CONNECTOR_NO_FORWARD'if r['has_KV_transfer']else'EMPTY_MODEL_OUTPUT')and not r['actual_predicates_attested'])
    check('theory_and_MFU_open',demand['theory']['status']=='INCOMPLETE'and demand['theory']['lowest_module_MFU'].startswith('NOT_ESTABLISHED')and demand['new_native_phase_bindings']==0)
    if a.preflight:manifest=read(O/'checkpoint-manifest-v103.json')
    else:
        cp=read(O/'REVIEW-CHECKPOINT-v104.json');check('PROGRESSnotclosed',cp['status']=='PROGRESS'and not cp['overall_complete'])
        now=read(O/'module-phase-branch-matrix-v104.json');check('353rows26semantic',sum(len(m['branches'])for m in now['modules'])==353 and all(m.get('branch_schema')=='SEMANTIC_BRANCHES_V1'for m in now['modules']))
        check('all_conditions_still_open',not now['all_conditions_closed']and not now['all_phases_closed']and not now['overall_complete'])
        for mid in ['M00','M03']:
            m=next(m for m in now['modules']if m['id']==mid);hist=read(Path(m['legacy_branch_history']))['module']
            check('archived_history:'+mid,hist==next(m for m in prev['modules']if m['id']==mid))
        audit=read(O/'whole-scope-requirement-audit-v104.json');q=next(r for r in audit['requirements']if r['id']=='ALL_MODULE_CONDITIONAL_BRANCHES')
        check('semantic_format_gap_closed_only',q['semantic_modules']==26 and q['branch_records']==353 and q['chronological_only']==[]and q['state']=='PARTIAL')
        for name,keys in [('operator-analysis-coverage-ledger',['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']),('candidate-relation-coverage',['scoped_conditional_pairs','higher_order_groups','remaining_pair_relations'])]:
            old=read(O/(name+'-v103.json'));new=read(O/(name+'-v104.json'))
            for k in keys:check('unchanged:'+name+':'+k,old[k]==new[k])
        for r in read(O/'candidate-record-index-v104.json')['model_candidates']:check('master:'+r['id'],hashlib.sha256(Path(r['master_record']).read_bytes()).hexdigest()==r['master_sha256'])
        manifest=read(O/'checkpoint-manifest-v104.json')
    for r in manifest['artifacts']:check('manifest:'+r['path'],pinned(r))
    result=dict(status='PASS',checks=len(checks),preflight=a.preflight,zero_scheduled_entries=64,old_rows_retained=26,new_semantic_rows=15,semantic_modules=26,actual_dummy_or_zero_device_work_attested=False,overall_complete=False)
    if a.preflight:write(O/'operator-boundary-analysis-preflight-v104.json',result)
    print(json.dumps(result))

if __name__=='__main__':main()
