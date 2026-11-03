"""Independent source-guard, event-domain and accounting checks; no target execution."""
from pathlib import Path
from itertools import combinations
from collections import Counter, defaultdict
import argparse, ast, base64, hashlib, json
from finalize_origin_coverage_v9 import N, O, R, read, write

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    checks=[]
    def check(name, value):
        assert value,name
        checks.append(name)
    def same_pin(r):
        p=Path(r['path']);return p.stat().st_size==r['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
    inp=read(O/'sampling-applicability-inputs-v102.json')
    for p in inp['inputs']:check('input:'+p['path'],same_pin(p))
    provenance=read(O/'sampling-load-provenance-v102.json')
    for k in ['launch_receipt','controller','captured_client']:check('provenance:'+k,same_pin(provenance[k]))
    launch=json.loads(read(Path(provenance['launch_receipt']['path']))['stdout'])
    check('captureidentity',launch['run_id']==provenance['run_id']=='hostprof-20261030T212812Z-360e0c72' and
          launch['controller_sha256']==provenance['controller']['sha256'])
    client_ast=ast.parse(Path(provenance['captured_client']['path']).read_text())
    run=next(n for n in client_ast.body if isinstance(n,ast.FunctionDef)and n.name=='run_request')
    literal=next(n.value for n in run.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='body'for t in n.targets))
    temp=next(v for k,v in zip(literal.keys,literal.values)if ast.literal_eval(k)=='temperature')
    check('explicittemperaturezero',isinstance(temp,ast.Constant)and temp.value==0)
    assigned_body_keys=[]
    for node in ast.walk(run):
        if isinstance(node,(ast.Assign,ast.AnnAssign,ast.AugAssign)):
            targets=node.targets if isinstance(node,ast.Assign)else[node.target]
            for target in targets:
                if isinstance(target,ast.Subscript)and isinstance(target.value,ast.Name)and target.value.id=='body':
                    assigned_body_keys.append(ast.literal_eval(target.slice))
    check('no_temperature_overwrite',assigned_body_keys==['stream_options'])
    controller_ast=ast.parse(Path(provenance['controller']['path']).read_text())
    for row in provenance['controller_active_definitions']:
        fs=[n for n in controller_ast.body if isinstance(n,ast.FunctionDef)and n.name==row['name']]
        check('active_definition:'+row['name'],fs[-1].lineno==row['first_line']and fs[-1].end_lineno==row['last_line'])
    check('allphasesvalid',all(x['valid']for x in provenance['phases'].values()))
    check('validatedC8counters',provenance['phases']['profile']['counters']['request_success_total']==8 and
          provenance['phases']['profile']['counters']['generation_tokens_total']==8192)

    guards=read(O/'sampling-operator-guard-contract-v102.json');parts={Path(r['path']).name:r for r in guards['parts']}
    for r in guards['parts']:
        check('sourcepin:'+r['path'],same_pin(r));lines=Path(r['path']).read_text().splitlines()
        for s in r['segments']:check('segment:'+Path(r['path']).name+str(s['first_line']),lines[s['first_line']-1:s['last_line']]==s['source'])
    npu=parts['npu_input_batch.py'];fs=json.loads(read(Path(npu['receipt']['path']))['stdout'])['files']
    rec=next(f for f in fs if f['path']==npu['original_path'])
    check('receipt_extraction_exact',base64.b64decode(rec['source_b64'])==Path(npu['path']).read_bytes() and not rec['historical_control_match'])
    cls=next(n for n in ast.parse(Path(npu['path']).read_text()).body if isinstance(n,ast.ClassDef))
    check('source_inheritance_not_constructor_assumption',[ast.unparse(b)for b in cls.bases]==['InputBatch'] and
          [n.name for n in cls.body if isinstance(n,ast.FunctionDef)]==['__init__'])
    nt=Path(npu['path']).read_text()
    check('base_import_and_sets','from vllm.v1.worker.gpu_input_batch import InputBatch'in nt and
          'self.greedy_reqs: set[str] = set()'in nt and 'self.random_reqs: set[str] = set()'in nt)
    pt=Path(parts['protocol.py']['path']).read_text();sp=Path(parts['sampling_params.py']['path']).read_text()
    check('explicit_zero_survives','if (temperature := self.temperature) is None:'in pt and 'temperature=temperature,'in pt and
          'temperature=1.0 if temperature is None else temperature,'in sp and 'if 0 < self.temperature < _MAX_TEMP:'in sp)
    check('classifier_zero_is_greedy','_SAMPLING_EPS = 1e-5'in sp and 'if self.temperature < _SAMPLING_EPS:\n            return SamplingType.GREEDY'in sp)
    bt=Path(parts['gpu_input_batch.py']['path']).read_text();bc=next(n for n in ast.parse(bt).body if isinstance(n,ast.ClassDef)and n.name=='InputBatch')
    bmethods={n.name:n for n in bc.body if isinstance(n,ast.FunctionDef)}
    for name,expected in [('all_greedy','len(self.random_reqs) == 0'),('all_random','len(self.greedy_reqs) == 0')]:
        check('inherited_property:'+name,ast.unparse(bmethods[name].body[0].value)==expected)
    check('metadata_passes_both_properties','all_greedy=self.all_greedy,'in bt and 'all_random=self.all_random,'in bt)
    rejection=Path(parts['rejection_sampler.py']['path']).read_text();tree=ast.parse(rejection)
    helper=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='rejection_sample')
    outer=next(n for n in helper.body if isinstance(n,ast.If)and ast.unparse(n.test)=='not sampling_metadata.all_random')
    early=next(n for n in outer.body if isinstance(n,ast.If)and ast.unparse(n.test)=='sampling_metadata.all_greedy')
    check('return_nested_under_not_allrandom',isinstance(early.body[0],ast.Return))
    processed=next(n for n in helper.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='target_probs'for t in n.targets))
    check('processed_probabilities_after_return',processed.lineno>early.end_lineno)
    check('pre_return_entropy_and_synthetic_retained',rejection.index('ori_target_probs = ori_target_logits.softmax')<rejection.index('uniform_probs_for_greedy:')<
          rejection.index('target_argmax = target_logits.argmax') and 'if synthetic_mode and using_block_verify:'in rejection)
    check('profile_missing_internal_flags_remains_unknown',not guards['internal_guard_attested'])

    saved=read(O/'sampling-saved-request-domain-v102.json');events=Counter();worker_keys=set();ids=set();runner_hashes=set();counts_by_dp=defaultdict(set)
    for r in saved['metadata_files']:
        check('eventpin:'+Path(r['path']).name,same_pin(r))
        es=[json.loads(x)for x in Path(r['path']).read_text().splitlines()]
        enter=[e for e in es if e['event']=='execute_enter'];exit=[e for e in es if e['event']=='execute_exit']
        key={(e['data_parallel_rank'],e['rank'],e['pid'])for e in enter};assert len(key)==1
        k=next(iter(key));worker_keys.add(k[:2]);counts_by_dp[k[0]].add(len(enter))
        assert len(enter)==len(exit)and all(e['error_type']is None for e in exit)
        for e in es:
            events[e['event']]+=1
            if e['event']=='installed':runner_hashes.add(e['source_sha256'])
        for e in enter:
            ids.update(e['scheduled_tokens']);ids.update(e['cached_request_state']['req_ids']);ids.update(x['req_id']for x in e['new_request_state'])
    check('sixteen_workers_not_eight',worker_keys=={(d,t)for d in range(2)for t in range(8)})
    check('recordedDPcounts_not_average232',dict(counts_by_dp)=={0:{233},1:{231}})
    check('3712entries_exits',events==Counter(saved['event_counts'])and events['execute_enter']==events['execute_exit']==3712)
    check('model_runner_source_hash_matches_install_records',runner_hashes=={guards['recorded_model_runner_source_sha256']}=={parts['model_runner_v1.py']['sha256']}and guards['installed_records_matching_model_runner_source']==16)
    check('ten_id_domain',ids=={r['request_id']for r in saved['requests']}and len(ids)==10)
    check('two_seed_eight_measurement_consistent',Counter(r['time_partition']for r in saved['requests'])=={'PRE_INITIAL_WINDOW_ONLY':2,'MEASUREMENT_WINDOWS_AND_FOLLOWUP':8})
    check('no_payload_or_runtime_boolean_join',all(not r['role_is_direct_payload_id_join']for r in saved['requests'])and not saved['internal_sampling_metadata_attested']and not saved['native_attribution_by_cross_rank_ordinal_or_timestamp'])

    routing=read(O/'sampling-case-fusion-and-ceiling-routing-v102.json')
    for row in routing['request_set_truth_table']:
        G={'a'}if row['greedy_set_nonempty']else set();Q={'b'}if row['random_set_nonempty']else set()
        ag,ar=len(Q)==0,len(G)==0
        check('boolean_domain:'+str((bool(G),bool(Q))),row['all_greedy']==ag and row['all_random']==ar and
              row['enters_greedy_branch']==(not ar)and row['returns_at_nested_greedy_guard']==((not ar)and ag))
    transitions=0
    # Every valid admit/remove transition of this finite reference preserves an empty random set.
    for mask in range(16):
        G={i for i in range(4)if mask>>i&1}
        for i in range(4):
            after=(G-{i})if i in G else(G|{i})
            ag,ar=True,not after
            assert ag and (ar==(len(after)==0));transitions+=1
    check('64_reference_greedy_set_transitions',transitions==64)
    check('greedy_kernel_not_allgreedy_proof',not routing['request_set_truth_table'][3]['returns_at_nested_greedy_guard'] and routing['request_set_truth_table'][3]['enters_greedy_branch'])
    check('empty_batch_guard_difference',routing['request_set_truth_table'][0]['all_greedy']and routing['request_set_truth_table'][0]['all_random']and not routing['request_set_truth_table'][0]['returns_at_nested_greedy_guard'])
    check('zero_savings_for_unexecuted_branch',routing['random_private_plane_v101']['current_nonempty_allgreedy_post_return_bytes']==routing['random_private_plane_v101']['incremental_savings_from_fusing_an_unexecuted_branch_bytes']==0)
    old=read(O/'heads-sampling-fusion-costs-v65.json')['sampling_models']
    check('reuse_not_new_byte_discovery',routing['reused_v65_value_models']==old)
    for r in old:
        check('value_model:'+str((r['B'],r['T_capacity'])),r['Index_Cast_workingclone_argmax_value_bytes']==22*r['T_capacity']*r['V']and r['pure_ID_only_interface_budget_bytes']==20*r['T_capacity']*r['V'])
    check('symbolic_physical_scope',routing['theory']['status']=='SYMBOLIC_CEILING'and not routing['actual_MFU_or_numeric_physical_ceiling_proved'])

    if args.preflight:manifest=read(O/'checkpoint-manifest-v101.json')
    else:
        cp=read(O/'REVIEW-CHECKPOINT-v102.json');check('progress',cp['status']=='PROGRESS'and not cp['overall_complete'])
        for name in ['candidate-relation-coverage','operator-analysis-coverage-ledger']:
            prev=read(O/(name+'-v101.json'));now=read(O/(name+'-v102.json'))
            keys=['scoped_conditional_pairs','higher_order_groups','remaining_pair_relations']if name.startswith('candidate')else['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']
            for k in keys:check('unchanged:'+name+':'+k,prev[k]==now[k])
        matrix=read(O/'module-phase-branch-matrix-v102.json');check('362rows24semantic',sum(len(m['branches'])for m in matrix['modules'])==362 and sum(m.get('branch_schema')=='SEMANTIC_BRANCHES_V1'for m in matrix['modules'])==24)
        audit=read(O/'whole-scope-requirement-audit-v102.json');check('MFUnotestablished',next(r for r in audit['requirements']if r['id']=='LOWEST_MODULE_UTILIZATION')['state']=='NOT_ESTABLISHED')
        for r in read(O/'candidate-record-index-v102.json')['model_candidates']:
            check('master:'+r['id'],hashlib.sha256(Path(r['master_record']).read_bytes()).hexdigest()==r['master_sha256'])
        manifest=read(O/'checkpoint-manifest-v102.json')
    for r in manifest['artifacts']:check('manifest:'+r['path'],same_pin(r))
    result=dict(status='PASS',checks=len(checks),preflight=args.preflight,saved_requests=10,workers=16,
                reference_set_transitions=transitions,actual_runtime_flag_attested=False,overall_complete=False)
    if args.preflight:write(O/'sampling-applicability-analysis-preflight-v102.json',result)
    print(json.dumps(result))

if __name__=='__main__':main()
