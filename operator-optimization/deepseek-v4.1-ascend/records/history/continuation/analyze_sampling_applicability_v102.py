"""Bind saved client/load evidence to operator guards; no imports of target/runtime code."""
from pathlib import Path
from collections import Counter, defaultdict
import ast, base64, hashlib, json
from finalize_origin_coverage_v9 import N, O, W, read, write

CACHE = N/'original-profile-cache-v53'

def pin(p):
    p = Path(p)
    return dict(path=str(p), bytes=p.stat().st_size,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def section(p, ranges, **extra):
    lines = Path(p).read_text().splitlines()
    return dict(**pin(p), segments=[dict(first_line=a, last_line=b, source=lines[a-1:b])
                                    for a,b in ranges], **extra)

def last_function(text, name):
    return [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name==name][-1]

def source_segment(text, node):
    return '\n'.join(text.splitlines()[node.lineno-1:node.end_lineno])

def main():
    assert not (O/'REVIEW-CHECKPOINT-v102.json').exists()
    capability = CACHE/'analysis/profile-capabilities-index-v12-v2.json'
    cap = read(capability)
    launch_path = N/'launch-host-profile166-v12.json'
    launch_receipt = read(launch_path)
    launch = json.loads(launch_receipt['stdout'])
    assert launch_receipt['returncode']==0
    assert launch['run_id']==cap['summary']['run_id']=='hostprof-20261030T212812Z-360e0c72'
    controller = N/'host_profile_controller166_v12.py'
    assert pin(controller)['sha256']==launch['controller_sha256']=='107b8714e924ef47cd5655cfec96a6e539cda82c56601949a128544b21c25b57'
    client = CACHE/'runtime/official_profile_client.py'
    assert pin(client)['sha256']=='42a6d9e54ea0a922843e8ab37f4096e52c54fb4dbcdd23b77bb60d69293f3131'
    ctext=controller.read_text()
    prepare=last_function(ctext,'prepare'); client_launch=last_function(ctext,'client')
    assert "Frozen client source changed" in source_segment(ctext,prepare)
    assert pin(client)['sha256'] in source_segment(ctext,prepare)
    assert "shutil.copy2(client_source,dst/'official_profile_client.py')" in source_segment(ctext,prepare)
    assert "'WIO_PHASE='+phase" in source_segment(ctext,client_launch)
    body_nodes=[n for n in ast.walk(ast.parse(client.read_text())) if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id=='body' for t in n.targets)]
    assert len(body_nodes)==1 and isinstance(body_nodes[0].value,ast.Dict)
    body=body_nodes[0].value
    fields={ast.literal_eval(k):v for k,v in zip(body.keys,body.values)}
    assert ast.literal_eval(fields['temperature'])==0
    # These records attest a frozen client/capture domain, not a network packet or runtime boolean.
    provenance=dict(run_id=launch['run_id'], launch_receipt=pin(launch_path), controller=pin(controller),
        captured_client=pin(client), controller_active_definitions=[
            dict(name=n.name, first_line=n.lineno,last_line=n.end_lineno) for n in [prepare,client_launch]],
        client_request_body_temperature=0, client_all_run_request_calls_share_this_body=True,
        client_seeds_complete_before_profile_start=True,
        phases={k:dict(valid=v['valid'], input_sha256=v['input_sha256'], counters=v['counter_deltas'])
                for k,v in cap['clients'].items()},
        evidence_grade='DOCUMENTED_CAPTURE_ASSETS_AND_DERIVED_REQUEST_CONTRACT',
        source_scope='Saved client and historical controller receipt only; controller/client are parsed, never imported or executed.',
        limitation='No per-request HTTP body capture, effective SamplingParams or all_greedy/all_random scalar recorded. Pinned request intent is not direct runtime flag attestation.',
        new_server_or_device_work=False, overall_complete=False)
    write(O/'sampling-load-provenance-v102.json', provenance)

    windows=cap['clients']['profile']['windows']; records={}; workers=[]; events=Counter(); files=[]
    for descriptor in cap['metadata']:
        path=CACHE/'runtime/host-profile'/Path(descriptor['path']).name
        assert pin(path)['sha256']==descriptor['sha256'] and path.stat().st_size==descriptor['bytes']
        es=[json.loads(line) for line in path.read_text().splitlines()]
        assert es==descriptor['entries']
        enters=[e for e in es if e['event']=='execute_enter']; exits=[e for e in es if e['event']=='execute_exit']
        assert len(enters)==len(exits) and all(e['error_type'] is None for e in exits)
        keys={(e['data_parallel_rank'],e['rank'],e['pid']) for e in enters};assert len(keys)==1
        key=next(iter(keys));assert {e['tensor_parallel_size'] for e in enters}=={8}
        assert {e['speculative_method'] for e in enters}=={'dspark'}
        assert {e['scheduled_spec_tokens'] for e in enters}=={5}
        assert {e['call_id'] for e in enters}==set(range(1,len(enters)+1))
        assert {e['call_id'] for e in exits}==set(range(1,len(enters)+1))
        workers.append(dict(dp_rank=key[0],local_tp_rank=key[1],pid=key[2],execute_entries=len(enters),
                            request_ids=sorted({r for e in enters for r in e['scheduled_tokens']})))
        for e in es: events[e['event']]+=1
        for e in enters:
            observed=set(e['scheduled_tokens'])|set(e['cached_request_state']['req_ids'])|{x['req_id'] for x in e['new_request_state']}
            # Record identity unions; no scheduler ordinal is used to attribute native operators.
            for req in observed:
                r=records.setdefault(req,dict(request_id=req,worker_keys=set(),first_seen_ns=10**30,last_seen_ns=0,
                    scheduled_in_windows=Counter(),new_request_computed_tokens=set(),scheduled_tokens_by_worker=Counter()))
                r['worker_keys'].add(key[:2]);r['first_seen_ns']=min(r['first_seen_ns'],e['monotonic_ns'])
                r['last_seen_ns']=max(r['last_seen_ns'],e['monotonic_ns'])
                if req in e['scheduled_tokens']:
                    r['scheduled_tokens_by_worker'][key[:2]]+=e['scheduled_tokens'][req]
                    for w in windows:
                        if w['start_ns']<=e['monotonic_ns']<=w['stop_ns']:
                            r['scheduled_in_windows'][w['name']]+=1
                for x in e['new_request_state']:
                    if x['req_id']==req:r['new_request_computed_tokens'].add(x['num_computed_tokens'])
        files.append(pin(path))
    for r in records.values():
        r['worker_keys']=[list(k)for k in sorted(r['worker_keys'])]
        r['new_request_computed_tokens']=sorted(r['new_request_computed_tokens'])
        r['scheduled_in_windows']=dict(r['scheduled_in_windows'])
        r['scheduled_tokens_by_worker']=[dict(dp_rank=k[0],local_tp_rank=k[1],tokens=v)
                                         for k,v in sorted(r['scheduled_tokens_by_worker'].items())]
        r['time_partition']='PRE_INITIAL_WINDOW_ONLY' if r['last_seen_ns']<windows[0]['start_ns'] else 'MEASUREMENT_WINDOWS_AND_FOLLOWUP'
        r['role']='PREFIX_SEED_CONSISTENT_WITH_CLIENT' if r['time_partition']=='PRE_INITIAL_WINDOW_ONLY' else 'MEASUREMENT_REQUEST_CONSISTENT_WITH_CLIENT'
        r['role_is_direct_payload_id_join']=False
    assert len(workers)==16 and {(w['dp_rank'],w['local_tp_rank']) for w in workers}=={(d,t)for d in range(2)for t in range(8)}
    assert len(records)==10
    assert Counter(r['time_partition']for r in records.values())=={'PRE_INITIAL_WINDOW_ONLY':2,'MEASUREMENT_WINDOWS_AND_FOLLOWUP':8}
    assert all(len(r['worker_keys'])==8 and len({x[0]for x in r['worker_keys']})==1 for r in records.values())
    pre=[r for r in records.values()if r['time_partition']=='PRE_INITIAL_WINDOW_ONLY']
    post=[r for r in records.values()if r['time_partition']!='PRE_INITIAL_WINDOW_ONLY']
    assert all(r['new_request_computed_tokens']==[0] and {x['tokens']for x in r['scheduled_tokens_by_worker']}=={29696} for r in pre)
    assert all(r['first_seen_ns']>windows[0]['start_ns'] and r['scheduled_in_windows'].get('initial') and r['scheduled_in_windows'].get('later_decode_candidate')for r in post)
    request_domain=dict(metadata_files=files, workers=sorted(workers,key=lambda r:(r['dp_rank'],r['local_tp_rank'])),
        event_counts=dict(events), requests=sorted(records.values(),key=lambda r:r['request_id']),
        request_union=10,pre_initial_only_requests=2,measurement_window_requests=8,
        worker_count=16,execute_entries=3712,local_tp_rank_is_not_global_rank=True,
        recorded_guard_fields=[], internal_sampling_metadata_attested=False,
        evidence_grade='MEASURED_SAVED_SCALAR_SCHEDULE_METADATA',
        limitation='Ten IDs close the saved event surface only. Two pre-window requests agree with prefix seeding; eight later IDs agree with the validated C8 run. IDs alone do not prove HTTP body contents, absence of outside clients, dummy/capture coverage, or actual internal greedy flags. TP replicas must not be summed as unique token work.',
        native_attribution_by_cross_rank_ordinal_or_timestamp=False,overall_complete=False)
    write(O/'sampling-saved-request-domain-v102.json',request_domain)

    # Resolve only the inheritance and fields selecting sampling operators.
    receipt_path=N/'sampling-guard-source166-v55.json';receipt=read(receipt_path)
    fs=json.loads(receipt['stdout'])['files']
    f=next(x for x in fs if x['path'].endswith('/npu_input_batch.py'))
    raw=base64.b64decode(f['source_b64']);assert hashlib.sha256(raw).hexdigest()==f['sha256']
    extraction=O/'sampling-applicability-source-v102/npu_input_batch.py'
    extraction.parent.mkdir(exist_ok=True)
    if extraction.exists():assert extraction.read_bytes()==raw
    else:extraction.write_bytes(raw)
    cls=next(x for x in ast.parse(raw.decode()).body if isinstance(x,ast.ClassDef))
    assert cls.name=='NPUInputBatch' and [ast.unparse(x)for x in cls.bases]==['InputBatch']
    assert [x.name for x in cls.body if isinstance(x,ast.FunctionDef)]==['__init__']
    protocol=W/'frozen-source-snapshot/vllm/vllm/entrypoints/openai/completion/protocol.py'
    params=W/'contract-source-supplement/frozen-source-snapshot/vllm/vllm/sampling_params.py'
    batch=W/'frozen-source-snapshot/vllm/vllm/v1/worker/gpu_input_batch.py'
    runner=W/'frozen-source-snapshot/vllm-ascend/vllm_ascend/worker/model_runner_v1.py'
    rejection=W/'sampler-source-supplement/frozen-source-snapshot/vllm-ascend/vllm_ascend/sample/rejection_sampler.py'
    installed=[e for f in cap['metadata']for e in f['entries']if e['event']=='installed']
    assert len(installed)==16 and {e['source_sha256']for e in installed}=={pin(runner)['sha256']}
    parts=[section(client,[(34,42),(81,89),(133,136)]),section(controller,[(470,472),(511,515)]),
        section(protocol,[(330,333),(369,381)]),section(params,[(27,27),(454,456),(487,497),(533,538),(782,788)]),
        section(extraction,[(33,39),(146,153),(244,245)],receipt=pin(receipt_path),
            original_path=f['path'],historical_control_match=f['historical_control_match'],
            provenance_limit='Saved v55 source body; historical_control_match=False is not a pinned original loaded-module identity, nor evidence of a mismatch by itself.'),
        section(batch,[(397,406),(561,564),(939,945),(1113,1120)]),
        section(runner,[(238,238),(611,618),(2972,2999)]),
        section(rejection,[(209,222),(321,346),(435,447),(461,490),(509,554),(556,580)])]
    facts=[
        dict(id='REQ0',grade='DERIVED_FROM_PINNED_CLIENT',fact='All run_request calls submit explicit numeric zero temperature, for prefix seeds and measured requests. No truthiness-default assumption.'),
        dict(id='PARAM0',grade='DERIVED_IN_PINNED_SOURCE_DOMAIN',fact='Explicit zero bypasses only-None default; SamplingParams classifies temperature<1e-5 as GREEDY and normalizes top_p=1, top_k=0, min_p=0.'),
        dict(id='BATCH0',grade='DERIVED_IN_SOURCE_MRO_DOMAIN',fact='NPUInputBatch initializes empty greedy/random sets and defines only __init__; base add/remove/_make_sampling_metadata/all_greedy/all_random are inherited in this source MRO. Valid admitted zero-temperature requests populate greedy_reqs only; removal discards from both.'),
        dict(id='NONEMPTY0',grade='DERIVED_CONDITIONAL_INVARIANT',fact='If every valid active request has that contract, metadata is current, and the active greedy set is nonempty, random_reqs is empty: all_greedy=True and all_random=False. Empty active set instead makes both True and is not licensed as the same helper execution.'),
        dict(id='HELPER0',grade='DERIVED_IN_SOURCE_DOMAIN',fact='all_greedy skips temperature/top-k/top-p expansion in apply_sampling_constraints. In rejection_sample, !all_random enters target argmax/greedy writer; nested all_greedy returns before processed target softmax, ordinary uniform, recovery and random/block verifier.'),
        dict(id='KEEP0',grade='DOCUMENTED_SOURCE_ORDER',fact='Raw/working processors and snapshots precede helper; synthetic&&block error guard, optional ori softmax, output Fill and optional synthetic uniforms precede return. These cannot all be pruned from temperature alone.'),
        dict(id='NATIVE0',grade='MEASURED_SAVED_PROFILE',fact='72 observed greedy kernel chains corroborate greedy-capable execution; mixed requests also enter that kernel, so its presence does not identify all_greedy.'),
        dict(id='EPOCH0',grade='UNKNOWN',fact='Actual per-call effective flags, patched/loaded class/module identity, closed external admission, metadata epoch and unbound/dummy/capture cases are not directly attested. Source implication is not a global promotion.')]
    write(O/'sampling-operator-guard-contract-v102.json',dict(parts=parts,facts=facts,
        recorded_model_runner_source_sha256=pin(runner)['sha256'],
        installed_records_matching_model_runner_source=16,
        source_binding_limit='All saved installed records report the same model_runner source hash as this snapshot; no equivalent per-call source/flag attestation for InputBatch, SamplingParams or rejection helper is present.',
        guards_for_sealed_nonempty_greedy_domain=['explicit zero admitted unchanged','all valid active requests originate in this contract',
            'chosen pinned source/MRO selects these bodies without runtime overrides','current metadata epoch, valid request ownership and nonempty greedy set',
            'all earlier observable checks/processors/synthetic/raw/logprob/entropy/RNG effects preserved'],
        internal_guard_attested=False,whole_module_closed=False,overall_complete=False))

    prior_costs=read(O/'heads-sampling-fusion-costs-v65.json')['sampling_models']
    routing=[
        dict(id='GREEDY_SEALED',domain='Valid nonempty source-bound all-greedy request set',
             constraints_temperature_topk_topp_body=False,rejection_post_return_random_body=False,
             original_post_return_random_work=0,
             decision='Keep greedy Index/Cast/working clone/ArgMax/Fill/verify candidates. Post-return softmax/recovery fusion is an alternative-domain candidate; no current savings credit.'),
        dict(id='GREEDY_ENTROPY',domain='GREEDY_SEALED plus enable_entropy_verify and ori logits present',
             pre_return_ori_softmax_created=True,post_return_ori_content_consumed=False,
             decision='v33 dead-content candidate remains conditional; a successful content-elision proof preserves prior errors and all other readers. Temperature alone does not establish flag False.'),
        dict(id='GREEDY_SYNTHETIC',domain='GREEDY_SEALED plus synthetic mode, valid non-block config',
             pre_return_uniform_draws=True,
             decision='Preserve original greedy synthetic draw order/rates; do not claim temperature zero removes all RNG.'),
        dict(id='MIXED_OR_RANDOM',domain='At least one valid non-greedy active request',
             rejection_post_return_random_body=True,
             decision='Random/recovery/filter candidate graphs remain conditional; count actual selected read sets and backend-specific precision once.'),
        dict(id='EMPTY_SET',domain='Both request sets empty',all_greedy=True,all_random=True,
             same_as_nonempty_greedy=False,
             decision='No nonempty greedy early-return proof applies. Do not infer a valid B=0 helper ABI or a runtime dummy execution from this truth-table row.')]
    write(O/'sampling-case-fusion-and-ceiling-routing-v102.json',dict(domains=routing,
        request_set_truth_table=[dict(greedy_set_nonempty=g,random_set_nonempty=r,
            all_greedy=not r,all_random=not g,enters_greedy_branch=g,
            returns_at_nested_greedy_guard=g and not r,
            reaches_post_return_body_by_boolean_guards=not(g and not r),
            source_guards_only_not_helper_validity=True)for g in [False,True]for r in [False,True]],
        reused_v65_value_models=prior_costs,
        random_private_plane_v101=dict(materialized_reference_bytes='48*T*V',
            current_nonempty_allgreedy_post_return_bytes=0,
            incremental_savings_from_fusing_an_unexecuted_branch_bytes=0,
            limitation='Zero is only post-return body reachability in the stated source domain; not all probability/RNG work, actual runtime flags or whole-module cost.'),
        theory=dict(status='SYMBOLIC_CEILING',
            chosen_reference_domain='K explicitly required independent finite full-vocab score rows, exact IDs under the chosen comparator; no prior winner certificate or exploitable value structure.',
            comparison_work_lower='K*(V-1) comparisons in a comparison model; K*V observations of the independent score inputs.',
            fused_BF16_score_load_value_reference='2*K*V bytes if values start in a BF16 source accessible to the fused reduction; not a necessary HBM/L2 transfer claim.',
            latency_lower='max(L_comparison_model_typed_capacity, L_necessary_storage_by_layer, L_required_global_argmax_communication, L_proven_dependencies)',
            caveats='Chosen K is not padded T capacity or necessarily all head rows. No numeric typed comparison/communication/storage upper capacity, native comparator, all-reader/alias/epoch closure or critical-path speedup is proved. Greedy-domain routing does not tighten unrelated module ceilings.'),
        no_new_candidate_or_native_regime=True,actual_MFU_or_numeric_physical_ceiling_proved=False,
        no_framework_research=True,overall_complete=False))
    originals=[capability,launch_path,controller,client,receipt_path,protocol,params,batch,runner,rejection,
        O/'heads-sampling-fusion-costs-v65.json',O/'heads-sampling-native-chains-v65.json',
        O/'M17-probability-consumer-and-RNG-contract-v33.json',O/'M17-current-probability-and-guard-scope-v33.json',
        O/'whole-scope-requirement-audit-v101.json',O/'recovery-fallback-work-and-interface-models-v101.json']
    write(O/'sampling-applicability-inputs-v102.json',dict(inputs=[pin(p)for p in originals]+files,
        new_source_files_from_target=0,new_public_documents=0,
        new_work='Bind historical frozen client/controller and ten saved request identities to the exact nonempty-greedy operator domain; separate unreachable-body savings from applicable fusion work.',
        prior_facts_not_new=['v33 pre-return entropy and synthetic guards','v65 pure-ID byte budgets','v101 private recovery planes'],
        no_model_Torch_NPU_server_capture_or_implementation=True,overall_complete=False))
    print(json.dumps(dict(status='ANALYZED',saved_requests=10,workers=16,metadata_entries=3712,
        runtime_guard_attested=False,new_native_promotions=0,overall_complete=False)))

if __name__=='__main__':main()
