"""Local L13 D two-step native diagnostic audit; evidence only, never a certificate.

Reads frozen local reports and captured node JSON only. No SSH, network, model,
SDK, NPU or Git action. Reuses only the pure unpack/sdk_counts helpers from the
frozen L12D native audit; never calls its audit/main.
"""
import collections,hashlib,importlib.util,json,re,statistics,types
from decimal import Decimal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RID='20261019T-L13-sglang-C32-D-native-decode-two-step-profile-v1'
CAP='20261019T-L13-sglang-C32-D-two-step-native-v1'
BEFORE='20261019T-L12C-sglang-native-formal-capture-v1'
CONFIG='config/C32-L13-D-native-decode-v1/D-two-step-diagnostic-v1.json'
CONFIG_SHA='035d37267b9895790d0a733a4a219d15d15262382a211064305303b4aa2b9ee2'
RAW_REPORT=f'runs/{RID}/diagnostic-result.remote.json'
RAW_REPORT_SHA='95b333569d0614a685c211edd3792711746382ea6c97b9eb7f5179da0f03ad86'
RESULT='results/C32-L13-D-two-step-evidence-v1.json'
MANIFEST_PROOF='runs/20261017T2056-sglang-C32-fifth-suffix-inputs-v1/manifest-proof.json'
NATIVE_HELPER='scripts/audit_C32_L12D_native_formal_native_local_v2.py'
NATIVE_HELPER_SHA='5bc9bd543d421e0469d85f5ab62db2f440593e40654e5203d30a3137fef33356'
CACHE_HELPER='scripts/glm53_sglang_cache_annotation_review_local_v1.py'
CACHE_HELPER_SHA='9f408fdfd1df85d1d6ea9e3431d0f3e4dfd171cd50cc67ca3b11a0e5e0fbdb76'
CACHE_SOURCE='runs/20261017T1896-sglang-cache-annotation-source-v1/sglang.json'
PROFILER_SOURCE='runs/20261017T1818-native-profile-source-review-v1/source/sglang/sglang__srt__managers__scheduler_components__profiler_manager.py'
PROFILER_SOURCE_SHA='a962ab8283b468d6c3f10e53527613ea79569490cfdef340f3c22ffbee6bdcc3'
MODEL_RUNNER_SOURCE='runs/20261018T-C32-alignment-installed-source-v1/source/sglang/sgl-workspace__sglang__python__sglang__srt__model_executor__model_runner.py'
MODEL_RUNNER_SOURCE_SHA='df648164f669f9bbf823fa63e4ba7d7f3e27205eed301235364809e90baf8ce2'
STOP_ERROR="AssertionError('Native stop did not acknowledge completion or already-inactive state')"
OPTIONAL_EXTENSION='/usr/local/python3.12.13/lib/python3.12/site-packages/memfabric_hybrid/lib/libmf_hybm_accoffload.so'
CSV_NAMES=['api_statistic.csv','kernel_details.csv','op_statistic.csv','operator_details.csv','step_trace_time.csv','task_time.csv']
def read(relative):return json.loads((ROOT/relative).read_text())
def sha(relative):return hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()
def load(relative,digest,name):
    assert sha(relative)==digest
    spec=importlib.util.spec_from_file_location(name,ROOT/relative);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def distribution(values):
    assert values
    return dict(min=min(values),median=statistics.median(values),max=max(values),ranks=len(values))
def align_rows(obj):
    rows=[]
    for item in obj['alignment_audit'].values():
        raw=''.join(json.dumps(x,separators=(',',':'))+'\n' for x in item['rows']).encode()
        assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
        rows.extend(item['rows'])
    return rows
def assert_prefix(new_rows,old_rows,scope):
    assert new_rows[:len(old_rows)]==old_rows,f'{scope}: existing audit prefix changed'
def audit_graph(o,previous,initialized,node):
    aligned=align_rows(o)
    if node not in [36,41]:
        assert not aligned,'P nodes must not carry target Graph audit rows'
        return None
    for name,item in previous['alignment_audit'].items():
        assert name in o['alignment_audit'],f'node{node} alignment audit file missing: {name}'
        assert_prefix(o['alignment_audit'][name]['rows'],item['rows'],f'node{node} {name}')
    all_pids={x['pid'] for x in aligned};assert set(initialized)<=all_pids
    extras=all_pids-set(initialized)
    assert all(x['event'] in ['ops_compat_installed','target_graph_audit_installed','draft_compat_installed'] for x in aligned if x['pid'] in extras)
    installation={str(pid):dict(collections.Counter(x['event'] for x in aligned if x['pid']==pid)) for pid in sorted(extras)}
    older=align_rows(previous);productive_graph={};retention={}
    for pid in initialized:
        rows=[x for x in aligned if x['pid']==pid]
        assert all(x['revision']=='glm53-C32-target-graph-fullmesh-v2' for x in rows)
        for method in ['_npu_low_latency_dispatch','_npu_low_latency_combine']:
            first=[x for x in rows if x['event']=='ops_first_return' and x['method']==method]
            assert len(first)==1 and first[0]['group_size']==32 and first[0]['comm_alg']=='fullmesh_v1'
            if method.endswith('dispatch'):
                assert first[0]['quant_mode']==2 and first[0]['inputs']['topk_weights'] is None and first[0]['outputs'][0]['dtype']=='torch.int8'
            else:
                assert first[0]['comm_quant_mode']==0 and first[0]['inputs']['topk_weights']['shape']==[32,8] and first[0]['outputs']['dtype']=='torch.bfloat16'
        assert any(x['event']=='draft_graph_disabled' for x in rows),'Draft Graph must remain disabled'
        assert any(x['event']=='target_graph_decision' and x['eligible'] is True and x['mode']=='TARGET_VERIFY' and x['raw_bs']>0 for x in rows)
        assert not any(x['event']=='target_graph_replay_return' for x in rows)
        older_rows=[x for x in older if x['pid']==pid]
        assert_prefix(rows,older_rows,f'node{node} pid{pid}')
        new_rows=rows[len(older_rows):];decisions=[x for x in new_rows if x['event']=='target_graph_decision']
        productive=[x for x in decisions if x['eligible'] is True and x['mode']=='TARGET_VERIFY' and x['raw_bs']>0 and not x['is_draft']]
        assert productive,f'node{node} pid{pid}: no fresh productive TARGET_VERIFY decision'
        rank=next(x['rank'] for x in rows if x['event']=='ops_first_return');assert str(rank) not in productive_graph
        productive_graph[str(rank)]={'pid':pid,'productive_eligible_TARGET_VERIFY_decisions':len(productive),'productive_raw_bs_histogram':dict(collections.Counter(x['raw_bs'] for x in productive)),'ineligible_TARGET_VERIFY_decisions':sum(x['mode']=='TARGET_VERIFY' and x['eligible'] is False for x in decisions),'all_new_modes_histogram':dict(collections.Counter(x['mode'] for x in decisions)),'scope':'Native-qualified BEFORE snapshot to current post snapshot; covers this diagnostic without new warmup; complete original diagnostic D metric archive independently proves within-diagnostic Graph progression.'}
        retention[str(rank)]={'pid':pid,'draft_graph_disabled':True,'native_ops_fullmesh_v1':True,'productive_TARGET_VERIFY_decisions':len(productive)}
    native_graph={}
    for line in o['/metrics']['body'].splitlines():
        if not line.startswith('sglang:cuda_graph_passes_total{'):continue
        labels=dict(re.findall(r'(\w+)="([^"]*)"',line));rank=int(labels['moe_ep_rank'])
        assert labels['mode']=='decode_cuda_graph' and labels['engine_type']=='decode' and int(labels['tp_rank'])==rank and int(labels['dp_rank'])==rank//2 and rank not in native_graph
        native_graph[rank]=float(line.rsplit(' ',1)[1])
    assert set(native_graph)==set(range(16) if node==36 else range(16,32)) and all(x>0 for x in native_graph.values())
    for rank,value in native_graph.items():retention[str(rank)]['native_graph_counter']=value
    return {'productive_graph':productive_graph,'retention':retention,'installation_only_pids':installation}
def audit_profile(profile,initialized,node):
    assert profile['exists'] is True and profile['analysis_pass'] is True and profile['all16_local_model_workers_profiled'] is True
    assert profile['raw_traces_transferred_to_Mac'] is False and not profile.get('error') and RID in profile['root']
    assert len(profile['workers'])==16 and {x['namespace_pid'] for x in profile['workers'].values()}==set(initialized)
    assert {x['TP_rank'] for x in profile['workers'].values()}==set(range(16) if node==36 else range(16,32))
    assert {x['native_device_id'] for x in profile['workers'].values()}=={x['device_id'] for x in initialized.values()}
    index={x['relative']:x for x in profile['files']};assert len(index)==len(profile['files'])
    assert profile['total_bytes']==sum(x['bytes'] for x in profile['files'])
    expected_csv={'ASCEND_PROFILER_OUTPUT/'+name for name in CSV_NAMES};flat=[]
    for folder,worker in profile['workers'].items():
        assert type(worker['native_collection_duration_seconds']) in [int,float] and worker['native_collection_duration_seconds']>0
        source=worker['source_files'];assert len(source)==7 and expected_csv<=set(source)
        assert sum(Path(k).name.startswith('profiler_info_') and k.endswith('.json') for k in source)==1
        for relative,item in source.items():
            key=folder+'/'+relative
            assert key in index and item['bytes']==index[key]['bytes']>0 and item['sha256']==index[key]['sha256'] and re.fullmatch('[0-9a-f]{64}',item['sha256'])
        spans=worker['span_like_host_operator_aggregates'];assert spans;counts=collections.Counter()
        for row in spans:
            assert type(row['count']) is int and row['count']>0
            counts[row['name']]+=row['count']
        verify=sum(value for name,value in counts.items() if name.startswith('step[VERIFY '))
        draft=sum(value for name,value in counts.items() if name.startswith('step[DRAFT '))
        assert verify==2 and draft==10 and counts.get('draft')==2 and counts.get('draft_extend')==2,f'node{node} {folder}: span totals {verify}/{draft}/{counts.get("draft")}/{counts.get("draft_extend")}'
        flat.append(dict(node=node,folder=folder,**worker))
    markers=[x for x in profile['model_log_markers'] if RID in x['line']]
    starts=sum('Profiling starts' in x['line'] for x in markers);dones=sum('Profiling done.' in x['line'] for x in markers)
    assert starts==dones==16,f'INCONCLUSIVE: node{node} current-RID Profiling starts/done markers are {starts}/{dones}, expected16/16'
    return flat,{'current_RID_markers':len(markers),'starts':starts,'dones':dones,'worker_folders':len(profile['workers']),'indexed_files':len(profile['files']),'indexed_bytes':profile['total_bytes']}
def main():
    assert not (ROOT/RESULT).exists(),f'Output already exists: {RESULT}'
    assert sha(CONFIG)==CONFIG_SHA and sha(NATIVE_HELPER)==NATIVE_HELPER_SHA and sha(CACHE_HELPER)==CACHE_HELPER_SHA
    assert sha(PROFILER_SOURCE)==PROFILER_SOURCE_SHA and sha(MODEL_RUNNER_SOURCE)==MODEL_RUNNER_SOURCE_SHA
    settings=read(CONFIG);sources={CONFIG:CONFIG_SHA,NATIVE_HELPER:NATIVE_HELPER_SHA,CACHE_HELPER:CACHE_HELPER_SHA,PROFILER_SOURCE:PROFILER_SOURCE_SHA,MODEL_RUNNER_SOURCE:MODEL_RUNNER_SOURCE_SHA,CACHE_SOURCE:sha(CACHE_SOURCE)}
    assert settings['project_id']=='glm53' and settings['framework']=='sglang' and settings['concurrency']==32
    assert settings['run_id']==RID and settings['diagnostic_only'] is True and settings['performance_certificate_allowed'] is False
    client=settings['client_settings'];assert client['run_id']==RID and client['diagnostic_only'] is True and client['performance_certificate_allowed'] is False and client['concurrency']==32
    assert client['profile_settings']['run_id']==RID and client['profile_settings']['profile_body']['num_steps']==2
    assert RID in client['profile_settings']['profile_body']['output_dir'] and client['profile_settings']['profile_body']['profile_id']==RID
    assert set(settings['files'])==set(settings['files_sha256'])
    for name,relative in settings['files'].items():
        assert sha(relative)==settings['files_sha256'][name],f'Client source changed: {relative}'
        sources[relative]=settings['files_sha256'][name]
    for relative,digest in settings['prerequisite_sha256'].items():
        assert sha(relative)==digest,f'Prerequisite changed: {relative}'
        sources[relative]=digest
    assert set(settings['runtime_containers'])=={'34','35','36','41'}
    native=load(NATIVE_HELPER,NATIVE_HELPER_SHA,'native_helper');cache=load(CACHE_HELPER,CACHE_HELPER_SHA,'cache_helper')
    assert sha(RAW_REPORT)==RAW_REPORT_SHA
    report=read(RAW_REPORT);sources[RAW_REPORT]=RAW_REPORT_SHA
    assert report['project_id']=='glm53' and report['framework']=='sglang' and report['run_id']==RID and report['concurrency']==32
    assert report['diagnostic_only'] is True and report['performance_certificate_allowed'] is False and report['phase']=='COMPLETE'
    assert report['cache_namespace']==client['cache_namespace'] and report['input_manifest_sha256']==client['input_manifest_sha256']
    assert report['prior_qualification_report_sha256']==client['prior_report_sha256']==sha(f"runs/{client['prior_run_id']}/formal-result.remote.json")
    assert report['warmups']==0 and report['automatic_model_request_retries'] is False and report['SDK_init_calls']==0
    assert report['backend_HTTP_attempts']==64 and len(report['gateway'])==len(report['rows'])==32
    assert report['count_and_terminal_successes']==32 and report['pending_terminal_ids']==[]
    control=report['profile_control'];assert control['project_id']=='glm53' and control['run_id']==RID and control['diagnostic_only'] is True
    assert control['model_requests']==control['SDK_init_calls']==0 and control['profile_start_attempts']==control['profile_stop_attempts']==1
    start=control['start_response'];assert start['status']==200 and not control.get('error')
    raw_control_pass=control['pass'];stop=control.get('stop_response',{})
    duplicate_stop_pending_native=(not raw_control_pass and stop.get('status')==500 and stop.get('body')=='Internal Server Error' and control.get('stop_error')==STOP_ERROR)
    if raw_control_pass:assert stop.get('status')==200 and not control.get('stop_error')
    else:assert duplicate_stop_pending_native,'Other native control failures remain unqualified'
    assert report['D_metrics']['sample_count']==report['D_metrics']['complete_sample_count']>0
    proof=read(MANIFEST_PROOF);sources[MANIFEST_PROOF]=sha(MANIFEST_PROOF)
    assert proof['manifest_sha256']==client['input_manifest_sha256']
    manifest=proof['manifest'];assert manifest['concurrency']==32 and manifest['requests']==32 and manifest['warmups']==0 and manifest['fresh_suffix_lcp_exact'] is True
    expected={x['prefix_id']:x for x in manifest['rows']};assert set(expected)==set(range(32))
    assert {x['prefix_id'] for x in report['rows']}==set(range(32))
    cache_rows=[];D_storage_null_prefix_ids=[];output=0;p_output=0;mtp=dict(verify_steps=0,proposed_drafts=0,correct_drafts=0,histogram=[0]*6)
    formatter=cache.source_formatter()
    for row in report['rows']:
        pid=row['prefix_id'];meta=expected[pid];n=row['native'];extra=n['extra_details_data'];usage=extra['glm53_service_usage'];req=extra['glm53_request'];gw=row['terminal']['gateway']
        assert row['variant']==5 and meta['variant']==5 and meta['lcp_against_previous_four']==[74400]*4
        assert row['input_sha256_le_u32']==meta['ids_sha256_le_u32']==req['input_sha256_le_u32']
        assert row['count_and_terminal_pass'] and n['success']
        assert n['input_tokens']==usage['prompt_tokens']==req['input_tokens']==80000
        assert n['output_tokens']==usage['completion_tokens']==req['output_target']==600 and usage['total_tokens']==80600
        assert extra['glm53_finish_reason']=='length' and req['native_attempt_setting']==1
        assert req['namespace']==report['cache_namespace']==gw['namespace']==client['cache_namespace']
        assert row['request_id']==f"{client['cache_namespace']}-diag-L13-D-p{pid:02d}-v5"
        assert gw['prefill_usage']['prompt_tokens']==80000 and gw['prefill_usage']['completion_tokens']==1
        assert gw['status']=='D_STREAM_COMPLETE' and gw['prefill_http_status']==gw['decode_http_status']==200 and gw['logical_request_id']==row['request_id']
        P=gw['prefill_sglext']['cached_tokens_details'];D=extra['glm53_native_sglext']['cached_tokens_details'];checked=cache.check_tiers(P,D)
        request=types.SimpleNamespace(cached_tokens_device=P['device'],cached_tokens_host=P['host'],cached_tokens_storage=P['storage'],cached_tokens=sum(P[k] for k in ['device','host','storage']))
        for enabled,actual in [(True,P),(False,D)]:
            owner=types.SimpleNamespace(enable_hicache_storage=lambda:enabled,_get_storage_backend_type=lambda:'NpuMemcacheStore');assert formatter(owner,request)==actual
        cache_rows.append(dict(prefix_id=pid,**checked));D_storage_null_prefix_ids.extend([pid] if not checked['D_storage_field_present'] else [])
        output+=n['output_tokens'];p_output+=gw['prefill_usage']['completion_tokens']
        spec=extra['glm53_native_sglext']['spec_tokens_details'];verify=spec['spec_verify_ct'];proposed=spec['spec_num_proposed_drafts'];correct=spec['spec_num_correct_drafts'];hist=spec['spec_correct_drafts_histogram']
        assert verify>0 and proposed==5*verify and 0<=correct<=proposed and len(hist)==6 and all(type(x) is int and x>=0 for x in hist) and sum(hist)==verify and sum(i*x for i,x in enumerate(hist))==correct
        mtp['verify_steps']+=verify;mtp['proposed_drafts']+=proposed;mtp['correct_drafts']+=correct;mtp['histogram']=[a+b for a,b in zip(mtp['histogram'],hist)]
    assert output==19200 and p_output==32 and mtp['correct_drafts']>0
    assert report['raw_cache_observation_passes']==sum(row['cache_observation']['pass'] for row in report['rows'])
    sessions={};PD=[];read_workers=[];workers=[];errors={};preserved_error_counts={};new_error_counts={};runtime={};descriptor_bytes=0;graph_evidence={};retention_evidence={};installation={};profile_completion={}
    for node in [34,35,36,41]:
        path,o,events=native.unpack(CAP,node);before_path,old,old_events=native.unpack(BEFORE,node)
        sources[path]=sha(path);sources[before_path]=sha(before_path);runtime[str(node)]=o['container_id']
        wire=read(path);raw=wire['stdout'].encode();assert len(raw)==wire['native_output_transport']['raw_bytes'] and hashlib.sha256(raw).hexdigest()==wire['native_output_transport']['raw_sha256']
        assert o['container_id']==settings['runtime_containers'][str(node)]['container_id']==old['container_id']
        assert o['actual_model_identity']['namespace_pid']==o['supervisor']['model_pid'] and o['provider_phase']=='READY' and o['/health']['status']==200
        gauges=[float(x.rsplit(' ',1)[1]) for x in o['/metrics']['body'].splitlines() if x.startswith(('sglang:num_running_reqs{','sglang:num_queue_reqs{'))];assert gauges and all(x==0 for x in gauges)
        assert o['supervisor']['phase']=='MODEL_RUNNING' and 'model_exit_code' not in o['supervisor']
        env=o['actual_model_identity']['selected_environment'];assert env['DEEP_USE_MODE']==('ops' if node in [36,41] else 'alltoall') and env['HCCL_BUFFSIZE']==('256' if node in [36,41] else '')
        assert not env['GLM53_PACKED_DISPATCH'] and env['ASCEND_MF_TRANSFER_PROTOCOL']=='sdma' and env['SGLANG_MOONCAKE_SEND_AUX_TCP']=='0'
        argv=o['actual_model_identity']['actual_model_argv']
        def option(name):assert name in argv;return argv[argv.index(name)+1]
        assert option('--model-path')=='/data/wio/models/GLM-5.3-w8a8' and option('--quantization')=='modelslim'
        assert option('--dtype')==option('--kv-cache-dtype')=='bfloat16' and option('--tp-size')==option('--ep-size')=='32'
        assert option('--dp-size')==('16' if node in [36,41] else '4') and option('--speculative-num-steps')=='5' and option('--speculative-num-draft-tokens')=='6'
        if node in [36,41]:
            assert option('--deepep-mode')=='low_latency' and option('--cuda-graph-backend-decode')=='full' and option('--cuda-graph-max-bs-decode')=='4' and '--disable-decode-cuda-graph' not in argv
        else:assert option('--deepep-mode')=='normal' and '--disable-decode-cuda-graph' in argv
        assert all(log['full_archive_complete'] for log in o['logs'].values())
        assert all(log['native_sdma']['physical_link_bytes'] is None for log in o['logs'].values())
        assert all(log['native_sdma']['copy_calls']==0 or log['native_sdma']['copy_status_values']==[0] for log in o['logs'].values())
        initialized={x['pid']:x for x in events if x['event']=='initialize_complete'}
        assert len(initialized)==16 and all(x['rc']==0 and x['protocol']=='SDMA' and x['protocol_value']==1 for x in initialized.values())
        provider_pid=o['supervisor']['provider_pid'];assert provider_pid==7
        assert len(o['owned_npu_process_rows'])==18 and {x['namespace_pid'] for x in o['owned_npu_process_rows']}==({provider_pid,o['supervisor']['model_pid']}|set(initialized))
        sessions.update({x['session']:(node,x['device_id']) for x in initialized.values()})
        layouts=[x for x in events if x['event']=='model_state_registered'];assert len(layouts)==16
        for item in layouts:
            metadata=item['metadata'];assert metadata['hidden_kv_layers']==78 and metadata['draft_kv_layers']==1 and metadata['kv_buf_groups']==3
            assert len(metadata['kv_item_lens'])==237 and len(metadata['aux_item_lens'])==11 and metadata['page_size']==128 and metadata['kv_cache_dtype_str']=='bfloat16'
        current_errors=[x for log in o['logs'].values() for x in log['error_lines']];old_errors=[x for log in old['logs'].values() for x in log['error_lines']]
        assert current_errors[:len(old_errors)]==old_errors,'Existing native errors changed'
        added=current_errors[len(old_errors):]
        if duplicate_stop_pending_native and node==36:
            assert len(added)==3 and re.fullmatch(r'\[[^]]+\] ERROR:    Exception in ASGI application',added[0]) and added[1]=='Traceback (most recent call last):' and added[2]=='RuntimeError: Profiling is not in progress. Call /start_profile first.','Only exact source-qualified duplicate-stop ASGI error is allowed'
        else:assert not added,'New non-control native errors require review'
        errors[str(node)]=added;preserved_error_counts[str(node)]=len(old_errors);new_error_counts[str(node)]=len(added)
        for name,item in old['persistent_audit'].items():
            assert name in o['persistent_audit'],f'node{node} persistent audit file missing: {name}'
            assert_prefix(o['persistent_audit'][name]['rows'],item['rows'],f'node{node} {name}')
        previous={json.dumps(x,sort_keys=True) for x in old_events};new=[x for x in events if json.dumps(x,sort_keys=True) not in previous]
        assert not any(x['event'] in ['write_error','model_state_send_error','forbidden_tcp_model_payload'] for x in new)
        assert all(x['rc']==0 for x in new if x['event']=='model_state_send_complete')
        writes=[x for x in new if x['event']=='write_complete']
        for event in writes:
            assert event['rc']==0 and event['pid'] in initialized and event['payload_bytes_completed']==event['payload_bytes_requested']>0 and event['context']['kind'] in ['send_kvcache','send_aux']
            assert node in [34,35],'PD state writes must originate on P nodes'
            PD.append(event|{'node':node})
        if node in [34,35]:
            delta=sum(log['native_sdma']['copy_descriptor_bytes'] for log in o['logs'].values())-sum(log['native_sdma']['copy_descriptor_bytes'] for log in old['logs'].values())
            assert delta==sum(x['payload_bytes_completed'] for x in writes);descriptor_bytes+=delta
            before_counts=native.sdk_counts(old,initialized);after_counts=native.sdk_counts(o,initialized)
            for pid,tags in after_counts.items():
                if 'TP_MMC_PY_BATCH_GET' not in tags:continue
                deltas={name:{key:entry[key]-before_counts.get(pid,{}).get(name,{}).get(key,0) for key in ['begin','good','bad','raw_TOTAL_column_sum']} for name,entry in tags.items()}
                assert all(value['bad']==0 and value['begin']==value['good'] for value in deltas.values())
                if deltas['TP_MMC_PY_BATCH_GET']['good']>0:
                    read_workers.append({'node':node,'pid':pid,'device_id':initialized[pid]['device_id'],'counter_deltas':{name:{key:str(value) if isinstance(value,Decimal) else value for key,value in entry.items()} for name,entry in deltas.items()}})
            maps=o['native_optional_extension_worker_maps'];assert len(maps)==16
            for item in maps:
                assert item['namespace_pid'] in initialized and item['host_pid']>0 and item['loaded_extension_paths']==[OPTIONAL_EXTENSION] and re.fullmatch('[0-9a-f]{64}',item['process_maps_sha256'])
        else:
            native.sdk_counts(o,initialized)
            flat,completion=audit_profile(o['native_profile'],initialized,node);workers.extend(flat);profile_completion[str(node)]=completion
            graph=audit_graph(o,old,initialized,node);graph_evidence.update(graph['productive_graph']);retention_evidence.update(graph['retention']);installation[str(node)]=graph['installation_only_pids']
    assert len(PD)>0 and {x['context']['kind'] for x in PD}=={'send_kvcache','send_aux'}
    assert descriptor_bytes==sum(x['payload_bytes_completed'] for x in PD)
    assert len(read_workers)==32 and {(x['node'],x['device_id']) for x in read_workers}=={(node,d) for node in [34,35] for d in range(16)}
    actual_D=set()
    for event in PD:
        assert event['remote_session'] in sessions;node,device=sessions[event['remote_session']];assert node in [36,41];actual_D.add((node,device))
    assert actual_D=={(node,d) for node in [36,41] for d in range(16)}
    assert len(workers)==32 and set(graph_evidence)==set(retention_evidence)=={str(i) for i in range(32)}
    cpu_path=f'runs/{CAP}/node-43.json';cpu_wire=read(cpu_path);assert cpu_wire['rc']==0;sources[cpu_path]=sha(cpu_path);cpu=json.loads(cpu_wire['stdout'])
    identity=cpu['diagnostic_report_identity'];audit=cpu['independent_diagnostic_metric_archive_audit'];graphs=cpu['diagnostic_native_target_graph_counters']
    assert cpu['model_requests']==cpu['SDK_init_calls']==0 and identity['sha256']==RAW_REPORT_SHA and identity['bytes']==len((ROOT/RAW_REPORT).read_bytes())
    assert identity['namespace']==report['cache_namespace'] and identity['phase']=='COMPLETE' and identity['diagnostic_only'] is True and identity['performance_certificate_allowed'] is False
    assert audit['all_native_bodies_SHA_and_DP_gauges_recomputed'] is True and audit['errors']==[]
    assert audit['sample_pairs']==audit['complete_pairs']==report['D_metrics']['complete_sample_count'] and audit['archive_sha256']==report['D_metrics']['raw_archive_sha256']
    assert graphs['mode']=='decode_cuda_graph' and graphs['engine']=='decode' and graphs['all32productive_Graph_deltas'] is True
    deltas=graphs['delta_by_global_EP_rank'];assert set(deltas)=={str(i) for i in range(32)} and all(x>0 for x in deltas.values())
    assert graphs['native_endpoint_rows']==audit['native_HTTP_rows']
    result=dict(project_id='glm53',framework='sglang',run_id=RID,capture_id=CAP,before_capture_id=BEFORE,diagnostic_evidence_pass=True,diagnostic_only=True,performance_certificate_allowed=False,primary_performance_measurement=False,
                config_sha256=CONFIG_SHA,raw_report_sha256=RAW_REPORT_SHA,input_manifest_sha256=client['input_manifest_sha256'],prior_report_sha256=client['prior_report_sha256'],cache_namespace=report['cache_namespace'],
                raw_profile_control_pass=raw_control_pass,profile_HTTP_stop_acknowledged=raw_control_pass,duplicate_stop_native_completion_supplement=duplicate_stop_pending_native,
                raw_profile_start_HTTP_status=start['status'],raw_profile_stop_HTTP_status=stop.get('status'),raw_profile_stop_error=control.get('stop_error'),
                preserved_native_control_error_lines=errors,preserved_existing_error_line_counts=preserved_error_counts,new_error_line_counts=new_error_counts,profile_completion_by_node=profile_completion,
                profile_completion_basis='Actual all32rank native end-info/done markers filtered to the current diagnostic RID; raw HTTP stop 500 remains unacknowledged and is never relabeled.',
                profile_source_proof={'profiler_manager':{'path':PROFILER_SOURCE,'sha256':PROFILER_SOURCE_SHA},'model_runner_step_span':{'path':MODEL_RUNNER_SOURCE,'sha256':MODEL_RUNNER_SOURCE_SHA}},
                requested_profile_forward_steps=2,all32_D_native_profile_workers=True,
                diagnostic_requests=32,count_and_terminal_successes=report['count_and_terminal_successes'],pending_terminal_ids=len(report['pending_terminal_ids']),backend_HTTP_attempts=64,successful_backend_inferences=64,successful_D_output_tokens=19200,successful_P_output_tokens=32,all_successful_backend_output_tokens=19232,
                cache_verified_prefixes=32,cache_reviews=cache_rows,raw_cache_observation_passes=report['raw_cache_observation_passes'],D_storage_null_requests=len(D_storage_null_prefix_ids),D_storage_null_prefix_ids=D_storage_null_prefix_ids,raw_nulls_preserved=True,
                usage_contract='prompt_tokens/completion_tokens/total_tokens checked; other usage fields such as reasoning_tokens are preserved without exact-dict restriction.',
                native_PD_descriptor_bytes=descriptor_bytes,PD_descriptor_bytes_by_state=dict((kind,sum(x['payload_bytes_completed'] for x in PD if x['context']['kind']==kind)) for kind in ['send_kvcache','send_aux']),physical_wire_bytes=None,
                native_PD_request_ID_join=None,native_PD_request_ID_join_limitation='Existing SDK send audit exposes peer sessions and typed state; period transfers and all32 D destinations are verified. No per-request native transfer mapping is invented.',
                actual_P_MemCache_read_workers=len(read_workers),read_workers=read_workers,actual_D_SDMA_receiving_devices=len(actual_D),runtime_CIDs=runtime,
                diagnostic_MTP=mtp,profile_collection_duration_seconds=distribution([x['native_collection_duration_seconds'] for x in workers]),
                profile_view_separation={'interval_union':'Per-worker native CSV view only; never summed across ranks.','native_API_class_aggregates':'Per-worker API view only; never summed across ranks.','host_operator_spans':'Per-worker host span view only; never summed across ranks.','communication_name_union':'Name-based interval union; not physical bandwidth, link utilization or pure computation.','no_cross_rank_addition':True},
                workers=workers,retained_current_SDK_graph_evidence=retention_evidence,new_productive_graph_decisions_by_global_EP_rank=graph_evidence,
                diagnostic_native_target_graph_counters=graphs,independent_complete_D_metrics_archive=audit,diagnostic_report_identity=identity,non_SDK_installation_only_pids=installation,
                generic_execute_hook_replay_rows=0,generic_execute_hook_limitation='NPU override bypasses generic execute; native counters and source execute replay path provide actual Graph evidence',
                interpretation_limits=['Two requested forward steps are a bounded diagnostic sample; no claim of long-run temporal representativeness.','Profile timing includes instrumentation and export overhead; not substituted for unprofiled C32 performance.','Native interval unions, API sums and host spans are separate overlapping views; do not add them across ranks into a total latency.','A communication-name interval union is not physical wire bandwidth or NPU utilization.','Raw profile HTTP stop 500 is preserved as unacknowledged; native end-info/done markers supplement completion only.','No performance certificate or new throughput result follows from this diagnostic evidence.'],
                source_evidence_sha256=sources)
    path=ROOT/RESULT;assert not path.exists();path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({key:result[key] for key in ['diagnostic_evidence_pass','diagnostic_requests','successful_backend_inferences','D_storage_null_prefix_ids','profile_collection_duration_seconds']}))
if __name__=='__main__':main()
