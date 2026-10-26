"""Diagnostic native completion audit; preserve native duplicate-stop HTTP500 as non-acknowledged."""
import hashlib,importlib.util,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RID='20261018T-L11-sglang-C32-P-two-step-profile-v1'
CAP='20261018T-L11-sglang-C32-P-two-step-native-v1'
BEFORE='20261018T-L10-sglang-target-graph-post-native-v1'
def read(relative):return json.loads((ROOT/relative).read_text())
def sha(relative):return hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()
def load(relative,digest,name):
    assert sha(relative)==digest
    spec=importlib.util.spec_from_file_location(name,ROOT/relative);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def distribution(values):
    assert values
    return dict(min=min(values),median=statistics.median(values),max=max(values),ranks=len(values))
def main():
    native=load('scripts/audit_sglang_C32_L10_target_graph_native_local_v3.py','e183e0a3ae7cf8344b556235b9cb1de19d591ce51150e9423220a48e9da1dda7','native')
    cache=load('scripts/glm53_sglang_cache_annotation_review_local_v1.py',sha('scripts/glm53_sglang_cache_annotation_review_local_v1.py'),'cache')
    settings=read('config/C32-target-graph-runtime-v3/P-two-step-diagnostic-v2.json')
    report=read(f'runs/{RID}/diagnostic-result.remote.json');assert report['phase']=='COMPLETE' and report['diagnostic_only'] and report['performance_certificate_allowed'] is False and report['concurrency']==32
    assert report['backend_HTTP_attempts']==64 and len(report['gateway'])==len(report['rows'])==32 and report['count_and_terminal_successes']==32 and not report['pending_terminal_ids']
    assert report['automatic_model_request_retries'] is False and report['SDK_init_calls']==0
    control=report['profile_control'];assert control['profile_start_attempts']==control['profile_stop_attempts']==1 and control['start_response']['status']==200 and not control.get('error')
    stop=control.get('stop_response',{});raw_control_pass=control['pass']
    duplicate_stop_pending_native=(not raw_control_pass and stop.get('status')==500 and stop.get('body')=='Internal Server Error' and control.get('stop_error')=="AssertionError('Native stop did not acknowledge completion or already-inactive state')")
    assert raw_control_pass or duplicate_stop_pending_native,'Other native control failures remain unqualified'
    admission=[e for e in control['events'] if e['kind']=='P4_running_or_queued_admission'];assert len(admission)==1 and set(admission[0]['value'])=={'0','1','2','3'} and all(v>0 for v in admission[0]['value'].values())
    native_control_errors={}
    assert report['D_metrics']['sample_count']==report['D_metrics']['complete_sample_count']>0
    inputs=read('runs/20261017T2056-sglang-C32-fifth-suffix-inputs-v1/manifest-proof.json')['manifest'];expected={x['prefix_id']:x for x in inputs['rows']}
    assert {x['prefix_id'] for x in report['rows']}==set(range(32))
    cache_rows=[];output=0;mtp=dict(verify_steps=0,proposed_drafts=0,correct_drafts=0,histogram=[0]*6)
    formatter=cache.source_formatter()
    import types
    for row in report['rows']:
        pid=row['prefix_id'];n=row['native'];gw=row['terminal']['gateway'];extra=n['extra_details_data'];usage=extra['glm53_service_usage'];req=extra['glm53_request']
        assert row['variant']==5 and row['input_sha256_le_u32']==expected[pid]['ids_sha256_le_u32']==req['input_sha256_le_u32'] and row['count_and_terminal_pass']
        assert n['success'] and n['input_tokens']==usage['prompt_tokens']==req['input_tokens']==80000 and n['output_tokens']==usage['completion_tokens']==req['output_target']==600 and extra['glm53_finish_reason']=='length' and req['native_attempt_setting']==1
        assert req['namespace']==report['cache_namespace']==settings['client_settings']['cache_namespace']
        assert gw['prefill_usage']['prompt_tokens']==80000 and gw['prefill_usage']['completion_tokens']==1
        assert gw['status']=='D_STREAM_COMPLETE' and gw['prefill_http_status']==gw['decode_http_status']==200 and gw['logical_request_id']==row['request_id']
        P=gw['prefill_sglext']['cached_tokens_details'];D=extra['glm53_native_sglext']['cached_tokens_details'];checked=cache.check_tiers(P,D)
        request=types.SimpleNamespace(cached_tokens_device=P['device'],cached_tokens_host=P['host'],cached_tokens_storage=P['storage'],cached_tokens=sum(P[k] for k in ['device','host','storage']))
        for enabled,actual in [(True,P),(False,D)]:
            owner=types.SimpleNamespace(enable_hicache_storage=lambda:enabled,_get_storage_backend_type=lambda:'NpuMemcacheStore');assert formatter(owner,request)==actual
        cache_rows.append(checked);output+=n['output_tokens']
        spec=extra['glm53_native_sglext']['spec_tokens_details'];verify=spec['spec_verify_ct'];proposed=spec['spec_num_proposed_drafts'];correct=spec['spec_num_correct_drafts'];hist=spec['spec_correct_drafts_histogram']
        assert verify>0 and proposed==5*verify and len(hist)==6 and all(type(x) is int and x>=0 for x in hist) and sum(hist)==verify and sum(i*x for i,x in enumerate(hist))==correct
        mtp['verify_steps']+=verify;mtp['proposed_drafts']+=proposed;mtp['correct_drafts']+=correct;mtp['histogram']=[a+b for a,b in zip(mtp['histogram'],hist)]
    assert output==19200
    sources={f'runs/{RID}/diagnostic-result.remote.json':sha(f'runs/{RID}/diagnostic-result.remote.json')};devices=set();sessions={};PD=[];reads=set();workers=[];descriptor_bytes=0
    for node in [34,35,36,41]:
        path,o,events=native.unpack(CAP,node);before_path,old,old_events=native.unpack(BEFORE,node);sources[path]=sha(path);sources[before_path]=sha(before_path)
        wire=read(path);raw=wire['stdout'].encode();assert len(raw)==wire['native_output_transport']['raw_bytes'] and hashlib.sha256(raw).hexdigest()==wire['native_output_transport']['raw_sha256']
        assert o['container_id']==settings['runtime_containers'][str(node)]['container_id']==old['container_id'] and o['provider_phase']=='READY' and o['/health']['status']==200
        gauges=[float(x.rsplit(' ',1)[1]) for x in o['/metrics']['body'].splitlines() if x.startswith(('sglang:num_running_reqs{','sglang:num_queue_reqs{'))];assert gauges and all(x==0 for x in gauges)
        assert o['supervisor']['phase']=='MODEL_RUNNING' and 'model_exit_code' not in o['supervisor'] and all(x['full_archive_complete'] for x in o['logs'].values())
        current_errors=[x for log in o['logs'].values() for x in log['error_lines']];old_errors=[x for log in old['logs'].values() for x in log['error_lines']];assert current_errors[:len(old_errors)]==old_errors,'Existing native errors changed'
        added=current_errors[len(old_errors):]
        if duplicate_stop_pending_native and node==34:
            import re
            assert len(added)==3 and re.fullmatch(r'\[[^]]+\] ERROR:    Exception in ASGI application',added[0]) and added[1]=='Traceback (most recent call last):' and added[2]=='RuntimeError: Profiling is not in progress. Call /start_profile first.','Only exact source-qualified duplicate-stop ASGI error is allowed'
        else:assert not added,'New non-control native errors require review'
        native_control_errors[str(node)]=added
        initialized={x['pid']:x for x in events if x['event']=='initialize_complete'};assert len(initialized)==16 and all(x['rc']==0 and x['protocol']=='SDMA' and x['protocol_value']==1 for x in initialized.values())
        assert len(o['owned_npu_process_rows'])==18 and {x['namespace_pid'] for x in o['owned_npu_process_rows']}==({7,o['supervisor']['model_pid']}|set(initialized))
        assert all(x['native_sdma']['copy_calls']==0 or x['native_sdma']['copy_status_values']==[0] for x in o['logs'].values())
        sessions.update({x['session']:(node,x['device_id']) for x in initialized.values()})
        previous={json.dumps(x,sort_keys=True) for x in old_events};new=[x for x in events if json.dumps(x,sort_keys=True) not in previous]
        assert not any(x['event'] in ['write_error','model_state_send_error','forbidden_tcp_model_payload'] for x in new)
        assert all(x['rc']==0 for x in new if x['event']=='model_state_send_complete')
        writes=[x for x in new if x['event']=='write_complete']
        for event in writes:assert event['rc']==0 and event['payload_bytes_completed']==event['payload_bytes_requested']>0 and node in [34,35];PD.append(event)
        if node in [34,35]:
            delta=sum(x['native_sdma']['copy_descriptor_bytes'] for x in o['logs'].values())-sum(x['native_sdma']['copy_descriptor_bytes'] for x in old['logs'].values());assert delta==sum(x['payload_bytes_completed'] for x in writes);descriptor_bytes+=delta
            before=native.sdk_counts(old,initialized);after=native.sdk_counts(o,initialized)
            for pid,tags in after.items():
                if tags.get('TP_MMC_PY_BATCH_GET',{}).get('good',0)>before.get(pid,{}).get('TP_MMC_PY_BATCH_GET',{}).get('good',0):reads.add((node,initialized[pid]['device_id']))
        if node in [34,35]:
            profile=o['native_profile'];assert sum('Profiling done.' in x['line'] for x in profile['model_log_markers'])==16,'All local ranks must finish native stop path'
            assert len(profile['files'])>0 and profile['total_bytes']>0 and profile['raw_traces_transferred_to_Mac'] is False
            assert profile['analysis_pass'] and profile['all16_local_model_workers_profiled'] and len(profile['workers'])==16
            assert {x['namespace_pid'] for x in profile['workers'].values()}==set(initialized)
            workers.extend(dict(node=node,**worker) for worker in profile['workers'].values())
    for event in PD:assert event['remote_session'] in sessions;devices.add(sessions[event['remote_session']])
    assert devices=={(node,d) for node in [36,41] for d in range(16)} and reads=={(node,d) for node in [34,35] for d in range(16)}
    assert {x['context']['kind'] for x in PD}=={'send_kvcache','send_aux'}
    assert len(workers)==32 and descriptor_bytes==sum(x['payload_bytes_completed'] for x in PD)
    for worker in workers:
        target=[x for x in worker['top_host_operator_self_time'] if x['name'].startswith('step[EXTEND ')]
        draft=[x for x in worker['span_like_host_operator_aggregates'] if x['name']=='draft_extend']
        assert len(target)==2 and len(draft)==1 and draft[0]['count']==2,'All32 native workers must record exactly two target and draft forward spans'
    result=dict(project_id='glm53',run_id=RID,diagnostic_evidence_pass=True,raw_profile_control_pass=raw_control_pass,profile_HTTP_stop_acknowledged=raw_control_pass,duplicate_stop_native_completion_supplement=duplicate_stop_pending_native,preserved_native_control_error_lines=native_control_errors,profile_completion_basis='Actual all32rank native end-info/done markers, post-stop32request completion, idle metrics and exactsourcea962profilerstoplogic; raw controlHTTP500 is preserved and never labeled acknowledged.',audit_pattern_source_sha256='4e4f6d77810cb82689b05507e2dac35e2385905fdaf902cfa713cde5f385eeea',performance_certificate_allowed=False,primary_performance_measurement=False,
                requested_profile_forward_steps=2,native_target_step_spans_per_worker=2,native_draft_step_spans_per_worker=2,all32_P_native_profile_workers=True,diagnostic_requests=32,backend_HTTP_attempts=64,successful_backend_inferences=64,
                successful_D_output_tokens=19200,successful_P_output_tokens=32,all_successful_backend_output_tokens=19232,
                cache_verified_prefixes=32,D_storage_null_requests=sum(not x['D_storage_field_present'] for x in cache_rows),raw_nulls_preserved=True,
                native_PD_descriptor_bytes=descriptor_bytes,physical_wire_bytes=None,actual_P_MemCache_read_workers=32,actual_D_SDMA_receiving_devices=32,
                diagnostic_MTP=mtp,profile_collection_duration_seconds=distribution([x['native_collection_duration_seconds'] for x in workers]),
                workers=workers,source_evidence_sha256=sources,
                interpretation_limits=['Two requested forward steps are a bounded diagnostic sample; no claim of long-run temporal representativeness.','Profile timing includes instrumentation and export overhead; not substituted for unprofiled C32 performance.','Native interval unions and API sums are separate overlapping views; do not add them into total latency.','No optimization or causal speedup acceptance follows from a diagnostic alone.'])
    p=ROOT/'results/C32-L11-P-two-step-evidence-v2.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['diagnostic_evidence_pass','diagnostic_requests','successful_backend_inferences','profile_collection_duration_seconds']}))
if __name__=='__main__':main()
