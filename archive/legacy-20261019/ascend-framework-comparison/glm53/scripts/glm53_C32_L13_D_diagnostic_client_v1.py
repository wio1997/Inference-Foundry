"""One C32 diagnostic cohort, reusing qualified prefixes with fresh fifth tails.

No warmup/SDK/flush/retry. Only diagnostic; never a performance certificate.
Frozen original AIS adapter/gateway preserves real count and terminal handling.
"""
import argparse, asyncio, hashlib, importlib.metadata as md, json, time
from pathlib import Path
import aiohttp, torch
from aiohttp import web
from ais_bench.benchmark.models.output import RequestOutput
from glm53_formal_ais_client_v1 import Observed, ids_sha, dump, cache_observation
from glm53_request_terminal_v1 import RequestTerminals
from glm53_runtime_metrics_v1 import MetricRecorder
from glm53_sglang_pd_gateway_v3 import SGLangPDGateway
import glm53_C32_L13_D_profile_control_v1 as profiler

async def run(settings):
    assert torch.__version__=='2.10.0+cpu' and md.version('ais_bench_benchmark')=='3.1.0'
    assert settings['project_id']=='glm53' and settings['diagnostic_only'] and not settings['performance_certificate_allowed'] and settings['concurrency']==32
    output=Path(settings['output']);assert output.is_dir()
    with (output/'diagnostic-client.once').open('x') as file:file.write(settings['run_id'])
    prior_path=Path(settings['prior_report']);prior_raw=prior_path.read_bytes()
    assert hashlib.sha256(prior_raw).hexdigest()==settings['prior_report_sha256']
    prior=json.loads(prior_raw);assert prior['phase']=='COMPLETE' and prior['concurrency']==32 and prior['run_id']==settings['prior_run_id'] and prior['cache_gate']['pass']
    assert prior['summary']['strict_count_lifecycle_successes']==128 and prior['summary']['strict_count_lifecycle_failures']==0
    namespace=prior['namespace'];assert namespace==settings['cache_namespace']
    manifest_path=Path(settings['dataset'])/'manifest.json';raw=manifest_path.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==settings['input_manifest_sha256']
    manifest=json.loads(raw);assert manifest['fresh_suffix_lcp_exact'] and manifest['concurrency']==32 and manifest['requests']==32 and manifest['warmups']==0
    inputs=[]
    for index,meta in enumerate(manifest['rows']):
        item=json.loads((manifest_path.parent/meta['file']).read_text())
        assert item['prefix_id']==index and item['variant']==5 and len(item['input_ids'])==80000 and ids_sha(item['input_ids'])==meta['ids_sha256_le_u32'] and ids_sha(item['input_ids'][:74400])==meta['prefix_sha256_le_u32']
        assert meta['prefix_sha256_le_u32']==prior['cache_gate']['prefixes'][index]['prefix_ids_sha256_le_u32'] and item['lcp_against_previous_four']==[74400]*4
        inputs.append(item)
    report=dict(project_id='glm53',framework='sglang',run_id=settings['run_id'],concurrency=32,
                diagnostic_only=True,performance_certificate_allowed=False,cache_namespace=namespace,
                prior_qualification_report_sha256=settings['prior_report_sha256'],input_manifest_sha256=settings['input_manifest_sha256'],
                warmups=0,phase='STARTING',rows=[],gateway=[],automatic_model_request_retries=False,SDK_init_calls=0)
    runner=None;recorder=None;metrics_task=None;profile_task=None;stop=asyncio.Event()
    try:
        async with aiohttp.ClientSession(trust_env=False,timeout=aiohttp.ClientTimeout(total=900),connector=aiohttp.TCPConnector(limit=0,force_close=True)) as upstream:
            terminals=RequestTerminals();gateway=SGLangPDGateway(upstream,[namespace],report['gateway'],terminal_registry=terminals)
            runner=web.AppRunner(gateway.application());await runner.setup();site=web.TCPSite(runner,'127.0.0.1',0);await site.start();port=site._server.sockets[0].getsockname()[1]
            async with aiohttp.ClientSession(trust_env=False,timeout=aiohttp.ClientTimeout(total=900),connector=aiohttp.TCPConnector(limit=0,force_close=True)) as client:
                model=Observed(path='',model='glm53-w8a8',stream=True,host_ip='127.0.0.1',host_port=port,run_namespace=namespace,expected_input_tokens=80000,expected_output_tokens=600)
                endpoints=[{'role':'D','node':n,'url':f'http://172.16.12.{n}:30600/metrics'} for n in [36,41]]
                recorder=MetricRecorder('sglang',endpoints,output/'diagnostic-D-metrics.jsonl.gz');before=await recorder.sample(upstream)
                assert before['complete_all_D_DP'] and before['sampled_total_running']==before['sampled_total_waiting']==0
                metrics_task=asyncio.create_task(recorder.run(upstream,stop))
                def profile_save(value):
                    path=output/'D-profile-control-progress.json';temporary=path.with_suffix('.tmp');dump(temporary,value);temporary.replace(path)
                profile_task=asyncio.create_task(asyncio.to_thread(profiler.control,settings['profile_settings'],profiler.request,save=profile_save))
                admission=asyncio.Event();inflight=0;events=[]
                async def submit(item):
                    nonlocal inflight
                    await admission.wait()
                    rid=namespace+f'-diag-L13-D-p{item["prefix_id"]:02d}-v5'
                    out=RequestOutput(perf_mode=True);out.uuid=rid;terminals.reserve(rid)
                    begin=time.perf_counter_ns();inflight+=1;events.append(dict(event='dispatch',request_id=rid,perf_counter_ns=begin,inflight=inflight))
                    row=dict(request_id=rid,prefix_id=item['prefix_id'],variant=5,input_sha256_le_u32=item['ids_sha256_le_u32'],outer_dispatch_perf_counter_ns=begin)
                    try:
                        await model.generate(item['input_ids'],600,out,session=client)
                        row['terminal']=await terminals.wait_terminal(rid,15)
                    except Exception as error:row['error']=repr(error)
                    finally:
                        out.input=None;row['native']=out.to_dict();row['outer_terminal_perf_counter_ns']=time.perf_counter_ns()
                        gw=row.get('terminal',{}).get('gateway',{});row['cache_observation']=cache_observation('sglang',gw,row['native'])
                        usage=row['native'].get('extra_details_data',{}).get('glm53_service_usage',{})
                        row['count_and_terminal_pass']=bool(not row.get('error') and row['native']['success'] and row['native']['input_tokens']==usage.get('prompt_tokens')==80000 and row['native']['output_tokens']==usage.get('completion_tokens')==600 and row['native'].get('extra_details_data',{}).get('glm53_finish_reason')=='length' and gw.get('status')=='D_STREAM_COMPLETE')
                        inflight-=1;events.append(dict(event='terminal',request_id=rid,perf_counter_ns=row['outer_terminal_perf_counter_ns'],inflight=inflight));report['rows'].append(row)
                        dump(output/'diagnostic-progress.json',report)
                tasks=[asyncio.create_task(submit(item)) for item in inputs]
                report['phase']='DIAGNOSTIC';admission.set();await asyncio.gather(*tasks)
                await gateway.wait_idle(15);report['pending_terminal_ids']=terminals.pending
                stop.set();await metrics_task;metrics_task=None;report['D_metrics']=recorder.close();recorder=None
                report['profile_control']=await profile_task;profile_task=None
                assert inflight==0 and len(report['rows'])==32
                report['client_inflight_events']=events
                report['full_diagnostic_clock_seconds']=(max(x['outer_terminal_perf_counter_ns'] for x in report['rows'])-min(x['outer_dispatch_perf_counter_ns'] for x in report['rows']))/1e9
                report['count_and_terminal_successes']=sum(x['count_and_terminal_pass'] for x in report['rows'])
                report['raw_cache_observation_passes']=sum(x['cache_observation']['pass'] for x in report['rows'])
                report['phase']='COMPLETE'
    except BaseException as error:
        report['phase']='FAILED';report['error']=repr(error);raise
    finally:
        stop.set()
        if metrics_task:await metrics_task
        if profile_task:report['profile_control']=await profile_task
        if recorder:report['D_metrics']=recorder.close()
        if runner:await runner.cleanup()
        report['backend_HTTP_attempts']=sum(int('prefill_dispatch_monotonic_ns' in g)+int('decode_dispatch_monotonic_ns' in g) for g in report['gateway'])
        dump(output/'diagnostic-result.json',report)
        print(json.dumps({k:report.get(k) for k in ['project_id','run_id','phase','diagnostic_only','performance_certificate_allowed','backend_HTTP_attempts','count_and_terminal_successes','raw_cache_observation_passes']}),flush=True)
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--settings',required=True);parser.add_argument('--sha256',required=True);args=parser.parse_args()
    raw=Path(args.settings).read_bytes();assert hashlib.sha256(raw).hexdigest()==args.sha256
    asyncio.run(run(json.loads(raw)))
