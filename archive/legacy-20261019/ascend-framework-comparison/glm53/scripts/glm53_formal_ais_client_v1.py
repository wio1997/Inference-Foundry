"""Frozen-input AISBench closed loop; external cache gate precedes formal clock."""
import argparse, asyncio, hashlib, importlib.metadata as md, json, struct, time
from pathlib import Path
import aiohttp, numpy as np, torch
from aiohttp import web
from ais_bench.benchmark.models.output import RequestOutput
from aisbench_glm53_token_ids import GLM53TokenIDAPI
from glm53_request_terminal_v1 import RequestTerminals
from glm53_closed_loop_v1 import closed_loop
from glm53_aisbench_native_steady_v1 import native_steady
from glm53_runtime_metrics_v1 import MetricRecorder

MANIFESTS={16:'dff4e3943c14f8d12c1964f0aad9ed419441e6b94274f40507d2fdd34f4e3ab9',32:'9bdba915c372eb2c23da018a18c2c9653ded9c5012d875db8b496effb5983b67',64:'c4a6222088e7a8d7b9fafd1666e58601f3a0cf5055350871b1cb6a60aa7145d3'}

def default(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)

def dump(path,value):path.write_text(json.dumps(value,default=default,ensure_ascii=False,indent=2)+'\n')
def ids_sha(ids):return hashlib.sha256(struct.pack('<'+'I'*len(ids),*ids)).hexdigest()

class Observed(GLM53TokenIDAPI):
    async def parse_stream_response(self,data,output):
        if data.get('sglext') is not None:output.extra_details_data['glm53_native_sglext']=data['sglext']
        await super().parse_stream_response(data,output)

def cache_observation(framework,gateway,native):
    # Independent native page contracts: rc1 raw block128; B103 MTP bigram128.
    expected=(74400//128*128 if framework=='vllm' else (74400-1)//128*128)
    result={'expected_usable_prefix_tokens':expected,'observed':None,'pass':False}
    if framework=='vllm':
        usage=gateway.get('prefill_usage') or {};details=usage.get('prompt_tokens_details') or {};cached=details.get('cached_tokens')
        result['observed']={'P_usage':usage}
        result['pass']=type(cached) is int and expected<=cached<=74400
    else:
        tiers={}
        for role,extension in [('P',gateway.get('prefill_sglext')),('D',native.get('extra_details_data',{}).get('glm53_native_sglext'))]:
            details=extension.get('cached_tokens_details') if isinstance(extension,dict) else None
            valid=isinstance(details,dict) and all(type(details.get(k)) is int and details[k]>=0 for k in ['device','host','storage'])
            good=bool(valid and details['host']==0 and expected<=sum(details[k] for k in ['device','host','storage'])<=74400)
            tiers[role]={'native_details':details,'pass':good}
        result['observed']=tiers;result['pass']=all(v['pass'] for v in tiers.values())
    return result

def summarize(run,concurrency):
    rows=run['rows'];seconds=(run['last_terminal_perf_counter_ns']-run['first_dispatch_perf_counter_ns'])/1e9
    assert seconds>0 and len(rows)==concurrency*4
    native=[row.get('result',{}).get('native',{}) for row in rows]
    good=[row for row in rows if row.get('result',{}).get('logical_success')]
    successful=[row for row in rows if row.get('result',{}).get('transport_success')]
    output=sum(row['result']['native']['output_tokens'] for row in successful)
    cache_good=[row for row in good if row['result']['cache_observation']['pass']]
    # Input is a logical request count, never summed across P and D.
    input_count=sum(row['result']['native']['input_tokens'] for row in successful)
    distributions={}
    for key in ['ttft_seconds','tpot_seconds','e2e_seconds']:
        values=([(row['outer_terminal_perf_counter_ns']-row['outer_dispatch_perf_counter_ns'])/1e9 for row in rows] if key=='e2e_seconds' else [row['result'][key] for row in rows if row.get('result',{}).get(key) is not None])
        distributions[key]={'measured_requests':len(values),'missing_requests':len(rows)-len(values),'P50':float(np.percentile(values,50)) if values else None,'P90':float(np.percentile(values,90)) if values else None,'P99':float(np.percentile(values,99)) if values else None}
    ttft=distributions['ttft_seconds'];tpot=distributions['tpot_seconds']
    complete=len(good)==len(rows) and all(v['measured_requests']==len(rows) for v in distributions.values())
    sla={'complete_correct_requests':complete,'pass':bool(complete and ttft['P50']<4 and ttft['P90']<12 and ttft['P99']<30 and tpot['P50']<.018 and tpot['P90']<.040)}
    # The secondary native window may not select successful survivors from failure.
    steady=native_steady(native,concurrency) if complete else {'status':'N/A','reason':'At least one logical/count/cache/lifecycle failure; primary full clock retains every request.'}
    return {'full_clock_seconds':seconds,'formal_requests':len(rows),'transport_successes':len(successful),'strict_count_lifecycle_successes':len(good),'strict_count_lifecycle_failures':len(rows)-len(good),'cache_verified_strict_successes':len(cache_good),'successful_output_tokens':output,'successful_output_tokens_per_second':output/seconds,'cache_and_count_qualified_output_tokens_per_second':sum(row['result']['native']['output_tokens'] for row in cache_good)/seconds,'successful_logical_input_tokens':input_count,'successful_logical_input_tokens_per_second':input_count/seconds,'offered_logical_input_tokens':len(rows)*80000,'input_metric_includes_cached_tokens':True,'quantiles':distributions,'aggregate_SLA':sla,'goodput_tokens_per_second':None,'goodput_reason':'Only aggregate quantile SLA is specified; no invented per-request SLA allocation. Cache/count-qualified throughput is reported separately.','native_AISBench_steady':steady,'request_count_and_clock_complete':complete,'all_formal_prefix_hits_verified':len(cache_good)==len(rows)}

def load_inputs(dataset,concurrency):
    raw=(dataset/'manifest.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==MANIFESTS[concurrency]
    manifest=json.loads(raw);assert manifest['concurrency']==concurrency and manifest['formal_requests']==4*concurrency
    prefixes=[];groups=[[] for _ in range(concurrency)]
    for i,meta in enumerate(manifest['prefixes']):
        p=json.loads((dataset/f'prefix-{i:02d}.json').read_text());assert p['prefix_id']==i and len(p['input_ids'])==74400 and ids_sha(p['input_ids'])==meta['ids_sha256_le_u32']==p['ids_sha256_le_u32'];prefixes.append(p)
    for meta in manifest['formal']:
        p=json.loads((dataset/meta['file']).read_text());ids=p['input_ids'];prefix=p['prefix_id'];assert len(ids)==80000 and ids[:74400]==prefixes[prefix]['input_ids'] and ids_sha(ids)==p['ids_sha256_le_u32']==meta['ids_sha256_le_u32'];assert all(p[k]==meta[k] for k in ['request_id','prefix_id','variant']);groups[prefix].append(p)
    for group in groups:assert [x['variant'] for x in group]==[1,2,3,4]
    return manifest,prefixes,groups

async def wait_cache_gate(output,request,timeout=300):
    pending=output/'need-cache-gate.tmp';dump(pending,request);pending.replace(output/'need-cache-gate.json');deadline=time.monotonic()+timeout
    while not (output/'cache-gate.json').is_file():
        if time.monotonic()>deadline:raise TimeoutError('External exact-key/storage-readback gate timed out before formal dispatch')
        await asyncio.sleep(.2)
    gate=json.loads((output/'cache-gate.json').read_text())
    assert gate['project_id']=='glm53' and gate['namespace']==request['namespace'] and gate['framework']==request['framework']
    assert gate['prefix_count']==request['concurrency'] and gate['all_expected_keys_present'] and gate['actual_payload_samples_read'] and gate['close_rc']==0 and gate['pass'],gate
    return gate

async def run(settings):
    assert torch.__version__=='2.10.0+cpu' and md.version('ais_bench_benchmark')=='3.1.0'
    fw=settings['framework'];concurrency=settings['concurrency'];assert fw in ['vllm','sglang'] and concurrency in MANIFESTS
    dataset=Path(settings['dataset']);output=Path(settings['output']);assert output.is_dir() and not (output/'formal-client.once').exists()
    namespace=settings['namespace'];assert namespace.startswith('glm53-formal-') and namespace.endswith(f'-C{concurrency}') and settings['run_id'] in namespace
    with (output/'formal-client.once').open('x') as f:f.write(namespace)
    manifest,prefixes,groups=load_inputs(dataset,concurrency)
    for group in groups:
        for item in group:item['accepted_request_id']=item['request_id'];item['request_id']=namespace+f'-p{item["prefix_id"]:02d}-v{item["variant"]}'
    if fw=='vllm':
        from glm53_vllm_pd_gateway_v3 import VLLMPullGateway as Gateway
        endpoints=[{'role':'D','node':n,'url':f'http://172.16.12.{n}:30400/metrics'} for n in [43,45]]
    else:
        from glm53_sglang_pd_gateway_v3 import SGLangPDGateway as Gateway
        endpoints=[{'role':'D','node':n,'url':f'http://172.16.12.{n}:30600/metrics'} for n in [36,41]]
    gwrows=[];report={'project_id':'glm53','framework':fw,'concurrency':concurrency,'namespace':namespace,'run_id':settings['run_id'],'input_manifest_sha256':MANIFESTS[concurrency],'phase':'WARMUP','warmup':[],'gateway':gwrows,'formal':None,'baseline_accepted':False};runner=None;recorder=None;metrics_task=None;stop=asyncio.Event()
    try:
        async with aiohttp.ClientSession(trust_env=False,timeout=aiohttp.ClientTimeout(total=900),connector=aiohttp.TCPConnector(limit=0)) as upstream:
            terminals=RequestTerminals();gateway=Gateway(upstream,[namespace],gwrows,terminal_registry=terminals);runner=web.AppRunner(gateway.application());await runner.setup();site=web.TCPSite(runner,'127.0.0.1',0);await site.start();port=site._server.sockets[0].getsockname()[1]
            async with aiohttp.ClientSession(trust_env=False,timeout=aiohttp.ClientTimeout(total=900),connector=aiohttp.TCPConnector(limit=0,force_close=True)) as client:
                warm_model=Observed(path='',model='glm53-w8a8',stream=True,host_ip='127.0.0.1',host_port=port,run_namespace=namespace,expected_input_tokens=74400,expected_output_tokens=1)
                formal_model=Observed(path='',model='glm53-w8a8',stream=True,host_ip='127.0.0.1',host_port=port,run_namespace=namespace,expected_input_tokens=80000,expected_output_tokens=600)
                for prefix in prefixes:
                    rid=namespace+f'-p{prefix["prefix_id"]:02d}-warm';out=RequestOutput(perf_mode=True);out.uuid=rid;terminals.reserve(rid);await warm_model.generate(prefix['input_ids'],1,out,session=client);terminal=await terminals.wait_terminal(rid,15);out.input=None;native=out.to_dict();usage=native.get('extra_details_data',{}).get('glm53_service_usage') or {}
                    passed=bool(native['success'] and native['input_tokens']==74400 and native['output_tokens']==1 and usage.get('prompt_tokens')==74400 and usage.get('completion_tokens')==1 and terminal['gateway']['status']=='D_STREAM_COMPLETE');report['warmup'].append({'request_id':rid,'pass':passed,'native':native,'terminal':terminal});dump(output/'formal-progress.json',report);assert passed,'Real prefix warmup failed; no formal dispatch'
                report['cache_gate']=await wait_cache_gate(output,{'project_id':'glm53','framework':fw,'namespace':namespace,'concurrency':concurrency,'manifest_sha256':MANIFESTS[concurrency],'readback_policy':'All exact complete-page keys and first/last-page payload samples for every independent prefix; preserve natural L1/staging.'})
                recorder=MetricRecorder(fw,endpoints,output/'formal-D-metrics.jsonl.gz');before=await recorder.sample(upstream);assert before['complete_all_D_DP'] and before['sampled_total_running']==0 and before['sampled_total_waiting']==0
                metrics_task=asyncio.create_task(recorder.run(upstream,stop));report['phase']='FORMAL'
                dispatch_times={}
                def on_event(event):
                    if event['event']=='dispatch':dispatch_times[event['request_id']]=event['perf_counter_ns']/1e9
                async def submit(item):
                    out=RequestOutput(perf_mode=True);out.uuid=item['request_id'];terminals.reserve(out.uuid);begin=dispatch_times[out.uuid];error=None;terminal=None
                    try:
                        await formal_model.generate(item['input_ids'],600,out,session=client)
                    except Exception as e:error=repr(e)
                    try:terminal=await terminals.wait_terminal(out.uuid,15)
                    except Exception as e:error=(error or '')+'; terminal: '+repr(e)
                    end=time.perf_counter();out.input=None;native=out.to_dict();extra=native.get('extra_details_data',{});usage=extra.get('glm53_service_usage') or {};valid=native.get('extra_perf_data',{}).get('glm53_valid_output_chunk_times',[]);gw=terminal['gateway'] if terminal else {};count=bool(native['success'] and native['input_tokens']==80000 and native['output_tokens']==600 and usage.get('prompt_tokens')==80000 and usage.get('completion_tokens')==600 and extra.get('glm53_finish_reason')=='length' and valid)
                    cache=cache_observation(fw,gw,native);transport=bool(native['success'] and type(native.get('output_tokens')) is int and native['output_tokens']>0 and type(native.get('input_tokens')) is int and gw.get('status')=='D_STREAM_COMPLETE' and error is None);good=bool(count and transport)
                    return {'accepted_request_id':item['accepted_request_id'],'input_ids_sha256_le_u32':item['ids_sha256_le_u32'],'native':native,'terminal':terminal,'error':error,'count_pass':count,'cache_observation':cache,'logical_success':good,'transport_success':transport,'ttft_seconds':valid[0]-begin if valid else None,'tpot_seconds':(valid[-1]-valid[0])/599 if count else None,'e2e_seconds':end-begin}
                report['formal']=await closed_loop(groups,submit,on_event=on_event);stop.set();await metrics_task;metrics_task=None
                await gateway.wait_idle(15);report['pending_terminal_ids']=terminals.pending;report['metrics_after']=await recorder.sample(upstream);report['D_metrics']=recorder.close();recorder=None
                report['summary']=summarize(report['formal'],concurrency);report['baseline_accepted']=bool(report['summary']['request_count_and_clock_complete'] and report['summary']['all_formal_prefix_hits_verified'] and not terminals.pending and report['D_metrics']['complete_sample_count']==report['D_metrics']['sample_count']);report['phase']='COMPLETE'
    except BaseException as error:
        report['phase']='FAILED';report['error']=repr(error);raise
    finally:
        stop.set()
        if metrics_task:await metrics_task
        if recorder:report['D_metrics']=recorder.close()
        if runner:await runner.cleanup()
        report['backend_HTTP_attempts']=sum(int('prefill_dispatch_monotonic_ns' in g)+int('decode_dispatch_monotonic_ns' in g) for g in gwrows)
        dump(output/'formal-result.json',report)
        print('GLM53_FORMAL_RESULT_FILE='+json.dumps({'path':str(output/'formal-result.json'),'phase':report['phase'],'baseline_accepted':report['baseline_accepted'],'backend_HTTP_attempts':report['backend_HTTP_attempts']}),flush=True)
    return report

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--settings',required=True);parser.add_argument('--sha256',required=True);args=parser.parse_args();raw=Path(args.settings).read_bytes();assert hashlib.sha256(raw).hexdigest()==args.sha256;asyncio.run(run(json.loads(raw)))
if __name__=='__main__':main()
