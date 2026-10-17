"""Synthetic loopback integration of the full formal harness; zero model/SDK calls."""
import asyncio, hashlib, json, re, struct, sys
from pathlib import Path
import aiohttp
from aiohttp import web
import glm53_formal_ais_client_v2 as client
import glm53_sglang_pd_gateway_v3 as sg
import glm53_vllm_pd_gateway_v3 as vl
from glm53_runtime_metrics_v1 import MetricRecorder as NativeRecorder

ROOT=Path('/run/glm53');received=[];state={'scenario':''};C=16
def sha(ids):return hashlib.sha256(struct.pack('<'+'I'*len(ids),*ids)).hexdigest()

def dataset():
    path=ROOT/'synthetic-inputs';path.mkdir();prefixes=[];formal=[]
    for p in range(C):
        ids=[p+17]*74400;meta={'prefix_id':p,'ids_sha256_le_u32':sha(ids)};prefixes.append(meta);client.dump(path/f'prefix-{p:02d}.json',meta|{'input_ids':ids})
        for v in range(1,5):
            full=ids+[1000+v]*5600;name=f'request-p{p:02d}-v{v}.json';record={'request_id':f'fixture-p{p:02d}-v{v}','prefix_id':p,'variant':v,'ids_sha256_le_u32':sha(full)};client.dump(path/name,record|{'input_ids':full});formal.append(record|{'file':name})
    manifest={'concurrency':C,'formal_requests':C*4,'prefixes':prefixes,'formal':formal};client.dump(path/'manifest.json',manifest)
    # Fixture-only checksum override, never changes the production source file.
    client.MANIFESTS[C]=hashlib.sha256((path/'manifest.json').read_bytes()).hexdigest();return path

async def fake(request):
    fw=request.match_info['fw'];role=request.match_info['role'];body=await request.json();rid=body.get('rid',body.get('request_id'));pid=int(re.search(r'-p(\d+)-',rid)[1]);received.append({'scenario':state['scenario'],'role':role,'rid':rid})
    if role=='P':
        if state['scenario']=='failure' and '-p15-v4.P' in rid:
            await asyncio.sleep(.5);return web.json_response({'error':'Synthetic lateP503'},status=503)
        result={'id':rid,'choices':[{'text':'x','finish_reason':'length'}],'usage':{'prompt_tokens':len(body['prompt']),'completion_tokens':1,'prompt_tokens_details':{'cached_tokens':74368}}}
        if fw=='sglang':result['sglext']={'cached_tokens_details':{'device':0,'host':0,'storage':74368}}
        else:
            dp=pid%4;result['kv_transfer_params']={'remote_num_prompt_tokens':len(body['prompt']),'remote_block_ids':[[1]],'do_remote_prefill':True,'do_remote_decode':False,'remote_host':'172.16.12.'+str(30 if dp<2 else 31),'remote_port':26032+dp,'remote_request_id':rid,'remote_engine_id':'fixture'}
        return web.json_response(result)
    if '-p15-v1.D' in rid:await asyncio.sleep(.4)
    response=web.StreamResponse(headers={'Content-Type':'text/event-stream'});await response.prepare(request)
    count=body['max_tokens'];await response.write(b'data: {"choices":[{"text":"","finish_reason":null}]}\n\n')
    for text in ['x'*(count//2),'x'*(count-count//2)]:
        if text:await response.write(('data: '+json.dumps({'choices':[{'text':text,'finish_reason':None}]})+'\n\n').encode());await asyncio.sleep(.001)
    result={'choices':[{'text':'','finish_reason':'length'}],'usage':{'prompt_tokens':len(body['prompt']),'completion_tokens':count,'total_tokens':len(body['prompt'])+count},'sglext':{'cached_tokens_details':{'device':0,'host':0,'storage':74368}}}
    await response.write(('data: '+json.dumps(result)+'\n\ndata: [DONE]\n\n').encode());await response.write_eof();return response

async def metrics(request):
    fw=request.match_info['fw'];start=int(request.match_info['start']);lines=[]
    for p in range(start,start+8):
        if fw=='sglang':lines += [f'sglang:num_running_reqs{{dp_rank="{p}",engine_type="decode"}} 0',f'sglang:num_queue_reqs{{dp_rank="{p}",engine_type="decode"}} 0']
        else:lines += [f'vllm:num_requests_running{{engine="{p}"}} 0',f'vllm:num_requests_waiting{{engine="{p}"}} 0']
    return web.Response(text='\n'.join(lines)+'\n')

class VLLMUpstream:
    def __init__(self,session,port):self.session=session;self.port=port
    def post(self,url,**kwargs):return self.session.post(f'http://127.0.0.1:{self.port}/vllm/'+('P' if ':30300/' in url else 'D'),**kwargs)

async def cache_gate(output,request,timeout=300):
    return {'project_id':'glm53','namespace':request['namespace'],'framework':request['framework'],'prefix_count':C,'all_expected_keys_present':True,'actual_payload_samples_read':True,'close_rc':0,'pass':True,'fixture_only':'No SDK or storage reads; exercises external-gate handoff in client only.'}

async def main():
    path=dataset();app=web.Application(client_max_size=16*1024**2);app.router.add_post('/{fw}/{role}',fake);app.router.add_get('/metrics/{fw}/{start}',metrics);runner=web.AppRunner(app);await runner.setup();site=web.TCPSite(runner,'127.0.0.1',0);await site.start();port=site._server.sockets[0].getsockname()[1]
    original_sg=sg.SGLangPDGateway;original_vl=vl.VLLMPullGateway
    sg.SGLangPDGateway=lambda session,namespaces,rows,**kw:original_sg(session,namespaces,rows,fixture_endpoints=(f'http://127.0.0.1:{port}/sglang/P',f'http://127.0.0.1:{port}/sglang/D'),**kw)
    vl.VLLMPullGateway=lambda session,namespaces,rows,**kw:original_vl(VLLMUpstream(session,port),namespaces,rows,**kw)
    client.MetricRecorder=lambda fw,endpoints,output:NativeRecorder(fw,[{'role':'D','node':e['node'],'url':f'http://127.0.0.1:{port}/metrics/{fw}/{i*8}'} for i,e in enumerate(endpoints)],output)
    client.wait_cache_gate=cache_gate;cases=[]
    try:
        for fw,scenario in [('vllm','success'),('sglang','success'),('sglang','failure')]:
            state['scenario']=scenario;output=ROOT/(fw+'-'+scenario);output.mkdir();rid='fixture-v1-'+fw+'-'+scenario
            report=await client.run({'framework':fw,'concurrency':C,'dataset':str(path),'output':str(output),'run_id':rid,'namespace':f'glm53-formal-{rid}-C16'})
            assert report['http_connection_policy']=={'upstream_force_close':True,'frontend_force_close':True,'automatic_model_request_retries':False,'revision':'v2'}
            assert len(report['warmup'])==C and len(report['formal']['rows'])==C*4 and len(report['gateway'])==5*C
            assert report['baseline_accepted'] is (scenario=='success') and report['summary']['strict_count_lifecycle_failures']==(1 if scenario=='failure' else 0)
            events=report['formal']['client_inflight_events'];early=next(e['perf_counter_ns'] for e in events if e['event']=='dispatch' and e['prefix_id']==0 and e['variant']==2);slow=next(e['perf_counter_ns'] for e in events if e['event']=='terminal' and e['prefix_id']==15 and e['variant']==1);assert early<slow,'Full harness introduced round/refill barrier'
            if scenario=='failure':
                failed=next(r for r in report['formal']['rows'] if not r['result']['logical_success']);assert failed['result']['native']['success'] and failed['result']['terminal']['gateway']['status']=='FAILED';assert failed['outer_terminal_perf_counter_ns']==report['formal']['last_terminal_perf_counter_ns'];assert report['summary']['native_AISBench_steady']['status']=='N/A'
            cases.append({'framework':fw,'scenario':scenario,'baseline_flag_only_in_synthetic_fixture':report['baseline_accepted'],'summary':report['summary']})
    finally:await runner.cleanup()
    result={'project_id':'glm53','pass':True,'model_requests':0,'NPU_devices':False,'SDK_calls':0,'fake_backend_HTTP_requests':len(received),'formal_fake_requests':C*4*3,'cases':cases,'qualification':'FULL_CLIENT_LOOPBACK_ONLY_NOT_MODEL_BASELINE_OR_STORAGE_READBACK'};client.dump(ROOT/'fixture-result.json',result);print('GLM53_FORMAL_HARNESS_FIXTURE='+json.dumps(result,default=client.default),flush=True)
if __name__=='__main__':asyncio.run(main())
