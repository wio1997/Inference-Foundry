"""Actual AIS client + local fake PD gateways, per-request drain/refill proof."""
import asyncio,json,time
import aiohttp
from aiohttp import web
from ais_bench.benchmark.models.output import RequestOutput
from aisbench_glm53_token_ids import GLM53TokenIDAPI
from glm53_sglang_pd_gateway_v3 import SGLangPDGateway
from glm53_vllm_pd_gateway_v3 import VLLMPullGateway
from glm53_request_terminal_v1 import RequestTerminals
received=[];events={}
async def fake(request):
    body=await request.json();fw=request.match_info['fw'];role=request.match_info['role'];rid=body.get('rid',body.get('request_id'));key=rid[:-2]
    events.setdefault(key,asyncio.Event());received.append({'framework':fw,'role':role,'id':rid,'input_count':len(body['prompt'])})
    if role=='P':
        if fw=='sg':
            await events[key].wait()
            await asyncio.sleep(3.0 if '-slow.' in rid else .1)
        if '-reject.' in rid:return web.json_response({'error':'fixture503'},status=503)
        result={'id':rid,'choices':[{'text':'x','finish_reason':'length'}],'usage':{'prompt_tokens':len(body['prompt']),'completion_tokens':1}}
        if fw=='vllm':
            rank=int(rid.split('-p')[1][:2])%4
            result['kv_transfer_params']={'remote_num_prompt_tokens':len(body['prompt']),'remote_block_ids':[[1]],'do_remote_prefill':True,'do_remote_decode':False,'remote_host':'172.16.12.'+str(30 if rank<2 else 31),'remote_port':26032+rank,'remote_request_id':rid,'remote_engine_id':'fixture'}
        else:result['sglext']={'cached_tokens_details':{'device':0,'host':0,'storage':74368}}
        return web.json_response(result)
    response=web.StreamResponse(headers={'Content-Type':'text/event-stream'});await response.prepare(request)
    await response.write(b'data: {"choices":[{"text":"","finish_reason":null}]}\n\n')
    count=body['max_tokens'];await response.write(('data: '+json.dumps({'choices':[{'text':'x'*count,'finish_reason':None}]})+'\n\n').encode())
    await response.write(('data: '+json.dumps({'choices':[{'text':'','finish_reason':'length'}],'usage':{'prompt_tokens':len(body['prompt']),'completion_tokens':count,'total_tokens':len(body['prompt'])+count}})+'\n\ndata: [DONE]\n\n').encode());await response.write_eof();events[key].set();return response
class ExitDelay:
    def __init__(self,context,delay):self.context=context;self.delay=delay
    async def __aenter__(self):return await self.context.__aenter__()
    async def __aexit__(self,*args):
        result=await self.context.__aexit__(*args);await asyncio.sleep(self.delay);return result
class VLLMUpstream:
    def __init__(self,session,port):self.session=session;self.port=port
    def post(self,url,**kwargs):
        role='P' if ':30300/' in url else 'D';rid=kwargs['json']['request_id'];delay=(3.0 if '-slow.' in rid else .1) if role=='D' else 0
        return ExitDelay(self.session.post(f'http://127.0.0.1:{self.port}/vllm/{role}',**kwargs),delay)
async def serve(app):
    r=web.AppRunner(app);await r.setup();s=web.TCPSite(r,'127.0.0.1',0);await s.start();return r,s._server.sockets[0].getsockname()[1]
async def main():
    app=web.Application(client_max_size=16*1024**2);app.router.add_post('/{fw}/{role}',fake);ur,port=await serve(app);results={}
    try:
        async with aiohttp.ClientSession(trust_env=False,timeout=aiohttp.ClientTimeout(total=15)) as session:
            for fw in ['sg','vllm']:
                rows=[];terminals=RequestTerminals();ns='glm53-terminal-fixture-'+fw
                gateway=SGLangPDGateway(session,[ns],rows,fixture_endpoints=(f'http://127.0.0.1:{port}/sg/P',f'http://127.0.0.1:{port}/sg/D'),terminal_registry=terminals) if fw=='sg' else VLLMPullGateway(VLLMUpstream(session,port),[ns],rows,terminal_registry=terminals)
                gr,gport=await serve(gateway.application());completed=[]
                async def submit(label,pid,inputs=80000,outputs=600):
                    rid=f'{ns}-p{pid:02d}-{label}';terminals.reserve(rid)
                    model=GLM53TokenIDAPI(path='',model='glm53-w8a8',stream=True,host_ip='127.0.0.1',host_port=gport,run_namespace=ns,expected_input_tokens=inputs,expected_output_tokens=outputs)
                    out=RequestOutput(perf_mode=True);out.uuid=rid;await model.generate([17]*inputs,outputs,out,session=session);native_return=time.perf_counter_ns();terminal=await terminals.wait_terminal(rid,5)
                    row=terminal['gateway'];assert row['logical_request_id']==rid
                    if label=='reject':
                        assert row['status'] in ['FAILED','PREFILL_REJECTED']
                        assert out.success is (fw=='sg'), 'SG late P failure arrives after native D stream success; terminal row must reject it'
                    else:
                        assert out.success and out.content=='x'*outputs and out.output_tokens==outputs and out.input_tokens==inputs and row['status']=='D_STREAM_COMPLETE'
                        if fw=='sg':assert row['prefill_sglext']['cached_tokens_details']['storage']==74368
                    completed.append(label);return {'case':label,'native_success':out.success,'native_return_perf_counter_ns':native_return,'terminal':terminal}
                try:
                    slow=asyncio.create_task(submit('slow',0));key=f'{ns}-p00-slow'
                    while key not in events:await asyncio.sleep(.005)
                    await asyncio.wait_for(events[key].wait(),3)
                    fast=await submit('fast',1);assert not slow.done() and gateway.active==1 and 'slow' not in completed
                    refill=await submit('refill',2,74400,1);assert not slow.done() and gateway.active==1
                    slow_result=await slow;reject=await submit('reject',3);await gateway.wait_idle(5)
                    assert not terminals.pending and gateway.active==0 and len(rows)==4
                    results[fw]={'cases':[fast,refill,slow_result,reject],'fast_and_refill_completed_while_slow_draining':True}
                finally:await gr.cleanup()
    finally:await ur.cleanup()
    print('GLM53_TERMINAL_FIXTURE='+json.dumps({'pass':True,'project_id':'glm53','model_requests':0,'fake_HTTP_requests':len(received),'frameworks':results,'received':received}),flush=True)
asyncio.run(main())
