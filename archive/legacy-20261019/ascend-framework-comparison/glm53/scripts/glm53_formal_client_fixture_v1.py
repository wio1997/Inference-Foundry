"""Native AIS client/steady calculator with synthetic loopback SSE only."""
import asyncio,collections,importlib.metadata,json,time
from pathlib import Path
import aiohttp,numpy as np,torch
from aiohttp import web
from ais_bench.benchmark.models.output import RequestOutput
from aisbench_glm53_token_ids import GLM53TokenIDAPI
from glm53_closed_loop_v1 import closed_loop
from glm53_aisbench_native_steady_v1 import native_steady

assert torch.__version__=='2.10.0+cpu'
assert importlib.metadata.version('ais_bench_benchmark')=='3.1.0'
received=[];active=0;peak=0
def default(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)

async def completion(request):
    global active,peak
    body=await request.json();received.append(body);rid=body['request_id']
    assert body['prompt']==[17,23,41] and body['max_tokens']==body['min_tokens']==4
    active+=1;peak=max(peak,active)
    try:
        if 'failure-p01-v4' in rid:
            await asyncio.sleep(.04)
            return web.json_response({'error':'Synthetic HTTP503'},status=503)
        response=web.StreamResponse(headers={'Content-Type':'text/event-stream'})
        await response.prepare(request)
        if 'failure-p00-v3' in rid:
            await response.write(b'data: invalid-json\n\ndata: [DONE]\n\n')
            await response.write_eof();return response
        await response.write(b'data: {"choices":[{"text":"","finish_reason":null}]}\n\n')
        delay=.003 if '-p00-' in rid else .02
        for token in 'ABCD':
            await asyncio.sleep(delay)
            await response.write(('data: '+json.dumps({'choices':[{'text':token,'finish_reason':None}]})+'\n\n').encode())
        payload={'choices':[{'text':'','finish_reason':'length'}],
                 'usage':{'prompt_tokens':3,'completion_tokens':4,'total_tokens':7}}
        await response.write(('data: '+json.dumps(payload)+'\n\ndata: [DONE]\n\n').encode())
        try:await response.write_eof()
        except ConnectionResetError:pass
        return response
    finally:active-=1

async def scenario(label,port):
    model=GLM53TokenIDAPI(path='',model='glm53-w8a8',stream=True,
        host_ip='127.0.0.1',host_port=port,run_namespace='glm53-formal-client-fixture-v1-'+label,
        expected_input_tokens=3,expected_output_tokens=4)
    groups=[[{'prefix_id':p,'variant':v,'request_id':f'glm53-fixture-{label}-p{p:02d}-v{v}'} for v in range(1,5)] for p in range(2)]
    async with aiohttp.ClientSession(trust_env=False,timeout=aiohttp.ClientTimeout(total=10)) as session:
        async def submit(request):
            out=RequestOutput(perf_mode=True);out.uuid=request['request_id']
            await model.generate([17,23,41],4,out,session=session)
            out.input=None
            return out.to_dict()
        run=await closed_loop(groups,submit)
    assert len(run['rows'])==8 and len(run['client_inflight_events'])==16
    outputs=[r['result'] for r in run['rows']]
    successes=[o for o in outputs if o['success']]
    assert len(successes)==(8 if label=='success' else 6)
    for out in successes:
        assert out['input_tokens']==3 and out['output_tokens']==4 and out['content']=='ABCD'
        assert out['time_points'][1]<out['extra_perf_data']['glm53_valid_output_chunk_times'][0]
    counts=collections.Counter(body['request_id'] for body in received if label in body['request_id'])
    assert len(counts)==8 and set(counts.values())=={1},'Every native request exactly once, including failures'
    events=run['client_inflight_events']
    early=next(e['perf_counter_ns'] for e in events if e['event']=='dispatch' and e['prefix_id']==0 and e['variant']==2)
    slow=next(e['perf_counter_ns'] for e in events if e['event']=='terminal' and e['prefix_id']==1 and e['variant']==1)
    assert early<slow and max(e['client_inflight'] for e in events)==2
    steady=native_steady(outputs,2)
    assert steady['status']==('CALCULATED' if label=='success' else 'N/A'),steady
    if label=='success':assert steady['sample_count']>0
    else:
        failed=[r for r in run['rows'] if not r['result']['success']]
        successful=[r for r in run['rows'] if r['result']['success']]
        assert len(failed)==2 and all(not r['result']['time_points'] for r in failed)
        assert run['last_terminal_perf_counter_ns']==max(r['outer_terminal_perf_counter_ns'] for r in failed)
        assert run['last_terminal_perf_counter_ns']>max(r['outer_terminal_perf_counter_ns'] for r in successful)
    return {'scenario':label,'run':run,'native_steady':steady,
            'synthetic_successful_outputs':sum(o['output_tokens'] for o in successes)}

async def main():
    app=web.Application();app.router.add_post('/v1/completions',completion)
    runner=web.AppRunner(app);await runner.setup();site=web.TCPSite(runner,'127.0.0.1',0);await site.start();port=site._server.sockets[0].getsockname()[1]
    try:
        cases=[await scenario(label,port) for label in ['success','failure']]
        assert len(received)==16 and active==0 and peak==2
        result={'project_id':'glm53','pass':True,'model_requests':0,'synthetic_HTTP_requests':16,
                'synthetic_successful_outputs':sum(c['synthetic_successful_outputs'] for c in cases),
                'native_AISBench_version':importlib.metadata.version('ais_bench_benchmark'),
                'closed_loop_no_round_barrier':True,'failure_outer_clock_preserved':True,
                'native_steady_failure_is_NA':True,'cases':cases}
        (Path('/run/glm53')/'fixture-result.json').write_text(json.dumps(result,default=default,indent=2)+'\n')
        print('GLM53_FORMAL_CLIENT_FIXTURE='+json.dumps(result,default=default),flush=True)
    finally:await runner.cleanup()

asyncio.run(main())
