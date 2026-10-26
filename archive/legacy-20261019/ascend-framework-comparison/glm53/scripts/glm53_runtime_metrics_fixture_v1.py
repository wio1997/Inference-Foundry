"""CPU loopback fixture: complete metrics beyond64KiB, failure/cap coverage."""
import asyncio,gzip,hashlib,importlib.metadata,json
from pathlib import Path
import aiohttp
from aiohttp import web
import glm53_runtime_metrics_v1 as metrics
assert importlib.metadata.version('ais_bench_benchmark')=='3.1.0'
fault=None
async def fake(request):
    fw=request.match_info['fw'];node=int(request.match_info['node'])
    if fault=='503' and node==1:return web.Response(status=503,text='fixture unavailable')
    text='# Native metrics padding\n'*3500
    for dp in range(node*8,node*8+8):
        if fw=='sglang':
            labels=f'dp_rank="{dp}",engine_type="decode",model_name="glm53-w8a8"';pairs=[('sglang:num_running_reqs',2),('sglang:num_queue_reqs',dp%3)]
        else:
            labels=f'engine="{dp}",model_name="glm53-w8a8"';pairs=[('vllm:num_requests_running',2),('vllm:num_requests_waiting',dp%3)]
        text+=''.join(f'{name}{{{labels}}} {value}.0\n' for name,value in pairs)
    response=web.StreamResponse(headers={'Content-Type':'text/plain'});await response.prepare(request)
    raw=text.encode()
    try:
        for start in range(0,len(raw),7000):await response.write(raw[start:start+7000]);await asyncio.sleep(.001)
        await response.write_eof()
    except ConnectionResetError:pass
    return response
async def main():
    global fault
    app=web.Application();app.router.add_get('/{fw}/{node}',fake);runner=web.AppRunner(app);await runner.setup();site=web.TCPSite(runner,'127.0.0.1',0);await site.start();port=site._server.sockets[0].getsockname()[1];out={}
    try:
        async with aiohttp.ClientSession(trust_env=False) as session:
            for fw in ['sglang','vllm']:
                endpoints=[{'node':n,'role':'D','url':f'http://127.0.0.1:{port}/{fw}/{n}'} for n in [0,1]]
                rec=metrics.MetricRecorder(fw,endpoints,Path('/run/glm53')/(fw+'-metrics.gz'))
                try:
                    first=await rec.sample(session);assert first['complete_all_D_DP'] and first['sampled_total_running']==32 and len(first['per_DP'])==16
                    assert all(n['bytes']>65536 for n in first['native'])
                    stop=asyncio.Event();task=asyncio.create_task(rec.run(session,stop,.01))
                    while len(rec.samples)<3:await asyncio.sleep(.005)
                    stop.set();await task
                    fault='503';bad=await rec.sample(session);assert not bad['complete_all_D_DP'] and bad['errors']
                    fault=None;oldcap=metrics.CAP;metrics.CAP=1024
                    try:
                        cap=await rec.sample(session);assert not cap['complete_all_D_DP'] and len(cap['errors'])==2
                    finally:metrics.CAP=oldcap
                finally:summary=rec.close();fault=None
                rows=[json.loads(line) for line in gzip.decompress(Path(summary['raw_archive']).read_bytes()).splitlines()]
                for row in rows:
                    if 'raw_utf8' in row:
                        raw=row['raw_utf8'].encode();assert hashlib.sha256(raw).hexdigest()==row['sha256'] and len(raw)==row['bytes']
                assert summary['complete_sample_count']>=3 and summary['sampled_peak_total_D_running']==32 and set(summary['sampled_peak_D_running_by_DP'].values())=={2}
                out[fw]={'sample_count':summary['sample_count'],'complete_sample_count':summary['complete_sample_count'],'sampled_peak_total_D_running':32,'full_native_response_larger_than64KiB':True,'HTTP503_and_oversize_not_complete':True,'raw_roundtrip_sha_pass':True}
    finally:await runner.cleanup()
    print('GLM53_RUNTIME_METRICS_FIXTURE='+json.dumps({'pass':True,'model_requests':0,'frameworks':out}),flush=True)
asyncio.run(main())
