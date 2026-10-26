"""Exercise actual frozen AISBench against local synthetic SSE, zero model calls."""
import asyncio
import hashlib
import importlib.metadata as md
import json
from pathlib import Path
import sys
import time

import aiohttp
from aiohttp import web
import torch
from ais_bench.benchmark.models.output import RequestOutput
from aisbench_glm53_token_ids import GLM53TokenIDAPI

assert torch.__version__ == '2.10.0+cpu'
assert md.version('ais_bench_benchmark') == '3.1.0'
assert not torch.cuda.is_available()
requests = []


async def completion(request):
    body = await request.json();requests.append(body)
    if body['request_id'].endswith('-fail'):
        return web.json_response({'error':'fixture HTTP503'},status=503)
    response = web.StreamResponse(headers={'Content-Type':'text/event-stream'})
    await response.prepare(request)
    if body['request_id'].endswith('-badjson'):
        await response.write(b'data: not-json\n\n');await response.write_eof();return response
    await response.write(b'data: {"choices":[{"text":"","finish_reason":null}]}\n\n')
    await asyncio.sleep(.02)
    for token in ('A','B','C','D'):
        payload={'choices':[{'text':token,'finish_reason':None}]}
        await response.write(('data: '+json.dumps(payload)+'\n\n').encode())
        await asyncio.sleep(.01)
    payload={'choices':[{'text':'','finish_reason':'length'}],
             'usage':{'prompt_tokens':3,'completion_tokens':4,'total_tokens':7}}
    await response.write(('data: '+json.dumps(payload)+'\n\ndata: [DONE]\n\n').encode())
    await response.write_eof();return response


async def main():
    app=web.Application();app.router.add_post('/v1/completions',completion)
    runner=web.AppRunner(app);await runner.setup();site=web.TCPSite(runner,'127.0.0.1',0);await site.start()
    port=site._server.sockets[0].getsockname()[1]
    model=GLM53TokenIDAPI(path='',model='glm53-w8a8',stream=True,host_ip='127.0.0.1',host_port=port,
         run_namespace='glm53-client-fixture-v1',expected_input_tokens=3,expected_output_tokens=4)
    rows=[]
    try:
        async with aiohttp.ClientSession(trust_env=False,timeout=aiohttp.ClientTimeout(total=15)) as session:
            for label in ('ok','fail','badjson'):
                output=RequestOutput(perf_mode=True);output.uuid='glm53-fixture-'+label
                begin=time.perf_counter();await model.generate([17,23,41],4,output,session=session);end=time.perf_counter()
                row={'case':label,'outer_dispatch':begin,'outer_terminal':end,'native':output.to_dict()}
                rows.append(row)
                assert end>=begin and len([r for r in requests if r['request_id']==output.uuid])==1
                if label=='ok':
                    assert output.success and output.input_tokens==3 and output.output_tokens==4 and output.content=='ABCD'
                    valid=output.extra_perf_data['glm53_valid_output_chunk_times']
                    assert len(valid)==4 and output.time_points[1]<valid[0]
                    assert output.extra_details_data['glm53_finish_reason']=='length'
                else:assert not output.success
        assert len(requests)==3
        for request in requests:
            assert request['prompt']==[17,23,41] and request['max_tokens']==request['min_tokens']==4
            assert request['stream_options']['include_usage'] and not request['return_token_ids']
        print('GLM53_AISBENCH_FIXTURE='+json.dumps({'rows':rows,'received_bodies':requests,
              'versions':{'torch':torch.__version__,'ais_bench_benchmark':md.version('ais_bench_benchmark'),
                          'aiohttp':md.version('aiohttp')},'model_requests':0,
              'qualification':'CLIENT_FIXTURE_ONLY_NOT_INFERENCE_OR_PERFORMANCE'}),flush=True)
    finally:await runner.cleanup()


asyncio.run(main())
