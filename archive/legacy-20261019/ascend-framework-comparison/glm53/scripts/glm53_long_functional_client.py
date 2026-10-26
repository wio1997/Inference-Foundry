"""Native AISBench client on actual80K/600 cold/direct/warm PD; no performance baseline."""
import asyncio
import hashlib
import importlib.metadata as md
import json
from pathlib import Path
import struct
import time
import urllib.request

import aiohttp
from aiohttp import web
import numpy as np
import torch
from ais_bench.benchmark.models.output import RequestOutput
from aisbench_glm53_token_ids import GLM53TokenIDAPI
from glm53_vllm_pd_gateway import VLLMPullGateway,route

RUN=Path('/run/glm53')
NAMESPACE='glm53-long-functional-20261017-v1'
EXPECTED_MANIFEST='dff4e3943c14f8d12c1964f0aad9ed419441e6b94274f40507d2fdd34f4e3ab9'
assert torch.__version__=='2.10.0+cpu' and md.version('ais_bench_benchmark')=='3.1.0'
assert not torch.cuda.is_available()
dataset=Path('/dataset')
assert hashlib.sha256((dataset/'manifest.json').read_bytes()).hexdigest()==EXPECTED_MANIFEST
manifest=json.loads((dataset/'manifest.json').read_text())
record=json.loads((dataset/'request-p00-v1.json').read_text())
prefix=json.loads((dataset/'prefix-00.json').read_text())
ids=record['input_ids'];prefix_ids=prefix['input_ids']
assert len(ids)==80000 and len(prefix_ids)==74400 and ids[:74400]==prefix_ids
for data in (record,prefix):
    actual=hashlib.sha256(struct.pack('<'+'I'*len(data['input_ids']),*data['input_ids'])).hexdigest()
    assert actual==data['ids_sha256_le_u32']
for value in range(64):
    mapping=route(value)
    for role,starts in [('P',{30:0,31:2}),('D',{43:0,45:8})]:
        choice=mapping[role];assert starts[choice['node']]+choice['wire_local_dp']==choice['global_dp']


def json_default(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)


def metrics_snapshot(label):
    result={}
    for node in (30,31,43,45):
        port=30300 if node in (30,31) else 30400
        try:
            with urllib.request.urlopen(f'http://172.16.12.{node}:{port}/metrics',timeout=5) as response:
                raw=response.read()
            name=f'metrics-{label}-node{node}.txt';(RUN/name).write_bytes(raw)
            result[node]={'file':name,'sha256':hashlib.sha256(raw).hexdigest()}
        except Exception as error:result[node]={'error':repr(error)}
    return result


class ObservedAPI(GLM53TokenIDAPI):
    async def parse_stream_response(self,data,output):
        if data.get('id'):
            seen=output.extra_details_data.setdefault('glm53_server_response_ids',[])
            if data['id'] not in seen:seen.append(data['id'])
        await super().parse_stream_response(data,output)


async def main():
    results=[];gateway_rows=[];runner=None
    report={'project_id':'glm53','namespace':NAMESPACE,'qualification':'FUNCTIONAL_LONG_ONLY_NOT_PERFORMANCE_BASELINE',
            'dataset_manifest_sha256':EXPECTED_MANIFEST,'cases':results,'gateway':gateway_rows,
            'native_client_versions':{name:md.version(name) for name in ('ais_bench_benchmark','aiohttp','torch')},
            'clock_info':{name:str(time.get_clock_info(name)) for name in ('monotonic','perf_counter')}}
    report['metrics_snapshots']={'before':metrics_snapshot('before')}
    try:
        for node in (30,31,43,45):
            port=30300 if node in (30,31) else 30400
            with urllib.request.urlopen(f'http://172.16.12.{node}:{port}/health',timeout=3) as response:assert response.status==200
        timeout=aiohttp.ClientTimeout(total=900)
        async with aiohttp.ClientSession(trust_env=False,timeout=timeout,connector=aiohttp.TCPConnector(limit=0)) as upstream:
            gateway=VLLMPullGateway(upstream,[NAMESPACE+'-cold',NAMESPACE+'-warm'],gateway_rows)
            runner=web.AppRunner(gateway.application());await runner.setup()
            site=web.TCPSite(runner,'127.0.0.1',0);await site.start()
            gateway_port=site._server.sockets[0].getsockname()[1]
            async with aiohttp.ClientSession(trust_env=False,timeout=timeout,connector=aiohttp.TCPConnector(limit=0)) as client:
                specs=[('direct-D',80000,600,NAMESPACE+'-direct',30400,'172.16.12.43'),
                       ('cold-PD',80000,600,NAMESPACE+'-cold',gateway_port,'127.0.0.1'),
                       ('prefix-warmup-PD',74400,1,NAMESPACE+'-warm',gateway_port,'127.0.0.1'),
                       ('warm-PD',80000,600,NAMESPACE+'-warm',gateway_port,'127.0.0.1')]
                for label,input_tokens,output_tokens,namespace,port,host in specs:
                    model=ObservedAPI(path='',model='glm53-w8a8',stream=True,host_ip=host,host_port=port,
                         run_namespace=namespace,expected_input_tokens=input_tokens,expected_output_tokens=output_tokens)
                    if label=='direct-D':model.headers['X-data-parallel-rank']='0'
                    output=RequestOutput(perf_mode=True);output.uuid=f'{NAMESPACE}-p00-{label}'
                    begin=time.monotonic_ns()
                    await model.generate(prefix_ids if input_tokens==74400 else ids,output_tokens,output,session=client)
                    terminal=time.monotonic_ns();content=output.content
                    usage=output.extra_details_data.get('glm53_service_usage')
                    valid=output.extra_perf_data.get('glm53_valid_output_chunk_times',[])
                    passed=bool(output.success and output.input_tokens==input_tokens and output.output_tokens==output_tokens
                        and usage and usage.get('prompt_tokens')==input_tokens and usage.get('completion_tokens')==output_tokens
                        and output.extra_details_data.get('glm53_finish_reason')=='length' and valid)
                    output.input=None
                    row={'case':label,'request_id':output.uuid,'outer_dispatch_monotonic_ns':begin,
                         'outer_terminal_monotonic_ns':terminal,'required_input_tokens':input_tokens,
                         'required_output_tokens':output_tokens,'pass':passed,'native':output.to_dict(),
                         'output_text_sha256':hashlib.sha256(content.encode()).hexdigest()}
                    results.append(row)
                    report['metrics_snapshots'][label]=metrics_snapshot(label)
                    (RUN/'functional-progress.json').write_text(json.dumps(report,default=json_default,indent=2)+'\n')
                    print('GLM53_LONG_CASE='+json.dumps(row,default=json_default),flush=True)
                    assert passed,f'Long case {label} failed; preserve all raw fields, no formal benchmark'
        assert len(results)==4 and len(gateway_rows)==3
        assert all(row['status']=='D_STREAM_COMPLETE' for row in gateway_rows)
        reference=results[0]['output_text_sha256']
        report['cold_PD_equals_direct_D']=results[1]['output_text_sha256']==reference
        report['warm_PD_equals_direct_D']=results[3]['output_text_sha256']==reference
        report['long_output_pass']=report['cold_PD_equals_direct_D'] and report['warm_PD_equals_direct_D']
        assert report['long_output_pass'],'Long greedy output mismatch; retain raw outputs for investigation'
    finally:
        if runner:await runner.cleanup()
        (RUN/'functional-result.json').write_text(json.dumps(report,default=json_default,indent=2)+'\n')
        print('GLM53_LONG_RESULT='+json.dumps(report,default=json_default),flush=True)


asyncio.run(main())
