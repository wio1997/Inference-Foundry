"""Actual AISBench/client + gateway lifecycle fixture, local fake upstream only."""
import asyncio,json,time
import aiohttp
from aiohttp import web
from ais_bench.benchmark.models.output import RequestOutput
from aisbench_glm53_token_ids import GLM53TokenIDAPI
from glm53_vllm_pd_gateway_v2 import VLLMPullGateway

received=[]
async def fake(request):
 body=await request.json();role=request.match_info['role'];received.append({'role':role,'body':{k:v for k,v in body.items() if k!='prompt'},'input_count':len(body['prompt']),'rank_header':request.headers.get('X-data-parallel-rank')})
 if role=='P':
  if body['request_id'].endswith('-reject.P'):return web.json_response({'error':'fixture prefill rejection'},status=503)
  rank=int(body['request_id'].split('-p')[1][:2])%4
  return web.json_response({'usage':{'prompt_tokens':len(body['prompt']),'completion_tokens':1},'kv_transfer_params':{'remote_num_prompt_tokens':len(body['prompt']),'remote_block_ids':[[1]],'do_remote_prefill':True,'do_remote_decode':False,'remote_host':'172.16.12.'+str(30 if rank<2 else 31),'remote_port':26032+rank,'remote_request_id':body['request_id'],'remote_engine_id':'fixture'}})
 response=web.StreamResponse(headers={'Content-Type':'text/event-stream'});await response.prepare(request)
 await response.write(b'data: {"choices":[{"text":"","finish_reason":null}]}\n\n')
 await asyncio.sleep(.01)
 count=body['max_tokens'];chunk='x'*min(5,count)
 for _ in range(count//len(chunk)):
  await response.write(('data: '+json.dumps({'choices':[{'text':chunk,'finish_reason':None}]})+'\n\n').encode())
 await response.write(('data: '+json.dumps({'choices':[{'text':'','finish_reason':'length'}],'usage':{'prompt_tokens':len(body['prompt']),'completion_tokens':count,'total_tokens':len(body['prompt'])+count}})+'\n\ndata: [DONE]\n\n').encode())
 await response.write_eof();return response

class SlowExit:
 def __init__(self,context):self.context=context
 async def __aenter__(self):return await self.context.__aenter__()
 async def __aexit__(self,*args):
  result=await self.context.__aexit__(*args)
  await asyncio.sleep(.15)
  return result

class Upstream:
 def __init__(self,session,port):self.session=session;self.port=port
 def post(self,url,**kwargs):
  role='P' if ':30300/' in url else 'D'
  return SlowExit(self.session.post(f'http://127.0.0.1:{self.port}/{role}',**kwargs))

async def serve(app):
 runner=web.AppRunner(app);await runner.setup();site=web.TCPSite(runner,'127.0.0.1',0);await site.start();return runner,site._server.sockets[0].getsockname()[1]

async def main():
 app=web.Application(client_max_size=16*1024**2);app.router.add_post('/{role}',fake)
 upstream_runner,upstream_port=await serve(app);gateway_runner=None;rows=[];tests=[]
 try:
  async with aiohttp.ClientSession(trust_env=False,timeout=aiohttp.ClientTimeout(total=10)) as session:
   gateway=VLLMPullGateway(Upstream(session,upstream_port),['glm53-gateway-fixture-v2'],rows)
   gateway_runner,port=await serve(gateway.application())
   for label,prefix,inputs,outputs in [('full',8,80000,600),('warmup',15,74400,1),('reject',0,80000,600)]:
    model=GLM53TokenIDAPI(path='',model='glm53-w8a8',stream=True,host_ip='127.0.0.1',host_port=port,run_namespace='glm53-gateway-fixture-v2',expected_input_tokens=inputs,expected_output_tokens=outputs)
    output=RequestOutput(perf_mode=True);output.uuid=f'glm53-gateway-fixture-v2-p{prefix:02d}-{label}'
    await model.generate([17]*inputs,outputs,output,session=session)
    state_at_native_return={'active':gateway.active,'status':rows[-1]['status'],'row_count':len(rows)}
    await gateway.wait_idle(5)
    assert gateway.active==0 and len(rows)==len(tests)+1
    if label!='reject':
     assert output.success and output.input_tokens==inputs and output.output_tokens==outputs
     assert state_at_native_return['active']==1 and state_at_native_return['status']=='DECODE'
     assert rows[-1]['status']=='D_STREAM_COMPLETE'
     assert len(output.content)==outputs and output.content=='x'*outputs
    else:assert not output.success and rows[-1]['status']=='PREFILL_REJECTED'
    tests.append({'case':label,'native_success':output.success,'state_at_native_return':state_at_native_return,'terminal_gateway_row':dict(rows[-1])})
   assert len(received)==5 and [r['role'] for r in received]==['P','D','P','D','P']
   assert [r['rank_header'] for r in received]==['0','0','1','7','0']
   assert received[0]['body']['max_tokens']==1 and received[1]['body']['max_tokens']==600
   assert received[1]['body']['kv_transfer_params']['remote_num_prompt_tokens']==80000
   print('GLM53_GATEWAY_FIXTURE='+json.dumps({'project_id':'glm53','model_requests':0,'fixture_only':True,'tests':tests,'received':received,'pass':True}),flush=True)
 finally:
  if gateway_runner:await gateway_runner.cleanup()
  await upstream_runner.cleanup()
asyncio.run(main())
