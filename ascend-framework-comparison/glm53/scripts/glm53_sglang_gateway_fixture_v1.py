"""Actual frozen AISBench CPU fixture for concurrent native SGLang bootstrap relay."""
import asyncio,json
import aiohttp
from aiohttp import web
from ais_bench.benchmark.models.output import RequestOutput
from aisbench_glm53_token_ids import GLM53TokenIDAPI
from glm53_sglang_pd_gateway_v1 import SGLangPDGateway
received=[];d_events={}
async def fake(request):
 body=await request.json();role=request.match_info['role'];room=body['bootstrap_room'];d_events.setdefault(room,asyncio.Event());received.append({'role':role,'body':{k:v for k,v in body.items() if k!='prompt'},'input_count':len(body['prompt'])})
 assert body['bootstrap_host']=='172.16.12.34' and body['bootstrap_port']==29898 and 'request_id' not in body
 if role=='P':
  assert body['max_tokens']==body['min_tokens']==1 and not body['stream']
  await asyncio.wait_for(d_events[room].wait(),2)
  if 'prefill-reject' in body['rid']:return web.json_response({'error':'fixtureP503'},status=503)
  await asyncio.sleep(.15)
  return web.json_response({'id':body['rid'],'choices':[{'text':'x','finish_reason':'length'}],'usage':{'prompt_tokens':len(body['prompt']),'completion_tokens':1}})
 d_events[room].set();assert body['disagg_prefill_dp_rank']==int(body['rid'].split('-p')[1][:2])%4
 if 'decode-reject' in body['rid']:return web.json_response({'error':'fixtureD503'},status=503)
 if 'prefill-reject' in body['rid']:
  await asyncio.sleep(.03)
 response=web.StreamResponse(headers={'Content-Type':'text/event-stream'});await response.prepare(request)
 await response.write(b'data: {"choices":[{"text":"","finish_reason":null}]}\n\n');await asyncio.sleep(.01)
 count=body['max_tokens'];await response.write(('data: '+json.dumps({'id':body['rid'],'choices':[{'text':'x'*count,'finish_reason':None}]})+'\n\n').encode())
 await response.write(('data: '+json.dumps({'choices':[{'text':'','finish_reason':'length'}],'usage':{'prompt_tokens':len(body['prompt']),'completion_tokens':count,'total_tokens':len(body['prompt'])+count}})+'\n\ndata: [DONE]\n\n').encode());await response.write_eof();return response
async def serve(app):
 r=web.AppRunner(app);await r.setup();site=web.TCPSite(r,'127.0.0.1',0);await site.start();return r,site._server.sockets[0].getsockname()[1]
async def main():
 app=web.Application(client_max_size=16*1024**2);app.router.add_post('/{role}',fake);up,port=await serve(app);gr=None;rows=[];tests=[]
 try:
  async with aiohttp.ClientSession(trust_env=False,timeout=aiohttp.ClientTimeout(total=10)) as session:
   gateway=SGLangPDGateway(session,['glm53-sglang-gateway-fixture-v1'],rows,fixture_endpoints=(f'http://127.0.0.1:{port}/P',f'http://127.0.0.1:{port}/D'));gr,gport=await serve(gateway.application())
   for label,prefix,inputs,outputs in [('full',8,80000,600),('warmup',15,74400,1),('decode-reject',14,80000,600),('prefill-reject',13,80000,600),('duplicate',8,80000,600)]:
    model=GLM53TokenIDAPI(path='',model='glm53-w8a8',stream=True,host_ip='127.0.0.1',host_port=gport,run_namespace='glm53-sglang-gateway-fixture-v1',expected_input_tokens=inputs,expected_output_tokens=outputs)
    output=RequestOutput(perf_mode=True);output.uuid=f'glm53-sglang-gateway-fixture-v1-p{prefix:02d}-'+('full' if label=='duplicate' else label)
    await model.generate([17]*inputs,outputs,output,session=session);atreturn={'active':gateway.active,'status':rows[-1]['status']};await gateway.wait_idle(5)
    assert gateway.active==0 and len(rows)==len(tests)+1
    if label in ['full','warmup']:
     assert output.success and output.input_tokens==inputs and output.output_tokens==outputs and output.content=='x'*outputs
     assert atreturn['active']==1 and rows[-1]['status']=='D_STREAM_COMPLETE'
    else:assert not output.success and rows[-1]['status'] in ('FAILED','DECODE_REJECTED')
    tests.append({'case':label,'native_success':output.success,'state_at_native_return':atreturn,'terminal_gateway_row':dict(rows[-1])})
   assert len(received)==8
   for index in range(0,8,2):
    pair=received[index:index+2];assert {x['role'] for x in pair}=={'P','D'};assert pair[0]['body']['bootstrap_room']==pair[1]['body']['bootstrap_room']
   assert [x['body']['routed_dp_rank'] for x in received if x['role']=='P']==[0,3,2,1]
   assert [x['body']['routed_dp_rank'] for x in received if x['role']=='D']==[8,15,14,13]
   print('GLM53_SGLANG_GATEWAY_FIXTURE='+json.dumps({'project_id':'glm53','model_requests':0,'fixture_only':True,'tests':tests,'received':received,'pass':True}),flush=True)
 finally:
  if gr:await gr.cleanup()
  await up.cleanup()
asyncio.run(main())
