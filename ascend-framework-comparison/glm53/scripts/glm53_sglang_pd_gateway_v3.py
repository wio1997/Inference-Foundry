"""B103 native concurrent P/D bootstrap relay with explicit global DP affinity."""
import asyncio,hashlib,json,re,struct,time
from aiohttp import web

def route(prefix):
 assert type(prefix) is int and 0<=prefix<64
 return {'P':{'api_host':'172.16.12.34','api_port':30500,'global_dp':prefix%4},'D':{'api_host':'172.16.12.36','api_port':30600,'global_dp':prefix%16}}

def bootstrap_room(request_id,prefill_dp):
 # Native follow_bootstrap_room selects P DP by room modulo the P DP size.
 assert isinstance(request_id,str) and request_id and type(prefill_dp) is int and 0<=prefill_dp<4
 value=int.from_bytes(hashlib.sha256(request_id.encode()).digest()[:8],'little')&((1<<63)-1)
 return (value//4)*4+prefill_dp

class SGLangPDGateway:
 def __init__(self,session,allowed_namespaces,rows,*,fixture_endpoints=None,terminal_registry=None):
  self.terminals=terminal_registry;self.session=session;self.namespaces=frozenset(allowed_namespaces);self.rows=rows;self.active=0;self.idle=asyncio.Event();self.idle.set();self.rooms=set();self.fixture_endpoints=fixture_endpoints
 def application(self):
  app=web.Application(client_max_size=16*1024**2);app.router.add_post('/v1/completions',self.completion);return app
 async def wait_idle(self,timeout=10):await asyncio.wait_for(self.idle.wait(),timeout)
 async def prefill(self,body,url,row):
  row['prefill_dispatch_monotonic_ns']=time.monotonic_ns()
  async with self.session.post(url,json=body,headers={'X-Request-Id':body['rid']}) as response:
   row['prefill_http_status']=response.status;raw=await response.text()
   if response.status!=200:
    row['prefill_error_body']=raw;raise RuntimeError(f'Native SGLang prefill HTTP {response.status}')
   data=json.loads(raw);row['prefill_usage']=data.get('usage');row['prefill_response_id']=data.get('id');row['prefill_sglext']=data.get('sglext')
   assert data['usage']['prompt_tokens']==len(body['prompt']) and data['usage']['completion_tokens']==1
   assert len(data['choices'])==1 and data['choices'][0]['finish_reason']=='length'
  row['prefill_terminal_monotonic_ns']=time.monotonic_ns()
 async def completion(self,request):
  row={'gateway_dispatch_monotonic_ns':time.monotonic_ns(),'status':'VALIDATING'};self.rows.append(row);self.active+=1;self.idle.clear();stream=None;p_task=None;terminal_ticket=None
  try:
   body=await request.json();ids=body.get('prompt');rid=body.get('request_id','')
   if self.terminals is not None:terminal_ticket=self.terminals.bind(rid,row)
   assert isinstance(ids,list) and all(type(x) is int and x>=0 for x in ids)
   assert (len(ids),body.get('max_tokens')) in ((74400,1),(80000,600))
   assert body.get('min_tokens')==body['max_tokens'] and body.get('ignore_eos') is True and body.get('temperature')==0 and body.get('seed')==20261017
   assert body.get('stream') is True and body.get('model')=='glm53-w8a8' and body.get('cache_salt') in self.namespaces
   assert body.get('return_token_ids') is False and body.get('skip_special_tokens') is False
   assert not any(key in body for key in ('bootstrap_host','bootstrap_room','kv_transfer_params','routed_dp_rank','rid'))
   match=re.search(r'(?:^|-)p(\d+)(?:-|$)',rid);assert match;pid=int(match.group(1));selected=route(pid)
   room=bootstrap_room(rid,selected['P']['global_dp']);assert room%4==selected['P']['global_dp'] and room not in self.rooms;self.rooms.add(room)
   row.update(logical_request_id=rid,namespace=body['cache_salt'],prefix_id=pid,route=selected,bootstrap_room=room,input_tokens=len(ids),output_target=body['max_tokens'],input_sha256_le_u32=hashlib.sha256(struct.pack('<'+'I'*len(ids),*ids)).hexdigest())
   common=dict(body);common.pop('request_id');common.update(bootstrap_host='172.16.12.34',bootstrap_port=29898,bootstrap_room=room,return_cached_tokens_details=True,return_spec_tokens_details=True)
   p=selected['P'];d=selected['D'];prefill=dict(common);prefill.update(rid=rid+'.P',routed_dp_rank=p['global_dp'],max_tokens=1,min_tokens=1,stream=False);prefill.pop('stream_options',None)
   decode=dict(common);decode.update(rid=rid+'.D',routed_dp_rank=d['global_dp'],disagg_prefill_dp_rank=p['global_dp'])
   p_url=f"http://{p['api_host']}:{p['api_port']}/v1/completions";d_url=f"http://{d['api_host']}:{d['api_port']}/v1/completions"
   if self.fixture_endpoints:p_url,d_url=self.fixture_endpoints
   row['status']='CONCURRENT_PD';p_task=asyncio.create_task(self.prefill(prefill,p_url,row));row['decode_dispatch_monotonic_ns']=time.monotonic_ns()
   async with self.session.post(d_url,json=decode,headers={'X-Request-Id':decode['rid']}) as response:
    row['decode_http_status']=response.status;row['decode_headers_monotonic_ns']=time.monotonic_ns()
    if response.status!=200:
     raw=await response.text();row['upstream_error_body']=raw;row['status']='DECODE_REJECTED';return web.Response(status=response.status,text=raw,content_type='application/json')
    stream=web.StreamResponse(status=200,headers={'Content-Type':'text/event-stream','Cache-Control':'no-cache'});await stream.prepare(request);tail=b'';done=False
    async for chunk in response.content.iter_any():
     if p_task.done():p_task.result()
     if 'first_forward_monotonic_ns' not in row:row['first_forward_monotonic_ns']=time.monotonic_ns()
     await stream.write(chunk);tail=(tail+chunk)[-512:]
     if b'\ndata: [DONE]\n' in b'\n'+tail:
      done=True;row['native_done_forwarded_monotonic_ns']=time.monotonic_ns();break
    row['native_done_forwarded']=done
    if not done:raise RuntimeError('Native SGLang decode ended without DONE')
    try:await stream.write_eof()
    except ConnectionResetError as error:row['downstream_eof_after_done']=repr(error)
   await asyncio.wait_for(asyncio.shield(p_task),10);row['status']='D_STREAM_COMPLETE';return stream
  except asyncio.CancelledError:
   row['status']='CANCELLED';row['error']='Gateway cancelled; native SGLang owns transfer cancellation';raise
  except Exception as error:
   row['status']='FAILED';row['error']=repr(error)
   if stream is None:return web.json_response({'error':{'message':str(error),'type':'GLM53SGLangGatewayError'}},status=502)
   raise
  finally:
   if p_task and not p_task.done():
    # Bounded native HTTP drain; no transfer ACK or SDK lifetime operation here.
    try:await asyncio.wait_for(asyncio.shield(p_task),10)
    except BaseException as error:
     row['prefill_drain_error']=repr(error)
     if not p_task.done():p_task.cancel();await asyncio.gather(p_task,return_exceptions=True)
   elif p_task:
    errors=await asyncio.gather(p_task,return_exceptions=True)
    if isinstance(errors[0],BaseException):row['prefill_terminal_error']=repr(errors[0])
   row['gateway_terminal_monotonic_ns']=time.monotonic_ns();self.active-=1
   if terminal_ticket is not None:self.terminals.finish(terminal_ticket,row)
   if self.active==0:self.idle.set()
   print('GLM53_SGLANG_GATEWAY_EVENT='+json.dumps(row,sort_keys=True),flush=True)
