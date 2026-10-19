#!/usr/bin/env python3
"""Real loopback HTTP/SSE with exact wrapper/generator, original IPC and injected upstream."""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
import ast,asyncio,datetime,hashlib,json,pathlib,socket,sys,traceback
from http import HTTPStatus
R=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'.deps/L031-msgspec020'));O=R/'runs/L031-R009-http';O.mkdir(parents=True,exist_ok=False);checks=[];cases=[];current='load'
def ck(v,label):
 checks.append({'case':current,'check':label,'pass':bool(v)})
 if not v:raise AssertionError(current+': '+label)
try:
 import httpx,uvicorn
 from fastapi import FastAPI,Request
 import L031_frontend_components as F
 assert F.msgspec.__version__=='0.20.0'
 original=F.Chat._generate_chat_stream;candidate_path=R/'candidates/L031-status-code/python/sglang/srt/entrypoints/openai/serving_chat.py';t=ast.parse(candidate_path.read_text());cl=next(x for x in t.body if isinstance(x,ast.ClassDef) and x.name=='OpenAIServingChat');fn=next(x for x in cl.body if isinstance(x,ast.AsyncFunctionDef) and x.name=='_generate_chat_stream');mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fn],type_ignores=[]);ast.fix_missing_locations(mod);exec(compile(mod,str(candidate_path),'exec'),F.ns);candidate=F.ns['_generate_chat_stream']
 async def one(name,pickled=True,incremental=False,use_candidate=True,reason=None,framed=False):
  global current
  current=name;F.Chat._generate_chat_stream=candidate if use_candidate else original;m,cfg=F.manager(incremental);wire=F.Wire(pickled);app=FastAPI();ctx={};producer_tasks=[];dispatch_event=asyncio.Event();release=asyncio.Event();client_close=False
  def dispatch(req):
   decoded=wire.exchange(req);m.dispatched.append(decoded);dispatch_event.set()
  m._dispatch_to_scheduler=dispatch
  @app.post('/v1/chat/completions')
  async def endpoint(raw:Request):
   request=F.protocol.ChatCompletionRequest.model_validate(await raw.json());rid='HTTP-synthetic';adapted=F.GenerateFixture(rid=rid,is_single=True)
   if name=='pre-validation-error':
    async def rejected(adapted,raw):
     if False:yield None
     raise ValueError('synthetic pre-validation failure')
    m.generate_request=rejected;c=F.chat(m);return await c._handle_streaming_request(adapted,request,raw)
   up=F.Upstream([rid],128,reason);obj,state=F.add_state(m,rid,up.prompt_ids);prepared=m._wait_one_response(obj,raw);ctx.update(upstream=up,state=state,obj=obj)
   async def generated(adapted,raw):
    async for out in prepared:
     yield out
     if name=='late-generation-error':raise ValueError('synthetic post-first-frame failure')
   m.generate_request=generated;c=F.chat(m)
   async def produce():
    while up.pos<up.budget:
     await m._handle_batch_output(wire.exchange(up.next()))
     if name in ['disconnect','late-generation-error'] and up.turn>=2:await release.wait()
     await asyncio.sleep(0)
   producer_tasks.append(asyncio.create_task(produce()));return await c._handle_streaming_request(obj,request,raw)
  sock=socket.socket();sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);sock.bind(('127.0.0.1',0));sock.listen(8);port=sock.getsockname()[1];server=uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=port,log_level='error',lifespan='off',loop='asyncio',timeout_keep_alive=1));task=asyncio.create_task(server.serve(sockets=[sock]));frames=[];status=None;contenttype=None;prebody=None
  try:
   for _ in range(500):
    if server.started:break
    if task.done():await task
    await asyncio.sleep(.01)
   ck(server.started,'task-owned loopback server started');headers={'x-sglext-ids-framed':'1'} if framed else {}
   async with httpx.AsyncClient(timeout=15,trust_env=False) as client:
    async with client.stream('POST',f'http://127.0.0.1:{port}/v1/chat/completions',json=F.request_model().model_dump(mode='json'),headers=headers) as response:
     status=response.status_code;contenttype=response.headers.get('content-type','')
     if name=='pre-validation-error':prebody=json.loads(await response.aread())
     else:
      lines=[]
      async for line in response.aiter_lines():
       if line:lines.append(line);continue
       if not lines:continue
       frame='\n'.join(lines)+'\n\n';lines=[];frames.append(frame)
       if name=='disconnect':
        parsed=F.parse_sse([frame]);visible=any((c.get('delta',{}).get('content') or '') for d in parsed if isinstance(d,dict) for c in d.get('choices',[]))
        if visible:client_close=True;break
   result={'id':name,'codec':'pickle' if pickled else 'msgpack','incremental_mode':incremental,'candidate':use_candidate,'framed_ids':framed,'HTTP_status':status,'content_type':contenttype,'server':'uvicorn/FastAPI/Starlette on127.0.0.1 ephemeral port','full_endpoint_constructors_or_model_executed':False}
   if name=='pre-validation-error':
    ck(status==400 and 'application/json' in contenttype,'pre-first-frame ValueError becomes HTTP400 JSON');ck(prebody['code']==400 and prebody['message']=='synthetic pre-validation failure','exact source error response body');result['error_code']=prebody['code']
   elif name=='disconnect':
    ck(status==200 and client_close,'real HTTP client closes after visible text');await asyncio.wait_for(dispatch_event.wait(),6);ck(len(m.dispatched)==1 and m.dispatched[0].rid=='HTTP-synthetic' and not m.dispatched[0].abort_all,'background source abort dispatches exact AbortReq once');ck(ctx['state'].abort_sent and 'HTTP-synthetic' in m.rid_to_state,'abort sent flag set; actual scheduler terminal/KV boundary remains');result.update(abort_dispatched=1,RID_still_pending_without_real_scheduler=True,actual_client_disconnect=True,KV_free_proven=False)
   else:
    ck(status==200 and 'text/event-stream' in contenttype,'actual HTTP200 SSE response');parsed=F.parse_sse(frames);errors=[z['error'] for z in parsed if isinstance(z,dict) and z.get('error')];choices=[c for z in parsed if isinstance(z,dict) for c in z.get('choices',[])];text=''.join(c.get('delta',{}).get('content') or '' for c in choices);fin=[c['finish_reason'] for c in choices if c.get('finish_reason')];ext=[z['sglext'] for z in parsed if isinstance(z,dict) and z.get('sglext')];usage=[z['usage'] for z in parsed if isinstance(z,dict) and z.get('usage')];ck(parsed[-1]=='[DONE]' and parsed.count('[DONE]')==1,'one terminal DONE over network')
    if name in ['coded-error-candidate','late-generation-error']:
     want=503 if name=='coded-error-candidate' else 400;ck(len(errors)==1 and errors[0]['code']==want,'expected coded SSE error over network');ck(not fin,'error path omits ordinary finish');ck(not ext,'error path omits token-ID extension')
    elif name=='coded-error-baseline':
     ck(not errors and fin==['abort'] and bool(ext),'known baseline bug reproduced over actual HTTP');result['semantic_verdict']='OBSERVED_BASELINE_BUG_NOT_ACCEPTED'
    else:
     up=ctx['upstream'];gold=F.tokenizer.decode(up.IDs['HTTP-synthetic'],skip_special_tokens=True,spaces_between_special_tokens=True);ck(not errors and text==gold,'HTTP SSE reassembles exact fixture text');ck(fin==(['abort'] if reason else ['length']),'normal/unknown-code original finish behavior preserved');ck(len(ext)==1 and ext[0]['output_ids']==[up.IDs['HTTP-synthetic']],'HTTP token IDs exact once');ck(usage[-1]['completion_tokens']==128 and usage[-1]['prompt_tokens']==len(up.prompt_ids),'HTTP usage matches exact fixture token counts')
     if framed:ck(sum(s.startswith('event: sglext_ids\n') for s in frames)==1,'named token-ID frame received intact')
    result.update(frames=len(parsed),finish_reasons=fin,error_codes=[z['code'] for z in errors],extension_frames=len(ext),text_sha256=hashlib.sha256(text.encode()).hexdigest())
   return result
  finally:
   release.set()
   for p in producer_tasks:
    if not p.done():p.cancel()
   await asyncio.gather(*producer_tasks,return_exceptions=True)
   server.should_exit=True
   try:await asyncio.wait_for(task,10)
   except asyncio.TimeoutError:server.force_exit=True;task.cancel();await asyncio.gather(task,return_exceptions=True)
   sock.close();wire.close();m.rid_to_state.clear()
 async def main():
  configs=[('normal-default',dict(pickled=True,incremental=False,use_candidate=False,framed=True)),('normal-msgpack-incremental',dict(pickled=False,incremental=True)),('coded-error-baseline',dict(pickled=False,use_candidate=False,reason={'type':'abort','message':'synthetic code503','status_code':HTTPStatus.SERVICE_UNAVAILABLE})),('coded-error-candidate',dict(pickled=False,reason={'type':'abort','message':'synthetic code503','status_code':HTTPStatus.SERVICE_UNAVAILABLE})),('unknown-integer-code',dict(pickled=False,reason={'type':'abort','message':'unknown code preserves source fallback','status_code':599})),('string-code',dict(pickled=False,reason={'type':'abort','message':'string code preserves source fallback','status_code':'503'})),('pre-validation-error',{}),('late-generation-error',{}),('disconnect',{})]
  for name,args in configs:cases.append(await one(name,**args))
 asyncio.run(main());result={'status':'PASS','case_count':len(cases),'semantic_checks':len(checks),'checks':checks,'cases':cases,'native_executed':False,'real_loopback_HTTP_executed':True,'actual_client_disconnect_to_abort_dispatch_executed':True,'real_scheduler_abort_KV_reclaim_proven':False,'explicit_boundaries':F.BOUNDARIES+['Actual HTTP/TCP and client close tested; scheduling/model generate_request remains deterministic injected producer. Background2-second sleep unchanged; no timing/performance interpretation.','Request/KV state remains pending after abort dispatch until real scheduler handles it, which is untested; fixture finally clears state for cleanup.'],'performance':'NOT_MEASURED','finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};(O/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'status':'PASS','cases':len(cases),'semantic_checks':len(checks)}))
except BaseException as e:
 (O/'result.json').write_text(json.dumps({'status':'FAIL','failed_case':current,'error_type':type(e).__name__,'error':str(e),'completed_cases':cases,'checks':checks,'native_executed':False},ensure_ascii=False,indent=2)+'\n');traceback.print_exc();sys.exit(1)
