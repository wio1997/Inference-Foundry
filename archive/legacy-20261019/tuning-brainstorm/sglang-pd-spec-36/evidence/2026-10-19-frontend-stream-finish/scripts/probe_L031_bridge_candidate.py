#!/usr/bin/env python3
"""Frozen normal/abort stream contract across original typed IPC and frontend bodies."""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
import asyncio,datetime,hashlib,json,pathlib,sys,traceback
from collections import Counter
from http import HTTPStatus
R=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'.deps/L031-msgspec020'));O=R/'runs/L031-R008-bridge-candidate';O.mkdir(parents=True,exist_ok=False);checks=[];cases=[];current='load';F=None
# Optional candidate mode is a separate frozen command and output, original default remains baseline.
def ck(v,label):checks.append({'case':current,'check':label,'pass':bool(v)})
def checksum(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
async def bridge(pickled,incremental,bs,budget,backlog,notify=16,reason=None,continuous=False):
 m,cfg=F.manager(incremental,notify);rids=[f'fixture-{i}' for i in range(bs)];up=F.Upstream(rids,budget,reason);states={};prepared={}
 for rid in rids:
  obj,state=F.add_state(m,rid,up.prompt_ids);states[rid]=state;prepared[rid]=m._wait_one_response(obj)
 c=F.chat(m,prepared);wire=F.Wire(pickled);received={};errors={};outputs={}
 async def collect(rid):
  try:outputs[rid]=[s async for s in c._generate_chat_stream(states[rid].obj,F.request_model(True,continuous),None)]
  except BaseException as e:errors[rid]={'type':type(e).__name__,'message':str(e)}
 tasks=[asyncio.create_task(collect(r)) for r in rids];await asyncio.sleep(0)
 try:
  while up.pos<budget:
   payload=wire.exchange(up.next());await m._handle_batch_output(payload)
   if not backlog:await asyncio.sleep(0)
  await asyncio.wait_for(asyncio.gather(*tasks),5)
  for rid in rids:
   ck(rid not in errors,'SSE generator terminates without exception');frames=F.parse_sse(outputs.get(rid,[]));errors_out=[x['error'] for x in frames if isinstance(x,dict) and 'error' in x];choices=[ch for x in frames if isinstance(x,dict) for ch in x.get('choices',[])];text=''.join(ch.get('delta',{}).get('content') or '' for ch in choices);fin=[ch['finish_reason'] for ch in choices if ch.get('finish_reason')];roles=[ch.get('delta',{}).get('role') for ch in choices if ch.get('delta',{}).get('role')];sglext=[x['sglext'] for x in frames if isinstance(x,dict) and x.get('sglext')];usage=[x['usage'] for x in frames if isinstance(x,dict) and x.get('usage')];expected_error=reason is not None and reason.get('status_code') is not None;gold=F.tokenizer.decode(up.IDs[rid],skip_special_tokens=True,spaces_between_special_tokens=True)
   ck(frames.count('[DONE]')==1 and frames[-1]=='[DONE]','one terminal DONE')
   if expected_error:
    ck(len(errors_out)==1 and errors_out[0]['code']==int(reason['status_code']),'coded scheduler abort becomes one SSE error regardless of codec')
    ck(not fin,'coded error does not become ordinary abort finish')
    ck(not any(z.get('input_ids') or z.get('output_ids') for z in sglext),'coded error omits sglext token IDs')
   else:
    ck(not errors_out,'normal/graceful abort has no coded error');ck(text==gold,'SSE text reconstructs exact upstream tokenizer text');ck(len(fin)==1 and fin[0]==('abort' if reason else 'length'),'one correct finish reason');ck(roles==['assistant'],'one assistant role');ck(len(sglext)==1 and sglext[0]['output_ids']==[up.IDs[rid]],'sglext IDs exact once across queued modes');ck(sglext[0]['input_ids']==up.prompt_ids,'input IDs preserved');ck(usage and usage[-1]['completion_tokens']==budget and usage[-1]['prompt_tokens']==len(up.prompt_ids),'final usage counts match fixture IDs');ck(usage[-1]['total_tokens']==budget+len(up.prompt_ids),'usage total tokens')
   ck(not m.rid_to_state and states[rid].finished,'finish removes manager RID, captured state remains valid');received[rid]={'text_sha256':hashlib.sha256(text.encode()).hexdigest(),'text_chars':len(text),'frames':len(frames),'finish':fin,'error_codes':[e['code'] for e in errors_out],'sglext_present':bool(sglext),'usage_completion':usage[-1]['completion_tokens'] if usage else None}
  return {'codec':'pickle' if pickled else 'msgpack','incremental_mode':incremental,'concurrency_fixture':bs,'output_budget':budget,'consumer_backlog':backlog,'batch_notify_size':notify,'reason':{k:int(v) if isinstance(v,HTTPStatus) else v for k,v in reason.items()} if reason else None,'received':received,'wire_records_digest':checksum(wire.records),'wire_finish_code_types':wire.records[-1]['finish_code_types'],'whole_BatchStrOutput_schema':True,'errors':errors}
 finally:
  for t in tasks:
   if not t.done():t.cancel()
  await asyncio.gather(*tasks,return_exceptions=True);wire.close()
async def main():
 global current
 for pickled in [True,False]:
  for inc in [False,True]:
   for bs,budget in [(2,128),(3,256)]:
    for backlog in [False,True]:
     current=f'normal-{pickled}-{inc}-c{bs}-b{budget}-backlog{backlog}';x=await bridge(pickled,inc,bs,budget,backlog,16);cases.append({'id':current,**x})
 for inc in [False,True]:
  current=f'notify1-interleave-{inc}';cases.append({'id':current,**await bridge(True,inc,3,128,False,1,continuous=True)})
 for pickled in [True,False]:
  for inc in [False,True]:
   for code in [None,HTTPStatus.BAD_REQUEST,HTTPStatus.INTERNAL_SERVER_ERROR,HTTPStatus.SERVICE_UNAVAILABLE]:
    reason={'type':'abort','message':'synthetic scheduler terminal reason'}
    if code is not None:reason['status_code']=code
    current=f'abort-{pickled}-{inc}-{code}';cases.append({'id':current,**await bridge(pickled,inc,2,128,True,16,reason)})
 current='abort-dispatch-and-retry'
 m,cfg=F.manager();obj,state=F.add_state(m,'abort-rid',[]);m.abort_request('');ck(not m.dispatched,'empty non-global abort ignored');m.abort_request('abort-rid');m.abort_request('abort-rid');ck(len(m.dispatched)==1 and state.abort_sent,'abort dispatch deduplicated');m.abort_request('missing');ck(len(m.dispatched)==1,'single worker missing RID ignored');m.abort_request(abort_all=True);ck(len(m.dispatched)==2 and m.dispatched[-1].abort_all,'explicit abort-all dispatches');cases.append({'id':current,'dispatches':len(m.dispatched)})
 current='abort-dispatch-failure-retry'
 m,cfg=F.manager();obj,state=F.add_state(m,'retry-rid',[])
 def fail(req):raise RuntimeError('synthetic dispatch failure')
 m._dispatch_to_scheduler=fail
 try:m.abort_request('retry-rid');ck(False,'dispatch failure must surface')
 except RuntimeError:ck(not state.abort_sent,'failed dispatch resets dedup flag')
 m._dispatch_to_scheduler=m.dispatched.append;m.abort_request('retry-rid');ck(len(m.dispatched)==1 and state.abort_sent,'retry after failed dispatch not suppressed');cases.append({'id':current,'retry_dispatched':True})
try:
 import L031_frontend_components as F
 import ast
 candidate_path=R/'candidates/L031-status-code/python/sglang/srt/entrypoints/openai/serving_chat.py'
 tree=ast.parse(candidate_path.read_text());cl=next(z for z in tree.body if isinstance(z,ast.ClassDef) and z.name=='OpenAIServingChat');fn=next(z for z in cl.body if isinstance(z,ast.AsyncFunctionDef) and z.name=='_generate_chat_stream')
 module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fn],type_ignores=[]);ast.fix_missing_locations(module);exec(compile(module,str(candidate_path),'exec'),F.ns);F.Chat._generate_chat_stream=F.ns['_generate_chat_stream']
 F.records.append({'path':str(candidate_path.relative_to(R)),'symbol':'_generate_chat_stream','AST_sha256':hashlib.sha256(ast.dump(fn,include_attributes=False).encode()).hexdigest(),'candidate_override':True})
 assert F.msgspec.__version__=='0.20.0',F.msgspec.__version__
 asyncio.run(main());failures=[z for z in checks if not z['pass']];status='FAIL' if failures else 'PASS';result={'status':status,'semantic_checks':len(checks),'case_count':len(cases),'failed_checks':failures,'checks':checks,'cases':cases,'source_bodies':F.records,'boundaries':F.BOUNDARIES,'native_executed':False,'HTTP_server_executed':False,'SSE_generator_executed':True,'original_full_BatchStrOutput_schema_and_array_codec_executed':True,'candidate_applied_in_CPU_fixture_only':True,'msgspec_version':F.msgspec.__version__,'dependency_overlay':'.deps/L031-msgspec020','default_source_IPC':'pickle=True, sourceenviron; actual runtime UNKNOWN','performance':'NOT_MEASURED','finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};(O/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'status':status,'semantic_checks':len(checks),'case_count':len(cases),'failed_checks':len(failures)}));sys.exit(1 if failures else 0)
except SystemExit:raise
except BaseException as e:
 (O/'result.json').write_text(json.dumps({'status':'HARNESS_ERROR','failed_case':current,'error_type':type(e).__name__,'error':str(e),'completed_checks':len(checks),'completed_cases':cases,'native_executed':False},ensure_ascii=False,indent=2)+'\n');traceback.print_exc();sys.exit(2)
