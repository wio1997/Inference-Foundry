#!/usr/bin/env python3
import asyncio,hashlib,json,sys,traceback
from array import array
from L036_streaming_components_v2 import R,ns,records,registry,manager,TimeBoundary,branch,ComposedView,composed_scheduler,BOUNDARIES_L036
O=R/'runs/L036-R004-streaming-reference';O.mkdir(parents=True,exist_ok=False)
checks=[];cases=[];stage='setup';error=None
Ref=registry.LoRARef
A=Ref(lora_id=Ref.deterministic_id('A','/synthetic/A'),lora_name='A',lora_path='/synthetic/A')
B=Ref(lora_id=Ref.deterministic_id('B','/synthetic/B'),lora_name='B',lora_path='/synthetic/B')
def ck(name,value,**detail):checks.append({'stage':stage,'name':name,'pass':bool(value),'detail':detail})

async def submit(m,s,sid,rid,ids,budget,lora='A',key='tenant',salt='salt'):
 q=ns['GenerateReqInput'](rid=rid,input_ids=list(ids),lora_path=lora,extra_key=key,cache_salt=salt,session_params={'id':sid},sampling_params={'max_new_tokens':budget},bootstrap_host='cpu-P',bootstrap_port=8998,bootstrap_room=101,stream=True)
 q.normalize_batch_and_arguments();await m._validate_and_resolve_lora(q);m.rid_to_state[q.rid]=type('State',(),{'time_stats':TimeBoundary()})();t=m._create_tokenized_object(q,None,q.input_ids);s.handle_generate_request(t);return s.queued[-1],q,t

async def seed(ids,budget,page,npu,mode,sid,finish=True):
 branch.npu=npu;v=ComposedView(page,True,len(ids)+4*page+128);s=composed_scheduler(v,mode);m=manager([A,B]);out=s.session_controller.open(ns['OpenSessionReqInput'](capacity_of_str_len=100000,session_id=sid,streaming=True));ck('original controller open',out.success)
 req,q,t=await submit(m,s,sid,sid+'-first',ids,budget);ck('first admitted through session branch',req.session is s.session_controller.get(sid) and req.to_finish is None and req.lora_id==A.lora_id);ck('requested budget retained',req.sampling_params.max_new_tokens==budget)
 req.init_next_round_input(v);ck('first falls through real empty Core',len(req.prefix_indices)==0 and v.tree_core.calls==1)
 req.output_ids.extend([7,8,9]);req._refresh_fill_ids();req.finished_len=3;allocated=len(ids)+3+2*page+7;req.kv.req_pool_idx=0;req.kv.kv_committed_len=len(ids)+3;req.kv.kv_allocated_len=allocated;v.token_to_kv_pool_allocator.own(v.req_to_token_pool.req_to_token[0,:allocated]);oldkv=req.kv
 if finish:finish_seed(v,req,oldkv)
 await m.lora_registry.release(q.lora_id)
 return v,s,m,req,oldkv

def finish_seed(v,req,kv):
 req.finished_reason=ns['FINISH_LENGTH'](3);ck('original finish handler',v.session.try_cache_finished_req(req));slot=v.session.slots[req.session.session_id];ck('slot takes original KV record',slot.kv is kv and not req.kv.holds_kv);ck('finish commits synthetic output3',slot.kv.kv_committed_len==len(req.origin_input_ids)+3 and slot.kv.kv_allocated_len==len(req.origin_input_ids)+3);ck('finish clears inflight',req.session._inflight is False)

def cleanup_abort(v,s,req,kv):
 sid=req.session.session_id;req.finished_reason=ns['FINISH_ABORT']('CPU lifecycle fixture');ck('mid-abort handler',v.session.try_cache_finished_req(req));ck('mid-abort drops slot and inflight',sid not in v.session.slots and not s.session_controller.get(sid)._inflight);s.session_controller.close(ns['CloseSessionReqInput'](session_id=sid));ck('controller closes after abort',sid not in s.session_controller)
 ck('CPU row released once',v.req_to_token_pool.freed_rows==[0] and not kv.holds_kv);ck('CPU pages fully released without duplicates',v.token_to_kv_pool_allocator.freed==v.token_to_kv_pool_allocator.owned)

async def main():
 global stage
 manifest=json.loads((R/'fixtures/M001-v1/manifest.json').read_text());fixtures={x['label']:x for x in manifest['fixtures']};inputs={}
 for label in ['c2-short-0','c3-long-0']:
  f=fixtures[label];p=R/f['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256'];inputs[label]=json.loads(p.read_text())['input_ids'];assert len(inputs[label])==f['input_tokens']
 for label,budget in [('c2-short-0',128),('c3-long-0',256)]:
  for mode in [ns['DisaggregationMode'].PREFILL,ns['DisaggregationMode'].DECODE]:
   for page,npu in [(1,False),(128,False),(128,True)]:
    for change in ['same','adapter','caller-key','salt']:
     stage=str((label,mode.value,page,npu,change));sid='s-'+str(len(cases));v,s,m,first,kv=await seed(inputs[label],budget,page,npu,mode,sid);before=(first.lora_id,first.extra_key,first.cache_salt);lora='B' if change=='adapter' else 'A';key='other-tenant' if change=='caller-key' else 'tenant';salt='other-salt' if change=='salt' else 'salt'
     req,q,t=await submit(m,s,sid,sid+'-next',[10,11],budget,lora,key,salt);after=(req.lora_id,req.extra_key,req.cache_salt);ck('second passes original admission',req.to_finish is None and req.session is s.session_controller.get(sid));ck('identity propagated through session constructor',(before==after)==(change=='same'));ck('session reconstructs append history',list(req.origin_input_ids)==inputs[label]+[7,8,9,10,11]);ck('second budget retained',req.sampling_params.max_new_tokens==budget)
     rawcalls=v.tree_core.calls;req.init_next_round_input(v);hit=len(req.prefix_indices);want=len(inputs[label])+3;want=(want//page)*page if npu and page>1 else want
     ck('facade dispatch and prefix length coherent',v.tree_core.calls==rawcalls+(0 if hit else 1) and hit in (0,want));ck('returned metadata ownership and provenance',(req.kv is kv and req.prefix_indices.tolist()==v.req_to_token_pool.req_to_token[0,:hit].tolist()) if hit else not req.kv.holds_kv)
     if change=='same':ck('CONTRACT same identity reuses metadata',hit>0)
     else:ck('CONTRACT changed identity must not restore old metadata',hit==0,identity=change,hit_indices=hit)
     freelen=len(v.token_to_kv_pool_allocator.calls);indices=req.prefix_indices.tolist();stablekv=req.kv;req.init_next_round_input(v);ck('lookup retry idempotent',req.kv is stablekv and req.prefix_indices.tolist()==indices and len(v.token_to_kv_pool_allocator.calls)==freelen)
     cleanup_abort(v,s,req,kv);await m.lora_registry.release(q.lora_id);ck('registry fully released',all(c.value()==0 for c in m.lora_registry._counters.values()));cases.append({'kind':'identity','input_fixture':label,'input_tokens':len(inputs[label]),'output_budget':budget,'mode':mode.value,'page_size':page,'simulated_is_npu_branch':npu,'change':change,'hit_indices':hit,'raw_Core_calls_across_second_and_retry':v.tree_core.calls-rawcalls,'freed_pages':len(v.token_to_kv_pool_allocator.freed),'native':False})
 stage='lifecycle-idle-close';v,s,m,first,kv=await seed(inputs['c2-short-0'],128,128,True,ns['DisaggregationMode'].PREFILL,'idle');s.session_controller.close(ns['CloseSessionReqInput'](session_id='idle'));ck('idle close releases slot and controller','idle' not in s.session_controller and not v.session.slots);ck('idle close row/pages ledger',v.req_to_token_pool.freed_rows==[0] and not kv.holds_kv and v.token_to_kv_pool_allocator.freed==v.token_to_kv_pool_allocator.owned);cases.append({'kind':stage,'native':False})
 stage='lifecycle-deferred-close';v,s,m,first,kv=await seed(inputs['c2-short-0'],128,128,True,ns['DisaggregationMode'].DECODE,'deferred',finish=False);s.session_controller.close(ns['CloseSessionReqInput'](session_id='deferred'));ck('close while inflight defers','deferred' in s.session_controller and first.session.close_on_finish and kv.holds_kv and not v.req_to_token_pool.freed_rows);finish_seed(v,first,kv);s.session_controller.maybe_reap(10**12);ck('reap closes after finish','deferred' not in s.session_controller and not v.session.slots);ck('deferred row/pages released once',v.req_to_token_pool.freed_rows==[0] and not kv.holds_kv and v.token_to_kv_pool_allocator.freed==v.token_to_kv_pool_allocator.owned);cases.append({'kind':stage,'native':False})
 stage='lifecycle-preabort-recover';v,s,m,first,kv=await seed(inputs['c2-short-0'],128,128,True,ns['DisaggregationMode'].PREFILL,'preabort');s.max_req_input_len=1;bad,q,t=await submit(m,s,'preabort','too-long',[10,11],128);ck('original length guard preaborts',isinstance(bad.to_finish,ns['FINISH_ABORT']));bad.init_next_round_input(v);ck('preabort detaches and preserves slot',bad.session is None and not bad.kv.holds_kv and v.session.slots['preabort'].kv is kv and kv.holds_kv and not s.session_controller.get('preabort')._inflight);await m.lora_registry.release(q.lora_id);s.max_req_input_len=65536;good,q,t=await submit(m,s,'preabort','recover',[12,13],128);ck('aborted append rolled back',list(good.origin_input_ids)==inputs['c2-short-0']+[7,8,9,12,13]);good.init_next_round_input(v);ck('recovery restores original slot',good.kv is kv and len(good.prefix_indices)==512);cleanup_abort(v,s,good,kv);await m.lora_registry.release(q.lora_id);cases.append({'kind':stage,'native':False})
 stage='lifecycle-small-NPU-branch';v,s,m,first,kv=await seed(inputs['c2-short-0'][:20],128,128,True,ns['DisaggregationMode'].PREFILL,'tiny');req,q,t=await submit(m,s,'tiny','tiny-next',[10,11],128);prior=v.tree_core.calls;req.init_next_round_input(v);ck('subpage original branch releases and falls back',not v.session.slots and not kv.holds_kv and not req.kv.holds_kv and len(req.prefix_indices)==0 and v.tree_core.calls==prior+1);ck('subpage ledger releases all once',v.req_to_token_pool.freed_rows==[0] and v.token_to_kv_pool_allocator.freed==v.token_to_kv_pool_allocator.owned);s.session_controller.get('tiny').abort_req();s.session_controller.close(ns['CloseSessionReqInput'](session_id='tiny'));await m.lora_registry.release(q.lora_id);cases.append({'kind':stage,'input_tokens':20,'page_size':128,'simulated_is_npu_branch':True,'native':False})

try:asyncio.run(main())
except Exception as e:error={'stage':stage,'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
failed=[x for x in checks if not x['pass']];result={'status':'INVALID' if error else 'FAIL' if failed else 'PASS','harness_error':error,'case_count':len(cases),'semantic_checks':len(checks),'failed_checks':len(failed),'checks':checks,'cases':cases,'source_components':records,'boundaries':BOUNDARIES_L036,'native_executed':False,'current_deployment_fault_proven':False,'candidate_accepted':False,'production_applied':False,'performance':'NOT_MEASURED'};(O/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['status','case_count','semantic_checks','failed_checks','harness_error']}));sys.exit(2 if error else 1 if failed else 0)
