#!/usr/bin/env python3
import asyncio,hashlib,json,pathlib,sys,dataclasses,importlib.metadata as metadata
from types import SimpleNamespace as NS
from array import array
from L034_extension_components import R,ns,registry,identity,records,BOUNDARIES,flags,model_cfg,mm_cfg,Image,np,torch,msgspec
O=R/'runs/L034-R004-extension-cpu';O.mkdir(parents=True,exist_ok=False);checks=[];group='';observations=[]
def ck(name,value,**detail):checks.append({'group':group,'name':name,'pass':bool(value),'detail':detail})
def raises(fn,typ):
 try:fn()
 except typ:return True
 return False
Req=ns['Req'];Session=ns['Session'];Slot=ns['SessionSlot'];Params=ns['SessionParams'];sp=NS(custom_params=None,max_new_tokens=2)
def req(rid,ids=(1,2),**kw):return Req(rid=rid,origin_input_text=None,origin_input_ids=array('q',ids),sampling_params=sp,**kw)
def carrier(rid,ids=(3,),**kw):
 c=ns['TokenizedGenerateReqInput'];fields=msgspec.structs.fields(c);v={x.name:None for x in fields if x.default is msgspec.NODEFAULT and x.default_factory is msgspec.NODEFAULT};v.update(rid=rid,input_ids=array('q',ids),sampling_params=sp,session_params=Params(id='s'),return_logprob=False,top_logprobs_num=0,stream=True,**kw);return c(**v)
def key(q):return (q.extra_key,q.cache_salt)
async def main():
 global group
 group='registry-live-holders-reload'
 a=registry.LoRARef(lora_name='A',lora_path='/synthetic/A',pinned=True);b=registry.LoRARef(lora_name='B',lora_path='/synthetic/B',pinned=False);reg=registry.LoRARegistry([a,b]);got=await reg.acquire(['A','A',None,'B']);ck('batchIDs',got==[a.lora_id,a.lora_id,None,b.lora_id]);ck('holdercounts',reg._counters[a.lora_id].value()==2 and reg._counters[b.lora_id].value()==1);ck('pinned excluded LRU',await reg.lru_lora_name(exclude_pinned=True)=='B')
 old=await reg.unregister('A');wait=asyncio.create_task(reg.wait_for_unload(old));await asyncio.sleep(0);ck('unload blocks onholders',not wait.done());new=registry.LoRARef(lora_name='A',lora_path='/synthetic/A-v2');await reg.register(new);newid=await reg.acquire('A');ck('reload distinct identity',newid!=old);await reg.release([old,None]);await asyncio.sleep(0);ck('one old holder still blocks',not wait.done());await reg.release(old);await asyncio.wait_for(wait,2);ck('old counter removed new alive',old not in reg._counters and reg._counters[newid].value()==1);await reg.release([newid,b.lora_id]);ck('new balanced',all(c.value()==0 for c in reg._counters.values()))
 group='registry-invalid-batch-and-cancel'
 before={k:v.value() for k,v in reg._counters.items()}
 try:await reg.acquire(['A','missing'])
 except ValueError:ck('unknown batch rejected',True)
 else:ck('unknown batch rejected',False)
 ck('no partial counter increment',before=={k:v.value() for k,v in reg._counters.items()});ids=await asyncio.gather(*[reg.acquire('B') for _ in range(3)]);await reg.unregister('B');w=asyncio.create_task(reg.wait_for_unload(b.lora_id));await asyncio.sleep(0);w.cancel()
 try:await w
 except asyncio.CancelledError:pass
 ck('cancelled wait preserves counter',reg._counters[b.lora_id].value()==3);await asyncio.gather(*[reg.release(x) for x in ids]);await asyncio.wait_for(reg.wait_for_unload(b.lora_id),2);ck('retry wait removes zero counter',b.lora_id not in reg._counters)
 group='deterministic-adapter-id'
 aid=registry.LoRARef.deterministic_id('A','/synthetic/A');ck('deterministic32hex',len(aid)==32 and aid==registry.LoRARef.deterministic_id('A','/synthetic/A'));ck('path changes ID',aid!=registry.LoRARef.deterministic_id('A','/synthetic/B'));ck('name changes ID',aid!=registry.LoRARef.deterministic_id('B','/synthetic/A'))
 group='request-cache-namespace'
 from L034_extension_components import load
 load('python/sglang/srt/mem_cache/radix_cache.py',classes=['RadixKey']);RadixKey=ns['RadixKey'];base=req('base');adapt=req('adapt',lora_id=aid);other=req('other',lora_id=registry.LoRARef.deterministic_id('B','/synthetic/B'));ck('normal base adapter distinct',key(base)!=key(adapt));ck('normal two adapters distinct',key(adapt)!=key(other));salted=req('salt',lora_id=aid,cache_salt='tenant2');ck('salt changes childkey',RadixKey(adapt.origin_input_ids,extra_key=adapt.extra_key).child_key()!=RadixKey(salted.origin_input_ids,extra_key=salted.extra_key,cache_salt=salted.cache_salt).child_key())
 collision=req('base-caller-key',extra_key=aid);kb=RadixKey(collision.origin_input_ids,extra_key=collision.extra_key);ka=RadixKey(adapt.origin_input_ids,extra_key=adapt.extra_key);ck('CONTRACT base-vs-adapter caller key must separate',kb.child_key()!=ka.child_key(),base_lora_id=collision.lora_id,adapter_lora_id=adapt.lora_id,base_extra_key=collision.extra_key,adapter_extra_key=adapt.extra_key,actual_match=kb.match(ka));observations.append({'kind':'conditional_namespace_collision','equal_child_key':kb.child_key()==ka.child_key(),'full_prefix_matched':kb.match(ka),'current_deployment_reachability':'UNKNOWN'})
 group='streaming-session-append-abort-heal'
 s=Session(1000,'s',streaming=True);q=s.create_req(carrier('turn1',(1,2)),None,100);q.output_ids.extend([4,5]);q.full_untruncated_fill_ids=array('q',[1,2,4,5]);q.finished_reason=ns['FINISH_LENGTH'](2);s.finish_req(q);nextq=s.create_req(carrier('turn2',(6,)),None,100);ck('append committed output',list(nextq.origin_input_ids)==[1,2,4,5,6]);ck('shared origin',nextq.origin_input_ids is q.origin_input_ids);blocked=s.create_req(carrier('parallel',(7,)),None,100);ck('second inflight rejected',isinstance(blocked.to_finish,ns['FINISH_ABORT']));s.abort_req();healed=s.create_req(carrier('turn3',(8,)),None,100);ck('aborted append trimmed',list(healed.origin_input_ids)==[1,2,4,5,8]);ck('carry fill matches',list(healed.full_untruncated_fill_ids)==[1,2,4,5,8]);s.abort_req()
 group='streaming-slot-identity-precondition'
 first=req('first',lora_id=aid,extra_key='tenant',cache_salt='salt');first.kv.req_pool_idx=3;first.kv.kv_committed_len=2;first.kv.kv_allocated_len=2;first.session=Session(1000,'identity',streaming=True);slot=Slot();slot.save_from_req(first,True);streaming=ns['StreamingSession'](None);streaming.slots['identity']=slot
 for name,kw in [('same',{'lora_id':aid,'extra_key':'tenant','cache_salt':'salt'}),('adapter',{'lora_id':other.lora_id,'extra_key':'tenant','cache_salt':'salt'}),('callerkey',{'lora_id':aid,'extra_key':'other','cache_salt':'salt'}),('salt',{'lora_id':aid,'extra_key':'tenant','cache_salt':'other'})]:
  incoming=req('next-'+name,session=first.session,**kw);found=streaming.find_active_slot(incoming)
  if name=='same':ck('same identity finds slot',found is slot)
  else:ck('CONTRACT changed '+name+' identity must reject saved slot',found is None,new_namespace=key(incoming),saved_namespace=key(first),metadata_row=slot.kv.req_pool_idx)
  if found:found.restore_to_req(incoming);ck('actual metadata record transferred '+name,incoming.kv is slot.kv)
 group='media-content-and-artifact-identity'
 red=Image.new('RGB',(2,2),(255,0,0));snap=identity.snapshot_media(red);red.putpixel((0,0),(0,0,0));ck('PIL snapshot independent',snap.data.getpixel((0,0))==(255,0,0));changed=identity.snapshot_media(red);ck('PIL mutation changes digest',changed.content_digest!=snap.content_digest);rgb=identity.snapshot_media(Image.new('RGB',(2,2)));gray=identity.snapshot_media(Image.new('L',(2,2)));ck('PIL mode distinct',rgb.content_digest!=gray.content_digest)
 a=np.arange(6,dtype=np.int32).reshape(2,3);b=a.copy();n=identity.snapshot_media(a);a[0,0]=99;ck('numpy snapshot independent',n.data[0,0]==0);ck('numpy equal snapshots same digest',n.content_digest==identity.snapshot_media(b).content_digest);ck('dtype distinct',n.content_digest!=identity.snapshot_media(b.astype(np.int64)).content_digest);t=torch.arange(6,dtype=torch.int32).reshape(2,3);ts=identity.snapshot_media(t);t[0,0]=99;ck('torch snapshot independent',ts.data[0,0].item()==0)
 fingerprint='sha256:'+'a'*64;k=identity.build_artifact_key(snap.content_digest,modality='image',processor_fingerprint=fingerprint,preprocess_kwargs={'x':1,'y':2});ck('mapping order stable',k==identity.build_artifact_key(snap.content_digest,modality='image',processor_fingerprint=fingerprint,preprocess_kwargs={'y':2,'x':1}));ck('preprocess settings distinct',k!=identity.build_artifact_key(snap.content_digest,modality='image',processor_fingerprint=fingerprint,preprocess_kwargs={'x':2,'y':2}));ck('modality distinct',k!=identity.build_artifact_key(snap.content_digest,modality='video',processor_fingerprint=fingerprint,preprocess_kwargs={'x':1,'y':2}));ck('processor distinct',k!=identity.build_artifact_key(snap.content_digest,modality='image',processor_fingerprint='sha256:'+'b'*64,preprocess_kwargs={'x':1,'y':2}));ck('invalid hash rejected',raises(lambda:identity.parse_content_hash('sha256:zz'),ValueError));h=identity.resolve_multimodal_item_hash(existing_hash=7,namespace=k);ck('artifact namespace scopes itemhash',h!=identity.resolve_multimodal_item_hash(existing_hash=7,namespace='sha256:'+'c'*64));ck('itemhash deterministic',h==identity.resolve_multimodal_item_hash(existing_hash=7,namespace=k))
 group='lora-spec-source-predicates'
 for alg,adaptive,plan,ragged,want in [('EAGLE',False,False,'static',True),('EAGLE3',False,False,'static',True),('DSPARK',False,False,'static',True),('DSPARK',False,False,'cap_accept',False),('EAGLE',True,False,'static',False),('EAGLE',False,True,'static',False),('FROZEN_KV_MTP',False,False,'static',False),('NEXTN',False,False,'static',False)]:
  flags.SGLANG_RAGGED_VERIFY_MODE.v=ragged;flags.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.v=plan;cfg=NS(speculative_algorithm=alg,speculative_adaptive=adaptive,moe_runner_backend='cpu-fixture',speculative_moe_runner_backend='cpu-fixture');ok=not raises(lambda:ns['check_lora_speculative_compatibility'](cfg),ValueError);ck('resolved '+str((alg,adaptive,plan,ragged)),ok==want)
asyncio.run(main())
failed=[x for x in checks if not x['pass']];result={'status':'FAIL' if failed else 'PASS','case_groups':len({x['group'] for x in checks}),'semantic_checks':len(checks),'passed_checks':len(checks)-len(failed),'failed_checks':len(failed),'checks':checks,'observations':observations,'source_components':records,'boundaries':BOUNDARIES,'packages':{n:metadata.version(n) for n in ['torch','msgspec','numpy','transformers']},'Pillow':'12.0.0 task overlay','native_executed':False,'current_deployment_fault_proven':False,'candidate_accepted':False,'performance':'NOT_MEASURED'};(O/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['status','case_groups','semantic_checks','failed_checks','native_executed']}));sys.exit(1 if failed else 0)
