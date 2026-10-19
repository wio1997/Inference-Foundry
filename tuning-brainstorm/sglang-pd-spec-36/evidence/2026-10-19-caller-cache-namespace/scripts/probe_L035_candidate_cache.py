#!/usr/bin/env python3
import asyncio,json,pathlib,sys,traceback,hashlib
from types import SimpleNamespace as NS
from L035_candidate_components import R,ns,records,registry,CoreView,manager,scheduler,TimeBoundary,lora_cfg,BOUNDARIES_L035,AlgoBoundary
O=R/'runs/L035-R008-candidate-cache';O.mkdir(parents=True,exist_ok=False);checks=[];cases=[];stage='setup';error=None
G=ns['GenerateReqInput'];Req=ns['Req'];Ref=registry.LoRARef
A=Ref(lora_id=Ref.deterministic_id('A','/synthetic/A'),lora_name='A',lora_path='/synthetic/A');B=Ref(lora_id=Ref.deterministic_id('B','/synthetic/B'),lora_name='B',lora_path='/synthetic/B')
def ck(name,value,**detail):checks.append({'stage':stage,'name':name,'pass':bool(value),'detail':detail})
async def main():
 global stage
 manifest=json.loads((R/'fixtures/M001-v1/manifest.json').read_text());fixtures={x['label']:x for x in manifest['fixtures']}
 for label,num,budget in [('c2-short-0',2,128),('c3-long-0',3,256)]:
  f=fixtures[label];p=R/f['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256'];data=json.loads(p.read_text());ids=data['input_ids'];assert len(ids)==f['input_tokens'] and max(ids)<154856
  for page in [1,128]:
   for eagle in [False,True]:
    for mode in [ns['DisaggregationMode'].PREFILL,ns['DisaggregationMode'].DECODE]:
     stage=str((label,page,eagle,mode.value));m=manager([A,B]);rids=[stage+'-'+str(i) for i in range(num)];obj=G(rid=rids,input_ids=[ids[:]]*num,lora_path=[None,'A']+(['B'] if num==3 else []),extra_key=[A.lora_id,'']+([''] if num==3 else []),cache_salt=['']*num,sampling_params={'max_new_tokens':budget},bootstrap_host='cpu-fixture-prefill',bootstrap_port=8998,bootstrap_room=101);obj.normalize_batch_and_arguments();ck('batch rooms unique',len(set(obj.bootstrap_room))==num and obj.bootstrap_room[0]==101);cached=obj[1];await m._validate_and_resolve_lora(obj);ck('normalized caller key preserved',obj.extra_key[0]==A.lora_id);ck('cached subobject resolved ID propagated',obj[1] is cached and cached.lora_id==A.lora_id);ck('real registry counts',m.lora_registry._counters[A.lora_id].value()==1)
     view=CoreView(page,eagle);sched=scheduler(view,mode);reqs=[]
     for i in range(num):
      sub=obj[i];m.rid_to_state[sub.rid]=NS(time_stats=TimeBoundary());t=m._create_tokenized_object(sub,None,sub.input_ids);ck('carrier identity fields preserved',t.extra_key==sub.extra_key and t.cache_salt==sub.cache_salt and t.lora_id==sub.lora_id);sched.handle_generate_request(t);req=sched.queued[-1];ck('scheduler retained PD metadata',req.bootstrap_room==t.bootstrap_room and req.bootstrap_port==t.bootstrap_port and req.bootstrap_host==t.bootstrap_host and req.to_finish is None,carrier_room=t.bootstrap_room,scheduled_room=req.bootstrap_room);ck('output budget preserved',req.sampling_params.max_new_tokens==budget);reqs.append(req)
     values,insert_result=view.insert(reqs[0]);reqs[1].init_next_round_input(view);hit=len(reqs[1].prefix_indices);ck('CONTRACT adapted query must not hit base metadata',hit==0,hit_tokens=hit,caller_base_key=reqs[0].extra_key,adapter_key=reqs[1].extra_key);ck('actual cross-hit indices provenance',(hit==0) or reqs[1].prefix_indices.tolist()==values[:hit].tolist());reqs[0].init_next_round_input(view);ck('same base namespace still hits',len(reqs[0].prefix_indices)>0 and reqs[0].prefix_indices.tolist()==values[:len(reqs[0].prefix_indices)].tolist())
     if num==3:reqs[2].init_next_round_input(view);ck('other adapter misses',len(reqs[2].prefix_indices)==0)
     for rid in obj.lora_id:
      if rid is not None:await m.lora_registry.release(rid)
     ck('holders released',all(c.value()==0 for c in m.lora_registry._counters.values()));cases.append({'case':label,'requests':num,'input_tokens':len(ids),'output_budget':budget,'page_size':page,'is_eagle':eagle,'mode':mode.value,'cross_hit_tokens':hit,'metadata_values_origin':'base request inserted Torch CPU indices','native':False})
 stage='normalizer-controls'
 for field in ['extra_key','cache_salt']:
  q=G(input_ids=[1,2],**{field:''});q.normalize_batch_and_arguments();ck('empty '+field+' normalized',getattr(q,field) is None)
  try:G(input_ids=[1,2],**{field:123}).normalize_batch_and_arguments()
  except ValueError:ck('nonstr '+field+' rejected',True)
  else:ck('nonstr '+field+' rejected',False)
 stage='disabled-lora-control';m=manager([A]);m.enable_lora=False;q=G(input_ids=[1,2],lora_path='A');q.normalize_batch_and_arguments()
 try:await m._validate_and_resolve_lora(q)
 except ValueError:ck('disabled adapter rejected',True)
 else:ck('disabled adapter rejected',False)
 stage='target-draft-init-gate'
 for enabled,draft,want in [(True,False,1),(True,True,0),(False,False,0)]:
  lora_cfg.enable_lora=enabled;r=ns['ModelRunner']();r.spec_algorithm=AlgoBoundary();r.is_draft_worker=draft;calls=[];r.init_lora_manager=lambda:calls.append('init-boundary');r.maybe_init_lora_manager();ck('original runner gate '+str((enabled,draft)),len(calls)==want)
 lora_cfg.enable_lora=True
 stage='legacy-cache-migration'
 from L035_candidate_components import OriginalReq
 old_style_key=json.dumps(["sglang.cache-namespace.v1",None,A.lora_id],ensure_ascii=False,separators=(",",":"))
 old=OriginalReq(rid='legacy-base',origin_input_text=None,origin_input_ids=__import__('array').array('q',list(range(512))),sampling_params=ns['SamplingParams'](max_new_tokens=128),extra_key=old_style_key)
 new=Req(rid='new-adapter',origin_input_text=None,origin_input_ids=__import__('array').array('q',list(range(512))),sampling_params=ns['SamplingParams'](max_new_tokens=128),lora_id=A.lora_id)
 mixed=CoreView(128,True);values,_=mixed.insert(old,start=900000);new.init_next_round_input(mixed);legacy_hit=len(new.prefix_indices)
 ck('CONTRACT candidate must not reuse legacy base metadata',legacy_hit==0,legacy_hit_tokens=legacy_hit,legacy_key=old.extra_key,candidate_key=new.extra_key)
 ck('legacy metadata hit provenance',legacy_hit>0 and new.prefix_indices.tolist()==values[:legacy_hit].tolist())
 cases.append({'case':'legacy-cache-migration','input_tokens':512,'page_size':128,'is_eagle':True,'cross_hit_tokens':legacy_hit,'native':False,'precondition':'An old cache entry was retained; full cache/storage generation fencing absent'})
try:asyncio.run(main())
except Exception as e:error={'stage':stage,'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
failed=[x for x in checks if not x['pass']];result={'status':'INVALID' if error else 'FAIL' if failed else 'PASS','harness_error':error,'case_count':len(cases),'semantic_checks':len(checks),'failed_checks':len(failed),'checks':checks,'cases':cases,'source_components':records,'boundaries':BOUNDARIES_L035,'native_executed':False,'current_deployment_fault_proven':False,'candidate_accepted':False,'performance':'NOT_MEASURED'};(O/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['status','case_count','semantic_checks','failed_checks','harness_error']}));sys.exit(2 if error else 1 if failed else 0)
