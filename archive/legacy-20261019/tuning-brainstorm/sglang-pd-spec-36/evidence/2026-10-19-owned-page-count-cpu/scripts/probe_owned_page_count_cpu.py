#!/usr/bin/env python3
"""Producer-level CPU differential with real allocator and explicit boundary fixtures."""
import argparse,ast,builtins,contextlib,copy,datetime,hashlib,importlib,importlib.util,inspect,json,logging,os,pathlib,subprocess,sys,traceback,types
from collections import defaultdict
ap=argparse.ArgumentParser()
for name in ['source-root','expected-hashes','model-hashes','case-key','protocol','fixture-root']:ap.add_argument('--'+name,required=True)
ap.add_argument('--stdout',action='store_true');args=ap.parse_args();root=pathlib.Path('/sgl-workspace/sglang');snapshot=pathlib.Path(args.source_root);expected=json.loads(args.expected_hashes);models=json.loads(args.model_hashes);p=json.loads(pathlib.Path(args.protocol).read_text());sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest();C=snapshot/p['candidate_root']
assert os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0' and os.environ.get('SGLANG_USE_CPU_ENGINE')=='1';nodes=sorted({str(q) for pat in ['davinci*','devmm_svm','hisi_hdc','ascend*'] for q in pathlib.Path('/dev').glob(pat)});assert not nodes
head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip();assert head==p['source_commit']
for f,h in expected.items():assert sha(root/f)==h==sha(snapshot/f);assert subprocess.check_output(['git','-C',str(root),'show',head+':'+f])==(root/f).read_bytes()
for f,h in models.items():assert sha(pathlib.Path('/model')/f)==h
for f,h in p['candidate_hashes'].items():assert sha(C/f)==h
result={'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'probe_status':'PENDING','correctness':'NOT_EVALUATED','source_commit':head,'source_hashes':expected,'model_hashes':models,'candidate_hashes':p['candidate_hashes'],'script_sha256':sha(pathlib.Path(__file__)),'protocol_sha256':sha(pathlib.Path(args.protocol)),'accelerator_device_nodes':nodes,'source_bodies':{},'cases':[],'gate_cases':[],'semantic_checks':0,'native_executed':False,'candidate_accepted':False,'production_applied':False,'performance':'NOT_MEASURED','scope':'Full actual baseline/candidate NPU leaf modules; complete selected owner/bridge source bodies on CPU. Req-slot allocation, eviction, req-table write/aux and DSV4 hook are explicit fixtures. Native import raises before import/launch. No fake native output or complete scheduler/PD/stream/performance proof.'};stage='imports'
try:
 import torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1);canonical='sglang.srt.hardware_backend.npu.allocator_npu';old=importlib.import_module(canonical);npu_path=p['paths']['npu'];spec=importlib.util.spec_from_file_location('if_owned_candidate_npu',C/npu_path);new=importlib.util.module_from_spec(spec);sys.modules[spec.name]=new;spec.loader.exec_module(new);sizing=importlib.import_module('sglang.srt.mem_cache.allocation_sizing');page_mod=importlib.import_module('sglang.srt.mem_cache.allocator.page_interleave');common=importlib.import_module('sglang.srt.utils.common');parent=importlib.import_module('sglang.srt.hardware_backend.npu');result['environment']={'python':sys.version,'torch':torch.__version__};result['full_NPU_modules_imported']=True
 def cls_ast(path):return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='NPUPagedTokenToKVPoolAllocator')
 assert ast.dump(cls_ast(root/npu_path),include_attributes=False)==ast.dump(cls_ast(C/npu_path),include_attributes=False);result['allocator_class_AST_unchanged']=True
 class NativeBoundary(RuntimeError):pass
 class DeviceLens(torch.Tensor):
  reads=0
  @staticmethod
  def wrap(v):return torch.Tensor._make_subclass(DeviceLens,v,False)
  def item(self,*a,**kw):DeviceLens.reads+=1;return torch.Tensor.item(self,*a,**kw)
 @contextlib.contextmanager
 def bindings(mod,events):
  saved=sys.modules[canonical];attr=parent.allocator_npu;original=builtins.__import__;sys.modules[canonical]=mod;parent.allocator_npu=mod
  def guard(name,globals=None,locals=None,fromlist=(),level=0):
   if name=='sgl_kernel_npu.mem_cache.allocator':events.append(name);raise NativeBoundary('Before native import/launch; not success')
   return original(name,globals,locals,fromlist,level)
  builtins.__import__=guard
  try:yield
  finally:builtins.__import__=original;sys.modules[canonical]=saved;parent.allocator_npu=attr
 def check(v,msg):
  assert v,msg
  result['semantic_checks']+=1
 def digest(t):
  if t is None:return None
  if isinstance(t,DeviceLens):t=t.as_subclass(torch.Tensor)
  return {'dtype':str(t.dtype),'shape':list(t.shape),'sha256':hashlib.sha256(t.detach().contiguous().numpy().tobytes()).hexdigest()}
 class PoolFixture:
  def __init__(self,bs,width):self.req_to_token=torch.zeros((bs+1,width),dtype=torch.int32);self.aux_calls=0
  def alloc_aux_to_lengths(self,**kwargs):self.aux_calls+=1
 def table_assign(indices,table,start,end,out,bs):
  offset=0
  for row,lo,hi in zip(indices.tolist(),start.tolist(),end.tolist()):n=hi-lo;table[row,lo:hi]=out[offset:offset+n].to(torch.int32);offset+=n
  assert offset==out.numel()
 def table_write(out,idx,idx_cpu,prefix_dev,prefix_cpu,seq_dev,seq_cpu,extend_dev,extend_cpu,prefix_values,pool):
  for row,v in zip(idx_cpu.tolist(),prefix_values):pool.req_to_token[row,:len(v)]=v.to(torch.int32)
  table_assign(idx_cpu,pool.req_to_token,prefix_cpu,seq_cpu,out,len(idx_cpu))
 def env_for(which,reserve,platform=True):
  module=types.ModuleType('if_owner_'+which+'_'+str(len(result['cases'])));sys.modules[module.__name__]=module;env=module.__dict__;env.update(torch=torch,defaultdict=defaultdict,logger=logging.getLogger('if_owner'),_is_npu=platform,_is_hip=False,_is_cuda=False,attention_backends=lambda:('ascend','ascend'),page_interleave_shard_size=page_mod.page_interleave_shard_size,is_pin_memory_available=lambda device:False,is_deepseek_v4=lambda cfg:False,get_spec=lambda:types.SimpleNamespace(speculative_eagle_topk=1),get_alloc_reserve_per_decode=lambda:reserve,page_aligned_decode_alloc_lens=sizing.page_aligned_decode_alloc_lens,available_and_evictable_str=lambda c:'fixture',evict_from_tree_cache=lambda c,n:None,write_cache_indices=table_write,assign_req_to_token_pool_func=table_assign,maybe_write_dsv4_extend=lambda *a:None)
  def alloc_rows(pool,reqs,cache):
   for i,r in enumerate(reqs,1):r.kv.req_pool_idx=i
   return list(range(1,len(reqs)+1))
  env['alloc_req_slots']=alloc_rows;future=ast.parse('from __future__ import annotations').body[0]
  def load(path,names,changed=True):
   source=(C/path if which=='candidate' and path in p['candidate_hashes'] and changed else root/path).read_text();t=ast.parse(source);sel=[next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name) for name in names]
   for n in sel:result['source_bodies'][which+':'+path+':'+n.name]=hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest()
   exec(compile(ast.Module(body=[future,*copy.deepcopy(sel)],type_ignores=[]),path,'exec'),env);return t
  t=load(p['paths']['allocation'],['get_last_loc_torch','get_last_loc','_alloc_page_size','alloc_token_slots','alloc_paged_token_slots_extend','alloc_for_extend','alloc_for_spec_decode']);load(p['paths']['PD'],['alloc_for_decode_prealloc']);load(p['paths']['eagle'],['eagle_prepare_for_decode']);load('python/sglang/srt/hardware_backend/npu/dsv4/dsv4_allocator.py',['alloc_paged_token_slots_extend_npu'],False);env['_alloc_paged_token_slots_extend_npu']=env['alloc_paged_token_slots_extend_npu'];mapping=next(n for n in t.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='ALLOC_EXTEND_FUNCS' for x in n.targets));exec(compile(ast.Module(body=[mapping],type_ignores=[]),'dispatch','exec'),env);return env
 def run(which,case):
  mod=new if which=='candidate' else old;ps=case['page'];bs=case.get('batch',2);kind=case['family'];prefix=case.get('prefix',[0]*bs);ends=case.get('seq',[512]*bs);reserve=case.get('reserve',8);reqs=[types.SimpleNamespace(prefix_indices=None,kv=types.SimpleNamespace(kv_allocated_len=x,kv_committed_len=x,req_pool_idx=None),decode_batch_idx=0) for x in prefix]
  if kind in ['EAGLE','EAGLE-noalloc']:prefix=[ps if kind=='EAGLE' else 2*ps]*bs;ends=[max(x,((ps-1+reserve+ps-1)//ps)*ps) for x in prefix];reqs=[types.SimpleNamespace(prefix_indices=None,kv=types.SimpleNamespace(kv_allocated_len=x,kv_committed_len=ps-1,req_pool_idx=i+1),decode_batch_idx=0) for i,x in enumerate(prefix)]
  need=sum((e+ps-1)//ps-(s+ps-1)//ps for s,e in zip(prefix,ends));prefix_pages=sum((x+ps-1)//ps for x in prefix);pages=prefix_pages if case.get('exhaust') else prefix_pages+max(need,0)+32;a=mod.NPUPagedTokenToKVPoolAllocator(pages*ps,ps,torch.float32,'cpu',None,True);a.debug_mode=False;width=max(max(ends),max(prefix))+ps+1;pool=PoolFixture(bs,width);values=[]
  for i,(r,n) in enumerate(zip(reqs,prefix),1):v=a.alloc(((n+ps-1)//ps)*ps)[:n];values.append(v);r.prefix_indices=v;pool.req_to_token[i,:n]=v.to(torch.int32)
  captured=[];original=a.alloc_extend;sig=inspect.signature(original)
  def observed(*args,**kwargs):
   bound=sig.bind(*args,**kwargs);bound.apply_defaults();v=bound.arguments;oracle=sum((int(e)+ps-1)//ps-(int(s)+ps-1)//ps for s,e in zip(v['prefix_lens_cpu'].tolist(),v['seq_lens_cpu'].tolist())) if v['prefix_lens_cpu'] is not None and v['seq_lens_cpu'] is not None else need;captured.append({'count':v['num_new_pages'],'kwarg_present':'num_new_pages' in kwargs,'oracle':oracle,'prefix_dtype':str(v['prefix_lens'].dtype)});v['prefix_lens']=DeviceLens.wrap(v['prefix_lens']);v['seq_lens']=DeviceLens.wrap(v['seq_lens']);return original(**v)
  a.alloc_extend=observed
  class StrictForward:
   def __getattr__(self,name):return getattr(a,name)
   def alloc_extend(self,prefix_lens,prefix_lens_cpu,seq_lens,seq_lens_cpu,last_loc,extend_num_tokens):return a.alloc_extend(prefix_lens,prefix_lens_cpu,seq_lens,seq_lens_cpu,last_loc,extend_num_tokens)
  target=StrictForward() if case.get('strict') else a;cache=types.SimpleNamespace(token_to_kv_pool_allocator=target,page_size=ps,pretty_print=lambda:None);batch=types.SimpleNamespace(reqs=reqs,device=torch.device('cpu'),tree_cache=cache,token_to_kv_pool_allocator=target,req_to_token_pool=pool,req_pool_indices=torch.arange(1,bs+1,dtype=torch.int64),sampling_info=types.SimpleNamespace(penalizer_orchestrator=types.SimpleNamespace(is_required=False)),batch_size=lambda:bs,maybe_evict_swa=lambda:None,is_dllm=lambda:False,model_config=types.SimpleNamespace(hf_config=types.SimpleNamespace()),prefix_lens=prefix,extend_lens=[e-s for s,e in zip(prefix,ends)],extend_num_tokens=sum(e-s for s,e in zip(prefix,ends)),seq_lens=torch.tensor(ends,dtype=torch.int64),seq_lens_cpu=torch.tensor(ends,dtype=torch.int64));env=env_for(which,reserve,case.get('platform',True));before=a.get_all_free_pages().clone();DeviceLens.reads=0;events=[];out=None;outcome='RETURN'
  with bindings(mod,events):
   try:
    if kind=='ordinary':out=env['alloc_for_extend'](batch)[0]
    elif kind in ['EAGLE','EAGLE-noalloc']:env['eagle_prepare_for_decode'](batch)
    elif kind=='PD':out=env['alloc_for_decode_prealloc'](target,req=reqs[0],fill_len=ends[0],delta_len=ends[0]-prefix[0],prefix_len=prefix[0],total_prefix_len=prefix[0],prefix_indices=values[0],uses_swa_tail=False,swa_tail_len=0)
    elif kind in ['legacy-helper','wrapper-opt-in']:
     pt=torch.tensor(prefix,dtype=torch.int64);st=torch.tensor(ends,dtype=torch.int64);last=torch.tensor([int(v[-1]) if len(v) else -1 for v in values]);fn=env['alloc_paged_token_slots_extend'] if kind=='legacy-helper' else env['alloc_paged_token_slots_extend_npu'];kw={'batch':batch}
     if kind=='wrapper-opt-in' and which=='candidate':kw['num_new_pages']=need
     out=fn(cache,pt,pt.clone(),st,st.clone(),last,batch.extend_num_tokens,**kw)
    else:raise AssertionError(kind)
   except NativeBoundary:outcome='NATIVE_IMPORT_BOUNDARY'
   except RuntimeError as exc:
    if case.get('exhaust') and 'Prefill out of memory' in str(exc):outcome='CAPACITY_REJECT'
    else:raise
  return {'outcome':outcome,'output':digest(out),'pages':digest(a.get_all_free_pages()),'initial':digest(before),'available':a.available_size(),'table':digest(pool.req_to_token),'aux_calls':pool.aux_calls,'reqs':[(r.kv.kv_allocated_len,r.kv.kv_committed_len,r.kv.req_pool_idx,r.decode_batch_idx) for r in reqs],'captured':captured,'reads':DeviceLens.reads,'native_imports':len(events),'need':need}
 def compare(case):
  result['in_progress']=case;l=run('baseline',case);r=run('candidate',case)
  for key in ['outcome','output','pages','initial','available','table','aux_calls','reqs','native_imports']:check(l[key]==r[key],'differential '+key)
  check(len(l['captured'])==len(r['captured']),'same allocator call count');eligible=case['family'] in ['ordinary','PD','EAGLE','wrapper-opt-in'] and case['page']>1 and not case.get('strict') and case.get('platform',True)
  for left,right in zip(l['captured'],r['captured']):
   check(left['count'] is None and not left['kwarg_present'],'baseline default remains absent');check(right['count']==right['oracle'] if eligible else right['count'] is None,'candidate count matches owner oracle or remains absent');check(right['kwarg_present']==eligible,'count keyword only for explicit eligible route')
  check(r['reads']==(0 if eligible else l['reads']),'read behavior matches eligibility');result['cases'].append({**case,'status':'PASS','outcome':l['outcome'],'new_pages':l['need'],'baseline_reads':l['reads'],'candidate_reads':r['reads'],'allocator_calls':len(r['captured']),'candidate_counts':[x['count'] for x in r['captured']],'eligible_explicit_count':eligible,'native_success_proven':False});result.pop('in_progress')
 stage='producer-differential'
 for case in p['cases']:compare(case)
 stage='direct-allocator-legacy-stale'
 for ps in p['pages']:
  states=[]
  for mod in [old,new]:
   a=mod.NPUPagedTokenToKVPoolAllocator(512*ps,ps,torch.float32,'cpu',None,True);a.debug_mode=False;pt=torch.zeros(2,dtype=torch.int64);st=torch.tensor([100*ps,100*ps]);cpu_stale=torch.tensor([100*ps,101*ps]);DeviceLens.reads=0;out=a.alloc_extend(DeviceLens.wrap(pt),pt,DeviceLens.wrap(st),cpu_stale,torch.tensor([-1,-1]),200*ps);states.append((digest(out),digest(a.get_all_free_pages()),DeviceLens.reads))
  check(states[0]==states[1] and states[1][2]==1,'direct stale/default API unchanged');result['cases'].append({'family':'direct-allocator-stale-regression','page':ps,'status':'PASS','native_success_proven':False})
 stage='private-helper-gates'
 for ps,variant in [(1,'page1'),(128,'subclass'),(128,'c128'),(128,'unsupported')]:
  if variant=='unsupported':a=object()
  else:
   cls=type('AllocatorSubclass',(new.NPUPagedTokenToKVPoolAllocator,),{}) if variant=='subclass' else new.NPUPagedTokenToKVPoolAllocator;a=cls(ps*4,ps,torch.float32,'cpu',None,True)
   if variant=='c128':a.c128_attn_allocator=object()
  check(new._num_new_pages_from_owned_cpu(a,torch.tensor([0]),torch.tensor([ps])) is None,'private helper opt-in gate');result['gate_cases'].append({'fixture':variant,'status':'PASS','real_backend_support_claim':False})
 check(sys.modules[canonical] is old and parent.allocator_npu is old,'canonical module binding restored');result.update(probe_status='PASS',correctness='PASS',whole_caller_modules_imported=False,native_executed=False,candidate_accepted=False)
except Exception as exc:
 result.update(probe_status='UNAVAILABLE' if stage=='imports' else 'FAIL',correctness='NOT_EVALUATED' if stage=='imports' else 'FAIL',failed_stage=stage,error=repr(exc),traceback=traceback.format_exc())
print(json.dumps(result,ensure_ascii=False));sys.exit(0 if result['probe_status'] in ['PASS','UNAVAILABLE'] else 1)
