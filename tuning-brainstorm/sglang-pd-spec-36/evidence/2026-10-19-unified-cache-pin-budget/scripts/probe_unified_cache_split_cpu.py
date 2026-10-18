#!/usr/bin/env python3
"""Canonical whole Python tree/component bodies; explicit CPU allocator boundary."""
import argparse,ast,copy,dataclasses,datetime,enum,hashlib,heapq,importlib.util,json,logging,os,pathlib,subprocess,sys,traceback,types,typing
from array import array
from collections import defaultdict
from abc import ABC,abstractmethod
ap=argparse.ArgumentParser()
for name in ['source-root','expected-hashes','model-hashes','case-key','protocol','fixture-root']:ap.add_argument('--'+name,required=True)
ap.add_argument('--stdout',action='store_true');args=ap.parse_args();root=pathlib.Path('/sgl-workspace/sglang');snapshot=pathlib.Path(args.source_root);expected=json.loads(args.expected_hashes);models=json.loads(args.model_hashes);p=json.loads(pathlib.Path(args.protocol).read_text());sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
assert os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0' and os.environ.get('SGLANG_USE_CPU_ENGINE')=='1'
nodes=sorted({str(q) for pat in ['davinci*','devmm_svm','hisi_hdc','ascend*'] for q in pathlib.Path('/dev').glob(pat)});assert not nodes
head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip();assert head==p['source_commit']
for f,h in expected.items():assert sha(root/f)==h==sha(snapshot/f);assert subprocess.check_output(['git','-C',str(root),'show',head+':'+f])==(root/f).read_bytes()
for f,h in models.items():assert sha(pathlib.Path('/model')/f)==h
result={'probe_status':'PENDING','correctness':'NOT_EVALUATED','source_commit':head,'source_hashes':expected,'model_hashes':models,'script_sha256':sha(pathlib.Path(__file__)),'protocol_sha256':sha(pathlib.Path(args.protocol)),'accelerator_device_nodes':nodes,'source_bodies':{},'actual_modules':[],'cases':[],'semantic_checks':0,'scope':'Whole unchanged default Python UnifiedTreeCore/FULL component source bodies, actual Torch CPU indices and leaf imports. Allocator and action draining are explicit metadata fixtures. No UnifiedRadixCache facade constructor, physical KV, native kernels, serving, timing or PD/MTP performance evidence.'};stage='full-module-import-once'
try:
 import torch,msgspec,numpy
 torch.set_num_threads(1);torch.set_num_interop_threads(1);result['environment']={'python':sys.version,'torch':torch.__version__}
 result['full_facade_import']={'status':'NOT_RETRIED','prior_run':'L021-R007','prior_status':'UNAVAILABLE','reason':'sgl_kernel absent in isolated CPU dependency view'}
 stage='source-body-loading';module=types.ModuleType('if_unified_source_bodies');sys.modules[module.__name__]=module;env=module.__dict__;env.update(vars(typing));env['__name__']='if_unified_source_bodies';env.update(torch=torch,msgspec=msgspec,float64=numpy.float64,dataclasses=dataclasses,dataclass=dataclasses.dataclass,ABC=ABC,abstractmethod=abstractmethod,Enum=enum.Enum,IntFlag=enum.IntFlag,auto=enum.auto,array=array,defaultdict=defaultdict,heapq=heapq,logging=logging,sys=sys,NodeId=int,logger=logging.getLogger('if_unified_probe'),envs=types.SimpleNamespace())
 # Disabled placement-event callbacks still receive a storage-medium tag.
 env['StorageMedium']=enum.Enum('StorageMedium',{'GPU':'GPU','CPU':'CPU'})
 def tree(f):return ast.parse((root/f).read_text())
 def execute(f,names=None,assigns=False):
  body=[]
  for n in tree(f).body:
   if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and (names is None or n.name in names):body.append(n)
   elif assigns and isinstance(n,(ast.Assign,ast.AnnAssign)):body.append(n)
  assert body
  for n in body:
   label=getattr(n,'name',ast.unparse(n).split('=')[0].strip());result['source_bodies'][f+':'+label]={'ast_sha256':hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest(),'unchanged':True}
  exec(compile(ast.Module(body=[ast.parse('from __future__ import annotations').body[0],*copy.deepcopy(body)],type_ignores=[]),f,'exec'),env)
 def leaf(f):
  name='if_leaf_'+pathlib.Path(f).stem;spec=importlib.util.spec_from_file_location(name,root/f);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);result['actual_modules'].append({'path':f,'sha256':sha(root/f),'full_file':True});return m
 b='python/sglang/srt/mem_cache/';ct=leaf(b+'unified_cache/component_type.py');env.update(ComponentType=ct.ComponentType,BASE_COMPONENT_TYPE=ct.BASE_COMPONENT_TYPE)
 init=leaf(b+'cache_init_params.py');policy=leaf(b+'evict_policy.py');env.update({k:v for k,v in vars(policy).items() if k.endswith('Strategy')})
 execute(b+'utils.py',['get_eviction_strategy','split_node_hash_value'])
 # The factory mapping below is copied from the canonical AST, not reimplemented.
 mapping=next(n for n in tree(b+'utils.py').body if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name) and n.target.id=='_EVICTION_POLICY_FACTORIES');exec(compile(ast.Module(body=[ast.parse('from __future__ import annotations').body[0],mapping],type_ignores=[]),'eviction-factory-map','exec'),env)
 execute(b+'base_prefix_cache.py',['CacheRequestHandle','MatchPrefixParams','InsertParams','InsertResult','EvictParams','EvictResult','IncLockRefResult','DecLockRefParams','DecLockRefResult','MatchResult','_dfs_weight_order'])
 execute(b+'radix_cache.py',['RadixKey']);execute(b+'events.py',['KVCacheEventRecorder'])
 execute(b+'unified_cache/cache_action.py',assigns=True);execute(b+'unified_cache/unified_tree_core_interface.py',assigns=True)
 execute(b+'unified_cache/components/base.py',assigns=True);execute(b+'unified_cache/components/full.py');execute(b+'unified_cache/unified_tree_core.py',assigns=True);execute(b+'prefill_budget.py',['PrefillBudget'])
 CT=ct.ComponentType;Core=env['UnifiedTreeCore'];Full=env['FullComponent'];Key=env['RadixKey'];IP=env['InsertParams'];MP=env['MatchPrefixParams'];Budget=env['PrefillBudget']
 def check(x,message):
  assert x,message
  result['semantic_checks']+=1
 class AllocatorFixture:
  device=torch.device('cpu')
  def __init__(self,page):self.page_size=page;self.free=262144;self.next=0;self.live=set();self.frees=[]
  def available_size(self):return self.free
  def allocate(self,n):
   out=torch.arange(self.next,self.next+n,dtype=torch.int64);self.next+=n;self.free-=n;self.live.update(out.tolist());return out
  def release(self,value):
   ids=value.tolist();assert len(ids)==len(set(ids)) and set(ids)<=self.live,'double/unowned fixture free';self.live.difference_update(ids);self.free+=len(ids);self.frees.append(len(ids))
 def setup(page,eagle):
  a=AllocatorFixture(page);params=init.CacheInitParams(False,None,a,page,is_eagle=eagle,tree_components=(CT.FULL,));cache=types.SimpleNamespace(enable_session_radix_cache=False,token_to_kv_pool_allocator=a);comp=Full(cache,params);core=Core(params,{CT.FULL:comp});cache.tree_core=core
  # Budget sees the actual tree's counters; non-Mamba classification is a boundary adapter.
  budget_view=types.SimpleNamespace(evictable_size=core.evictable_size,full_evictable_size=core.full_evictable_size,supports_mamba=lambda:False)
  return core,a,Budget(a,budget_view)
 def drain_actions(actions,a):
  for action in actions:
   assert isinstance(action,(env['FreeDeviceKV'],env['FreeDeviceKVFullOnly'])),'unexpected action '+type(action).__name__
   for value in action.indices:a.release(value)
 def insert(core,a,tokens,**kw):
  values=a.allocate(len(tokens));step=core.begin_insert(IP(key=Key(array('q',tokens),**kw),value=values));drain_actions(step.actions,a)
  while step.result is None:step=core.resume_insert();drain_actions(step.actions,a)
  drain_actions(core.end_insert(),a);return step.result,values
 def match(core,tokens,**kw):return core.match_prefix(MP(key=Key(array('q',tokens),**kw)))
 def drain_eviction(ret,a):
  counts=dict(ret.tracker)
  for values in ret.device_frees.values():
   for value in values:a.release(value)
  assert not ret.host_frees;ret.device_frees.clear();ret.host_frees.clear();return counts
 def evict(core,a,n):
  tracker=defaultdict(int);core.evict_device_start(CT.FULL,n)
  try:
   while True:
    step=core.evict_device_next_node(CT.FULL,tracker);nid=step.node_id;progress=step.made_progress
    for k,v in drain_eviction(step,a).items():tracker[k]+=v
    if nid is not None:
     leaf=core.evict_device_leaf(nid,False)
     for k,v in drain_eviction(leaf,a).items():tracker[k]+=v
    elif not progress:break
  finally:core.evict_device_end(CT.FULL)
  return tracker[CT.FULL]
 stage='pin-before-split'
 for length in p['lengths']:
  for trigger in ['match','insert']:
   case={'page_size':128,'is_eagle':True,'raw_length':length,'split_trigger':trigger};result['in_progress']=case;core,a,budget=setup(128,True);cacheable=((length-1)//128)*128;shared=128;ts=list(range(length));other=ts[:129]+list(range(100000,100000+length-129));r,original=insert(core,a,ts);nid=r.last_device_node;rec=core.inc_lock_ref(nid);check(rec.delta==cacheable and core.protected_size()==cacheable,'initial pin before split');base=budget.remaining_current
   if trigger=='match':
    partial=match(core,ts[:129]);check(len(partial.device_indices)==128 and core.protected_size()==cacheable,'partial match splits locked leaf');check(budget.remaining_current==base,'match-only split preserves budget')
   r2,original2=insert(core,a,other);parent=next(iter(core.root_node.children.values()));leaf=core.node_by_id(nid);check(leaf.parent is parent and len(parent.key)==shared,'existing node handle survives split');check(parent.component_data[CT.FULL].lock_ref==leaf.component_data[CT.FULL].lock_ref==1,'split copies existing lock coverage');check(core.protected_size()==cacheable and core.evictable_size()==cacheable-shared,'new branch outside existing receipt');check(budget.remaining_current==base-(length-cacheable),'new branch leaves only uncached tail capacity charge');check(sum(a.frees)==shared,'one duplicate prefix action drained')
   before=match(core,ts).device_indices.clone();returned=match(core,ts).device_indices;returned.fill_(-1);check(torch.equal(match(core,ts).device_indices,before),'match metadata output is independent copy');check(parent.component_data[CT.FULL].value.data_ptr()!=original.data_ptr() and leaf.component_data[CT.FULL].value.data_ptr()!=original.data_ptr(),'both split metadata tensors cloned')
   snapshot=(core.evictable_size(),core.protected_size());skip=core.inc_lock_ref(nid,skip_lock_components=(CT.FULL,));check(snapshot==(core.evictable_size(),core.protected_size()),'skip FULL receipt leaves ledger');core.dec_lock_ref(nid,skip.to_dec_params());check(snapshot==(core.evictable_size(),core.protected_size()),'skipped release preserves real holder')
   rec2=core.inc_lock_ref(r2.last_device_node);check(rec2.delta==cacheable-shared and core.evictable_size()==0,'second branch pins only new tail');core.dec_lock_ref(nid,rec.to_dec_params());check(core.protected_size()==cacheable and parent.component_data[CT.FULL].lock_ref==1,'old receipt releases split path without stealing shared holder');avail=budget.remaining_current;check(evict(core,a,1)==cacheable-shared,'released tail evicts while shared pin remains');check(budget.remaining_current==avail,'drained action keeps live total capacity');core.dec_lock_ref(r2.last_device_node,rec2.to_dec_params());check(core.protected_size()==0,'last receipt releases inherited ancestor');remaining=core.evictable_size();check(evict(core,a,262144)==remaining,'final eviction drains shared ancestor');check(len(a.live)==2*(length-cacheable),'caller retains only uncached tails')
   result['cases'].append({**case,'status':'PASS','cacheable_per_request':cacheable,'protected_after_split':cacheable,'evictable_after_split':cacheable-shared,'metadata_copy_only':True});result.pop('in_progress')
 result.update(probe_status='PASS',correctness='PASS',status_scope='Source-class CPU contract only; full facade import status independent; native acceptance remains E0')
except Exception as exc:
 result.update(probe_status='UNAVAILABLE' if stage in ['full-module-import-once','source-body-loading'] else 'FAIL',correctness='NOT_EVALUATED' if stage in ['full-module-import-once','source-body-loading'] else 'FAIL',failed_stage=stage,error=repr(exc),traceback=traceback.format_exc())
print(json.dumps(result,ensure_ascii=False))
sys.exit(0 if result['probe_status'] in ['PASS','UNAVAILABLE'] else 1)
