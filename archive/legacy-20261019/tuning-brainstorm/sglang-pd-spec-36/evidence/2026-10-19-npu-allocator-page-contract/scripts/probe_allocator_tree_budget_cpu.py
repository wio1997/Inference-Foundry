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
result={'probe_status':'PENDING','correctness':'NOT_EVALUATED','source_commit':head,'source_hashes':expected,'model_hashes':models,'script_sha256':sha(pathlib.Path(__file__)),'protocol_sha256':sha(pathlib.Path(args.protocol)),'accelerator_device_nodes':nodes,'source_bodies':{},'actual_modules':[],'cases':[],'semantic_checks':0,'scope':'Actual full NPU allocator CPU metadata plus unchanged tree/FULL/action/factory source bodies. Partial facade field adapter, fixture factory context and kvcache=None. No full facade constructor, KV payload, native kernels, serving or performance.'};stage='full-module-import-once'
try:
 import torch,msgspec,numpy
 torch.set_num_threads(1);torch.set_num_interop_threads(1);result['environment']={'python':sys.version,'torch':torch.__version__}
 npu=importlib.import_module('sglang.srt.hardware_backend.npu.allocator_npu');RealAlloc=npu.NPUPagedTokenToKVPoolAllocator;os.environ['SGLANG_DEBUG_MEMORY_POOL']='true';budget_module=None
 result['full_facade_import']={'status':'NOT_RETRIED','prior_run':'L021-R007','prior_status':'UNAVAILABLE missing sgl_kernel'}
 try:
  budget_module=importlib.import_module('sglang.srt.mem_cache.prefill_budget');result['full_budget_import']={'status':'PASS'}
 except Exception as exc:result['full_budget_import']={'status':'UNAVAILABLE','error':repr(exc),'traceback':traceback.format_exc(),'attempt_count':1}
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
 result['actual_modules'].append({'path':'python/sglang/srt/hardware_backend/npu/allocator_npu.py','sha256':sha(pathlib.Path(npu.__file__)),'full_file':True});assert sha(pathlib.Path(npu.__file__))==expected['python/sglang/srt/hardware_backend/npu/allocator_npu.py']
 CT=ct.ComponentType;Core=env['UnifiedTreeCore'];Full=env['FullComponent'];Key=env['RadixKey'];IP=env['InsertParams'];MP=env['MatchPrefixParams'];Budget=env['PrefillBudget']
 def check(x,message):
  assert x,message
  result['semantic_checks']+=1
 # Source facade methods execute on a small field adapter, without facade construction.
 ft=tree(b+'unified_radix_cache.py');fc=next(n for n in ft.body if isinstance(n,ast.ClassDef) and n.name=='UnifiedRadixCache');method_names=['_apply_cache_actions','_apply_cache_action','_drain_device_frees','supports_mamba','evictable_size','full_evictable_size'];methods=[next(n for n in fc.body if isinstance(n,ast.FunctionDef) and n.name==name) for name in method_names]
 for node in methods:
  result['source_bodies']['facade-method:'+node.name]={'ast_sha256':hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest(),'unchanged':True}
  exec(compile(ast.Module(body=[future,node],type_ignores=[]),node.name,'exec'),env)
 Adapter=type('CacheFieldAdapter',(),{name:env[name] for name in method_names})
 def check(x,message):
  assert x,message
  result['semantic_checks']+=1
 def setup(page):
  alloc=RealAlloc(2048*page,page,torch.float32,'cpu',None,True);params=init.CacheInitParams(False,None,alloc,page,is_eagle=True,tree_components=(CT.FULL,));cache=Adapter();cache.enable_session_radix_cache=False;cache.is_mamba_enabled=False;cache.is_swa_enabled=False;cache.token_to_kv_pool_allocator=alloc;cache.linker=None;comp=Full(cache,params);cache.components={CT.FULL:comp};core=Core(params,cache.components);cache.tree_core=core
  budget=alloc.create_prefill_budget(cache) if budget_module is not None else Budget(alloc,cache)
  return core,alloc,cache,budget
 def insert(core,alloc,cache,tokens):
  values=alloc.alloc(len(tokens));assert values is not None;step=core.begin_insert(IP(key=Key(array('q',tokens)),value=values));cache._apply_cache_actions(step.actions);assert not step.actions
  while step.result is None:step=core.resume_insert();cache._apply_cache_actions(step.actions);assert not step.actions
  cache._apply_cache_actions(core.end_insert());return step.result,values
 def match(core,tokens):return core.match_prefix(MP(key=Key(array('q',tokens))))
 def drain(ret,cache):
  counts=dict(ret.tracker);cache._drain_device_frees(ret.device_frees);assert not ret.device_frees and not ret.host_frees;return counts
 def evict(core,cache,n):
  tracker=defaultdict(int);core.evict_device_start(CT.FULL,n)
  try:
   while True:
    step=core.evict_device_next_node(CT.FULL,tracker);nid=step.node_id;progress=step.made_progress
    for k,v in drain(step,cache).items():tracker[k]+=v
    if nid is not None:
     leaf=core.evict_device_leaf(nid,False)
     for k,v in drain(leaf,cache).items():tracker[k]+=v
    elif not progress:break
  finally:core.evict_device_end(CT.FULL)
  return tracker[CT.FULL]
 stage='actual-allocator-source-tree-budget'
 for scenario in p['scenarios']:
  for bs in p['batch_sizes']:
   page,length,shared=scenario['page'],scenario['length'],scenario['shared'];core,alloc,cache,budget=setup(page);C=((length-1)//page)*page;common=shared+1;tokens=[list(range(common))+list(range(100000+i*100000,100000+i*100000+length-common)) for i in range(bs)];case={'family':'real-allocator-tree-budget','page_size':page,'raw_length':length,'shared_logical_tokens':shared,'batch':bs};result['in_progress']=case;rows=[];values=[];receipts=[];initial=alloc.available_size()
   r,v=insert(core,alloc,cache,tokens[0]);rows.append(r.last_device_node);values.append(v);check(budget.remaining_current==initial-(length-C),'cold insert leaves only caller tail charged');receipts.append(core.inc_lock_ref(rows[0]));check(receipts[0].delta==C and budget.remaining_current==initial-length,'pin before split charges real live capacity')
   for ts in tokens[1:]:r,v=insert(core,alloc,cache,ts);rows.append(r.last_device_node);values.append(v)
   check(core.protected_size()==C and core.evictable_size()==(bs-1)*(C-shared),'new branches do not inherit unrelated pin');check(alloc.available_size()==initial-bs*length+(bs-1)*shared,'duplicate actions actually return page capacity');check(budget.remaining_current==initial-length-(bs-1)*(length-C),'budget sees actual allocator and tree getters')
   for ts in tokens:check(len(match(core,ts).device_indices)==C,'actual allocated indices match logical cached floor')
   for nid in rows[1:]:
    rec=core.inc_lock_ref(nid);receipts.append(rec);check(rec.delta==C-shared,'new holder pins only new tail')
   check(core.evictable_size()==0 and budget.remaining_current==alloc.available_size(),'all cached pages protected');budget.reserve(129,1);check(budget.remaining_current-budget.remaining_total==1,'future and current reservations remain separate');check(evict(core,cache,1)==0,'no live pinned page returned to allocator');core.dec_lock_ref(rows[0],receipts[0].to_dec_params());check(core.evictable_size()==C-shared,'first receipt releases only private tail');before=(alloc.available_size(),budget.remaining_current);freed=evict(core,cache,1);check(freed==C-shared and alloc.available_size()==before[0]+freed,'core frees traverse actual facade and FULL action bodies');check(budget.remaining_current==before[1],'evictable-to-free budget conservation with real allocator');check(len(match(core,tokens[0]).device_indices)==shared,'remaining hit is the shared protected prefix')
   for nid,rec in zip(rows[1:],receipts[1:]):core.dec_lock_ref(nid,rec.to_dec_params())
   remaining=core.evictable_size();check(core.protected_size()==0 and evict(core,cache,initial)==remaining,'final holders release all tree pages');check(alloc.available_size()==initial-bs*(length-C),'only caller-owned tail pages remain allocated')
   for v in values:alloc.free_segment(v[C:],start_pos=C)
   check(alloc.available_size()==initial and torch.unique(alloc.get_all_free_pages()).numel()==2048,'caller cleanup restores unique original page pool');result['cases'].append({**case,'status':'PASS','cacheable_per_request':C,'first_leaf_freed':freed,'uncached_tail_slots_before_cleanup':bs*(length-C),'budget_binding':'actual allocator.create_prefill_budget' if budget_module else 'source PrefillBudget body; full import unavailable'});result.pop('in_progress')
 stage='source-factory-actual-npu-construction'
 factory_tree=tree('python/sglang/srt/mem_cache/kv_cache_configurator.py');f=[n for n in ast.walk(factory_tree) if isinstance(n,ast.FunctionDef) and n.name=='_build_token_to_kv_pool_allocator'];assert len(f)==1;f=f[0];result['source_bodies']['factory:'+f.name]={'ast_sha256':hashlib.sha256(ast.dump(f,include_attributes=False).encode()).hexdigest(),'unchanged':True};exec(compile(ast.Module(body=[future,f],type_ignores=[]),'allocator-factory','exec'),env);factory=env[f.name]
 env.update(_is_npu=True,current_platform=types.SimpleNamespace(is_out_of_tree=lambda:False),get_exec=lambda:types.SimpleNamespace(kernel=types.SimpleNamespace(attention_backend='ascend')),get_schedule=lambda:types.SimpleNamespace(page_size=128),get_memory=lambda:types.SimpleNamespace(enable_hisparse=False))
 for cp in [1,2,4]:
  for mode in ['prefill','decode','null']:
   env['get_parallel']=lambda:types.SimpleNamespace(attn_dcp_size=cp);env['get_disagg']=lambda:types.SimpleNamespace(disaggregation_mode=mode);self=types.SimpleNamespace(kv_cache_dtype=torch.float32,device='cpu',is_hybrid_swa=False,hybrid_gdn_config=None,is_draft_worker=False);sizes=types.SimpleNamespace(max_total_num_tokens=1024);a=factory(self,sizes=sizes,token_to_kv_pool=None,is_dsv4_model=False,req_to_token_pool=types.SimpleNamespace(),token_to_kv_pool_allocator=None);check(type(a) is RealAlloc,'in-tree Ascend/full source factory selects actual NPU class');check(a.size==1024*cp and a.page_size==128*cp and a.num_pages==8,'DCP expands size and page together');check(a.need_sort==(mode in ['prefill','decode']),'PD enables staged sorted frees');check(a.available_size()==1024*cp,'logical capacity follows source constructor');self.is_draft_worker=True;check(factory(self,sizes=sizes,token_to_kv_pool=None,is_dsv4_model=False,req_to_token_pool=types.SimpleNamespace(),token_to_kv_pool_allocator=a) is a,'nonhybrid draft reuses supplied allocator');result['cases'].append({'family':'source-factory-with-fixture-context','attn_dcp_size':cp,'PD_mode_fixture':mode,'status':'PASS','task_runtime_confirmed':False})
 result['actual_allocator_module']=npu.__name__;result['real_allocator_executed']=True;result['full_facade_constructed']=False;result['native_executed']=False;result['actual_create_prefill_budget_executed']=budget_module is not None
 result.update(probe_status='PASS',correctness='PASS',status_scope='Source-class CPU contract only; full facade import status independent; native acceptance remains E0')
except Exception as exc:
 result.update(probe_status='UNAVAILABLE' if stage in ['full-module-import-once','source-body-loading'] else 'FAIL',correctness='NOT_EVALUATED' if stage in ['full-module-import-once','source-body-loading'] else 'FAIL',failed_stage=stage,error=repr(exc),traceback=traceback.format_exc())
print(json.dumps(result,ensure_ascii=False))
sys.exit(0 if result['probe_status'] in ['PASS','UNAVAILABLE'] else 1)
