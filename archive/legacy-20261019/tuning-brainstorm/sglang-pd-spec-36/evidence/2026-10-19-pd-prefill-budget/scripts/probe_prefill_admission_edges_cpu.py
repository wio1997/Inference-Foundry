#!/usr/bin/env python3
"""Unchanged fixed-pool budget and whole PrefillAdder class on explicit CPU boundaries."""
import argparse,ast,copy,dataclasses,datetime,enum,functools,hashlib,importlib,json,logging,os,pathlib,subprocess,sys,traceback,types
from array import array
from contextlib import contextmanager
ap=argparse.ArgumentParser()
for name in ['source-root','expected-hashes','model-hashes','case-key','protocol','fixture-root']:ap.add_argument('--'+name,required=True)
ap.add_argument('--stdout',action='store_true');args=ap.parse_args();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();root=pathlib.Path('/sgl-workspace/sglang');snapshot=pathlib.Path(args.source_root);model=pathlib.Path('/model');pp=pathlib.Path(args.protocol);p=json.loads(pp.read_text());expected=json.loads(args.expected_hashes);models=json.loads(args.model_hashes)
assert os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0' and os.environ.get('SGLANG_USE_CPU_ENGINE')=='1';nodes=sorted({str(q) for pat in ['davinci*','devmm_svm','hisi_hdc','ascend*'] for q in pathlib.Path('/dev').glob(pat)});assert not nodes
head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip();assert head==p['source_commit'];hashes={}
for f,h in expected.items():hashes[f]=sha(root/f);assert hashes[f]==h==sha(snapshot/f);assert subprocess.check_output(['git','-C',str(root),'show',head+':'+f])==(root/f).read_bytes()
for f,h in models.items():assert sha(model/f)==h
result={'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'probe_status':'PENDING','correctness':'NOT_EVALUATED','source_commit':head,'source_hashes':hashes,'model_hashes':models,'script_sha256':sha(pathlib.Path(__file__)),'protocol_sha256':sha(pp),'accelerator_device_nodes':nodes,'source_bodies':{},'budget_cases':[],'batch_cases':[],'transaction_cases':[],'selection_cases':[],'continuation_cases':[],'scope':'Whole unchanged PrefillBudget/PrefillAdder class bodies and selected Req/PD/cache dataclass bodies. Actual common.Range module import and TorchCPU tensors. Allocator/cache capacity, locks, load-back and negotiation are explicit boundary fixtures; no real allocator, cache tree, H2D, complete scheduler/bootstrap module or native/model/performance.'};stage='imports'
try:
 import torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1);common=importlib.import_module('sglang.srt.utils.common');assert sha(pathlib.Path(common.__file__))==expected[p['common_source']];result['actual_module']={'name':common.__name__,'file':common.__file__,'sha256':sha(pathlib.Path(common.__file__)),'Range_unmodified':True};result['environment']={'python':sys.version,'torch':torch.__version__};future=ast.parse('from __future__ import annotations').body[0]
 module=types.ModuleType('if_prefill_source_bodies');sys.modules[module.__name__]=module;env=module.__dict__;schedule=types.SimpleNamespace(schedule_policy='fcfs');env.update(torch=torch,Range=common.Range,dataclass=dataclasses.dataclass,dataclasses=dataclasses,Enum=enum.Enum,auto=enum.auto,contextmanager=contextmanager,lru_cache=functools.lru_cache,array=array,logger=logging.getLogger('if_prefill_probe'),CLIP_MAX_NEW_TOKENS=4096,_IS_HIP=False,PREFILL_TILE_BUDGET=1,PREFILL_TILE_BUDGET_MODE='causal',get_schedule=lambda:schedule,get_exec=lambda:types.SimpleNamespace(features=types.SimpleNamespace(enable_encoder_swa_bounded_replay=False)),envs=types.SimpleNamespace(SGLANG_EXACT_CHUNK_FILL=types.SimpleNamespace(get=lambda:True)),is_gfx95_supported=lambda:False)
 for name in ['SWATokenToKVPoolAllocator','DeepSeekV4HiSparseTokenToKVPoolAllocator','UnifiedMambaTokenToKVPoolAllocator','UnifiedMambaSWATokenToKVPoolAllocator']:env[name]=type(name+'Sentinel',(),{})
 def tree(path):return ast.parse((root/path).read_text())
 def bodyhash(n):return hashlib.sha256(ast.dump(ast.Module(body=n.body,type_ignores=[]),include_attributes=False).encode()).hexdigest()
 def execute(nodes,label):
  for n in nodes:result['source_bodies'][label+':'+n.name]={'body_ast_sha256':bodyhash(n),'body_unchanged':True,'decorators_preserved':True}
  exec(compile(ast.Module(body=[future,*copy.deepcopy(nodes)],type_ignores=[]),label,'exec'),env)
 bt=tree(p['budget_source']);execute([next(n for n in bt.body if isinstance(n,ast.ClassDef) and n.name=='PrefillBudget')],'budget')
 ct=tree(p['cache_source']);execute([next(n for n in ct.body if isinstance(n,ast.ClassDef) and n.name==name) for name in ['DecLockRefParams','IncLockRefResult','InitLoadBackParams']],'cache-records')
 st=tree(p['policy_source']);execute([next(n for n in st.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name) for name in ['_use_exact_chunk_fill','AddReqResult','_PrefillAdmission','PrefillAdder']],'policy')
 rt=tree(p['req_source']);rcls=next(n for n in rt.body if isinstance(n,ast.ClassDef) and n.name=='Req');rnames=['set_extend_range','get_fill_ids','needs_host_load_back','materialized_host_hit_len','fulfilled_storage_hit_len'];rnodes=[next(n for n in rcls.body if isinstance(n,ast.FunctionDef) and n.name==name) for name in rnames];execute(rnodes,'req-methods');execute([next(n for n in rt.body if isinstance(n,ast.FunctionDef) and n.name=='split_cached_prefix_by_tier')],'cache-tier-split');Req=type('ReqFixture',(),{name:env[name] for name in rnames})
 pt=tree(p['pd_source']);proc=[m for c in pt.body if isinstance(c,ast.ClassDef) for m in c.body if isinstance(m,ast.FunctionDef) and m.name=='_process_req'];assert len(proc)==1;execute(proc,'PD-local-process');process=env['_process_req'];Budget=env['PrefillBudget'];Adder=env['PrefillAdder'];Verdict=env['AddReqResult'];Receipt=env['IncLockRefResult']
 class Allocator:
  def __init__(self,free):self.free=free;self.page_size=128
  def available_size(self):return self.free
  def create_prefill_budget(self,cache,*,num_mixed_decode_tokens=0):return Budget(self,cache,num_mixed_decode_tokens=num_mixed_decode_tokens)
 class Cache:
  disable=False
  def __init__(self,evictable=0,mamba=False):self.base=evictable;self.mamba=mamba;self.nodes={};self.events=[];self.load_calls=0;self.load_result=None;self.finished=[]
  def node(self,tokens=0):
   n=types.SimpleNamespace(key=len(self.nodes)+1,tokens=tokens,refs=0);self.nodes[n.key]=n;return n
  def supports_mamba(self):return self.mamba
  def evictable_size(self):return self.base-sum(n.tokens for n in self.nodes.values() if n.refs)
  def full_evictable_size(self):return self.evictable_size()+64
  def is_tree_cache(self):return True
  def inc_lock_ref(self,node):node.refs+=1;self.events.append(('inc',node.key));return Receipt(node_id=node.key)
  def dec_lock_ref(self,node,params=None):assert params is None or params.node_id==node.key;assert node.refs>0;node.refs-=1;self.events.append(('dec',node.key))
  def init_load_back(self,params):
   self.load_calls+=1;self.events.append(('load',params.host_hit_length))
   if self.load_result is None:return None
   return torch.arange(self.load_result,dtype=torch.int64),self.node(0)
  def finish_storage_prefetch_admission(self,handle,**kw):self.finished.append((handle,kw))
  def discard_storage_prefetch_accounting(self,handle):raise AssertionError('Unexpected retraction path')
 class Running:
  reqs=[]
  def batch_size(self):return 0
 class Delayer:
  def __init__(self,allow):self.allow=allow;self.calls=[]
  def negotiate_should_allow_prefill(self,**kw):self.calls.append(kw);return self.allow
 def make_adder(free=262144,evictable=0,chunk=16384,input_cap=16384,delay=None):
  a=Allocator(free);c=Cache(evictable);ad=Adder(128,c,a,Running(),0.6,input_cap,chunk,max_prefill_bs=3,max_running_requests=64,prefill_delayer_single_pass=delay);assert not ad.exact_chunk_fill and not ad.is_hybrid_swa and ad._mamba_slot_cost==0;return ad,a,c
 def request(c,L,prefix=0,host=0):
  r=Req();r.rid='fixture-'+str(len(c.nodes));r.full_untruncated_fill_ids=array('q',range(L));r.origin_input_ids=r.full_untruncated_fill_ids;r.output_ids=array('q');r.prefix_indices=torch.arange(prefix,dtype=torch.int64);r.sampling_params=types.SimpleNamespace(max_new_tokens=512,ignore_eos=False);process(None,r);assert r.sampling_params.max_new_tokens==1;r.extend_range=None;r.retracted_stain=False;r.last_node=c.node(-(-prefix//128)*128);r.best_match_node=r.last_node;r.host_hit_length=host;r.host_loaded_length=0;r.swa_host_hit_length=r.mamba_host_hit_length=0;r.storage_hit_length=0;r.storage_hit_start=None;r.host_hit_is_storage=False;r.cache_request_handle=r.rid;r.lock_receipt=None;r.kv=types.SimpleNamespace(holds_mamba=False,req_pool_idx=None,cache_protected_len=prefix);return r
 def snap(ad,r,c):return {'range':list(r.extend_range) if r.extend_range is not None else None,'prefix_len':len(r.prefix_indices),'host_loaded':r.host_loaded_length,'last_node':r.last_node.key,'total_offset':ad.memory_budget.total_offset,'current_offset':ad.memory_budget.current_offset,'input_remaining':ad.rem_input_tokens,'chunk_remaining':ad.rem_chunk_tokens,'admitted':len(ad.can_run_list),'new_chunked':ad.new_chunked_req is not None,'locks':{str(k):n.refs for k,n in c.nodes.items()}}
 stage='batch-admission'
 for case in p['batch_cases']:
  result['in_progress']=case;prefixes=case.get('prefixes',[0]*len(case['lengths']));ev=sum(-(-x//128)*128 for x in prefixes);ad,a,c=make_adder(free=case['free'],evictable=ev,chunk=case['chunk'],input_cap=case['input_cap']);reqs=[request(c,L,pre) for L,pre in zip(case['lengths'],prefixes)];steps=[]
  for r in reqs:
   before=len(ad.can_run_list);v=ad.add_one_req(r,has_chunked_req=ad.new_chunked_req is not None,truncation_align_size=None);steps.append({'verdict':v.name,'committed_this_call':len(ad.can_run_list)>before})
   if v is not Verdict.CONTINUE:break
  lens=[r.extend_range.length for r in ad.can_run_list];assert lens==case['rows'];assert ad.cur_rem_tokens==case['expected_current_remaining'] and ad.rem_total_tokens==case['expected_total_remaining'] and steps[-1]['verdict']==case['expected_verdict'];assert ad.memory_budget.current_offset==case['current_offset'] and ad.memory_budget.total_offset==case['total_offset'];assert ad.rem_input_tokens==case['input_cap']-case['compute_charge'];assert ad.rem_chunk_tokens==(None if case['chunk'] is None else case['chunk']-case['compute_charge']);assert ad.log_input_tokens==sum(lens);assert all(len(r.get_fill_ids())==r.extend_range.end for r in ad.can_run_list);assert all(n.refs==(1 if any(r.last_node is n for r in ad.can_run_list) else 0) for n in c.nodes.values());assert c.load_calls==0
  result['batch_cases'].append({**case,'status':'PASS','actual_fixture_rows':sum(lens),'ranges':[list(r.extend_range) for r in ad.can_run_list],'steps':steps,'remaining_input':ad.rem_input_tokens,'remaining_chunk':ad.rem_chunk_tokens,'remaining_current':ad.cur_rem_tokens,'remaining_total':ad.rem_total_tokens,'scope':'Actualclass CPUrequest/Range operations; syntheticIDs andcache/allocator boundaryfixtures, notfullscheduler/devicebatch'});result.pop('in_progress')
 stage='transaction-boundaries'
 for case in p['transaction_cases']:
  result['in_progress']=case;prefix=case.get('prefix',0);delay=Delayer(case.get('allow',True));ad,a,c=make_adder(free=case.get('free',262144),evictable=-(-prefix//128)*128,chunk=case.get('chunk',16384),delay=delay);r=request(c,case.get('length',512),prefix,case.get('host',128));c.load_result=case.get('loaded');before=snap(ad,r,c);caught=None
  try:v=ad.add_one_req(r,has_chunked_req=False,truncation_align_size=None);verdict=v.name
  except RuntimeError as exc:
   assert case.get('error')=='partial-load' and 'all promised FULL tokens or none' in str(exc);caught=str(exc);verdict='ERROR'
  after=snap(ad,r,c);assert verdict==case['verdict'] and len(delay.calls)==case['negotiation_calls'] and c.load_calls==case['load_calls'];rows=r.extend_range.length if r.extend_range is not None else 0;assert rows==case['rows'];assert ad.memory_budget.current_offset==case['current_offset'] and ad.memory_budget.total_offset==case['total_offset']
  if not case.get('error') and not rows:assert before==after
  if case.get('error'):assert r.host_loaded_length==64 and r.extend_range is None and all(n.refs==0 for n in c.nodes.values())
  else:assert sum(n.refs for n in c.nodes.values())==(1 if rows else 0)
  result['transaction_cases'].append({**case,'status':'PASS','before':before,'after':after,'error_observation':caught,'scope':'Explicit load-back/negotiation/lock fixtures; no realH2D orphysicalallocation effect modeled. Partialloadfailure maymutate request metadata, notactivebugclaim.'});result.pop('in_progress')
 stage='pure-selection'
 for case in p['selection_cases']:
  ad,a,c=make_adder(free=case.get('free',262144),evictable=-(-case['prefix']//128)*128);r=request(c,case['length'],case['prefix']);before=snap(ad,r,c);total=case['length']-case['prefix']+1+128;out=ad._select_prefill_admission(r,total_tokens=total,host_hit_length=0,swa_host_hit_length=0,truncation_align_size=None);assert snap(ad,r,c)==before and not c.events and not c.load_calls
  if isinstance(out,Verdict):observed=out.name
  else:
   observed=[out.prefix_len,out.extend_len,out.max_new_tokens,out.is_chunked]
   try:out.extend_len=1
   except dataclasses.FrozenInstanceError:pass
   else:raise AssertionError('Admission notimmutable')
  assert observed==case['expected'];result['selection_cases'].append({**case,'status':'PASS','state_unchanged':True,'scope':'Pureselection only; outeradd_one_req temporarilylocks prefix beforecallingselection'})
 stage='continuations'
 for case in p['continuation_cases']:
  ad,a,c=make_adder(free=65536,evictable=16384);r=request(c,32768,16384);c.inc_lock_ref(r.last_node);ad.memory_budget.total_offset=65536-case['remaining_total'];initial_current=ad.cur_rem_tokens;out=ad.add_chunked_req(r);assert r.extend_range.length==case['rows'] and (out is r)==case['truncated'];assert len(ad.can_run_list)==1 and initial_current==65536 and r.last_node.refs==1
  result['continuation_cases'].append({**case,'status':'PASS','range':list(r.extend_range),'initial_current':initial_current,'final_remaining_total':ad.rem_total_tokens,'final_remaining_current':ad.cur_rem_tokens,'scope':'Alreadylocked continuation andfuture-offset fixtures; no physicalfreecapacityzero ornativeallocation guarantee'})
 assert 'torch_npu' not in sys.modules and 'sgl_kernel_npu' not in sys.modules
 result.update({'probe_status':'PASS','correctness':'PASS','counts':{k:len(result[k]) for k in ['budget_cases','batch_cases','transaction_cases','selection_cases','continuation_cases']},'native_executed':False,'full_budget_or_policy_module_imported':False,'full_scheduler_or_PD_queue_executed':False,'performance':'NOT_MEASURED','fixed_pool_only':True,'D_output_policy_validated':False})
except ImportError as exc:result.update({'probe_status':'UNAVAILABLE','correctness':'NOT_EVALUATED','stage':stage,'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()})
except Exception as exc:result.update({'probe_status':'ERROR','correctness':'FAIL','stage':stage,'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()})
for f,h in models.items():assert sha(model/f)==h
print(json.dumps(result));sys.exit(1 if result['probe_status']=='ERROR' else 0)
