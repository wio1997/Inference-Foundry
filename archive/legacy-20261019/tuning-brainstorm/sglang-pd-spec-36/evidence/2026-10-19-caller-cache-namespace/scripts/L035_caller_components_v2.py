"""Unchanged caller/Core bodies. Service config, timers, queues and physical capacity are explicit CPU boundaries."""
import ast,copy,dataclasses,enum,heapq,sys,types,typing
from collections import Counter,defaultdict
from abc import ABC,abstractmethod
from types import SimpleNamespace as NS
from L034_extension_components_v4 import R,S,ns,load,import_source,records,registry,flags,Flag,torch,msgspec,np,BOUNDARIES
ns.update(float64=np.float64,NodeId=int,Counter=Counter,defaultdict=defaultdict,heapq=heapq,ABC=ABC,abstractmethod=abstractmethod,sys=sys)
load('python/sglang/srt/utils/common.py',functions=['flatten_nested_list']);load('python/sglang/srt/multimodal/mm_utils.py',functions=['has_valid_data'])
load('python/sglang/srt/managers/io_struct.py',classes=['GenerateReqInput'])
sampling=import_source('L035_real_sampling_params','python/sglang/srt/sampling/sampling_params.py');ns['SamplingParams']=sampling.SamplingParams
load('python/sglang/srt/managers/tokenizer_manager.py',methods={'TokenizerManager':['_validate_and_resolve_lora','_resolve_lora_path','_create_tokenized_object']})
load('python/sglang/srt/disaggregation/utils.py',classes=['DisaggregationMode','TransferBackend'])
# Capture source methods without importing accelerator-owned module initialization.
load('python/sglang/srt/managers/schedule_batch.py',methods={'Req':['__init__','detach_kv','finished','set_finish_with_abort','_refresh_fill_ids','init_next_round_input','_compute_max_prefix_len','is_prefill_only']},bases={'Req':['ReqDllmMixin']})
load('python/sglang/srt/managers/utils.py',functions=['validate_input_length'])
load('python/sglang/srt/beam_search/coordinator.py',methods={'BeamCoordinator':['request_beam_width']})
load('python/sglang/srt/managers/scheduler.py',methods={'Scheduler':['handle_generate_request','init_req_max_new_tokens','_maybe_namespace_elastic_radix_cache']})
load('python/sglang/srt/model_executor/model_runner.py',methods={'ModelRunner':['maybe_init_lora_manager']})
class TimeBoundary:
 def __init__(self,**kw):self.tokenized=0
 @classmethod
 def new_from_obj(cls,obj):return cls()
 def set_metrics_collector(self,*a):pass
 def set_scheduler_recv_time(self):pass
 def set_tokenize_finish_time(self):self.tokenized+=1
ns['SchedulerReqTimeStats']=TimeBoundary
lora_cfg=NS(enable_lora=True,max_loaded_loras=8);ns['get_lora']=lambda:lora_cfg
ns['get_disagg']=lambda:NS(disaggregation_transfer_backend='ascend',disaggregation_bootstrap_port=8998)
ns['get_parallel']=lambda:NS(tp_rank=1,pp_rank=1,attn_dcp_size=1)
ns['get_spec']=lambda:NS(speculative_algorithm='EAGLE',speculative_num_steps=3,speculative_eagle_topk=1,speculative_num_draft_tokens=4)
ns['get_serving']=lambda:NS(allow_auto_truncate=False)
ns['get_exec']=lambda:NS(moe=NS(elastic_ep_backend=None))
ns['get_device']=lambda:NS(mlx_enable_sampling=False)
flags.SGLANG_RADIX_FORCE_MISS=Flag(False)
class AlgoBoundary:
 label='EAGLE reference; not a constructed speculative worker'
 def is_dflash_family(self):return False
 def is_uno(self):return False
 def is_none(self):return False

def all_bodies(path,assigns=False):
 p=S/path;nodes=[n for n in ast.parse(p.read_text()).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) or assigns and isinstance(n,(ast.Assign,ast.AnnAssign))]
 for n in nodes:records.append({'path':path,'symbol':getattr(n,'name',ast.unparse(n).split('=')[0]),'AST_sha256':__import__('hashlib').sha256(ast.dump(n,include_attributes=False).encode()).hexdigest(),'whole_original_body':True})
 module=ast.Module(body=[ast.parse('from __future__ import annotations').body[0]]+nodes,type_ignores=[]);ast.fix_missing_locations(module);exec(compile(module,str(p),'exec'),ns)
b='python/sglang/srt/mem_cache/'
ct=import_source('L035_component_type',b+'unified_cache/component_type.py');ns.update(ComponentType=ct.ComponentType,BASE_COMPONENT_TYPE=ct.BASE_COMPONENT_TYPE)
init=import_source('L035_cache_init',b+'cache_init_params.py');ns['CacheInitParams']=init.CacheInitParams
policy=import_source('L035_evict_policy',b+'evict_policy.py');ns.update({k:v for k,v in vars(policy).items() if k.endswith('Strategy')})
load(b+'utils.py',functions=['get_eviction_strategy','split_node_hash_value'])
mapping=next(n for n in ast.parse((S/b/'utils.py').read_text()).body if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name) and n.target.id=='_EVICTION_POLICY_FACTORIES');exec(compile(ast.Module(body=[ast.parse('from __future__ import annotations').body[0],mapping],type_ignores=[]),'original-eviction-map','exec'),ns)
load(b+'base_prefix_cache.py',classes=['CacheRequestHandle','MatchPrefixParams','InsertParams','InsertResult','EvictParams','EvictResult','IncLockRefResult','DecLockRefParams','DecLockRefResult','MatchResult'],functions=['_dfs_weight_order'])
load(b+'radix_cache.py',classes=['RadixKey']);load('python/sglang/srt/disaggregation/kv_events.py',classes=['StorageMedium']);load(b+'events.py',classes=['KVCacheEventRecorder'])
all_bodies(b+'unified_cache/cache_action.py',True);all_bodies(b+'unified_cache/unified_tree_core_interface.py',True);all_bodies(b+'unified_cache/components/base.py',True);all_bodies(b+'unified_cache/components/full.py');all_bodies(b+'unified_cache/unified_tree_core.py',True)
class CapacityBoundary:
 device=torch.device('cpu')
 # Represents an adequate CPU test budget, not an actual NPU memory pool.
 def max_new_tokens_for_memory(self,input_len,max_new_tokens,**kw):return max_new_tokens
class CoreView:
 def __init__(self,page,eagle):
  self.capacity=CapacityBoundary();params=init.CacheInitParams(False,None,self.capacity,page,is_eagle=eagle,tree_components=(ct.ComponentType.FULL,));self.enable_session_radix_cache=False;self.token_to_kv_pool_allocator=self.capacity;self.cache_controller=None
  self.comp=ns['FullComponent'](self,params);self.tree_core=ns['UnifiedTreeCore'](params,{ct.ComponentType.FULL:self.comp});self.page_size=page;self.actions=[]
 def match_prefix(self,params):return self.tree_core.match_prefix(params)
 def supports_mamba(self):return False
 def swa_reprefill_tail_tokens(self):return 0
 def is_tree_cache(self):return True
 def insert(self,req,start=1000):
  vals=torch.arange(start,start+len(req.origin_input_ids),dtype=torch.int64);step=self.tree_core.begin_insert(ns['InsertParams'](key=ns['RadixKey'](req.origin_input_ids,req.extra_key,cache_salt=req.cache_salt),value=vals));self.actions.extend(step.actions)
  while step.result is None:step=self.tree_core.resume_insert();self.actions.extend(step.actions)
  self.actions.extend(self.tree_core.end_insert());return vals,step.result

def manager(refs):
 m=ns['TokenizerManager']();m.enable_lora=True;m.lora_registry=registry.LoRARegistry(refs);m.lora_ref_cache={r.lora_name:r for r in refs};m.preferred_sampling_params=None;m.sampling_params_class=sampling.SamplingParams;m.tokenizer=None;m.model_config=NS(vocab_size=154856);m.rid_to_state={};return m

def scheduler(view,mode):
 s=ns['Scheduler']();s.enable_session_radix_cache=False;s.model_config=NS(vocab_size=154856,hf_eos_token_id={154820});s.metrics_reporter=NS(enable_metrics=False);s.dllm_config=None;s.tokenizer=None;s.disaggregation_mode=mode;s.transfer_backend=ns['TransferBackend'].ASCEND;s.disable_radix_cache=False;s.tree_cache=view;s.spec_algorithm=AlgoBoundary();s.enable_overlap=False;s.max_req_input_len=65536;s.max_req_len=65536;s.max_new_tokens_limit=None;s.token_to_kv_pool_allocator=view.capacity;s.max_total_num_tokens=131072;s.sliding_window_size=None;s.chunked_prefill_size=32768;s.grammar_manager=NS(process_req_with_grammar=lambda req:False);s.validate_dllm_request=lambda req:None;s.queued=[];s._add_request_to_queue=s.queued.append;return s
BOUNDARIES_L035=BOUNDARIES+['Whole GenerateReqInput normalizer and SamplingParams module plus original registry-resolve/carrier/Scheduler request bodies and Req cache-lookup methods; no HTTP/Pydantic endpoint dispatch or process IPC','Full original UnifiedTreeCore/FULL component classes and actual Torch CPU index arrays; CoreView adapts capabilities, no full UnifiedRadixCache constructor/collectives or physical KV values','Scheduler config, metrics/time, adequate capacity, no-DLLM/no-grammar and queue dispatch are explicit periphery fixtures; no admission, transfer, NPU/forward or model-output correctness','EAGLE3step/topk1/draft4 represented as reference config; no native speculative worker constructed. is_eagle True/False are separate CPU tree-key controls; page1 is not NPU support evidence']
