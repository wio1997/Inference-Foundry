"""Exercise cross-step ownership, fallback and restoration using changing inputs."""
from pathlib import Path
import importlib.util, json, os, tempfile, types
import numpy as np
r=Path(__file__).resolve().parent;repo=r.parents[3]
s=importlib.util.spec_from_file_location('reuse',repo/'glm5-3-vllm/runtime-patches/p-first-seed-v2/rope_step_reuse.py')
h=importlib.util.module_from_spec(s);s.loader.exec_module(h)
os.environ['GLM53_P_FIRST_DRAFT_EXPORT']='0'
N=types.SimpleNamespace
device=N(type='npu')
class Tensor:
 def __init__(self, a, dtype='bf16', ptr=None):self.a=np.array(a,copy=True);self.dtype=dtype;self.device=device;self.ptr=ptr or id(self)
 @property
 def shape(self):return self.a.shape
 def stride(self):return tuple(s//self.a.itemsize for s in self.a.strides)
 def data_ptr(self):return self.ptr
 def copy(self):return Tensor(self.a,self.dtype)
checks=[];native_calls=[];capturing={'v':False};throw={'v':False}
def getter(pos,use_cache=False):
 native_calls.append((pos.a.copy(),use_cache))
 x=np.repeat(pos.a[:,None,None,None],64,axis=3)
 return Tensor(x),Tensor(-x)
class SFA:pass
class Indexer:pass
indexer=N(AscendSFAIndexerMetadataBuilder=Indexer,get_cos_and_sin_mla=getter)
primary_group=N(get_metadata_builder=lambda:SFA(),layer_names=['sfa'])
cache_group=N(get_metadata_builder=lambda:Indexer(),layer_names=['idx'])
class Proposer:
 method='mtp'
 def _build_cache_only_group_next_step_attn_metadata(self,cad,draft_index,num_input_tokens,primary_group,primary_metadata,cache_only_groups):
  c,s=indexer.get_cos_and_sin_mla(cad.positions,False)
  if throw['v']:raise RuntimeError('copy failed')
  return dict(sfa=primary_metadata,idx=N(cos=c.copy(),sin=s.copy()))
class Runner:
 def __init__(self):
  self.parallel_config=N(tensor_parallel_size=16,data_parallel_size=1,pipeline_parallel_size=1,decode_context_parallel_size=1,prefill_context_parallel_size=1)
  self.is_kv_consumer=True;self.is_kv_producer=False;self.use_async_scheduling=True;self.num_spec_tokens=5;self.model_config=N(use_mla=True,is_hybrid=False);self.dynamic_eplb=False;self.input_batch=N(req_ids=['a']);self.requests={'a':N(sampling_params=N(max_tokens=64))};self.tp_rank=0
  self.proposer=Proposer();self.positions=Tensor([1,2,3],dtype='i64');self.draft=1;self.groups=[cache_group];self.primary_group=primary_group
 def sample_tokens(self):
  pair=getter(self.positions);self.primary=N(positions=self.positions,cos=pair[0],sin=pair[1])
  return self.proposer._build_cache_only_group_next_step_attn_metadata(N(positions=self.positions),self.draft,3,self.primary_group,self.primary,self.groups)
with tempfile.TemporaryDirectory() as tmp:
 root=Path(tmp)
 h.install(dict(torch=N(int64='i64',bfloat16='bf16',equal=lambda a,b:np.array_equal(a.a,b.a)),NPUModelRunner=Runner,enable_sp=lambda:False),root,dict(indexer=indexer,SFA=SFA,Proposer=Proposer,ctx=lambda:N(capturing=capturing['v']),ctx_available=lambda:True))
 runner=Runner()
 def run(expected_native):
  before=len(native_calls);v=runner.sample_tokens();assert len(native_calls)-before==expected_native
  assert indexer.get_cos_and_sin_mla is getter
  assert np.array_equal(v['sfa'].cos.a,v['idx'].cos.a)
  assert v['sfa'].cos.data_ptr()!=v['idx'].cos.data_ptr()
  return v
 run(2);checks.append('default_off_calls_native_getter')
 (root/'enable').touch();run(2);checks.append('observer_compares_original_and_preserves_independent_indexer_copy')
 (root/'disabled').touch();a=run(1);checks.append('observer_off_removes_only_duplicate_getter')
 runner.positions.a[:]=[100,7,8191];b=run(1)
 assert not np.array_equal(a['idx'].cos.a,b['idx'].cos.a) and np.array_equal(a['idx'].cos.a,np.repeat(np.array([1,2,3])[:,None,None,None],64,axis=3))
 checks.append('same_position_address_changed_values_use_new_step_source_and_keep_previous_copy')
 for attr,value in [('draft',0),('draft',5),('groups',[]),('groups',[cache_group,cache_group]),('primary_group',N(get_metadata_builder=lambda:object()))]:
  old=getattr(runner,attr);setattr(runner,attr,value);run(2);setattr(runner,attr,old)
 checks.append('draft_boundary_and_group_topology_fallback')
 capturing['v']=True;run(2);capturing['v']=False;checks.append('capture_stays_native')
 runner.proposer.method='eagle3';run(2);runner.proposer.method='mtp';checks.append('non_mtp_stays_native')
 runner.positions.dtype='i32';run(2);runner.positions.dtype='i64';checks.append('casted_position_source_stays_native')
 runner.parallel_config.tensor_parallel_size=8;run(2);runner.parallel_config.tensor_parallel_size=16;checks.append('unsupported_parallel_stays_native')
 runner.requests['a'].sampling_params.max_tokens=1;run(2);runner.requests['a'].sampling_params.max_tokens=64;checks.append('singleton_stays_native')
 throw['v']=True
 try:runner.sample_tokens()
 except RuntimeError:pass
 else:raise AssertionError('exception lost')
 assert indexer.get_cos_and_sin_mla is getter;throw['v']=False;run(1);checks.append('exception_restores_getter_and_frame')
 (root/'enable').unlink();run(2);checks.append('native_restored_after_candidate')
 # A callback outside sample scope must not inherit any memo or selector.
 before=len(native_calls);runner.proposer._build_cache_only_group_next_step_attn_metadata(N(positions=runner.positions),1,3,primary_group,runner.primary,[cache_group]);assert len(native_calls)==before+1
 checks.append('no_cross_callback_or_outside_scope_memo')
out=dict(completed=True,passed=len(checks),checks=checks,scope='CPU ownership/guard/error checks with changing same-address inputs. No NPU, full-model correctness or performance claim.')
(r/'rope-step-reuse-cpu-check-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
