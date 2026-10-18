import importlib.util,json,os,pathlib,tempfile,types
from enum import Enum
class Mode(Enum):NONE=0;PIECEWISE=1;FULL=2
class Tensor:
 def __init__(self,n):self.shape=(n,4);self.values=[7]*n
 def __getitem__(self,key):
  parent=self
  class View:
   def zero_(self):parent.values[key]=[0]*len(parent.values[key])
  return View()
class SFA:
 def _sfa_preprocess_prolog_v3(self,hidden_states,*a,**kw):return hidden_states
 def _finalize_o_proj(self,attn_output,*a,**kw):return None
 def forward(self,layer_name,hidden_states,kv_cache,attn_metadata,output=None):
  n=attn_metadata.num_input_tokens
  x=types.SimpleNamespace(shape=(n,4));self._sfa_preprocess_prolog_v3(x);self._finalize_o_proj(x)
  for i in range(n):output.values[i]=3
  return output
class Runner:
 def __init__(self,n=257,padded=1024):
  self.n=n;self.padded=padded;self.mode=Mode.PIECEWISE;self.use_sp=False;self.sparse_kv_offload_enabled=False
  self.parallel_config=types.SimpleNamespace(tensor_parallel_size=16,data_parallel_size=1,pipeline_parallel_size=1,decode_context_parallel_size=1,prefill_context_parallel_size=1)
  self.is_kv_producer=True;self.is_kv_consumer=False;self.model_config=types.SimpleNamespace(is_hybrid=False,use_mla=True);self.num_spec_tokens=1;self.uniform_decode_query_len=2;self.tp_rank=0
  self.input_batch=types.SimpleNamespace(req_ids=['r'],num_computed_tokens_cpu=[0],num_prompt_tokens=[n]);self.requests={'r':types.SimpleNamespace(sampling_params=types.SimpleNamespace(max_tokens=1),mm_features=[])};self.descriptor=types.SimpleNamespace(num_tokens=padded);self.spec=False;self.capture=False;self.ubatch=None;self.cascade=None;self.scheduled=[n];self.meta_padded=None
 def _build_attention_metadata(self,num_tokens,num_reqs,max_query_len,num_tokens_padded=None,num_reqs_padded=None,ubatch_slices=None,logits_indices=None,use_spec_decode=False,for_cudagraph_capture=False,num_scheduled_tokens=None,num_scheduled_tokens_np=None,dcp_dummy_metadata=None,cascade_attn_prefix_lens=None,skip_gdn_state_update=False,cudagraph_runtime_mode=None,batch_descriptor=None,offload_dummy=False):
  self.meta_padded=num_tokens_padded;assert batch_descriptor is self.descriptor
  return types.SimpleNamespace(num_input_tokens=num_tokens_padded,num_actual_tokens=num_tokens)
 def execute_model(self):
  m=self._build_attention_metadata(self.n,1,self.n,num_tokens_padded=self.padded,num_reqs_padded=1,num_scheduled_tokens_np=self.scheduled,cudagraph_runtime_mode=self.mode,batch_descriptor=self.descriptor,ubatch_slices=self.ubatch,use_spec_decode=self.spec,for_cudagraph_capture=self.capture,cascade_attn_prefix_lens=self.cascade)
  out=Tensor(self.padded);result=SFA().forward('target',types.SimpleNamespace(shape=(self.padded,4)),None,m,out);assert result is out;return out
spec=importlib.util.spec_from_file_location('attention','glm5-3-vllm/runtime-patches/p-first-seed-v2/prefill_attention.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);os.environ['GLM53_P_FIRST_DRAFT_EXPORT']='1';current=[None];rows=[]
with tempfile.TemporaryDirectory() as temp:
 root=pathlib.Path(temp);mod.install({'NPUModelRunner':Runner,'CUDAGraphMode':Mode,'enable_sp':lambda:current[0].use_sp},root,sfa_cls=SFA)
 cases=[('control',None,257,1024,False,False),('selected',None,257,1024,True,True),('exact256',None,256,256,True,False),('exact1024',None,1024,1024,True,False),('DP2',lambda x:setattr(x.parallel_config,'data_parallel_size',2),257,1024,True,False),('TP8',lambda x:setattr(x.parallel_config,'tensor_parallel_size',8),257,1024,True,False),('PP2',lambda x:setattr(x.parallel_config,'pipeline_parallel_size',2),257,1024,True,False),('DCP2',lambda x:setattr(x.parallel_config,'decode_context_parallel_size',2),257,1024,True,False),('PCP2',lambda x:setattr(x.parallel_config,'prefill_context_parallel_size',2),257,1024,True,False),('SP',lambda x:setattr(x,'use_sp',True),257,1024,True,False),('FULL',lambda x:setattr(x,'mode',Mode.FULL),257,1024,True,False),('NONE',lambda x:setattr(x,'mode',Mode.NONE),257,1024,True,False),('consumer',lambda x:setattr(x,'is_kv_consumer',True),257,1024,True,False),('hybrid',lambda x:setattr(x.model_config,'is_hybrid',True),257,1024,True,False),('offload',lambda x:setattr(x,'sparse_kv_offload_enabled',True),257,1024,True,False),('P_output64',lambda x:setattr(x.requests['r'].sampling_params,'max_tokens',64),257,1024,True,False),('spec_target',lambda x:setattr(x,'spec',True),257,1024,True,False),('capture',lambda x:setattr(x,'capture',True),257,1024,True,False),('ubatch',lambda x:setattr(x,'ubatch',[1]),257,1024,True,False),('cascade',lambda x:setattr(x,'cascade',[1]),257,1024,True,False),('tiny_query',None,1,256,True,False),('long_tail',lambda x:(x.input_batch.num_computed_tokens_cpu.__setitem__(0,1024),x.input_batch.num_prompt_tokens.__setitem__(0,1537)),513,1024,True,True),('prefix_hit',lambda x:(x.input_batch.num_computed_tokens_cpu.__setitem__(0,128),x.input_batch.num_prompt_tokens.__setitem__(0,257)),129,256,True,True),('warmup',lambda x:(x.input_batch.req_ids.clear(),setattr(x,'scheduled',None)),257,1024,True,False)]
 for name,change,n,padded,enabled,selected in cases:
  obj=Runner(n,padded);current[0]=obj
  if change:change(obj)
  if enabled:(root/'enable').touch()
  else:(root/'enable').unlink(missing_ok=True)
  out=obj.execute_model();assert obj.descriptor.num_tokens==padded;assert obj.meta_padded==(n if selected else padded),(name,obj.meta_padded)
  assert out.values[:n]==[3]*n
  assert out.values[n:]==([0]*(padded-n) if selected else [3]*(padded-n)),name
  rows.append({'case':name,'selected':selected,'attention_tokens':obj.meta_padded,'unchanged_graph_tokens':padded,'actual_prefix_preserved_and_tail_defined':True})
report={'passed':True,'cases':rows,'scope':'CPU argument/guard/buffer contract only, fake attention body. Native source pins match installed image; device shapes/quality/performance remain unproven.'}
pathlib.Path('glm5-3-vllm/evidence/20261018-p-first-seed-device-v2/recovery-v2/prefill-attention-cpu-check-v2.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'passed':True,'cases':len(rows)}))
