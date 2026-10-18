import importlib.util,json,os,pathlib,tempfile,types
from enum import Enum
class Mode(Enum):NONE=0;PIECEWISE=1;FULL=2
class Runner:
 def __init__(self,n=257):
  self.parallel_config=types.SimpleNamespace(tensor_parallel_size=16,data_parallel_size=1,pipeline_parallel_size=1,decode_context_parallel_size=1,prefill_context_parallel_size=1)
  self.is_kv_producer=True;self.is_kv_consumer=False;self.model_config=types.SimpleNamespace(is_hybrid=False);self.num_spec_tokens=1;self.uniform_decode_query_len=2;self.tp_rank=0;self.alignment=1;self.force_mode=Mode.PIECEWISE
  self.input_batch=types.SimpleNamespace(req_ids=['r'],num_computed_tokens_cpu=[0],num_prompt_tokens=[n]);self.requests={'r':types.SimpleNamespace(sampling_params=types.SimpleNamespace(max_tokens=1))};self.calls=[]
 def _pad_for_sequence_parallelism(self,n):return ((n+self.alignment-1)//self.alignment)*self.alignment
 def _determine_batch_execution_and_padding(self,num_tokens,num_reqs,num_scheduled_tokens_np,max_num_scheduled_tokens,use_cascade_attn,allow_microbatching=False,force_eager=False,force_uniform_decode=None,force_has_lora=None,force_num_active_loras=None,num_encoder_reqs=0):
  self.calls.append({'force_eager':force_eager,'cascade':use_cascade_attn,'encoder':num_encoder_reqs})
  mode=Mode.NONE if force_eager else self.force_mode
  size=self._pad_for_sequence_parallelism(num_tokens) if force_eager else (256 if num_tokens<=256 else 1024)
  return mode,types.SimpleNamespace(num_tokens=size),False,None,{'mode':mode.name,'tokens':size}
spec=importlib.util.spec_from_file_location('padding','glm5-3-vllm/runtime-patches/p-first-seed-v2/prefill_padding.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);os.environ['GLM53_P_FIRST_DRAFT_EXPORT']='1';rows=[]
with tempfile.TemporaryDirectory() as temp:
 root=pathlib.Path(temp);module.install({'NPUModelRunner':Runner,'CUDAGraphMode':Mode},root)
 for name,change,n,enabled,expect in [
 ('control',None,257,False,1024),('selected',None,257,True,257),('no_padding',None,256,True,256),
 ('DP2',lambda x:setattr(x.parallel_config,'data_parallel_size',2),257,True,1024),
 ('PP2',lambda x:setattr(x.parallel_config,'pipeline_parallel_size',2),257,True,1024),
 ('CP2',lambda x:setattr(x.parallel_config,'prefill_context_parallel_size',2),257,True,1024),
 ('consumer',lambda x:setattr(x,'is_kv_consumer',True),257,True,1024),
 ('hybrid',lambda x:setattr(x.model_config,'is_hybrid',True),257,True,1024),
 ('direct_P_multioutput',lambda x:setattr(x.requests['r'].sampling_params,'max_tokens',64),257,True,1024),
 ('FULL_preserved',lambda x:setattr(x,'force_mode',Mode.FULL),257,True,1024),
 ('small_query_preserved',None,1,True,256),
 ('SP_alignment_preserved',lambda x:setattr(x,'alignment',16),257,True,272),
 ('partial_prefill',lambda x:(x.input_batch.num_computed_tokens_cpu.__setitem__(0,1024),x.input_batch.num_prompt_tokens.__setitem__(0,1537)),513,True,513)]:
  obj=Runner(n)
  if change:change(obj)
  if enabled:(root/'enable').touch()
  else:(root/'enable').unlink(missing_ok=True)
  result=obj._determine_batch_execution_and_padding(num_tokens=n,num_reqs=1,num_scheduled_tokens_np=[n],max_num_scheduled_tokens=n,use_cascade_attn=True,num_encoder_reqs=0)
  assert result[1].num_tokens==expect,(name,result[1].num_tokens,expect)
  assert all(z['cascade'] is True and z['encoder']==0 for z in obj.calls)
  rows.append({'case':name,'selected_native_eager':len(obj.calls)==2,'result_tokens':result[1].num_tokens,'result_mode':result[0].name})
report={'passed':True,'cases':rows,'scope':'CPU signature/guard/native-argument preservation only, fake native descriptor; not framework/device correctness or benefit'}
path=pathlib.Path('glm5-3-vllm/evidence/20261018-p-first-seed-device-v2/recovery-v2/prefill-padding-cpu-check-v1.json');path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
