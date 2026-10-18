import importlib.util,json,os,pathlib,tempfile,types
from enum import Enum
class Mode(Enum):NONE=0;PIECEWISE=1;FULL=2
class SFA:pass
class Other:pass
class Tensor:
 def data_ptr(self):return 42
ctx=types.SimpleNamespace(cudagraph_runtime_mode=Mode.FULL,capturing=False,device_metadata_executor=None,batch_descriptor='key')
flags={'draft':False,'updatable':False,'sp':False}
class Graph:
 def __init__(self):
  self.runtime_mode=Mode.FULL;self.enable_enpu=False;self.attn_backend=SFA;self.is_debugging_mode=False;self.syncs=0;self.replays=0;self.output=object();self.fail=False
  def replay():
   self.replays+=1
   if self.fail:raise RuntimeError('injected replay failure')
  self.graph=types.SimpleNamespace(tasks=[],replay=replay);self.concrete_aclgraph_entries={'key':types.SimpleNamespace(aclgraph=self.graph,output=self.output,input_addresses=[42])}
 def __call__(self,*args,**kwargs):
  e=self.concrete_aclgraph_entries.get(ctx.batch_descriptor)
  if ctx.capturing or e is None or e.aclgraph is None:return self.output
  if self.runtime_mode==Mode.FULL and not self.enable_enpu and not flags['draft']:self.syncs+=1
  e.aclgraph.replay();return e.output
class Runner:
 def __init__(self):
  self.input_batch=types.SimpleNamespace(req_ids=['r']);self.requests={'r':types.SimpleNamespace(sampling_params=types.SimpleNamespace(max_tokens=64))};self.parallel_config=types.SimpleNamespace(tensor_parallel_size=16,data_parallel_size=1,pipeline_parallel_size=1,decode_context_parallel_size=1,prefill_context_parallel_size=1)
  self.is_kv_consumer=True;self.is_kv_producer=False;self.use_async_scheduling=True;self.num_spec_tokens=5;self.model_config=types.SimpleNamespace(use_mla=True,is_hybrid=False);self.tp_rank=0;self.model=Graph()
 def _model_forward(self,*args,**kwargs):return self.model(*args,**kwargs)
spec=importlib.util.spec_from_file_location('order','glm5-3-vllm/runtime-patches/p-first-seed-v2/sfa_replay_order.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);os.environ['GLM53_P_FIRST_DRAFT_EXPORT']='0';rows=[]
with tempfile.TemporaryDirectory() as temp:
 root=pathlib.Path(temp);mod.install({'NPUModelRunner':Runner,'CUDAGraphMode':Mode,'enable_sp':lambda:flags['sp'],'torch':types.SimpleNamespace(Tensor=Tensor)},root,{'Graph':Graph,'SFA':SFA,'get_context':lambda:ctx,'is_draft':lambda:flags['draft'],'updatable':lambda backend:flags['updatable'],'logger':types.SimpleNamespace(info_once=lambda text:None)})
 cases=[('control',None,False,1),('selected',None,True,0),('DP2',lambda x:setattr(x.parallel_config,'data_parallel_size',2),True,1),('TP8',lambda x:setattr(x.parallel_config,'tensor_parallel_size',8),True,1),('PP2',lambda x:setattr(x.parallel_config,'pipeline_parallel_size',2),True,1),('DCP2',lambda x:setattr(x.parallel_config,'decode_context_parallel_size',2),True,1),('PCP2',lambda x:setattr(x.parallel_config,'prefill_context_parallel_size',2),True,1),('producer',lambda x:setattr(x,'is_kv_producer',True),True,1),('sync_scheduler',lambda x:setattr(x,'use_async_scheduling',False),True,1),('hybrid',lambda x:setattr(x.model_config,'is_hybrid',True),True,1),('other_backend',lambda x:setattr(x.model,'attn_backend',Other),True,1),('updatable_backend',lambda x:flags.update(updatable=True),True,1),('host_update_tasks',lambda x:x.model.graph.tasks.append('task'),True,1),('device_metadata_executor',lambda x:setattr(ctx,'device_metadata_executor',object()),True,1),('SP',lambda x:flags.update(sp=True),True,1),('nativeENPU',lambda x:setattr(x.model,'enable_enpu',True),True,0),('FULL_capture',lambda x:setattr(ctx,'capturing',True),True,0),('NONE_context',lambda x:setattr(ctx,'cudagraph_runtime_mode',Mode.NONE),True,1),('PIECEWISE_wrapper',lambda x:setattr(x.model,'runtime_mode',Mode.PIECEWISE),True,0),('draft',lambda x:flags.update(draft=True),True,0),('whole_max1_task_declined',lambda x:setattr(x.requests['r'].sampling_params,'max_tokens',1),True,1),('missing_entry',lambda x:x.model.concrete_aclgraph_entries.clear(),True,0),('uncaptured_entry',lambda x:setattr(x.model.concrete_aclgraph_entries['key'],'aclgraph',None),True,0),('warmup',lambda x:x.input_batch.req_ids.clear(),True,1),('debug_addresses_preserved',lambda x:setattr(x.model,'is_debugging_mode',True),True,0)]
 for name,change,enabled,syncs in cases:
  ctx.cudagraph_runtime_mode=Mode.FULL;ctx.capturing=False;ctx.device_metadata_executor=None;flags.update(draft=False,updatable=False,sp=False);obj=Runner()
  if change:change(obj)
  if enabled:(root/'enable').touch()
  else:(root/'enable').unlink(missing_ok=True)
  result=obj._model_forward(Tensor());assert result is obj.model.output;assert obj.model.syncs==syncs,(name,obj.model.syncs,syncs);assert obj.model.enable_enpu==(name=='nativeENPU');rows.append({'case':name,'native_host_waits':obj.model.syncs,'replays':obj.model.replays,'same_output_handle':True})
 # A failed selected replay must reset per-call state; later direct calls remain native.
 ctx.cudagraph_runtime_mode=Mode.FULL;ctx.capturing=False;ctx.device_metadata_executor=None;flags.update(draft=False,updatable=False,sp=False);obj=Runner();obj.model.fail=True
 try:obj._model_forward();raise AssertionError('fault not raised')
 except RuntimeError:pass
 obj.model.fail=False;obj.model();assert obj.model.syncs==1
 rows.append({'case':'exception_resets_context','passed':True})
report={'passed':True,'cases':rows,'scope':'CPU guards/native replay identity and ContextVar cleanup; fake graph, no device correctness/performance proof.'};pathlib.Path('glm5-3-vllm/evidence/20261018-p-first-seed-device-v2/recovery-v2/sfa-replay-order-cpu-check-v1.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'passed':True,'cases':len(rows)}))
