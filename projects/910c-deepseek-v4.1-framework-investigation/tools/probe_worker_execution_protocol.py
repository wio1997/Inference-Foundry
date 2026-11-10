#!/usr/bin/env python3
"""Run actual worker Python protocol bodies with explicit non-device callbacks.

This controls ordering/ownership, not numerical operations, native events or bits.
"""
import ast,copy,json,itertools
from pathlib import Path
from hashlib import sha256
from contextlib import contextmanager
from functools import partial
from types import SimpleNamespace as NS,MethodType
ROOT=Path('audit/execution-search')
def main():
 deps=json.loads((ROOT/'worker-protocol-dependencies.json').read_text());sources=deps['sources'];excerpts=[];events=[];ctx=NS()
 for s in sources:assert sha256(Path(s['local_path']).read_bytes()).hexdigest()==s['runtime_sha256']
 def method(suffix,owner,name,g):
  s=next(s for s in sources if s['runtime_path'].endswith(suffix));p=Path(s['local_path']);T=ast.parse(p.read_text());c=next(n for n in T.body if isinstance(n,ast.ClassDef) and n.name==owner);n=copy.deepcopy(next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name==name));n.decorator_list=[];excerpts.append(dict(runtime_path=s['runtime_path'],sha256=s['runtime_sha256'],owner=owner,method=name,first=n.lineno,last=n.end_lineno,lines=p.read_text().splitlines()[n.lineno-1:n.end_lineno]));mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),n],type_ignores=[]);exec(compile(ast.fix_missing_locations(mod),str(p),'exec'),g);return g[name]
 @contextmanager
 def scope(label,result=None):
  events.append('enter:'+label)
  try:yield result
  finally:events.append('exit:'+label)
 @contextmanager
 def forward_context(*a,**k):
  ctx.cudagraph_runtime_mode=k['aclgraph_runtime_mode'];ctx.capturing=False;ctx.device_metadata_executor=k['device_metadata_executor']
  with scope('forward_context'):yield
 class Arr(list):
  def __getitem__(self,i):
   if isinstance(i,list):return Arr([list.__getitem__(self,j) for j in i])
   value=list.__getitem__(self,i);return Arr(value) if isinstance(i,slice) else value
  def max(self):return max(self)
 class Tensor:
  def __getitem__(self,key):events.append('select_logits_rows');return self
 def repeat(xs,ns):return Arr([x for x,n in zip(xs,ns) for _ in range(n)])
 def add(a,b,*,out):events.append('CPU_position_shadow');out[:]=[x+y for x,y in zip(a,b)]
 class Event:
  def synchronize(self):events.append('prior_CPU_copy_lease_wait')
  def record(self):events.append('CPU_copy_lease_record')
 class Model:
  def prepare_engram_inputs(self,*a,**k):
   events.append('Engram_prepare')
   if self.error=='engram':raise ValueError('explicit opaque Engram error')
   return {}
  def set_attn_backend(self,*a):events.append('set_backend')
  def __call__(self,**k):
   events.append('model_call')
   if self.error=='model':raise ValueError('explicit opaque model error')
   return Tensor()
  def compute_logits(self,h):events.append('compute_logits');return Tensor()
 g=dict(torch=NS(npu=NS(is_current_stream_capturing=lambda:False)),get_forward_context=lambda:ctx,CUDAGraphMode=NS(FULL='FULL'),partial=partial,_start_profiling_chunk_timing=lambda *a:None,_finish_profiling_chunk_timing=lambda *a:None,get_pp_group=lambda:NS(world_size=1,is_last_rank=True),has_kv_transfer_group=lambda:False,has_ec_transfer=lambda:False,record_function_or_nullcontext=lambda n:scope(n),EMPTY_MODEL_RUNNER_OUTPUT='EMPTY',np=NS(array=lambda xs,**kw:Arr(xs),int32='int32',repeat=repeat,add=add),logger=NS(isEnabledFor=lambda level:False),logging=NS(DEBUG='DEBUG'),maybe_create_ubatch_slices=lambda *a:(None,None),enable_sp=lambda:True,update_cos_sin=lambda p:events.append('update_cos_sin'),set_ascend_forward_context=forward_context,ExecuteModelState=lambda *a:tuple(a))
 execute=method('/worker/model_runner_v1.py','NPUModelRunner','execute_model',g);forward=method('/worker/model_runner_v1.py','NPUModelRunner','_model_forward',g);prepare=method('/worker/model_runner_v1.py','NPUModelRunner','_prepare_device_metadata_for_forward',g)
 guard=method('/worker/gpu_model_runner.py','GPUModelRunner','synchronize_input_prep',g);guard=contextmanager(guard)
 cases=[]
 specs=[dict(label='normal_'+mode+'_async'+str(async_),mode=mode,async_=async_) for mode,async_ in itertools.product(['NONE','FULL'],[False,True])]
 specs += [dict(label='pending_state',pending=True),dict(label='empty_mp',total=0),dict(label='empty_external',total=0,backend='external_launcher'),dict(label='empty_effective',empty_effective=True),dict(label='noncompress_tail_counterfactual',compress=False,async_=True),dict(label='preprocess_error',error='preprocess'),dict(label='Engram_prepare_error',error='engram'),dict(label='model_error',error='model'),dict(label='FULL_without_external_events_negative',mode='FULL',external=False)]
 for spec in specs:
  events.clear();mode=spec.get('mode','NONE');async_=spec.get('async_',True);compress=spec.get('compress',True);error=spec.get('error');pending=spec.get('pending',False);total=spec.get('total',12);model=Model();model.error=error
  def release():events.append('metadata_release');executor.submission_in_flight=False
  executor=NS(submission_in_flight=True,uses_external_events=spec.get('external',True),release=release)
  count=0 if spec.get('empty_effective') else 2;batch=NS(prev_req_id_to_index={'b':0},num_reqs=count,req_ids=['a','b'][:count],num_computed_tokens_cpu=Arr([100,200]))
  states={'a':NS(prev_num_draft_len=5),'b':NS(prev_num_draft_len=5)}
  def correction():events.append('CPU_accept_count_wait_and_repair');batch.num_computed_tokens_cpu[0]-=4
  def update(s):events.append('update_states');return correction if async_ else None
  def preproc(*a):
   events.append('preprocess')
   if error=='preprocess':raise ValueError('explicit opaque preprocess error')
   return 'ids',None,'positions',None,{},None
  b=NS(_cpp_execution_time_ms=None,ascend_config=NS(scheduler_config=NS(profiling_chunk_config=object())),execute_model_state='PENDING' if pending else None,speculative_config=NS(use_ngram_gpu=lambda:False),use_async_scheduling=async_,num_spec_tokens=5,input_batch=batch,requests=states,_update_states=update,parallel_config=NS(distributed_executor_backend=spec.get('backend','mp'),data_parallel_size=2,enable_dbo=False,num_ubatches=1),_dummy_run=lambda *a,**k:events.append('external_empty_dummy'),cache_config=NS(kv_sharing_fast_prefill=False,mamba_cache_mode='none'),num_prompt_logprobs={},_start_dump_data=lambda **kw:events.append('start_dump'),_prepare_inputs=lambda *a:(Tensor(),'specmeta',total),cascade_attn_enabled=False,model_config=NS(enforce_eager=False,use_mla=True,is_encoder_decoder=False,requires_raw_input_tokens=False),_determine_batch_execution_and_padding=lambda **k:(mode,NS(num_tokens=total,num_reqs=2),False,Arr([total,total]),'graphstats'),dynamic_eplb=False,use_compress=compress,arange_np=Arr(range(24)),query_pos=NS(np=Arr(range(24))),_dsa_positions_np_buf=Arr([0]*24),_pad_query_start_loc_for_fia=lambda *a:events.append('FIA_pad') or 2,query_start_loc=object(),_build_attention_metadata=lambda **kw:(events.append('build_attention') or {'target':object()},'specCM'),_sanitize_placeholder_input_ids_for_forward=lambda *a:events.append('sanitize_IDs'),_preprocess=preproc,kvpp=NS(scheduler=None,complete_forward=lambda:events.append('kvpp_complete')),broadcast_pp_output=False,is_pooling_model=False,use_aux_hidden_state_outputs=False,device_metadata_executor=executor,model=model,attn_backend='DSA',enable_enpu=False,_get_engram_device_inputs=lambda:{},_update_full_graph_params_if_needed=lambda *a:events.append('legacy_graph_guard'),maybe_get_kv_connector_output=lambda *a,**k:scope('kv_connector',None),prepare_inputs_event=Event())
  b.vllm_config=object();b._has_sinks=False
  b.synchronize_input_prep=MethodType(guard,b);b._model_forward=MethodType(forward,b);b._prepare_device_metadata_for_forward=MethodType(prepare,b)
  sched=NS(total_num_scheduled_tokens=total,scheduled_cached_reqs=NS(req_ids=['a','b']),num_scheduled_tokens={'a':6,'b':6},scheduled_spec_decode_tokens={'a':[1,2,3,4,5]},scheduled_encoder_inputs={})
  result=None;raised=None
  try:result=execute(b,sched)
  except (ValueError,RuntimeError) as e:raised=type(e).__name__
  if pending:assert raised=='RuntimeError' and events==[]
  elif not total or spec.get('empty_effective'):
   assert result=='EMPTY' and not raised and b.execute_model_state is None and 'CPU_copy_lease_record' in events and 'model_call' not in events
   assert ('external_empty_dummy' in events)==(spec.get('backend')=='external_launcher' and not total)
  elif not spec.get('external',True):assert raised=='RuntimeError' and 'model_call' not in events and executor.submission_in_flight
  elif error:
   assert raised=='ValueError'
   assert ('metadata_release' in events)==(error=='model')
   assert 'kvpp_complete' not in events
  else:
   assert result is None and len(b.execute_model_state)==12 and events.count('metadata_release')==1
   if async_:
    assert states['a'].prev_num_draft_len==0 and states['b'].prev_num_draft_len==5
    assert events.count('CPU_accept_count_wait_and_repair')==1
    if compress:assert events.index('CPU_accept_count_wait_and_repair')<events.index('CPU_position_shadow')<events.index('build_attention')
    else:assert events.index('CPU_accept_count_wait_and_repair')>events.index('compute_logits')
  cases.append(dict(label=spec['label'],spec=spec,result='None' if result is None else result,raised=raised,events=list(events),state_pending_after=b.execute_model_state is not None,metadata_in_flight_after=executor.submission_in_flight,qualification='Original execute/_model_forward/pre-forward guard/CPU lease bodies; opaque model, events, arrays and nested callbacks. No Torch/NPU/native values or runtime branch population.'))
 out=dict(status='PASS_BOUNDED_ORIGINAL_WORKER_PROTOCOL_WITH_OPAQUE_CALLBACKS',count=len(cases),cases=cases,source_pins=len(sources),whole_chain_complete=False,new_primitive_admissions=0,new_framework_replacement_admissions=0,new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[],reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
 (ROOT/'worker-execution-protocol-controls.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');(ROOT/'worker-execution-protocol-source-excerpts.json').write_text(json.dumps(dict(qualification=cases[0]['qualification'],methods=excerpts),ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(cases=len(cases),new_admissions=0,new_NPU_work=0)))
if __name__=='__main__':main()
