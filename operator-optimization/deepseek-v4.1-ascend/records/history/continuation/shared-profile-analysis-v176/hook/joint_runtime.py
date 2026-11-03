"""Native PyTorch ET tensor/storage identities plus bounded CPU control metadata.

No optimization,Tensor values from the model,extra model device kernels or
Python TorchDispatch replacement. The16device interface selftest passed in v175; this run performs no new selftest.
No process-wide resource limit is changed. ET does not establish physical lifetime/alias independence.
"""
import traceback,contextlib,dataclasses,enum,functools,hashlib,inspect,json,os,resource,threading,time,uuid,weakref
from pathlib import Path
ROOT=Path(os.environ['WIO_META_ROOT'])
PINNED={
 'vllm_ascend.worker.model_runner_v1':'568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c',
 'vllm_ascend.worker.worker':'51cfb3da18046af8c72ccefe940de5f24d9502bfdae75b941e1798c0a165cda4',
 'vllm_ascend.compilation.updatable_graph':'d41d912ce8f4183228fdc182dbdb6a504f07642fddb8872c657e7f4429ac0729',
 'vllm_ascend.sample.rejection_sampler':'bb1d39de851540c16c2dd2b16df60c60a1fc4957cedb88bdce9417efc0d55eda',
 'vllm_ascend.spec_decode.llm_base_proposer':'458fb50e6839f3a353ddc7181ace4fdae02e35093d8ccd1c4327dc24ae5fff31',
 'vllm_ascend.worker.device_metadata':'efe13f0e801b70d7fba7498ac44062a8eccfd78e5b6303cf658d33dd22217580',
}
_state=None;_installed=set();_graph_hooks=False
def sensitive(name):
 k=str(name).lower()
 return k in {'hf_token','api_key','access_token','auth_token','password','secret','credentials'} or k.endswith(('_api_key','_password','_secret','_access_token'))
class State:
 def __init__(self):
  self.pid=os.getpid();self.uid=uuid.uuid4().hex;self.path=ROOT/'workers'/(str(self.pid)+'-'+self.uid);self.path.mkdir(parents=True,exist_ok=True)
  self.log=(self.path/'bindings.jsonl').open('a',buffering=1);self.seq=0;self.et=None;self.armed=False;self.closed=False;self.worker_step=0;self.runner=None;self.lock=threading.RLock();self.catalogued=False;self.errors={};self.npu_selftest_done=False
  self.emit('process_header',native_ET=True,model_tensor_values_read=False,physical_retirement_not_inferred_from_ET_ids=True)
 def emit(self,kind,**data):
  with self.lock:
   self.seq+=1
   if self.seq>1048576 or self.log.tell()>128*1024**2:raise RuntimeError('Bounded metadata sidecar exceeded')
   self.log.write(json.dumps({'kind':kind,'pid':self.pid,'process_uuid':self.uid,'seq':self.seq,'worker_step':self.worker_step,'host_monotonic_ns':time.monotonic_ns(),**data},ensure_ascii=False)+'\n')
 def ensure(self):
  import torch
  if self.et is None:
   self.et=torch.profiler.ExecutionTraceObserver().register_callback(str(self.path/'execution-trace.json'))
   assert self.et.is_registered,'Native ET registration failed'
 def running(self):return bool(self.et and self.et.is_running())
 def start(self):self.ensure();self.et.start()
 def stop(self):
  if self.et:self.et.stop()
 def check_budget(self):
  f=self.path/'execution-trace.json'
  if f.exists() and f.stat().st_size>768*1024**2:raise RuntimeError('Native ET 768MiB stop threshold exceeded;no automatic retry')
 def tensor_meta(self,t):
  m={'python_id':id(t),'tensor_impl':getattr(t,'_cdata',None),'dtype':str(t.dtype),'device':str(t.device),'shape':list(t.shape),'stride':list(t.stride()),'storage_offset_elements':int(t.storage_offset()),'numel':int(t.numel()),'itemsize':int(t.element_size())}
  try:
   st=t.untyped_storage();m.update(storage_impl=getattr(st,'_cdata',None),storage_base_address=st.data_ptr(),storage_nbytes=st.nbytes(),tensor_data_address=t.data_ptr())
  except Exception as e:m['storage_metadata_error']=type(e).__name__
  try:m['version']=t._version
  except RuntimeError:m['version']=None
  return m
 def flatten(self,obj,detail=False):
  import torch
  tensors=[];paths=[];controls={};seen=set();omitted=[]
  def walk(v,path,depth):
   if isinstance(v,torch.Tensor):
    if len(tensors)>=128:omitted.append(path+':tensor_cap');return
    tensors.append(v);paths.append(path)
    if detail:controls[path]=self.tensor_meta(v)
    return
   if v is None or type(v) in (bool,int,float):controls[path]=v;return
   if isinstance(v,str):controls[path]=v if len(v)<=256 else {'string_length':len(v),'sha256':hashlib.sha256(v.encode()).hexdigest()};return
   if isinstance(v,(torch.dtype,torch.device)):controls[path]=str(v);return
   if isinstance(v,enum.Enum):controls[path]={'enum':type(v).__qualname__,'name':v.name};return
   if callable(v) and not isinstance(v,torch.nn.Module):controls[path]={'callable':getattr(v,'__module__','')+'.'+getattr(v,'__qualname__',type(v).__qualname__)};return
   if id(v) in seen:controls[path]={'object_ref':id(v)};return
   seen.add(id(v))
   if depth>6:controls[path]={'type':type(v).__module__+'.'+type(v).__qualname__,'depth_limit':True};return
   if isinstance(v,(tuple,list)):
    if len(v)>64:controls[path]={'type':type(v).__name__,'length':len(v),'bulk_not_dumped':True};return
    for i,z in enumerate(v):walk(z,path+'['+str(i)+']',depth+1)
    return
   if isinstance(v,dict):
    if len(v)>64:controls[path]={'type':'dict','length':len(v),'bulk_not_dumped':True};return
    for k,z in v.items():
     if sensitive(k):controls[path+'.'+str(k)]='REDACTED';continue
     walk(z,path+'.'+str(k),depth+1)
    return
   if isinstance(v,torch.nn.Module):controls[path]={'module':type(v).__module__+'.'+type(v).__qualname__,'python_id':id(v)};return
   # Do not execute arbitrary properties or tensor scalar conversion. Objects
   # with explicit CPU fields expose controls and Tensor bank references only.
   attrs=getattr(v,'__dict__',None)
   if dataclasses.is_dataclass(v):attrs={f.name:getattr(v,f.name) for f in dataclasses.fields(v)}
   controls[path+'@type']=type(v).__module__+'.'+type(v).__qualname__
   if isinstance(attrs,dict):
    for k,z in list(attrs.items())[:96]:
     if sensitive(k):controls[path+'.'+k]='REDACTED';continue
     if k.startswith('__') or k in ['_parameters','_modules','_buffers','model','model_runner','requests','encoder_cache','tmp_encoder_cache','cpu_encoder_cache']:continue
     walk(z,path+'.'+k,depth+1)
    if len(attrs)>96:omitted.append(path+':attribute_cap')
  walk(obj,'root',0)
  return tensors,paths,controls,omitted
 @contextlib.contextmanager
 def bind(self,name,values,detail=False):
  import torch
  if not self.running():yield;return
  if self.seq%256==0:self.check_budget()
  ts,paths,controls,omitted=self.flatten(values,detail)
  self.emit('binding',name=name,tensor_paths=paths,controls=controls,omitted=omitted)
  handle=torch.autograd._record_function_with_args_enter('wio176/'+name,self.seq,json.dumps(paths,separators=(',',':')),*ts)
  try:yield
  finally:torch.autograd._record_function_with_args_exit(handle)
 def catalogue(self,runner,phase):
  import torch
  self.runner=weakref.ref(runner)
  roots=[('target',runner.get_model())]
  drafter=getattr(runner,'drafter',None);dm=getattr(drafter,'model',None)
  if isinstance(dm,torch.nn.Module):roots.append(('draft',dm))
  for rootname,model in roots:
   names=[];batch=[]
   def flush():
    if batch:
     with self.bind('catalogue/'+phase+'/'+rootname+'/'+str(len(names)-len(batch)),dict(zip(names[-len(batch):],batch)),detail=True):pass
     batch.clear()
   for kind,iterator in [('parameter',model.named_parameters(remove_duplicate=False)),('buffer',model.named_buffers(remove_duplicate=False))]:
    for name,t in iterator:
     names.append(kind+':'+name);batch.append(t)
     if len(batch)==32:flush()
    flush()
   for name,module in model.named_modules():
    small={k:v for k,v in vars(module).items() if (type(v) in (bool,int,float,str) or v is None or isinstance(v,(torch.dtype,torch.device,enum.Enum))) and not k.startswith('__') and not sensitive(k)}
    small={k:(str(v) if isinstance(v,(torch.dtype,torch.device,enum.Enum)) else v) for k,v in small.items()}
    quant=getattr(module,'quant_method',None);impl=getattr(module,'impl',None)
    hooks={k:len(v) for k,v in vars(module).items() if k in ['_forward_hooks','_forward_pre_hooks','_backward_hooks'] and isinstance(v,dict)}
    self.emit('module_identity',root=rootname,name=name,python_id=id(module),qualified_class=type(module).__module__+'.'+type(module).__qualname__,scalar_attributes=small,hook_counts=hooks,quant_method=None if quant is None else type(quant).__module__+'.'+type(quant).__qualname__,impl_class=None if impl is None else type(impl).__module__+'.'+type(impl).__qualname__)
    def has_tensor(v,depth=0):
     if isinstance(v,torch.Tensor):return True
     if depth>=3:return False
     if isinstance(v,(tuple,list)) and len(v)<=64:return any(has_tensor(z,depth+1) for z in v)
     if isinstance(v,dict) and len(v)<=64:return any(has_tensor(z,depth+1) for z in v.values())
     return False
    for key,value in vars(module).items():
     if key in ['_parameters','_buffers','_modules'] or sensitive(key):continue
     if has_tensor(value):
      with self.bind('module_bank/'+phase+'/'+rootname+'/'+name+'/'+key,{key:value},True):pass
    for label,aux in [('quant_method',quant),('impl',impl)]:
     if aux is None or isinstance(aux,torch.nn.Module):continue
     fields=getattr(aux,'__dict__',{})
     simple={k:v for k,v in fields.items() if not sensitive(k) and (type(v) in (bool,int,float,str) or v is None)}
     self.emit('auxiliary_operator_identity',root=rootname,module_name=name,field=label,qualified_class=type(aux).__module__+'.'+type(aux).__qualname__,scalar_attributes=simple)
     for key,value in fields.items():
      if not sensitive(key) and has_tensor(value):
       with self.bind('aux_bank/'+phase+'/'+rootname+'/'+name+'/'+label+'/'+key,{key:value},True):pass
  # Persistent banks and explicit CPU selector fields; never model tensor data.
  bank={k:v for k,v in vars(runner).items() if k not in ['model','drafter','requests','encoder_cache','tmp_encoder_cache','cpu_encoder_cache'] and (isinstance(v,torch.Tensor) or k in ['input_batch','positions','query_start_loc','seq_lens','num_computed_tokens','num_accepted_tokens','prev_positions','num_decode_draft_tokens','device_metadata_executor','ascend_config','vllm_config'])}
  # Split fields so a large catalogue cannot silently drop an unrelated bank.
  for k,v in bank.items():
   with self.bind('runner_bank/'+phase+'/'+k,{k:v},detail=True):pass
  self.catalogued=True
 def finish(self):
  if self.closed:return
  self.stop()
  if self.et:self.et.unregister_callback()
  self.closed=True;self.emit('footer',native_ET_closed=True,npu_selftest_done=self.npu_selftest_done,errors=self.errors)
  (self.path/'FINAL.json').write_text(json.dumps({'pid':self.pid,'process_uuid':self.uid,'events':self.seq,'worker_steps':self.worker_step,'catalogued':self.catalogued,'npu_selftest_done':self.npu_selftest_done,'execution_trace':str(self.path/'execution-trace.json'),'execution_trace_bytes':(self.path/'execution-trace.json').stat().st_size if (self.path/'execution-trace.json').exists() else None,'errors':self.errors},indent=2)+'\n')

def state():
 global _state
 if _state is None or _state.pid!=os.getpid():_state=State()
 return _state

def selftest_npu(worker):
 import torch,torch_npu,gc
 s=state();p=s.path/'tiny-npu-selftest.json';e=torch.profiler.ExecutionTraceObserver().register_callback(str(p));e.start()
 a=torch.arange(16,dtype=torch.float32,device=worker.device).view(4,4);b=a[:,1:];z=a+1
 h=torch.autograd._record_function_with_args_enter('wio176/tiny-alias-inputs',a,b,z);torch.autograd._record_function_with_args_exit(h);del h
 b.add_(2);torch_npu.npu.synchronize()
 golden=torch.arange(16,dtype=torch.float32).view(4,4);expected=golden.clone();expected[:,1:]+=2
 correct=bool(torch.equal(z.cpu(),golden+1) and torch.equal(a.cpu(),expected))
 raw_alias_meta=[s.tensor_meta(t) for t in [a,b,z]]
 weak_storage=[t.untyped_storage()._weak_ref() for t in [a,z]]
 del a,b,z;gc.collect();torch_npu.npu.synchronize()
 nonowning=all(torch.UntypedStorage._expired(w) for w in weak_storage)
 for w in weak_storage:torch.UntypedStorage._free_weak_ref(w)
 e.stop();e.unregister_callback();d=json.loads(p.read_text());alias=[];devices=set()
 for n in d['nodes']:
  vals=n.get('inputs',{}).get('values',[])+n.get('outputs',{}).get('values',[])
  for v in vals:
   if isinstance(v,list) and len(v)==6 and isinstance(v[-1],str) and v[-1].startswith(('npu','privateuseone')):
    devices.add(v[-1]);alias.append(v)
 assert correct and nonowning and devices and len({x[1] for x in alias})>=2,'NativeETNPUtensor/storage selftestfailed'
 s.npu_selftest_done=True;s.emit('tiny_npu_selftest',correct=correct,nonowning_NPU_storage=nonowning,device=str(worker.device),tensor_records=len(alias),native_ET_schema=d.get('schema'),model_loaded=False,base_input_elements=16,model_tensor_values_read=False,alias_inputs_raw_metadata=raw_alias_meta)
 torch_npu.npu.empty_cache()

def observe_method(cls,name,detail=False):
 old=getattr(cls,name,None)
 if old is None or getattr(old,'_wio176',False):return
 @functools.wraps(old)
 def wrapped(obj,*a,**k):
  import torch
  s=state()
  with torch.profiler.record_function('wio176/scope/'+type(obj).__qualname__+'.'+name) if s.running() else contextlib.nullcontext():
   with s.bind(type(obj).__qualname__+'.'+name+'/entry',{'args':a,'kwargs':k,'self_scalars':{q:v for q,v in vars(obj).items() if type(v) in (bool,int,float,str) or v is None}},detail):pass
   result=old(obj,*a,**k)
   after={q:v for q,v in vars(obj).items() if type(v) in (bool,int,float,str) or v is None or isinstance(v,enum.Enum)}
   with s.bind(type(obj).__qualname__+'.'+name+'/return',{'result':result,'self_scalars_after':after},detail):pass
   return result
 wrapped._wio176=True;setattr(cls,name,wrapped)

def install(module):
 global _graph_hooks
 name=module.__name__
 if name in _installed:return
 actual=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest();assert actual==PINNED[name],(name,actual)
 s=state();s.emit('module_source_bound',module=name,sha256=actual)
 if name=='vllm_ascend.worker.worker':
  cls=module.NPUWorker;old_init=cls.init_device;old_profile=cls.profile
  @functools.wraps(old_init)
  def init_device(obj,*a,**k):
   result=old_init(obj,*a,**k);state().emit("predecessor_NPU_interface_selftest_reused",cohort="joint175",devices_passed=16,no_new_NPU_selftest=True)
   identities={'rank':getattr(obj,'rank',None),'local_rank':getattr(obj,'local_rank',None),'device':str(obj.device)}
   try:
    from vllm.distributed import get_ep_group,get_dp_group,get_tp_group
    for label,group in [('EP',get_ep_group()),('DP',get_dp_group()),('TP',get_tp_group())]:identities[label]={'rank_in_group':int(group.rank_in_group),'world_size':int(group.world_size),'ranks':list(getattr(group,'ranks',[]))}
   except Exception as exc:identities['group_query_error']=type(exc).__name__
   state().emit('worker_identity',**identities);return result
  @functools.wraps(old_profile)
  def profile(obj,is_start=True,*a,**k):
   st=state()
   if is_start:
    st.start();st.armed=True
    runner=getattr(obj,'model_runner',None) or (st.runner() if st.runner else None)
    if runner is not None:st.catalogue(runner,'formal_start')
    st.emit('formal_start',native_ET=True,NPU_profiler=False,no_extra_model_device_readers=True)
   else:st.armed=False;st.finish()
   # Existing endpoint is the collection control. NativeET supplies the new
   # metadata;old independent PMU traces remain the performance evidence.
   return None
  cls.init_device=init_device;cls.profile=profile
  old_load=cls.load_model
  @functools.wraps(old_load)
  def load(obj,*a,**k):
   state().emit('worker_load_model_enter')
   try:
    result=old_load(obj,*a,**k);state().emit('worker_load_model_return');return result
   except BaseException:
    state().emit('worker_load_model_failure',traceback=traceback.format_exc());raise
  cls.load_model=load
 elif name=='vllm_ascend.worker.model_runner_v1':
  cls=module.NPUModelRunner;old_capture=cls.capture_model;old_execute=cls.execute_model
  @functools.wraps(old_capture)
  def capture(obj,*a,**k):
   import torch
   st=state();st.start();st.catalogue(obj,'capture')
   try:
    with torch.profiler.record_function('wio176/scope/capture_model'):
     with st.bind('capture_model',{'args':a,'kwargs':k},True):pass
     return old_capture(obj,*a,**k)
   finally:st.stop()
  @functools.wraps(old_execute)
  def execute(obj,*a,**k):
   import torch
   st=state();st.runner=weakref.ref(obj)
   if not st.armed:return old_execute(obj,*a,**k)
   st.worker_step+=1;st.check_budget()
   batch=getattr(obj,'input_batch',None);sampling=getattr(batch,'sampling_metadata',None)
   with torch.profiler.record_function('wio176/scope/execute_model/'+str(st.worker_step)):
    with st.bind('execute_model',{'args':a,'kwargs':k,'sampling_metadata':sampling},True):pass
    return old_execute(obj,*a,**k)
  cls.capture_model=capture;cls.execute_model=execute
  for n in ['_dummy_run','_should_build_dummy_attn_metadata','_determine_batch_execution_and_padding','_build_attn_state','_sync_num_accepted_tokens','_correct_optimistic_seq_lens_cpu','_prepare_inputs','_sample','_calc_spec_decode_metadata','_copy_spec_decode_metadata_to_device','_prepare_device_metadata_for_forward','_build_attention_metadata','_sanitize_placeholder_input_ids_for_forward','_bookkeeping_sync','_skip_drafting','sample_tokens']:
   observe_method(cls,n,True)
 elif name=='vllm_ascend.compilation.updatable_graph':observe_method(module.UpdatableGraph,'update',True)
 elif name=='vllm_ascend.worker.device_metadata':
  for n in ['submit','release']:observe_method(module.DeviceMetadataExecutor,n,True)
 elif name=='vllm_ascend.spec_decode.llm_base_proposer':
  for n in ['_run_merged_draft','prepare_inputs_padded','prepare_next_token_ids_padded','_sample_draft_from_logits']:observe_method(module.AscendSpecDecodeBaseProposer,n,True)
 elif name=='vllm_ascend.sample.rejection_sampler':
  observe_method(module.AscendRejectionSampler,'forward',True)
  for fname,old in list(vars(module).items()):
   if not inspect.isfunction(old) or not any(x in fname for x in ['rejection_sample','rejection_greedy','apply_sampling_constraints']):continue
   def make(fn,label):
    @functools.wraps(fn)
    def wrapped(*a,**k):
     import torch
     st=state()
     with torch.profiler.record_function('wio176/scope/helper/'+label) if st.running() else contextlib.nullcontext():
      with st.bind('helper/'+label,{'args':a,'kwargs':k},True):pass
      return fn(*a,**k)
    return wrapped
   setattr(module,fname,make(old,fname))
 if name in ['vllm_ascend.worker.model_runner_v1','vllm_ascend.spec_decode.llm_base_proposer']:
  fn=getattr(module,'lmhead_tp_enable',None)
  if callable(fn):
   def make_predicate(original,module_name):
    @functools.wraps(original)
    def predicate(*a,**k):
     result=original(*a,**k);st=state()
     with st.bind('predicate/'+module_name+'.lmhead_tp_enable',{'result':result}):pass
     return result
    return predicate
   module.lmhead_tp_enable=make_predicate(fn,name)
 if not _graph_hooks and name=='vllm_ascend.worker.worker':
  import torch_npu
  cls=torch_npu.npu.NPUGraph
  for method in ['capture_begin','capture_end','replay','reset']:
   original=getattr(cls,method)
   def make(fn,label):
    @functools.wraps(fn)
    def wrapped(obj,*a,**k):
     import torch
     st=state()
     with torch.profiler.record_function('wio176/scope/NPUGraph.'+label) if st.running() else contextlib.nullcontext():
      with st.bind('NPUGraph.'+label,{'python_graph_id':id(obj),'args':a,'kwargs':k},True):pass
      return fn(obj,*a,**k)
    return wrapped
   setattr(cls,method,make(original,method))
  _graph_hooks=True
 _installed.add(name)
