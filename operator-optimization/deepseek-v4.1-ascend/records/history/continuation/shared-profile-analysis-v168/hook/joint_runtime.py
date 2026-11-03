"""Read-only MC2 diagnostic values,creator profiles and explicit replay epochs.

The observer adds device readers and work. Its timings are diagnostic only.
It does not establish baseline liveness or implement an optimization.
"""
import contextlib,functools,hashlib,json,os,time,uuid,resource
from pathlib import Path
ROOT=Path(os.environ['WIO_JOINT_ROOT'])
EPOCHS=4096;SLOTS=384;ID_SLOTS=32;WIDTH=26;ID_WIDTH=144
SENTINEL=-2147483648
_state=None;_installed=set();_original_profile=None;_graph_patched=False
PINNED={
 'vllm_ascend.ops.fused_moe.moe_comm_method':'7040f691355bd9acb3c668e698de3f7d3854edbc22e3f979f9b03d99800364d7',
 'vllm_ascend.worker.model_runner_v1':'568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c',
 'vllm_ascend.compilation.updatable_graph':'d41d912ce8f4183228fdc182dbdb6a504f07642fddb8872c657e7f4429ac0729',
 'vllm_ascend.worker.worker':'51cfb3da18046af8c72ccefe940de5f24d9502bfdae75b941e1798c0a165cda4'}
class State:
 def __init__(self):
  self.pid=os.getpid();self.uid=uuid.uuid4().hex;self.path=ROOT/'workers'/(str(self.pid)+'-'+self.uid)
  self.path.mkdir(parents=True,exist_ok=True);self.log=(self.path/'events.jsonl').open('a',buffering=1)
  self.events=0;self.prepared=False;self.armed=False;self.dumped=False;self.epoch=0;self.step=0;self.step_depth=0;self.step_epoch=0;self.eager_occ={}
  self.graphs={};self.graph_refs={};self.capture=None;self.replay_depth=0;self.replay_serial=0;self.banks=[];self.eager_banks={};self.id_slots=0;self.layers={};self.startup_profiles=0
  self.emit('process_created',application_arena_bytes=SLOTS*EPOCHS*WIDTH*4+ID_SLOTS*EPOCHS*ID_WIDTH*8+8+24*4+ID_WIDTH*8,
   runtime_overhead_additional=True,all_output_values_unchanged=True,diagnostic_readers_are_not_baseline_readers=True)
 def emit(self,kind,**data):
  self.events+=1
  if self.events>32768 or self.log.tell()>32*1024**2:raise RuntimeError('Diagnostic host metadata budget exceeded')
  self.log.write(json.dumps({'kind':kind,'pid':self.pid,'process_uuid':self.uid,'event':self.events,'host_monotonic_ns':time.monotonic_ns(),**data},ensure_ascii=False)+'\n')
 def prepare(self):
  if self.prepared:return
  import torch,torch_npu
  assert not torch_npu.npu.is_current_stream_capturing(),'History arenas must be allocated before graph capture'
  self.history=torch.full((SLOTS,EPOCHS,WIDTH),SENTINEL,dtype=torch.int32,device='npu')
  self.ids=torch.full((ID_SLOTS,EPOCHS,ID_WIDTH),-1,dtype=torch.int64,device='npu')
  self.index=torch.zeros((1,),dtype=torch.int64,device='npu')
  self.zeros=torch.zeros(24,dtype=torch.int32,device='npu');self.zero_ids=torch.zeros(ID_WIDTH,dtype=torch.int64,device='npu')
  torch_npu.npu.synchronize();self.prepared=True
  self.emit('arenas_prepared',device=str(self.index.device),history_shape=list(self.history.shape),ids_shape=list(self.ids.shape),allocated_torch_bytes=torch_npu.npu.memory_allocated(),reserved_torch_bytes=torch_npu.npu.memory_reserved())
 def graph(self,obj):
  key=id(obj)
  if key not in self.graphs:
   # A strong reference prevents Python object-id reuse during this cohort.
   self.graph_refs[key]=obj;self.graphs[key]={'token':len(self.graphs)+1,'capture_epoch':0,'site':0,'destroyed':False}
  return self.graphs[key]
 def next_epoch(self,kind,**data):
  self.epoch+=1
  slot=self.epoch if self.epoch<EPOCHS else 0
  self.index.fill_(slot)
  self.emit('epoch',epoch=self.epoch,history_slot=slot,kind_detail=kind,worker_step=self.step,overflow=self.epoch>=EPOCHS,**data)
  return slot
 def bank(self,inp,result,module,kind,graph=None,occurrence=0):
  import torch
  from vllm.distributed import get_ep_group,get_dp_group,get_tp_group
  from vllm_ascend.ascend_config import get_ascend_config
  x=inp.hidden_states;ids=inp.topk_ids;counts=result.expert_tokens;mask=inp.routing.mc2_mask;layer=inp.layer
  role='draft' if bool(module._EXTRA_CTX.is_draft_model) else 'target';name=layer.layer_name
  if role=='target' and name not in self.layers:self.layers[name]=len(self.layers)
  ep=get_ep_group();M=int(x.shape[0]);K=int(ids.shape[1]);E=int(counts.numel())
  assert counts.dtype==torch.int32 and counts.ndim==1 and 0<E<=24
  assert counts.device==ids.device==x.device and ids.ndim==2 and ids.shape[0]==M
  assert mask is None or (mask.device==x.device and mask.numel()>=M)
  assert 0<=M<=8192 and 0<K<=6
  if graph is not None:
   assert self.capture is not None and M<=24,(M,name,'Unexpected captured metadata shape')
   graph['site']+=1;key=('graph',graph['token'],graph['capture_epoch'],graph['site'])
  else:key=('eager',role,name,occurrence)
  if key in self.eager_banks:return self.eager_banks[key]
  slot=len(self.banks);assert slot<SLOTS,'History bank budget exceeded'
  idslot=None
  if role=='target' and self.layers.get(name) in (0,20) and M<=24:
   idslot=self.id_slots;self.id_slots+=1;assert idslot<ID_SLOTS,'ID history bank budget exceeded'
  meta={'slot':slot,'execution_kind':kind,'graph_token':None if graph is None else graph['token'],'capture_epoch':None if graph is None else graph['capture_epoch'],'capture_site':None if graph is None else graph['site'],
   'role':role,'layer_name':name,'target_layer_observation_order':self.layers.get(name),'E_local':E,'E_global':int(layer.global_num_experts),'M_at_creation':M,'K_at_creation':K,
   'ep_rank':int(ep.rank_in_group),'ep_world_size':int(ep.world_size),'ep_group_ranks':list(getattr(ep,'ranks',[])),'dp_rank':int(get_dp_group().rank_in_group),'tp_rank':int(get_tp_group().rank_in_group),
   'capacity_max_output_size':int(get_ascend_config().mega_moe_max_tokens),'group_name':str(getattr(inp,'_unused_group','')),
   'counts_data_ptr_at_creation':counts.data_ptr(),'counts_dtype':str(counts.dtype),'ids_dtype':str(ids.dtype),'mask_shape_at_creation':None if mask is None else list(mask.shape),'id_slot':idslot,
   'history_fields':'counts[0:E],valid_sender_rows[E],device_epoch[E+1],zero_padding[E+2:26]','valid_sender_rows_semantics':'sum first M mask entries;M if maskNone','new_observer_reader':True}
  self.banks.append(meta);self.eager_banks[key]=meta;self.emit('history_bank',**meta)
  return meta
 def observe(self,inp,result,module,bank):
  import torch
  counts=result.expert_tokens;E=int(counts.numel());M=int(inp.hidden_states.shape[0]);mask=inp.routing.mc2_mask
  valid=counts.new_full((1,),M) if mask is None else mask.reshape(-1)[:M].sum(dtype=torch.int32).reshape(1)
  payload=torch.cat((counts,valid,self.index.to(torch.int32),self.zeros[:24-E]))
  self.history[bank['slot']].scatter_(0,self.index.expand(1,WIDTH),payload.reshape(1,WIDTH))
  if bank['id_slot'] is not None:
   ids=inp.topk_ids.reshape(-1).to(torch.int64);assert ids.numel()<=ID_WIDTH
   packed=torch.cat((ids,self.zero_ids[:ID_WIDTH-ids.numel()]))
   self.ids[bank['id_slot']].scatter_(0,self.index.expand(1,ID_WIDTH),packed.reshape(1,ID_WIDTH))
 def dump(self):
  if self.dumped:return
  import torch,torch_npu
  torch_npu.npu.synchronize();self.armed=False;last=min(self.epoch,EPOCHS-1)
  if self.prepared:
   torch.save({'history':self.history[:len(self.banks),:last+1].cpu(),'ids':self.ids[:self.id_slots,:last+1].cpu()},self.path/'history.pt')
  meta={'schema':1,'process_uuid':self.uid,'pid':self.pid,'epochs':self.epoch,'history_last_index':last,'overflow':self.epoch>=EPOCHS,'banks':self.banks,'graphs':list(self.graphs.values()),'target_layer_order':self.layers,'inference_outputs_modified':False,'baseline_reader_liveness_not_implied':True}
  (self.path/'FINAL.json').write_text(json.dumps(meta,indent=2)+'\n');self.dumped=True
  self.emit('history_dumped',banks=len(self.banks),id_banks=self.id_slots,epochs=self.epoch,overflow=self.epoch>=EPOCHS)
def state():
 global _state
 if _state is None or _state.pid!=os.getpid():_state=State()
 return _state
@contextlib.contextmanager
def mark(name):
 import torch_npu
 r=torch_npu.npu.mstx.range_start(name)
 try:yield
 finally:torch_npu.npu.mstx.range_end(r)
def configure():
 global _original_profile,_graph_patched
 import torch_npu
 if _original_profile is not None:return
 _original_profile=torch_npu.profiler.profile;handler=torch_npu.profiler.tensorboard_trace_handler
 @functools.wraps(_original_profile)
 def formal_profile(*a,**k):
  if a:raise RuntimeError('Unexpected positional profiler factory')
  k.update(activities=[torch_npu.profiler.ProfilerActivity.NPU],record_shapes=False,profile_memory=False,with_stack=False,with_modules=False,
   experimental_config=torch_npu.profiler._ExperimentalConfig(profiler_level=torch_npu.profiler.ProfilerLevel.Level1,aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,mstx=True,export_type=[torch_npu.profiler.ExportType.Db]))
  resource.setrlimit(resource.RLIMIT_FSIZE,(1024**3,1024**3))
  state().emit('formal_profiler_constructed',activities=['NPU'],level='Level1',PMU=False)
  return _original_profile(**k)
 @functools.wraps(handler)
 def raw_handler(*a,**k):k['analyse_flag']=False;return handler(*a,**k)
 torch_npu.profiler.profile=formal_profile;torch_npu.profiler.tensorboard_trace_handler=raw_handler
 cls=torch_npu.npu.NPUGraph;old_begin=cls.capture_begin;old_end=cls.capture_end;old_replay=cls.replay;old_reset=cls.reset
 @functools.wraps(old_begin)
 def begin(obj,*a,**k):
  s=state();s.prepare();g=s.graph(obj);assert s.capture is None,'Nested diagnostic graph capture'
  g['capture_epoch']+=1;g['site']=0;g['destroyed']=False;s.capture=g
  s.emit('graph_capture_begin',**g)
  try:return old_begin(obj,*a,**k)
  except BaseException:s.capture=None;raise
 @functools.wraps(old_end)
 def end(obj,*a,**k):
  s=state();g=s.graph(obj)
  try:return old_end(obj,*a,**k)
  finally:s.emit('graph_capture_end',**g);s.capture=None
 @functools.wraps(old_replay)
 def replay(obj,*a,**k):
  s=state();g=s.graph(obj);assert not g['destroyed']
  root=s.replay_depth==0;s.replay_depth+=1
  epoch=s.next_epoch('graph_replay',graph_token=g['token'],capture_epoch=g['capture_epoch']) if root and s.armed else 0
  s.replay_serial+=1
  message='wio_joint168/'+s.uid+'/replay/'+str(g['token'])+'/'+str(g['capture_epoch'])+'/'+str(epoch)+'/r'+str(s.replay_serial)
  s.emit('replay_scope',marker=message,graph_token=g['token'],capture_epoch=g['capture_epoch'],history_epoch=epoch,armed=s.armed,root=root)
  try:
   with mark(message):return old_replay(obj,*a,**k)
  finally:s.replay_depth-=1
 @functools.wraps(old_reset)
 def reset(obj,*a,**k):
  s=state();g=s.graph(obj);s.emit('graph_reset',**g);g['destroyed']=True;return old_reset(obj,*a,**k)
 cls.capture_begin=begin;cls.capture_end=end;cls.replay=replay;cls.reset=reset
 _graph_patched=True

def install(module,label):
 if module.__name__ in _installed:return
 expected=PINNED[module.__name__];actual=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest();assert actual==expected,(module.__name__,actual)
 configure();s=state();s.emit('module_hook_installed',module=module.__name__,source_sha256=actual,label=label)
 if label=='moe':
  cls=module.FusedMC2CommImpl;original=cls.fused_experts
  @functools.wraps(original)
  def observed(comm,*a,**k):
   s=state();inp=k.get('fused_experts_input',a[0] if a else None);assert inp is not None
   import torch_npu
   capturing=bool(torch_npu.npu.is_current_stream_capturing())
   role='draft' if bool(module._EXTRA_CTX.is_draft_model) else 'target';name=inp.layer.layer_name
   if role=='target' and name not in s.layers:s.layers[name]=len(s.layers)
   if not capturing and not s.armed:return original(comm,*a,**k)
   assert s.prepared
   if not capturing:
    if s.step_depth:
     epoch=s.step_epoch;s.index.fill_(epoch);key=(role,name);occ=s.eager_occ.get(key,0);s.eager_occ[key]=occ+1
    else:epoch=s.next_epoch('eager_outside_worker_step');occ=0
   else:epoch=0;occ=0
   binding=('g'+str(s.capture['token'])+'/c'+str(s.capture['capture_epoch'])+'/s'+str(s.capture['site']+1)) if capturing else ('eager/e'+str(epoch)+'/occ'+str(occ))
   site_message='wio_joint168/'+s.uid+'/mc2/'+binding+'/'+role+'/'+name
   with mark(site_message):
    result=original(comm,*a,**k)
    assert not bool(module._EXTRA_CTX.use_mega_moe) and comm.enable_fused_mc2==1,'Unexpected MC2 implementation;not silently observed'
    bank=s.bank(inp,result,module,'captured' if capturing else 'eager',s.capture if capturing else None,occ)
    if 'source_marker_at_creation' not in bank:bank['source_marker_at_creation']=site_message
    # The actual operator group and source schema are captured without native pointer reads.
    if not bank['group_name']:
     bank['group_name']=str(comm.token_dispatcher.moe_all_to_all_group_name)
     s.emit('history_bank_group',slot=bank['slot'],group_name=bank['group_name'])
    s.observe(inp,result,module,bank)
    if not capturing:s.emit('eager_mc2_binding',epoch=epoch,history_slot=epoch,bank_slot=bank['slot'],M=int(inp.hidden_states.shape[0]),K=int(inp.topk_ids.shape[1]),worker_step=s.step,occurrence=occ,source_marker=site_message)
    return result
  cls.fused_experts=observed
 elif label=='updatable':
  cls=module.UpdatableGraph;old_update=cls.update
  @functools.wraps(old_update)
  def update(obj,*a,**k):
   s=state();g=s.graph(obj);g['update_calls']=g.get('update_calls',0)+1
   tasks=k.get('resolved_tasks',a[1] if len(a)>1 else ())
   s.emit('graph_update',graph_token=g['token'],capture_epoch=g['capture_epoch'],update_call=g['update_calls'],task_count=len(tasks),operations=[getattr(getattr(t,'operation',None),'__qualname__',str(type(getattr(t,'operation',None)))) for t in tasks[:64]],not_native_parameter_or_liveness_proof=True)
   with mark('wio_joint168/'+s.uid+'/update/'+str(g['token'])+'/'+str(g['capture_epoch'])+'/'+str(g['update_calls'])):return old_update(obj,*a,**k)
  cls.update=update
 elif label=='runner':
  cls=module.NPUModelRunner;old_capture=cls.capture_model;old_execute=cls.execute_model
  @functools.wraps(old_capture)
  def capture(obj,*a,**k):
   s=state();s.prepare()
   if s.startup_profiles:return old_capture(obj,*a,**k)
   s.startup_profiles+=1
   import torch_npu
   prof=_original_profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],record_shapes=True,profile_memory=False,with_stack=True,
    experimental_config=torch_npu.profiler._ExperimentalConfig(profiler_level=torch_npu.profiler.ProfilerLevel.Level2,aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,mstx=True,export_type=[torch_npu.profiler.ExportType.Db]),
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(ROOT/'startup'),worker_name='creation-'+s.uid,analyse_flag=False))
   s.emit('creation_profile_begin',CPU=True,NPU=True,stack=True,memory=False,level='Level2')
   try:
    with prof:return old_capture(obj,*a,**k)
   finally:s.emit('creation_profile_end',actual_profile_files_require_validation=True)
  @functools.wraps(old_execute)
  def execute(obj,*a,**k):
   s=state()
   if not s.armed:return old_execute(obj,*a,**k)
   root=s.step_depth==0;s.step_depth+=1
   if root:
    s.step+=1;s.eager_occ={};s.step_epoch=s.next_epoch('worker_step')
    scheduler=k.get('scheduler_output',a[0] if a else None);scheduled=getattr(scheduler,'num_scheduled_tokens',{})
    bounded={str(key)[:192]:int(v) for key,v in scheduled.items()} if type(scheduled) is dict and len(scheduled)<=64 else None
    s.emit('worker_step',worker_step=s.step,history_epoch=s.step_epoch,scheduled_by_request=bounded,total_num_scheduled_tokens=getattr(scheduler,'total_num_scheduled_tokens',None),host_step_is_not_global_collective_ordinal=True)
   try:
    with mark('wio_joint168/'+s.uid+'/worker_step/'+str(s.step)):return old_execute(obj,*a,**k)
   finally:s.step_depth-=1
  cls.capture_model=capture;cls.execute_model=execute
 elif label=='worker':
  cls=module.NPUWorker;old_profile=cls.profile
  @functools.wraps(old_profile)
  def profile(obj,*a,**k):
   s=state();start=k.get('is_start',a[0] if a else True)
   import torch_npu
   if start:
    s.prepare();assert not s.armed and not s.dumped,'One formal collection per service incarnation'
    s.epoch=0;s.step=0;s.index.zero_();torch_npu.npu.synchronize();s.armed=True;s.emit('formal_armed',rank=int(obj.rank))
    return old_profile(obj,*a,**k)
   torch_npu.npu.synchronize()
   result=old_profile(obj,*a,**k)
   if s.armed:s.dump()
   return result
  cls.profile=profile
 else:raise RuntimeError(label)
 _installed.add(module.__name__)
