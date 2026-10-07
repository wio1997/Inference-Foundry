"""Execute pinned source AST; all device/graph/stream operations are doubles."""
from __future__ import annotations
import ast, contextlib, dataclasses, hashlib, inspect, itertools, json, math, os, sys, threading, time, types, weakref
from pathlib import Path
from types import SimpleNamespace as NS
R=Path(__file__).resolve().parent
UP=R.parent/'vllm__vllm__v1__worker__gpu_model_runner.py'
log=[]; state=NS(enabled=False, graph=None, mode=0, fail=False, pending=False, invalid=False)
class Tensor:
 def __init__(self,n=8,pool=None,value=None,offset=0): self.n=n;self.pool=pool;self.value=value;self.offset=offset;self.ptr=id(self)
 def numel(self):return self.n
 def element_size(self):return 1
 def data_ptr(self):return self.ptr+self.offset
 def __getitem__(self,s):return self
 def view(self,*a):return self
 def reshape(self,*a):return self
class Mode:
 def __init__(self,name):self.name=name
 def is_valid_runtime_mode(self):return True
NONE=Mode('NONE');FULL=Mode('FULL');Modes=NS(NONE=NONE,FULL=FULL)
@dataclasses.dataclass(frozen=True)
class Desc:
 num_tokens:int=2
 uniform:bool=True
 num_active_loras:int=0
ctx=NS(cudagraph_runtime_mode=FULL,batch_descriptor=Desc(),capturing=False)
class Graph:
 def __init__(self):self.ops=[]
 def capture_begin(self,pool=None):self.pool=pool;assert state.graph is None;state.graph=self;log.append(('begin',pool))
 def capture_end(self):state.graph=None;log.append(('end',self.pool))
 def replay(self):log.append(('replay',self.pool));[f() for f in self.ops]
class Stream:
 def synchronize(self):log.append(('stream_sync',))
stream=Stream()
@contextlib.contextmanager
def cm(name):
 log.append(('enter',name))
 try:yield
 finally:log.append(('exit',name))
def sync():log.append(('accelerator_sync',))
def empty():log.append(('empty_cache',))
def set_enabled(v):state.enabled=v;log.append(('enabled',v))
def validate():assert state.enabled
logger=NS(info_once=lambda *a,**k:None,debug=lambda *a,**k:None,warning=lambda *a,**k:None,info=lambda *a,**k:None)
torch=NS(Tensor=Tensor,uint8=NS(itemsize=1),empty=lambda shape,**k:Tensor(math.prod(shape)),accelerator=NS(synchronize=sync,empty_cache=empty,get_memory_info=lambda:(10**9,2*10**9)),cuda=NS(CUDAGraph=Graph),npu=NS(current_stream=lambda:stream),profiler=NS(record_function=lambda s:cm('record:'+s)))
platform=NS(get_global_graph_pool=lambda:'default',graph_pool_handle=lambda:'fallback')
offloader=NS(sync_prev_onload=lambda:log.append(('offload_sync',)),join_after_forward=lambda:log.append(('offload_join',)))
sys.modules['oracle']=types.ModuleType('oracle')
ns=dict(__name__='oracle',dataclasses=dataclasses,torch=torch,logger=logger,envs=NS(VLLM_LOGGING_LEVEL='INFO',VLLM_DEBUG_WORKSPACE=False),current_platform=platform,weakref=weakref,threading=threading,dataclass=dataclasses.dataclass,get_forward_context=lambda:ctx,is_forward_context_available=lambda:True,CUDAGraphMode=Modes,BatchDescriptor=Desc,gc=NS(collect=lambda:log.append(('gc',))),get_offloader=lambda:offloader,weak_ref_tensors=lambda t:t,set_graph_pool_id=lambda p:log.append(('set_pool',p)),validate_cudagraph_capturing_enabled=validate,_EXTRA_CTX=NS(is_draft_model=False),get_graph_params=lambda:None,get_draft_graph_params=lambda:None,get_draft_graph_prefill_params=lambda:None,weak_ref_workspaces=lambda p:log.append(('weak_workspace',)),_h9_mode=lambda:0,_h9_record=lambda *a:log.append(('h9_record',)),os=os,inspect=inspect,prod=math.prod,accumulate=itertools.accumulate,round_up=lambda a,n:(a+n-1)//n*n,dbo_current_ubatch_id=lambda:0,_MB=1024**2,_manager=None)
def load(path,names):
 nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
 assert {n.name for n in nodes}==set(names)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])),str(path),'exec'),ns)
load(R/'workspace.py',['_compute_bytes','WorkspaceManager','current_workspace_manager','lock_workspace'])
wm=ns['WorkspaceManager']('cpu-double');ns['_manager']=wm
w=wm._ensure_workspace_size(256);ns['lock_workspace']();ns['lock_workspace']()
assert wm._ensure_workspace_size(256) is w and wm._ensure_workspace_size(128) is w
try:wm._ensure_workspace_size(257);raise RuntimeError('growth accepted')
except AssertionError:pass
load(R/'breakable_cudagraph.py',['BreakableCUDAGraphCapture','_BreakableEntry','BreakableCUDAGraphWrapper'])
load(R/'breakable_aclgraph.py',['BreakableACLGraphWrapper'])
config=NS(compilation_config=NS(cudagraph_mode=FULL,cudagraph_num_of_warmups=1),profiler_config=NS(capture_torch_profiler=False))
def model(**kwargs):
 log.append(('raw_model',state.mode));assert wm._ensure_workspace_size(128) is w
 if state.fail and state.graph is not None:raise RuntimeError('injected capture failure')
 out=Tensor(pool=state.graph.pool if state.graph else None,value=state.mode)
 if state.graph is not None:
  captured=state.mode;state.graph.ops.append(lambda: setattr(out,'value',captured))
 return out
A=ns['BreakableACLGraphWrapper'](model,config);B=ns['BreakableACLGraphWrapper'](model,config);A.graph_pool='poolA';B.graph_pool='poolB'
# Extract real runner methods into classes; do not replace their bodies.
def methods(path,clsname,wanted,newname,bases=[]):
 tree=ast.parse(path.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==clsname)
 nodes=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in wanted];assert len(nodes)==len(wanted)
 c=ast.ClassDef(name=newname,bases=[ast.Name(id=x,ctx=ast.Load()) for x in bases],keywords=[],body=nodes,decorator_list=[])
 ast.fix_missing_locations(c)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),c],type_ignores=[])),str(path),'exec'),ns)
world=types.ModuleType('vllm.distributed.parallel_state');world.get_world_group=lambda:NS(local_rank=0)
sys.modules['vllm.distributed.parallel_state']=world
ns.update(time=time,nullcontext=contextlib.nullcontext,instrument=lambda **k:lambda f:f,set_cudagraph_capturing_enabled=set_enabled,graph_capture=lambda **k:cm('graph_capture'),compilation_counter=NS(num_gpu_runner_capture_triggers=0),lock_workspace=ns['lock_workspace'],is_global_first_rank=lambda:False,check_ubatch_thresholds=lambda **k:False,_get_gpu_model_runner_module_name=lambda r:'actual',_torch_cuda_wrapper=lambda:cm('torch_cuda_wrapper'),_replace_gpu_model_runner_function_wrapper=lambda n:cm('replace_gpu_functions'))
methods(UP,'GPUModelRunner',['capture_model','_warmup_and_capture','_capture_cudagraphs'],'GPUModelRunner')
methods(R/'model_runner_v1.py','NPUModelRunner',['capture_model'],'TestRunner',['GPUModelRunner'])
runner=ns['TestRunner']();runner.compilation_config=config.compilation_config;runner.vllm_config=config;runner.device='cpu-double';runner.encoder_cudagraph_manager=None;runner.update_stream=None;runner.parallel_config=NS(use_ubatching=False);runner.lora_config=None
runner._maybe_init_encoder_cudagraph_manager=lambda:None;runner._freeze_gc=lambda:cm('freeze_gc');runner.maybe_remove_all_loras=lambda x:None
runner.cudagraph_dispatcher=NS(get_capture_descs=lambda:[(FULL,[Desc()])])
def dummy(n,cudagraph_runtime_mode,**kwargs):
 ctx.cudagraph_runtime_mode=cudagraph_runtime_mode;ctx.batch_descriptor=Desc();ctx.capturing=False
 return runner.model(input_ids=shared)
shared=Tensor();runner._dummy_run=dummy
for mode,bank in [(0,A),(1,B)]:
 state.mode=mode;runner.model=bank;runner.capture_model();assert not state.enabled
entryA=A.entries[Desc()];entryB=B.entries[Desc()];assert entryA is not entryB and entryA.output is not entryB.output
assert entryA.capture.pool=='poolA' and entryB.capture.pool=='poolB'
assert A.runnable is B.runnable is model
raw_count=sum(x[0]=='raw_model' for x in log)
# Proposed controller boundary, not behavior supplied by upstream wrapper.
def select(bank):
 assert not state.pending and not state.invalid,'bank selection requires idle valid lifecycle'
 runner.model=bank
for mode,bank in [(0,A),(1,B),(0,A),(1,B)]:
 select(bank);state.mode=mode;ctx.cudagraph_runtime_mode=FULL
 out=runner.model(input_ids=shared);assert out is bank.entries[Desc()].output and out.value==mode
assert sum(x[0]=='raw_model' for x in log)==raw_count
for mode,bank in [(1,A),(0,B)]:
 select(bank);state.mode=mode;ctx.cudagraph_runtime_mode=NONE
 assert runner.model(input_ids=shared).value==mode
state.pending=True
try:select(A);raise RuntimeError('busy selection allowed')
except AssertionError:pass
state.pending=False
C=ns['BreakableACLGraphWrapper'](model,config);C.graph_pool='poolFailure';runner.model=C;state.fail=True
try:runner.capture_model();raise AssertionError('capture failure swallowed')
except RuntimeError as e:assert 'injected' in str(e);state.invalid=True
assert state.enabled # actual upstream has no finally; recovery is mandatory
try:select(A);raise RuntimeError('invalid lifecycle resumed')
except AssertionError:pass
assert A.entries[Desc()] is entryA and B.entries[Desc()] is entryB
assert ns['BreakableCUDAGraphCapture'].current() is None
assert sum(x[0]=='offload_join' for x in log)==2
assert sum(x[0]=='h9_record' for x in log)==4
assert sum(x[0]=='stream_sync' for x in log)==4
assert sum(x[0]=='offload_sync' for x in log)==7
assert [x[1] for x in log if x[0]=='enter'].count('torch_cuda_wrapper')==3
assert [x[1] for x in log if x[0]=='exit'].count('torch_cuda_wrapper')==3
assert [x[1] for x in log if x[0]=='enter'].count('replace_gpu_functions')==3
assert [x[1] for x in log if x[0]=='exit'].count('replace_gpu_functions')==3
assert 'torch_npu' not in sys.modules and 'torch' not in sys.modules
result=dict(passed=True,actual_AST=['workspace allocation/lock','base wrapper and capture object','Ascend capture/replay H9off','Ascend runner capture_model','upstream capture_model/_warmup_and_capture/_capture_cudagraphs'],checks=dict(independent_pools=True,same_raw_model=True,same_descriptor=True,private_outputs_ABAB=True,NONE_current_prepare=True,workspace_equal_smaller_stable=True,workspace_growth_rejected=True,second_lock_idempotent=True,capture_failure_requires_recovery=True,idle_selection_gate=True,contexts_sync_offloader_preserved=True),source_sha256={str(p.name):hashlib.sha256(p.read_bytes()).hexdigest() for p in [R/'workspace.py',R/'breakable_cudagraph.py',R/'breakable_aclgraph.py',R/'model_runner_v1.py',UP]},NPU_requests=0,torch_imported=False,torch_npu_imported=False,limitations=['graph/device/stream/pool/tensor allocations and weak references are doubles; no hardware allocator/HCCL proof','idle and invalid-lifecycle guards are proposed controller gates, not upstream guarantees','dummy_run and model body are doubles; actual metadata/KV/async side-copy content is not executed','H9 witness I/O is intercepted; H9 mode fixed0; no live files touched'],event_counts={k:sum(x[0]==k for x in log) for k in sorted({x[0] for x in log})},log=log)
(R/'lifecycle_CPU_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('log','source_sha256')}))
