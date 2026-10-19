"""Pinned C32 target-graph compatibility: fullmesh MC2, bounded padding, eager draft."""
import functools,hashlib,importlib.abc,importlib.machinery,inspect,json,os,sys
from pathlib import Path
REVISION='glm53-C32-target-graph-fullmesh-v2'
OPS='deep_ep.strategies.low_latency_strategy'
DRAFT='sglang.srt.speculative.eagle_worker_v2'
GRAPH='sglang.srt.model_executor.runner.decode_cuda_graph_runner'
SOURCE_SHA={OPS:'1b9fe42cc46425c8d7e97d4f4e8a0501a6465e9b49033522d89b31362d363c46',DRAFT:'67a76083ecc365604b7fa9e9d8dd64aa0d4cf3f023c9050e98dc48829985716d',GRAPH:'7967c824367fcdbe83a9960268e6d724d78f486d654e4c5715095634fe2f65df'}
_fds={}
def gate():
 assert os.environ.get('GLM53_C32_TARGET_GRAPH_ALIGN')=='1'
 assert os.environ.get('GLM53_NODE') in ['36','41']
 assert os.environ.get('DEEP_USE_MODE')=='ops'
 assert os.environ.get('HCCL_BUFFSIZE')=='256'
 assert os.environ.get('DEEPEP_HCCL_BUFFSIZE','')==''
 assert os.environ.get('SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK')=='32'
 assert os.environ.get('TASK_QUEUE_ENABLE')=='0'
def emit(event,**data):
 pid=os.getpid()
 if pid not in _fds:
  p=Path(os.environ.get('GLM53_ALIGNMENT_AUDIT_ROOT','/run/glm53/alignment-audit'));p.mkdir(parents=True,exist_ok=True)
  _fds[pid]=os.open(p/('worker-'+str(pid)+'.jsonl'),os.O_WRONLY|os.O_APPEND|os.O_CREAT,0o600)
 row={'project_id':'glm53','revision':REVISION,'node':os.environ['GLM53_NODE'],'pid':pid,'event':event,**data}
 os.write(_fds[pid],(json.dumps(row,separators=(',',':'))+'\n').encode())
def metadata(x):
 if x is None:return None
 return {'shape':list(x.shape),'dtype':str(x.dtype),'device':str(x.device)}
def install_ops(module):
 cls=module.OpsLowLatencyCommStrategy
 assert not getattr(cls,'_glm53_C32_target_graph',False)
 original_init=cls.__init__
 @functools.wraps(original_init)
 def init(self,runtime,group,comm_alg='hierarchy'):
  gate();assert comm_alg=='hierarchy','Unexpected caller algorithm'
  original_init(self,runtime,group,comm_alg='fullmesh_v1')
  assert self.comm_alg=='fullmesh_v1'
 cls.__init__=init
 for name in ['_npu_low_latency_dispatch','_npu_low_latency_combine']:
  original=getattr(cls,name);sig=inspect.signature(original);assert 'comm_alg' in sig.parameters and 'topk_weights' in sig.parameters
  def make(original,sig,name):
   first=set()
   @functools.wraps(original)
   def wrapped(self,*args,**kwargs):
    bound=sig.bind(self,*args,**kwargs);bound.apply_defaults();v=bound.arguments
    assert self.comm_alg==v['comm_alg']=='fullmesh_v1'
    assert self.group_size==32 and v['num_experts']==256
    assert v['num_max_dispatch_tokens_per_rank']==32
    if name.endswith('dispatch'):
     assert tuple(v['x'].shape)==(32,6144) and tuple(v['topk_idx'].shape)==(32,8)
     assert tuple(v['x_active_mask'].shape)==(32,)
     # Fullmesh dispatch does not support expertScales; weighted combine below retains them.
     bound.arguments['topk_weights']=None
    else:
     assert v['topk_weights'] is not None
     # Opaque dispatch auxiliary output must pass to combine unchanged (None or Tensor).
    result=original(*bound.args,**bound.kwargs)
    if not first:
     gate();emit('ops_first_return',method=name,rank=self.rank,group_size=self.group_size,comm_alg=self.comm_alg,quant_mode=v.get('quant_mode'),comm_quant_mode=v.get('comm_quant_mode'),inputs={k:metadata(v[k]) for k in ['x','topk_idx','topk_weights','x_active_mask'] if k in v},outputs=[metadata(x) for x in result] if name.endswith('dispatch') else metadata(result),source_sha256=SOURCE_SHA[OPS]);first.add(True)
    return result
   return wrapped
  setattr(cls,name,make(original,sig,name))
 cls._glm53_C32_target_graph=True
 emit('ops_compat_installed',comm_alg='fullmesh_v1',dispatch_padding=32,HCCL_BUFFSIZE_MiB=256,dispatch_expert_scales=None,combine_expert_scales='unchanged')
def install_draft(module):
 cls=module.EagleDraftWorker;original=cls._capture_cuda_graphs
 assert not getattr(cls,'_glm53_C32_eager_draft',False)
 @functools.wraps(original)
 def eager(self):
  gate();self.cuda_graph_runner=None;self.cuda_graph_runner_for_draft_extend=None
  emit('draft_graph_disabled',target_graph_config_unchanged=True,source_sha256=SOURCE_SHA[DRAFT])
 cls._capture_cuda_graphs=eager;cls._glm53_C32_eager_draft=True
 emit('draft_compat_installed')
def install_graph(module):
 cls=module.DecodeCudaGraphRunner
 assert not getattr(cls,'_glm53_C32_graph_audit',False)
 original_can=cls.can_run_graph;original_execute=cls.execute
 @functools.wraps(original_can)
 def can(self,forward_batch):
  result=original_can(self,forward_batch)
  emit('target_graph_decision',eligible=bool(result),raw_bs=int(forward_batch.batch_size),mode=forward_batch.forward_mode.name,is_draft=bool(self.model_runner.is_draft_worker))
  return result
 @functools.wraps(original_execute)
 def execute(self,forward_batch,*args,**kwargs):
  result=original_execute(self,forward_batch,*args,**kwargs)
  key=self._replay_graph_key
  emit('target_graph_replay_return',raw_bs=int(forward_batch.batch_size),key_size=int(key.size),capture_request_width=int(self.captured_req_width),mode=forward_batch.forward_mode.name,is_draft=bool(self.model_runner.is_draft_worker),backend_type=type(self.backend).__name__)
  return result
 cls.can_run_graph=can;cls.execute=execute;cls._glm53_C32_graph_audit=True
 emit('target_graph_audit_installed',source_sha256=SOURCE_SHA[GRAPH],tensor_value_reads=0,synchronizations_added=0)
def install(module):
 gate();name=module.__name__;assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()==SOURCE_SHA[name]
 {OPS:install_ops,DRAFT:install_draft,GRAPH:install_graph}[name](module)
class Loader(importlib.abc.Loader):
 def __init__(self,original):self.original=original
 def create_module(self,spec):return self.original.create_module(spec) if hasattr(self.original,'create_module') else None
 def exec_module(self,module):self.original.exec_module(module);install(module)
 def __getattr__(self,name):return getattr(self.original,name)
class Finder(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname not in SOURCE_SHA:return None
  spec=importlib.machinery.PathFinder.find_spec(fullname,path,target)
  assert spec is not None and spec.loader is not None and not isinstance(spec.loader,Loader)
  spec.loader=Loader(spec.loader);return spec
def activate():
 gate();assert not any(x in sys.modules for x in SOURCE_SHA)
 assert not any(isinstance(x,Finder) for x in sys.meta_path)
 sys.meta_path.insert(0,Finder())
