"""Observe native ops calls once after return; never inspect tensor values or alter returns."""
import functools, hashlib, importlib.abc, importlib.machinery, inspect, json, os, sys
from pathlib import Path

TARGET='deep_ep.strategies.low_latency_strategy'
SOURCE_SHA='1b9fe42cc46425c8d7e97d4f4e8a0501a6465e9b49033522d89b31362d363c46'
REVISION='glm53-C32-native-ops-window-audit-v2'

def tensor_metadata(tensor):
 if tensor is None:return None
 return {'shape':list(tensor.shape),'dtype':str(tensor.dtype),'device':str(tensor.device)}

def install(module):
 assert os.environ.get('GLM53_C32_OPS_AUDIT')=='1' and os.environ.get('GLM53_NODE') in ['36','41']
 assert os.environ.get('DEEP_USE_MODE')=='ops'
 assert os.environ.get('HCCL_BUFFSIZE')==os.environ.get('DEEPEP_HCCL_BUFFSIZE')=='1024'
 assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()==SOURCE_SHA
 cls=module.OpsLowLatencyCommStrategy
 assert not getattr(cls,'_glm53_C32_ops_audit',False)
 for name in ['_npu_low_latency_dispatch','_npu_low_latency_combine']:
  original=getattr(cls,name);signature=inspect.signature(original)
  assert 'comm_alg' in signature.parameters and 'x_active_mask' in signature.parameters
  def make_wrapper(original,signature,name):
   emitted=set()
   @functools.wraps(original)
   def wrapper(self,*args,**kwargs):
    if not emitted:assert os.environ.get('HCCL_BUFFSIZE')==os.environ.get('DEEPEP_HCCL_BUFFSIZE')=='1024'
    result=original(self,*args,**kwargs)
    if not emitted:
     bound=signature.bind(self,*args,**kwargs);bound.apply_defaults();values=bound.arguments
     # Metadata-only properties; no .cpu/.item/.tolist/synchronization/copies.
     tensors={key:tensor_metadata(values[key]) for key in ['x','topk_idx','topk_weights','x_active_mask'] if key in values}
     outputs=[tensor_metadata(value) for value in result] if name.endswith('dispatch') else tensor_metadata(result)
     row={'project_id':'glm53','revision':REVISION,'node':os.environ['GLM53_NODE'],'pid':os.getpid(),'method':name,'group_size':self.group_size,'rank':self.rank,'comm_alg':values['comm_alg'],'num_experts':values.get('num_experts'),'quant_mode':values.get('quant_mode'),'comm_quant_mode':values.get('comm_quant_mode'),'num_max_dispatch_tokens_per_rank':values.get('num_max_dispatch_tokens_per_rank'),'inputs':tensors,'outputs':outputs,'original_method_returned':True,'upstream_sha256':SOURCE_SHA,'HCCL_BUFFSIZE_environment':os.environ['HCCL_BUFFSIZE'],'DEEPEP_HCCL_BUFFSIZE_environment':os.environ['DEEPEP_HCCL_BUFFSIZE']}
     print('GLM53_C32_OPS_FIRST_USE='+json.dumps(row),flush=True);emitted.add(True)
    return result
   return wrapper
  setattr(cls,name,make_wrapper(original,signature,name))
 cls._glm53_C32_ops_audit=True
 return {'revision':REVISION,'upstream_sha256':SOURCE_SHA,'methods':['_npu_low_latency_dispatch','_npu_low_latency_combine'],'tensor_value_reads':0,'NPU_synchronizations_added':0}

class Loader(importlib.abc.Loader):
 def __init__(self,original):self.original=original
 def create_module(self,spec):return self.original.create_module(spec) if hasattr(self.original,'create_module') else None
 def exec_module(self,module):
  self.original.exec_module(module);result=install(module)
  print('GLM53_C32_OPS_AUDIT_INSTALLED='+json.dumps({'project_id':'glm53','pid':os.getpid(),'node':os.environ['GLM53_NODE'],**result}),flush=True)
 def __getattr__(self,name):return getattr(self.original,name)
class Finder(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname!=TARGET:return None
  spec=importlib.machinery.PathFinder.find_spec(fullname,path,target)
  assert spec is not None and spec.loader is not None and not isinstance(spec.loader,Loader)
  spec.loader=Loader(spec.loader);return spec
def activate():
 assert os.environ.get('GLM53_C32_OPS_AUDIT')=='1' and os.environ.get('GLM53_NODE') in ['36','41'] and os.environ.get('DEEP_USE_MODE')=='ops'
 assert TARGET not in sys.modules and not any(isinstance(x,Finder) for x in sys.meta_path)
 sys.meta_path.insert(0,Finder())
