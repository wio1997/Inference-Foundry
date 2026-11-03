"""Temporary opt-in import adapter;original installed framework files unchanged."""
import os
if os.environ.get('WIO_JOINT_ENABLE')=='v168':
 import importlib.abc,importlib.machinery,sys
 wanted={'vllm_ascend.ops.fused_moe.moe_comm_method':'moe','vllm_ascend.worker.model_runner_v1':'runner','vllm_ascend.worker.worker':'worker','vllm_ascend.compilation.updatable_graph':'updatable'}
 class Loader(importlib.abc.Loader):
  def __init__(self,base,label):self.base=base;self.label=label
  def create_module(self,spec):
   f=getattr(self.base,'create_module',None);return f(spec) if f else None
  def exec_module(self,module):
   self.base.exec_module(module)
   import joint_runtime
   joint_runtime.install(module,self.label)
  def __getattr__(self,key):return getattr(self.base,key)
 class Finder(importlib.abc.MetaPathFinder):
  def find_spec(self,fullname,path=None,target=None):
   if fullname not in wanted:return None
   spec=importlib.machinery.PathFinder.find_spec(fullname,path)
   if spec is not None and spec.loader is not None:spec.loader=Loader(spec.loader,wanted[fullname])
   return spec
 sys.meta_path.insert(0,Finder())
 for fullname,label in wanted.items():
  if fullname in sys.modules:
   import joint_runtime
   joint_runtime.install(sys.modules[fullname],label)
