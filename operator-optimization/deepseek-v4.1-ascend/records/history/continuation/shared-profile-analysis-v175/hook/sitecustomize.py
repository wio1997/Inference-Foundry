"""Opt-in native execution trace and scalar/bank bindings for the owned cohort."""
import os
if os.environ.get('WIO_META_ENABLE')=='v175':
 import sys,json,importlib.abc,importlib.machinery
 from pathlib import Path
 # Preserve the existing serve script/environment. Add only the profiler
 # configuration used to expose its existing start/stop control endpoints.
 if Path(sys.argv[0]).name=='vllm' and 'serve' in sys.argv[:3] and '--profiler-config' not in sys.argv:
  sys.argv+=['--profiler-config',json.dumps({'profiler':'torch','torch_profiler_dir':os.environ['WIO_META_ROOT']+'/unused-npu-profile','ignore_frontend':True,'max_iterations':0,'torch_profiler_with_memory':False,'torch_profiler_with_stack':False,'torch_profiler_record_shapes':False})]
 import joint_runtime
 wanted=joint_runtime.PINNED
 class Loader(importlib.abc.Loader):
  def __init__(self,base):self.base=base
  def create_module(self,spec):
   f=getattr(self.base,'create_module',None);return f(spec) if f else None
  def exec_module(self,module):self.base.exec_module(module);joint_runtime.install(module)
  def __getattr__(self,key):return getattr(self.base,key)
 class Finder(importlib.abc.MetaPathFinder):
  def find_spec(self,fullname,path=None,target=None):
   if fullname not in wanted:return None
   spec=importlib.machinery.PathFinder.find_spec(fullname,path)
   if spec is not None and spec.loader is not None:spec.loader=Loader(spec.loader)
   return spec
 sys.meta_path.insert(0,Finder())
 for name in wanted:
  if name in sys.modules:joint_runtime.install(sys.modules[name])
