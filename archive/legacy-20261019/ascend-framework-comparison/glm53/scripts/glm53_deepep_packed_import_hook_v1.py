"""D-only lazy import hook; installs the reviewed overlay in spawned interpreters."""
import hashlib,importlib.abc,importlib.machinery,json,os,sys
from pathlib import Path

TARGET='deep_ep.strategies.normal_strategy'
PACK_SHA='336a64b8f6657541f24daa49588891072d350b4a0d9d94b383c6f3fe08cea5cb'
OPT_SHA='daf4825ab2579af4cca617cf2746b8a97d30020e7abde47277bc4abe7e009363'

class Loader(importlib.abc.Loader):
 def __init__(self,original):self.original=original
 def create_module(self,spec):
  return self.original.create_module(spec) if hasattr(self.original,'create_module') else None
 def exec_module(self,module):
  self.original.exec_module(module)
  import glm53_deepep_metadata_batch_v1 as optimizer
  assert hashlib.sha256(Path(optimizer.__file__).read_bytes()).hexdigest()==OPT_SHA
  installed=optimizer.install(module)
  print('GLM53_DEEPEP_METADATA_BATCH_INSTALLED='+json.dumps({'project_id':'glm53','pid':os.getpid(),'node':os.environ['GLM53_NODE'],'source_path':module.__file__,'optimizer_sha256':OPT_SHA,**installed}),flush=True)
  import glm53_deepep_packed_dispatch_v1 as packed
  assert hashlib.sha256(Path(packed.__file__).read_bytes()).hexdigest()==PACK_SHA
  result=packed.install(module)
  print('GLM53_C32_PACKED_DISPATCH_INSTALLED='+json.dumps({'project_id':'glm53','pid':os.getpid(),'node':os.environ['GLM53_NODE'],'optimizer_sha256':PACK_SHA,**result}),flush=True)
 def __getattr__(self,name):return getattr(self.original,name)

class Finder(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname!=TARGET:return None
  spec=importlib.machinery.PathFinder.find_spec(fullname,path,target)
  assert spec is not None and spec.loader is not None and not isinstance(spec.loader,Loader)
  spec.loader=Loader(spec.loader);return spec

def activate():
 assert os.environ.get('GLM53_PACKED_DISPATCH')=='1'
 assert os.environ.get('GLM53_DEEPEP_METADATA_BATCH')=='1'
 assert os.environ.get('GLM53_NODE') in ['36','41'],'Metadata optimization is D-only'
 assert not any(isinstance(x,Finder) for x in sys.meta_path)
 assert TARGET not in sys.modules,'Register hook before native DeepEP imports'
 sys.meta_path.insert(0,Finder())
 print('GLM53_DEEPEP_METADATA_HOOK_READY='+json.dumps({'project_id':'glm53','pid':os.getpid(),'node':os.environ['GLM53_NODE'],'optimizer_sha256':OPT_SHA}),flush=True)
