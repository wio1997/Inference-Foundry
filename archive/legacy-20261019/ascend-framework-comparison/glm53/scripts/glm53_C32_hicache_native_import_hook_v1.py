"""Lazy P-only native HiCache MTP compatibility; no eager framework imports."""
import hashlib,importlib.abc,importlib.machinery,json,os,sys
from pathlib import Path
TARGET='sglang.srt.mem_cache.pool_host.mla'
HELPER_SHA='50d7cc8040b9956092dd40fe204bc5d6d3633bf04f1ba6937a8273fc72ac0b2f'
class Loader(importlib.abc.Loader):
    def __init__(self,original):self.original=original
    def create_module(self,spec):
        return self.original.create_module(spec) if hasattr(self.original,'create_module') else None
    def exec_module(self,module):
        self.original.exec_module(module)
        import glm53_C32_hicache_native_mtp_v1 as compatibility
        assert hashlib.sha256(Path(compatibility.__file__).read_bytes()).hexdigest()==HELPER_SHA
        result=compatibility.install(module)
        print('GLM53_C32_NATIVE_HICACHE_INSTALLED='+json.dumps({'project_id':'glm53','pid':os.getpid(),'node':os.environ['GLM53_NODE'],'source_path':module.__file__,'helper_sha256':HELPER_SHA,**result}),flush=True)
    def __getattr__(self,name):return getattr(self.original,name)
class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname!=TARGET:return None
        spec=importlib.machinery.PathFinder.find_spec(fullname,path,target)
        assert spec and spec.loader and not isinstance(spec.loader,Loader)
        spec.loader=Loader(spec.loader);return spec
def activate():
    assert os.environ.get('GLM53_C32_NATIVE_HICACHE_MTP')=='1'
    assert os.environ.get('SGLANG_HICACHE_HOST_MEM_BACKEND')=='memfabric'
    assert os.environ.get('GLM53_NODE') in ['34','35'],'Native HiCache IO is P-only'
    assert TARGET not in sys.modules and not any(isinstance(x,Finder) for x in sys.meta_path)
    sys.meta_path.insert(0,Finder())
    print('GLM53_C32_NATIVE_HICACHE_HOOK_READY='+json.dumps({'project_id':'glm53','pid':os.getpid(),'node':os.environ['GLM53_NODE'],'helper_sha256':HELPER_SHA}),flush=True)
