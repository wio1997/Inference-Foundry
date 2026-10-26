"""Lazy P-only native HiCache MTP and existing MC mapping compatibility; no eager framework imports."""
import hashlib,importlib.abc,importlib.machinery,json,os,sys
from pathlib import Path
MLA_TARGET='sglang.srt.mem_cache.pool_host.mla'
MC_TARGET='sglang.srt.mem_cache.storage.npu_memcache.npu_memcache_store'
MC_HELPER_SHA='daec07cca112741d54bc766d206ca6ccabb27258d5ceb26f79a0573878514a82'
HELPER_SHA='50d7cc8040b9956092dd40fe204bc5d6d3633bf04f1ba6937a8273fc72ac0b2f'
class Loader(importlib.abc.Loader):
    def __init__(self,original):self.original=original
    def create_module(self,spec):
        return self.original.create_module(spec) if hasattr(self.original,'create_module') else None
    def exec_module(self,module):
        self.original.exec_module(module)
        if module.__name__==MLA_TARGET:
            import glm53_C32_hicache_native_mtp_v1 as compatibility
            expected=HELPER_SHA
        else:
            assert module.__name__==MC_TARGET
            import glm53_C32_native_MC_mapping_reuse_v1 as compatibility
            expected=MC_HELPER_SHA
        assert hashlib.sha256(Path(compatibility.__file__).read_bytes()).hexdigest()==expected
        result=compatibility.install(module)
        print('GLM53_C32_NATIVE_HICACHE_INSTALLED='+json.dumps({'project_id':'glm53','pid':os.getpid(),'node':os.environ['GLM53_NODE'],'module':module.__name__,'source_path':module.__file__,'helper_sha256':expected,**result}),flush=True)
    def __getattr__(self,name):return getattr(self.original,name)
class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname not in [MLA_TARGET,MC_TARGET]:return None
        spec=importlib.machinery.PathFinder.find_spec(fullname,path,target)
        assert spec and spec.loader and not isinstance(spec.loader,Loader)
        spec.loader=Loader(spec.loader);return spec
def activate():
    assert os.environ.get('GLM53_C32_NATIVE_MC_MAPPING_REUSE')=='1'
    assert os.environ.get('GLM53_C32_NATIVE_HICACHE_MTP')=='1'
    assert os.environ.get('SGLANG_HICACHE_HOST_MEM_BACKEND')=='memfabric'
    assert os.environ.get('GLM53_NODE') in ['34','35'],'Native HiCache IO is P-only'
    assert MLA_TARGET not in sys.modules and MC_TARGET not in sys.modules and not any(isinstance(x,Finder) for x in sys.meta_path)
    sys.meta_path.insert(0,Finder())
    print('GLM53_C32_NATIVE_HICACHE_HOOK_READY='+json.dumps({'project_id':'glm53','pid':os.getpid(),'node':os.environ['GLM53_NODE'],'helper_sha256':HELPER_SHA}),flush=True)
