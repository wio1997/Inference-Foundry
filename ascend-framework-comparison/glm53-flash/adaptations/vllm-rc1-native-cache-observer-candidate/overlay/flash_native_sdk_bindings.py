# SPDX-License-Identifier: Apache-2.0
"""Original vendor native extensions only; no SDK package/module stubs.
Candidate import entrypoint, NOT a qualified cache/runtime repair.
"""
import importlib.util,importlib.metadata,pathlib,hashlib,sys,threading
_LOCK=threading.Lock();_CACHE=None
_IDENTITIES={'_pymf_transfer':'b51dd8c9373e2953ea8f790d2b12e03bc55353a2d6a353785b2112b0c6f8c96c','_pymf_hybrid':'bd642be8d13319623571baf4709897caa4c240d1f6887b5425c48a9bf7d462f1','_pymmc':'0c18c360818e3d31d2374310cd6ea091bd98044d651265bca42c18ab8905108d'}
def _binding(distribution,package,module):
 dist=importlib.metadata.distribution(distribution);assert dist.version=='1.2.0';candidates=[dist.locate_file(f) for f in dist.files if str(f).startswith(package+'/'+module+'.') and str(f).endswith('.so')];assert len(candidates)==1
 path=pathlib.Path(candidates[0]);sha=hashlib.sha256(path.read_bytes()).hexdigest()
 if module in _IDENTITIES:assert sha==_IDENTITIES[module]
 if module in sys.modules:
  native=sys.modules[module];assert pathlib.Path(native.__file__).resolve()==path.resolve();return native
 spec=importlib.util.spec_from_file_location(module,path);native=importlib.util.module_from_spec(spec);sys.modules[module]=native;spec.loader.exec_module(native);return native

def load_bindings():
 global _CACHE
 with _LOCK:
  if _CACHE is None:
   assert 'memfabric_hybrid' not in sys.modules and 'memcache_hybrid' not in sys.modules,'Unexpected publicSDKpackagepreloaded'
   # Same original pybind types/functions/classes exported by vendor packages.
   # BM native types must precede MMC defaults, as installed __init__.py does.
   hybrid=_binding('memfabric-hybrid','memfabric_hybrid','_pymf_hybrid')
   transfer=_binding('memfabric-hybrid','memfabric_hybrid','_pymf_transfer')
   mmc=_binding('memcache-hybrid','memcache_hybrid','_pymmc')
   _CACHE=(hybrid,transfer,mmc)
  return _CACHE
