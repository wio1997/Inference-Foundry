"""Process-local canonical loader for the exact rc1 MemCache1.2 wheel."""
import ctypes
import hashlib
import importlib.util
from pathlib import Path
import sys

MC_INIT_SHA256 = '7a6708755c1b6b67c323123d7a9315cfb9dded160d89a7b36ea278d61898c618'
MC_CANONICAL_SHA256 = '7b6b15909f1cfd647b095a2e6f1d0d3c20abced1fd4a2a8d4462beb48e2691bc'


def load_memcache():
    # load_memfabric is supplied by the independently frozen MF loader module.
    load_memfabric()
    if 'memcache_hybrid' in sys.modules:
        module = sys.modules['memcache_hybrid']
        if not getattr(module, '_glm53_guarded', False):
            raise RuntimeError('MemCache already imported without canonical guard')
        return module
    spec = importlib.util.find_spec('memcache_hybrid')
    if spec is None or spec.origin is None:
        raise ImportError('MemCache unavailable in fixed image')
    origin = Path(spec.origin)
    data = origin.read_bytes()
    if hashlib.sha256(data).hexdigest() != MC_INIT_SHA256:
        raise RuntimeError('MemCache initializer SHA changed')
    canonical = origin.parent.parent / 'memcache_hybrid.libs' / 'libmf_memcache-7c15eaaf.so'
    if hashlib.sha256(canonical.read_bytes()).hexdigest() != MC_CANONICAL_SHA256:
        raise RuntimeError('MemCache canonical library SHA changed')
    handles = []

    def load_library(name):
        path = canonical if name == 'libmf_memcache.so' else origin.parent / 'lib' / name
        handle = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        handles.append(handle)
        return handle

    source = data.decode()
    needle = 'ctypes.CDLL(so_path)'
    if source.count(needle) != 1:
        raise RuntimeError('Unexpected MemCache preload structure')
    source = source.replace(needle, '_glm53_mmc_load(lib_name)')
    module = importlib.util.module_from_spec(spec)
    module.__dict__.update(_glm53_mmc_load=load_library, _glm53_handles=handles,
                           _glm53_guarded=True, _glm53_original_init_sha256=MC_INIT_SHA256)
    sys.modules['memcache_hybrid'] = module
    try:
        exec(compile(source, str(origin), 'exec'), module.__dict__)
    except BaseException:
        del sys.modules['memcache_hybrid']
        raise
    return module
