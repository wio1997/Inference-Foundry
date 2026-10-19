"""Isolated loader repair for the exact rc1 image's MemFabric 1.2 wheel.

No installed file is changed. Reuse the auditwheel libraries already required by
the extension and retain CDLL handles, avoiding a second unversioned SDK copy.
Deployment and acceptance require a separate reviewed ZCode run.
"""
import ctypes
import hashlib
import importlib.util
from pathlib import Path
import sys

INIT_SHA256 = '988298d70bae124e3e0e943d44d74324a1f3fceef227cc78df0ad370b58ba92f'
LIBRARIES = {
    'libmf_hybm_core.so': ('libmf_hybm_core-00225bf3.so', 'e5c81008b5f479fa9e51fc5a7f082823ba960d1becf435ddceef5d2f90f1e9cb'),
    'libmf_smem.so': ('libmf_smem-2154e855.so', 'aaf716211fcddc1a586908a90a5015388f5f4822421d516c40fdcb6a36b2c6d6'),
    'libmf_acc_offload.so': ('libmf_acc_offload-0c99b633.so', '556050429955edf2918f45d3e0c87003c598d39a46adef372766746087aa0717'),
    'libboundscheck.so': ('libboundscheck-58fadf91.so', 'a55d20f5ec61af5e11369ea13087779980d690000b9dce475b7b270eab10299d'),
}


def load_memfabric():
    if 'memfabric_hybrid' in sys.modules:
        module = sys.modules['memfabric_hybrid']
        if not getattr(module, '_glm53_guarded', False):
            raise RuntimeError('MemFabric already imported without GLM53 loader guard')
        return module
    spec = importlib.util.find_spec('memfabric_hybrid')
    if spec is None or spec.origin is None:
        raise ImportError('MemFabric unavailable in fixed image')
    origin = Path(spec.origin)
    data = origin.read_bytes()
    if hashlib.sha256(data).hexdigest() != INIT_SHA256:
        raise RuntimeError('MemFabric __init__ SHA changed; refuse compatibility overlay')
    canonical = origin.parent.parent / 'memfabric_hybrid.libs'
    for name, (filename, digest) in LIBRARIES.items():
        path = canonical / filename
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'MemFabric canonical library SHA changed: {name}')
    handles = []

    def load_library(name):
        path = canonical / LIBRARIES[name][0] if name in LIBRARIES else origin.parent / 'lib' / name
        handle = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        handles.append(handle)
        return handle

    source = data.decode()
    needle = 'ctypes.CDLL(os.path.join(libs_path, lib), mode=ctypes.RTLD_GLOBAL)'
    if source.count(needle) != 2:
        raise RuntimeError('Unexpected MemFabric loader structure')
    source = source.replace(needle, '_glm53_load_library(lib)')
    module = importlib.util.module_from_spec(spec)
    module.__dict__.update(_glm53_load_library=load_library,
                           _glm53_handles=handles,
                           _glm53_guarded=True,
                           _glm53_original_init_sha256=INIT_SHA256)
    sys.modules['memfabric_hybrid'] = module
    try:
        exec(compile(source, str(origin), 'exec'), module.__dict__)
    except BaseException:
        del sys.modules['memfabric_hybrid']
        raise
    return module
