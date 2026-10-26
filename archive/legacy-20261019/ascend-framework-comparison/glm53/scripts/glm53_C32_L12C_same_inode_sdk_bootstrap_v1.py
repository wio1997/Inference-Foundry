"""CPU-only single SDK image/handle through an inode-identical private loader alias."""
import ctypes, hashlib, importlib.metadata, json, os, sys
from pathlib import Path
CORE = Path('/usr/local/python3.12.13/lib/python3.12/site-packages/memfabric_hybrid/lib/libmf_acc_offload.so')
LIB = CORE.parent
ALIAS = Path(os.environ['GLM53_NATIVE_LOADER_DIR']) / 'libmf_acc_offload.so'
EXT = ALIAS.with_name('libmf_hybm_accoffload.so')

def load_sdk():
    assert os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] == '0' and (not os.environ.get('LD_PRELOAD')) and (not os.environ.get('MEMFABRIC_HYBRID_EXTEND_LIB_PATH'))
    assert hashlib.sha256(CORE.read_bytes()).hexdigest() == '53736834872ab58915869c60ada0a719473b9b3628c112ded4dc2de8272bd126'
    assert ALIAS.is_symlink() and ALIAS.resolve() == CORE.resolve() and ((ALIAS.stat().st_dev, ALIAS.stat().st_ino) == (CORE.stat().st_dev, CORE.stat().st_ino))
    assert hashlib.sha256(EXT.read_bytes()).hexdigest() == '3e87649f4d2ea85936cd94b439afd3473973bd62abcc06f777c9b875cf1a9d46'
    stock = CORE.with_name('libmf_hybm_accoffload.so')
    assert stock.stat().st_size == 0
    import torch
    assert torch._C._GLIBCXX_USE_CXX11_ABI is True and 'torch_npu' not in sys.modules
    assert 'libmf_acc_offload.so' not in Path('/proc/self/maps').read_text()
    handles = [ctypes.CDLL(str(LIB / name), mode=ctypes.RTLD_GLOBAL) for name in ['libmf_hybm_core.so', 'libmf_smem.so']]
    aliased = ctypes.CDLL(str(ALIAS), mode=ctypes.RTLD_GLOBAL)
    import memfabric_hybrid as mf
    import memcache_hybrid as mc
    assert importlib.metadata.version('memfabric_hybrid') == importlib.metadata.version('memcache_hybrid') == '1.3.0'
    assert mf.__file__.startswith(str(CORE.parents[1]) + '/') and mc.__file__.startswith('/usr/local/python3.12.13/lib/python3.12/site-packages/memcache_hybrid/')
    canonical = ctypes.CDLL(str(CORE), mode=ctypes.RTLD_GLOBAL)
    assert mf._lib_handles[2]._handle == aliased._handle == canonical._handle, 'Multiple SDK instances forbidden'

    class DlInfo(ctypes.Structure):
        _fields_ = [('dli_fname', ctypes.c_char_p), ('dli_fbase', ctypes.c_void_p), ('dli_sname', ctypes.c_char_p), ('dli_saddr', ctypes.c_void_p)]
    dl = ctypes.CDLL('libdl.so.2')
    dl.dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
    dl.dladdr.restype = ctypes.c_int
    symbol = ctypes.c_void_p.in_dll(aliased, '_ZN3ock7offload19AccOffloadLaunchApi18gAccOffloadLibNameE')
    assert ctypes.cast(symbol.value, ctypes.c_char_p).value == b'libmf_hybm_accoffload.so'
    info = DlInfo()
    assert dl.dladdr(ctypes.c_void_p(ctypes.addressof(symbol)), ctypes.byref(info)) == 1
    assert info.dli_fname.decode() == str(ALIAS), info.dli_fname
    assert Path(info.dli_fname.decode()).parent.joinpath('libmf_hybm_accoffload.so').resolve() == EXT.resolve()

    class PhdrInfo(ctypes.Structure):
        _fields_ = [('dlpi_addr', ctypes.c_void_p), ('dlpi_name', ctypes.c_char_p)]
    rows = []
    callback_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(PhdrInfo), ctypes.c_size_t, ctypes.c_void_p)

    @callback_type
    def visit(ptr, size, data):
        name = (ptr.contents.dlpi_name or b'').decode()
        if Path(name).name in ['libmf_hybm_core.so', 'libmf_smem.so', 'libmf_acc_offload.so']:
            rows.append({'path': name, 'base': ptr.contents.dlpi_addr})
        return 0
    libc = ctypes.CDLL(None)
    libc.dl_iterate_phdr.argtypes = [callback_type, ctypes.c_void_p]
    libc.dl_iterate_phdr.restype = ctypes.c_int
    assert libc.dl_iterate_phdr(visit, None) == 0
    assert len(rows) == 3 and all((sum((Path(x['path']).name == name for x in rows)) == 1 for name in ['libmf_hybm_core.so', 'libmf_smem.so', 'libmf_acc_offload.so']))
    assert next((x['base'] for x in rows if Path(x['path']).name == 'libmf_acc_offload.so')) == info.dli_fbase
    assert stock.stat().st_size == 0 and hashlib.sha256(CORE.read_bytes()).hexdigest() == '53736834872ab58915869c60ada0a719473b9b3628c112ded4dc2de8272bd126'
    assert 'torch_npu' not in sys.modules
    return (mf, mc, torch, {'single_core_instance': True, 'same_inode_alias': True, 'SDK_loader_directory': str(ALIAS.parent), 'global_optional_placeholder_unchanged': True})
