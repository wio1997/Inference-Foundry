"""Native CPU comparison of one fixed capability predicate and its cache."""
import ctypes
import hashlib
import json
import pathlib
import statistics
import time
import torch
import torch_npu

ROOT = pathlib.Path(__file__).resolve().parent
assert not torch.npu.is_initialized()
path = pathlib.Path(torch_npu.__file__).parent / "lib/libtorch_npu.so"
assert hashlib.sha256(path.read_bytes()).hexdigest() == (
    "83fb9a0eb249aef6bca7f8463047fcb4d3062cc4849f17ddf31a5e050c838842")
installed = ctypes.CDLL(str(path))
lookup = getattr(installed, "_Z16GetOpApiFuncAddrPKc")
lib = ctypes.CDLL(str(ROOT / "mc2_cache_cpu.so"))
lib.set_lookup.argtypes = [ctypes.c_size_t]
lib.set_lookup.restype = ctypes.c_int
assert lib.set_lookup(ctypes.cast(lookup, ctypes.c_void_p).value) == 0
assert lib.correctness() == 0
lib.run_predicate.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                              ctypes.POINTER(ctypes.c_uint64)]
lib.run_predicate.restype = ctypes.c_uint64
rows = []
for op, name in enumerate(("Dispatch", "Combine")):
    for mode in (0, 1):
        ns = ctypes.c_uint64()
        assert lib.run_predicate(op, mode, 20, ctypes.byref(ns)) == 20
    samples = {0: [], 1: []}
    for repeat in range(11):
        for mode in ((0, 1) if repeat % 2 == 0 else (1, 0)):
            ns = ctypes.c_uint64()
            cpu = time.process_time_ns()
            assert lib.run_predicate(op, mode, 100, ctypes.byref(ns)) == 100
            samples[mode].append(dict(native_wall_us=ns.value / 100 / 1000,
                process_cpu_us=(time.process_time_ns() - cpu) / 100 / 1000))
    rows.append(dict(operation=name, samples=samples,
        medians={str(mode): statistics.median(z["native_wall_us"]
                     for z in samples[mode]) for mode in (0, 1)}))
assert not torch.npu.is_initialized()
print(json.dumps(dict(rows=rows, NPU_initialized=False, model_request=False,
    library_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    helper_sha256=hashlib.sha256((ROOT / "mc2_cache_cpu.so").read_bytes()).hexdigest(),
    correctness=dict(concurrent_first_call_threads=8, calls_per_case=8000,
                     positive_queries=2, missing_api_queries=1,
                     missing_workspace_queries=2, passed=True),
    limits=["Same exported installed lookup, source-identical native predicate body; no Python timing loop per predicate.",
            "Standalone host process, not resident model critical-path timing or measured E2E saving.",
            "Libraries remain fixed for worker lifetime; library replacement requires restart."])))
