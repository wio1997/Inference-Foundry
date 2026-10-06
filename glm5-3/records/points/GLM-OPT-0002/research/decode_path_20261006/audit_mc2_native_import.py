"""Fresh-process CPU import/ABI gate; never calls a device or model operator.

The launcher must put the candidate directory first in LD_LIBRARY_PATH before
starting Python. A candidate must not be loaded into an already imported model.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--library", type=Path, required=True)
parser.add_argument("--sha256", required=True)
parser.add_argument("--stock-schema", type=Path)
args = parser.parse_args()
expected = args.library.resolve()
assert hashlib.sha256(expected.read_bytes()).hexdigest() == args.sha256

import torch
import torch_npu

assert not torch.npu.is_initialized()
mapped = sorted({str(Path(line.split(maxsplit=5)[5]).resolve())
                 for line in Path("/proc/self/maps").read_text().splitlines()
                 if len(line.split(maxsplit=5)) == 6
                 and line.split(maxsplit=5)[5].endswith("/libtorch_npu.so")})
assert mapped == [str(expected)], mapped
assert hashlib.sha256(expected.read_bytes()).hexdigest() == args.sha256
schemas = {}
for name in ("npu_moe_distribute_dispatch_v2", "npu_moe_distribute_combine_v2"):
    schemas[name] = str(getattr(torch.ops.npu, name).default._schema)
if args.stock_schema:
    assert schemas == json.loads(args.stock_schema.read_text())["schemas"]

# Same lookup used by the measured CPU predicate. Do not invoke a kernel,
# GetWorkspaceSize function, stream, device query, or NPU tensor allocation.
library = ctypes.CDLL(str(expected))
lookup = getattr(library, "_Z16GetOpApiFuncAddrPKc")
lookup.argtypes = [ctypes.c_char_p]
lookup.restype = ctypes.c_void_p
symbols = {name: bool(lookup(name.encode())) for name in (
    "aclnnMoeDistributeDispatchV4", "aclnnMoeDistributeDispatchV4GetWorkspaceSize",
    "aclnnMoeDistributeCombineV4", "aclnnMoeDistributeCombineV4GetWorkspaceSize")}
assert all(symbols.values()), symbols
assert not torch.npu.is_initialized()
print(json.dumps(dict(
    library=str(expected), library_sha256=args.sha256,
    mapped_libtorch_npu=mapped, torch_version=torch.__version__,
    torch_npu_version=torch_npu.__version__,
    cxx11_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI), schemas=schemas,
    V4_symbols=symbols, NPU_initialized=False, model_request=False,
    CPU_import_passed=True, model_correctness=False, matched_AB=False,
    limits=["Fresh-process load/schema/lookup gate only; no device/model correctness.",
            "All-rank resident library and selector witnesses are required before timing."],
)))
