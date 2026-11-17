import os

import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
opp_path = os.path.join(current_dir, "vendors", "hwcomputing")
lib_path = os.path.join(current_dir, "vendors", "hwcomputing", "op_api", "lib")
# Set environment variables related to custom operators
if os.environ.get("SGLANG_DSV41_FUSED_MOE_ENABLE") == "1":
    _fused_provider = os.environ["SGLANG_DSV41_FUSED_MOE_PROVIDER"]
    _existing = [p for p in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(":")
                 if p and p not in {_fused_provider, opp_path}]
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = ":".join(_existing + [_fused_provider, opp_path])
else:
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = f"{opp_path}:{os.environ.get('ASCEND_CUSTOM_OPP_PATH', '')}"
os.environ["LD_LIBRARY_PATH"] = f"{lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"

# Preload libcust_opapi.so by full path so that subsequent bare-name
# dlopen("libcust_opapi.so") calls (used by EXEC_NPU_CMD in the C++ layer)
# can find it via SONAME lookup.  glibc caches LD_LIBRARY_PATH at process
# startup, so the os.environ update above is not sufficient on its own.
import ctypes

_cust_lib = os.path.join(lib_path, "libcust_opapi.so")
if os.path.exists(_cust_lib):
    try:
        ctypes.CDLL(_cust_lib)
    except OSError:
        pass

from deep_ep_cpp import Config

# Import strategies to register them
from . import strategies
from .buffer import Buffer
from .ep_strategy import LowLatencyStrategy, NormalStrategy
from .utils import EventOverlap
