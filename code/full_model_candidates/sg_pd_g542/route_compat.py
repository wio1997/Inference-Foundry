"""Reuse native wrapper and fused router with a pinned Ascend math lowering."""
from pathlib import Path
import hashlib,inspect,json,types
_FUSED=None
def get_gate(root):
 global _FUSED
 if _FUSED is not None:return _FUSED
 import sglang
 package=Path(sglang.__file__).resolve().parent.parent
 for name,h in json.loads((root/'native_route_sha256.json').read_text()).items():assert hashlib.sha256((package/name).read_bytes()).hexdigest()==h,name
 import sglang.kernels.ops.moe.moe_fused_gate as native
 from route_kernel import _router_triton_kernel
 prior=inspect.unwrap(native.moe_fused_gate);assert '_router_triton_kernel' in prior.__globals__
 env=dict(prior.__globals__);env['_router_triton_kernel']=_router_triton_kernel
 _FUSED=types.FunctionType(prior.__code__,env,prior.__name__,prior.__defaults__,prior.__closure__);_FUSED.__kwdefaults__=prior.__kwdefaults__
 return _FUSED
