"""Task-only norm2 adapter to the existing donor fused A3 router; no new math."""
import os
from functools import cache
from pathlib import Path


@cache
def get_router_op():
    import torch
    if os.environ.get("SGLANG_NPU_V41_A3_COMPAT") != "1":
        raise RuntimeError("Router donor bridge requires explicit task compatibility gate")
    path=Path(os.environ["SGLANG_V41_A3_ROUTER_EXTENSION"])
    if not path.is_absolute() or not path.is_file():
        raise ValueError("Set an existing absolute target-Torch2.13 router extension path")
    torch.ops.load_library(str(path.resolve()))
    return torch.ops.sglang_v41_router_g2.moe_gating_top_k


def moe_gating_top_k_norm2(x,k,*,bias=None,input_ids=None,tid2eid=None,
        k_group=1,group_count=1,group_select_mode=0,renorm=0,norm_type=2,
        out_flag=False,routed_scaling_factor=1.0,eps=1e-20):
    import torch
    if norm_type!=2 or input_ids is not None or tid2eid is not None:
        raise ValueError("This bridge serves the exact non-hash sqrtsoftplus route")
    if x.dtype!=torch.float32 or x.ndim!=2 or (x.shape[1],k) not in ((384,6),(128,3)):
        raise ValueError("Unexpected target/draft router geometry or dtype")
    if (k_group,group_count,group_select_mode,renorm)!=(1,1,0,0):
        raise ValueError("This task uses ungrouped normalized sqrtsoftplus routing")
    if bias is not None and (bias.dtype!=x.dtype or tuple(bias.shape)!=(x.shape[1],)):
        raise ValueError("Router bias must match the native FP32 expert vector")
    weights,ids,scores=get_router_op()(x,k,k_group,group_count,group_select_mode,
        renorm,norm_type,out_flag,float(routed_scaling_factor),float(eps),bias)
    return weights.to(x.dtype),ids.to(torch.int32),(
        scores if out_flag else torch.empty(0,dtype=torch.float32,device=x.device))
