"""Select existing native Triton split/Sinkhorn on NPU, preserving equations."""
import torch
from sglang.kernels.ops.layernorm.mhc import _hc_split_sinkhorn_triton


def split_sinkhorn(mixes, scale, base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    assert mixes.device.type == 'npu' and mixes.dtype == torch.float32
    assert mixes.ndim == 3 and mixes.shape[-1] == (2+hc_mult)*hc_mult
    assert hc_mult == 4 and sinkhorn_iters == 20
    b,s,_=mixes.shape
    if b*s==0:
        return (mixes.new_empty(b,s,hc_mult),mixes.new_empty(b,s,hc_mult),mixes.new_empty(b,s,hc_mult,hc_mult))
    # The native kernel indexes flat contiguous rows; make that contract explicit.
    return _hc_split_sinkhorn_triton(mixes.contiguous(),scale,base,hc_mult,sinkhorn_iters,eps)
