# SPDX-License-Identifier: Apache-2.0
"""Bounded Flash mHC diagnostic using native Torch math at fixed row geometry.

Only stateless operator rows are padded. They are not requests, model tokens,
cache slots, or persistent state. All returned values belong to real input rows.
No weight, quantization, dtype, epsilon, or Sinkhorn iteration is changed.
Graph/MTP/efficient production execution require separate qualification.
"""
import torch
from sglang.kernels.ops.layernorm import mhc

FIXED_ROWS = 32

def _check(x, hc_mult):
    if x.ndim != 2 or x.dtype != torch.bfloat16 or hc_mult != 4:
        raise ValueError('Bounded Flash mHC requires BF16 [T,H*4] and HC4')
    if x.device.type != 'npu' or x.shape[0] > 4096:
        raise ValueError('Bounded Flash mHC requires NPU and at most4096 actual rows')
    if mhc._use_tilelang_mhc_pre() or mhc._use_tilelang_mhc_post():
        raise ValueError('Fixed-row diagnostic requires the qualified native Torch mHC path')

def _pad(x, count):
    p = x.new_zeros((FIXED_ROWS, *x.shape[1:]))
    p[:count].copy_(x)
    return p

def flash_hc_pre_fixed(x, hc_fn, hc_scale, hc_base, hc_mult, rms_eps,
                       hc_eps, sinkhorn_iters, post_mult_value=2.0,
                       hc_norm_weight=None, out_norm_weight=None, out_norm_eps=None):
    _check(x,hc_mult)
    if x.shape[1] != 16384 or hc_fn.dtype != torch.float32 or hc_scale.dtype != torch.float32 or hc_base.dtype != torch.float32:
        raise ValueError('Flash H4096 mHC params must retain native FP32 storage')
    if not x.shape[0]:
        return mhc.hc_pre(x,hc_fn,hc_scale,hc_base,hc_mult,rms_eps,hc_eps,
                         sinkhorn_iters,post_mult_value,hc_norm_weight,out_norm_weight,out_norm_eps)
    pieces = [[],[],[]]
    for lo in range(0,x.shape[0],FIXED_ROWS):
        count = min(FIXED_ROWS,x.shape[0]-lo)
        result = mhc.hc_pre(_pad(x[lo:lo+count],count),hc_fn,hc_scale,hc_base,
            hc_mult,rms_eps,hc_eps,sinkhorn_iters,post_mult_value,
            hc_norm_weight,out_norm_weight,out_norm_eps)
        if result[3] is not False:
            raise ValueError('Unexpected fused normalization in qualified native Torch path')
        for i in range(3):
            pieces[i].append(result[i][:count])
    return tuple(torch.cat(p,dim=0) for p in pieces)+(False,)

def flash_hc_post_fixed(x, residual, h_post, h_res, hc_mult):
    _check(residual,hc_mult)
    if x.shape != (residual.shape[0],4096) or residual.shape[1]!=16384:
        raise ValueError('Flash mHC post requires actual matching H4096 residual rows')
    if not x.shape[0]:
        return mhc.hc_post(x,residual,h_post,h_res,hc_mult)
    pieces=[]
    for lo in range(0,x.shape[0],FIXED_ROWS):
        count=min(FIXED_ROWS,x.shape[0]-lo)
        args=[_pad(t[lo:lo+count],count) for t in (x,residual,h_post,h_res)]
        pieces.append(mhc.hc_post(*args,hc_mult)[:count])
    return torch.cat(pieces,dim=0)
