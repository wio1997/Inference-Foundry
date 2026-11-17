# SPDX-License-Identifier: Apache-2.0
"""G5 unfused functional candidate; no custom ACLNN dependency.

The two private numerical functions below are copied verbatim from the pinned
installed SGLang kernels/ops/layernorm/mhc.py lines 1901-1958. They use FP32
projection/reductions/mixes and cast the materialized branch/output to BF16.
This restores a mathematical execution path, not a performance claim.
"""
from typing import Tuple
import torch

ABI_VERSION = "glm53_mhc_g5_native_torch_unfused"

def _mhc_pre_torch(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    import torch.nn.functional as F

    s, n, h = residual.shape
    dtype = residual.dtype

    x_flat = residual.view(s, n * h).float()
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + rms_eps)
    mixes = F.linear(x_flat, fn) * rsqrt

    pre_raw = mixes[:, :n]
    post_raw = mixes[:, n : 2 * n]
    comb_raw = mixes[:, 2 * n :].view(s, n, n)
    pre_base = hc_base[:n]
    post_base = hc_base[n : 2 * n]
    comb_base = hc_base[2 * n :].view(n, n)

    pre = torch.sigmoid(pre_raw * hc_scale[0] + pre_base) + hc_pre_eps
    post = hc_post_mult_value * torch.sigmoid(post_raw * hc_scale[1] + post_base)
    comb = comb_raw * hc_scale[2] + comb_base

    comb = comb.softmax(-1) + hc_sinkhorn_eps
    comb = comb / (comb.sum(-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + hc_sinkhorn_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + hc_sinkhorn_eps)

    layer_input = (pre.unsqueeze(-1) * residual.float()).sum(dim=1).to(dtype)
    return post.unsqueeze(-1), comb, layer_input


def _mhc_post_torch(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    out = post_layer_mix * x.unsqueeze(1) + (
        comb_res_mix.unsqueeze(-1) * residual.unsqueeze(2)
    ).sum(dim=1)
    return out.type_as(x)


def _validate_pre(x, fn, scale, base, n):
    if n != 4 or x.ndim != 2 or x.shape[1] % n or x.dtype != torch.bfloat16:
        raise ValueError("Requires GLM BF16[T,4*H], HC4")
    if tuple(fn.shape) != (n*(n+2), x.shape[1]) or tuple(scale.shape) != (3,) or tuple(base.shape) != (n*(n+2),):
        raise ValueError("Invalid checkpoint mHC coefficient shapes")
    if any(t.dtype != torch.float32 for t in (fn, scale, base)):
        raise TypeError("Checkpoint mHC fn/scale/base must remain FP32")
    if any(t.device != x.device for t in (fn, scale, base)):
        raise ValueError("mHC inputs must share device")


def hc_pre(x, hc_fn, hc_scale, hc_base, hc_mult, rms_eps, hc_eps, sinkhorn_iters,
           post_mult_value=2.0, hc_norm_weight=None, out_norm_weight=None, out_norm_eps=None):
    _validate_pre(x, hc_fn, hc_scale, hc_base, hc_mult)
    if not isinstance(sinkhorn_iters, int) or sinkhorn_iters < 1:
        raise ValueError("Sinkhorn iterations must be a positive Python int")
    t, total = x.shape
    h = total // hc_mult
    if t == 0:
        return (x.new_zeros((0,h)), torch.zeros((0,16), device=x.device, dtype=torch.float32),
                torch.zeros((0,4), device=x.device, dtype=torch.float32), False)
    # Match installed SGLang: fold only optional input HC norm into fn.
    # Output RMSNorm stays with the decoder, indicated by norm_fused=False.
    fn = hc_fn if hc_norm_weight is None else hc_fn * hc_norm_weight
    post, comb, layer_input = _mhc_pre_torch(
        x.reshape(t,hc_mult,h), fn, hc_scale, hc_base, rms_eps, hc_eps,
        hc_eps, post_mult_value, sinkhorn_iters)
    return layer_input, comb.reshape(t,hc_mult*hc_mult), post.reshape(t,hc_mult), False


def hc_post(x, residual, h_post, h_res, hc_mult):
    if hc_mult != 4 or x.ndim != 2 or x.dtype != torch.bfloat16 or residual.dtype != x.dtype:
        raise ValueError("Requires HC4 BF16 branch and residual")
    t,h = x.shape
    if tuple(residual.shape) != (t,4*h) or tuple(h_post.shape) != (t,4) or tuple(h_res.shape) != (t,16):
        raise ValueError("Invalid flat SGLang mHC post contract")
    if h_post.dtype != torch.float32 or h_res.dtype != torch.float32:
        raise TypeError("Mix coefficients must remain FP32")
    if any(v.device != x.device for v in (residual,h_post,h_res)):
        raise ValueError("mHC inputs must share device")
    if t == 0:
        return x.new_zeros((0,4*h))
    # comb[in,out], reduce input streams. FP32 coefficient multiplication
    # promotes BF16 residual before reduction, exactly as installed source.
    return _mhc_post_torch(x, residual.reshape(t,4,h), h_post.reshape(t,4,1),
                           h_res.reshape(t,4,4)).reshape(t,4*h)
