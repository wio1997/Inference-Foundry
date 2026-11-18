"""E4M3FN positive scale rounding without an unsupported NPU FP8 cast.

Only the compressed-KV scale conversion changes; per-16 scale, FP4 E2M1
ties-to-even, clipping and final dtype remain native. No binary replacement.
"""
import torch


def positive_e4m3_scale(x):
    x=x.clamp(min=2.0**-9,max=448.0)
    exponent=((x.contiguous().view(torch.int32)>>23)&255)-127
    step_exponent=torch.maximum(exponent-3,torch.full_like(exponent,-9))
    step=((step_exponent+127)<<23).view(torch.float32)
    scaled=x/step
    lower=torch.floor(scaled)
    fraction=scaled-lower
    odd=(lower.to(torch.int32)&1)!=0
    increment=(fraction>0.5)|((fraction==0.5)&odd)
    return ((lower+increment.to(torch.float32))*step).clamp(max=448.0)


def fake_quant_compressed_kv(x):
    from sglang.kernels.ops.quantization.fp4_fake_quant import round_fp4
    blocks=x.float().unflatten(-1,(-1,16))
    amax=blocks.abs().amax(dim=-1,keepdim=True)
    scale=positive_e4m3_scale(amax*(1.0/6.0))
    scaled=(blocks/scale).clamp(-6.0,6.0)
    return (round_fp4(scaled)*scale).flatten(-2).to(x.dtype)
