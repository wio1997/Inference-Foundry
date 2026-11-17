"""E4M3FN positive scale rounding without an unsupported NPU FP8 cast.

Only the compressed-KV scale conversion changes; per-16 scale, FP4 E2M1
ties-to-even, clipping and final dtype remain native. No binary replacement.
"""
import torch


def positive_e4m3_scale(x):
    x=x.clamp(min=2.0**-9,max=448.0)
    exponent=torch.floor(torch.log2(x))
    step=torch.exp2(torch.maximum(exponent-3.0,torch.full_like(exponent,-9.0)))
    return (torch.round(x/step)*step).clamp(max=448.0)


def fake_quant_compressed_kv(x):
    from sglang.kernels.ops.quantization.fp4_fake_quant import round_fp4
    blocks=x.float().unflatten(-1,(-1,16))
    amax=blocks.abs().amax(dim=-1,keepdim=True)
    scale=positive_e4m3_scale(amax*(1.0/6.0))
    scaled=(blocks/scale).clamp(-6.0,6.0)
    return (round_fp4(scaled)*scale).flatten(-2).to(x.dtype)
