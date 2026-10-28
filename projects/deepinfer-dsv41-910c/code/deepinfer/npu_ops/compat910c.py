"""Small correctness implementations for the opt-in 910C profile.

Only BF16/FP32 compute and explicit byte-format decode are supported here.
These functions do not emulate the 950 runtime, DeepEP or a model loader.
"""
from __future__ import annotations

import torch


def bf16_gemm_nt(a: torch.Tensor, b: torch.Tensor, d: torch.Tensor, c: torch.Tensor | None = None) -> None:
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("expected A[M,K] and B[N,K]")
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise TypeError("910C reference GEMM requires BF16 operands")
    if d.shape != (a.shape[0], b.shape[0]) or d.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("output must be BF16/FP32 [M,N]")
    if a.device != b.device or a.device != d.device:
        raise ValueError("all operands must share a device")
    if c is not None and (c.shape not in (d.shape, (1, d.shape[1])) or c.dtype != d.dtype or c.device != d.device):
        raise ValueError("C must match D or be a [1,N] bias with D dtype/device")
    # Preserve FP32 accumulation and defer output rounding until C is added.
    result = torch.mm(a.float(), b.float().T)
    if c is not None:
        result = result + c.float()
    d.copy_(result.to(d.dtype))


def norm(
    x: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor | None = None,
    eps: float = 1e-6, out: torch.Tensor | None = None, *, fmt: str = "",
    num_per_channels: int = 0, use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False, use_packed_ue8m0: bool = False,
    extra_sf_layout: str | None = None, extra_sf_out: torch.Tensor | None = None,
    extra_sf_block_m: int = 0, return_bf16: bool = False, out_scale: float = 1.0,
):
    if fmt or num_per_channels or use_tma_aligned_col_major_sf or round_sf or use_packed_ue8m0 or extra_sf_layout or extra_sf_out is not None or extra_sf_block_m:
        raise NotImplementedError("910C reference RMSNorm supports unquantized output only")
    if x.ndim not in (2, 3) or x.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("expected BF16/FP32 [M,N] or [B,M,N]")
    if weight.shape != (x.shape[-1],) or weight.device != x.device or weight.dtype != x.dtype:
        raise ValueError("weight must match the input last dimension, dtype and device")
    if residual is not None and (residual.shape != x.shape or residual.dtype != x.dtype or residual.device != x.device):
        raise ValueError("residual must match input")
    if out is not None and (out.shape != x.shape or out.dtype != x.dtype or out.device != x.device):
        raise ValueError("out must match input")
    # The upstream contract rounds residual addition to x.dtype before statistics.
    residual_out = x + residual if residual is not None else None
    values = x if residual_out is None else residual_out
    fp32 = values.float()
    result = ((fp32 * torch.rsqrt(fp32.square().mean(-1, keepdim=True) + eps) * weight.float()) * out_scale).to(x.dtype)
    if out is not None:
        out.copy_(result); result = out
    output = (result, result.to(torch.bfloat16)) if return_bf16 else result
    return (output, residual_out) if residual is not None else output


def decode_e4m3_bytes(data: torch.Tensor) -> torch.Tensor:
    """Decode E4M3FN bytes using FP32 ops, including signed zero and NaN.

    FP8 tensor allocation/casts are deliberately avoided on 910C.
    """
    if data.dtype != torch.uint8:
        raise TypeError("E4M3FN storage must be uint8 bytes")
    bits = data.to(torch.int32)
    exponent = (bits >> 3) & 15
    mantissa = bits & 7
    # Construct normal FP32 representations exactly; NPU Pow approximates
    # powers of two and fails bit-exact decoding even for tiny FP8 values.
    normal = (((exponent + 120) << 23) | (mantissa << 20)).contiguous().view(torch.float32)
    magnitude = torch.where(exponent == 0, mantissa.float() * (2.0 ** -9), normal)
    magnitude = torch.where((exponent == 15) & (mantissa == 7), float("nan"), magnitude)
    return torch.where(bits >= 128, -magnitude, magnitude)


def decode_e2m1_bytes(data: torch.Tensor) -> torch.Tensor:
    """Decode packed E2M1, low nibble before high nibble, along the last dim."""
    if data.dtype not in (torch.uint8, torch.int8):
        raise TypeError("FP4 storage must contain packed bytes")
    bits = data.to(torch.int32) & 255
    nibbles = torch.stack((bits & 15, bits >> 4), -1).flatten(-2)
    index = nibbles & 7
    # Explicit selection is portable to NPU and independent of float4 support.
    magnitude = torch.zeros_like(nibbles, dtype=torch.float32)
    for i, value in enumerate((0., .5, 1., 1.5, 2., 3., 4., 6.)):
        magnitude = torch.where(index == i, value, magnitude)
    return torch.where(nibbles >= 8, -magnitude, magnitude)


def decode_block_weight(data: torch.Tensor, scales: torch.Tensor, *, fmt: str, block_shape: tuple[int, int] = (32, 32), scale_fmt: str = "ue8m0") -> torch.Tensor:
    """Decode one untransposed HF weight matrix to BF16, no implicit format guess.

    Scales are unpacked [ceil(N/br),ceil(K/bc)] bytes (UE8M0) or FP32.
    Kernel-specific packed int16 MN-major scales are deliberately rejected.
    """
    if data.ndim != 2 or scales.ndim != 2 or data.device != scales.device:
        raise ValueError("expected weight/scales matrices on the same device")
    if fmt == "e4m3": values = decode_e4m3_bytes(data)
    elif fmt == "e2m1": values = decode_e2m1_bytes(data)
    else: raise ValueError(f"unsupported weight encoding: {fmt}")
    br, bc = block_shape
    if br <= 0 or bc <= 0: raise ValueError("block dimensions must be positive")
    n,k = values.shape
    if scales.shape != ((n+br-1)//br,(k+bc-1)//bc):
        raise ValueError("scale shape does not match the decoded matrix and block geometry")
    if scale_fmt == "ue8m0":
        if scales.dtype != torch.uint8: raise TypeError("UE8M0 scales require unpacked bytes")
        sf = (scales.to(torch.int32) << 23).contiguous().view(torch.float32)
        sf = torch.where(scales == 0, 2.0 ** -127, sf)
        sf = torch.where(scales == 255, float("nan"), sf)
    elif scale_fmt == "fp32":
        if scales.dtype != torch.float32: raise TypeError("FP32 scale format requires FP32 storage")
        sf = scales
    else: raise ValueError(f"unsupported scaling format: {scale_fmt}")
    return (values * sf.repeat_interleave(br,0).repeat_interleave(bc,1)[:n,:k]).to(torch.bfloat16)
