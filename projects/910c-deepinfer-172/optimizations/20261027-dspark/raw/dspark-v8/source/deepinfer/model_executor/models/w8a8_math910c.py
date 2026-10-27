# SPDX-License-Identifier: Apache-2.0
"""CPU-safe mixed-precision contracts from installed Ascend V4.1 code.

These functions do not import PLATFORM, torch_npu, TileLang or device kernels.
They are shared by the eager 910C integration and offline contract checks.
"""
from dataclasses import dataclass, field
import math

import torch
import torch.nn.functional as F


def rms_norm_mixed(x, weight, eps):
    value = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps) * weight.float()
    return value.bfloat16(), value


def rope_frequencies(config, *, compressed):
    dim = config['qk_rope_head_dim']
    base = config['compress_rope_theta'] if compressed else config['rope_theta']
    frequencies = base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    if not compressed or not config.get('rope_scaling'):
        return frequencies.reciprocal()
    yarn = config['rope_scaling']
    original = yarn['original_max_position_embeddings']
    correction = lambda beta: dim * math.log(original / (beta * 2 * math.pi)) / (2 * math.log(base))
    low = max(math.floor(correction(yarn['beta_fast'])), 0)
    high = min(math.ceil(correction(yarn['beta_slow'])), dim - 1)
    if low == high:
        high += .001
    mask = 1 - ((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low)).clamp(0, 1)
    return (1 / (yarn['factor'] * frequencies)) * (1 - mask) + frequencies.reciprocal() * mask


def device_rope_frequencies(config, device, *, compressed):
    cpu=rope_frequencies(config,compressed=compressed)
    frequencies=cpu.to(device)
    from deepinfer.npu_ops.rope910c import register
    return register(frequencies,cpu)


def rotate_partial(x, position, frequencies, *, inverse=False):
    """Interleaved complex rotation on the final RoPE channels, BF16 output."""
    dim = frequencies.numel() * 2
    if x.shape[-1] < dim:
        raise ValueError('RoPE dimension exceeds head width')
    angles = frequencies * position
    cos, sin = angles.cos(), angles.sin()
    if inverse:
        sin = -sin
    even, odd = x[..., -dim::2].float(), x[..., -dim + 1::2].float()
    rotated = torch.stack((even * cos - odd * sin, odd * cos + even * sin), -1).flatten(-2)
    return torch.cat((x[..., :-dim], rotated.to(x.dtype)), -1)


def rotate_positions(x, positions, frequencies, *, inverse=False):
    """Vectorized trailing-channel RoPE for packed prefill/decode tokens."""
    if (x.ndim < 2 or positions.shape != x.shape[:1]
            or positions.dtype not in (torch.int32, torch.int64)):
        raise ValueError('one integer RoPE position is required per packed token')
    dim = frequencies.numel()*2
    if dim > x.shape[-1]:
        raise ValueError('RoPE dimension exceeds head width')
    if x.dtype==torch.bfloat16 and hasattr(frequencies,'_native_rope_table'):
        from deepinfer.npu_ops.rope910c import apply
        return apply(x,positions,frequencies,inverse=inverse)
    angles = positions.float()[:, None]*frequencies[None]
    shape = (x.shape[0], *((1,)*(x.ndim-2)), frequencies.numel())
    cos, sin = angles.cos().reshape(shape), angles.sin().reshape(shape)
    if inverse:
        sin = -sin
    even, odd = x[..., -dim::2].float(), x[..., -dim+1::2].float()
    rotated = torch.stack((even*cos-odd*sin, odd*cos+even*sin), -1).flatten(-2)
    return torch.cat((x[..., :-dim], rotated.to(x.dtype)), -1)


@dataclass
class W8A8SequenceCache:
    """Single-sequence BF16 KV and FP32 incomplete-compressor state."""
    max_tokens: int
    swa: dict = field(default_factory=dict)
    compressed: dict = field(default_factory=dict)
    pending: dict = field(default_factory=dict)
    next_position: int = 0


def compress_step(x, position, ratio, wkv, wgate, norm_weight, eps, pending):
    """Return (completed BF16 latent or None, source RoPE position).

    C2 projects and retains FP32 K/S until the pair completes; BF16 rounding
    occurs before RMS normalization. C1 uses the BF16 projection boundary.
    """
    if ratio == 1:
        projected = F.linear(x, wkv)
        return rms_norm_mixed(projected, norm_weight, eps)[0], position
    if ratio != 2:
        raise ValueError('only source compression ratios 1 and 2 are supported')
    kv, scores = F.linear(x.float(), wkv.float()), F.linear(x.float(), wgate.float())
    if position % 2 == 0:
        if pending:
            raise ValueError('unconsumed compressor state')
        pending.extend((kv, scores))
        return None, None
    if len(pending) != 2:
        raise ValueError('C2 requires the preceding position in the same sequence')
    previous_kv, previous_scores = pending
    pooled = (torch.stack((previous_kv, kv)) * torch.stack((previous_scores, scores)).softmax(0)).sum(0).bfloat16()
    pending.clear()
    return rms_norm_mixed(pooled, norm_weight, eps)[0], position - 1


def rotated_engram_gate(hidden, kv, q_weight, k_weight, rotation, token_mask, eps):
    """Gate in the original basis; value projection stays checkpoint-rotated."""
    channels, width = hidden.shape[-2:]
    key, value = kv.split((channels * width, width), -1)
    key = key.reshape_as(hidden).float()
    original = hidden.float()
    if rotation is not None:
        original = (original.unflatten(-1, (-1, rotation.shape[0])) @ rotation.float().T).flatten(-2)
    rstd = torch.rsqrt(original.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps)
    dot = (original * (q_weight.float() * k_weight.float()) * key).sum(-1) * rstd * width ** -.5
    magnitude = dot.abs().clamp_min(1e-6).sqrt()
    gate = torch.sigmoid(torch.where(torch.signbit(dot), -magnitude, magnitude))
    gate = gate.masked_fill(~token_mask[:, None], 0.)
    return (hidden.float() + gate[..., None] * value.float()[:, None]).bfloat16()
