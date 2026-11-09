# SPDX-License-Identifier: Apache-2.0
"""910C local E2M1/UE8M0 adapter with pinned-reference validated arithmetic.

UE8M0 conversion follows DeepGEMM-Ascend tests/test_attention.py at
3a35326271b069b8b5bf1a754435acd6cc91575b. E2M1 ties and packed bytes
were compared against TileKernels2.1.0 CPU cast. Model integration is opt-in.
"""
import torch


def ceil_to_ue8m0_with_byte(x):
    bits = x.abs().float().view(torch.int)
    exp = (((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()).clamp(1, 254)
    return (exp << 23).view(torch.float), exp.to(torch.uint8)


def quantize_group32(x):
    if x.dtype != torch.bfloat16 or x.shape[-1] != 128:
        raise ValueError('expected finite BF16 rows with 128 channels')
    group = x.float().reshape(*x.shape[:-1], 4, 32)
    amax = group.abs().amax(-1).clamp_min(6.0 * 2.0**-126)
    # Reuse the official reference bit conversion. NPU log2/ceil can move
    # exact scale boundaries up by an exponent; avoid transcendental ops here.
    scales, _ = ceil_to_ue8m0_with_byte(amax / 6.0)
    magnitude = group.abs() / scales[..., None]
    codes = torch.zeros_like(group, dtype=torch.int8)
    # Nearest E2M1, ties to even encoding. Comparisons differ at alternate ties.
    for threshold, inclusive, code in ((.25,False,1),(.75,True,2),
                                       (1.25,False,3),(1.75,True,4),
                                       (2.5,False,6),(3.5,True,8),(5.,False,12)):
        selected = magnitude >= threshold if inclusive else magnitude > threshold
        codes = torch.where(selected, code, codes)
    return torch.where(group < 0, -codes, codes), scales


def pack_group32(raw, codes, scales):
    if raw.dtype != torch.bfloat16 or raw.shape[-1] != 128:
        raise ValueError('expected BF16 rows with128channels')
    magnitude = codes.reshape_as(raw).to(torch.int32).abs()
    bits = torch.where(magnitude <= 4, magnitude,
                       torch.where(magnitude == 6, 5,
                                   torch.where(magnitude == 8, 6, 7)))
    sign = (raw.view(torch.int16).to(torch.int32) & 0x8000) != 0
    bits = (bits | (sign.to(torch.int32) << 3)).to(torch.uint8)
    pairs = bits.reshape(*raw.shape[:-1], 64, 2)
    values = pairs[..., 0] | (pairs[..., 1] << 4)
    scale_bytes = (scales.contiguous().view(torch.int32) >> 23).to(torch.uint8)
    return values.view(torch.int8), scale_bytes


def unpack_group32(values, scale_bytes, *, levels=None):
    if values.dtype != torch.int8 or values.shape[-1] != 64:
        raise ValueError('expected64packedE2M1bytes')
    values = values.view(torch.uint8)
    bits = torch.stack((values & 15, values >> 4),dim=-1).reshape(*values.shape[:-1],128)
    if levels is None:
        levels = torch.tensor([0,1,2,3,4,6,8,12],dtype=torch.int8,device=values.device)
    codes = levels[(bits & 7).long()]
    codes = torch.where((bits & 8) != 0,-codes,codes).reshape(*values.shape[:-1],4,32)
    scales = (scale_bytes.to(torch.int32) << 23).view(torch.float32)
    return codes, scales
