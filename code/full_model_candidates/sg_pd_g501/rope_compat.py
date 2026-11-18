"""Reuse native FP32 complex RoPE on the existing quantized position cache.

Integration candidate: operator compatibility, no performance/quality acceptance.
"""
import torch
from sglang.srt.layers.attention.dsv4.dsv41_sparse import rope_tail


def apply_rotary_mul_inplace(q_rope, kv_rope, cos4, sin4, qk_nope_dim=0):
    assert q_rope.device.type == 'npu'
    assert q_rope.dtype == cos4.dtype == sin4.dtype == torch.bfloat16
    dim = cos4.shape[-1]
    assert dim == 64 and tuple(cos4.shape) == tuple(sin4.shape)
    assert cos4.shape[1:3] == (1, 1)
    # Keep the same BF16 cache values and inverse sign; never regenerate angles.
    freq = torch.complex(cos4[:, 0, 0, ::2].float(), sin4[:, 0, 0, ::2].float())
    for x in (q_rope, kv_rope):
        if x is None:
            continue
        assert x.dtype == q_rope.dtype and x.device == q_rope.device
        assert x.shape[0] == cos4.shape[0] and x.shape[-1] == qk_nope_dim + dim
        tail = x[..., qk_nope_dim:]
        # Native rope_tail casts inputs to FP32, rotates, then casts once.
        # Mutate only the tail to preserve Q/K aliases and non-rotary features.
        tail.copy_(rope_tail(tail, freq, dim))
