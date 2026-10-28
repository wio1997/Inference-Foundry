# SPDX-License-Identifier: Apache-2.0
"""Vectorized C1/C2 compression with per-request incomplete-pair state."""
import torch
import torch.nn.functional as F

from .w8a8_math910c import rms_norm_mixed


def compress_batch(x, batch, request_slots, ratio, wkv, wgate, norm, eps, pending):
    """Return unrotated completed latents, source positions and request slots.

    Source positions are the first token of each completed pair. A trailing
    even token retains its FP32 projection, without BF16 rounding, until the
    next chunk supplies its odd partner. Requests never share pending state.
    """
    slots = list(request_slots)
    if len(slots) != len(batch.query_lengths) or len(set(slots)) != len(slots) or x.shape[0] != sum(batch.query_lengths):
        raise ValueError('packed compressor token/request shape mismatch')
    if ratio == 1:
        latent = rms_norm_mixed(F.linear(x, wkv), norm, eps)[0]
        positions, requests = [], []
        for q, s, slot in zip(batch.query_lengths, batch.seq_lengths, slots):
            positions.extend(range(s-q, s)); requests.extend([slot]*q)
        return latent, positions, requests
    if ratio != 2 or wgate is None:
        raise ValueError('C2 requires an FP32 gate projection')
    projected, scores = F.linear(x.float(), wkv.float()), F.linear(x.float(), wgate.float())
    latents, positions, requests = [], [], []
    offset = 0
    for q, s, slot in zip(batch.query_lengths, batch.seq_lengths, slots):
        start = s-q
        kv, gate = projected[offset:offset+q], scores[offset:offset+q]
        offset += q
        if start % 2:
            previous = pending.pop(slot, None)
            if previous is None:
                raise ValueError('C2 odd chunk start lacks preceding even-token state')
            kv, gate = torch.cat((previous[0], kv)), torch.cat((previous[1], gate))
            start -= 1
        elif slot in pending:
            raise ValueError('C2 even chunk start has unconsumed pending state')
        if kv.shape[0] % 2:
            pending[slot] = (kv[-1:].clone(), gate[-1:].clone())
            kv, gate = kv[:-1], gate[:-1]
        pairs = kv.shape[0]//2
        if pairs:
            pooled = (kv.reshape(pairs, 2, -1)*gate.reshape(pairs, 2, -1).softmax(1)).sum(1).bfloat16()
            latents.append(rms_norm_mixed(pooled, norm, eps)[0])
            positions.extend(range(start, start+2*pairs, 2)); requests.extend([slot]*pairs)
    return torch.cat(latents) if latents else x.new_empty((0, norm.numel())), positions, requests
