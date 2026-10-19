# SPDX-License-Identifier: Apache-2.0
"""Bounded eager correctness candidate: own compression inputs through score.

No new kernel math. No CPU tensor reads/hashes. Not Graph/80K/performance
qualified. Fresh allocation removes original storage alias/registration;
physical NPU format and real-model outcome remain runtime questions.
"""
from __future__ import annotations

import torch

_NAMES = ('tail_key', 'tail_score', 'cache', 'k', 'gate', 'ape',
          'positions', 'query_ends', 'seq_lens', 'tail_slots',
          'tail_table', 'index_slots')
_MAX_BYTES = 16 * 1024 * 1024


def owned_compress(compress, tensors, index_kpool, *, audit_context=None):
    """Run unchanged native callable; commit only exactly writable row sets.

    Caller must retain returned tensors until immediate score launch. Existing
    model input/ownership/stream fences remain mandatory. Copying full cache
    or full tails back would overwrite unrelated concurrent request writes.
    """
    if set(tensors) != set(_NAMES) or index_kpool != 4:
        raise ValueError('Owned candidate requires the complete KPool4 boundary')
    total = sum(t.numel() * t.element_size() for t in tensors.values())
    if total > _MAX_BYTES or tensors['k'].shape[0] > 4096:
        raise ValueError('Owned eager diagnostic budget exceeded')
    if not all(t.is_contiguous() for t in tensors.values()):
        raise ValueError('This candidate covers original contiguous logical views only')
    devices = {t.device for t in tensors.values()}
    if len(devices) != 1:
        raise ValueError('Mixed device boundary')
    # .contiguous() may return the original alias. Always allocate and copy.
    owned = {name: torch.empty(tuple(t.shape), dtype=t.dtype, device=t.device,
                              memory_format=torch.contiguous_format).copy_(t)
             for name, t in tensors.items()}
    count, dim = owned['k'].shape
    cache = owned['cache']
    tail = owned['tail_key']
    if cache.ndim != 4 or cache.shape[2] != 1 or cache.shape[3] != dim:
        raise ValueError('Unexpected paged indexer layout')
    if tail.ndim != 3 or tail.shape[-1] != dim or tail.shape[1] != 4:
        raise ValueError('Unexpected tail layout')
    if owned['tail_score'].shape != tail.shape or owned['positions'].numel() != count:
        raise ValueError('Boundary cardinality mismatch')
    rows = torch.arange(count, dtype=torch.int64, device=cache.device)
    pos = owned['positions'].reshape(-1)
    slots = owned['index_slots'].reshape(-1).to(torch.int64)
    # Exactly the original compression kernel's write predicate. A valid slot
    # on a non-completion token MUST NOT be copied back.
    valid = ((slots >= 0) & (slots < cache.shape[0] * cache.shape[1])
             & (rows < owned['query_ends'][-1]) & (pos >= 0)
             & ((pos + 1).remainder(index_kpool) == 0))
    cache_rows = torch.unique(slots[valid])
    ends = owned['query_ends'].reshape(-1).to(torch.int64)
    starts = torch.cat((torch.zeros_like(ends[:1]), ends[:-1]))
    retained = ends[:, None] - 1 - torch.arange(tail.shape[1], device=tail.device)[None, :]
    valid_retained = ((retained >= starts[:, None]) & (retained >= 0) & (retained < count))
    retained = retained[valid_retained]
    tail_slots = owned['tail_slots'].reshape(-1).index_select(0, retained).to(torch.int64)
    retained_pos = pos.index_select(0, retained)
    tail_rows = torch.unique(tail_slots[(retained_pos >= 0) & (tail_slots >= 0)
                                        & (tail_slots < tail.shape[0] * tail.shape[1])])
    args = [owned[name] for name in _NAMES]
    if audit_context is None:
        compress(*args, index_kpool)
    else:
        compress(*args, index_kpool, audit_context=audit_context)
    # If native callable raises, no persistent commit occurs. Both real kernel
    # launches and these copies use the existing current stream ordering.
    tensors['cache'].view(-1, dim).index_copy_(
        0, cache_rows, owned['cache'].view(-1, dim).index_select(0, cache_rows))
    for name in ('tail_key', 'tail_score'):
        tensors[name].view(-1, dim).index_copy_(
            0, tail_rows, owned[name].view(-1, dim).index_select(0, tail_rows))
    return owned
