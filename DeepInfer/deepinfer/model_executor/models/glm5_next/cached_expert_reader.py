# SPDX-License-Identifier: Apache-2.0
"""Opt-in ModelSlim reader adapter for the existing selected-expert bank.

Routed expert payloads are read via a rank-owned, byte-bounded host LRU.
Shared experts and non-expert tensors use the underlying reader unchanged.
This is a local loader seam: it does not dispatch routes, communicate, or make
non-owned experts available. The caller must explicitly assemble valid routes.
"""
import re
import torch
from .rank_expert_cache import ExpertSliceKey, RankExpertHostCache

_ROUTED = re.compile(r'^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)$')

class CachedRankExpertReader:
    def __init__(self, reader, cache, *, max_assembled_bytes):
        if not isinstance(cache, RankExpertHostCache) or cache.reader is not reader:
            raise ValueError('cache must belong to the supplied reader')
        if type(max_assembled_bytes) is not int or max_assembled_bytes < 1:
            raise ValueError('positive explicit assembled-payload budget required')
        self.reader, self.cache, self.max_assembled_bytes = reader, cache, max_assembled_bytes

    def __getattr__(self, name):
        return getattr(self.reader, name)

    def linear_payload(self, prefix, *, scale_dtype=torch.bfloat16, output_slice=None):
        match = _ROUTED.fullmatch(prefix)
        if match is None:
            # Only shared/non-expert prefixes may fall back. Malformed expert
            # names must never bypass the ownership check.
            if '.mlp.experts.' in prefix:
                raise ValueError('invalid routed expert prefix')
            return self.reader.linear_payload(prefix, scale_dtype=scale_dtype, output_slice=output_slice)
        layer, expert, projection = int(match[1]), int(match[2]), match[3]
        if scale_dtype not in (torch.bfloat16, torch.float32):
            raise ValueError('expected BF16/F32 runtime scale dtype')
        if expert % self.cache.world_size != self.cache.rank:
            raise ValueError('routed expert is not owned by this rank')
        descriptor = self.reader.descriptor(prefix+'.weight')[2]
        if descriptor['dtype'] != 'I8' or len(descriptor['shape']) != 2:
            raise ValueError('expected expert INT8 matrix')
        rows, width = descriptor['shape']
        if output_slice is None:
            start, stop = 0, rows
        elif isinstance(output_slice, slice) and output_slice.step in (None, 1):
            start, stop, _ = output_slice.indices(rows)
        else:
            raise ValueError('output sharding requires a contiguous row slice')
        count = stop-start
        if count < 1:
            raise ValueError('nonempty expert slice required')
        # Limit output weight + runtime scales. The LRU, returned copies,
        # conversion scratch and allocator overhead are separate limits.
        if count*(width+torch.empty((),dtype=scale_dtype).element_size()) > self.max_assembled_bytes:
            raise ValueError('assembled expert payload exceeds explicit budget')
        chunk = min(self.cache.max_bytes//(width+8), self.reader.max_read_bytes//width,
                    self.reader.max_read_bytes//4)
        if chunk < 1:
            raise ValueError('cache or reader budget cannot hold one row')
        weight = torch.empty((count,width),dtype=torch.int8)
        scales = torch.empty((count,),dtype=scale_dtype)
        for first in range(start, stop, chunk):
            last=min(first+chunk,stop)
            part=self.cache.read(ExpertSliceKey(layer,expert,projection,first,last))
            rounded=part.scale.flatten().to(scale_dtype)
            if not bool(torch.isfinite(rounded).all() and (rounded>0).all()):
                raise ValueError('scale invalid after runtime dtype conversion')
            weight[first-start:last-start].copy_(part.weight)
            scales[first-start:last-start].copy_(rounded)
        return weight, scales
