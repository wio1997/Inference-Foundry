# SPDX-License-Identifier: Apache-2.0
"""Token-axis ownership for native TP attention and sequence-parallel trunk.

Attention caches retain full request coordinates on every TP rank. Only the
mHC residual stream, normalization and FFN own disjoint token slices; gather
normalized attention inputs and reduce-scatter the partial output projection.
"""
import torch.distributed as dist


class SequenceShard:
    def __init__(self, group):
        self.group = group
        self.world = dist.get_world_size(group) if group is not None else 1
        self.rank = dist.get_rank(group) if group is not None else 0

    def _pad(self, value, *, fill=0):
        tokens = value.shape[0]
        width = (tokens + self.world - 1) // self.world
        if width * self.world == tokens:
            return value.contiguous(), width
        padded = value.new_full((width * self.world, *value.shape[1:]), fill)
        padded[:tokens].copy_(value)
        return padded, width

    def split(self, value, *, fill=0):
        padded, width = self._pad(value, fill=fill)
        return padded[self.rank * width:(self.rank + 1) * width].contiguous()

    def gather(self, local, tokens):
        if self.world == 1:
            return local[:tokens]
        output = local.new_empty((local.shape[0] * self.world, *local.shape[1:]))
        dist.all_gather_into_tensor(output, local.contiguous(), group=self.group)
        return output[:tokens]

    def reduce(self, partial):
        padded, width = self._pad(partial)
        if self.world == 1:
            return padded
        local = padded.new_empty((width, *padded.shape[1:]))
        dist.reduce_scatter_tensor(local, padded, group=self.group)
        return local
