# SPDX-License-Identifier: Apache-2.0
"""Token-axis ownership for native TP attention and sequence-parallel trunk.

Attention caches retain full request coordinates on every TP rank. Only the
mHC residual stream, normalization and FFN own disjoint token slices; gather
normalized attention inputs and reduce-scatter the partial output projection.
"""
import torch
import torch.distributed as dist
from deepinfer import envs


def ordered_sum(partial, group, *, scatter=False):
    """Sum TP contributions in rank order when batch invariance is requested.

    HCCL transports the contributions without reducing them. Each destination
    then adds the same ranks in FP32, so changing a row's batch position or
    destination rank cannot change its reduction order.
    """
    world = dist.get_world_size(group)
    if world == 1 or partial.shape[0] == 0:
        return partial
    if scatter:
        if partial.shape[0] % world:
            raise ValueError('ordered reduce-scatter requires padded token rows')
        received = torch.empty_like(partial)
        dist.all_to_all_single(received, partial.contiguous(), group=group)
        pieces = received.reshape(world, partial.shape[0] // world, *partial.shape[1:])
    else:
        received = partial.new_empty((world * partial.shape[0], *partial.shape[1:]))
        dist.all_gather_into_tensor(received, partial.contiguous(), group=group)
        pieces = received.reshape(world, *partial.shape)
    result = pieces[0].float()
    for rank in range(1, world):
        result.add_(pieces[rank])
    return result.to(partial.dtype)


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
        if envs.DEEPINFER_BATCH_INVARIANT:
            return ordered_sum(padded,self.group,scatter=True)
        local = padded.new_empty((width, *padded.shape[1:]))
        dist.reduce_scatter_tensor(local, padded, group=self.group)
        return local
