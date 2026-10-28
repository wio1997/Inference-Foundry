# SPDX-License-Identifier: Apache-2.0
"""Exact greedy selection from ordered TP vocabulary shards on910C."""
import torch
import torch.distributed as dist


def global_argmax(logits,group,vocabulary):
    """Gather only each shard's top score/ID, keeping lowest-ID tie breaks.

    FP32 represents these vocabulary IDs exactly. Every rank in one TP group
    receives the same result; no complete logits gather or target replay.
    """
    if logits.shape[-1]==vocabulary:return logits.argmax(-1).long()
    size=dist.get_world_size(group)
    if vocabulary%size or not 0<vocabulary<2**24 or logits.shape[-1]!=vocabulary//size:
        raise ValueError('target logits must contain one complete ordered vocabulary shard')
    shape=logits.shape[:-1]
    values,ids=logits.reshape(-1,logits.shape[-1]).max(-1)
    local=torch.stack((values.float(),ids.float()+dist.get_rank(group)*(vocabulary//size)),-1)
    gathered=local.new_empty((size*local.shape[0],2))
    dist.all_gather_into_tensor(gathered,local.contiguous(),group=group)
    gathered=gathered.reshape(size,local.shape[0],2)
    winner=gathered[:,:,0].argmax(0)
    rows=torch.arange(local.shape[0],device=logits.device)
    return gathered[winner,rows,1].long().reshape(shape)
