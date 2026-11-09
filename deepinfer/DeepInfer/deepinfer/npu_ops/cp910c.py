# SPDX-License-Identifier: Apache-2.0
"""Query-token CP with replicated KV, matching the pinned V4.1 DSACP layout.

Each query belongs to one TP rank and owns all query heads. Cache writes
retain the original global request layout. Exchange heads before the existing
TP output projection. No vLLM Python model or runtime is imported.
"""
from dataclasses import dataclass
import torch.distributed as dist


@dataclass(frozen=True)
class QuerySlice:
    start: int
    end: int
    width: int
    request_indices: tuple[int, ...]
    query_lengths: tuple[int, ...]
    seq_lengths: tuple[int, ...]

    @classmethod
    def build(cls, query_lengths, seq_lengths, world, rank):
        if world < 1 or not 0 <= rank < world:
            raise ValueError('invalid CP rank/world')
        if len(query_lengths) != len(seq_lengths) or any(q < 1 or s < q for q, s in zip(query_lengths, seq_lengths)):
            raise ValueError('invalid global request coordinates')
        total = sum(query_lengths)
        width = (total + world - 1) // world
        start, end = min(rank * width, total), min((rank + 1) * width, total)
        indices, queries, seqs = [], [], []
        offset = 0
        for i, (q, s) in enumerate(zip(query_lengths, seq_lengths)):
            left, right = max(offset, start), min(offset + q, end)
            if right > left:
                indices.append(i)
                queries.append(right - left)
                seqs.append(s - (offset + q - right))
            offset += q
        return cls(start, end, width, tuple(indices), tuple(queries), tuple(seqs))


def restore_tp_heads(output, group, *, total_tokens, width):
    """[local tokens, full heads] -> [global tokens, local heads].

    All ranks exchange the same padded token width, including empty query
    ranks. The receive order is rank-major, equal to global packed token order.
    """
    world = dist.get_world_size(group) if group is not None else 1
    if output.ndim != 3 or output.shape[0] > width or output.shape[1] % world:
        raise ValueError('invalid CP attention output layout')
    if world == 1:
        return output[:total_tokens]
    padded = output.new_zeros((width, *output.shape[1:]))
    padded[:output.shape[0]].copy_(output)
    heads, dim = output.shape[1:]
    local_heads = heads // world
    send = padded.view(width, world, local_heads, dim).permute(1, 0, 2, 3).contiguous().view(-1, local_heads, dim)
    recv = send.new_empty(send.shape)
    dist.all_to_all_single(recv, send, group=group)
    return recv[:total_tokens]
