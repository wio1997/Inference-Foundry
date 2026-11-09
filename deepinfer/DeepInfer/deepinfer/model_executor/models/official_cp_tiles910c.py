"""Preserve absolute positions when calling row-local 910C consumers."""

import torch

WIDTH = 32


def plan(query_batch):
    if query_batch is None:
        return ()
    segments = []
    offset = 0
    for width,end in zip(query_batch.query_lengths,query_batch.seq_lengths,strict=True):
        position = end-width
        while position < end:
            slot = position % WIDTH
            length = min(end-position,WIDTH-slot)
            segments.append((offset,length,slot))
            offset += length
            position += length
    return tuple(segments)


def apply(operation, segments, *values):
    """Only retain real output slices and separately cover native CP padding."""
    count = values[0].shape[0]
    if not count:
        return operation(*values)
    if any(v.shape[0]!=count for v in values):
        raise ValueError('local CP operands do not share the row dimension')
    valid = segments[-1][0]+segments[-1][1] if segments else 0
    if valid > count:
        raise ValueError('CP tile plan exceeds the local rows')
    padding = tuple((first,min(count-first,WIDTH),0) for first in range(valid,count,WIDTH))
    parts = []
    multiple = None
    for first,length,slot in (*segments,*padding):
        packed = [v.new_zeros((WIDTH,*v.shape[1:])) for v in values]
        for x,v in zip(packed,values,strict=True):
            x[slot:slot+length].copy_(v[first:first+length])
        result = operation(*packed)
        multiple = isinstance(result,tuple)
        outputs = result if multiple else (result,)
        parts.append(tuple(o[slot:slot+length] for o in outputs))
    joined = tuple(torch.cat(v,dim=0) for v in zip(*parts,strict=True))
    return joined if multiple else joined[0]
