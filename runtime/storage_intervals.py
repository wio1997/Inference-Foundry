"""Conservative tensor storage intervals for concurrent scratch alias audits.

A strided view can have holes. Overlapping hulls are treated as possible alias;
disjoint hulls prove the underlying elements are disjoint.
"""
from __future__ import annotations

import torch


def tensor_interval(name: str, tensor: torch.Tensor) -> dict[str, object]:
    shape = tuple(int(n) for n in tensor.shape)
    strides = tuple(int(n) for n in tensor.stride())
    offset = int(tensor.storage_offset())
    itemsize = int(tensor.element_size())
    base = int(tensor.untyped_storage().data_ptr())
    if tensor.numel() == 0:
        start = end = base + offset * itemsize
    else:
        low = offset + sum(min(0, (n - 1) * stride)
                           for n, stride in zip(shape, strides))
        high = offset + sum(max(0, (n - 1) * stride)
                            for n, stride in zip(shape, strides))
        start = base + low * itemsize
        end = base + (high + 1) * itemsize
    return dict(name=name, device=str(tensor.device), storage_base=base, storage_offset=offset,
                shape=shape, strides=strides, itemsize=itemsize,
                start=start, end=end, is_empty=tensor.numel() == 0)


def audit_disjoint(private: dict[str, torch.Tensor],
                   protected: dict[str, torch.Tensor]) -> dict[str, object]:
    writes = [tensor_interval(name, value) for name, value in private.items()]
    live = [tensor_interval(name, value) for name, value in protected.items()]
    conflicts = []
    for index, left in enumerate(writes):
        if left["is_empty"]:
            continue
        for right in writes[index + 1:] + live:
            if right["is_empty"] or left["device"] != right["device"]:
                continue
            if max(left["start"], right["start"]) < min(left["end"], right["end"]):
                conflicts.append([left["name"], right["name"]])
    return dict(pass_=not conflicts, private=writes, protected=live,
                possible_aliases=conflicts,
                meaning="Conservative hull check, including storage base, offset, shape and stride")
