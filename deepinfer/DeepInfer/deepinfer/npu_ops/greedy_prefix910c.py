# SPDX-License-Identifier: Apache-2.0
"""Fixed four-slot greedy prefix verification for the Ascend Triton backend.

This is a standalone primitive.  It neither schedules a model nor implements
ZOS.  Each slot has six physical target rows and at most five draft tokens.
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@triton.jit
def _greedy_prefix(draft, target, lengths, remaining, finished,
                   accepted_out, commit_out, last_out, emitted_out,
                   accept_mask_out, final_mask_out, status_out):
    slot = tl.arange(0, 4)
    col = tl.arange(0, 8)
    length = tl.load(lengths + slot)
    left = tl.load(remaining + slot)
    done = tl.load(finished + slot)
    valid = (length >= 1) & (length <= 6) & (left >= 0)
    active = valid & (~done) & (left > 0)

    pred = tl.load(target + slot[:, None] * 6 + col[None, :],
                   col[None, :] < 6, other=0)
    proposed = tl.load(draft + slot[:, None] * 5 + col[None, :],
                       col[None, :] < 5, other=0)
    verification = col[None, :] < (length[:, None] - 1)
    mismatch = verification & (proposed != pred)
    # The final fresh row at length-1 is the fallback when all drafts pass.
    first = tl.min(tl.where(mismatch, col[None, :], length[:, None] - 1), axis=1)
    accepted = tl.where(active, first, 0)
    commit = tl.where(active, tl.minimum(first + 1, left), 0)
    # Select from the same 2-D rows already loaded for verification.  This
    # avoids a second per-slot dynamically indexed global load, which produced
    # an incorrect value on the pinned Ascend Triton backend.  Token IDs are
    # nonnegative, so -1 is a safe masked reduction identity.
    selected = (col[None, :] == commit[:, None] - 1) & (commit[:, None] > 0)
    last = tl.max(tl.where(selected, pred, -1), axis=1)

    tl.store(accepted_out + slot, accepted)
    tl.store(commit_out + slot, commit)
    tl.store(last_out + slot, last)
    tl.store(status_out + slot, tl.where(valid, 0, 1))
    physical = col[None, :] < 6
    offsets = slot[:, None] * 6 + col[None, :]
    tl.store(emitted_out + offsets,
             tl.where(active[:, None] & (col[None, :] < commit[:, None]), pred, -1),
             physical)
    tl.store(accept_mask_out + offsets,
             active[:, None] & (col[None, :] < accepted[:, None]), physical)
    tl.store(final_mask_out + offsets,
             active[:, None] & (col[None, :] == accepted[:, None]), physical)


@dataclass
class GreedyPrefixBuffers:
    accepted: torch.Tensor
    commit: torch.Tensor
    last: torch.Tensor
    emitted: torch.Tensor
    accept_mask: torch.Tensor
    final_mask: torch.Tensor
    status: torch.Tensor

    @classmethod
    def allocate(cls, device):
        def make(shape, dtype):
            return torch.empty(shape, dtype=dtype, device=device)
        return cls(make((4,), torch.int32), make((4,), torch.int32),
                   make((4,), torch.int32), make((4, 6), torch.int32),
                   make((4, 6), torch.bool), make((4, 6), torch.bool),
                   make((4,), torch.int32))


def verify_greedy_prefix(draft, target, lengths, remaining, finished, out):
    """Enqueue verification on the current NPU stream with persistent buffers.

    draft/target: contiguous int32 [4,5]/[4,6] *global* vocabulary IDs;
    lengths: int32 [4], valid range 1..6, including the final fresh row;
    remaining: nonnegative int32 [4]; finished: bool [4].
    All inputs and buffers must live on the same NPU, with no aliasing.

    The accept/final masks match upstream prefix verification for active
    slots; completion and output clipping are separate serving semantics.
    ``status`` is 1 for invalid length/remaining and 0 otherwise.  Callers
    must inspect it before consuming untrusted inputs.  The graph-safe hot
    path does not synchronize or read tensor values on the CPU.
    """
    if not isinstance(out, GreedyPrefixBuffers):
        raise TypeError('out must be persistent GreedyPrefixBuffers')
    tensors = ((draft, (4, 5), torch.int32), (target, (4, 6), torch.int32),
               (lengths, (4,), torch.int32), (remaining, (4,), torch.int32),
               (finished, (4,), torch.bool),
               (out.accepted, (4,), torch.int32), (out.commit, (4,), torch.int32),
               (out.last, (4,), torch.int32), (out.emitted, (4, 6), torch.int32),
               (out.accept_mask, (4, 6), torch.bool), (out.final_mask, (4, 6), torch.bool),
               (out.status, (4,), torch.int32))
    device = target.device
    if device.type != 'npu':
        raise ValueError('all tensors must be on an Ascend NPU')
    for value, shape, dtype in tensors:
        if value.device != device or tuple(value.shape) != shape or value.dtype != dtype or not value.is_contiguous():
            raise ValueError(f'expected contiguous {shape} {dtype} on {device}')
    ranges = sorted((value.data_ptr(), value.data_ptr() + value.numel() * value.element_size())
                    for value, _, _ in tensors)
    if any(end > next_start for (_, end), (next_start, _) in zip(ranges, ranges[1:])):
        raise ValueError('input and output tensors must not alias')
    _greedy_prefix[(1,)](draft, target, lengths, remaining, finished,
                          out.accepted, out.commit, out.last, out.emitted,
                          out.accept_mask, out.final_mask, out.status, num_warps=4)
    return out
