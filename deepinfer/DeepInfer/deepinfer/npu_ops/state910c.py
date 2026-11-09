# SPDX-License-Identifier: Apache-2.0
"""Fixed four-slot raw-history commit primitive, independent of the service.

This is only the state transition after a target verification. It is not a
sampler, C2 commit, graph scheduler, streaming server, or full ZOS.
"""
from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@triton.jit
def _commit_state(old_pos, old_remaining, old_golden, old_finished, old_tail,
                  target_input, commit, last_emitted,
                  new_pos, new_remaining, new_golden, new_finished, new_tail,
                  status, H: tl.constexpr, VOCAB: tl.constexpr, MAX_POS: tl.constexpr):
    slot = tl.arange(0, 4)
    col = tl.arange(0, 8)
    pos = tl.load(old_pos + slot)
    remaining = tl.load(old_remaining + slot)
    golden = tl.load(old_golden + slot)
    finished = tl.load(old_finished + slot)
    count = tl.load(commit + slot)
    last = tl.load(last_emitted + slot)
    tail = tl.load(old_tail + slot[:, None] * H + col[None, :], col[None, :] < H, other=-1)
    x = tl.load(target_input + slot[:, None] * 6 + col[None, :], col[None, :] < 6, other=-1)

    bad_count = (count < 0) | (count > 6) | (count > remaining) | (finished & (count != 0))
    bad_position = (pos < 0) | (pos > MAX_POS) | (remaining < 0) | ((pos.to(tl.int64) + count.to(tl.int64)) > MAX_POS)
    bad_tail = tl.max(tl.where(col[None, :] < H,
                              ((tail < -1) | (tail >= VOCAB)).to(tl.int32), 0), axis=1) != 0
    bad_x = tl.max(tl.where(col[None, :] < count[:, None],
                           ((x < 0) | (x >= VOCAB)).to(tl.int32), 0), axis=1) != 0
    bad_tokens = bad_tail | bad_x | ((count > 0) & ((golden < 0) | (golden >= VOCAB) | (last < 0) | (last >= VOCAB)))
    first_x = tl.max(tl.where(col[None, :] == 0, x, -2147483648), axis=1)
    bad_first = (count > 0) & (first_x != golden)
    error = (bad_count.to(tl.int32) + 2 * bad_position.to(tl.int32) +
             4 * bad_tokens.to(tl.int32) + 8 * bad_first.to(tl.int32))
    safe_count = tl.where(error == 0, count, 0)
    updated_remaining = remaining - safe_count
    tl.store(new_pos + slot, pos + safe_count)
    tl.store(new_remaining + slot, updated_remaining)
    tl.store(new_golden + slot, tl.where(safe_count > 0, last, golden))
    tl.store(new_finished + slot, finished | ((safe_count > 0) & (updated_remaining == 0)))
    tl.store(status + slot, error)

    # The committed raw history is the last H entries of old_tail + X[:c].
    # Use only fixed-column row reductions rather than dynamic global gathers;
    # the pinned Ascend Triton backend previously failed a dynamic row load.
    source_index = safe_count[:, None] + col[None, :]
    result = tl.full((4, 8), -1, tl.int32)
    for j in tl.static_range(H):
        token = tl.max(tl.where(col[None, :] == j, tail, -2147483648), axis=1)
        result = tl.where(source_index == j, token[:, None], result)
    for j in tl.static_range(6):
        token = tl.max(tl.where(col[None, :] == j, x, -2147483648), axis=1)
        result = tl.where(source_index == H + j, token[:, None], result)
    tl.store(new_tail + slot[:, None] * H + col[None, :], result, col[None, :] < H)


@dataclass
class StateBuffers:
    pos: torch.Tensor
    remaining: torch.Tensor
    golden: torch.Tensor
    finished: torch.Tensor
    tail: torch.Tensor
    status: torch.Tensor

    @classmethod
    def allocate(cls, device, history_width):
        def empty(shape, dtype):
            return torch.empty(shape, dtype=dtype, device=device)
        return cls(empty((4,), torch.int32), empty((4,), torch.int32),
                   empty((4,), torch.int32), empty((4,), torch.bool),
                   empty((4, history_width), torch.int32), empty((4,), torch.int32))


def commit_state(pos, remaining, golden, finished, tail, target_input,
                 commit, last_emitted, out, *, vocab_size, max_positions):
    """Enqueue one state transition; all inputs and outputs are disjoint NPU tensors.

    ``tail`` stores raw IDs oldest-to-newest, using -1 for missing old tokens.
    ``target_input`` contains the old golden followed by five draft tokens;
    only its first ``commit`` entries are appended. ``last_emitted`` is the
    already-clipped target token at commit-1, or -1 when commit is zero.
    Invalid slots are copied unchanged and receive a nonzero status bitmask.
    The function does not synchronize or inspect device values from Python.
    """
    if not isinstance(out, StateBuffers):
        raise TypeError('out must be preallocated StateBuffers')
    if (type(vocab_size) is not int or not 0 < vocab_size < 2**31 or
            type(max_positions) is not int or not 0 < max_positions < 2**31):
        raise ValueError('invalid vocab size or context limit')
    h = tail.shape[1] if tail.ndim == 2 else 0
    if not 1 <= h <= 7:
        raise ValueError('expected history width 1..7')
    arrays = ((pos, (4,), torch.int32), (remaining, (4,), torch.int32),
              (golden, (4,), torch.int32), (finished, (4,), torch.bool),
              (tail, (4, h), torch.int32), (target_input, (4, 6), torch.int32),
              (commit, (4,), torch.int32), (last_emitted, (4,), torch.int32),
              (out.pos, (4,), torch.int32), (out.remaining, (4,), torch.int32),
              (out.golden, (4,), torch.int32), (out.finished, (4,), torch.bool),
              (out.tail, (4, h), torch.int32), (out.status, (4,), torch.int32))
    device = pos.device
    if device.type != 'npu':
        raise ValueError('all state must be on an Ascend NPU')
    for value, shape, dtype in arrays:
        if value.device != device or tuple(value.shape) != shape or value.dtype != dtype or not value.is_contiguous():
            raise ValueError(f'expected contiguous {shape} {dtype} on {device}')
    ranges = sorted((v.data_ptr(), v.data_ptr() + v.numel() * v.element_size()) for v, _, _ in arrays)
    if any(end > next_start for (_, end), (next_start, _) in zip(ranges, ranges[1:])):
        raise ValueError('state inputs and outputs must not overlap')
    _commit_state[(1,)](pos, remaining, golden, finished, tail, target_input, commit,
                        last_emitted, out.pos, out.remaining, out.golden, out.finished,
                        out.tail, out.status, H=h, VOCAB=vocab_size,
                        MAX_POS=max_positions, num_warps=4)
    return out
