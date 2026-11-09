# SPDX-License-Identifier: Apache-2.0
"""Tensor layout bridge for the existing four-slot 910C small target graphs.

The caller owns the graph lease and publishes state only after status and
native host acceptance agree. Inputs and results stay on the caller's stream.
"""
from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class SmallTargetLayout:
    lengths: tuple[int, int, int, int]
    capacity: int | None = None
    offsets: tuple[int, ...] = field(init=False)
    physical_rows: int = field(init=False)

    def __post_init__(self):
        if (type(self.lengths) is not tuple or len(self.lengths) != 4
                or any(type(q) is not int or not 1 <= q <= 6 for q in self.lengths)):
            raise ValueError('expected four positive verification lengths')
        offsets = [0]
        for length in self.lengths:
            offsets.append(offsets[-1] + length)
        real_rows = offsets[-1]
        if not 4 <= real_rows <= 7:
            raise ValueError('layout is outside the small target graph pool')
        object.__setattr__(self, 'offsets', tuple(offsets))
        capacity = (4 if real_rows == 4 else 8) if self.capacity is None else self.capacity
        if capacity not in (4, 8) or (capacity == 4 and real_rows != 4):
            raise ValueError('physical graph cannot contain this query budget')
        object.__setattr__(self, 'physical_rows', capacity)

    @property
    def scratch_rows(self):
        return self.physical_rows - self.offsets[-1]

    def pack_predictions(self, predictions):
        """Map global top-1 IDs to the fixed sampler packet; omit scratch."""
        if (predictions.shape != (self.physical_rows,)
                or predictions.dtype != torch.int64 or not predictions.is_contiguous()):
            raise ValueError('invalid small target prediction vector')
        packed = predictions.new_zeros((4, 6))
        if self.physical_rows == 4:
            packed[:, 0].copy_(predictions)
        else:
            for slot, length in enumerate(self.lengths):
                packed[slot, :length].copy_(predictions[self.offsets[slot]:self.offsets[slot + 1]])
        return packed.flatten()

    def c2_row(self, slot, commit):
        """Return a real row for a nonzero commit, or None for saved C2."""
        if (type(slot) is not int or not 0 <= slot < 4 or type(commit) is not int
                or not 0 <= commit <= self.lengths[slot]):
            raise ValueError('commit exceeds the target span for this physical slot')
        return self.offsets[slot] + commit - 1 if commit else None


def _require(tensor, shape, dtype, device):
    if (tensor.shape != shape or tensor.dtype != dtype or tensor.device != device
            or not tensor.is_contiguous()):
        raise ValueError('invalid small4 tensor layout')


def stage_small4_windows(tail, golden, bases, *, vocab_size, max_pos):
    """Build four newest-first raw contexts and physical used=base+1."""
    if type(vocab_size) is not int or not 0 < vocab_size < 2**31:
        raise ValueError('invalid vocabulary size')
    if type(max_pos) is not int or not 1 <= max_pos < 2**31:
        raise ValueError('invalid position limit')
    device = bases.device
    _require(tail, (4, 3), torch.int32, device)
    _require(golden, (4,), torch.int32, device)
    _require(bases, (4,), torch.int32, device)
    status = ((bases < 0) | (bases >= max_pos)).to(torch.int32)
    status += 2 * ((golden < 0) | (golden >= vocab_size)).to(torch.int32)
    status += 4 * ((tail < -1) | (tail >= vocab_size)).any(dim=1).to(torch.int32)
    windows = torch.cat((golden[:, None], tail.flip((1,))), dim=1)
    used = torch.where(status == 0, bases + 1, 0)
    return windows, used, status


def stage_small7_inputs(layout, tokens, windows, bases, *, scratch_base, noise, vocab_size, max_pos):
    """Pack the existing 8-row graph inputs and newest-first Engram windows."""
    if layout.physical_rows != 8 or not 0 <= noise < vocab_size:
        raise ValueError('invalid small7 layout or noise token')
    if not 0 <= scratch_base <= max_pos-layout.scratch_rows:
        raise ValueError('small7 scratch exceeds context capacity')
    device = bases.device
    _require(bases, (4,), torch.int32, device)
    _require(tokens, (4, 6), torch.int32, device)
    _require(windows, (24, 4), torch.int32, device)
    lengths = (*layout.lengths, layout.scratch_rows)
    rows = [slot*6+j for slot, length in enumerate(layout.lengths) for j in range(length)]
    slots = [slot for slot, length in enumerate(lengths) for _ in range(length)]
    offsets = [j for length in lengths for j in range(length)]
    # Only the frozen shape is host supplied. Positions and tokens come from
    # the device's current/next view, with scratch private to this graph.
    index = torch.tensor(rows, dtype=torch.int64, device=device)
    cu = torch.tensor((*layout.offsets, 8), dtype=torch.int32, device=device)
    requests = torch.tensor(slots, dtype=torch.int32, device=device)
    relative = torch.tensor(offsets, dtype=torch.int32, device=device)
    starts = torch.cat((bases, bases.new_full((1,), scratch_base)))
    ids = torch.cat((tokens.flatten().index_select(0, index),
                     bases.new_full((layout.scratch_rows,), noise)))
    positions = starts.index_select(0, requests.long()) + relative
    used = starts + (cu[1:]-cu[:-1])
    scratch = [[noise]*(j+1) + [-1]*(3-j) for j in range(layout.scratch_rows)]
    compact_windows = torch.cat((windows.index_select(0, index), bases.new_tensor(scratch)))
    return torch.cat((ids, used, cu, requests, positions)), compact_windows


def stage_small7_device_inputs(lengths, tokens, windows, bases, *, scratch_base, noise, vocab_size, max_pos):
    """Stage corrected device Q without reading its allocation back to the host.

    Shapes remain fixed at eight rows. The fifth request owns positive scratch
    padding; it never participates in the four-slot verifier packet. Invalid
    discrete lengths stage a safe shape and report status for the sample phase.
    """
    device = bases.device
    _require(lengths, (4,), torch.int32, device)
    _require(tokens, (4, 6), torch.int32, device)
    _require(windows, (24, 4), torch.int32, device)
    _require(bases, (4,), torch.int32, device)
    if not 0 <= scratch_base <= max_pos - 4 or not 0 <= noise < vocab_size:
        raise ValueError('invalid private scratch context')
    bad = ((lengths < 1) | (lengths > 6)).any() | (lengths.sum() > 7)
    safe = torch.where(bad, torch.ones_like(lengths), lengths)
    all_lengths = torch.cat((safe, (8 - safe.sum()).to(torch.int32).reshape(1)))
    cu = torch.cat((safe.new_zeros(1), all_lengths.cumsum(0).to(torch.int32)))
    row = torch.arange(8, dtype=torch.int32, device=device)
    requests = (row[:, None] >= cu[1:][None, :]).sum(1).to(torch.int32)
    relative = row - cu.index_select(0, requests.long())
    real = requests < 4
    fixed_index = torch.where(real, requests * 6 + relative, 0).long()
    starts = torch.cat((bases, bases.new_full((1,), scratch_base)))
    ids = torch.where(real, tokens.flatten().index_select(0, fixed_index), noise)
    positions = starts.index_select(0, requests.long()) + relative
    used = starts + all_lengths
    scratch = bases.new_tensor([[noise] * (j + 1) + [-1] * (3 - j) for j in range(4)])
    scratch_windows = scratch.index_select(0, relative.clamp(0, 3).long())
    compact_windows = torch.where(real[:, None], windows.index_select(0, fixed_index), scratch_windows)
    return torch.cat((ids, used, cu, requests, positions)), compact_windows, bad.to(torch.int32)


def select_small7_c2(before, rows, bases, counts, cu, *, max_pos):
    """Select real pending rows by compact offset, excluding scratch."""
    device = bases.device
    _require(bases, (4,), torch.int32, device)
    _require(counts, (4,), torch.int32, device)
    _require(cu, (6,), torch.int32, device)
    if set(before) != {2, 8, 14} or set(rows) != set(before):
        raise ValueError('unexpected compact C2 sources')
    lengths = cu[1:5]-cu[:4]
    end = bases.to(torch.int64) + counts
    bad_count = (counts < 0) | (counts > lengths)
    bad_pos = (bases < 0) | (bases > max_pos) | (end > max_pos)
    status = bad_count.to(torch.int32) + 2*bad_pos.to(torch.int32)
    rejected = status != 0
    odd = end.remainder(2) != 0
    valid = torch.where(rejected, bases.remainder(2) != 0, odd)
    use_row = (~rejected & odd & (counts > 0))[:, None]
    use_before = (rejected | (odd & (counts == 0)))[:, None]
    index = (cu[:4] + torch.where(use_row[:, 0], counts-1, 0)).long()
    result = {}
    for source in (2, 8, 14):
        if len(before[source]) != 2 or len(rows[source]) != 2:
            raise ValueError('C2 requires value and gate')
        payload = []
        for saved, current in zip(before[source], rows[source]):
            if saved.ndim!=2 or saved.shape[1]<=0:
                raise ValueError('expected nonempty compact C2 feature dimension')
            width = saved.shape[1]
            _require(saved, (5, width), torch.float32, device)
            _require(current, (8, width), torch.float32, device)
            payload.append(torch.where(use_row, current.index_select(0, index),
                                       torch.where(use_before, saved[:4], 0.0)))
        result[source] = (*payload, valid, status)
    return result


def select_small4_c2(before, rows, bases, counts, *, max_pos):
    """Return independent C2 value/gate buffers with the existing packet ABI."""
    if type(max_pos) is not int or not 1 <= max_pos < 2**31:
        raise ValueError('invalid position limit')
    if set(before) != {2, 8, 14} or set(rows) != set(before):
        raise ValueError('unexpected C2 source layout')
    device = bases.device
    _require(bases, (4,), torch.int32, device)
    _require(counts, (4,), torch.int32, device)
    base64, count64 = bases.to(torch.int64), counts.to(torch.int64)
    end = base64 + count64
    bad_count = (count64 < 0) | (count64 > 1)
    bad_pos = (base64 < 0) | (base64 > max_pos) | (end > max_pos)
    status = bad_count.to(torch.int32) + 2 * bad_pos.to(torch.int32)
    rejected = status != 0
    odd = end.remainder(2) != 0
    valid = torch.where(rejected, bases.remainder(2) != 0, odd)
    use_row = (~rejected & odd & (counts == 1))[:, None]
    use_before = (rejected | (odd & (counts == 0)))[:, None]

    result = {}
    for source in (2, 8, 14):
        if len(before[source]) != 2 or len(rows[source]) != 2:
            raise ValueError('C2 requires value and gate')
        payload = []
        for saved, current in zip(before[source], rows[source]):
            if saved.ndim != 2 or saved.shape[1] <= 0:
                raise ValueError('expected nonempty C2 feature dimension')
            width = saved.shape[1]
            _require(saved, (4, width), torch.float32, device)
            _require(current, (4, 1, width), torch.float32, device)
            payload.append(torch.where(use_row, current[:, 0], torch.where(use_before, saved, 0.0)))
        result[source] = (*payload, valid, status)
    return result
