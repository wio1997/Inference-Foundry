"""Independent C2 pending transition, four rows and six proposal tokens.

Device source and destination buffers must be disjoint. Run on the same stream
as producer and consumer; see README for the exact contract.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _payload(before_value, before_gate, rows_value, rows_gate, bases, counts,
             out_value, out_gate, DV: tl.constexpr, DG: tl.constexpr,
             MAX_POS: tl.constexpr, BLOCK: tl.constexpr):
    slot = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    base = tl.load(bases + slot).to(tl.int64)
    count = tl.load(counts + slot).to(tl.int64)
    end = base + count
    invalid = (count < 0) | (count > 6) | (base < 0) | (base > MAX_POS) | (end > MAX_POS)
    odd = (end % 2) != 0
    old_v = tl.load(before_value + slot * DV + offsets, offsets < DV, other=0)
    old_g = tl.load(before_gate + slot * DG + offsets, offsets < DG, other=0)
    chosen_v = old_v
    chosen_g = old_g
    # Constant row addresses avoid the dynamic gather issue observed on Ascend.
    for row in tl.static_range(0, 6):
        v = tl.load(rows_value + (slot * 6 + row) * DV + offsets, offsets < DV, other=0)
        g = tl.load(rows_gate + (slot * 6 + row) * DG + offsets, offsets < DG, other=0)
        chosen_v = tl.where(count == row + 1, v, chosen_v)
        chosen_g = tl.where(count == row + 1, g, chosen_g)
    result_v = tl.where(invalid, old_v, tl.where(odd, chosen_v, 0.0))
    result_g = tl.where(invalid, old_g, tl.where(odd, chosen_g, 0.0))
    tl.store(out_value + slot * DV + offsets, result_v, offsets < DV)
    tl.store(out_gate + slot * DG + offsets, result_g, offsets < DG)


@triton.jit
def _metadata(before_valid, bases, counts, out_valid, status, MAX_POS: tl.constexpr):
    # One program owns all four packed metadata entries, avoiding cross-core writes.
    slot = tl.arange(0, 4)
    base = tl.load(bases + slot).to(tl.int64)
    count = tl.load(counts + slot).to(tl.int64)
    end = base + count
    bad_count = (count < 0) | (count > 6)
    bad_pos = (base < 0) | (base > MAX_POS) | (end > MAX_POS)
    error = bad_count.to(tl.int32) + 2 * bad_pos.to(tl.int32)
    before = tl.load(before_valid + slot)
    valid = tl.where(error != 0, before, (end % 2) != 0)
    tl.store(out_valid + slot, valid)
    tl.store(status + slot, error)


def allocate(before_value, before_gate):
    """Allocate separate output tensors outside any graph capture."""
    return (torch.empty_like(before_value), torch.empty_like(before_gate),
            torch.empty((4,), dtype=torch.bool, device=before_value.device),
            torch.empty((4,), dtype=torch.int32, device=before_value.device))


def launch(before_value, before_gate, before_valid, rows_value, rows_gate,
           bases, counts, outputs, max_pos=2**31-1):
    """Enqueue updates on the current NPU stream; returns the supplied outputs."""
    ov, og, valid, status = outputs
    assert 0 <= max_pos <= 2**31-1
    assert before_value.ndim == before_gate.ndim == 2
    dv, dg = before_value.shape[1], before_gate.shape[1]
    assert dv > 0 and dg > 0 and dv % 8 == 0 and dg % 8 == 0
    tensors = (before_value, before_gate, before_valid, rows_value, rows_gate,
               bases, counts, ov, og, valid, status)
    device = before_value.device
    assert device.type == 'npu' and all(t.device == device and t.is_contiguous() for t in tensors)
    shapes = ((4,dv),(4,dg),(4,),(4,6,dv),(4,6,dg),(4,),(4,),
              (4,dv),(4,dg),(4,),(4,))
    dtypes = (torch.float32,torch.float32,torch.bool,torch.float32,torch.float32,
              torch.int32,torch.int32,torch.float32,torch.float32,torch.bool,torch.int32)
    assert all(tuple(t.shape) == shape and t.dtype == dtype
               for t,shape,dtype in zip(tensors,shapes,dtypes))
    intervals = sorted((t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()) for t in tensors)
    assert all(end <= start for (_,end),(start,_) in zip(intervals,intervals[1:])), 'inputs and outputs must be disjoint'
    assert all(t.data_ptr() % 32 == 0 for t in (before_value,before_gate,rows_value,rows_gate,ov,og))
    _payload[(4, triton.cdiv(max(dv,dg),128))](before_value,before_gate,rows_value,rows_gate,
        bases,counts,ov,og,dv,dg,max_pos,128,num_warps=4)
    _metadata[(1,)](before_valid,bases,counts,valid,status,max_pos,num_warps=4)
    return outputs
