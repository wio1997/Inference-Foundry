# SPDX-License-Identifier: Apache-2.0
"""Fixed 4x6 device Engram raw windows; isolated, not a serving entrypoint.

The caller owns lifetime, stream dependencies and correctness/EP consensus.
The kernel produces status per slot; consumers must not use invalid windows.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _full24_windows(tail, tokens, positions, context, used, status,
                    VOCAB:tl.constexpr, MAX_POS:tl.constexpr):
    slot=tl.program_id(0)
    q=tl.arange(0,8)
    shift=tl.arange(0,4)
    pos=tl.load(positions+slot).to(tl.int64)
    # Each source element has a fixed address. Avoid dynamic per-lane gathers
    # on the pinned Ascend Triton path, which failed in earlier probes.
    idx=q[:,None]-shift[None,:]
    result=tl.full((8,4),-1,tl.int32)
    invalid_tokens=tl.full((),False,tl.int1)
    invalid_tail=tl.full((),False,tl.int1)
    for j in tl.static_range(6):
        value=tl.load(tokens+slot*6+j)
        invalid_tokens |= (value<0)|(value>=VOCAB)
        result=tl.where(idx==j,value,result)
    for j in tl.static_range(3):
        value=tl.load(tail+slot*3+j)
        invalid_tail |= (value < -1)|(value>=VOCAB)
        result=tl.where(idx==j-3,value,result)
    error=((pos<0)|(pos+6>MAX_POS)).to(tl.int32)
    error+=2*invalid_tokens.to(tl.int32)+4*invalid_tail.to(tl.int32)
    tl.store(context+((slot*6+q[:,None])*4+shift[None,:]),result,q[:,None]<6)
    tl.store(used+slot,tl.where(error==0,pos+6,0).to(tl.int32))
    tl.store(status+slot,error)



def build_windows(tail, tokens, positions, context, used, status, *, vocab_size, max_positions):
    """Enqueue [24,4] newest-first windows and used positions on caller stream."""
    if type(vocab_size) is not int or not 0 < vocab_size < 2**31:
        raise ValueError('invalid vocabulary size')
    if type(max_positions) is not int or not 6 <= max_positions < 2**31:
        raise ValueError('invalid maximum position')
    device=tokens.device
    expected=((tail,(4,3),torch.int32),(tokens,(4,6),torch.int32),
              (positions,(4,),torch.int32),(context,(24,4),torch.int32),
              (used,(4,),torch.int32),(status,(4,),torch.int32))
    if device.type!='npu':raise ValueError('device must be Ascend NPU')
    for value,shape,dtype in expected:
        if value.device!=device or tuple(value.shape)!=shape or value.dtype!=dtype or not value.is_contiguous():
            raise ValueError('invalid device window tensor layout')
    spans=sorted((v.data_ptr(),v.data_ptr()+v.numel()*v.element_size()) for v,_,_ in expected)
    if any(end>start for (_,end),(start,_) in zip(spans,spans[1:])):
        raise ValueError('window inputs and outputs must be disjoint')
    _full24_windows[(4,)](tail,tokens,positions,context,used,status,
                           VOCAB=vocab_size,MAX_POS=max_positions,num_warps=4)
    return context,used,status
