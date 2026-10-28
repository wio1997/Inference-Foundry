# SPDX-License-Identifier: Apache-2.0
"""CANN interleaved RoPE with exact FP32 angle tables on the pinned A3 ABI.

Reuse only the public torch_npu operator. Tables follow the same NPU FP32
position/frequency multiply and trig operations as the existing reference.
Share angle gathers within one target/draft forward, including graph capture.
"""
import os

import torch

_TABLES={}
_ACTIVE={}


def begin_forward():
    _ACTIVE.clear()


def register(frequencies, cpu_frequencies):
    if os.environ.get('DEEPINFER_910C_FUSED_ROPE')!='1':
        return frequencies
    limit=int(os.environ.get('DEEPINFER_910C_ROPE_MAX_TOKENS','34816'))
    key=(str(frequencies.device),limit,tuple(cpu_frequencies.tolist()))
    table=_TABLES.get(key)
    if table is None:
        positions=torch.arange(limit,dtype=torch.float32,device=frequencies.device)
        angles=positions[:,None]*frequencies[None]
        cos=angles.cos().repeat_interleave(2,-1).contiguous()
        sin=angles.sin().repeat_interleave(2,-1).contiguous()
        table=(cos,sin,-sin)
        _TABLES[key]=table
    frequencies._native_rope_table=table
    return frequencies


def apply(x,positions,frequencies,*,inverse=False):
    import torch_npu
    tables=frequencies._native_rope_table
    key=(id(tables),positions.data_ptr(),positions.numel(),tuple(positions.stride()))
    cached=_ACTIVE.get(key)
    if cached is None:
        ids=positions.long()
        cached=(positions,tuple(table.index_select(0,ids)[:,None,None,:]
                                for table in tables))
        _ACTIVE[key]=cached
    cos,sin,negative=cached[1]
    dim=frequencies.numel()*2
    tail=x[...,-dim:]
    shape=tail.shape
    rotated=torch_npu.npu_rotary_mul(tail.reshape(x.shape[0],1,-1,dim),cos,
                                    negative if inverse else sin,
                                    rotary_mode='interleave').reshape(shape)
    return torch.cat((x[...,:-dim],rotated),-1)
