"""Eager sparse MLA for 910C correctness integration.

Public sparse-attention semantics follow FlashMLA tests/ref.py. This module
uses actual NPU tensor operations, never 950 device binaries or native FP8.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from deepinfer.npu_ops.compat910c import decode_e2m1_bytes, decode_e4m3_bytes


@dataclass
class ReferenceMLASchedMeta:
    """Per-attention state for eager MLA; no compiled tile schedule is needed."""

    have_initialized: bool = False
    config: object | None = None
    tile_scheduler_metadata: torch.Tensor | None = None
    num_splits: torch.Tensor | None = None


def _mask(indices, capacity, lengths, *, decode):
    if indices.dtype not in (torch.int32, torch.int64) or indices.ndim != 2:
        raise ValueError("indices must be a 2D int32/int64 tensor")
    active = torch.ones_like(indices, dtype=torch.bool)
    if lengths is not None:
        if lengths.shape != indices.shape[:1] or lengths.dtype not in (torch.int32, torch.int64):
            raise ValueError("lengths must be a per-query integer vector")
        active = torch.arange(indices.shape[1], device=indices.device)[None, :] < lengths[:, None]
    if decode and bool((((indices < -1) | (indices >= capacity)) & active).any()):
        raise ValueError("active decode indices must be -1 or a valid physical cache slot")
    return active & (indices >= 0) & (indices < capacity)


def _gather_dense(kv, indices, valid):
    if kv.shape[0] == 0:
        return torch.zeros((*indices.shape, kv.shape[-1]), device=kv.device, dtype=torch.float32)
    selected = kv.index_select(0, indices.clamp(0, kv.shape[0] - 1).long().flatten()).reshape(*indices.shape, kv.shape[-1]).float()
    return torch.where(valid[..., None], selected, 0.)


def _cache_rows(cache, indices, valid, dim):
    if cache.ndim != 4 or cache.shape[-2] != 1:
        raise ValueError("expected cache [blocks,block_size,1,width]")
    flat = cache.reshape(-1, cache.shape[-1])
    if cache.dtype == torch.bfloat16:
        if flat.shape[-1] != dim: raise ValueError("BF16 cache width mismatch")
        selected = _gather_dense(flat, indices, valid)
    elif cache.dtype == torch.uint8 and dim == 512:
        raw = _gather_dense(flat, indices, valid).to(torch.uint8)
        if flat.shape[-1] == 528:
            values = decode_e4m3_bytes(raw[..., :512])
            sfbytes = raw[..., 512:]
            scales = (sfbytes.int() << 23).contiguous().view(torch.float32)
            scales = torch.where(sfbytes == 0, 2.**-127, scales)
            scales = torch.where(sfbytes == 255, float('nan'), scales)
            selected = values * scales.repeat_interleave(32, -1)
        elif flat.shape[-1] == 288:
            values = decode_e2m1_bytes(raw[..., :256])
            scales = decode_e4m3_bytes(raw[..., 256:])
            selected = values * scales.repeat_interleave(16, -1)
        else:
            raise NotImplementedError("910C sparse decode supports V41 528-byte FP8 and 288-byte FP4 caches")
    else:
        raise NotImplementedError("unsupported sparse cache dtype or logical dimension")
    # FlashMLA's decode reference explicitly cleans NaN cache payloads.
    return torch.where(torch.isnan(selected) | ~valid[..., None], 0., selected)


def _attend(query, gathered, valid, scale, sink, dim_v):
    q,h,d = query.shape
    if not 0 < dim_v <= d: raise ValueError("invalid value prefix dimension")
    if gathered.shape[1] == 0:
        return (query.new_zeros((q,h,dim_v)), torch.full((q,h),float('inf'),device=query.device))
    logits = torch.bmm(query.float(), gathered.transpose(1,2)) * scale
    logits = logits.masked_fill(~valid[:,None,:], float('-inf'))
    lse = torch.logsumexp(logits, -1)
    denominator = lse
    if sink is not None:
        if sink.shape != (h,) or sink.dtype != torch.float32 or sink.device != query.device:
            raise ValueError("sink must be FP32 [heads] on the query device")
        denominator = torch.logsumexp(torch.stack((lse, sink[None,:].expand_as(lse))), 0)
    denominator = torch.where(denominator == float('-inf'), float('inf'), denominator)
    probabilities = torch.exp(logits - denominator[...,None])
    output = torch.bmm(probabilities, gathered[...,:dim_v]).to(torch.bfloat16)
    return output, torch.where(lse == float('-inf'), float('inf'), lse)


def _validate(query, indices):
    if query.ndim != 3 or query.dtype != torch.bfloat16 or indices.shape[0] != query.shape[0] or query.device != indices.device:
        raise ValueError("expected BF16 [queries,heads,dim] with matching index rows/device")


def prefill(query, kv, indices, softmax_scale, attn_sink=None, index_length=None, fused_norm_rope_cast_kwargs=None):
    if fused_norm_rope_cast_kwargs is not None: raise NotImplementedError("910C fused quantized MLA is not implemented")
    _validate(query,indices)
    if kv.ndim!=3 or kv.shape[1]!=1 or kv.shape[-1]!=query.shape[-1] or kv.device!=query.device or kv.dtype!=query.dtype:
        raise ValueError("expected matching BF16 KV [keys,1,dim]")
    outputs=[];lses=[]
    # Bound gathered KV memory independently of the sequence length.
    for start in range(0,query.shape[0],16):
        idx=indices[start:start+16];lengths=None if index_length is None else index_length[start:start+16]
        valid=_mask(idx,kv.shape[0],lengths,decode=False)
        gathered=_gather_dense(kv[:,0],idx,valid)
        out,lse=_attend(query[start:start+16],gathered,valid,softmax_scale,attn_sink,query.shape[-1]);outputs.append(out);lses.append(lse)
    if not outputs:return query.new_empty(query.shape),torch.empty(query.shape[:2],device=query.device,dtype=torch.float32)
    return torch.cat(outputs),torch.cat(lses)


def decode(query,key_cache,head_dim_v,indices_in_kvcache,sched_meta,softmax_scale,attn_sink=None,batch_invariant=False,index_length=None,key_cache_extra=None,indices_in_kvcache_extra=None,index_length_extra=None,fused_norm_rope_cast_kwargs=None):
    if fused_norm_rope_cast_kwargs is not None or batch_invariant:
        raise NotImplementedError("910C reference MLA does not implement fused casting or batch-invariant guarantees")
    _validate(query,indices_in_kvcache)
    if (key_cache_extra is None)!=(indices_in_kvcache_extra is None):raise ValueError("extra cache and its indices must be provided together")
    if key_cache_extra is None and index_length_extra is not None:raise ValueError("extra lengths require an extra cache")
    if key_cache.device!=query.device:raise ValueError("cache and query devices differ")
    outputs=[];lses=[]
    for start in range(0,query.shape[0],16):
        idx=indices_in_kvcache[start:start+16];lengths=None if index_length is None else index_length[start:start+16]
        valid=_mask(idx,key_cache.shape[0]*key_cache.shape[1],lengths,decode=True)
        gathered=_cache_rows(key_cache,idx,valid,query.shape[-1])
        if key_cache_extra is not None:
            if key_cache_extra.device!=query.device or indices_in_kvcache_extra.shape[0]!=query.shape[0]:raise ValueError("extra cache/indices mismatch")
            extra_idx=indices_in_kvcache_extra[start:start+16];extra_len=None if index_length_extra is None else index_length_extra[start:start+16]
            extra_valid=_mask(extra_idx,key_cache_extra.shape[0]*key_cache_extra.shape[1],extra_len,decode=True)
            gathered=torch.cat((gathered,_cache_rows(key_cache_extra,extra_idx,extra_valid,query.shape[-1])),1);valid=torch.cat((valid,extra_valid),1)
        out,lse=_attend(query[start:start+16],gathered,valid,softmax_scale,attn_sink,head_dim_v);outputs.append(out);lses.append(lse)
    if not outputs:return query.new_empty((*query.shape[:2],head_dim_v)),torch.empty(query.shape[:2],device=query.device,dtype=torch.float32)
    return torch.cat(outputs),torch.cat(lses)
