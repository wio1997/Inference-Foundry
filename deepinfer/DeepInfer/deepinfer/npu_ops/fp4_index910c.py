# SPDX-License-Identifier: Apache-2.0
"""Opt-in local FP4 index arithmetic and paged selection for Ascend910C.

Reuse DeepInfer projections, page ownership, source-K sharing and graph inputs.
The backend keeps E2M1/UE8M0 storage and candidate block-max/top-k semantics;
it does not claim the launch efficiency of the native Ascend950 FP4 kernel.
"""
import torch
import torch.nn.functional as F
from .fp4_index_quant910c import quantize_group32, pack_group32, unpack_group32

_RUNTIME = {}
_SENTINEL = 2**24


def prepare_runtime(device):
    """Create immutable small device constants before graph capture."""
    device = torch.device(device)
    key = str(device)
    if key not in _RUNTIME:
        _RUNTIME[key] = (torch.tensor([0,1,2,3,4,6,8,12], dtype=torch.int8, device=device),
                         torch.ones(1, dtype=torch.float32, device=device))
    return _RUNTIME[key]


def quantize_packed(value):
    codes, scales = quantize_group32(value)
    return pack_group32(value, codes, scales)


def _integer_matmul(query, key, one):
    if query.device.type == 'cpu':
        return torch.matmul(query.int(), key.int())
    import torch_npu
    return torch_npu.npu_quant_matmul(query, key, one, output_dtype=torch.int32)


def _dense_logits(q, qs, k, ks, weights, one):
    # q [T,H,4,32], k [N,4,32], logits [T,N].
    t, h = q.shape[:2]
    n = k.shape[0]
    dot = _integer_matmul(q.permute(2,0,1,3).reshape(4,t*h,32).contiguous(),
                          k.permute(1,2,0).contiguous(), one).reshape(4,t,h,n)
    head = torch.zeros((t,h,n), dtype=torch.float32, device=q.device)
    for group in range(4):
        scale = qs[:,:,group].float()[:,:,None] * ks[:,group].float()[None,None,:] / 4.0
        head.add_(dot[group].float() * scale)
    return (head.relu() * weights.float()[:,:,None]).sum(1).bfloat16()


def _sparse_logits(q, qs, k, ks, weights, one):
    # Per query candidate keys: k [T,N,4,32], dot [T,4,H,N].
    t, h = q.shape[:2]
    n = k.shape[1]
    dot = _integer_matmul(q.permute(0,2,1,3).contiguous(),
                          k.permute(0,2,3,1).contiguous(), one)
    head = torch.zeros((t,h,n), dtype=torch.float32, device=q.device)
    for group in range(4):
        scale = qs[:,:,group].float()[:,:,None] * ks[:,:,group].float()[:,None,:] / 4.0
        head.add_(dot[:,group].float() * scale)
    return (head.relu() * weights.float()[:,:,None]).sum(1).bfloat16()


def _sorted_indices(indices, valid, width):
    """Match logical chronological indices, with -1 padding and fixed width."""
    values = torch.where(valid, indices, _SENTINEL).float().sort(-1).values
    if values.shape[-1] < width:
        values = F.pad(values, (0,width-values.shape[-1]), value=_SENTINEL)
    values = values[:,:width]
    return torch.where(values < _SENTINEL, values.int(), -1)


def _top_tokens(logits, token_ids, width):
    count = min(width, logits.shape[1])
    values, local = logits.float().topk(count, dim=-1, sorted=False)
    if token_ids.ndim == 1:
        indices = token_ids[local]
    else:
        indices = token_ids.gather(1, local)
    return _sorted_indices(indices, values != float('-inf'), width)


def _top_blocks(logits, visible, width, block_size):
    pad = (-logits.shape[1]) % block_size
    scores = F.pad(logits.float(), (0,pad), value=float('-inf')) if pad else logits.float()
    scores = scores.reshape(scores.shape[0], -1, block_size).amax(-1)
    count = min(width, scores.shape[1])
    values, ids = scores.topk(count, dim=-1, sorted=False)
    valid_blocks = torch.div(visible+block_size-1, block_size, rounding_mode='floor')
    return _sorted_indices(ids, (values != float('-inf')) & (ids < valid_blocks[:,None]), width)


def select(query, key, weights, query_scale, key_scale, batch, block_table,
           *, ratio, candidate_mode=3, candidates=None, topk=512,
           candidate_topk_blocks=2048, candidate_block_size=8, query_chunk=8):
    """Packed FP4 paged QLI replacement; lengths/causality use device positions.

    candidate_mode1 emits max-score blocks, mode2 computes only candidate keys,
    mode3 computes dense scores. This uses the same compressed logical indices
    as the existing DeepInfer attention caller. Page tables may be shuffled.
    """
    if (query.ndim != 3 or query.dtype != torch.int8 or query.shape[-1] != 64
            or query_scale.shape != (*query.shape[:2],4) or query_scale.dtype != torch.uint8
            or key.ndim != 4 or key.shape[-2:] != (1,64) or key.dtype != torch.int8
            or key_scale.shape != (*key.shape[:3],4) or key_scale.dtype != torch.uint8
            or weights.shape != query.shape[:2] or weights.dtype not in (torch.bfloat16,torch.float16)
            or query.shape[0] != sum(batch.query_lengths)):
        raise ValueError('invalid packedFP4/UE8M0 index ABI')
    if ratio not in (1,2) or candidate_mode not in (1,2,3) or (candidate_mode==2) != (candidates is not None):
        raise ValueError('invalid compressed ratio or candidate mode')
    if (block_table.dtype != torch.int32 or block_table.shape[0] != len(batch.query_lengths)
            or topk < 1 or candidate_topk_blocks < 1 or candidate_block_size not in (8,16)
            or query_chunk < 1):
        raise ValueError('invalid index metadata')
    levels, one = _RUNTIME[str(query.device)]
    if candidates is not None:
        if candidates.ndim==3 and candidates.shape[1]==1:
            candidates=candidates[:,0]
        if candidates.shape != (query.shape[0],candidate_topk_blocks) or candidates.dtype != torch.int32:
            raise ValueError('candidate rows must match the current query')
    positions = batch.positions()
    if positions.shape != query.shape[:1]:
        raise ValueError('position vector must match query rows')
    block_size = key.shape[1]
    capacity = block_table.shape[1]*block_size
    if capacity >= _SENTINEL or capacity % 32:
        raise ValueError('key capacity must be a multiple of32 and less than2**24')
    flat_key = key.flatten(0,1)
    flat_scale = key_scale.flatten(0,1)
    outputs=[]; emitted=[]; offset=0
    for request, count in enumerate(batch.query_lengths):
        if candidate_mode != 2:
            # Capacity is static in a graph; actual visible length remains a
            # device value. Never truncate to stale host seq_lengths.
            pages = block_table[request].long()
            kp=key.index_select(0,pages).reshape(capacity,64)
            sp=key_scale.index_select(0,pages).reshape(capacity,4)
            k,ks=unpack_group32(kp,sp,levels=levels)
            token_ids=torch.arange(capacity,dtype=torch.int32,device=query.device)
        for first in range(offset,offset+count,query_chunk):
            last=min(first+query_chunk,offset+count)
            q,qs=unpack_group32(query[first:last],query_scale[first:last],levels=levels)
            visible=torch.div(positions[first:last]+1,ratio,rounding_mode='floor')
            if candidate_mode == 2:
                blocks=candidates[first:last]
                offsets=torch.arange(candidate_block_size,dtype=torch.int32,device=query.device)
                logical=(blocks[:,:,None]*candidate_block_size+offsets).flatten(1)
                valid=(blocks[:,:,None]>=0).expand(-1,-1,candidate_block_size).flatten(1)
                valid=valid & (logical>=0) & (logical<visible[:,None]) & (logical<capacity)
                safe=logical.clamp(0,capacity-1)
                pages=block_table[request][torch.div(safe,block_size,rounding_mode='floor').long()]
                physical=pages.long()*block_size+safe.remainder(block_size).long()
                kp=flat_key.index_select(0,physical.flatten()).reshape(last-first,-1,64)
                sp=flat_scale.index_select(0,physical.flatten()).reshape(last-first,-1,4)
                sk,sks=unpack_group32(kp,sp,levels=levels)
                logits=_sparse_logits(q,qs,sk,sks,weights[first:last],one)
                logits=logits.masked_fill(~valid,float('-inf'))
                outputs.append(_top_tokens(logits,logical,topk))
            else:
                logits=_dense_logits(q,qs,k,ks,weights[first:last],one)
                logits=logits.masked_fill(token_ids[None,:]>=visible[:,None],float('-inf'))
                outputs.append(_top_tokens(logits,token_ids,topk))
                if candidate_mode==1:
                    emitted.append(_top_blocks(logits,visible,candidate_topk_blocks,candidate_block_size))
        offset+=count
    selected=torch.cat(outputs,0)
    return selected, torch.cat(emitted,0)[:,None] if emitted else None
