# SPDX-License-Identifier: Apache-2.0
"""NoPE via stock Sparse API with exactly zero auxiliary features.

The model's latent cache remains BF16 dim512. App-local zero tensors satisfy
CANN's required dim64 RoPE inputs: q.k + 0.0 == q.k, with scale unchanged.
These immutable constants are reconstructible, not model/KV/PD state. This is
an eager bounded candidate until operator and full-model gates pass.
"""
from functools import lru_cache
import torch
import torch_npu


@lru_cache(maxsize=2)
def _zero_key_rope(device, shape):
    return torch.zeros((*shape[:-1], 64), device=device, dtype=torch.bfloat16)


def flash_sparse_nope(query, cache, indices, table, qends, kvlens, scale):
    if torch.npu.current_device() != query.device.index:
        raise RuntimeError('NoPE query must be on the current NPU device')
    if query.dtype != torch.bfloat16 or cache.dtype != torch.bfloat16:
        raise TypeError('Qualified Flash Sparse cache/query must be BF16')
    if query.ndim != 3 or query.shape[-1] != 512 or not 0 < query.shape[0] <= 4096:
        raise ValueError('Initial eager NoPE reference requires [1..4096,H,512] query')
    if cache.ndim != 4 or tuple(cache.shape[1:]) != (128,1,512):
        raise ValueError('Expected native latent cache [pages,128,1,512]')
    if scale != .0625:
        raise ValueError('NoPE scale must stay .0625; auxiliary zeros do not change it')
    if indices.ndim == 2:
        indices = indices.unsqueeze(1)
    if tuple(indices.shape) != (query.shape[0],1,2051):
        raise ValueError('Expected padded Sparse indices [T,1,2051]')
    for t in [cache,indices,table,qends,kvlens]:
        if t.device != query.device:
            raise ValueError('NoPE query/cache/metadata must share a device')
    qzero=torch.zeros((*query.shape[:-1],64),device=query.device,dtype=query.dtype)
    kzero=_zero_key_rope(str(cache.device),tuple(cache.shape))
    return torch_npu.npu_sparse_flash_attention(
        query=query.contiguous(),key=cache,value=cache,
        sparse_indices=indices.contiguous(),block_table=table,
        actual_seq_lengths_query=qends,actual_seq_lengths_kv=kvlens,
        query_rope=qzero,key_rope=kzero,scale_value=scale,
        sparse_block_size=1,layout_query='TND',layout_kv='PA_BSND',
        sparse_mode=3,attention_mode=2,return_softmax_lse=False)[0]
