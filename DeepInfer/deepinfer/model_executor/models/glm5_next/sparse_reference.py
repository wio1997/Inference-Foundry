# SPDX-License-Identifier: Apache-2.0
"""Bounded CPU mathematical contracts for GLM selection and sparse MLA.

Not a production fallback, model forward or performance implementation. Pool
scores follow the extracted GLM Triton path: FP32 weighted-query head sum then
dot with compressed keys, WITHOUT per-head ReLU. Installed CANN selection and
native tie ordering require separate validation. CPU ties choose lower pool ID.
"""
from dataclasses import dataclass
import math

import torch

from .cache import expand_pool_indices


def _cpu(*tensors):
    if any(t.device.type!='cpu' for t in tensors):
        raise ValueError('bounded mathematical reference is CPU-only')


def _bound(value,name):
    if type(value) is not int or value<1:raise ValueError(f'{name} must be a positive integer')


@dataclass(frozen=True)
class PoolSelection:
    pool_ids: torch.Tensor  # INT32 packed prefix, -1 padding
    pool_scores: torch.Tensor  # FP32 ranked, -inf padding
    token_indices: torch.Tensor  # INT32 expanded complete pools + raw causal tail
    visible_pools: torch.Tensor  # INT64 per query


@torch.inference_mode()
def select_pools_reference(query,head_weights,compressed,positions,layout,*,active_rows=None,max_score_elements=262144):
    """One request's completed cache; head weights are already scaled once.

    The compressed cache must cover all completed pools at every active query.
    Future cache entries are masked by query position. No page lookup is implied.
    """
    _bound(max_score_elements,'max_score_elements');_cpu(query,head_weights,compressed,positions)
    if query.ndim!=3 or query.shape[-1]!=layout.index_dim or query.dtype!=torch.bfloat16:
        raise ValueError('query must be BF16 [tokens,heads,index_dim]')
    tokens,heads,width=query.shape
    if heads<1 or head_weights.shape!=(tokens,heads) or head_weights.dtype!=torch.bfloat16:
        raise ValueError('head weights must be scaled BF16 [tokens,heads]')
    if compressed.ndim!=2 or compressed.shape[-1]!=width or compressed.dtype!=torch.bfloat16:
        raise ValueError('compressed keys must be BF16 [pools,index_dim]')
    if positions.shape!=(tokens,) or positions.dtype not in (torch.int32,torch.int64):raise ValueError('query positions must be integer [tokens]')
    if tokens*compressed.shape[0]>max_score_elements:raise ValueError('score matrix exceeds explicit CPU reference bound')
    if active_rows is None:active_rows=torch.ones(tokens,dtype=torch.bool)
    _cpu(active_rows)
    if active_rows.shape!=(tokens,) or active_rows.dtype!=torch.bool:raise ValueError('active_rows must be a boolean vector')
    if bool(((positions<0)&active_rows).any()):raise ValueError('active query positions must be nonnegative')
    visible=torch.where(active_rows,(positions.long()+1)//layout.pool_size,0)
    if bool((visible>compressed.shape[0]).any()):raise ValueError('compressed history does not cover active query position')
    count=layout.topk_tokens//layout.pool_size
    ids=torch.full((tokens,count),-1,dtype=torch.int32);selected=torch.full((tokens,count),float('-inf'))
    visible_max=int(visible.max()) if tokens else 0
    if any(not bool(torch.isfinite(t).all()) for t in (query[active_rows],head_weights[active_rows],compressed[:visible_max])):raise ValueError('visible indexer operands must be finite')
    qbar=(query.float()*head_weights.float().unsqueeze(-1)).sum(dim=1)
    scores=torch.nn.functional.linear(qbar,compressed[:visible_max].float())
    for row,n in enumerate(visible.tolist()):
        k=min(count,n)
        if k:
            if not bool(torch.isfinite(scores[row,:n]).all()):raise ValueError('pool score overflow')
            # Stable descending sort keeps ascending IDs for exact CPU ties.
            ranked=torch.argsort(scores[row,:n],descending=True,stable=True)[:k]
            ids[row,:k]=ranked.int();selected[row,:k]=scores[row,ranked]
    indices=expand_pool_indices(ids,torch.where(active_rows,positions,0).clamp_min(0),layout)
    indices[~active_rows]=-1
    return PoolSelection(ids,selected,indices,visible)


@torch.inference_mode()
def sparse_mla_reference(query,keys,values,indices,positions,*,scale=None,max_work_elements=1048576):
    """Single-request BF16 Q/K/V, FP32 logits/softmax/reduction, BF16 output.

    Input keys are expanded MLA QK heads, NOT latent512 nor index128. Caller
    binds native kv_b/absorbed decode separately. Empty/padded selections return
    zero output and -inf LSE; valid rows must be unique causal packed prefixes.
    """
    _bound(max_work_elements,'max_work_elements');_cpu(query,keys,values,indices,positions)
    if query.ndim!=3 or keys.ndim!=3 or values.ndim!=3 or query.shape[1:]!=keys.shape[1:] or keys.shape[:2]!=values.shape[:2]:
        raise ValueError('invalid MLA Q/K/V token/head geometry')
    tokens,heads,qk=query.shape;value_dim=values.shape[-1]
    if heads<1 or qk<1 or value_dim<1 or any(t.dtype!=torch.bfloat16 for t in (query,keys,values)):
        raise ValueError('MLA Q/K/V require positive BF16 head dimensions')
    if indices.ndim!=2 or indices.shape[0]!=tokens or indices.dtype not in (torch.int32,torch.int64):raise ValueError('selected indices must be an integer matrix')
    if positions.shape!=(tokens,) or positions.dtype not in (torch.int32,torch.int64):raise ValueError('MLA positions must be an integer vector')
    if bool((indices < -1).any()):raise ValueError('only -1 padding is valid')
    selected_count=int((indices>=0).sum())
    if selected_count*heads*(qk+value_dim)>max_work_elements:raise ValueError('selected work exceeds explicit CPU reference bound')
    scale=qk**-.5 if scale is None else scale
    if not isinstance(scale,(int,float)) or not math.isfinite(scale) or scale<=0:raise ValueError('invalid MLA score scale')
    output=torch.zeros(tokens,heads,value_dim,dtype=torch.bfloat16);lse=torch.full((tokens,heads),float('-inf'))
    for row in range(tokens):
        ids=indices[row];valid=ids[ids>=0].long();count=valid.numel()
        if bool((ids[:count]<0).any()) or bool((ids[count:]!=-1).any()):raise ValueError('selected IDs must form a packed valid prefix')
        if not count:continue
        if valid.unique().numel()!=count or bool(((valid>=keys.shape[0])|(valid>positions[row])).any()):raise ValueError('selected IDs must be unique, in range and causal')
        if any(not bool(torch.isfinite(t).all()) for t in (query[row],keys[valid],values[valid])):raise ValueError('selected MLA operands must be finite')
        logits=torch.einsum('hd,shd->hs',query[row].float(),keys[valid].float())*scale
        if not bool(torch.isfinite(logits).all()):raise ValueError('MLA logit overflow')
        probs=torch.softmax(logits,dim=-1)
        output[row]=torch.einsum('hs,shv->hv',probs,values[valid].float()).bfloat16()
        lse[row]=torch.logsumexp(logits,dim=-1)
    return output,lse
