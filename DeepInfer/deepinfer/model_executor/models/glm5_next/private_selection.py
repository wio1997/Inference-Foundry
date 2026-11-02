# SPDX-License-Identifier: Apache-2.0
"""Explicit bounded CPU candidate index selection -> absorbed MLA reference.

No default callback, native operator/context/cache binding or model fallback is
installed. Caller supplies query/head weights and selected-head BF16 MLA weights.
This correctness bridge gathers bounded dense latent history;it is not a serving
memory design or a validation of CANN selection/ties/SFA/SMLA semantics.
"""
from dataclasses import dataclass
import torch
from .absorbed_mla import absorbed_sparse_mla_cpu
from .private_request import CandidateDSAReadView


@dataclass(frozen=True)
class PrivateSparseResult:
    selection: object
    attention: object


@torch.inference_mode()
def private_absorbed_mla_cpu(view,index_query,head_weights,query,weights,*,sink_policy,
                             max_tokens=128,max_latent_tokens=4096,max_score_elements=262144,max_work_elements=32*1024**2):
    """Read completed candidate compressed rows, choose pools/tail, read latent.

    Positions are bound to the candidate append. Head IDs must already be sliced
    consistently with weights by the caller. CPU stable lower-ID pool ties and
    original QK scale are inherited from the existing references. No extra score
    scaling, ReLU or sink is inserted. Returned tensors own reference buffers.
    """
    if sink_policy!='absent':raise ValueError('only explicit absent sink supported')
    if not isinstance(view,CandidateDSAReadView):raise ValueError('explicit candidate CPU read view required')
    for value in (max_tokens,max_latent_tokens,max_score_elements,max_work_elements):
        if type(value) is not int or value<1:raise ValueError('positive CPU bounds required')
    c=weights.contract;l=view.layout
    if (c.kv_rank,c.pool_size,c.topk_tokens)!=(l.latent_dim,l.pool_size,l.topk_tokens):raise ValueError('selection/absorbed geometry differs')
    if len(view.positions)>max_tokens or view.tokens_seen>max_latent_tokens:raise ValueError('candidate token/history bound exceeded')
    # Query must match this exact staged append, before any page read/selection.
    if not isinstance(query,torch.Tensor) or query.device.type!='cpu' or query.dtype!=torch.bfloat16 or query.shape!=(len(view.positions),c.heads,c.qk_dim):
        raise ValueError('candidate query must be CPU BF16 [step_tokens,selected_heads,QK]')
    selection=view.select_pools(index_query,head_weights,max_history_pools=max(1,max_latent_tokens//l.pool_size),max_score_elements=max_score_elements)
    latent=view.gather(torch.arange(view.tokens_seen,dtype=torch.int64))
    result=absorbed_sparse_mla_cpu(query,latent,selection.token_indices,torch.tensor(view.positions,dtype=torch.int64),weights,
        sink_policy=sink_policy,max_tokens=max_tokens,max_latent_tokens=max_latent_tokens,max_work_elements=max_work_elements)
    return PrivateSparseResult(selection,result)
