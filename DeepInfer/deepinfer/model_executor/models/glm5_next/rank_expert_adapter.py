# SPDX-License-Identifier: Apache-2.0
"""Opt-in installation of replicated-input rank-owned MoE for native layers."""
from torch import nn
from .expert_bank import CheckpointSelectedExperts
from .hybrid_decoder import GlmMoEMLP
from .replicated_rank_moe import replicated_rank_moe

class ReplicatedRankExperts(nn.Module):
    """Diagnostic one-token local expert interface; no production EP dispatch."""
    def __init__(self, reader, layer_id, num_experts, *, rank, world_size,
                 cache_bytes, max_assembled_bytes, max_packed_rows, group=None, cache_session=None):
        super().__init__()
        if type(num_experts) is not int or num_experts<1 or reader.config['text_config']['n_routed_experts']!=num_experts:
            raise ValueError('reader and router expert namespaces must agree')
        if type(layer_id) is not int or not 0<=layer_id<len(reader.config['text_config']['mlp_layer_types']) or reader.config['text_config']['mlp_layer_types'][layer_id]!='sparse':
            raise ValueError('sparse layer required')
        self.reader,self.layer_id,self.num_experts=reader,layer_id,num_experts
        self.rank,self.world_size=rank,world_size
        self.cache_bytes,self.max_assembled_bytes,self.max_packed_rows=cache_bytes,max_assembled_bytes,max_packed_rows
        self.group=group;self.last_metadata=None;self.cache_session=cache_session
        if cache_session is not None:
            from .cached_expert_session import CachedExpertBankSession
            if not isinstance(cache_session,CachedExpertBankSession):raise ValueError('explicit reusable expert bank session required')
            cache_session.validate_binding(reader,rank=rank,world_size=world_size,cache_bytes=cache_bytes,max_assembled_bytes=max_assembled_bytes)

    def forward(self,hidden_states,routes):
        result,metadata=replicated_rank_moe(self.reader,hidden_states,routes,layer_id=self.layer_id,
            rank=self.rank,world_size=self.world_size,cache_bytes=self.cache_bytes,
            max_assembled_bytes=self.max_assembled_bytes,max_packed_rows=self.max_packed_rows,group=self.group,cache_session=self.cache_session)
        self.last_metadata=metadata
        return result


def install_rank_experts(layer,reader,*,rank,world_size,cache_bytes,max_assembled_bytes,max_packed_rows,group=None,cache_session=None):
    if not isinstance(layer.mlp,GlmMoEMLP) or not isinstance(layer.mlp.experts,CheckpointSelectedExperts):
        raise ValueError('native selected-expert MoE layer required')
    if layer.mlp.experts.reader is not reader or layer.mlp.experts.layer_id!=layer.layer_id:
        raise ValueError('decoder and supplied checkpoint reader differ')
    adapter=ReplicatedRankExperts(reader,layer.layer_id,layer.mlp.num_experts,rank=rank,world_size=world_size,
        cache_bytes=cache_bytes,max_assembled_bytes=max_assembled_bytes,max_packed_rows=max_packed_rows,group=group,cache_session=cache_session)
    layer.mlp.experts=adapter
    return layer
