# SPDX-License-Identifier: Apache-2.0
"""Owned mixed-attention snapshots; no scheduler or model forward is implied."""
from dataclasses import dataclass
from types import MappingProxyType

import torch

from .cache import DSAState,validate_dsa_state
from .kda import KDAState


def _clone_kda(state):
    return KDAState(tuple(t.clone() for t in state.conv),state.recurrent.clone(),state.tokens_seen)


@dataclass(frozen=True)
class MixedRequestState:
    request_id: str
    tokens_seen: int
    layers: object

    @classmethod
    def capture(cls,request_id,config,layout,layers):
        """Validate all layers and copy buffers so caller mutations cannot leak."""
        if not isinstance(request_id,str) or not request_id:raise ValueError('request ID must be nonempty')
        specs=config.layers
        if set(layers)!={s.layer_id for s in specs}:raise ValueError('state must cover every backbone layer exactly once')
        counts={s.tokens_seen for s in layers.values()}
        if len(counts)!=1:raise ValueError('mixed layers must agree on the committed token count')
        count=next(iter(counts))
        if type(count) is not int or count<0:raise ValueError('invalid committed token count')
        owned={}
        linear=config.linear_attn_config;h,d,w=(linear[k] for k in ('num_heads','head_dim','short_conv_kernel_size'))
        for spec in specs:
            state=layers[spec.layer_id]
            if spec.attention=='linear_attention':
                if not isinstance(state,KDAState) or len(state.conv)!=3 or any(t.shape!=(w-1,h*d) for t in state.conv) or state.recurrent.shape!=(h,d,d):
                    raise ValueError('invalid KDA state type/geometry')
                if state.recurrent.dtype!=torch.float32 or any(t.dtype!=torch.bfloat16 for t in state.conv) or any(t.device!=state.recurrent.device for t in state.conv):
                    raise ValueError('invalid KDA state dtype/device')
                owned[spec.layer_id]=_clone_kda(state)
            elif spec.attention=='deepseek_sparse_attention':
                if not isinstance(state,DSAState):raise ValueError('DSA layer requires DSA state')
                validate_dsa_state(state,layout);owned[spec.layer_id]=state.clone()
            else:raise ValueError('unknown attention kind')
        return cls(request_id,count,MappingProxyType(owned))

    def advance(self,config,layout,updates,*,num_tokens):
        if type(num_tokens) is not int or num_tokens<0:raise ValueError('num_tokens must be nonnegative')
        result=self.capture(self.request_id,config,layout,updates)
        if result.tokens_seen!=self.tokens_seen+num_tokens:raise ValueError('mixed update must advance every layer by exactly num_tokens')
        return result
