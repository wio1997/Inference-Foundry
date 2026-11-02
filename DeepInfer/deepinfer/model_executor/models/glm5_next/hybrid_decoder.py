# SPDX-License-Identifier: Apache-2.0
"""Prepared hybrid decoder composition with explicit single-request state.

No default DSA operator, model registry, generation, paged allocator, INT8 CPU
fallback or distributed execution. Callers supply prepared attention/FFN
components. The stack stages private state copies, then validates every layer
before returning a committed snapshot. Copying is correctness preparation, not
an optimized serving transaction protocol or a memory/performance claim.
"""
from dataclasses import dataclass

import torch
from torch import nn

from .cache import state_from_aligned_prefix,validate_dsa_state,DSAState
from .decoder import Glm5NextDenseDecoder
from .kda import KDAState
from .mhc import GlmMHC
from .request_state import MixedRequestState
from .rank_dsa import CPUProjectedDSAOperation,CPUProjectedDSAResult


class GlmMoEMLP(nn.Module):
    """Route once, then call the explicit expert adapter (shared handled there)."""
    def __init__(self,router,experts,*,hidden_size,num_experts):
        super().__init__()
        if type(hidden_size) is not int or hidden_size<1 or type(num_experts) is not int or num_experts<1:
            raise ValueError('invalid MoE dimensions')
        if experts.num_experts!=num_experts:raise ValueError('router/expert global namespaces disagree')
        if router.weight.shape!=(num_experts,hidden_size):raise ValueError('router geometry disagrees with MoE')
        self.router,self.experts,self.hidden_size,self.num_experts=router,experts,hidden_size,num_experts

    @torch.inference_mode()
    def forward(self,hidden):
        if hidden.ndim!=2 or hidden.shape[-1]!=self.hidden_size or hidden.dtype!=torch.bfloat16:
            raise ValueError('MoE hidden must be BF16 [tokens,hidden]')
        if not len(hidden):return torch.empty_like(hidden)
        result=self.experts(hidden,self.router(hidden))
        if result.shape!=hidden.shape or result.dtype!=hidden.dtype or result.device!=hidden.device:
            raise ValueError('expert adapter must preserve hidden geometry/dtype/device')
        return result


@dataclass(frozen=True)
class DSAStep:
    mla: object  # MLAProjection (query rank, BF16 query, normalized latent)
    indexer: object  # IndexerProjection (query/head weights, FP32 raw K/gates)
    positions: torch.Tensor
    scale: float


class PreparedDSAAttention(nn.Module):
    """Bind projections to an explicit operation over unpaged DSAState.

    operation(step,state) returns BF16 per-head values and the new DSAState.
    CPUProjectedDSAOperation explicitly returns a projected result and skips
    mla.finish; ordinary operations still return per-head values.
    It owns cache update/selection/attention exactly once. No implicit CPU
    reference fallback is installed. Native paged state is a separate interface.
    """
    def __init__(self,mla,indexer,operation):
        super().__init__()
        if mla.kv_rank!=indexer.layout.latent_dim or mla.hidden_size!=indexer.hidden_size or mla.q_rank!=indexer.q_rank:
            raise ValueError('MLA/indexer projection dimensions disagree')
        if operation is None:raise ValueError('DSA operation must be provided explicitly')
        self.mla,self.indexer,self.operation=mla,indexer,operation
        self.layout=indexer.layout

    @torch.inference_mode()
    def forward(self,hidden,state=None,*,operation=None):
        if hidden.ndim!=2 or hidden.shape[1]!=self.mla.hidden_size or hidden.dtype!=torch.bfloat16:
            raise ValueError('DSA hidden must be BF16 [tokens,hidden]')
        if state is None:
            state=state_from_aligned_prefix(hidden.new_empty((0,self.layout.latent_dim)),hidden.new_empty((0,self.layout.index_dim)),self.layout)
        if not isinstance(state,DSAState):raise ValueError('DSA requires explicit DSAState')
        validate_dsa_state(state,self.layout)
        if state.latent.device!=hidden.device:raise ValueError('DSA state/hidden device mismatch')
        selected=self.operation if operation is None else operation
        projected=type(selected) is CPUProjectedDSAOperation
        if projected:selected.validate_input(hidden,state,self.mla,self.layout)
        start=state.tokens_seen
        if not len(hidden):return torch.empty_like(hidden),state.clone()
        mla=selected.project_input(self.mla,hidden) if projected else self.mla.project(hidden)
        indexer=self.indexer(hidden,mla.query_rank)
        step=DSAStep(mla,indexer,torch.arange(start,start+len(hidden),device=hidden.device,dtype=torch.int64),self.mla.qk_dim**-.5)
        values,next_state=selected(step,state.clone() if projected else state)
        if not isinstance(next_state,DSAState):raise ValueError('DSA operation returned wrong state type')
        validate_dsa_state(next_state,self.layout)
        if next_state.tokens_seen!=start+len(hidden):raise ValueError('DSA operation must advance exactly one token per input row')
        if next_state.latent.device!=hidden.device:raise ValueError('DSA operation changed state device')
        if projected:
            result=values.hidden.contiguous().clone()
        else:
            if type(values) is CPUProjectedDSAResult:raise ValueError('projected output requires explicit CPUProjectedDSAOperation')
            result=self.mla.finish(values)
        if result.shape!=hidden.shape or result.dtype!=hidden.dtype or result.device!=hidden.device:raise ValueError('DSA output contract mismatch')
        return result,next_state.clone() if projected else next_state


class GlmHybridDecoder(Glm5NextDenseDecoder):
    """Reuse materialized mHC/RMSNorm order for KDA/DSA and dense/MoE FFNs."""
    def __init__(self,*,attention_kind,mlp_kind,**components):
        if attention_kind not in ('linear_attention','deepseek_sparse_attention') or mlp_kind not in ('dense','sparse'):
            raise ValueError('unknown decoder component kind')
        if (mlp_kind=='sparse')!=isinstance(components['mlp'],GlmMoEMLP):raise ValueError('MoE kind requires an explicit GlmMoEMLP adapter')
        super().__init__(**components)
        self.attention_kind,self.mlp_kind=attention_kind,mlp_kind
        self.native_tokenwise=False  # explicitly enabled by diagnostic native factory

    @classmethod
    def from_checkpoint(cls,*args,**kwargs):
        raise ValueError('hybrid composition requires explicit attention/FFN; use from_checkpoint_components')

    @classmethod
    def from_checkpoint_components(cls,reader,layer_id,*,attention,mlp,device='cpu'):
        """Load FLOAT mHC/norms only; attention/FFN are mandatory supplied modules.

        This does not load/dequantize INT8 projections or experts. No runnable
        full-model factory or native operator authorization is implied.
        """
        from deepinfer.hugging_face.glm5_next_config import Glm5NextConfig
        config=Glm5NextConfig.from_checkpoint(reader.root)
        if type(layer_id) is not int or not 0<=layer_id<config.num_hidden_layers:raise ValueError('invalid backbone layer ID')
        spec=config.layers[layer_id];text=config.hf_config['text_config']
        if (spec.mlp=='sparse')!=isinstance(mlp,GlmMoEMLP):raise ValueError('configured FFN kind requires explicit matching component')
        if not text['mhc']:raise ValueError('this prepared decoder requires mHC')
        root=f'model.language_model.layers.{layer_id}.';norms=[]
        for suffix in ('input_layernorm.weight','post_attention_layernorm.weight'):
            name=root+suffix
            if reader.quant.get(name)!='FLOAT':raise ValueError('decoder norm must be declared FLOAT')
            value,dtype=reader.tensor(name)
            if dtype not in ('BF16','F32') or value.shape!=(config.hidden_size,):raise ValueError('invalid decoder norm payload')
            norms.append(value.to(device))
        return cls(layer_id=layer_id,attention_kind=spec.attention,mlp_kind=spec.mlp,
            attn_hc=GlmMHC.from_checkpoint(reader,layer_id,'attn',device),ffn_hc=GlmMHC.from_checkpoint(reader,layer_id,'ffn',device),
            input_norm=norms[0],ffn_norm=norms[1],attention=attention,mlp=mlp,norm_eps=text['rms_norm_eps']).eval()

    @torch.inference_mode()
    def forward(self,hidden,state=None,*,dsa_operation=None):
        if dsa_operation is not None and (self.attention_kind!='deepseek_sparse_attention' or not isinstance(self.attention,PreparedDSAAttention) or not callable(dsa_operation)):
            raise ValueError('explicit DSA operation requires prepared DSA attention')
        selected=self.attention.operation if dsa_operation is None and isinstance(self.attention,PreparedDSAAttention) else dsa_operation
        if type(selected) is CPUProjectedDSAOperation and selected.plan.layer_id!=self.layer_id:raise ValueError('projected DSA operation layer differs')
        if hidden.ndim not in (2,3) or hidden.dtype!=torch.bfloat16:raise ValueError('hybrid residual must be BF16 collapsed or materialized streams')
        expected=KDAState if self.attention_kind=='linear_attention' else DSAState
        if state is not None and not isinstance(state,expected):raise ValueError('decoder attention state kind mismatch')
        # Stable per-token native execution avoids batch-shape-dependent BF16
        # perturbations amplifying across layers. This is a correctness mode,
        # not a throughput optimization; ordinary prepared paths remain batched.
        if self.native_tokenwise and len(hidden)>1:
            if type(selected) is CPUProjectedDSAOperation:raise ValueError('CPU projected plans cannot be split by native tokenwise mode')
            current=state;parts=[]
            for row in hidden.split(1,dim=0):
                output,current=self.forward(row,current,dsa_operation=dsa_operation)
                parts.append(output)
            return torch.cat(parts,dim=0),current
        start=0 if state is None else state.tokens_seen
        attention=None if dsa_operation is None else lambda x,s:self.attention(x,s,operation=dsa_operation)
        output,next_state=super().forward(hidden,state,attention_forward=attention)
        if not isinstance(next_state,expected) or next_state.tokens_seen!=start+len(hidden):raise ValueError('decoder state must advance exactly by input rows')
        if output.ndim!=3 or output.shape!=(len(hidden),self.attn_hc.streams,self.attn_hc.hidden_size) or output.dtype!=hidden.dtype or output.device!=hidden.device:
            raise ValueError('decoder must preserve materialized residual stream contract')
        return output,next_state


class PreparedDecoderStack(nn.Module):
    """Caller-built backbone component stack; no embedding/logits/generation.

    All layers must be present in config. Smaller test configs are synthetic
    seams, not evidence of a runnable45-layer model. A failed step returns no
    commit and cannot mutate the caller's hidden/state through staged buffers.
    """
    def __init__(self,config,layout,layers):
        super().__init__()
        expected=[s.layer_id for s in config.layers]
        if not expected or sorted(layers)!=expected or expected!=list(range(len(expected))):raise ValueError('stack must cover all contiguous backbone layer IDs')
        for spec in config.layers:
            layer=layers[spec.layer_id]
            if layer.layer_id!=spec.layer_id or (layer.attention_kind,layer.mlp_kind)!=(spec.attention,spec.mlp):raise ValueError('decoder modules disagree with configured layer kinds')
        self.config,self.layout=config,layout
        self.layers=nn.ModuleList([layers[i] for i in expected])

    @torch.inference_mode()
    def forward(self,hidden,state,*,dsa_operations=None):
        if dsa_operations is not None:
            expected={s.layer_id for s in self.config.layers if s.attention=='deepseek_sparse_attention'}
            if set(dsa_operations)!=expected or any(not callable(v) for v in dsa_operations.values()):
                raise ValueError('step-scoped operations must cover every DSA layer exactly')
        if not isinstance(state,MixedRequestState):raise ValueError('stack requires a complete request state snapshot')
        for layer in self.layers:
            selected=(dsa_operations or {}).get(layer.layer_id)
            if selected is None and isinstance(getattr(layer,'attention',None),PreparedDSAAttention):selected=layer.attention.operation
            if type(selected) is CPUProjectedDSAOperation and selected.plan.request_id!=state.request_id:raise ValueError('projected DSA operation request differs')
            if type(selected) is CPUProjectedDSAOperation and selected.plan.layer_id!=layer.layer_id:raise ValueError('projected DSA operation layer differs')
        working=MixedRequestState.capture(state.request_id,self.config,self.layout,state.layers)
        if working.tokens_seen!=state.tokens_seen:raise ValueError('request token counter disagrees with layer states')
        value=hidden.clone();updates={}
        for layer in self.layers:
            kwargs={} if dsa_operations is None or layer.layer_id not in dsa_operations else {'dsa_operation':dsa_operations[layer.layer_id]}
            value,updates[layer.layer_id]=layer(value,working.layers[layer.layer_id],**kwargs)
        committed=state.advance(self.config,self.layout,updates,num_tokens=len(hidden))
        return value,committed
