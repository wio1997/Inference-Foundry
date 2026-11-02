"""Explicit bounded FP32 full-masked expanded-KV attention experiment; never auto-installed."""
import torch
from torch import nn
import types
from deepinfer.model_executor.models.glm5_next.cache import append_dsa
UPSTREAM_CODE = 'from __future__ import annotations\ndef repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:\n    """\n    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,\n    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)\n    """\n    batch, num_key_value_heads, slen, head_dim = hidden_states.shape\n    if n_rep == 1:\n        return hidden_states\n    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)\n    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)\n\ndef eager_attention_forward(\n    module: nn.Module,\n    query: torch.Tensor,\n    key: torch.Tensor,\n    value: torch.Tensor,\n    attention_mask: torch.Tensor | None,\n    scaling: float,\n    dropout: float = 0.0,\n    **kwargs: Unpack[TransformersKwargs],\n):\n    key_states = repeat_kv(key, module.num_key_value_groups)\n    value_states = repeat_kv(value, module.num_key_value_groups)\n\n    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling\n    if attention_mask is not None:\n        attn_weights = attn_weights + attention_mask\n\n    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)\n    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)\n    attn_output = torch.matmul(attn_weights, value_states)\n    attn_output = attn_output.transpose(1, 2).contiguous()\n\n    return attn_output, attn_weights\n\nclass ReferenceExpansion:\n    def expand_kv(self, kv_nope: torch.Tensor, k_rot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:\n            """Expands the compressed latents into key and value states. Args:\n                - kv_nope: key + value without positional encoding, shape [batch_size, 1, seqlen, self.kv_lora_rank]\n                - k_rot: shared key with positional encoding, shape [batch_size, 1, seqlen, self.qk_rope_head_dim]\n            Returns the key and value states, two tensors of shape [batch, num_heads, seq, (k or v)_head_dim].\n            """\n            batch_size, _, seq_length, _ = kv_nope.shape\n            key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)\n\n            kv_nope = self.kv_b_proj(kv_nope).view(key_shape).transpose(1, 2)\n            k_nope, value_states = torch.split(kv_nope, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)\n            k_rot = k_rot.expand(-1, k_nope.shape[1], -1, -1)  # does not affect the underlying storage\n\n            # Concatenate k_nope and k_rot in a new tensor\n            key_states = kv_nope.new_empty(*kv_nope.shape[:-1], self.qk_nope_head_dim + self.qk_rope_head_dim)\n            key_states[..., : self.qk_nope_head_dim].copy_(k_nope)\n            key_states[..., self.qk_nope_head_dim :].copy_(k_rot)\n            return key_states, value_states\n'
exec(compile(UPSTREAM_CODE,'preserved_transformers_glm5_next','exec'),globals())

class ExperimentalFp32ExpandedDSA:
    def __init__(self, mla, indexer, operator, selector, *, max_tokens):
        if not callable(selector) or type(max_tokens) is not int or not 1 <= max_tokens <= 2304:
            raise ValueError('explicit selector and 1..2304 token bound required')
        self.mla, self.indexer, self.selector, self.max_tokens = mla, indexer, selector, max_tokens
        self._module = types.SimpleNamespace(num_key_value_groups=1, training=False)
        self._expansion = ReferenceExpansion()
        self._expansion.kv_b_proj = lambda x: torch.nn.functional.linear(x, self.mla.kv_b.weight)
        self._expansion.qk_nope_head_dim, self._expansion.v_head_dim, self._expansion.qk_rope_head_dim = 256, 256, 0

    @torch.inference_mode()
    def __call__(self, step, state):
        start=state.tokens_seen;end=start+len(step.positions)
        if len(step.positions)!=1 or end>self.max_tokens or step.scale!=1/16 or int(step.positions[0])!=start:
            raise ValueError('requires one bounded continuation')
        updated=append_dsa(step.mla.latent,step.indexer.raw_keys,step.indexer.gate_scores,self.indexer.ape,state,self.indexer.layout,start_position=start)
        ids=self.selector(step.indexer.query[0],step.indexer.head_weights[0],updated,self.indexer.layout,max_tokens=self.max_tokens)
        # Match upstream masking semantics by preserving chronological order.
        valid=ids[ids>=0].long().sort().values
        expected=min(end//4,512)*4+end%4
        if valid.numel()!=expected or not bool((valid<end).all()) or (valid.numel()>1 and not bool((valid[1:]>valid[:-1]).all())):
            raise ValueError('invalid causal sparse selection')
        # Keep expanded keys, values, scores and reduction in FP32, then round once at the attention output.
        latents=updated.latent
        expanded=torch.nn.functional.linear(latents.float(),self.mla.kv_b.weight.float()).reshape(1,end,64,512).transpose(1,2)
        keys,values=torch.split(expanded,[256,256],dim=-1)
        query=step.mla.query.transpose(0,1).unsqueeze(0).float()
        mask=torch.full((1,1,1,end),float('-inf'),device=latents.device,dtype=torch.float32)
        mask[...,valid]=0
        output,_=eager_attention_forward(self._module,query,keys,values,mask,step.scale,dropout=0.0)
        result=output.reshape(1,64,256).to(torch.bfloat16)
        if result.dtype!=torch.bfloat16 or not bool(torch.isfinite(result).all()):
            raise ValueError('invalid expanded attention result')
        return result,updated

"""Opt-in ReLU pool selection from the adapter's unpaged DSA state."""
import torch

def select_from_state(query, head_weights, state, layout, *, max_tokens=2304):
    n=state.tokens_seen
    if query.shape!=(32,128) or head_weights.shape!=(32,) or layout.pool_size!=4 or layout.topk_tokens!=2048:
        raise ValueError('unexpected selector geometry')
    if type(n) is not int or n<1 or n>max_tokens or state.compressed.shape!=(n//4,128):
        raise ValueError('invalid state or context bound')
    if query.device!=state.compressed.device or head_weights.device!=query.device:
        raise ValueError('device mismatch')
    if not bool(torch.isfinite(query).all() and torch.isfinite(head_weights).all() and torch.isfinite(state.compressed).all()):
        raise ValueError('nonfinite input')
    count=n//4;take=min(count,512)
    if take:
        scores=(head_weights.float()[:,None]*(query.float()@state.compressed.float().T).relu()).sum(0)
        ids=scores.topk(take).indices
        tokens=(ids[:,None]*4+torch.arange(4,device=query.device)).reshape(-1).to(torch.int32)
    else:tokens=torch.empty((0,),dtype=torch.int32,device=query.device)
    tail=torch.arange(count*4,n,device=query.device,dtype=torch.int64)
    if tail.numel() and not torch.equal(state.tail_positions[tail%layout.tail_capacity],tail):
        raise ValueError('incomplete pool tags differ')
    out=torch.full((2051,),-1,dtype=torch.int32,device=query.device)
    out[:tokens.numel()]=tokens;out[tokens.numel():tokens.numel()+tail.numel()]=tail.int()
    return out

class BoundExperimentalFp32ExpandedDSA:
    """Explicit, one-token experimental DSA; no default or accuracy guarantee."""
    def __init__(self, *, max_total_tokens):
        if type(max_total_tokens) is not int or not 1 <= max_total_tokens <= 2304:
            raise ValueError('explicit 1..2304 token bound required')
        self.max_total_tokens=max_total_tokens
        self._operation=None

    def bind(self, mla, indexer):
        if self._operation is not None:
            raise ValueError('experimental DSA already bound')
        if (mla.num_heads,mla.qk_dim,mla.kv_rank,mla.value_dim)!=(64,256,512,256):
            raise ValueError('GLM MLA geometry differs')
        if (indexer.layout.latent_dim,indexer.layout.index_dim,indexer.layout.pool_size,indexer.layout.topk_tokens,indexer.layout.logical_block_size,indexer.layout.lookahead)!=(512,128,4,2048,128,0):
            raise ValueError('GLM DSA cache geometry differs')
        weight=getattr(mla.kv_b,'weight',None)
        if not isinstance(weight,torch.Tensor) or weight.shape!=(64*512,512) or weight.dtype!=torch.bfloat16:
            raise ValueError('FLOAT BF16 kv_b NK weight required')
        self._operation=ExperimentalFp32ExpandedDSA(mla,indexer,None,select_from_state,max_tokens=self.max_total_tokens)
        return self

    def __call__(self, step, state):
        if self._operation is None:
            raise ValueError('experimental DSA must bind checkpoint projections')
        if step.mla.query.device.type!='npu':
            raise ValueError('experimental DSA requires NPU')
        return self._operation(step,state)

def make_experimental_fp32_expanded_dsa_operations(config, *, max_total_tokens):
    """Create explicit experimental per-layer operations; caller must opt in."""
    return {s.layer_id:BoundExperimentalFp32ExpandedDSA(max_total_tokens=max_total_tokens)
            for s in config.layers if s.attention=='deepseek_sparse_attention'}
