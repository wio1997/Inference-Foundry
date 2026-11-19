# SPDX-License-Identifier: Apache-2.0
"""NoPE preparation reusing existing SGLang/NPU MLA projections and core."""
import torch
import torch_npu
from sglang.srt.runtime_context import get_parallel
from sglang.srt.state_capturer.indexer_topk import maybe_capture_indexer_topk


def prepare_nope(m,positions,hidden_states,forward_batch,zero_allocator,input_on_attn_tp_slices,prev_topk_indices=None):
    if m.qk_rope_head_dim!=0 or not m.use_dsa:
        raise ValueError('Adapter entry is only for NoPE sparse MLA')
    if get_parallel().attn_cp_size!=1 or get_parallel().attn_dcp_size!=1:
        raise NotImplementedError('Initial NoPE MLA probe uses CP1/DCP1')
    from sglang.srt.model_executor.forward_context import get_attn_backend
    from glm53_sglang_adapter.decode_endpoint import refresh_decode_endpoint
    backend=get_attn_backend()
    refresh_decode_endpoint(getattr(backend,'full_attn_backend',backend),forward_batch,positions)
    if input_on_attn_tp_slices and hidden_states.shape[0]!=positions.numel():
        from sglang.srt.layers.attention.dsa.dsa_npu_indexer import scattered_to_tp_attn_full
        hidden_states=scattered_to_tp_attn_full(hidden_states,forward_batch)
    latent=m.fused_qkv_a_proj_with_mqa(hidden_states)[0]
    q,kv=latent.split([m.q_lora_rank,m.kv_lora_rank],dim=-1)
    q_lora=m.q_a_layernorm(q)
    kv=m.kv_a_layernorm(kv)
    q_full=m.q_b_proj(q_lora)[0].view(-1,m.num_local_heads,m.qk_nope_head_dim)
    q_latent=torch_npu.npu_transpose_batchmatmul(q_full,m.w_kc,perm_x1=(1,0,2),perm_x2=(0,1,2),perm_y=(1,0,2))
    q_empty=q_full.new_empty((q_full.shape[0],m.num_local_heads,0))
    k_nope=kv.reshape(kv.shape[0],1,m.kv_lora_rank)
    k_empty=kv.new_empty((kv.shape[0],1,0))
    if m.skip_topk and not (m.is_nextn and prev_topk_indices is None):
        topk=prev_topk_indices
    else:
        topk=m.indexer(hidden_states,q_lora,positions,forward_batch,m.layer_id)
    topk=maybe_capture_indexer_topk(m.layer_id,topk)
    return q_empty,k_empty,q_latent,k_nope,topk,forward_batch,zero_allocator,positions,False
