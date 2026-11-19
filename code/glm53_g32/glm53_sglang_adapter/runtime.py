# SPDX-License-Identifier: Apache-2.0
"""Eager SGLang IndexerKPool forward using registered NPU child-pool tensors.

Graph/CP/PD acceptance are separate gates. No vLLM scheduler/model is used.
"""
import torch
import torch.nn.functional as F
from sglang.srt.model_executor.forward_context import get_attn_backend,get_token_to_kv_pool
from sglang.srt.runtime_context import get_parallel
from sglang.srt.model_executor.runner import get_is_capture_mode
from .bridge import KPoolStorage,run_kpool
from .pool import NPUKPoolTokenToKVPool
from .eager_metadata import EagerLaneContract,build_eager_inputs

MTP_EAGER_CONTRACT=None

def bind_mtp_eager_contract(*,topk,verify_width,disable_overlap_schedule,attn_dp_size):
    """Integrating owner must call on each worker after resolved args are known."""
    global MTP_EAGER_CONTRACT
    if not disable_overlap_schedule or attn_dp_size!=1:
        raise NotImplementedError("First MTP candidate requires synchronous scheduler and attention DP1")
    MTP_EAGER_CONTRACT=EagerLaneContract(topk,verify_width,True,False)



def forward_npu(self,x,q_lora,positions,forward_batch,layer_id,input_on_attn_tp_slices=False,dynamic_scale=None,*,return_indices=True,**kwargs):
    if kwargs:
        raise TypeError(f'Unmapped indexer call fields: {sorted(kwargs)}')
    if get_parallel().attn_cp_size != 1 or get_parallel().attn_dcp_size != 1:
        raise NotImplementedError('CP/DCP KPool metadata bridge is not validated')
    if dynamic_scale is not None:
        raise NotImplementedError('Indexer requires normalized unquantized q_lora; dynamic-scale caller must supply BF16 latent')
    if input_on_attn_tp_slices and x.shape[0] != positions.numel():
        from sglang.srt.layers.attention.dsa.dsa_npu_indexer import scattered_to_tp_attn_full
        x=scattered_to_tp_attn_full(x,forward_batch)
    if isinstance(q_lora,tuple) or (not self.skip_rope and self.rope_head_dim):
        raise NotImplementedError('Frozen GLM probe requires unquantized q_lora and indexer rope_dim=0')
    pool=get_token_to_kv_pool()
    local_layer=layer_id
    if hasattr(pool,'full_kv_pool'):
        local_layer=pool._transfer_full_attention_id(layer_id)
        pool=pool.full_kv_pool
    if not isinstance(pool,NPUKPoolTokenToKVPool):
        raise TypeError('Indexer requires registered NPUKPoolTokenToKVPool; private sidecar allocation is forbidden')
    mode=forward_batch.forward_mode
    if mode.is_idle():
        return torch.full((x.shape[0],self.index_topk+3),-1,device=x.device,dtype=torch.int32) if return_indices else None
    # Initial service hook is eager only: graph-bound metadata needs persistent buffers.
    if get_is_capture_mode():
        from .mtp_graph_bucket import enabled
        if not enabled():raise NotImplementedError('Graph metadata buffers require reviewed candidate integration')
    backend=get_attn_backend()
    backend=getattr(backend,'full_attn_backend',backend)
    if backend.graph_mode:
        from .mtp_graph_bucket import graph_forward
        return graph_forward(self,x,q_lora,positions,forward_batch,local_layer,pool,return_indices=return_indices)
    fm=backend.forward_metadata
    from .decode_endpoint import active_decode_view
    physical_tokens = positions.numel()
    forward_batch,positions,active_tokens=active_decode_view(backend,forward_batch,positions)
    if active_tokens != physical_tokens:
        if x.shape[0] != physical_tokens or q_lora.shape[0] != physical_tokens:
            raise RuntimeError('KPool inputs disagree with native TP padding witness')
        x=x[:active_tokens];q_lora=q_lora[:active_tokens]
    reqids=forward_batch.req_pool_indices.long()
    tk,tg=pool.get_compress_tail_buffers(local_layer)
    original_query_positions=positions
    positions,raw_slots,lengths,qlens=build_eager_inputs(
        forward_batch,positions,forward_batch.out_cache_loc,reqids,tk.shape[1],
        contract=MTP_EAGER_CONTRACT)
    # Only active request rows enter temporary combined donor layout. Registered
    # separate arrays remain the authoritative states read by PD/cache lifecycle.
    scratch=torch.stack((tk.index_select(0,reqids),tg.index_select(0,reqids)),dim=1)
    storage=KPoolStorage(pool.get_index_k_buffer(local_layer),scratch,pool.page_size)
    # Preserve donor's FP32 index-key normalization/gate arithmetic; no full-model
    # FP32 path is introduced. Query and stored completed pools remain BF16.
    if not hasattr(self,'_adapter_wk_f32'):
        self._adapter_wk_f32=self.wk.weight.detach().float()
        self._adapter_gate_f32=self.index_kpool_compress_gate.detach().float()
    projected=F.linear(x.float(),self._adapter_wk_f32)
    norm=self.k_norm
    key=F.layer_norm(projected,(128,),norm.weight.float(),norm.bias.float() if norm.bias is not None else None,
                     getattr(norm,'eps',getattr(norm,'variance_epsilon',1e-6)))
    gate=F.linear(x.float(),self._adapter_gate_f32)
    query=self.wq_b(q_lora)[0].view(-1,self.n_heads,128) if return_indices else None
    weights=(self.weights_proj(x.float())[0].to(query.dtype)*(self.n_heads**-0.5*self.softmax_scale)) if return_indices else None
    result=run_kpool(storage,key=key,query=query,head_weights=weights,gate_score=gate,
                     ape=self.index_kpool_compress_ape,positions=positions,raw_slots=raw_slots,
                     raw_seq_lens=lengths,query_lens=qlens,raw_page_table=fm.block_tables,
                     req_pool_indices=torch.arange(reqids.numel(),device=reqids.device),topk=self.index_topk,
                     compute_topk=return_indices,
                     speculative_lookahead=(MTP_EAGER_CONTRACT.verify_width if MTP_EAGER_CONTRACT is not None else 0),
                     query_positions=(original_query_positions if mode.is_draft_extend_v2() else None))
    tk.index_copy_(0,reqids,scratch[:,0]);tg.index_copy_(0,reqids,scratch[:,1])
    if result is None:return None
    result=result[:,0]
    if active_tokens != physical_tokens:
        padded=torch.full((physical_tokens,*result.shape[1:]),-1,device=result.device,dtype=result.dtype)
        padded[:active_tokens].copy_(result)
        return padded
    return result


def bind_mtp_ordered_contract(*,topk,verify_width,attn_dp_size):
    from .mtp_graph_bucket import enabled
    if not enabled() or topk!=1 or verify_width!=4 or attn_dp_size!=1:
        raise RuntimeError('Ordered candidate requires native one-stream W4/DP1 integration')
    global MTP_EAGER_CONTRACT
    MTP_EAGER_CONTRACT=EagerLaneContract(topk,verify_width,True,False)

