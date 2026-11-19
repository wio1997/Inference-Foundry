"""Eager-only NoPE MTP diagnostic opt-in. Not full-model MTP acceptance."""
import os
import torch
import torch_npu
_POOLS={}
OPTIN='GLM53_NOPE_EAGER_MTP_TOPK1_W4'

def _mtp_enabled():
 from sglang.srt.runtime_context import get_spec
 s=get_spec()
 if s.speculative_num_draft_tokens is None:return False
 assert os.environ.get(OPTIN)=='1','NoPE MTP requires explicit diagnostic opt-in'
 assert s.speculative_eagle_topk==1 and s.speculative_num_steps==3 and s.speculative_num_draft_tokens==4,'Only topk1/steps3/W4 is covered'
 return True

def _pool_contract(backend,child):
 from glm53_sglang_adapter.pool import NPUKPoolTokenToKVPool
 assert isinstance(child,NPUKPoolTokenToKVPool),'Actual registered target/draft pool class required'
 assert child.tail_extra_slots==4
 tk=child._compress_tail_k;tg=child._compress_tail_score
 assert tk and len(tk)==len(tg)==child.num_indexer_layers
 for k,g in zip(tk,tg):
  assert k.shape==g.shape and k.ndim==3 and k.shape[1:]==(8,128)
  assert k.dtype==g.dtype==torch.float32 and k.device==g.device
 registered_tail=set(child.get_compress_tail_buf_infos()[0])
 assert registered_tail=={x.data_ptr() for x in list(tk)+list(tg)}, 'C8 tails must use actual registered storage'
 registered_index=set(child.get_state_buf_infos()[0])
 assert {x.data_ptr() for x in child.index_k_buffer}<=registered_index, 'pooled index cache must be registered'
 arrays=list(child.k_buffer)+list(child.index_k_buffer)+list(tk)+list(tg)
 ptrs={x.untyped_storage().data_ptr() for x in arrays}
 role='draft' if backend.is_draft_worker else 'target';device=str(child.k_buffer[0].device)
 # Same-role multi-step backends may legitimately refer to the same draft pool.
 peers=_POOLS.setdefault(device,{})
 other=peers.get('target' if role=='draft' else 'draft')
 if other is not None:assert not ptrs.intersection(other['pointers']),'Target/draft model KV/index/tails must not alias'
 peers[role]={'child':child,'pointers':ptrs}
 backend.glm53_nope_mtp_device=device

def _initialize_eager_nope_consumer_g17(backend,*,max_query_tokens,query_heads):
 from sglang.srt.runtime_context import get_parallel
 from glm53_sglang_adapter.zero_rope_consumer import ZeroRopeSparseFA
 p=get_parallel();assert p.attn_cp_size==p.attn_dcp_size==1
 mtp=_mtp_enabled()
 if mtp:assert p.attn_dp_size==1,'Only eager single-DP lane is covered'
 else:assert not backend.is_draft_worker
 assert backend.use_dsa and backend.qk_rope_head_dim==0
 assert backend.kv_lora_rank==512 and backend.kv_cache_dtype==torch.bfloat16
 child=getattr(backend.token_to_kv_pool,'full_kv_pool',backend.token_to_kv_pool)
 assert child.index_kpool==4 and child.qk_rope_head_dim==0 and not child.use_fia_nz
 if mtp:_pool_contract(backend,child)
 key=child.k_buffer[0];assert key.ndim==4 and key.shape[-2:]==(1,512)
 backend.glm53_nope_sparse_consumer=ZeroRopeSparseFA(key,max_query_tokens=max_query_tokens,query_heads=query_heads,rope_dim=64,sparse_width=2112)
 backend.glm53_nope_mtp_optin=mtp
 return backend.glm53_nope_sparse_consumer.allocation_metadata()

def _begin_eager_nope_metadata_g17(backend,fb):
 if not getattr(backend,'glm53_nope_mtp_optin',False):return
 assert not backend.graph_mode,'No graph in this candidate'
 # Snapshot before native .int() += W could alias an int32 CPU mirror.
 source=fb.seq_lens_cpu if fb.seq_lens_cpu is not None else fb.seq_lens
 backend._glm53_nope_mtp_base_lengths=source.clone()

def finish_nope_mtp_metadata(backend,fb):
 if not getattr(backend,'glm53_nope_mtp_optin',False):return
 bs=fb.seq_lens.shape[0]
 if bs==0:return
 m=backend.forward_metadata;mode=fb.forward_mode
 base=backend._glm53_nope_mtp_base_lengths.to(device=backend.device,dtype=torch.int32)
 if mode.is_target_verify():
  assert not backend.is_draft_worker and fb.spec_info.topk==1 and fb.spec_info.draft_token_num==4
  assert fb.positions.numel()==bs*4,'Ragged/tree/padded batch unsupported by eager W4 gate'
  kv=base+4;width=4
 elif mode.is_draft_extend_v2():
  assert backend.is_draft_worker and fb.positions.numel()==bs*4
  kv=base;width=4 # FB length already includes the full verify window, not accepted A.
 elif mode.is_decode_or_idle() and fb.spec_info is not None:
  assert backend.is_draft_worker and fb.positions.numel()==bs
  kv=fb.positions.to(device=backend.device,dtype=torch.int32)+1;width=1
 else:return # ordinary target/draft prefill uses the unchanged native contract
 m.actual_seq_lengths_kv=kv
 m.actual_seq_lengths_q=torch.arange(width,(bs+1)*width,width,dtype=torch.int32,device=backend.device)
 # The SFA block-table must cover the same END length as the operator operand.
 # No CPU scalar is read: gather a full fixed-capacity page table in eager mode.
 m.block_tables=(backend.req_to_token[fb.req_pool_indices,::backend.page_size]//backend.page_size).to(torch.int32)

def _sparse_eager_consumer_g17(backend):
 if backend.use_dsa and backend.qk_rope_head_dim==0:
  assert not backend.graph_mode,'Graph capture/replay remains unsupported'
  if getattr(backend,'glm53_nope_mtp_optin',False):
   assert os.environ.get(OPTIN)=='1'
   peers=_POOLS[backend.glm53_nope_mtp_device]
   assert set(peers)=={'target','draft'},'Both actual target/draft registered pool instances must be checked before MTP forward'
  c=getattr(backend,'glm53_nope_sparse_consumer',None);assert c is not None
  from .decode_endpoint import call_active_sparse_consumer
  return lambda **kwargs: call_active_sparse_consumer(c,backend.forward_metadata,**kwargs)
 return torch_npu.npu_sparse_flash_attention


def initialize_nope_consumer(backend, *, max_query_tokens, query_heads):
    from .mtp_graph_bucket import enabled
    if enabled():
        from .nope_consumer_graph_g15 import initialize_nope_consumer as graph_init
        return graph_init(backend,max_query_tokens=max_query_tokens,query_heads=query_heads)
    return _initialize_eager_nope_consumer_g17(backend,max_query_tokens=max_query_tokens,query_heads=query_heads)

def begin_nope_mtp_metadata(backend,fb):
    from .mtp_graph_bucket import enabled
    if enabled():
        from .nope_consumer_graph_g15 import begin_nope_mtp_metadata as graph_begin
        return graph_begin(backend,fb)
    return _begin_eager_nope_metadata_g17(backend,fb)

def sparse_consumer_for(backend):
    if backend.graph_mode:
        from .nope_consumer_graph_g15 import sparse_consumer_for as graph_consumer
        return graph_consumer(backend)
    return _sparse_eager_consumer_g17(backend)
