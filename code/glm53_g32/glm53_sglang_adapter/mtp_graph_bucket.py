"""Opt-in W4 persistent graph metadata. Numerical/NPUGraph acceptance pending."""
from types import SimpleNamespace
import os
import torch
OPTIN='GLM53_MTP_GRAPH_ASYNC_CANDIDATE'
_STREAMS={}
def enabled():return os.environ.get(OPTIN)=='1'

def register_native_streams(forward_stream,*plan_streams):
 if not enabled():return
 if forward_stream is None:raise RuntimeError('Actual native forward stream required')
 if forward_stream == torch.npu.default_stream(forward_stream.device):raise RuntimeError('Native MTP forward stream must be explicit non-default')
 device=str(forward_stream.device)
 streams=tuple(x for x in (forward_stream,)+plan_streams if x is not None)
 if device in _STREAMS and _STREAMS[device]!=streams:raise RuntimeError('Native ordered stream ownership changed')
 _STREAMS[device]=streams
 # Actual native source pins plan.wait(forward) and forward.wait(plan) around prep.
 import json
 from pathlib import Path
 root=Path(os.environ['GLM53_LOAD_LEDGER_DIR'])
 path=root/('mtp_ordered_streams.pid%d.json'%os.getpid())
 path.write_text(json.dumps({'device':device,'forward_stream':str(forward_stream),'forward_stream_verified_non_default':True,'joined_native_plan_streams':[str(x) for x in plan_streams if x is not None],'scope':'actual stream identities; native wait-chain source bound; runtime NPU correctness unverified'},indent=2)+'\n')

class MTPBucket:
 def __init__(self,backend,fb,width):
  pool=getattr(backend.token_to_kv_pool,'full_kv_pool',backend.token_to_kv_pool)
  self.pool=pool
  self.owner_ptrs=tuple(t.data_ptr() for t in list(pool.k_buffer)+list(pool.v_buffer)+list(pool.index_k_buffer)+list(pool._compress_tail_k)+list(pool._compress_tail_score))
  self.b=fb.batch_size;self.w=width;self.n=self.b*width;self.c=8
  self.device=pool._compress_tail_k[0].device;self.dummy=backend.req_to_token.shape[0]
  if any(t.shape[0]!=self.dummy+1 or t.shape[1:]!=(8,128) for t in pool._compress_tail_k):raise RuntimeError('Native C8 plus private dummy row required')
  self.active=torch.zeros(self.b,dtype=torch.bool,device=self.device)
  self.reqids=torch.empty(self.b,dtype=torch.int64,device=self.device)
  self.accepted=torch.empty(self.b,dtype=torch.int32,device=self.device)
  self.rawlen=torch.empty(self.b,dtype=torch.int32,device=self.device)
  self.poollen=torch.empty_like(self.rawlen)
  self.positions=torch.empty(self.n,dtype=torch.int64,device=self.device)
  self.query_positions=torch.empty_like(self.positions)
  self.slots=torch.empty_like(self.positions);self.completed=torch.empty_like(self.positions);self.tail_slots=torch.empty_like(self.positions)
  self.token_active=torch.empty(self.n,dtype=torch.bool,device=self.device)
  self.write_active=torch.empty_like(self.token_active)
  self.step=torch.arange(width,dtype=torch.int32,device=self.device)[None,:].expand(self.b,width)
  self.localrows=torch.arange(self.b,dtype=torch.int64,device=self.device)
  self.rowbase=self.localrows[:,None].expand(self.b,width).reshape(-1)*self.c
  self.qends=torch.arange(width,(self.b+1)*width,width,dtype=torch.int32,device=self.device)
  self.tail_planes=torch.empty((2,self.b,8,128),dtype=torch.float32,device=self.device)
  self.tail=self.tail_planes.permute(1,0,2,3)
  self.output=torch.empty((self.n,2051),dtype=torch.int32,device=self.device)
  self.meta=SimpleNamespace(block_table=backend.forward_metadata.block_tables,slot_mapping=self.completed,seq_lens=self.poollen,raw_seq_lens=self.rawlen,cum_query_lens=self.qends)
  self.tail_meta=SimpleNamespace(slot_mapping=self.tail_slots,block_table=self.localrows[:,None],block_size=8)
  self.query_meta=SimpleNamespace(positions=self.query_positions,cum_query_lens=self.qends,num_actual_tokens=self.n)
  self.extend=fb.forward_mode.is_draft_extend_v2()
 def stage(self,fb,in_capture):
  pool=self.pool
  current=tuple(t.data_ptr() for t in list(pool.k_buffer)+list(pool.v_buffer)+list(pool.index_k_buffer)+list(pool._compress_tail_k)+list(pool._compress_tail_score))
  if current!=self.owner_ptrs:raise RuntimeError('Registered pool storage changed; graph recapture required')
  actual_mode=getattr(fb,'actual_forward_mode',fb.forward_mode)
  inactive=in_capture or bool(getattr(actual_mode,'is_idle',lambda:False)())
  raw=0 if inactive else self.b-int(getattr(fb,'num_padding',0))
  if not 0<=raw<=self.b:raise RuntimeError('Invalid actual graph live request count')
  if not in_capture:
   stream=torch.npu.current_stream(self.device);allowed=_STREAMS.get(str(self.device),())
   if not allowed or not any(stream==owner for owner in allowed):raise RuntimeError('Graph staging must use registered native forward/explicitly joined plan stream')
  self.publish_inputs(fb,raw)
 def publish_inputs(self,fb,raw):
  # Host-known live count; called by actual stage after one-stream validation.
  if not 0<=raw<=self.b:raise RuntimeError('Bad live count')
  self.active.fill_(False);self.active[:raw].fill_(True)
  self.reqids.copy_(fb.req_pool_indices[:self.b]);self.reqids.masked_fill_(~self.active,self.dummy)
  if self.extend:
   accepted=fb.spec_info.num_accept_tokens
   if accepted is None or accepted.numel()<self.b:raise RuntimeError('Native accepted including-bonus buffer required')
   self.accepted.copy_(accepted[:self.b])
  else:self.accepted.fill_(self.w)
 def refresh_endpoint(self,positions):
  # Captured from actual per-layer forward positions. Draft loop mutates them.
  if positions.ndim!=1 or positions.numel()!=self.n:raise RuntimeError('Fixed graph positions shape changed')
  self.query_positions.copy_(positions)
  self.rawlen.copy_(positions.view(self.b,self.w)[:,-1]);self.rawlen.add_(1)
  self.rawlen.masked_fill_(~self.active,0)
  torch.div(self.rawlen,4,rounding_mode='floor',out=self.poollen)
 def refresh_write(self,positions,slots):
  self.refresh_endpoint(positions)
  self.positions.copy_(positions);self.slots.copy_(slots)
  self.token_active.copy_(self.active[:,None].expand(self.b,self.w).reshape(-1))
  self.write_active.copy_(self.token_active)
  if self.extend:self.write_active.logical_and_((self.step<self.accepted[:,None]).reshape(-1))
  self.positions.masked_fill_(~self.write_active,-1);self.slots.masked_fill_(~self.write_active,-1)
  self.completed.copy_(torch.div(self.slots,4,rounding_mode='floor'))
  self.completed.masked_fill_(~(self.write_active & ((self.positions+1).remainder(4)==0)),-1)
  self.tail_slots.copy_(self.rowbase+self.positions.remainder(8));self.tail_slots.masked_fill_(~self.write_active,-1)
 def gather(self,k,g):
  torch.index_select(k,0,self.reqids,out=self.tail[:,0]);torch.index_select(g,0,self.reqids,out=self.tail[:,1])
 def scatter(self,k,g):
  k.index_copy_(0,self.reqids,self.tail[:,0]);g.index_copy_(0,self.reqids,self.tail[:,1])

def stage_backend(backend,fb,in_capture=False):
 if not (backend.use_dsa and backend.qk_rope_head_dim==0):return
 if not enabled():raise RuntimeError('MTP graph requires explicit isolated-candidate opt-in')
 mode=fb.forward_mode
 if mode.is_target_verify() or mode.is_draft_extend_v2():width=4
 elif mode.is_decode_or_idle():width=1
 else:raise RuntimeError('Captured prefill/ragged graph not supported by W4 candidate')
 if getattr(fb,'ragged_verify_layout',None) is not None:raise RuntimeError('Ragged MTP graph unsupported')
 b=fb.batch_size;key=(b,width);buckets=getattr(backend,'glm53_mtp_graph_buckets',None)
 if buckets is None:buckets=backend.glm53_mtp_graph_buckets={}
 if key not in buckets:
  if not in_capture:raise RuntimeError('Bucket must be allocated before native warmup/capture')
  buckets[key]=MTPBucket(backend,fb,width)
 bucket=buckets[key]
 if bucket.meta.block_table.data_ptr()!=backend.forward_metadata.block_tables.data_ptr():raise RuntimeError('Graph block-table address changed; recapture required')
 bucket.stage(fb,in_capture);backend.glm53_mtp_graph_bucket=bucket
 backend.forward_metadata.actual_seq_lengths_q=bucket.qends
 backend.forward_metadata.actual_seq_lengths_kv=bucket.rawlen
 consumer=backend.glm53_nope_sparse_consumer;consumer.select_lane(0)
 backend.forward_metadata.glm53_nope_req_pool_indices=consumer.stage_request_ids(bucket.reqids)

def graph_forward(indexer,x,q_lora,positions,fb,layer_id,pool,*,return_indices=True):
 import torch.nn.functional as F
 from sglang.srt.model_executor.forward_context import get_attn_backend
 from .vendor.sparse_attn_indexer_kpool import SparseAttnIndexerKpool
 backend=get_attn_backend();backend=getattr(backend,'full_attn_backend',backend)
 if not enabled() or not backend.graph_mode:raise RuntimeError('Actual graph backend/opt-in required')
 bucket=backend.glm53_mtp_graph_bucket
 if positions.numel()!=bucket.n or x.shape[0]!=bucket.n:raise RuntimeError('Captured token shape changed')
 bucket.refresh_write(positions,fb.out_cache_loc)
 tk,tg=pool.get_compress_tail_buffers(layer_id);bucket.gather(tk,tg)
 if not hasattr(indexer,'_adapter_wk_f32'):
  indexer._adapter_wk_f32=indexer.wk.weight.detach().float();indexer._adapter_gate_f32=indexer.index_kpool_compress_gate.detach().float()
 projected=F.linear(x.float(),indexer._adapter_wk_f32);norm=indexer.k_norm
 key=F.layer_norm(projected,(128,),norm.weight.float(),norm.bias.float() if norm.bias is not None else None,getattr(norm,'eps',getattr(norm,'variance_epsilon',1e-6)))
 gate=F.linear(x.float(),indexer._adapter_gate_f32)
 query=indexer.wq_b(q_lora)[0].view(-1,indexer.n_heads,128) if return_indices else None
 weights=indexer.weights_proj(x.float())[0].to(query.dtype)*(indexer.n_heads**-0.5*indexer.softmax_scale) if return_indices else None
 result=SparseAttnIndexerKpool(indexer.index_topk,128)(key.float(),query,weights,bucket.positions,pool.get_index_k_buffer(layer_id),bucket.tail,bucket.meta,bucket.tail_meta,gate_score=gate.float(),compress_ape=indexer.index_kpool_compress_ape.float(),index_kpool=4,max_pool_seq_len=bucket.meta.block_table.shape[1]*pool.get_index_k_buffer(layer_id).shape[1],compute_topk=return_indices,output_buffer=bucket.output,allow_cache_packing=False,query_metadata=bucket.query_meta)
 bucket.scatter(tk,tg)
 return result[:,0] if result is not None else None
