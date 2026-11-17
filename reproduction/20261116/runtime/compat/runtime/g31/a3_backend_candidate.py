# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the vLLM project; Copyright SGLang Team.
"""Isolated complete-shape V4.1/A3 backend candidate, not installed by import.

Current image ABI must first be bound explicitly and tested. Metadata kernels
run outside capture and copy into fixed-address buffers; all source state,
indexer/candidate and sparse-attention operations remain in the forward graph.
"""
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable

import torch

from a3_source_components import (A3SourceMetadata,native_pair_source_mixin,
                                  project_a3_index_query_weights,run_a3_qli)
from quantize_indexer_query import quantize_indexer_query
from prepare_indexer_indices import prepare_indexer_indices


@dataclass(frozen=True)
class A3Bindings:
    # Exact donor call contracts; adapt spelling only after target ABI inventory.
    qli: Callable
    qli_metadata: Callable
    sparse_metadata: Callable
    sparse: Callable
    # Binder must preserve the exact candidate-v3 contract, not a V2 fallback.


def make_backend_class(bindings:A3Bindings):
    from sglang.srt.hardware_backend.npu.attention.ascend_dsv4_backend import DeepseekV4AscendAttnBackend
    from sglang.srt.layers.attention.dsv4.dsv41_sparse import token_req_indices
    PairMixin=native_pair_source_mixin()

    class A3V41AttentionBackend(PairMixin,DeepseekV4AscendAttnBackend):
        def __init__(self,model_runner,speculative_step_id=0):
            super().__init__(model_runner,speculative_step_id=speculative_step_id)
            cfg=model_runner.model_config.hf_config
            self._v41_cfg=cfg
            self._v41_enabled=not getattr(model_runner,'is_draft_worker',False)
            self._v41_ratios=tuple(cfg.compress_ratios)
            if not set(self._v41_ratios)<={0,1,2}:raise ValueError('A3 V41 candidate accepts0/1/2 only')
            # Preserve existing SWA/DSpark draft implementation, without making
            # its C4/C128 compressor metadata consume C2's generic pair state.
            self._dsv4_compress_ratios=[]
            self._dsv4_unique_compress_ratios=[]
            self._dsv4_has_c4=self._dsv4_has_c128=False
            self._dsv4_state_pools_by_ratio={}
            self._v41_buffers={}
            self._v41_eager_buffer=None
            self._v41_index_sources=tuple(cfg.index_source_layer_ids)
            self._v41_kv_sources=tuple(cfg.kv_source_layer_ids)
            self._v41_index_source_of={i:max((j for j in self._v41_index_sources if j<=i and self._v41_ratios[j]==self._v41_ratios[i]),default=-1) for i in range(len(self._v41_ratios))}

        @property
        def requires_eager_metadata(self):
            # CPU shape/length scheduling runs on every replay. The model
            # forward graph stays enabled; only metadata glue is ineligible.
            return True

        def _v41_attach_metadata(self,fb):
            if not self._v41_enabled:return
            fm=self.forward_metadata
            if fb.forward_mode.is_idle():return
            T=int(fb.positions.shape[0]);bs=int(fb.batch_size)
            pages=int(fm.block_tables.shape[1])
            # Graph workspaces belong only to actual capture geometry. Eager
            # context/page growth must not retain a new ~192MiB fat workspace
            # for each prefill chunk. Keep exactly one eager workspace.
            key=(str(fb.forward_mode),T,bs)
            buf=self._v41_buffers.get(key) if self.graph_mode else self._v41_eager_buffer
            if buf is not None and not self.graph_mode:
                if buf.seq.numel()!=bs or buf.candidates.shape[0]!=T:
                    buf=None
            if buf is None:
                dev=fb.positions.device
                buf=SimpleNamespace(
                    full_table=torch.empty((bs,pages),dtype=torch.int32,device=dev),
                    cu=torch.empty((bs+1,),dtype=torch.int32,device=dev),
                    seq=torch.empty((bs,),dtype=torch.int32,device=dev),
                    candidates=torch.full((T,1,int(self._v41_cfg.candidate_topk_blocks)),-1,dtype=torch.int32,device=dev),
                    indices={i:torch.full((T,int(self._v41_cfg.index_topk)),-1,dtype=torch.int32,device=dev) for i in self._v41_index_sources},
                    lengths={i:torch.zeros((T,1),dtype=torch.int32,device=dev) for i in self._v41_index_sources},
                    by_ratio={})
                if self.graph_mode:self._v41_buffers[key]=buf
                else:self._v41_eager_buffer=buf
            elif tuple(buf.full_table.shape)!=(bs,pages):
                if self.graph_mode:
                    raise ValueError('Native graph page-table shape changed for a captured bucket')
                # Slim page table alone grows/shrinks; reuse the fat token,
                # candidate and per-indexer arrays for this eager geometry.
                buf.full_table=torch.empty((bs,pages),dtype=torch.int32,device=fb.positions.device)
            buf.full_table.copy_(fm.block_tables[:bs])
            buf.cu.copy_(fm.actual_seq_lengths_q_pa[:bs+1])
            # Existing NPU replay code owns DSpark live/final-prefix semantics.
            # Reuse its final CPU mirror in graph paths, including capture,
            # rather than adding speculative width a second time.
            if self.graph_mode and fb.forward_mode.is_target_verify():
                ctx=self._build_dsv4_graph_replay_ctx(fb)
                final_cpu=ctx.final_seq_lens_cpu[:bs].to(device='cpu',dtype=torch.int32)
                buf.seq.copy_(final_cpu.to(buf.seq.device))
            else:
                buf.seq.copy_(fm.actual_seq_lengths_kv[:bs])
                final_cpu=getattr(fm,'seq_lens_cpu_int',None)
                if final_cpu is None:final_cpu=buf.seq.cpu()
                final_cpu=final_cpu[:bs].to(device='cpu',dtype=torch.int32)
            max_seq=int(final_cpu.max().item()) if bs else 0
            cu_cpu=buf.cu.cpu()
            max_q=int((cu_cpu[1:]-cu_cpu[:-1]).max().item()) if bs else 0
            if self.graph_mode:
                max_q=self._dsv4_graph_tokens_per_req if fb.forward_mode.is_target_verify() else 1
            buf.max_q=max_q
            for ratio in (1,2):
                seq=torch.div(buf.seq,ratio,rounding_mode='floor')
                residual=(buf.seq%ratio) if ratio==2 else None
                # Scheduling capacity is positive even before C2 completes a
                # group, so capture records the same path as later replays.
                # Exact device seq/residual above still determine visibility.
                cmp_capacity=max(1,max_seq//ratio)
                # These call contracts are copied from actual A3 metadata
                # builders. Returned scheduling data is copied into persistent
                # storage BEFORE replay; no stale capture-time lengths.
                qli=bindings.qli_metadata(int(self._v41_cfg.index_n_heads),1,
                    int(self._v41_cfg.index_head_dim),int(self._v41_cfg.index_topk),2,
                    cu_seqlens_q=buf.cu,seqused_k=seq,cmp_residual_k=residual,
                    batch_size=bs,max_seqlen_q=max_q,max_seqlen_k=cmp_capacity,
                    layout_q='TND',layout_k='PA_BBND',mask_mode=3,cmp_ratio=ratio)
                smla=bindings.sparse_metadata(self._dsv4_q_head_num,1,self._dsv4_head_dim,
                    cu_seqlens_q=buf.cu,seqused_ori_kv=buf.seq,seqused_cmp_kv=seq,
                    cmp_residual_kv=residual,batch_size=bs,max_seqlen_q=max_q,
                    max_seqlen_ori_kv=max_seq,max_seqlen_cmp_kv=cmp_capacity,
                    ori_topk=(fm.ori_sparse_indices.shape[-1] if fm.ori_sparse_indices is not None else 0),
                    ori_topk_length=fm.ori_topk_length,cmp_topk=int(self._v41_cfg.index_topk),
                    cmp_ratio=ratio,ori_mask_mode=fm.ori_mask_mode,cmp_mask_mode=3,
                    ori_win_left=fm.ori_win_left,ori_win_right=fm.ori_win_right,
                    layout_q='TND',layout_kv='PA_BBND',has_ori_kv=True,has_cmp_kv=True)
                old=buf.by_ratio.get(ratio)
                if old is None:
                    old=SimpleNamespace(seq=torch.empty_like(seq),residual=None if residual is None else torch.empty_like(residual),
                                        qli=torch.empty_like(qli),smla=torch.empty_like(smla))
                    buf.by_ratio[ratio]=old
                if old.qli.shape!=qli.shape or old.smla.shape!=smla.shape:
                    raise ValueError('Kernel metadata shape changed across graph bucket; ABI integration required')
                old.seq.copy_(seq)
                if residual is not None:old.residual.copy_(residual)
                old.qli.copy_(qli);old.smla.copy_(smla)
            fm.v41=buf

        def init_forward_metadata(self,forward_batch):
            super().init_forward_metadata(forward_batch)
            self._v41_attach_metadata(forward_batch)

        def init_forward_metadata_out_graph(self,forward_batch,in_capture=False):
            super().init_forward_metadata_out_graph(forward_batch,in_capture=in_capture)
            self._v41_attach_metadata(forward_batch)

        def forward_low_ratio_sources(self,*,layer,x,q_lora,positions,forward_batch,
                                       run_compressor=True,run_indexer=True):
            if forward_batch.forward_mode.is_idle():return
            fm=self.forward_metadata;buf=fm.v41
            raw=forward_batch.out_cache_loc[:positions.shape[0]].to(torch.int64)
            c2=torch.where((raw>0)&((positions+1)%2==0),raw//2,torch.full_like(raw,-1))
            fm.core_metadata=SimpleNamespace(raw_out_loc=raw,c1_out_loc=raw,c2_out_loc=c2)
            req=token_req_indices(forward_batch,num_tokens=positions.shape[0])
            if run_compressor and layer.compressor is not None:
                if forward_batch.encoder_swa_replay:raise NotImplementedError('encoder SWA replay requires a separately validated source replay contract')
                self._low_ratio_compress_torch(layer,x,req,positions)
            if run_indexer and layer.indexer is not None:
                idx=layer.indexer;ratio=layer.compress_ratio;meta=buf.by_ratio[ratio]
                q,w=project_a3_index_query_weights(idx,layer,x,q_lora,positions)
                source=A3SourceMetadata(buf.cu,meta.seq,meta.residual,buf.full_table,meta.qli,
                                       int(forward_batch.spec_info.draft_token_num) if forward_batch.forward_mode.is_target_verify() else 1)
                # max query length is a scheduler hint, not visibility. Store
                # exact outside-graph value when attaching metadata.
                source.max_query_len=buf.max_q
                selected,candidates=run_a3_qli(q,w,positions,self.token_to_kv_pool.get_index_kv_npu(layer.layer_id),source,
                    compress_ratio=ratio,index_topk=int(self._v41_cfg.index_topk),
                    is_candidate_source=idx.is_candidate_source,uses_candidate_filter=idx.uses_candidates,
                    candidate_topk_blocks=int(self._v41_cfg.candidate_topk_blocks),candidate_block_size=int(self._v41_cfg.candidate_block_size),
                    candidates=buf.candidates,quantize_query=quantize_indexer_query,finalize_indices=prepare_indexer_indices,
                    indices_output=buf.indices[layer.layer_id],lengths_output=buf.lengths[layer.layer_id],qli_op=bindings.qli)
                if idx.is_candidate_source:buf.candidates.copy_(candidates)

        def forward(self,q,k,v,layer,forward_batch,*,compress_ratio=0,attn_sink=None,save_kv_cache=True):
            if compress_ratio==0:
                return super().forward(q,k,v,layer,forward_batch,compress_ratio=0,attn_sink=attn_sink,save_kv_cache=save_kv_cache)
            if compress_ratio not in (1,2):raise ValueError('Unsupported A3 V41 ratio')
            if forward_batch.forward_mode.is_idle():return torch.zeros_like(q)
            if save_kv_cache:self.store_cache(layer_id=layer.layer_id,swa_k=k,forward_batch=forward_batch)
            fm=self.forward_metadata;buf=fm.v41;meta=buf.by_ratio[compress_ratio]
            index_source=self._v41_index_source_of[layer.layer_id]
            if index_source<0:raise ValueError('No preceding same-ratio index source')
            selected=buf.indices[index_source].view(-1,1,int(self._v41_cfg.index_topk))
            out,_=bindings.sparse(q,ori_kv=self.token_to_kv_pool.get_swa_buffer(layer.layer_id),
                cmp_kv=self.token_to_kv_pool.get_extra_key_buffer(layer.layer_id),
                ori_sparse_indices=fm.ori_sparse_indices,ori_topk_length=fm.ori_topk_length,
                cmp_sparse_indices=selected,ori_block_table=fm.swa_page_table,cmp_block_table=buf.full_table,
                cu_seqlens_q=buf.cu,seqused_ori_kv=buf.seq,seqused_cmp_kv=meta.seq,
                cmp_residual_kv=meta.residual,sinks=attn_sink,metadata=meta.smla,
                softmax_scale=layer.scaling,cmp_ratio=compress_ratio,
                ori_mask_mode=fm.ori_mask_mode,cmp_mask_mode=3,ori_win_left=fm.ori_win_left,ori_win_right=fm.ori_win_right,
                layout_q='TND',layout_kv='PA_BBND',topk_value_mode=1,return_softmax_lse=False)
            return out
    return A3V41AttentionBackend
