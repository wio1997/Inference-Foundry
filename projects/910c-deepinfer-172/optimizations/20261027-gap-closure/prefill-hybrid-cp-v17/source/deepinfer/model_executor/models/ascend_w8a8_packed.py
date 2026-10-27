# SPDX-License-Identifier: Apache-2.0
"""Native paged W8A8 text model with packed prefill/decode on 910C.

Uses the documented V4.1 source-sharing topology and the already validated
CANN operator ABI. The original scalar model remains a separate entry point.
No vLLM engine, scheduler, forward context or Python model is imported.
"""
from dataclasses import dataclass
import copy
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from deepinfer.npu_ops import mhc910c as mhc
from deepinfer.npu_ops.dsa910c import DSABatch, sparse_mla
from deepinfer.npu_ops.sp910c import SequenceShard
from deepinfer.npu_ops.cp910c import QuerySlice, restore_tp_heads
from .ascend_w8a8_layer import AscendW8A8Layer
from .w8a8_compressor910c import compress_batch
from .w8a8_engram910c import AscendW8A8Engram, CpuEngramHistory, load_rotation
from .w8a8_indexer910c import AscendW8A8Indexer
from .w8a8_math910c import rms_norm_mixed, rotate_positions
from .w8a8_paged_cache910c import W8A8PagedCache


@dataclass
class PackedForward:
    batch: DSABatch
    slots: list[int]
    positions: torch.Tensor
    swa_plan: object
    swa_table: torch.Tensor
    selected: torch.Tensor | None = None
    index_source: int | None = None
    candidates: torch.Tensor | None = None
    mc2_global_bs: int = 4096
    decode_frame: object | None = None
    source_metadata: dict | None = None
    sequence_parallel: bool = False
    context_query: object | None = None
    query_batch: object | None = None
    query_slots: list[int] | None = None
    query_swa_table: object | None = None


class AscendW8A8PackedLayer(AscendW8A8Layer):
    def __init__(self, reader, device, *, layer_id, group=None, tp_group=None):
        super().__init__(reader, device, layer_id=layer_id, group=group)
        self.tp_group = tp_group
        self.sp = SequenceShard(tp_group)
        self.eager_attention_calls = self.cann_attention_calls = 0
        # Reuse the same shifted-mHC kernels as the deployed V4.1 model.
        self.fused_mhc = (hasattr(torch.ops._C_ascend, 'npu_hc_pre_v3')
                          and hasattr(torch.ops._C_ascend, 'npu_hc_post'))
        self.fused_router_norm = hasattr(torch.ops._C_ascend, 'npu_rms_norm_cast')
        self.tp_size = dist.get_world_size(tp_group) if tp_group is not None else 1
        self.local_heads = self.config['num_attention_heads']//self.tp_size
        self.local_o_groups = self.config['o_groups']//self.tp_size
        if self.config['num_attention_heads'] % self.tp_size or self.config['o_groups'] % self.tp_size:
            raise ValueError('attention heads/output groups must divide TP size')
        self.context_parallel = (os.environ.get('DEEPINFER_910C_CONTEXT_PARALLEL') == '1' and self.tp_size > 1)
        self.context_parallel_calls = self.context_parallel_query_tokens = 0
        if self.context_parallel:
            self.cp_wq_b = reader.load_linear(self.prefix+'.attn.wq_b', device)
            self.register_buffer('cp_attn_sink', self.attn_attn_sink.clone())
        if self.tp_size > 1:
            from deepinfer.hugging_face.modelslim910c import FloatLinear
            from deepinfer.npu_ops.w8a8 import AscendW8A8Linear
            rank = dist.get_rank(tp_group)
            first = rank*self.local_heads
            self.projections['wq_b'] = reader.load_linear(self.prefix+'.attn.wq_b', device,
                output_slice=slice(first*self.config['head_dim'], (first+self.local_heads)*self.config['head_dim']))
            self.attn_attn_sink = self.attn_attn_sink[first:first+self.local_heads].clone()
            group_start = rank*self.local_o_groups
            self.wv = self.wv[group_start:group_start+self.local_o_groups].clone()
            width = self.local_o_groups*self.config['o_lora_rank']
            weight, scale = reader.linear_payload(self.prefix+'.attn.wo_b')
            weight = weight[:, rank*width:(rank+1)*width].contiguous().to(device)
            self.projections['wo_b'] = FloatLinear(weight) if scale is None else AscendW8A8Linear(weight, scale.to(device))
        self.fused_o_proj = os.environ.get('DEEPINFER_910C_FUSED_O_PROJ') == '1'
        self.fused_o_proj_calls = 0
        if self.fused_o_proj:
            if self.wv.dtype != torch.bfloat16:
                raise ValueError('fused O projection requires checkpoint BF16 FLOAT weights')
            # Pinned V4.1 BF16 ABI consumes [groups, input, output] in ND.
            self.register_buffer('wv_bmm', self.wv.transpose(1, 2).contiguous())
        self.index_source = max((s for s in self.config['index_source_layer_ids'] if s <= layer_id), default=None) if self.ratio else None
        if layer_id in self.config['index_source_layer_ids']:
            self.indexer = AscendW8A8Indexer(reader, device, layer_id=layer_id)
        self.mc2=None
        self.fused_mc2_prefill=os.environ.get('DEEPINFER_910C_FUSED_MC2_PREFILL')=='1'
        if os.environ.get('DEEPINFER_910C_MC2')=='1':
            from deepinfer.npu_ops.mc2_910c import NativeMC2
            self.mc2=NativeMC2(group)

    def load_local_experts(self):
        super().load_local_experts()
        from deepinfer.npu_ops.w8a8 import AscendW8A8ExpertBank
        self.local_expert_count=len(self.expert_cache)
        self.expert_bank=AscendW8A8ExpertBank(self.expert_cache.values())
        self.expert_cache.clear()
        if self.fused_mc2_prefill:
            if self.mc2 is None:
                raise ValueError('fused MC2 prefill requires native MC2 communicator')
            self.expert_bank.prepare_fused_mc2()

    def packed_ffn(self,x,router_x,global_bs,*,fused_mc2=False,already_sharded=False):
        from deepinfer.npu_ops.moe910c import dispatch_combine,route
        # Attention returns the same token batch on each TP rank. Dispatch
        # each token once to global EP16, then restore the TP-replicated batch.
        count=x.shape[0]
        width=count if already_sharded else (count+self.tp_size-1)//self.tp_size
        rank=dist.get_rank(self.tp_group) if self.tp_size>1 else 0
        first=0 if already_sharded else rank*width
        local=x[first:first+width]
        routing=router_x[first:first+width]
        valid=local.shape[0]
        if not valid:
            local=x[:1];routing=router_x[:1]
        logits=F.linear(routing,self.ffn_gate_weight)
        ids,weights=route(logits,self.ffn_gate_bias,self.config['num_experts_per_tok'],1.)
        if not valid and self.mc2 is None:
            ids=ids.fill_(-1);weights=weights.zero_()
        elif not valid:
            weights=weights.zero_()
        if self.mc2 is not None:
            routed=self.mc2(local,ids,weights,self.expert_bank,num_experts=self.config['n_routed_experts'],global_bs=global_bs,
                            fused=fused_mc2)
        else:
            routed=dispatch_combine(local,ids,weights,self.config['n_routed_experts'],None,self.group,
                weight_on_output=True,local_batch=self.expert_bank,trusted_indices=True)
        shared=self.shared_expert(local)
        output=(shared.float()+routed.float()*self.config['routed_scaling_factor']).bfloat16()
        output=output[:valid]
        if self.tp_size>1 and not already_sharded:
            padded=x.new_zeros((width,x.shape[1]))
            padded[:output.shape[0]].copy_(output)
            gathered=x.new_empty((width*self.tp_size,x.shape[1]))
            dist.all_gather_into_tensor(gathered,padded.contiguous(),group=self.tp_group)
            return gathered[:count]
        return output

    def packed_attention(self, x, cache, pending, state, *, reduce_results=True):
        c = self.config
        # The pinned pure-SWA kernel changes BF16 rounding with head grouping.
        # Preserve the proven TP head layout for SWA; compressed MLA uses CP.
        cp = state.context_query if self.ratio else None
        batch = state.query_batch if cp is not None else state.batch
        query_x = x[cp.start:cp.end] if cp is not None else x
        positions = state.positions[cp.start:cp.end] if cp is not None else state.positions
        slots = state.query_slots if cp is not None else state.slots
        swa_table = state.query_swa_table if cp is not None else state.swa_table
        heads = c['num_attention_heads'] if cp is not None else self.local_heads
        sink = self.cp_attn_sink if cp is not None else self.attn_attn_sink
        if query_x.shape[0]:
            qr = rms_norm_mixed(self.projections['wq_a'](query_x), self.attn_q_norm_weight, c['rms_norm_eps'])[0]
            q = (self.cp_wq_b if cp is not None else self.projections['wq_b'])(qr).unflatten(-1, (heads, c['head_dim']))
            q = rotate_positions(q, positions, self.rope_freq)
        kv = rms_norm_mixed(self.projections['wkv'](x), self.attn_kv_norm_weight, c['rms_norm_eps'])[0]
        kv = rotate_positions(kv, state.positions, self.rope_freq)
        cache.swa[self.layer_id].write(kv[:, None], state.swa_plan)
        if self.layer_id in c['kv_source_layer_ids']:
            if state.decode_frame is not None:
                latent,source_positions=state.decode_frame.compress(self,x,state.positions)
                plans=state.decode_frame.compressed_plans(self.layer_id,source_positions)
                has_latents=True
            else:
                latent, source_position_list, source_slots = compress_batch(x, state.batch, state.slots, self.ratio,
                    self.compressor_wkv_weight, getattr(self, 'compressor_wgate_weight', None),
                    self.compressor_norm_weight, c['rms_norm_eps'], pending.setdefault(self.layer_id, {}))
                has_latents=bool(source_position_list)
                if has_latents:
                    expected, source_positions, plans = state.source_metadata[self.layer_id]
                    if tuple(source_position_list) != expected:
                        raise RuntimeError('compressor rows differ from prepared cache coordinates')
            if has_latents:
                key, scale = self.indexer.key(latent, source_positions)
                rotated = rotate_positions(latent, source_positions, self.rope_freq)
                cache.compressed[self.layer_id].write(rotated[:, None], plans[0])
                cache.index_key[self.layer_id].write(key, plans[1])
                cache.index_scale[self.layer_id].write(scale[..., None], plans[2])
        if cp is not None:
            self.context_parallel_calls += 1
            self.context_parallel_query_tokens += query_x.shape[0]
        if query_x.shape[0]:
            has_compressed = self.ratio and max(batch.seq_lengths)//self.ratio > 0
            if self.ratio and hasattr(self, 'indexer'):
                if max(batch.seq_lengths)//self.ratio <= c['index_topk']:
                    # Exact shortcut: all completed keys fit top512, so scoring
                    # cannot exclude any key. Larger contexts use actual INT8 QLI.
                    keys = torch.arange(c['index_topk'], dtype=torch.int32, device=x.device)[None]
                    visible = torch.div(positions+1, self.ratio, rounding_mode='floor')[:, None]
                    state.selected = torch.where(keys < visible, keys, -1).contiguous()
                    if self.layer_id == c['candidate_source_layer_id']:
                        blocks = torch.arange(c['candidate_topk_blocks'], dtype=torch.int32, device=x.device)[None]
                        visible_blocks = torch.div(visible+c['candidate_block_size']-1, c['candidate_block_size'], rounding_mode='floor')
                        state.candidates = torch.where(blocks < visible_blocks, blocks, -1)[:, None].contiguous()
                else:
                    source = self.kv_source
                    selected, emitted = self.indexer.select(qr, query_x, batch,
                        cache.index_key[source].cache, cache.index_scale[source].cache.squeeze(-1),
                        state.decode_frame.tables['index_key'][source] if state.decode_frame is not None else cache.index_key[source].table(slots),
                        candidates=state.candidates if self.layer_id > c['candidate_source_layer_id'] else None)
                    state.selected = selected
                    if self.layer_id == c['candidate_source_layer_id']:
                        state.candidates = emitted
                state.index_source = self.layer_id
            if self.ratio and state.index_source != self.index_source:
                raise RuntimeError('compressed attention lacks its current-forward index source')
            kwargs = {}
            if has_compressed:
                kwargs = dict(compressed_cache=cache.compressed[self.kv_source].cache,
                              compressed_table=state.decode_frame.tables['compressed'][self.kv_source] if state.decode_frame is not None else cache.compressed[self.kv_source].table(slots),
                              compressed_indices=state.selected)
            short_prefix = (max(batch.seq_lengths) <= c['index_topk']
                            or (self.ratio and max(batch.seq_lengths)//self.ratio <= c['index_topk']))
            if short_prefix:
                self.eager_attention_calls += 1
                # Reuse the native eager sparse implementation for the exact
                # select-all prefix. The indexed long-context route stays CANN.
                from deepinfer.npu_ops.attention910c import decode, ReferenceMLASchedMeta
                requests = torch.tensor([slot for slot, count in zip(slots, batch.query_lengths)
                                         for _ in range(count)], dtype=torch.int32, device=x.device)
                window = positions[:, None]-c['sliding_window']+1+torch.arange(c['sliding_window'], dtype=torch.int32, device=x.device)[None]
                swa_indices = cache.swa[self.layer_id].physical_indices(window, requests)
                extra = {}
                if has_compressed:
                    extra = dict(key_cache_extra=cache.compressed[self.kv_source].cache,
                                 indices_in_kvcache_extra=cache.compressed[self.kv_source].physical_indices(state.selected, requests))
                context, _ = decode(q, cache.swa[self.layer_id].cache, 512, swa_indices,
                    ReferenceMLASchedMeta(), 512**-.5, sink, **extra)
            else:
                self.cann_attention_calls += 1
                context = sparse_mla(q, cache.swa[self.layer_id].cache, batch, swa_table,
                                     sink=sink, ratio=self.ratio if has_compressed else 0,
                                     window=c['sliding_window'], **kwargs)
            context = rotate_positions(context, positions, self.rope_freq, inverse=True)
        else:
            if hasattr(self, 'indexer'):
                state.selected = x.new_empty((0, c['index_topk']), dtype=torch.int32)
                state.index_source = self.layer_id
                if self.layer_id == c['candidate_source_layer_id']:
                    state.candidates = x.new_empty((0, 1, c['candidate_topk_blocks']), dtype=torch.int32)
            context = x.new_empty((0, heads, c['head_dim']))
        if cp is not None:
            context = restore_tp_heads(context, self.tp_group, total_tokens=x.shape[0], width=cp.width)
        grouped = context.reshape(x.shape[0], self.local_o_groups, -1)
        if self.fused_o_proj and state.sequence_parallel:
            import torch_npu
            projected = torch_npu.npu_transpose_batchmatmul(grouped, self.wv_bmm,
                bias=None, scale=None, perm_x1=(1, 0, 2), perm_x2=(0, 1, 2),
                perm_y=(1, 0, 2), batch_split_factor=1)
            self.fused_o_proj_calls += 1
        else:
            projected = torch.einsum('bgr,gdr->bgd', grouped.float(), self.wv.float()).bfloat16()
        output = self.projections['wo_b'](projected.flatten(1))
        if self.tp_size > 1 and reduce_results:
            dist.all_reduce(output, group=self.tp_group)
        return output

    def forward(self, trunk, previous, *, cache, pending, state):
        c = self.config
        if self.fused_mhc:
            x, post, comb, pre = self._hc_pre(trunk, previous, 'attn')
        else:
            x, pre, post, comb = self._pre(trunk, previous, 'attn')
        x = rms_norm_mixed(x, self.attn_norm_weight, c['rms_norm_eps'])[0]
        if state.sequence_parallel:
            x = self.sp.gather(x, sum(state.batch.query_lengths))
        attn = self.packed_attention(x, cache, pending, state,
                                     reduce_results=not state.sequence_parallel)
        if state.sequence_parallel:
            attn = self.sp.reduce(attn)
        after_attn = self._hc_post(attn, trunk, post, comb) if self.fused_mhc else mhc.mhc_post(attn, trunk, post, comb)
        if self.fused_mhc:
            x, post, comb, next_pre = self._hc_pre(after_attn, pre, 'ffn')
        else:
            x, next_pre, post, comb = self._pre(after_attn, pre, 'ffn')
        if self.fused_router_norm:
            x, router_x = torch.ops._C_ascend.npu_rms_norm_cast(x, self.ffn_norm_weight, c['rms_norm_eps'])
        else:
            x, router_x = rms_norm_mixed(x, self.ffn_norm_weight, c['rms_norm_eps'])
        ffn = self.packed_ffn(x,router_x,state.mc2_global_bs,
            fused_mc2=(self.fused_mc2_prefill and state.decode_frame is None
                       and state.mc2_global_bs>self.mc2.world),
            already_sharded=state.sequence_parallel)
        if self.fused_mhc:
            return self._hc_post(ffn, after_attn, post, comb), next_pre.unsqueeze(-1)
        return mhc.mhc_post(ffn, after_attn, post, comb), next_pre

    def _hc_pre(self, trunk, previous, kind):
        c = self.config
        return torch.ops._C_ascend.npu_hc_pre_v3(trunk, getattr(self, f'hc_{kind}_fn'),
            getattr(self, f'hc_{kind}_scale'), getattr(self, f'hc_{kind}_base'), previous.squeeze(-1),
            hc_mult=c['hc_mult'], hc_sinkhorn_iters=c['hc_sinkhorn_iters'],
            norm_eps=c['rms_norm_eps'], hc_eps=c['hc_eps'])

    @staticmethod
    def _hc_post(value, residual, post, combination):
        return torch.ops._C_ascend.npu_hc_post(value.unsqueeze(0), residual.unsqueeze(0),
                                            post.unsqueeze(0), combination.unsqueeze(0)).squeeze(0)


class AscendW8A8PackedModel(torch.nn.Module):
    def __init__(self, reader, device, *, max_tokens, requests=1, chunk_size=64, group=None, tp_group=None, progress=None):
        super().__init__()
        c = reader.config['text_config']
        if (not 1 <= max_tokens < 2**24 or requests < 1 or chunk_size < 1
                or c['head_dim'] != 512 or c['index_head_dim'] != 128
                or c['index_topk'] != 512 or c['n_shared_experts'] != 1
                or c['scoring_func'] != 'sqrtsoftplus' or not c['norm_topk_prob']
                or any(r not in (0, 1, 2) for r in c['compress_ratios'][:c['num_hidden_layers']])):
            raise ValueError('unsupported native paged W8A8 model/cache contract')
        self.reader, self.device, self.cfg = reader, torch.device(device), c
        self.sequence_parallel = (os.environ.get('DEEPINFER_910C_SEQUENCE_PARALLEL') == '1'
                                  and tp_group is not None and dist.get_world_size(tp_group) > 1)
        self.max_tokens, self.requests, self.chunk_size = max_tokens, requests, chunk_size
        progress = progress or (lambda event: None)
        progress({'stage': 'embedding_and_head', 'backend': 'native-paged'})
        self.register_buffer('embedding', reader.materialize('embed.weight', device))
        self.register_buffer('head', reader.materialize('head.weight', device))
        self.register_buffer('norm', reader.tensor('norm.weight')[0].to(device))
        history = CpuEngramHistory(reader) if c.get('engram_layer_ids') else None
        self.histories = [copy.copy(history) for _ in range(requests)] if history else []
        self.register_buffer('engram_rotation', load_rotation(reader, device) if history else None)
        self.engrams = torch.nn.ModuleDict({str(i): AscendW8A8Engram(reader, i, device) for i in c.get('engram_layer_ids', [])})
        self.layers = torch.nn.ModuleList()
        for i in range(c['num_hidden_layers']):
            layer = AscendW8A8PackedLayer(reader, device, layer_id=i, group=group, tp_group=tp_group)
            layer.load_local_experts(); self.layers.append(layer)
            progress({'stage': 'resident_layer', 'layer': i, 'local_experts': layer.local_expert_count, 'backend': 'native-paged'})
        self.sp = self.layers[0].sp
        self.cache = W8A8PagedCache(reader.config, device, max_tokens=max_tokens, requests=requests, max_query_tokens=chunk_size)
        self.reset()

    def reset(self):
        self.next_positions = [0]*self.requests
        self.restored_prefix_slots = set()
        self.pending = {}
        for history in self.histories:
            history.reset()
        for layer in self.layers:
            layer.eager_attention_calls = layer.cann_attention_calls = 0
            layer.fused_o_proj_calls = 0
            layer.context_parallel_calls = layer.context_parallel_query_tokens = 0
            if layer.mc2 is not None:
                layer.mc2.fused_calls=0
                layer.mc2.fused_max_output_size=0
        if getattr(self,'_decode_graph',None) is not None:
            self._decode_graph.reset_sequence()
        if getattr(self,'_verify_graph',None) is not None:
            self._verify_graph.reset_sequence()
        self.use_decode_graph=False
        self.sequence_parallel_prefill_forwards=0
        self.prompt_lookup_stats={}

    def capture_prefix(self,slot,token_ids):
        """Snapshot an exact prefix boundary, including SWA ring and Engram.

        Unlike retaining just compressed keys, this also preserves the SWA
        state at that boundary. A snapshot can only serve its exact token IDs.
        """
        length=self.next_positions[slot]
        if length!=len(token_ids):
            raise ValueError('prefix snapshot must be taken at the exact boundary')
        planes=[]
        for name in ('swa','compressed','index_key','index_scale'):
            for key,plane in getattr(self.cache,name).items():
                pages=plane.resident_pages
                if name!='swa':
                    ratio=self.cfg['compress_ratios'][key]
                    pages=min(pages,(length//ratio+plane.block_size-1)//plane.block_size)
                start=slot*plane.resident_pages
                planes.append((name,key,plane.cache[start:start+pages].clone()))
        pending={key:tuple(value.clone() for value in slots[slot])
                 for key,slots in self.pending.items() if slot in slots}
        return {'ids':tuple(token_ids),'planes':planes,'pending':pending,
                'history':list(self.histories[slot].history) if self.histories else None}

    def restore_prefix(self,snapshot,slot):
        for name,key,value in snapshot['planes']:
            plane=getattr(self.cache,name)[key]
            start=slot*plane.resident_pages
            plane.cache[start:start+value.shape[0]].copy_(value)
        for slots in self.pending.values():
            slots.pop(slot,None)
        for key,value in snapshot['pending'].items():
            self.pending.setdefault(key,{})[slot]=tuple(t.clone() for t in value)
        if self.histories:
            self.histories[slot].history=list(snapshot['history'])
        self.next_positions[slot]=len(snapshot['ids'])
        self.restored_prefix_slots.add(slot)

    def runtime_stats(self):
        return {'sequence_positions': list(self.next_positions),
                'native_eager_attention_calls': sum(layer.eager_attention_calls for layer in self.layers),
                'native_cann_attention_calls': sum(layer.cann_attention_calls for layer in self.layers),
                'fused_mhc_layers': sum(layer.fused_mhc for layer in self.layers),
                'fused_router_norm_layers': sum(layer.fused_router_norm for layer in self.layers),
                'mc2_layers': sum(layer.mc2 is not None for layer in self.layers),
                'fused_mc2_prefill_layers': sum(layer.fused_mc2_prefill for layer in self.layers),
                'fused_mc2_calls': sum(layer.mc2.fused_calls for layer in self.layers if layer.mc2 is not None),
                'fused_mc2_max_output_size': max((layer.mc2.fused_max_output_size for layer in self.layers
                                                if layer.mc2 is not None),default=0),
                'context_parallel_enabled': all(layer.context_parallel for layer in self.layers),
                'context_parallel_scope': 'cached-compressed-query-and-MLA',
                'context_parallel_swa_preserve_tp_heads': True,
                'context_parallel_requires_prefix_restore': True,
                'context_parallel_calls': sum(layer.context_parallel_calls for layer in self.layers),
                'context_parallel_query_tokens': sum(layer.context_parallel_query_tokens for layer in self.layers),
                'fused_o_proj_enabled': all(layer.fused_o_proj for layer in self.layers),
                'fused_o_proj_prefill_calls': sum(layer.fused_o_proj_calls for layer in self.layers),
                'sequence_parallel_enabled': self.sequence_parallel,
                'sequence_parallel_prefill_forwards': self.sequence_parallel_prefill_forwards,
                'sequence_parallel_tp_size': self.sp.world,
                'native_graph_captures': self._decode_graph.captures if getattr(self,'_decode_graph',None) is not None else 0,
                'native_graph_replays': self._decode_graph.replays if getattr(self,'_decode_graph',None) is not None else 0,
                'decode_host_staging_seconds': self._decode_graph.staging_seconds if getattr(self,'_decode_graph',None) is not None else 0.,
                'verification_graph_captures': self._verify_graph.captures if getattr(self,'_verify_graph',None) is not None else 0,
                'verification_graph_replays': self._verify_graph.replays if getattr(self,'_verify_graph',None) is not None else 0,
                'verification_host_staging_seconds': self._verify_graph.staging_seconds if getattr(self,'_verify_graph',None) is not None else 0.,
                'prompt_lookup':self.prompt_lookup_stats}

    def verify_chunks(self,chunks):
        """Return every target logit for fixed four-request verification."""
        from .w8a8_graph910c import NativeDecodeGraph
        width=len(chunks[0])
        if len(chunks)!=4 or any(len(chunk)!=width for chunk in chunks) or width>self.chunk_size:
            raise ValueError('verification requires four equal bounded chunks')
        if getattr(self,'_verify_graph',None) is None:
            self._verify_graph=NativeDecodeGraph(self,list(range(4)),query_tokens=width)
        if self._verify_graph.query_tokens!=width:
            raise ValueError('verification width must stay fixed for the resident graph')
        return self._verify_graph.step(chunks).reshape(4,width,-1)

    @torch.inference_mode()
    def forward_tokens(self, chunks, slots=None, *, sequence_parallel=False):
        chunks = [list(tokens) for tokens in chunks]
        slots = list(range(len(chunks))) if slots is None else list(slots)
        if (not chunks or len(chunks) != len(slots) or len(set(slots)) != len(slots)
                or any(type(s) is not int or not 0 <= s < self.requests for s in slots)
                or any(not 1 <= len(tokens) <= self.chunk_size for tokens in chunks)):
            raise ValueError('invalid packed requests or prefill chunk size')
        seq_lengths = [self.next_positions[s]+len(tokens) for tokens, s in zip(chunks, slots)]
        if max(seq_lengths) > self.max_tokens:
            raise ValueError('packed sequence exceeds cache context capacity')
        if (self.use_decode_graph and all(len(tokens)==1 for tokens in chunks)
                and slots==list(range(4)) and len(chunks)==4):
            if getattr(self,'_decode_graph',None) is None:
                from .w8a8_graph910c import NativeDecodeGraph
                self._decode_graph=NativeDecodeGraph(self,slots)
            return self._decode_graph.step(chunks)
        flat_ids = [token for tokens in chunks for token in tokens]
        image_ids = {self.reader.config.get('image_token_id'), self.reader.config.get('image_pad_token_id')}
        if any(type(t) is not int or not 0 <= t < self.embedding.shape[0] or t in image_ids for t in flat_ids):
            raise ValueError('native paged entry requires valid text token IDs')
        hashes = {}
        if self.histories:
            per_request = [self.histories[s].batch(tokens) for tokens, s in zip(chunks, slots)]
            hashes = {i: torch.cat([entry[i] for entry in per_request]) for i in self.cfg['engram_layer_ids']}
        batch = DSABatch.build([len(tokens) for tokens in chunks], seq_lengths, self.device)
        state = PackedForward(batch, slots, batch.positions(), self.cache.swa_plan(0, batch, slots), self.cache.swa[0].table(slots))
        state.source_metadata = self.cache.prepare_compressed(batch, slots)
        state.mc2_global_bs=getattr(self,'mc2_global_bs',4096)
        state.sequence_parallel=bool(sequence_parallel and self.sequence_parallel)
        if (state.sequence_parallel and self.layers[0].context_parallel
                and all(slot in self.restored_prefix_slots for slot in slots)):
            cp = QuerySlice.build(batch.query_lengths, batch.seq_lengths, self.sp.world, self.sp.rank)
            state.context_query = cp
            state.query_slots = [slots[i] for i in cp.request_indices]
            if cp.query_lengths:
                state.query_batch = DSABatch.build(cp.query_lengths, cp.seq_lengths, self.device)
                state.query_swa_table = self.cache.swa[0].table(state.query_slots)
        ids = torch.tensor(flat_ids, dtype=torch.int64, device=self.device)
        trunk = self.embedding.index_select(0, ids)[:, None].expand(-1, self.cfg['hc_mult'], -1).contiguous()
        previous = torch.zeros((len(flat_ids), self.cfg['hc_mult'], 1), dtype=torch.float32, device=self.device)
        previous[:, 0] = 1.
        if state.sequence_parallel:
            trunk=self.sp.split(trunk)
            previous=self.sp.split(previous)
            hashes={i:self.sp.split(value,fill=-1) for i,value in hashes.items()}
            self.sequence_parallel_prefill_forwards+=1
        for i, layer in enumerate(self.layers):
            if str(i) in self.engrams:
                trunk = self.engrams[str(i)](trunk, hashes[i], self.engram_rotation)
            trunk, previous = layer(trunk, previous, cache=self.cache, pending=self.pending, state=state)
        # Prefill only needs each request's final logit, not a vocabulary-sized
        # output tensor for every prompt token.
        last = batch.cu_query[1:].long()-1
        if state.sequence_parallel:
            local_hidden=(trunk.float()*previous).sum(-2).bfloat16()
            hidden=self.sp.gather(local_hidden,len(flat_ids)).index_select(0,last)
        else:
            hidden = (trunk.index_select(0, last).float()*previous.index_select(0, last)).sum(-2).bfloat16()
        hidden = rms_norm_mixed(hidden, self.norm, self.cfg['rms_norm_eps'])[0]
        logits = F.linear(hidden, self.head).float()
        for slot, end in zip(slots, seq_lengths):
            self.next_positions[slot] = end
        return logits

    def resident_bytes(self):
        result = {}
        for tensor in list(self.parameters())+list(self.buffers()):
            dtype = str(tensor.dtype)
            result[dtype] = result.get(dtype, 0)+tensor.numel()*tensor.element_size()
        result['paged_cache_payload'] = self.cache.resident_bytes()
        return result
