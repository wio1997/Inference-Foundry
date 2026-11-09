# SPDX-License-Identifier: Apache-2.0
"""Native paged W8A8 text model with packed prefill/decode on 910C.

Uses the documented V4.1 source-sharing topology and the already validated
CANN operator ABI. The original scalar model remains a separate entry point.
No vLLM engine, scheduler, forward context or Python model is imported.
"""
from dataclasses import dataclass
import copy
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

from deepinfer import envs
from deepinfer.npu_ops import mhc910c as mhc
from deepinfer.npu_ops.dsa910c import DSABatch, sparse_mla, validate_swa_window
from deepinfer.npu_ops.sp910c import SequenceShard, ordered_sum
from deepinfer.npu_ops.cp910c import QuerySlice, restore_tp_heads
from .ascend_w8a8_layer import AscendW8A8Layer
from .w8a8_compressor910c import compress_batch
from .w8a8_engram910c import AscendW8A8Engram, CpuEngramHistory, load_rotation
from .w8a8_indexer910c import AscendW8A8Indexer
from .w8a8_math910c import project, rms_norm_mixed, rotate_positions
from .w8a8_paged_cache910c import W8A8PagedCache


_layer20_probe = None
_layer21_probe = None
_consumer_end_probe = None
_consumer_scope_probe = None
_late_consumer_probes = {}


def _probe_for_layer(layer):
    return _layer20_probe if layer == 20 else _layer21_probe if layer == 21 else _late_consumer_probes.get(layer)

_swa_key_probe = None

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
    tail_ffn_indices: dict | None = None
    valid_query_lengths: tuple | None = None
    prefill_padding_margin: int = 0
    bounded_swa_base: int = 0
    bounded_swa_keep: int = 0
    pre_stored_source: int | None = None
    official_consumer: object | None = None
    eplb_rows: torch.Tensor | None = None
    state_history: object | None = None
    gate_force_random: torch.Tensor | None = None
    profile_routes: object | None = None
    images_seq_mask: torch.Tensor | None = None
    full_images_seq_mask: torch.Tensor | None = None
    full_gate_force_random: torch.Tensor | None = None

    def bind_row_controls(self, shard, images=None, gate=None, *, selected=None):
        """Keep original full-row controls aligned through SP/YOCO views."""
        if selected is not None:
            device = self.positions.device
            selected = torch.tensor(selected,dtype=torch.int64,device=device)
        for name,value in (('images_seq_mask',images),('gate_force_random',gate)):
            if value is not None and selected is not None:
                value = value.index_select(0,selected)
            setattr(self,'full_'+name,value)
            setattr(self,name,shard.split(value,fill=False)
                    if value is not None and self.sequence_parallel else value)


def repack_swa(plane, slots, base, keep, batch):
    """Gather an unaligned final SWA interval into reusable compact pages."""
    if (keep!=140 or base<0 or plane.block_size!=128 or not slots or
            base+keep>plane.logical_rows or len(set(slots))!=len(slots)):
        raise ValueError('invalid unaligned SWA replay interval')
    geometry=(plane.requests,plane.logical_pages,plane.resident_pages,plane.block_size)
    key=('bounded_swa_pack',base,keep,tuple(slots),geometry)
    item=batch.metadata_cache.get(key)
    if item is None:
        device=plane.cache.device
        requests=torch.tensor(slots,dtype=torch.int32,device=device)
        logical=torch.arange(base,base+keep,dtype=torch.int32,device=device)[None,:].expand(len(slots),-1)
        indices=plane.physical_indices(logical,requests).reshape(-1).long()
        pages=(keep+127)//128
        packed=plane.cache.new_zeros((len(slots),pages*128,1,plane.width))
        table=torch.arange(len(slots)*pages,dtype=torch.int32,device=device).reshape(len(slots),pages)
        item=(indices,packed,table)
        batch.metadata_cache[key]=item
    indices,packed,table=item
    values=plane.cache.flatten(0,1).index_select(0,indices).view(len(slots),keep,1,plane.width)
    packed[:,:keep].copy_(values)
    return packed.view(-1,128,1,plane.width),table


class AscendW8A8PackedLayer(AscendW8A8Layer):
    def __init__(self, reader, device, *, layer_id, group=None, tp_group=None):
        super().__init__(reader, device, layer_id=layer_id, group=group)
        # Original image routing belongs to target prefill. DFlash uses
        # InputMetadataV4 without an image mask and keeps its text-only gate.
        self.register_buffer('ffn_gate_bias_vl',reader.tensor(self.prefix+'.ffn.gate.bias_vl')[0].to(device).float()
                             if layer_id < self.config['num_hidden_layers'] else None)
        self.tp_group = tp_group
        self.sp = SequenceShard(tp_group)
        self.eager_attention_calls = self.cann_attention_calls = 0
        self.bounded_swa_calls = 0
        self.bounded_swa_packed_calls = 0
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
        self.kv_token_parallel = os.environ.get('DEEPINFER_910C_KV_TOKEN_PARALLEL') == '1'
        self.kv_token_parallel_calls = 0
        self.prefill_kv_overlap_started = 0
        self.prefill_kv_overlap_completed = 0
        self.tail_ffn_calls = self.tail_ffn_tokens = self.tail_ffn_skipped = 0
        if self.context_parallel:
            self.cp_wq_b = reader.load_linear(self.prefix+'.attn.wq_b', device)
            self.register_buffer('cp_attn_sink', self.attn_attn_sink.clone())
        self.cp_local_o = (self.context_parallel and layer_id == 20
                           and os.environ.get('DEEPINFER_910C_YOCO_CP_LOCAL_O') == '1')
        self.cp_o_available = self.cp_local_o or (
            self.context_parallel and 20 < layer_id < self.config['num_hidden_layers']
            and os.environ.get('DEEPINFER_910C_YOCO_CP_O_POOL') == '1')
        if self.cp_o_available:
            self.cp_wo_b = self.projections['wo_b']
            self.register_buffer('cp_wv_bmm', self.wv.transpose(1, 2).contiguous())
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
        self.fused_o_proj_decode = os.environ.get('DEEPINFER_910C_FUSED_O_PROJ_DECODE') == '1'
        self.fused_o_proj_calls = 0
        self.fused_o_proj_decode_calls = 0
        if self.fused_o_proj:
            if self.wv.dtype != torch.bfloat16:
                raise ValueError('fused O projection requires checkpoint BF16 FLOAT weights')
            # Pinned V4.1 BF16 ABI consumes [groups, input, output] in ND.
            self.register_buffer('wv_bmm', self.wv.transpose(1, 2).contiguous())
        self.index_source = max((s for s in self.config['index_source_layer_ids'] if s <= layer_id), default=None) if self.ratio else None
        if layer_id in self.config['index_source_layer_ids']:
            self.indexer = AscendW8A8Indexer(reader, device, layer_id=layer_id)
        self.yoco_merged_qb = None
        self.yoco_merged_qb_calls = 0
        if (self.context_parallel and 20 <= layer_id < self.config['num_hidden_layers']
                and layer_id in self.config['index_source_layer_ids']
                and not self.indexer.fp4_index
                and os.environ.get('DEEPINFER_910C_YOCO_MERGED_QB_POOL') == '1'):
            from .merged_qb910c import MergedQB
            self.yoco_merged_qb = MergedQB(self.cp_wq_b,self.indexer.query_projection)
        self.mc2=None
        self.shared_expert_stream=None
        self.fused_mc2_prefill=os.environ.get('DEEPINFER_910C_FUSED_MC2_PREFILL')=='1'
        self.fused_mc2_decode=os.environ.get('DEEPINFER_910C_FUSED_MC2_DECODE')=='1'
        if os.environ.get('DEEPINFER_910C_MC2')=='1':
            from deepinfer.npu_ops.mc2_910c import NativeMC2
            self.mc2=NativeMC2(group)

    def load_local_experts(self):
        from deepinfer.engine.eplb_model910c import load_physical_experts, requested
        if requested():
            load_physical_experts(self)
        else:
            super().load_local_experts()
        from deepinfer.npu_ops.w8a8 import AscendW8A8ExpertBank
        self.local_expert_count=len(self.expert_cache)
        self.expert_bank=AscendW8A8ExpertBank(self.expert_cache.values())
        self.expert_cache.clear()
        if self.fused_mc2_prefill or self.fused_mc2_decode:
            if self.mc2 is None:
                raise ValueError('fused MC2 prefill requires native MC2 communicator')
            self.expert_bank.prepare_fused_mc2()

    def packed_ffn(self,x,router_x,global_bs,*,fused_mc2=False,already_sharded=False,token_indices=None,decode_graph=False,router_plan=None,active_mask=None,gate_force_random=None,profile_routes=None,image_mask=None):
        probe = _probe_for_layer(self.layer_id)
        from deepinfer.npu_ops.moe910c import dispatch_combine,route
        # Attention returns the same token batch on each TP rank. Dispatch
        # each token once to global EP16, then restore the TP-replicated batch.
        count=x.shape[0]
        width=count if already_sharded else (count+self.tp_size-1)//self.tp_size
        rank=dist.get_rank(self.tp_group) if self.tp_size>1 else 0
        first=0 if already_sharded else rank*width
        local=x[first:first+width]
        routing=router_x[first:first+width]
        local_active=None if active_mask is None else active_mask[first:first+width]
        local_image=None if image_mask is None else image_mask[first:first+width]
        if token_indices is not None:
            if not already_sharded:
                raise ValueError('tail FFN pruning requires sequence-parallel ownership')
            local=local.index_select(0,token_indices)
            routing=routing.index_select(0,token_indices)
            if local_active is not None:local_active=local_active.index_select(0,token_indices)
            if local_image is not None:local_image=local_image.index_select(0,token_indices)
            self.tail_ffn_calls += 1
            self.tail_ffn_tokens += local.shape[0]
            self.tail_ffn_skipped += count-local.shape[0]
        valid=local.shape[0]
        if not valid:
            local=x[:1];routing=router_x[:1]
            if local_active is not None:local_active=torch.zeros(1,dtype=torch.bool,device=x.device)
            if local_image is not None:local_image=torch.zeros(1,dtype=torch.bool,device=x.device)
        if router_plan is None:
            logits=project(routing,self.ffn_gate_weight)
        else:
            from .official_cp_tiles910c import apply
            logits=apply(lambda rows: project(rows,self.ffn_gate_weight),router_plan,routing)
        if probe is not None:
            probe('router_logits', 'local', logits)
        # One model-owned stream per rank. The flag is frozen before graph
        # capture: changing it later cannot change an already captured graph.
        overlap = decode_graph and fused_mc2 and self.mc2 is not None and self.shared_expert_stream is not None
        if overlap:
            main_stream=torch.npu.current_stream()
            self.shared_expert_stream.wait_stream(main_stream)
            with torch.npu.stream(self.shared_expert_stream):
                local.record_stream(self.shared_expert_stream)
                shared=self.shared_expert(local)
                shared.record_stream(main_stream)
        if local_image is not None and self.ffn_gate_bias_vl is None:
            raise ValueError('image routing requires original target image-bias weights')
        ids,weights=route(logits,self.ffn_gate_bias,self.config['num_experts_per_tok'],1.,
            image_bias=self.ffn_gate_bias_vl if local_image is not None else None,image_mask=local_image)
        if probe is not None:
            probe('router_selection', 'local', ids, weights)
        if hasattr(self,'eplb_layer_state'):
            from deepinfer.engine.eplb_w8a8_910c import map_logical_routes
            state=self.eplb_layer_state
            ids=map_logical_routes(ids,state.logical_to_physical_map,state.logical_count,self.eplb_layout[4].rank())
        if gate_force_random is not None:
            from deepinfer.npu_ops.profile_routing910c import force_random_routes
            flags=gate_force_random[first:first+width]
            if token_indices is not None:
                flags=flags.index_select(0,token_indices)
            if not valid:
                flags=torch.zeros(ids.shape[0],dtype=torch.bool,device=ids.device)
            ids,weights=force_random_routes(ids,weights,flags,random_routes=profile_routes)
        if not valid and self.mc2 is None:
            ids=ids.fill_(-1);weights=weights.zero_()
        elif not valid:
            weights=weights.zero_()
        if self.mc2 is not None:
            routed=self.mc2(local,ids,weights,self.expert_bank,
                            num_experts=(self.eplb_layout[1] if hasattr(self,'eplb_layout') else self.config['n_routed_experts']),
                            global_bs=global_bs,fused=fused_mc2,active_mask=local_active)
        else:
            if local_active is not None:
                ids=torch.where(local_active[:,None],ids,-1)
                weights=torch.where(local_active[:,None],weights,0.)
            routed=dispatch_combine(local,ids,weights,
                self.eplb_layout[1] if hasattr(self,'eplb_layout') else self.config['n_routed_experts'],None,self.group,
                weight_on_output=True,local_batch=self.expert_bank,trusted_indices=True)
        if overlap:
            main_stream.wait_stream(self.shared_expert_stream)
        else:
            shared=self.shared_expert(local)
        if probe is not None:
            probe('routed', 'local', routed)
            probe('shared', 'local', shared)
        output=(shared.float()+routed.float()*self.config['routed_scaling_factor']).bfloat16()
        if probe is not None:
            probe('ffn_sum', 'local', output)
        output=output[:valid]
        if token_indices is not None:
            # Only a suffix can influence the final logit after the last KV
            # source. The discarded rows never feed a future valid SWA row.
            restored=torch.zeros_like(x)
            restored.index_copy_(0,token_indices,output)
            return restored
        if self.tp_size>1 and not already_sharded:
            padded=x.new_zeros((width,x.shape[1]))
            padded[:output.shape[0]].copy_(output)
            gathered=x.new_empty((width*self.tp_size,x.shape[1]))
            dist.all_gather_into_tensor(gathered,padded.contiguous(),group=self.tp_group)
            return gathered[:count]
        return output

    def _fixed_consumer(self, state, feature):
        active = (state.official_consumer is not None
                  and 20 <= self.layer_id < self.config['num_hidden_layers']
                  and (self.layer_id == 20
                       or os.environ.get('DEEPINFER_910C_YOCO_EXTEND_' + feature) == '1')
                  and os.environ.get('DEEPINFER_910C_YOCO_FIXED_' + feature) == '1')
        if _consumer_scope_probe is not None:
            _consumer_scope_probe(self.layer_id, feature, active)
        return active

    def _merge_qb_active(self, state, fixed_q_b):
        return bool(self.yoco_merged_qb is not None
                    and self.context_parallel and fixed_q_b
                    and 20 <= self.layer_id < self.config['num_hidden_layers']
                    and state.sequence_parallel and state.context_query is not None
                    and state.official_consumer is not None and state.decode_frame is None
                    and os.environ.get('DEEPINFER_910C_YOCO_MERGED_QB') == '1')

    def packed_attention(self, x, cache, pending, state, *, reduce_results=True, input_is_sharded=False):
        probe = _probe_for_layer(self.layer_id)
        c = self.config
        # The pinned pure-SWA kernel changes BF16 rounding with head grouping.
        # Preserve the proven TP head layout for SWA; compressed MLA uses CP.
        cp = state.context_query if self.ratio else None
        batch = state.query_batch if cp is not None else state.batch
        total_tokens = sum(state.batch.query_lengths)
        if input_is_sharded:
            if cp is None or self.layer_id in c['kv_source_layer_ids']:
                raise ValueError('token-local KV requires compressed non-source CP layer')
            query_x = x[:cp.end-cp.start]
        else:
            query_x = x[cp.start:cp.end] if cp is not None else x
        positions = state.positions[cp.start:cp.end] if cp is not None else state.positions
        slots = state.query_slots if cp is not None else state.slots
        swa_table = state.query_swa_table if cp is not None else state.swa_table
        heads = c['num_attention_heads'] if cp is not None else self.local_heads
        sink = self.cp_attn_sink if cp is not None else self.attn_attn_sink
        # The fixed eager CP path can submit the KV exchange before the
        # independent Q projection. Hold both tensors and Work through wait.
        wkv = self.projections['wkv']
        project_kv = wkv
        if self._fixed_consumer(state, 'WKV'):
            from .official_cp_tiles910c import apply, plan
            kv_plan = plan(state.query_batch if input_is_sharded else state.batch)
            project_kv = lambda values: apply(wkv, kv_plan, values)
        kv_work = None
        kv_gathered = None
        if (input_is_sharded and state.decode_frame is None and self.sp.world > 1
                and os.environ.get('DEEPINFER_910C_PREFILL_KV_OVERLAP') == '1'):
            linear_kv = project_kv(query_x)
            local_kv = rms_norm_mixed(linear_kv, self.attn_kv_norm_weight, c['rms_norm_eps'])[0]
            if probe is not None and self.layer_id != 20:
                probe('wkv_linear','local',linear_kv)
                probe('wkv_norm','local',local_kv)
            padded = local_kv.new_zeros((cp.width, c['head_dim']))
            padded[:local_kv.shape[0]].copy_(local_kv)
            kv_gathered = local_kv.new_empty((cp.width * self.sp.world, c['head_dim']))
            kv_work = dist.all_gather_into_tensor(kv_gathered, padded.contiguous(), group=self.sp.group, async_op=True)
            if kv_work is None:
                raise RuntimeError('asynchronous KV gather returned no work handle')
            self.prefill_kv_overlap_started += 1
        fixed_q_a = self._fixed_consumer(state, 'Q_A')
        fixed_q_b = self._fixed_consumer(state, 'Q_B')
        if fixed_q_a or fixed_q_b:
            from .official_cp_tiles910c import apply, plan
        query_plan = plan(state.query_batch) if fixed_q_a or fixed_q_b else None
        index_projection = None
        if query_x.shape[0]:
            q_a = self.projections['wq_a']
            q_linear_a = apply(q_a,query_plan,query_x) if fixed_q_a else q_a(query_x)
            if probe is not None:
                probe('q_linear_a', 'local', q_linear_a)
            qr = rms_norm_mixed(q_linear_a, self.attn_q_norm_weight, c['rms_norm_eps'])[0]
            if probe is not None:
                probe('q_norm_a', 'local', qr)
            q_b = self.cp_wq_b if cp is not None else self.projections['wq_b']
            if self._merge_qb_active(state,fixed_q_b):
                q,index_projection = apply(self.yoco_merged_qb,query_plan,qr)
                self.yoco_merged_qb_calls += 1
            else:
                q = apply(q_b,query_plan,qr) if fixed_q_b else q_b(qr)
            q = q.unflatten(-1, (heads, c['head_dim']))
            if probe is not None:
                probe('q_linear_b', 'local', q)
            q = rotate_positions(q, positions, self.rope_freq)
            if probe is not None:
                probe('q_rope', 'local', q)
        if input_is_sharded:
            # Query already owns local normalized tokens. Publish512 KV
            # channels instead of gathering5120 hidden channels. Keep global
            # RoPE/cache-write coordinates and all compressor source math.
            if kv_work is None:
                linear_kv = project_kv(query_x)
                local_kv = rms_norm_mixed(linear_kv, self.attn_kv_norm_weight, c['rms_norm_eps'])[0]
                if probe is not None and self.layer_id != 20:
                    probe('wkv_linear','local',linear_kv)
                    probe('wkv_norm','local',local_kv)
                padded = local_kv.new_zeros((cp.width, c['head_dim']))
                padded[:local_kv.shape[0]].copy_(local_kv)
                kv = self.sp.gather(padded, total_tokens)
            else:
                if kv_work.wait() is False:
                    raise RuntimeError('asynchronous KV gather failed')
                kv = kv_gathered[:total_tokens]
                self.prefill_kv_overlap_completed += 1
            self.kv_token_parallel_calls += 1
        else:
            key_probe = _swa_key_probe if self.layer_id == 20 else None
            projected_kv = project_kv(x)
            kv = rms_norm_mixed(projected_kv, self.attn_kv_norm_weight, c['rms_norm_eps'])[0]
            if key_probe is not None:
                assert state.official_consumer is not None
                key_probe.tensor('normalized_input', x)
                key_probe.tensor('linear_kv', projected_kv)
                key_probe.tensor('norm_kv', kv)
            if probe is not None and self.layer_id != 20:
                probe('wkv_linear','global',projected_kv)
                probe('wkv_norm','global',kv)
        kv = rotate_positions(kv, state.positions, self.rope_freq)
        if probe is not None and self.layer_id != 20:
            probe('wkv_rope','global',kv)
        consumer = state.official_consumer
        if consumer is None:
            cache.swa[self.layer_id].write(kv[:, None], state.swa_plan)
        else:
            key_probe = _swa_key_probe if self.layer_id == 20 else None
            if key_probe is not None:
                key_probe.positions(state.positions)
                key_probe.tensor('rope_kv', kv)
                key_probe.before_write(consumer, cache.swa[self.layer_id])
            consumer.write(cache.swa[self.layer_id],kv)
            if key_probe is not None:
                key_probe.pages(consumer)
        if self.layer_id in c['kv_source_layer_ids'] and state.pre_stored_source != self.layer_id:
            if state.decode_frame is not None:
                latent,source_positions=state.decode_frame.compress(self,x,state.positions)
                plans=state.decode_frame.compressed_plans(self.layer_id,source_positions)
                has_latents=True
            else:
                latent, source_position_list, source_slots = compress_batch(x, state.batch, state.slots, self.ratio,
                    self.compressor_wkv_weight, getattr(self, 'compressor_wgate_weight', None),
                    self.compressor_norm_weight, c['rms_norm_eps'], pending.setdefault(self.layer_id, {}),
                    trace=state.state_history, layer_id=self.layer_id)
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
                cache.index_scale[self.layer_id].write(scale[:, None, :], plans[2])
        if cp is not None:
            self.context_parallel_calls += 1
            self.context_parallel_query_tokens += query_x.shape[0]
        if query_x.shape[0]:
            has_compressed = self.ratio and (envs.DEEPINFER_BATCH_INVARIANT
                                             or max(batch.seq_lengths)//self.ratio > 0)
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
                        candidates=state.candidates if self.layer_id > c['candidate_source_layer_id'] else None,
                        projected_query=index_projection)
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
            if consumer is not None:
                if not has_compressed or self.ratio != 1:
                    raise RuntimeError('official YOCO consumer requires full C1 cache')
                context = consumer.attention(q,batch,slots,sink,kwargs['compressed_cache'],
                                             kwargs['compressed_table'],state.selected)
                self.cann_attention_calls += 1
            elif short_prefix and not envs.DEEPINFER_BATCH_INVARIANT:
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
                if state.bounded_swa_base:self.bounded_swa_calls += 1
                ori=cache.swa[self.layer_id].cache
                repacked=state.bounded_swa_keep==140
                if repacked:
                    ori,swa_table=repack_swa(cache.swa[self.layer_id],slots,state.bounded_swa_base,140,batch)
                    self.bounded_swa_packed_calls+=1
                context = sparse_mla(q, ori, batch, swa_table,
                                     sink=sink, ratio=self.ratio if has_compressed else 0,
                                     window=c['sliding_window'], swa_base=state.bounded_swa_base,
                                     swa_repacked=repacked, **kwargs)
            if probe is not None:
                probe('raw_context', 'local', context)
            context = rotate_positions(context, positions, self.rope_freq, inverse=True)
            if probe is not None:
                probe('inverse_rope', 'local', context)
        else:
            if hasattr(self, 'indexer'):
                state.selected = x.new_empty((0, c['index_topk']), dtype=torch.int32)
                state.index_source = self.layer_id
                if self.layer_id == c['candidate_source_layer_id']:
                    state.candidates = x.new_empty((0, 1, c['candidate_topk_blocks']), dtype=torch.int32)
            context = x.new_empty((0, heads, c['head_dim']))
        if self.layer_id==20 and getattr(state,'boundary20_trace_dir',None) is not None:
            from .official_source20_910c import record_index_contract
            record_index_contract(state,self.layer_id)
        cp_local_o = self.cp_local_o and consumer is not None
        if _consumer_scope_probe is not None and consumer is not None:
            _consumer_scope_probe(self.layer_id, 'CP_O', cp_local_o)
        if cp_local_o:
            if cp is None or reduce_results:
                raise RuntimeError('local YOCO projection requires sequence-parallel consumer rows')
            if context.shape[0] == 0:
                return x.new_zeros((cp.width, c['hidden_size']))
            grouped = context.reshape(context.shape[0], c['o_groups'], -1)
            wv_bmm = self.cp_wv_bmm
            wv = wv_bmm.transpose(1, 2)
            wo_b = self.cp_wo_b
        else:
            if cp is not None:
                context = restore_tp_heads(context, self.tp_group, total_tokens=total_tokens, width=cp.width)
                if probe is not None:
                    probe('tp_heads', 'global', context)
            grouped = context.reshape(total_tokens, self.local_o_groups, -1)
            wv_bmm = self.wv_bmm if self.fused_o_proj else None
            wv = self.wv
            wo_b = self.projections['wo_b']
        if envs.DEEPINFER_BATCH_INVARIANT:
            projected = torch.stack([project(grouped[:,g,:],wv[g].to(grouped.dtype))
                                     for g in range(grouped.shape[1])],1)
        elif self.fused_o_proj and (state.sequence_parallel or
                (self.fused_o_proj_decode and state.decode_frame is not None)):
            import torch_npu
            projected = torch_npu.npu_transpose_batchmatmul(grouped, wv_bmm,
                bias=None, scale=None, perm_x1=(1, 0, 2), perm_x2=(0, 1, 2),
                perm_y=(1, 0, 2), batch_split_factor=1)
            if state.sequence_parallel:self.fused_o_proj_calls += 1
        else:
            projected = torch.einsum('bgr,gdr->bgd', grouped.float(), wv.float()).bfloat16()
        if probe is not None:
            probe('o_a', 'local' if cp_local_o else 'global', projected)
        if cp_local_o:
            from .official_cp_tiles910c import apply, plan
            output = apply(wo_b,plan(state.query_batch),projected.flatten(1))
        else:
            output = wo_b(projected.flatten(1))
        if probe is not None:
            probe('o_b', 'local' if cp_local_o else 'global', output)
        if cp_local_o:
            pad = cp.width - output.shape[0]
            if pad:
                output = torch.cat((output, output.new_zeros((pad, output.shape[1]))), dim=0)
            return output
        if self.tp_size > 1 and reduce_results:
            if envs.DEEPINFER_BATCH_INVARIANT:
                output = ordered_sum(output, self.tp_group)
            else:
                dist.all_reduce(output, group=self.tp_group)
        return output

    def forward(self, trunk, previous, *, cache, pending, state):
        probe = _probe_for_layer(self.layer_id)
        c = self.config
        fused_mhc = self.fused_mhc or envs.DEEPINFER_BATCH_INVARIANT
        key_probe = _swa_key_probe if self.layer_id == 20 else None
        if key_probe is not None:
            key_probe.bind(state, trunk, previous)
        if probe is not None:
            probe('input', 'local', trunk, previous, state=state)
        if fused_mhc:
            if self._fixed_consumer(state, 'MHC_PRE'):
                from .official_cp_tiles910c import apply, plan
                x, post, comb, pre = apply(lambda a,b:self._hc_pre(a,b,'attn'),plan(state.query_batch),trunk,previous)
            else:
                x, post, comb, pre = self._hc_pre(trunk, previous, 'attn')
        else:
            x, pre, post, comb = self._pre(trunk, previous, 'attn')
        if probe is not None:
            probe('attn_mhc_pre', 'local', x, pre, post, comb)
        if key_probe is not None:
            key_probe.local('attn_mhc', x, pre, post, comb)
        x = rms_norm_mixed(x, self.attn_norm_weight, c['rms_norm_eps'])[0]
        if key_probe is not None:
            key_probe.local('local_norm', x)
        if probe is not None and self.layer_id != 20:
            probe('attn_normalized','local',x)
        kv_shard = bool(self.kv_token_parallel and state.sequence_parallel
                        and state.context_query is not None and self.ratio
                        and self.layer_id not in c['kv_source_layer_ids'])
        if state.sequence_parallel and not kv_shard:
            x = self.sp.gather(x, sum(state.batch.query_lengths))
        attn = self.packed_attention(x, cache, pending, state,
                                     reduce_results=not state.sequence_parallel,
                                     input_is_sharded=kv_shard)
        if state.sequence_parallel and not (self.cp_local_o and state.official_consumer is not None):
            if self.layer_id == 20 and state.official_consumer is not None and os.environ.get('DEEPINFER_910C_YOCO_FP32_TP_REFERENCE') == '1':
                attn = self.sp.reduce(attn.float()).to(attn.dtype)
            else:
                attn = self.sp.reduce(attn)
        if probe is not None:
            probe('tp_reduced', 'local', attn)
        after_attn = self._hc_post(attn, trunk, post, comb) if fused_mhc else mhc.mhc_post(attn, trunk, post, comb)
        if probe is not None:
            probe('attn_mhc_post', 'local', after_attn)
        if fused_mhc:
            if self._fixed_consumer(state, 'FFN_MHC_PRE'):
                from .official_cp_tiles910c import apply, plan
                x, post, comb, next_pre = apply(lambda a,b:self._hc_pre(a,b,'ffn'),plan(state.query_batch),after_attn,pre)
            else:
                x, post, comb, next_pre = self._hc_pre(after_attn, pre, 'ffn')
        else:
            x, next_pre, post, comb = self._pre(after_attn, pre, 'ffn')
        if probe is not None:
            probe('ffn_mhc_pre', 'local', x, next_pre, post, comb)
        if self.fused_router_norm:
            x, router_x = torch.ops._C_ascend.npu_rms_norm_cast(x, self.ffn_norm_weight, c['rms_norm_eps'])
        else:
            x, router_x = rms_norm_mixed(x, self.ffn_norm_weight, c['rms_norm_eps'])
        if probe is not None:
            probe('ffn_normalized', 'local', x, router_x)
        router_plan = None
        if self._fixed_consumer(state, 'ROUTER_LINEAR'):
            from .official_cp_tiles910c import plan
            assert state.sequence_parallel and state.tail_ffn_indices is None
            router_plan = plan(state.query_batch)
        active_mask=None
        if hasattr(self,"eplb_layer_state"):
            from deepinfer.engine.eplb_model910c import active_rows
            active_mask=active_rows(self,state)
        ffn = self.packed_ffn(x,router_x,state.mc2_global_bs,
            image_mask=state.images_seq_mask,
            active_mask=active_mask,
            gate_force_random=state.gate_force_random,
            profile_routes=None if state.profile_routes is None else state.profile_routes[self.layer_id],
            router_plan=router_plan,
            fused_mc2=(self.fused_mc2_prefill and state.decode_frame is None
                       and state.mc2_global_bs>self.mc2.world)
                       or (self.fused_mc2_decode and state.decode_frame is not None),
            already_sharded=state.sequence_parallel,
            token_indices=(state.tail_ffn_indices.get(self.layer_id)
                           if state.tail_ffn_indices is not None else None),
            decode_graph=state.decode_frame is not None)
        if probe is not None:
            probe('ffn_output', 'local', ffn)
        if fused_mhc:
            return self._hc_post(ffn, after_attn, post, comb), next_pre.unsqueeze(-1)
        return mhc.mhc_post(ffn, after_attn, post, comb), next_pre

    def _hc_pre(self, trunk, previous, kind):
        c = self.config
        if envs.DEEPINFER_BATCH_INVARIANT:
            from deepinfer.npu_ops.invariant910c import mhc_pre
            return mhc_pre(trunk,previous,getattr(self,f'hc_{kind}_fn'),
                getattr(self,f'hc_{kind}_scale'),getattr(self,f'hc_{kind}_base'),
                c['rms_norm_eps'],c['hc_eps'],c['hc_sinkhorn_iters'])
        return torch.ops._C_ascend.npu_hc_pre_v3(trunk, getattr(self, f'hc_{kind}_fn'),
            getattr(self, f'hc_{kind}_scale'), getattr(self, f'hc_{kind}_base'), previous.squeeze(-1),
            hc_mult=c['hc_mult'], hc_sinkhorn_iters=c['hc_sinkhorn_iters'],
            norm_eps=c['rms_norm_eps'], hc_eps=c['hc_eps'])

    @staticmethod
    def _hc_post(value, residual, post, combination):
        if envs.DEEPINFER_BATCH_INVARIANT:
            from deepinfer.npu_ops.invariant910c import mhc_post
            return mhc_post(value,residual,post,combination)
        return torch.ops._C_ascend.npu_hc_post(value.unsqueeze(0), residual.unsqueeze(0),
                                            post.unsqueeze(0), combination.unsqueeze(0)).squeeze(0)


class AscendW8A8PackedModel(torch.nn.Module):
    def __init__(self, reader, device, *, max_tokens, requests=1, chunk_size=64, group=None, tp_group=None, progress=None):
        super().__init__()
        c = reader.config['text_config']
        validate_swa_window(c['sliding_window'])
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
        self.tail_ffn = os.environ.get('DEEPINFER_910C_TAIL_FFN') == '1'
        self.tail_ffn_source = max(c['kv_source_layer_ids'])
        if self.tail_ffn and any(i > self.tail_ffn_source for i in c.get('engram_layer_ids', [])):
            raise ValueError('tail FFN pruning does not support downstream Engram')
        progress = progress or (lambda event: None)
        progress({'stage': 'embedding_and_head', 'backend': 'native-paged'})
        self.register_buffer('embedding', reader.materialize('embed.weight', device))
        self.register_buffer('head', reader.materialize('head.weight', device))
        self.tp_group=tp_group
        self.target_vocab_parallel=(os.environ.get('DEEPINFER_910C_TARGET_VOCAB_PARALLEL')=='1'
                                    and tp_group is not None and dist.get_world_size(tp_group)>1)
        size=dist.get_world_size(tp_group) if self.target_vocab_parallel else 1
        rank=dist.get_rank(tp_group) if self.target_vocab_parallel else 0
        if c['vocab_size']%size:raise ValueError('target vocabulary must divide TP')
        self.target_vocab_width=c['vocab_size']//size
        self.target_vocab_start=rank*self.target_vocab_width
        # A view retains the proven full head for prefill/ordinary AR. Only
        # verification reads a1/TP vocabulary slice, avoiding repeated logits.
        self.register_buffer('target_head',self.head.narrow(0,self.target_vocab_start,self.target_vocab_width))
        self.register_buffer('norm', reader.tensor('norm.weight')[0].to(device))
        history = CpuEngramHistory(reader) if c.get('engram_layer_ids') else None
        self.histories = [copy.copy(history) for _ in range(requests)] if history else []
        self.register_buffer('engram_rotation', load_rotation(reader, device) if history else None)
        self.engrams = torch.nn.ModuleDict({str(i): AscendW8A8Engram(reader, i, device) for i in c.get('engram_layer_ids', [])})
        self.engram_uva_runtime = None
        if history and os.environ.get('DEEPINFER_910C_ENGRAM_UVA_AVAILABLE') == '1':
            from .engram_uva910c import NativeUvaEngram
            self.engram_uva_runtime = NativeUvaEngram(reader,history,device,tp_group,progress)
        self.engram_device_hash = None
        if history and os.environ.get('DEEPINFER_910C_ENGRAM_DEVICE_HASH_AVAILABLE') == '1':
            from deepinfer.npu_ops.engram_hash910c import DeviceEngramHash
            self.engram_device_hash = DeviceEngramHash(history,device)
        # Create before any decode graph is captured; every layer on this rank
        # shares the stream and joins it before consuming its shared output.
        self.shared_expert_stream = (torch.npu.Stream(device=self.device)
            if os.environ.get('DEEPINFER_910C_SHARED_EXPERT_OVERLAP_DECODE') == '1' else None)
        self.layers = torch.nn.ModuleList()
        for i in range(c['num_hidden_layers']):
            layer = AscendW8A8PackedLayer(reader, device, layer_id=i, group=group, tp_group=tp_group)
            layer.shared_expert_stream = self.shared_expert_stream
            layer.load_local_experts(); self.layers.append(layer)
            progress({'stage': 'resident_layer', 'layer': i, 'local_experts': layer.local_expert_count, 'backend': 'native-paged'})
        self.sp = self.layers[0].sp
        self.cache = W8A8PagedCache(reader.config, device, max_tokens=max_tokens, requests=requests, max_query_tokens=chunk_size)
        self.dspark = None
        self.dspark_graphs = None
        self.engram_prefetch_runtime = None
        if os.environ.get('DEEPINFER_910C_DSPARK') == '1':
            from .dspark910c import DSpark910C
            self.dspark = DSpark910C(reader,device,requests=requests,
                max_tokens=max_tokens,group=group,tp_group=tp_group,progress=progress)
        from deepinfer.engine.eplb_model910c import initialize
        self.eplb_runtime=initialize(self)
        self.reset()

    def reset(self):
        if getattr(self, '_original_state_history', None) is not None:
            self._original_state_history.clear()
        if getattr(self,"eplb_runtime",None) is not None:self.eplb_runtime.reset_inputs()
        if self.engram_uva_runtime is not None:
            self.engram_uva_runtime.reset_stats()
        if self.engram_prefetch_runtime is not None:
            self.engram_prefetch_runtime.reset_stats()
        self.next_positions = [0]*self.requests
        self.yoco_swa_valid_start = [0]*self.requests
        self.yoco_official_metadata_calls = 0
        self.yoco_official_fallback_calls = 0
        self.restored_prefix_slots = set()
        self.pending = {}
        for history in self.histories:
            history.reset()
        for layer in self.layers:
            layer.eager_attention_calls = layer.cann_attention_calls = 0
            layer.bounded_swa_calls = 0
            layer.bounded_swa_packed_calls = 0
            layer.fused_o_proj_calls = 0
            layer.fused_o_proj_decode_calls = 0
            layer.context_parallel_calls = layer.context_parallel_query_tokens = 0
            layer.yoco_merged_qb_calls = 0
            layer.kv_token_parallel_calls = 0
            layer.prefill_kv_overlap_started = layer.prefill_kv_overlap_completed = 0
            layer.tail_ffn_calls = layer.tail_ffn_tokens = layer.tail_ffn_skipped = 0
            if layer.mc2 is not None:
                layer.mc2.fused_calls=0
                layer.mc2.fused_max_output_size=0
        if getattr(self,'_decode_graph',None) is not None:
            self._decode_graph.reset_sequence()
        if getattr(self,'_verify_graph',None) is not None:
            self._verify_graph.reset_sequence()
        self.use_decode_graph=False
        self.sequence_parallel_prefill_forwards=0
        self.prefill_padding_forwards=0
        self.prefill_padding_tokens=0
        self.prefill_wall_components={}
        self.prefill_engram_uva_calls=0
        self.prefill_engram_uva_tokens=0
        self.prefill_engram_uva_checked=0
        self.prefill_device_hash_calls=0
        self.prefill_device_hash_tokens=0
        self.prefill_device_hash_checked=0
        self.yoco_exact_compactions=0
        self.yoco_exact_discarded_tokens=0
        self.yoco_metadata_fusions=0
        self.yoco_preplan_prepared=0
        self.yoco_preplan_consumed=0
        self.yoco_bounded_compactions=0
        self.yoco_bounded_discarded_tokens=0
        self.target_vocab_top1_calls=0
        self.prompt_lookup_stats={}
        self.dspark_stats={}
        if self.dspark is not None:
            self.dspark.reset_stats()
        if self.dspark_graphs is not None:
            self.dspark_graphs.reset_sequence()

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
        return {'index_format':self.cache.index_format,'ids':tuple(token_ids),'planes':planes,'pending':pending,
                'yoco_swa_valid_start':self.yoco_swa_valid_start[slot],
                'history':list(self.histories[slot].history) if self.histories else None,
                'dspark':self.dspark.snapshot(slot) if self.dspark is not None else None}

    def restore_prefix(self,snapshot,slot):
        if type(slot) is not int or not 0 <= slot < len(self.next_positions):
            raise ValueError('invalid prefix destination slot')
        expected_format = self.cache.index_format
        if snapshot.get('index_format','int8-fp16-v1') != expected_format:
            raise ValueError('prefix index format differs from active cache')
        from .official_yoco_metadata910c import snapshot_valid
        valid=snapshot_valid(snapshot.get('yoco_swa_valid_start',0),len(snapshot['ids']))
        # Reuse the slot-handoff contract for every restore entry, including
        # native prefix hits. No plane, pending state or history may be copied
        # before coverage, exact page counts and draft payloads are validated.
        from ...engine.zos910c.zos_slot_handoff910c import validate_snapshot
        validate_snapshot(self,snapshot)
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
        if self.dspark is not None:
            if snapshot.get('dspark') is None:
                raise ValueError('DSpark requires matching draft context in prefix snapshots')
            self.dspark.restore(snapshot['dspark'],slot)
        if getattr(self, '_original_state_history', None) is not None:
            self._original_state_history.clear_slot(slot)
        self.next_positions[slot]=len(snapshot['ids'])
        self.yoco_swa_valid_start[slot]=valid
        self.restored_prefix_slots.add(slot)

    def runtime_stats(self):
        frames={}
        for frame in [getattr(self,'_decode_graph',None),getattr(self,'_verify_graph',None)]+(
                list(self.dspark_graphs.frames.values()) if self.dspark_graphs is not None else []):
            if frame is not None:frames[id(frame)]=frame
        return {'eplb':self.eplb_runtime.stats() if self.eplb_runtime is not None else {'enabled':False},
                'sequence_positions': list(self.next_positions),
                'yoco_official_metadata_calls':self.yoco_official_metadata_calls,
                'yoco_official_fallback_calls':self.yoco_official_fallback_calls,
                'yoco_merged_qb_calls':{str(i):layer.yoco_merged_qb_calls for i,layer in enumerate(self.layers) if layer.yoco_merged_qb_calls},
                'yoco_swa_valid_start':list(self.yoco_swa_valid_start),
                'engram_uva_enabled':os.environ.get('DEEPINFER_910C_ENGRAM_UVA')=='1',
                'engram_device_hash_enabled':os.environ.get('DEEPINFER_910C_ENGRAM_DEVICE_HASH')=='1',
                'prefill_engram_uva_calls':self.prefill_engram_uva_calls,
                'prefill_engram_uva_tokens':self.prefill_engram_uva_tokens,
                'prefill_engram_uva_checked':self.prefill_engram_uva_checked,
                'prefill_device_hash_calls':self.prefill_device_hash_calls,
                'prefill_device_hash_tokens':self.prefill_device_hash_tokens,
                'prefill_device_hash_checked':self.prefill_device_hash_checked,
                'engram_device_hash_calls':sum(f.engram_hash_calls for f in frames.values()),
                'engram_device_hash_tokens':sum(f.engram_hash_tokens for f in frames.values()),
                'engram_device_hash_frame_calls':{str(size):sum(f.engram_hash_calls for f in frames.values() if f.tokens==size)
                                                  for size in sorted({f.tokens for f in frames.values()})},
                'engram_uva':(dict(self.engram_uva_runtime.stats)
                              if self.engram_uva_runtime is not None else None),
                'engram_prefetch_enabled':os.environ.get('DEEPINFER_910C_ENGRAM_PREFETCH')=='1',
                'engram_prefetch':(dict(self.engram_prefetch_runtime.stats)
                                  if self.engram_prefetch_runtime is not None else None),
                'engram_pinned_staging_enabled':os.environ.get('DEEPINFER_910C_ENGRAM_PINNED_STAGING')=='1',
                'engram_pinned_staging_calls':sum(getattr(f,'engram_pinned_calls',0) for f in frames.values()),
                'engram_pinned_staging_bytes':sum(getattr(f,'engram_pinned_bytes',0) for f in frames.values()),
                'native_eager_attention_calls': sum(layer.eager_attention_calls for layer in self.layers),
                'native_cann_attention_calls': sum(layer.cann_attention_calls for layer in self.layers),
                'fused_mhc_layers': sum(layer.fused_mhc for layer in self.layers),
                'fused_router_norm_layers': sum(layer.fused_router_norm for layer in self.layers),
                'mc2_layers': sum(layer.mc2 is not None for layer in self.layers),
                'fused_mc2_prefill_layers': sum(layer.fused_mc2_prefill for layer in self.layers),
                'fused_mc2_decode_layers': sum(layer.fused_mc2_decode for layer in self.layers),
                'fused_mc2_calls': sum(layer.mc2.fused_calls for layer in self.layers if layer.mc2 is not None),
                'fused_mc2_max_output_size': max((layer.mc2.fused_max_output_size for layer in self.layers
                                                if layer.mc2 is not None),default=0),
                'fused_mc2_capacity_floor':(self.layers[0].mc2.capacity_floor
                    if self.layers[0].mc2 is not None else None),
                'fused_mc2_capacity_records':([{'global_bs':key[0],'local_tokens':key[1],
                    'topk':key[2],'max_output_size':value}
                    for key,value in sorted(self.layers[0].mc2.capacity_records.items())]
                    if self.layers[0].mc2 is not None else []),
                'context_parallel_enabled': all(layer.context_parallel for layer in self.layers),
                'kv_token_parallel_enabled': all(layer.kv_token_parallel for layer in self.layers),
                'kv_token_parallel_calls': sum(layer.kv_token_parallel_calls for layer in self.layers),
                'prefill_kv_overlap_started': sum(layer.prefill_kv_overlap_started for layer in self.layers),
                'prefill_kv_overlap_completed': sum(layer.prefill_kv_overlap_completed for layer in self.layers),
                'tail_ffn_enabled': self.tail_ffn,
                'tail_ffn_source_layer': self.tail_ffn_source,
                'tail_ffn_calls': sum(layer.tail_ffn_calls for layer in self.layers),
                'tail_ffn_active_tokens': sum(layer.tail_ffn_tokens for layer in self.layers),
                'tail_ffn_skipped_tokens': sum(layer.tail_ffn_skipped for layer in self.layers),
                'context_parallel_scope': 'cached-compressed-query-and-MLA',
                'context_parallel_swa_preserve_tp_heads': True,
                'context_parallel_requires_prefix_restore': True,
                'context_parallel_calls': sum(layer.context_parallel_calls for layer in self.layers),
                'context_parallel_query_tokens': sum(layer.context_parallel_query_tokens for layer in self.layers),
                'fused_o_proj_enabled': all(layer.fused_o_proj for layer in self.layers),
                'fused_o_proj_prefill_calls': sum(layer.fused_o_proj_calls for layer in self.layers),
                'fused_o_proj_decode_layers':sum(layer.fused_o_proj_decode for layer in self.layers),
                'fused_o_proj_decode_calls':sum(layer.fused_o_proj_decode_calls for layer in self.layers),
                'sequence_parallel_enabled': self.sequence_parallel,
                'sequence_parallel_prefill_forwards': self.sequence_parallel_prefill_forwards,
                'prefill_padding_forwards':self.prefill_padding_forwards,
                'prefill_padding_tokens':self.prefill_padding_tokens,
                'prefill_wall_components':self.prefill_wall_components,
                'yoco_exact_compactions':self.yoco_exact_compactions,
                'yoco_exact_discarded_tokens':self.yoco_exact_discarded_tokens,
                'yoco_metadata_fusions':self.yoco_metadata_fusions,
                'yoco_preplan_prepared':self.yoco_preplan_prepared,
                'yoco_preplan_consumed':self.yoco_preplan_consumed,
                'yoco_bounded_compactions':self.yoco_bounded_compactions,
                'yoco_bounded_discarded_tokens':self.yoco_bounded_discarded_tokens,
                'yoco_bounded_swa_calls':sum(layer.bounded_swa_calls for layer in self.layers),
                'yoco_bounded_swa_packed_calls':sum(layer.bounded_swa_packed_calls for layer in self.layers),
                'sequence_parallel_tp_size': self.sp.world,
                'native_graph_captures': self._decode_graph.captures if getattr(self,'_decode_graph',None) is not None else 0,
                'native_graph_replays': self._decode_graph.replays if getattr(self,'_decode_graph',None) is not None else 0,
                'decode_host_staging_seconds': self._decode_graph.staging_seconds if getattr(self,'_decode_graph',None) is not None else 0.,
                'verification_graph_captures': self._verify_graph.captures if getattr(self,'_verify_graph',None) is not None else 0,
                'verification_graph_replays': self._verify_graph.replays if getattr(self,'_verify_graph',None) is not None else 0,
                'verification_host_staging_seconds': self._verify_graph.staging_seconds if getattr(self,'_verify_graph',None) is not None else 0.,
                'target_vocab_parallel_enabled':self.target_vocab_parallel,
                'target_vocab_shard_width':self.target_vocab_width,
                'target_vocab_total_size':self.cfg['vocab_size'],
                'target_vocab_parallel_top1_calls':self.target_vocab_top1_calls,
                'prompt_lookup':self.prompt_lookup_stats,
                'dspark':self.dspark_stats,
                'dspark_varlen_graph_replays':({str(k):v.replays for k,v in self.dspark_graphs.frames.items()}
                                               if self.dspark_graphs is not None else {}),
                'dspark_weights_loaded':self.dspark is not None,
                'dspark_draft_layers':len(self.dspark.layers) if self.dspark is not None else 0}

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
    def forward_tokens(self, chunks, slots=None, *, sequence_parallel=False, bounded_keep=0,
                       official_yoco=None, return_hidden_states=False, state_history=None,
                       input_metadata=None):
        forward_started=time.perf_counter()
        from deepinfer.npu_ops.rope910c import begin_forward
        begin_forward()
        chunks = [list(tokens) for tokens in chunks]
        if type(bounded_keep) is not int or bounded_keep not in (0,140,256):
            raise ValueError('unsupported bounded replay width')
        if return_hidden_states and bounded_keep:
            raise ValueError('original hidden-state output requires the complete retained query layout')
        slots = list(range(len(chunks))) if slots is None else list(slots)
        if (not chunks or len(chunks) != len(slots) or len(set(slots)) != len(slots)
                or any(type(s) is not int or not 0 <= s < self.requests for s in slots)
                or any(not 1 <= len(tokens) <= self.chunk_size for tokens in chunks)):
            raise ValueError('invalid packed requests or prefill chunk size')
        seq_lengths = [self.next_positions[s]+len(tokens) for tokens, s in zip(chunks, slots)]
        official_view = None
        if official_yoco is not None:
            if (bounded_keep or not sequence_parallel or not self.sequence_parallel
                    or not self.layers[0].context_parallel or self.sp.world!=8
                    or self.tail_ffn_source!=20 or any(layer.ratio!=1 for layer in self.layers[20:])
                    or any(int(i)>=20 for i in self.engrams)
                    or (self.dspark is not None and min(self.dspark.target_layer_ids)<20)):
                raise ValueError('unsupported official YOCO first-consumer configuration')
            from .official_consumer910c import ChunkView
            official_view = ChunkView.from_metadata(official_yoco,[self.next_positions[s] for s in slots],seq_lengths)
        if max(seq_lengths) > self.max_tokens:
            raise ValueError('packed sequence exceeds cache context capacity')
        if (not return_hidden_states and official_view is None and self.use_decode_graph and all(len(tokens)==1 for tokens in chunks)
                and slots==list(range(4)) and len(chunks)==4):
            if getattr(self,'_decode_graph',None) is None:
                from .w8a8_graph910c import NativeDecodeGraph
                self._decode_graph=NativeDecodeGraph(self,slots)
            return self._decode_graph.step(chunks)
        actual_lengths=tuple(map(len,chunks))
        actual_ends=list(seq_lengths)
        if self.eplb_runtime is not None:
            ends=self.eplb_runtime.prompt_ends;ends.fill_(-1)
            ends.index_copy_(0,torch.tensor(slots,dtype=torch.int64,device=self.device),
                             torch.tensor(actual_ends,dtype=torch.int32,device=self.device))
            ends[-1]=-1
        padding_margin=0
        if (not return_hidden_states and official_view is None and sequence_parallel and self.sequence_parallel and
                os.environ.get('DEEPINFER_910C_PREFILL_QUERY_PADDING')=='1'):
            # Request arrival order changes which DP replica owns the shorter
            # cached prefix. Equalize query shapes before the model, rather
            # than letting the first formal request initialize many new GEMM
            # shapes inside the layer loop. No output/prefix token is counted
            # for the additional, strictly future query positions.
            eligible=(len(chunks)==4 and slots==list(range(4)) and
                all(slot in self.restored_prefix_slots for slot in slots) and
                all(end%2==0 for end in actual_ends))
            values=torch.tensor([int(not eligible),max(actual_lengths),
                -min(actual_lengths),max(self.next_positions[s] for s in slots)],
                dtype=torch.int32,device=self.device)
            dist.all_reduce(values,op=dist.ReduceOp.MAX)
            reject,width,negative_minimum,latest_start=values.cpu().tolist()
            if not reject and width+negative_minimum>0 and latest_start+width<=self.max_tokens:
                padding_margin=width+negative_minimum
                noise=self.cfg['dspark_noise_token_id']
                chunks=[row+[noise]*(width-len(row)) for row in chunks]
                seq_lengths=[self.next_positions[s]+width for s in slots]
                self.prefill_padding_forwards+=1
                self.prefill_padding_tokens+=sum(width-length for length in actual_lengths)
        flat_ids = [token for tokens in chunks for token in tokens]
        from deepinfer.multimodal_tokens import ImageTokenOffset
        limit = self.embedding.shape[0] + (ImageTokenOffset.NUM_SPECIAL if input_metadata is not None else 0)
        if any(type(t) is not int or not 0 <= t < limit for t in flat_ids):
            raise ValueError('native paged entry requires original valid token IDs')
        hashes = {}
        prefill_device_hash=(os.environ.get('DEEPINFER_910C_PREFILL_DEVICE_HASH')=='1' and
            os.environ.get('DEEPINFER_910C_PREFILL_ENGRAM_UVA')=='1' and
            os.environ.get('DEEPINFER_910C_ENGRAM_UVA')=='1' and
            os.environ.get('DEEPINFER_910C_ENGRAM_PREFETCH')!='1' and
            self.engram_uva_runtime is not None and self.engram_device_hash is not None and
            sequence_parallel and self.sequence_parallel and self.layers[0].context_parallel and
            bool(self.histories) and all(slot in self.restored_prefix_slots for slot in slots))
        hash_check=os.environ.get('DEEPINFER_910C_PREFILL_DEVICE_HASH_CHECK')=='1'
        hash_mismatch=False
        if self.histories and prefill_device_hash:
            from deepinfer.npu_ops.engram_hash910c import raw_contexts
            histories=[self.histories[s] for s in slots]
            lengths=[len(history.history) for history in histories]
            expected=None
            try:
                if hash_check:
                    clones=[copy.copy(history) for history in histories]
                    for clone,history in zip(clones,histories):clone.history=list(history.history)
                    expected_parts=[history.batch(tokens) for history,tokens in zip(clones,chunks)]
                    expected={i:torch.cat([entry[i] for entry in expected_parts]) for i in self.cfg['engram_layer_ids']}
                contexts=raw_contexts(histories,chunks)
                raw=torch.tensor(contexts,dtype=torch.int32,device=self.device)
                hashes=self.engram_device_hash.compute(raw)
                self.prefill_device_hash_calls+=1
                self.prefill_device_hash_tokens+=len(contexts)
                if hash_check:
                    for layer in self.cfg['engram_layer_ids']:
                        expected_layer=expected[layer].to(device=self.device,dtype=torch.int64)
                        hash_mismatch |= not torch.equal(hashes[layer].to(torch.int64),expected_layer)
                        self.prefill_device_hash_checked+=1
            except BaseException:
                for history,length in zip(histories,lengths):del history.history[length:]
                raise
        elif self.histories:
            per_request = [self.histories[s].batch(tokens) for tokens, s in zip(chunks, slots)]
            hashes = {i: torch.cat([entry[i] for entry in per_request]) for i in self.cfg['engram_layer_ids']}
        if hash_check:
            failed=torch.tensor([int(hash_mismatch)],dtype=torch.int32,device=self.device)
            dist.all_reduce(failed,op=dist.ReduceOp.MAX)
            if bool(failed.item()):raise RuntimeError('prefill device hash differs from CPU hash on an EP rank')
        batch = DSABatch.build([len(tokens) for tokens in chunks], seq_lengths, self.device)
        state = PackedForward(batch, slots, batch.positions(), self.cache.swa_plan(0, batch, slots), self.cache.swa[0].table(slots))
        state.state_history = state_history
        state.valid_query_lengths=actual_lengths
        state.prefill_padding_margin=padding_margin
        if padding_margin:
            last_rows=[];offset=0
            for valid,chunk in zip(actual_lengths,chunks):
                last_rows.append(offset+valid-1);offset+=len(chunk)
            valid_last=torch.tensor(last_rows,dtype=torch.int64,device=self.device)
        state.source_metadata = self.cache.prepare_compressed(batch, slots)
        state.mc2_global_bs=getattr(self,'mc2_global_bs',4096)
        state.sequence_parallel=bool(sequence_parallel and self.sequence_parallel)
        if (state.sequence_parallel and self.layers[0].context_parallel
                and (official_view is not None or all(slot in self.restored_prefix_slots for slot in slots))):
            cp = QuerySlice.build(batch.query_lengths, batch.seq_lengths, self.sp.world, self.sp.rank)
            state.context_query = cp
            state.query_slots = [slots[i] for i in cp.request_indices]
            if cp.query_lengths:
                state.query_batch = DSABatch.build(cp.query_lengths, cp.seq_lengths, self.device)
                state.query_swa_table = self.cache.swa[0].table(state.query_slots)
        if not return_hidden_states and self.tail_ffn and state.context_query is not None:
            # Once the final long-KV source is written, later long attention
            # reads fixed caches. Only SWA expands token dependence, by W-1
            # per layer. After layer i the final query needs exactly the tail
            # 1+(L-1-i)*(W-1). Its last128 input rows also retain every layer's
            # exact SWA state for subsequent decode. Indexer sharing remains
            # query-local and reads the unchanged full long-KV source.
            cp=state.context_query
            state.tail_ffn_indices={}
            for i in range(self.tail_ffn_source,len(self.layers)):
                required=1+(len(self.layers)-1-i)*(self.cfg['sliding_window']-1)+padding_margin
                indices=[];offset=0
                for q in batch.query_lengths:
                    left=max(offset+max(0,q-required),cp.start)
                    right=min(offset+q,cp.end)
                    indices.extend(range(left-cp.start,max(left,right)-cp.start))
                    offset+=q
                state.tail_ffn_indices[i]=torch.tensor(indices,dtype=torch.int64,device=self.device)
        if bounded_keep and (not self.tail_ffn or state.context_query is None or padding_margin):
            raise RuntimeError('bounded replay lost globally agreed prefill eligibility')
        if bounded_keep and os.environ.get('DEEPINFER_910C_YOCO_PREPLAN')=='1':
            # Like official metadata preparation, construct query layout before
            # the producer layers. All graph and gather operations remain intact.
            from .yoco_exact_compact910c import prepare_compaction_plan
            state.yoco_prepared_plan=prepare_compaction_plan(self,state,bounded_keep)
            if state.yoco_prepared_plan is not None:
                self.yoco_preplan_prepared+=1
        ids = torch.tensor(flat_ids, dtype=torch.int64, device=self.device)
        if input_metadata is None:
            embedded = self.embedding.index_select(0,ids)
        else:
            from deepinfer.model_executor.vision import w8a8_input_embeddings
            if input_metadata.input_ids.numel() != len(flat_ids):
                raise ValueError('original VL embeddings differ from the packed token layout')
            embedded = w8a8_input_embeddings(self,input_metadata)
        trunk = embedded[:,None].expand(-1,self.cfg['hc_mult'],-1).contiguous()
        previous = torch.zeros((len(flat_ids), self.cfg['hc_mult'], 1), dtype=torch.float32, device=self.device)
        previous[:, 0] = 1.
        prefill_uva=(os.environ.get('DEEPINFER_910C_PREFILL_ENGRAM_UVA')=='1' and
            state.sequence_parallel and state.context_query is not None and bool(hashes) and
            self.engram_uva_runtime is not None and os.environ.get('DEEPINFER_910C_ENGRAM_UVA')=='1' and
            os.environ.get('DEEPINFER_910C_ENGRAM_PREFETCH')!='1')
        full_hashes=hashes if prefill_uva else None
        if state.sequence_parallel:
            trunk=self.sp.split(trunk)
            previous=self.sp.split(previous)
            hashes={i:self.sp.split(value,fill=-1) for i,value in hashes.items()}
            self.sequence_parallel_prefill_forwards+=1
        state.bind_row_controls(self.sp,
            None if input_metadata is None else input_metadata.images_seq_mask,
            None if input_metadata is None else input_metadata.gate_force_random)
        draft_features=[]
        if self.dspark is not None:
            self.dspark.prepare_capture(state,self.sp)
        engram_fetch=None
        if hashes and os.environ.get('DEEPINFER_910C_ENGRAM_PREFETCH')=='1':
            if self.engram_prefetch_runtime is None:
                from .engram_prefetch910c import EngramPrefetch
                self.engram_prefetch_runtime=EngramPrefetch(self.reader,self.device)
            engram_fetch=self.engram_prefetch_runtime.begin(hashes)
        observe=bool(state.sequence_parallel and state.context_query is not None)
        split_timing=observe and os.environ.get('DEEPINFER_910C_PREFILL_TIMING_SPLIT')=='1'
        self._prefill_compaction_detail=None
        layer_times=[]
        before_layers=time.perf_counter()
        for i, layer in enumerate(self.layers):
            layer_start=time.perf_counter()
            if str(i) in self.engrams:
                if prefill_uva:
                    rows=self.engram_uva_runtime.prefill_sp_rows(i,full_hashes[i],self.sp)
                    valid=(hashes[i]>=0).all(-1).to(self.device)
                    if rows.shape[0]!=trunk.shape[0]:raise ValueError('prefill UVA SP row count mismatch')
                    trunk=self.engrams[str(i)].forward_rows(trunk,rows,self.engram_rotation,valid)
                    self.prefill_engram_uva_calls+=1
                    self.prefill_engram_uva_tokens+=rows.shape[0]
                    if os.environ.get('DEEPINFER_910C_PREFILL_ENGRAM_UVA_CHECK')=='1':
                        reference=self.reader.engram_rows(i,hashes[i],self.device).reshape_as(rows)
                        mismatch=not torch.equal(rows.view(torch.int16),reference.view(torch.int16))
                        self.prefill_engram_uva_checked+=1
                    else:mismatch=False
                elif engram_fetch is None:
                    trunk = self.engrams[str(i)](trunk, hashes[i], self.engram_rotation)
                else:
                    rows,valid=engram_fetch.get(i)
                    trunk=self.engrams[str(i)].forward_rows(trunk,rows,self.engram_rotation,valid)
                if os.environ.get('DEEPINFER_910C_PREFILL_ENGRAM_UVA_CHECK')=='1':
                    failed=torch.tensor([int(prefill_uva and mismatch)],dtype=torch.int32,device=self.device)
                    dist.all_reduce(failed,op=dist.ReduceOp.MAX)
                    if bool(failed.item()):raise RuntimeError('prefill UVA differs from original Engram rows on an EP rank')
            after_engram=time.perf_counter()
            if self.dspark is not None and i in self.dspark.target_layer_ids:
                draft_features.append(self.dspark.capture(trunk,state))
            after_capture=time.perf_counter()
            if official_view is not None and i==20:
                from .official_source20_910c import prestore_full_source
                from .official_consumer910c import enter
                prestore_full_source(self,layer,state,trunk,previous)
                before = sum(state.batch.query_lengths)
                trunk,previous = enter(self,state,trunk,previous,official_view)
                batch = state.batch
                self.yoco_bounded_compactions += 1
                self.yoco_bounded_discarded_tokens += before-sum(batch.query_lengths)
            trunk, previous = layer(trunk, previous, cache=self.cache, pending=self.pending, state=state)
            if _layer21_probe is not None and i==20:
                _layer21_probe('layer20_output','local',trunk,previous,state=state)
            if i in _late_consumer_probes:
                _late_consumer_probes[i]('output','local',trunk,previous,state=state)
            if i+1 in _late_consumer_probes:
                _late_consumer_probes[i+1]('previous_output','local',trunk,previous,state=state)
            if _consumer_end_probe is not None and state.official_consumer is not None:
                _consumer_end_probe(i, trunk, previous, state)
            after_layer = time.perf_counter() if split_timing else None
            if (official_view is None and (os.environ.get('DEEPINFER_910C_YOCO_EXACT_COMPACT') == '1' or bounded_keep)
                    and i == self.tail_ffn_source and self.tail_ffn
                    and state.context_query is not None and not padding_margin):
                from .yoco_exact_compact910c import compact
                before = sum(state.batch.query_lengths)
                compacted = compact(self, state, trunk, previous, bounded_keep=bounded_keep or None)
                if compacted is not None:
                    trunk, previous, state = compacted
                    batch = state.batch
                    if bounded_keep:
                        self.yoco_bounded_compactions += 1
                        self.yoco_bounded_discarded_tokens += before - sum(batch.query_lengths)
                    else:
                        self.yoco_exact_compactions += 1
                        self.yoco_exact_discarded_tokens += before - sum(batch.query_lengths)
            if observe:
                after_compaction = time.perf_counter()
                layer_record = {'layer':i,'engram_wall_seconds':after_engram-layer_start,
                    'feature_wall_seconds':after_capture-after_engram,
                    'layer_enqueue_wall_seconds':after_compaction-after_capture}
                if split_timing:
                    layer_record['layer_forward_host_seconds']=after_layer-after_capture
                    layer_record['post_layer_host_seconds']=after_compaction-after_layer
                layer_times.append(layer_record)
        if engram_fetch is not None:engram_fetch.finish()
        before_draft_context=time.perf_counter()
        if self.dspark is not None:
            consumer = state.official_consumer
            if consumer is None:
                self.dspark.update_main(draft_features,state.dspark_positions,state.dspark_requests,observer=self)
            elif consumer.store_plan is not None:
                self.dspark.update_main(draft_features,state.dspark_positions,state.dspark_requests,
                                        store_rows=consumer.store_plan[0],observer=self)
        before_head=time.perf_counter()
        # Prefill only needs each request's final logit, not a vocabulary-sized
        # output tensor for every prompt token.
        last = valid_last if padding_margin else batch.cu_query[1:].long()-1
        if state.sequence_parallel:
            local_hidden=(trunk.float()*previous).sum(-2).bfloat16()
            hidden=self.sp.gather(local_hidden,sum(state.batch.query_lengths))
            if not return_hidden_states:
                hidden=hidden.index_select(0,last)
        else:
            if return_hidden_states:
                hidden = (trunk.float()*previous).sum(-2).bfloat16()
            else:
                hidden = (trunk.index_select(0, last).float()*previous.index_select(0, last)).sum(-2).bfloat16()
        hidden = rms_norm_mixed(hidden, self.norm, self.cfg['rms_norm_eps'])[0]
        logits = None if return_hidden_states else project(hidden, self.head).float()
        after_head=time.perf_counter()
        if observe:
            self.prefill_wall_components={'scope':'CPU wall spans including implicit waits, not exclusive NPU kernel times',
                'preparation_seconds':before_layers-forward_started,'layers':layer_times,
                'draft_context_enqueue_seconds':before_head-before_draft_context,
                'head_enqueue_seconds':after_head-before_head,
                'total_enqueue_seconds':after_head-forward_started}
            if self._prefill_compaction_detail is not None:
                self.prefill_wall_components['compaction_detail']=self._prefill_compaction_detail
        for row,(slot,end) in enumerate(zip(slots,actual_ends,strict=True)):
            previous_valid=self.yoco_swa_valid_start[slot]
            if official_yoco is not None:
                from .official_yoco_metadata910c import valid_after
                self.yoco_swa_valid_start[slot]=valid_after(official_yoco[row],previous_valid)
            else:
                self.yoco_swa_valid_start[slot]=max(previous_valid,end-self.cfg['sliding_window'],0)
            self.next_positions[slot] = end
            if padding_margin:
                if self.histories:del self.histories[slot].history[end:]
                # Eligibility requires an even committed boundary. Future C2
                # pairs need no retained activation at that exact boundary.
                for pending_slots in self.pending.values():pending_slots.pop(slot,None)
        if return_hidden_states:
            from deepinfer.model_executor.model_output import CausalLMForwardOutput

            return CausalLMForwardOutput(hidden_states=hidden)
        return logits

    def resident_bytes(self):
        result = {}
        for tensor in list(self.parameters())+list(self.buffers()):
            dtype = str(tensor.dtype)
            result[dtype] = result.get(dtype, 0)+tensor.numel()*tensor.element_size()
        result['paged_cache_payload'] = self.cache.resident_bytes()
        return result
