# SPDX-License-Identifier: Apache-2.0
"""910C W8A8 backend for the upstream DeepInfer DSpark computation.

The checkpoint's three mtp blocks, target attention-input features, main
projection/cache, noncausal five-position draft attention, Markov correction
and confidence head follow models/mtp/dflash.py. Only device execution and
ModelSlim name/INT8 layout adapters are different. No text lookup proposer.
"""
import copy
import math
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from deepinfer import envs
from deepinfer.sampling_params import SAMPLING_EPS
from deepinfer.npu_ops.dsa910c import DSABatch
from deepinfer.npu_ops.sp910c import ordered_sum
from .ascend_w8a8_packed import AscendW8A8PackedLayer, PackedForward
from .w8a8_math910c import project, rms_norm_mixed, rotate_positions
from .w8a8_paged_cache910c import PagedPlane, CacheWritePlan


class DraftCheckpoint:
    """Map the validated block loader to mtp.i without rewriting weights."""
    def __init__(self, reader):
        self.reader = reader
        self.config = copy.deepcopy(reader.config)
        c = self.config['text_config']
        c['n_routed_experts'] = c['dspark_n_routed_experts']
        c['num_experts_per_tok'] = c['dspark_num_experts_per_tok']

    def name(self, name):
        if name.startswith('layers.'):
            _, number, rest = name.split('.', 2)
            return f"mtp.{int(number)-self.config['text_config']['num_hidden_layers']}.{rest}"
        return name

    def tensor(self, name):
        return self.reader.tensor(self.name(name))

    def linear_payload(self, name, **kwargs):
        return self.reader.linear_payload(self.name(name), **kwargs)

    def load_linear(self, name, device, **kwargs):
        return self.reader.load_linear(self.name(name), device, **kwargs)


def noncausal_draft_attention(query, plane, state, sink):
    """Pinned A3 sparse-MLA ABI, using explicit logical noncausal indices.

    All five positions see the same preceding 128 main keys and all five
    temporary draft keys, as in upstream get_fused_dflash_get_input_ref.
    """
    batch = state.batch
    indices, lengths = state.dspark_indices, state.dspark_lengths
    if envs.DEEPINFER_BATCH_INVARIANT:
        # Preserve the original explicit NONCAUSAL visibility, including all
        # draft positions. Only translate its logical rows to physical pages.
        from deepinfer.npu_ops.attention910c import decode, ReferenceMLASchedMeta
        request_ids = batch.metadata_cache.get('draft_request_ids')
        if request_ids is None:
            rows = torch.arange(query.shape[0],device=query.device,dtype=torch.int32)
            request_ids = (rows[:,None]>=batch.cu_query[None,1:]).sum(-1)
            batch.metadata_cache['draft_request_ids'] = request_ids
        logical = indices[:,0,:]
        pages = torch.div(logical.clamp_min(0),plane.block_size,rounding_mode='floor')
        valid = (logical>=0) & (pages<state.swa_table.shape[1])
        blocks = state.swa_table[request_ids[:,None],pages.clamp(0,state.swa_table.shape[1]-1).long()]
        physical = torch.where(valid,blocks*plane.block_size+logical.remainder(plane.block_size),-1).int()
        return decode(query,plane.cache,query.shape[-1],physical,ReferenceMLASchedMeta(),
            query.shape[-1]**-.5,sink,batch_invariant=True,index_length=lengths[:,0],
            _trusted_indices=True)[0]
    common = dict(cu_seqlens_q=batch.cu_query, seqused_ori_kv=batch.seq_used,
                  seqused_cmp_kv=None, cmp_residual_kv=None, cmp_ratio=0,
                  ori_mask_mode=0, cmp_mask_mode=0, ori_win_left=127,
                  ori_win_right=0, layout_q='TND', layout_kv='PA_BBND')
    key = ('dspark-noncausal', query.shape[1])
    metadata = batch.metadata_cache.get(key)
    if metadata is None:
        metadata = torch.ops._C_ascend.npu_sparse_flash_mla_metadata(
            query.shape[1], 1, 512, batch_size=len(batch.query_lengths),
            max_seqlen_q=max(batch.query_lengths),
            max_seqlen_ori_kv=max(batch.seq_lengths), max_seqlen_cmp_kv=0,
            ori_topk=indices.shape[-1], ori_topk_length=lengths, cmp_topk=0,
            has_ori_kv=True, has_cmp_kv=False, **common)
        batch.metadata_cache[key] = metadata
    result, _ = torch.ops._C_ascend.npu_sparse_flash_mla(
        query, ori_kv=plane.cache, cmp_kv=None,
        ori_sparse_indices=indices, ori_topk_length=lengths,
        ori_block_table=state.swa_table, cmp_block_table=None,
        sinks=sink, metadata=metadata, softmax_scale=512**-.5,
        topk_value_mode=1, return_softmax_lse=False, **common)
    return result


class DSparkDraftLayer(AscendW8A8PackedLayer):
    def packed_attention(self, x, cache, pending, state, *, reduce_results=True,
                         input_is_sharded=False):
        c = self.config
        qa = rms_norm_mixed(self.projections['wq_a'](x), self.attn_q_norm_weight,
                            c['rms_norm_eps'])[0]
        q = self.projections['wq_b'](qa).reshape(-1, self.local_heads, c['head_dim'])
        q = rotate_positions(q, state.positions, self.rope_freq)
        kv = rms_norm_mixed(self.projections['wkv'](x), self.attn_kv_norm_weight,
                            c['rms_norm_eps'])[0]
        kv = rotate_positions(kv, state.positions, self.rope_freq)
        cache.write(kv[:, None], state.swa_plan)
        context = noncausal_draft_attention(q, cache, state, self.attn_attn_sink)
        context = rotate_positions(context, state.positions, self.rope_freq, inverse=True)
        grouped = context.reshape(x.shape[0], self.local_o_groups, -1)
        if envs.DEEPINFER_BATCH_INVARIANT:
            projected = torch.stack([project(grouped[:,g,:],self.wv[g].to(grouped.dtype))
                                     for g in range(grouped.shape[1])],1)
        elif self.fused_o_proj and self.fused_o_proj_decode:
            import torch_npu
            projected=torch_npu.npu_transpose_batchmatmul(grouped,self.wv_bmm,
                bias=None,scale=None,perm_x1=(1,0,2),perm_x2=(0,1,2),
                perm_y=(1,0,2),batch_split_factor=1)
        else:
            projected = torch.einsum('bgr,gdr->bgd', grouped.float(), self.wv.float()).bfloat16()
        output = self.projections['wo_b'](projected.flatten(1))
        if self.tp_size > 1:
            if envs.DEEPINFER_BATCH_INVARIANT:
                output = ordered_sum(output,self.tp_group)
            else:
                dist.all_reduce(output, group=self.tp_group)
        return output


class DSpark910C(torch.nn.Module):
    def __init__(self, reader, device, *, requests, max_tokens, group=None,
                 tp_group=None, progress=None):
        super().__init__()
        self.reader, self.device = reader, torch.device(device)
        self.cfg = c = reader.config['text_config']
        self.tp_group = tp_group
        self.tp_size = dist.get_world_size(tp_group)
        self.tp_rank = dist.get_rank(tp_group)
        self.nextn = envs.DEEPINFER_USE_NEXTN_DECODE
        if self.nextn < 1:
            raise ValueError('draft resources require a positive original nextn setting')
        # Original DFlash uses runtime nextn for the noncausal draft block;
        # checkpoint block size is not the number of loaded transformer layers.
        self.sparse_width = math.ceil((c['sliding_window'] + self.nextn) / 64) * 64
        from deepinfer.worker.cuda_graph_runner import build_graph_batch_sizes
        sizes = build_graph_batch_sizes(max(1, requests - 1), envs.DEEPINFER_DFLASH_CUDAGRAPH_BATCH_SIZE_INTERVAL)
        # Rounded graph tails share only the private scratch slot. Keep their
        # non-overlapping temporary query spans inside its resident ring.
        max_padding = max(1, max(size - previous - 1 for previous, size in zip([0] + sizes[:-1], sizes)))
        padding_span = max_padding * self.nextn
        draft_pages = max(3, math.ceil((c['sliding_window'] + padding_span + 127) / 128))
        self.target_layer_ids = tuple(c['dspark_target_layer_ids'])
        if self.target_layer_ids != (37, 38, 39):
            raise ValueError('unexpected target attention-input feature contract')
        self.max_tokens, self.requests = max_tokens, requests
        # A final target row still launches the runtime-size draft block. Its
        # unused proposals may extend beyond the target admission limit.
        # Virtual coordinates include draft slack; resident pages also cover
        # larger configured blocks without aliasing their visible keys.
        self.max_draft_tokens = max(max_tokens + self.nextn, padding_span)
        self.graphs = {}
        self.slot_requests = {}
        self.proposals = 0
        self.context_updates = 0
        temperature = envs.DEEPINFER_DFLASH_CONFIDENCE_HEAD_TEMPERATURE
        temperatures = temperature if isinstance(temperature, tuple) else (temperature,) * self.nextn
        self.temperatures = tuple(max(value, SAMPLING_EPS) for value in temperatures)
        self.reset_stats()
        adapted = DraftCheckpoint(reader)
        self.layers = torch.nn.ModuleList()
        self.planes = []
        for i in range(c['num_nextn_predict_layers']):
            layer = DSparkDraftLayer(adapted, device,
                layer_id=c['num_hidden_layers']+i, group=group, tp_group=tp_group)
            layer.load_local_experts()
            self.layers.append(layer)
            self.planes.append(PagedPlane(requests=requests, logical_rows=self.max_draft_tokens,
                block_size=128, width=512, dtype=torch.bfloat16,
                device=device, resident_pages=draft_pages))
            if progress:
                progress({'stage':'resident_dspark_layer', 'layer':i,
                          'local_experts':layer.local_expert_count, 'quantization':'W8A8_DYNAMIC'})
        self.main_proj = reader.load_linear('mtp.0.main_proj', device)
        self.register_buffer('main_norm', reader.tensor('mtp.0.main_norm.weight')[0].to(device))
        self.register_buffer('norm', reader.tensor('mtp.2.norm.weight')[0].to(device))
        self.register_buffer('markov_embed', reader.materialize('mtp.2.markov_head.embed.weight',device))
        self.register_buffer('confidence_weight', reader.tensor('mtp.2.confidence_head.proj.weight')[0].to(device).bfloat16())
        vocabulary = c['vocab_size']
        if vocabulary % self.tp_size:
            raise ValueError('DSpark vocabulary must divide tensor parallel size')
        self.vocab_width = vocabulary//self.tp_size
        self.vocab_first = self.tp_rank*self.vocab_width
        first, end = self.vocab_first, self.vocab_first+self.vocab_width
        # Runtime parameters are BF16 in upstream. Accumulate output logits in
        # FP32, including the low-rank Markov add, before choosing a token.
        self.register_buffer('embedding',reader.row_slice('mtp.0.embed.weight',first,end)[0].to(device).bfloat16())
        self.register_buffer('head',reader.row_slice('mtp.2.head.weight',first,end)[0].bfloat16().float().to(device))
        self.register_buffer('markov_head',reader.row_slice('mtp.2.markov_head.head.weight',first,end)[0].float().to(device))

    def reset_stats(self):
        self.proposals = self.context_updates = 0

    def snapshot(self, slot):
        return [p.cache[slot*p.resident_pages:(slot+1)*p.resident_pages].clone()
                for p in self.planes]

    def restore(self, values, slot):
        from ...engine.zos910c.zos_slot_handoff910c import validate_draft_snapshot
        if type(slot) is not int or any(not 0 <= slot < plane.requests for plane in self.planes):
            raise ValueError('invalid draft destination slot')
        validate_draft_snapshot(self,values)
        for plane, value in zip(self.planes, values, strict=True):
            plane.cache[slot*plane.resident_pages:(slot+1)*plane.resident_pages].copy_(value)

    def prepare_capture(self, state, sp):
        rows, positions, requests = [], [], []
        offset = 0
        valid_lengths=state.valid_query_lengths or state.batch.query_lengths
        for slot,q,valid,end in zip(state.slots,state.batch.query_lengths,valid_lengths,state.batch.seq_lengths):
            take = valid if state.official_consumer is not None else min(valid,self.cfg['sliding_window'])
            rows.extend(range(offset+valid-take,offset+valid))
            positions.extend(range(end-q+valid-take,end-q+valid))
            requests.extend([slot]*take)
            offset += q
        state.dspark_positions = torch.tensor(positions,dtype=torch.int32,device=self.device)
        state.dspark_requests = torch.tensor(requests,dtype=torch.int32,device=self.device)
        if state.sequence_parallel:
            local_width = math.ceil(offset/sp.world)
            local_rows = torch.tensor(rows,dtype=torch.int64,device=self.device)-sp.rank*local_width
            state.dspark_local_rows = local_rows.clamp(0,local_width-1)
            state.dspark_own_rows = ((local_rows>=0)&(local_rows<local_width))[:,None]
        else:
            state.dspark_local_rows = torch.tensor(rows,dtype=torch.int64,device=self.device)
            state.dspark_own_rows = None

    def capture(self, trunk, state):
        # Upstream dflash_input_store averages the full HC trunk immediately
        # before the target layer's attention, before its norm/pre-mix.
        value = trunk.index_select(0,state.dspark_local_rows).float().mean(-2).bfloat16()
        if state.dspark_own_rows is not None:
            value = torch.where(state.dspark_own_rows,value,0.)
            dist.all_reduce(value,group=self.tp_group)
        return value

    def update_main(self, features, positions, requests, *, store_rows=None,observer=None):
        hidden = rms_norm_mixed(self.main_proj(torch.cat(features,-1)),self.main_norm,
                                self.cfg['rms_norm_eps'])[0]
        observed_keys=[] if observer is not None and os.environ.get('DEEPINFER_910C_INVARIANT_TRACE_DIR') else None
        for layer,plane in zip(self.layers,self.planes):
            kv = rms_norm_mixed(layer.projections['wkv'](hidden),layer.attn_kv_norm_weight,
                                self.cfg['rms_norm_eps'])[0]
            kv = rotate_positions(kv,positions,layer.rope_freq)
            if store_rows is None:
                physical = plane.physical_indices(positions[:,None],requests).squeeze(-1).long()
            else:
                pos = positions.index_select(0,store_rows)
                req = requests.index_select(0,store_rows)
                physical = plane.physical_indices(pos[:,None],req).squeeze(-1).long()
                kv = kv.index_select(0,store_rows)
            plane.write(kv[:,None],CacheWritePlan(physical,len(physical)))
            if observed_keys is not None:observed_keys.append(kv)
        if observed_keys is not None:
            from .decode_trace910c import main_update
            main_update(observer,features,hidden,positions,requests,observed_keys)

    def forward_hidden(self, golden, starts, slots, *, mc2_global_bs=None, active_requests=None):
        """Run the resident draft layers without imposing a sampling policy."""
        from deepinfer.npu_ops.rope910c import begin_forward
        begin_forward()
        count, n = len(slots), self.nextn
        if isinstance(slots, torch.Tensor):
            if slots.dtype != torch.int32 or slots.device != self.device or slots.ndim != 1:
                raise ValueError('draft graph slots require a native int32 device vector')
            reqs = slots.repeat_interleave(n)
            table = self.planes[0].block_table.index_select(0, slots.long())
        else:
            key = tuple(slots)
            if key not in self.slot_requests:
                self.slot_requests[key] = torch.tensor(slots, dtype=torch.int32,
                    device=self.device).repeat_interleave(n)
            reqs = self.slot_requests[key]
            table = self.planes[0].table(slots)
        offsets = torch.arange(n,dtype=torch.int32,device=self.device)
        positions = (starts[:,None]+offsets).flatten()
        ids = torch.full((count,n),self.cfg['dspark_noise_token_id'],dtype=torch.int64,device=self.device)
        ids[:,0] = golden
        local = ids.flatten()-self.vocab_first
        trunk = self.embedding.index_select(0,local.clamp(0,self.vocab_width-1))
        trunk = torch.where(((local>=0)&(local<self.vocab_width))[:,None],trunk,0.)
        dist.all_reduce(trunk,group=self.tp_group)
        trunk = trunk[:,None].expand(-1,self.cfg['hc_mult'],-1).contiguous()
        previous = torch.zeros((count*n,self.cfg['hc_mult'],1),dtype=torch.float32,device=self.device)
        previous[:,0] = 1.
        batch = DSABatch((n,)*count,(self.max_draft_tokens,)*count,
            torch.arange(count+1,dtype=torch.int32,device=self.device)*n,starts+n,
            position_values=positions)
        plan = CacheWritePlan(self.planes[0].physical_indices(positions[:,None],reqs).squeeze(-1).long(),count*n)
        state = PackedForward(batch,slots,positions,plan,table,
            mc2_global_bs=(dist.get_world_size()*math.ceil(count*n/self.tp_size)
                          if mc2_global_bs is None else mc2_global_bs))
        if active_requests is not None:
            if active_requests.shape != (count,) or active_requests.dtype != torch.bool or active_requests.device != self.device:
                raise ValueError('draft graph activity differs from physical request slots')
            state.eplb_rows = active_requests.repeat_interleave(n)
        elif hasattr(self.layers[0],"eplb_layer_state"):
            # propose() creates reqs before capture, already repeated per draft row.
            state.eplb_rows=self.layers[0].eplb_query_lengths.index_select(0,reqs.long())>0
        columns = torch.arange(self.sparse_width,dtype=torch.int32,device=self.device)
        left = (starts-self.cfg['sliding_window']).clamp_min(0)
        visible = starts+n-left
        indices = left[:,None]+columns
        indices = torch.where(columns[None]<visible[:,None],indices,-1)
        state.dspark_indices = indices[:,None,:].expand(-1,n,-1).reshape(count*n,1,self.sparse_width).contiguous()
        state.dspark_lengths = visible[:,None,None].expand(-1,n,1).reshape(count*n,1).contiguous()
        for layer,plane in zip(self.layers,self.planes):
            trunk,previous = layer(trunk,previous,cache=plane,pending=None,state=state)
        return (trunk.float()*previous).sum(-2).bfloat16().reshape(count,n,-1)

    def _forward(self, golden, starts, slots):
        count, n = len(slots), self.nextn
        hidden = self.forward_hidden(golden, starts, slots)
        normed = rms_norm_mixed(hidden,self.norm,self.cfg['rms_norm_eps'])[0]
        base_logits = project(normed.float(),self.head)
        prior = golden
        tokens,confidences = [],[]
        weight = self.confidence_weight.float().flatten()
        for step in range(n):
            markov = self.markov_embed.index_select(0,prior)
            logits = project(markov.float(),self.markov_head)+base_logits[:,step]
            score,token = logits.max(-1)
            local_best = torch.stack((score,token.float()+self.vocab_first),-1)
            gathered = local_best.new_empty((self.tp_size*count,2))
            dist.all_gather_into_tensor(gathered,local_best.contiguous(),group=self.tp_group)
            gathered = gathered.reshape(self.tp_size,count,2)
            winner = gathered[:,:,0].argmax(0)
            prior = gathered[winner,torch.arange(count,device=self.device),1].long()
            confidence_input = torch.cat((hidden[:,step],markov),-1).float()
            raw = (confidence_input*weight).sum(-1)/self.temperatures[step]
            confidences.append(torch.sigmoid(raw.clamp(-60,60)))
            tokens.append(prior)
        return torch.stack(tokens,1),torch.stack(confidences,1)

    @torch.inference_mode()
    def propose(self, golden, starts, slots):
        key = tuple(slots)
        if key not in self.slot_requests:
            self.slot_requests[key] = torch.tensor(slots,dtype=torch.int32,
                device=self.device).repeat_interleave(self.nextn)
        cached = self.graphs.get(key)
        device_inputs = isinstance(golden, torch.Tensor) or isinstance(starts, torch.Tensor)
        if device_inputs:
            if not (isinstance(golden, torch.Tensor) and isinstance(starts, torch.Tensor)):
                raise TypeError('draft golden and positions must both be tensors or both be host lists')
            for value, dtype, label in ((golden, torch.int32, 'golden'), (starts, torch.int32, 'positions')):
                if (value.device != self.device or value.dtype != dtype or
                        tuple(value.shape) != (len(slots),) or not value.is_contiguous()):
                    raise ValueError('invalid device draft '+label)
            # The graph owns its persistent inputs. Never capture a state buffer,
            # which may be swapped and reused by a later device transition.
            gold = golden.to(dtype=torch.int64).clone() if cached is None else golden
            position = starts.clone() if cached is None else starts
        else:
            gold = torch.tensor(golden,dtype=torch.int64,device=self.device)
            position = torch.tensor(starts,dtype=torch.int32,device=self.device)
        if cached is None:
            for _ in range(2):
                self._forward(gold,position,slots)
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                output = self._forward(gold,position,slots)
            cached = (graph,gold,position,output)
            self.graphs[key] = cached
        graph,input_ids,input_positions,output = cached
        input_ids.copy_(gold)
        input_positions.copy_(position)
        graph.replay()
        self.proposals += 1
        return output
