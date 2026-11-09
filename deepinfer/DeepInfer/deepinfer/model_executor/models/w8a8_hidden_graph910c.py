"""Normalized-hidden NPU graphs behind original Worker and Sampler interfaces.

Uses the existing transformer graph and the original graph-size selection and
profiling implementation. Only input layout, owned outputs and backend resource
lifetime are adapted; Scheduler and sampling policy remain in the original code.
"""
from bisect import bisect_left
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from deepinfer import envs
from deepinfer.model_executor.model_output import CausalLMForwardOutput
from .dspark_varlen910c import DSparkVarlenGraph
from .hidden_graph_layout910c import plan_hidden_rows, validate_hidden_capacity


@dataclass
class DeviceHiddenLayout:
    lengths: torch.Tensor
    starts: torch.Tensor
    ends: torch.Tensor
    output_rows: torch.Tensor


class OriginalHiddenGraph(DSparkVarlenGraph):
    def __init__(self, model, capacity, gate_force_random, *, eager=False):
        validate_hidden_capacity(capacity, model.max_tokens)
        # Incomplete C2 rows use [B,2B); completed scratch pairs use [0,B/2).
        for source, plane in model.cache.compressed.items():
            if model.cfg['compress_ratios'][source] == 2:
                validate_hidden_capacity(capacity, model.max_tokens, c2_logical_rows=plane.logical_rows)
        super().__init__(model, capacity, return_hidden_states=True)
        self.gate_force_random = gate_force_random
        self.eager = eager
        self.eager_calls = 0
        self.input_copy_done = None
        # Original force-random routes are deterministic per rank/local row.
        # Prepare them once, keeping Philox and insertion out of normal replay.
        from deepinfer.npu_ops.profile_routing910c import make_random_routes
        self.profile_routes = {}
        shared_routes = {}
        for layer in model.layers:
            width = math.ceil(capacity / layer.tp_size)
            rank = dist.get_rank(layer.tp_group) if layer.tp_size > 1 else 0
            rows = max(1, min(width, capacity - rank * width))
            experts = layer.eplb_layout[1] if hasattr(layer, 'eplb_layout') else layer.config['n_routed_experts']
            key = (rows, layer.config['num_experts_per_tok'], dist.get_rank(layer.group), experts)
            if key not in shared_routes:
                shared_routes[key] = make_random_routes(rows, key[1], device=model.device,
                                                       ep_rank=key[2], num_experts=experts)
            self.profile_routes[layer.layer_id] = shared_routes[key]

    def plan(self, chunks, slots):
        return plan_hidden_rows(chunks, slots, self.model.next_positions,
            requests=self.model.requests, capacity=self.capacity, max_tokens=self.model.max_tokens,
            query_limit=1 + (0 if self.model.dspark is None else self.model.dspark.nextn),
            vocab_size=self.model.cfg['vocab_size'], noise_token=self.model.cfg['dspark_noise_token_id'])

    def stage(self, layout):
        model = self.model
        self.valid.fill_(True)
        if self.input_copy_done is not None:
            self.input_copy_done.synchronize()
        self.cpu_inputs.copy_(torch.tensor(layout.values, dtype=torch.int32))
        self.inputs.copy_(self.cpu_inputs, non_blocking=True)
        scratch = model.requests - 1
        for pending in model.pending.values():
            pending.pop(scratch, None)
        if model.histories:
            model.histories[scratch].reset()
            active = [(model.histories[slot], chunk) for slot, chunk in enumerate(layout.chunks) if chunk]
            if self.rows:
                self._stage_engram([history for history, _ in active], [list(chunk) for _, chunk in active])
            else:
                for history, chunk in active:
                    history.history.extend(chunk)
        # Every replay, including a different graph size, starts from the
        # original committed C2 pair rather than another frame's last seed.
        self._prepare_sequence()
        self.input_copy_done = torch.npu.Event()
        self.input_copy_done.record()
        self.host_lengths, self.host_starts, self.host_cu = layout.lengths, layout.starts, layout.cu_query

    def stage_original(self, inputs, history):
        """Map original device metadata to the existing physical graph layout.

        State IDs and token order come from the original C++/ZOS producer.
        Sorting only restores this backend's physical state order; output_rows
        restores the original request/token order for the original Sampler.
        """
        model, count = self.model, inputs.input_ids.numel()
        if (inputs.is_prompt or not 1 <= count <= self.capacity
                or inputs.positions.shape != (count,)
                or inputs.state_block_idx.shape != (count,)
                or inputs.valid_token_mask.shape != (count,)):
            raise ValueError('original decode metadata exceeds the selected graph')
        device = model.device
        scratch = model.requests - 1
        pad = self.capacity - count
        slot = torch.where(inputs.state_block_idx > 0, inputs.state_block_idx - 1, scratch)
        valid = inputs.valid_token_mask
        # An empty DP owns scratch only. Its dummy token and padding form one
        # contiguous private query span, just as the normal host graph plan.
        positions = torch.where(valid, inputs.positions,
                                torch.arange(count, dtype=torch.int32, device=device))
        ids = torch.where(valid, inputs.input_ids, model.cfg['dspark_noise_token_id'])
        if pad:
            slot = torch.cat((slot, torch.full((pad,),scratch,dtype=torch.int32,device=device)))
            # A real batch has no source scratch rows; a placeholder has
            # count source rows. The device mask determines that distinction.
            scratch_count = (~valid).to(torch.int32).sum()
            positions = torch.cat((positions,torch.arange(pad,dtype=torch.int32,device=device)+scratch_count))
            ids = torch.cat((ids,torch.full((pad,),model.cfg['dspark_noise_token_id'],dtype=torch.int32,device=device)))
            valid = torch.cat((valid,torch.zeros(pad,dtype=torch.bool,device=device)))
        order = torch.argsort(slot, stable=True)
        slots = slot.index_select(0,order)
        position = positions.index_select(0,order)
        physical = torch.arange(model.requests,dtype=torch.int32,device=device)
        owner = slots[:,None] == physical[None,:]
        lengths = owner.sum(0).to(torch.int32)
        ends = torch.where(owner,position[:,None]+1,0).amax(0).to(torch.int32)
        starts = ends-lengths
        self.ids.copy_(ids.index_select(0,order))
        self.positions.copy_(position)
        self.query_requests.copy_(slots)
        self.used.copy_(ends)
        self.cu_query[0].zero_()
        self.cu_query[1:].copy_(lengths.cumsum(0,dtype=torch.int32))
        self.valid.copy_(valid.index_select(0,order))
        flags = inputs.gate_force_random
        if flags is not None:
            if pad:
                flags = torch.cat((flags,torch.zeros(pad,dtype=torch.bool,device=device)))
            self.gate_force_random.copy_(flags.index_select(0,order))
        history.seed_device(self,starts)
        if self.rows:
            hashes = model.engram_device_hash
            runtime = model.engram_uva_runtime
            if hashes is None or runtime is None or inputs.ngram_token_ids is None:
                raise ValueError('original ZOS decode requires the existing device hash/UVA resources')
            width = hashes.width
            ngrams = inputs.ngram_token_ids
            if pad:
                ngrams = torch.cat((ngrams,torch.full((pad,width),hashes.pad_id,
                    dtype=torch.int32,device=device)))
            sorted_ngrams = ngrams.index_select(0,order)
            image_free = self.valid.clone()
            image_free &= self.ids < hashes.token_map.numel()
            for image_id in hashes.image_ids[:2]:
                image_free &= self.ids != image_id
            self.valid.copy_(image_free)
            values = hashes.compute_compressed(sorted_ngrams,self.valid)
            for layer,buffer in self.rows.items():
                runtime.rows(layer,values[layer],buffer)
            self.engram_hash_calls += 1
            self.engram_hash_tokens += self.capacity
        return DeviceHiddenLayout(lengths,starts,ends,torch.argsort(order)[:count])

    def run_original(self, inputs, history):
        layout = self.stage_original(inputs,history)
        runtime = getattr(self.model,'eplb_runtime',None)
        if runtime is not None:
            runtime.bind_original_device_rows(layout.lengths,layout.ends)
        if self.eager:
            self.output = self._forward()
            self.eager_calls += 1
        else:
            if self.graph is None:
                raise RuntimeError('original hidden graph must be captured before decode')
            self.graph.replay()
            self.replays += 1
        hidden = self.output.index_select(0,layout.output_rows)
        span = history.retain_device(self,layout.starts)
        return CausalLMForwardOutput(hidden), layout, span

    def capture(self):
        if self.graph is not None:
            return
        for _ in range(2):
            self._forward()
        torch.npu.synchronize()
        self.graph = torch.npu.NPUGraph()
        with torch.npu.graph(self.graph):
            self.output = self._forward()
        self.captures += 1

    def run(self, layout, history):
        self.stage(layout)
        if self.graph is None:
            raise RuntimeError('original hidden graph must be captured before READY')
        self.graph.replay()
        self.replays += 1
        indices = torch.tensor(layout.output_rows, dtype=torch.int64, device=self.model.device)
        # index_select creates owned storage. Neither sampling nor a rollback
        # span may retain views into buffers overwritten by a later replay.
        hidden = self.output.index_select(0, indices)
        model = self.model
        for source, pair in self.even_rows.items():
            if layout.slots:
                projected, scores = (value.index_select(0, indices) for value in pair)
                history.record(source, layout.slots, layout.original_lengths, projected, scores)
                offset = 0
                for slot, length in zip(layout.slots, layout.original_lengths):
                    end = layout.starts[slot] + length
                    pending = model.pending.setdefault(source, {})
                    if end % 2:
                        pending[slot] = (projected[offset+length-1:offset+length],
                                         scores[offset+length-1:offset+length])
                    else:
                        pending.pop(slot, None)
                    offset += length
        for slot, length in zip(layout.slots, layout.original_lengths):
            end = layout.starts[slot] + length
            model.next_positions[slot] = end
            model.yoco_swa_valid_start[slot] = max(model.yoco_swa_valid_start[slot], end - 128, 0)
        return CausalLMForwardOutput(hidden)


def hidden_profile_identity(model, worker):
    """Use stable placement geometry, never process-local group objects."""
    manifest = Path(__file__).resolve().parents[3] / 'ascend910c/source-manifest.json'
    layers = []
    for layer in model.layers:
        entry = {name:getattr(layer,name,None) for name in
                 ('fused_mc2_decode','fused_o_proj_decode','fused_router_norm','fused_mhc')}
        layout = getattr(layer, 'eplb_layout', None)
        entry['eplb_layout'] = None if layout is None else (*layout[:4], layout[4].size())
        layers.append(entry)
    effective = dict(max_tokens=model.max_tokens, nextn=worker.nextn, layers=layers)
    return hashlib.sha256(manifest.read_bytes() + json.dumps(effective,sort_keys=True).encode()).hexdigest()


class OriginalHiddenGraphPool:
    def __init__(self, model, worker, *, capture_max_graph_only=False, eager=False):
        from deepinfer.worker.cuda_graph_runner import build_graph_batch_sizes

        self.model, self.worker = model, worker
        maximum = worker.scheduler_config.max_num_seqs * (1 + worker.nextn)
        self.maximum, self.eager = maximum, eager
        self.batch_sizes = build_graph_batch_sizes(maximum, envs.DEEPINFER_CUDAGRAPH_BATCH_SIZE_INTERVAL)
        if capture_max_graph_only:
            self.batch_sizes = self.batch_sizes[-1:]
        self.gate_force_random = torch.zeros(maximum, dtype=torch.bool, device=model.device)
        self.valid_token_mask = torch.ones_like(self.gate_force_random)
        self.frames = {} if eager else {size: OriginalHiddenGraph(model, size, self.gate_force_random[:size])
                                        for size in self.batch_sizes}
        self.graphs = {}
        self.curve = []
        self.ready = False
        self.profile_decode = None

    def select(self, total_q):
        if self.eager:
            if type(total_q) is not int or not 1 <= total_q <= self.maximum:
                raise ValueError('original eager query budget exceeds native capacity')
            if total_q not in self.frames:
                self.frames[total_q] = OriginalHiddenGraph(self.model,total_q,
                    self.gate_force_random[:total_q],eager=True)
            return total_q,self.frames[total_q]
        index = bisect_left(self.batch_sizes, total_q)
        if type(total_q) is not int or total_q < 1 or index == len(self.batch_sizes):
            raise ValueError('original hidden query budget exceeds captured graph sizes')
        size = self.batch_sizes[index]
        return size, self.frames[size]

    def warm_up(self, *, profile_decode):
        from deepinfer.worker.model_runner import ModelRunner

        if self.ready:
            if self.profile_decode != profile_decode:
                raise ValueError('hidden graph profiling mode differs from startup configuration')
            return self.curve if profile_decode else []
        if self.eager:
            if profile_decode:
                raise ValueError('original eager mode has no measured graph cost curve')
            self.ready,self.profile_decode = True,False
            return []
        for size in reversed(self.batch_sizes):
            self.model.mc2_global_bs = dist.get_world_size() * math.ceil(size / self.worker.parallel_config.context_parallel_size)
            frame = self.frames[size]
            frame.stage(frame.plan([], []))
            frame.capture()
            self.graphs[size] = SimpleNamespace(graph=frame.graph,
                input_metadata_source=SimpleNamespace(valid_token_mask=self.valid_token_mask[:size],
                                                     gate_force_random=self.gate_force_random[:size]))
        try:
            if profile_decode:
                # Retain the original warmup/5-sample/trim/EP-average/cache
                # algorithm. A source-specific model identity excludes old
                # logits-route curves without introducing another cache policy.
                identity = hidden_profile_identity(self.model, self.worker)
                adapter = SimpleNamespace(use_cuda_graph=True, cuda_graph_runner=self,
                    model_config=SimpleNamespace(model=f'{self.worker.model_config.model}:910c-hidden:{identity}'),
                    parallel_config=self.worker.parallel_config)
                self.curve = ModelRunner.profile_decode_perf(adapter)
                if ([size for size, _ in self.curve] != self.batch_sizes
                        or any(not math.isfinite(ms) or ms <= 0 for _, ms in self.curve)):
                    raise ValueError('original hidden decode profile is incomplete or non-finite')
        finally:
            self.gate_force_random.fill_(False)
            scratch = self.model.requests - 1
            if self.model.histories:
                self.model.histories[scratch].reset()
            self.model.next_positions[scratch] = 0
            for pending in self.model.pending.values():
                pending.pop(scratch, None)
        self.ready = True
        self.profile_decode = profile_decode
        return self.curve if profile_decode else []

    def destroy(self):
        work = getattr(self.model, '_original_device_work', None)
        if work is not None:
            work.drain()
        for frame in self.frames.values():
            if frame.input_copy_done is not None:
                frame.input_copy_done.synchronize()
            frame.graph = None
            frame.output = None
        self.graphs.clear()
        self.frames.clear()
        self.ready = False
