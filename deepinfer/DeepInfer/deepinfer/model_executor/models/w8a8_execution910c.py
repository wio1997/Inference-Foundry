"""Original metadata/Worker/Sampler interfaces over the resident W8A8 model.

No Scheduler, request queue, stopping policy or weight loader lives here.
Original ZOS metadata feeds the existing hidden graph on device. CPU history
and verification spans advance after the original Worker refreshes sampling
metadata, preserving its forward/previous-post/current-sample lifecycle.
"""
from dataclasses import dataclass
from contextlib import contextmanager
import math
import os

import torch
import torch.distributed as dist

from deepinfer import PLATFORM, envs
from deepinfer.model_executor.input_metadata import InputMetadata
from deepinfer.model_executor.layers.sampler import Sampler
from deepinfer.model_executor.model_output import CausalLMForwardOutput
from deepinfer.model_executor.parallel_utils.parallel_state import PG, cp_broadcast_object
from deepinfer.model_executor.utils import Movable
from deepinfer.worker.execution_resources910c import ExistingW8A8Resources
from deepinfer.worker.model_runner import MainModelOutput
from deepinfer.multimodal_tokens import ImageTokenOffset
from .w8a8_state_history910c import W8A8StateHistory


@dataclass(frozen=True)
class ExecutionRow:
    seq_id: int
    state_id: int
    seqlen_q: int
    seqlen_k: int
    block_table: tuple
    tokens: tuple
    full_len: int
    valid_start: int
    is_placeholder: bool = False
    restored_history: tuple | None = None


@dataclass(frozen=True)
class ExecutionPacket:
    is_prompt: bool
    rows: tuple
    yoco: object = None


@dataclass
class W8A8InputMetadata(InputMetadata):
    original_packet: ExecutionPacket | None = None


def project_original_step(metadata, outputs, model, metadata_zos=None, *, include_tokens=True):
    """Copy only selected token spans while C++ metadata/sequence owners live."""
    if not metadata:
        raise ValueError('original execution requires explicit metadata, including placeholders')
    prompt = bool(metadata[0].is_prompt)
    if any(bool(row.is_prompt) != prompt for row in metadata):
        raise ValueError('original execution cannot mix prompt and decode rows')
    if any(row.is_placeholder for row in metadata):
        if len(metadata) != 1 or not metadata[0].is_placeholder:
            raise ValueError('placeholder metadata cannot share an original real request batch')
        return ExecutionPacket(prompt, ())
    if outputs is None or len(outputs.scheduled_seqs) != len(metadata):
        raise ValueError('original selected sequence owners are required')
    q = [row.seqlen_q for row in metadata]
    k = [row.seqlen_k for row in metadata]
    remap = [-1] * len(metadata)
    previous_tokens = previous_drafts = None
    if metadata_zos is not None:
        q = metadata_zos.seqlens_q[:len(metadata)].detach().cpu().tolist()
        k = metadata_zos.seqlens_k[:len(metadata)].detach().cpu().tolist()
        remap = metadata_zos.remap_cpu[:len(metadata)].tolist()
        if any(index >= 0 for index in remap):
            previous_tokens = metadata_zos.prev_sampled_token_ids.detach().cpu().tolist()
            previous_drafts = ([[] for _ in previous_tokens] if metadata_zos.prev_predict_token_ids is None
                               else metadata_zos.prev_predict_token_ids.detach().cpu().tolist())
    rows, ids, states = [], set(), set()
    for number, (row, seq, count, end) in enumerate(zip(metadata, outputs.scheduled_seqs, q, k, strict=True)):
        if (row.seq_id != seq.seq_id or row.seq_id in ids or row.state_id in states
                or not 0 < row.state_id < model.requests
                or type(count) is not int or type(end) is not int
                or not 1 <= count <= min(end, model.chunk_size) or end > model.max_tokens):
            raise ValueError('invalid original sequence identity, state or query coordinates')
        if not include_tokens:
            tokens = ()
        elif prompt:
            tokens = tuple(seq.data.get_token_ids()[end-count:end])
        elif remap[number] >= 0:
            index = remap[number]
            if index >= len(previous_tokens) or index >= len(previous_drafts):
                raise ValueError('original ZOS remap exceeds previous sample owners')
            tokens = (int(previous_tokens[index]), *map(int, previous_drafts[index][:count-1]))
        else:
            tokens = (row.last_output_token, *row.last_next_n_token[:count-1])
        token_limit = model.cfg['vocab_size'] + (ImageTokenOffset.NUM_SPECIAL
            if prompt and row.multi_modal_data is not None else 0)
        if include_tokens and (len(tokens) != count or any(type(token) is not int or not 0 <= token < token_limit for token in tokens)):
            raise ValueError('original target token span is incomplete or outside vocabulary')
        blocks = tuple(row.block_table)
        if len(blocks) < (end + 127) // 128:
            raise ValueError('original metadata lacks allocated target blocks')
        if prompt and not 0 <= row.swa_valid_start <= end-count:
            raise ValueError('original prompt SWA visibility exceeds its committed boundary')
        restored_history = None
        if row.state_id - 1 in getattr(model, '_original_prefix_restore_positions', {}):
            restored_history = tuple(seq.data.get_token_ids()[:end-count])
        rows.append(ExecutionRow(row.seq_id, row.state_id, count, end, blocks, tokens,
                                 row.full_len, int(row.swa_valid_start), restored_history=restored_history))
        ids.add(row.seq_id); states.add(row.state_id)
    return ExecutionPacket(prompt, tuple(rows))


class OriginalW8A8Executor(ExistingW8A8Resources):
    def __init__(self, runner, *, mtp_module=None, registry=None, sampler=None):
        if envs.DEEPINFER_ENABLE_MTP_VARLEN and runner.model.device.type != 'npu':
            raise NotImplementedError('hidden-state variable graphs and their measured costs must be adapted before backend startup')
        super().__init__(runner, registry)
        if self.model.device.type == 'npu' and PLATFORM.tilelang_ops_backend != 'asc910c':
            raise ValueError('set DEEPINFER_910C_SOFTWARE_OPS=1 before importing the original execution backend')
        if envs.DEEPINFER_USE_NEXTN_DECODE and mtp_module is None:
            from .w8a8_dflash910c import W8A8DFlash
            mtp_module = W8A8DFlash(self.model)
        if envs.DEEPINFER_USE_NEXTN_DECODE and (
                mtp_module is None or not callable(getattr(mtp_module, 'decode_forward', None))
                or (getattr(mtp_module, 'mtp_info_pool', None) is None
                    and not callable(getattr(mtp_module, 'initialize_worker', None)))):
            raise NotImplementedError('original speculative sampling requires a compatible W8A8 DFlash/pool provider')
        self.mtp_module = mtp_module
        self.sampler = Sampler(self.model.cfg['vocab_size']) if sampler is None else sampler
        self.state_history = W8A8StateHistory(self.model)
        self.model._original_state_history = self.state_history
        self._yoco_api = None
        self.hidden_graphs = None
        self._row_trace_root = os.environ.get('DEEPINFER_910C_INVARIANT_TRACE_DIR')
        self._row_trace = None
        self.supports_batch_invariant = (self.model.device.type == 'npu'
                                        and PLATFORM.tilelang_ops_backend == 'asc910c')

    @contextmanager
    def _device_phase(self, name):
        if self.model.device.type != 'npu':
            yield
            return
        first, last = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        first.record()
        try:
            yield
        finally:
            last.record()
            self.registry.set(name, (first, last))
            self.record_device_work()

    def initialize_worker(self, worker):
        super().initialize_worker(worker)
        worker.model_runner.yoco_prefill_truncation_mode = envs.DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE
        error = None
        try:
            if getattr(self.model,'reader',None) is not None:
                from deepinfer.model_executor.model_metadata_utils import ModelMetadata
                from deepinfer.worker.cache_engine import get_kvstore_swa_block_size
                from deepinfer.model_executor.model_loader import _get_entry_architecture
                full = worker.model_config.model_cfg
                if hasattr(_get_entry_architecture(worker.model_config),'multi_modal_mapper'):
                    # Same config ownership as DeepseekVLForCausalLM.from_model_cfg.
                    full.gpt.vision = full.vl_aligner.vision.params
                    full.gpt.aligner = full.vl_aligner.aligner.params
                    if getattr(self.model,'vision_language_tuner',None) is None:
                        from deepinfer.model_executor.vision import load_ve_only
                        self.model.vision_language_tuner,self.model.image_processor = load_ve_only(
                            worker.model_config.model,checkpoint_reader=self.model.reader)
                self.model.model_metadata = ModelMetadata(full.gpt)
                runner = worker.model_runner
                runner.window_size = self.model.model_metadata.window_size
                if runner.window_size != self.model.cfg['sliding_window']:
                    raise ValueError('original model window differs from resident backend configuration')
                runner.swa_block_size = get_kvstore_swa_block_size(runner.window_size)
                runner.max_num_blocks = ((worker.model_config.runtime_max_model_len +
                    runner.block_size-1)//runner.block_size+63)//64*64
        except Exception as exc:
            error = str(exc)
        self._bootstrap_vote(error)
        if self.model.device.type == 'npu' and envs.DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE != 'off':
            error = None
            try:
                from deepinfer.engine.official_yoco_metadata910c import configure_consumer
                configure_consumer(self.model)
            except Exception as exc:
                error = str(exc)
            self._bootstrap_vote(error)
        if self.mtp_module is not None:
            initialize = getattr(self.mtp_module, 'initialize_worker', None)
            if initialize is not None:
                initialize(worker)

    def destroy_cuda_graphs(self):
        if self.mtp_module is not None:
            destroy_graphs = getattr(self.mtp_module, 'destroy_graphs', None)
            if destroy_graphs is not None:
                destroy_graphs()
        if self.hidden_graphs is not None:
            self.hidden_graphs.destroy()
            self.hidden_graphs = None
        super().destroy_cuda_graphs()
        if self.mtp_module is not None:
            destroy = getattr(self.mtp_module, 'destroy', None)
            if destroy is not None:
                destroy()

    def warm_up_model(self, *, capture_max_graph_only=False):
        runtime = getattr(self.model, 'eplb_runtime', None)
        if runtime is not None:
            runtime.end_session()
        if self.model.device.type != 'npu':
            if envs.DEEPINFER_ENABLE_MTP_VARLEN:
                raise NotImplementedError('actual hidden-state graphs require the initialized NPU backend')
            curve = super().warm_up_model(capture_max_graph_only=capture_max_graph_only)
            if runtime is not None:
                runtime.begin_session()
            return curve
        if envs.DEEPINFER_ENFORCE_EAGER:
            if envs.DEEPINFER_ENABLE_MTP_VARLEN:
                raise ValueError('original variable decode requires measured hidden graphs')
            if envs.DEEPINFER_ZERO_OVERHEAD_SCHEDULING and self.hidden_graphs is None:
                from .w8a8_hidden_graph910c import OriginalHiddenGraphPool
                self.hidden_graphs = OriginalHiddenGraphPool(self.model,self.worker,eager=True)
                self.hidden_graphs.warm_up(profile_decode=False)
            if runtime is not None:
                runtime.begin_session()
            return []
        if self.hidden_graphs is None:
            from .w8a8_hidden_graph910c import OriginalHiddenGraphPool
            error, pool = None, None
            try:
                self.cache_storage._assert_drained()
                if any(self.model.next_positions[:-1]) or self.state_history.spans:
                    raise ValueError('hidden graph warmup must precede original request admission')
                pool = OriginalHiddenGraphPool(self.model, self.worker,
                                               capture_max_graph_only=capture_max_graph_only)
            except Exception as exc:
                error = str(exc)
            self._bootstrap_vote(error)
            self.hidden_graphs = pool
        curve = self.hidden_graphs.warm_up(profile_decode=envs.DEEPINFER_ENABLE_MTP_VARLEN)
        if self.mtp_module is not None and not envs.DEEPINFER_DFLASH_ENFORCE_EAGER:
            self.mtp_module.capture_graphs()
        self.record_device_work()
        if runtime is not None:
            runtime.begin_session()
        return curve

    def correct_seqlens_qk(self, metadata, is_prompt, total_q):
        return self.worker._correct_original_seqlens_qk(metadata, is_prompt, total_q)

    def _placeholder_seq_metadata(self, is_prompt):
        return self.worker.model_runner._placeholder_original_seq_metadata(is_prompt)

    def _prepare_original_device_decode(self, metadata, metadata_zos):
        """Use the original metadata producer and original CP tensor transport."""
        from deepinfer.model_executor.parallel_utils.communication_op import broadcast_tensor_dict
        inputs, error = None, None
        if self.worker.is_driver_worker:
            try:
                inputs = self.worker.model_runner._prepare_input_tensors(metadata,metadata_zos=metadata_zos)
                inputs.total_q = inputs.input_ids.numel()
            except Exception as exc:
                error = str(exc)
        if PG.context_parallel_group().size() > 1:
            error = cp_broadcast_object(error)
        failed = torch.tensor([int(error is not None)],dtype=torch.int32,device='cpu')
        dist.all_reduce(failed,op=dist.ReduceOp.MAX,group=PG.cpu_ep_in_dp)
        if failed.item():
            raise RuntimeError(f'original ZOS input preparation failed before tensor broadcast: {error}')
        if PG.context_parallel_group().size() > 1:
            flat = None
            if inputs is not None:
                flat = dict(inputs.to_flatten_dict(),type=type(inputs))
            flat = broadcast_tensor_dict(flat,src=0)
            kind = flat.pop('type')
            if not self.worker.is_driver_worker:
                inputs = kind.from_flatten_dict(flat)
        return inputs

    def _prepare_original_vl(self, metadata, packet):
        """Original chunked MM metadata, using the original CP tensor transport."""
        from deepinfer.model_executor.input_metadata import InputMetadataV4VL
        from deepinfer.model_executor.parallel_utils.communication_op import broadcast_tensor_dict
        from deepinfer.ve.prepare_tensor_input_vl import prepare_input_metadata_vl
        inputs,error,present = None,None,False
        if self.worker.is_driver_worker:
            try:
                present = any(row.multi_modal_data is not None for row in metadata) or envs.DEEPINFER_FORCE_LOAD_BALANCE
                if present:
                    values = prepare_input_metadata_vl(metadata,
                        self.model.model_metadata.aligner_downsample_ratio)
                    ids = [token for row in packet.rows for token in row.tokens]
                    positions = [p for row in packet.rows for p in range(
                        row.seqlen_k-row.seqlen_q,row.seqlen_k)]
                    inputs = InputMetadataV4VL(is_prompt=True,
                        input_ids=torch.tensor(ids,dtype=torch.int32,device=self.model.device),
                        positions=torch.tensor(positions,dtype=torch.int32,device=self.model.device),
                        total_q=len(ids),num_reqs=len(packet.rows),**values)
                    inputs = inputs.recursive_fn(lambda value:value.to(self.model.device,non_blocking=True))
                    if envs.DEEPINFER_FORCE_LOAD_BALANCE:
                        inputs.gate_force_random = torch.ones(len(ids),dtype=torch.bool,device=self.model.device)
                    if inputs.images_raw is not None and getattr(self.model,'vision_language_tuner',None) is None:
                        raise ValueError('original vision resources were not initialized')
            except Exception as exc:
                error = str(exc)
        if PG.context_parallel_group().size() > 1:
            present,error = cp_broadcast_object((present,error))
        failed = torch.tensor([int(error is not None)],dtype=torch.int32,device='cpu')
        dist.all_reduce(failed,op=dist.ReduceOp.MAX,group=PG.cpu_ep_in_dp)
        if failed.item():
            raise RuntimeError(f'original MM metadata rejected before model collectives: {error}')
        if present and PG.context_parallel_group().size() > 1:
            flat = broadcast_tensor_dict(None if inputs is None else inputs.to_flatten_dict(),src=0)
            if not self.worker.is_driver_worker:
                inputs = InputMetadataV4VL.from_flatten_dict(flat)
        return inputs

    def _finish_original_device_decode(self, metadata, inputs):
        """Publish host state after original previous-post and metadata refresh.

        Original Worker calls this sampling phase only after its existing
        corrected-q event and update_metadata_for_sampling_zos. No device
        values are read to reconstruct the CPU history or C2 span here.
        """
        step = inputs._original_device_step
        packet, error = None, None
        if self.worker.is_driver_worker:
            try:
                packet = project_original_step(metadata,step['owners'],self.model)
            except Exception as exc:
                error = str(exc)
        if PG.context_parallel_group().size() > 1:
            packet,error = cp_broadcast_object((packet,error))
        if error is None:
            try:
                if packet.rows:
                    slots = [self.cache_storage.state_slots[row.state_id] for row in packet.rows]
                    lengths = [row.seqlen_q for row in packet.rows]
                    if sum(lengths) != inputs.input_ids.numel():
                        raise ValueError('original refreshed query count differs from submitted target rows')
                    history, model = self.state_history, self.model
                    history.prepare({slot:row.seqlen_k-row.seqlen_q
                        for slot,row in zip(slots,packet.rows,strict=True)})
                    history.begin(slots,lengths)
                    selected = {}
                    for source,pair in step['span'].projected.items():
                        selected[source] = tuple(t.index_select(0,step['layout'].output_rows) for t in pair)
                        history.record(source,slots,lengths,*selected[source])
                    offset = 0
                    for slot,row in zip(slots,packet.rows,strict=True):
                        if model.histories:
                            model.histories[slot].history.extend(row.tokens)
                        end = row.seqlen_k
                        for source,pair in selected.items():
                            pending = model.pending.setdefault(source,{})
                            if end % 2:
                                pending[slot] = tuple(t[offset+row.seqlen_q-1:offset+row.seqlen_q] for t in pair)
                            else:
                                pending.pop(slot,None)
                        model.next_positions[slot] = end
                        model.yoco_swa_valid_start[slot] = max(model.yoco_swa_valid_start[slot],end-model.cfg['sliding_window'],0)
                        offset += row.seqlen_q
                    history.finish()
                    history.publish_device(step['span'],slots)
                inputs.original_packet = packet
                del inputs._original_device_step
            except Exception as exc:
                error = str(exc)
        failed = torch.tensor([int(error is not None)],dtype=torch.int32,device='cpu')
        dist.all_reduce(failed,op=dist.ReduceOp.MAX,group=PG.cpu_ep_in_dp)
        if failed.item():
            raise RuntimeError(f'original ZOS deferred host bookkeeping failed before sampling: {error}')

    def execute_main_model(self, metadata, gpu_cache, *, metadata_zos=None, total_q=None, scheduler_outputs=None):
        if self._row_trace_root:
            from .row_trace910c import update
            self._row_trace=update(self.model,self._row_trace,self._row_trace_root)
        packet, error = None, None
        if self.worker.is_driver_worker:
            try:
                if envs.DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE != self.worker.model_runner.yoco_prefill_truncation_mode:
                    raise ValueError('YOCO mode differs from the original Engine/ModelRunner startup configuration')
                device_decode = (not metadata[0].is_prompt and envs.DEEPINFER_ZERO_OVERHEAD_SCHEDULING
                                 and self.hidden_graphs is not None)
                packet = project_original_step(metadata, scheduler_outputs, self.model,
                    None if device_decode else metadata_zos,include_tokens=not device_decode)
                if packet.is_prompt and packet.rows and envs.DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE != 'off':
                    from deepinfer.engine.official_yoco_metadata910c import prepare_original, load_bindings
                    from dataclasses import replace
                    if self._yoco_api is None:
                        self._yoco_api = load_bindings()
                    yoco = prepare_original(metadata, cp_size=self.runner.tp_size,
                        block_size=self.worker.cache_config.block_size,
                        layout=envs.DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE,
                        keep_tokens=envs.DEEPINFER_YOCO_PREFILL_KEEP_TOKENS,
                        window_size=self.worker.model_runner.window_size,
                        swa_block_size=self.worker.model_runner.swa_block_size,
                        max_backtracking_tokens=envs.DEEPINFER_MAX_BACKTRACKING_TOKENS, api=self._yoco_api)
                    # Reject an unsupported consumer view before any EP rank
                    # enters model kernels or modifies a physical cache.
                    from .official_consumer910c import ChunkView
                    ChunkView.from_metadata(yoco,
                        [row.seqlen_k-row.seqlen_q for row in packet.rows],
                        [row.seqlen_k for row in packet.rows])
                    packet = replace(packet, yoco=yoco)
            except Exception as exc:
                error = str(exc)
        if PG.context_parallel_group().size() > 1:
            packet, error = cp_broadcast_object((packet, error))
        group = PG.cpu_ep_in_dp
        tokens = sum(row.seqlen_q for row in packet.rows) if packet is not None and packet.rows else 1
        prompt = packet is not None and packet.is_prompt
        status = torch.tensor([int(error is not None), int(prompt), int(not prompt), tokens], dtype=torch.int32, device='cpu')
        dist.all_reduce(status, op=dist.ReduceOp.MAX, group=group)
        if status[0].item() or (status[1].item() and status[2].item()):
            raise RuntimeError(f'original backend input projection failed before EP: {error}')
        device_decode = (not packet.is_prompt and envs.DEEPINFER_ZERO_OVERHEAD_SCHEDULING
                         and self.hidden_graphs is not None)
        slots, chunks, graph_frame, graph_layout, graph_size = [], [], None, None, 0
        try:
            if packet.rows:
                positions = {self.cache_storage.state_slots[row.state_id]: row.seqlen_k-row.seqlen_q for row in packet.rows}
                self.cache_storage.restore_prefix_histories(packet.rows)
                if not device_decode:
                    for slot, position in positions.items():
                        self.state_history.validate_position(slot, position)
                slots = list(self.cache_storage.bind_metadata(packet.rows))
                if not device_decode:
                    self.state_history.prepare(positions)
                if packet.is_prompt:
                    for row, slot in zip(packet.rows, slots, strict=True):
                        self.model.yoco_swa_valid_start[slot] = row.valid_start
                chunks = [list(row.tokens) for row in packet.rows]
            else:
                slots = [self.cache_storage.scratch_slot]
                chunks = [[self.model.cfg['dspark_noise_token_id']]]
                if not device_decode:
                    self.state_history.prepare({slots[0]: 0})
            if not packet.is_prompt and self.hidden_graphs is not None:
                if not self.hidden_graphs.ready:
                    raise RuntimeError('hidden graph pool is not READY')
                graph_size, graph_frame = self.hidden_graphs.select(int(status[3].item()))
                if self.hidden_graphs.eager:
                    graph_size = 0
                if not device_decode:
                    graph_layout = graph_frame.plan(chunks if packet.rows else [], slots if packet.rows else [])
        except Exception as exc:
            error = str(exc)
        ready = torch.tensor([int(error is not None)], dtype=torch.int32, device='cpu')
        dist.all_reduce(ready, op=dist.ReduceOp.MAX, group=group)
        if ready.item():
            raise RuntimeError(f'original backend cache binding failed before EP: {error}')
        if packet.yoco is not None:
            self.model.yoco_official_metadata_calls += 1
        runtime = getattr(self.model, 'eplb_runtime', None)
        if runtime is not None and not device_decode:
            runtime.bind_original_rows(packet)
        self.model.mc2_global_bs = dist.get_world_size() * math.ceil(int(status[3].item()) / self.runner.tp_size)
        device_inputs = self._prepare_original_device_decode(metadata,metadata_zos) if device_decode else None
        vl_inputs = self._prepare_original_vl(metadata,packet) if packet.is_prompt else None
        record_state = not device_decode and not packet.is_prompt and (graph_frame is None or bool(packet.rows))
        if record_state:
            self.state_history.begin(slots, [len(chunk) for chunk in chunks])
        with self._device_phase('device_forward'):
            if device_decode:
                output,layout,span = graph_frame.run_original(device_inputs,self.state_history)
                device_inputs.original_packet = packet
                device_inputs._original_device_step = dict(layout=layout,span=span,owners=scheduler_outputs)
            elif graph_frame is not None:
                output = graph_frame.run(graph_layout, self.state_history)
            else:
                output = self.model.forward_tokens(chunks, slots,
                    sequence_parallel=packet.is_prompt and self.model.sequence_parallel,
                    official_yoco=packet.yoco, return_hidden_states=True,
                    state_history=self.state_history if not packet.is_prompt else None,
                    input_metadata=vl_inputs)
        if not isinstance(output, CausalLMForwardOutput):
            raise TypeError('resident backend must return original CausalLMForwardOutput')
        if record_state:
            self.state_history.finish()
        if device_decode:
            device_inputs._decode_observer_owner=graph_frame
            return MainModelOutput(output,device_inputs,graph_size=graph_size)
        ids = [token for chunk in chunks for token in chunk]
        positions = [position for row in packet.rows for position in range(row.seqlen_k-row.seqlen_q, row.seqlen_k)] if packet.rows else [0]
        inputs = W8A8InputMetadata(is_prompt=packet.is_prompt, num_reqs=len(chunks), total_q=len(ids),
            input_ids=torch.tensor(ids, dtype=torch.int32, device=self.model.device),
            positions=torch.tensor(positions, dtype=torch.int32, device=self.model.device), original_packet=packet)
        inputs._decode_observer_owner=graph_frame if graph_frame is not None else self.model
        return MainModelOutput(output, inputs, graph_size=graph_size)

    def execute_sample(self, metadata, inputs, model_output, gpu_cache):
        if hasattr(inputs,'_original_device_step'):
            self._finish_original_device_decode(metadata,inputs)
        sampler_inputs = self.worker.model_runner.prepare_sampler_inputs(metadata, inputs)
        output = model_output.move()
        from .decode_trace910c import observe,observe_draft
        with self._device_phase('device_sample'),observe(self,inputs,sampler_inputs) as tape:
            result = self.sampler(self.mtp_module, embedding=self.model.head,
                hidden_states=Movable(output.hidden_states), sampler_inputs=sampler_inputs,
                main_input_metadata=inputs, nextn_kv_caches=gpu_cache, seq_metadata_list=metadata)
            if tape is not None:
                if not inputs.is_prompt and self.mtp_module is not None:
                    # The original module has completed its enqueue phase;
                    # snapshots precede any later graph reuse on this stream.
                    observe_draft(self.mtp_module,len(inputs.original_packet.rows) or 1)
                result.post_funcs.append(tape.write)
            checks = getattr(self.mtp_module, 'take_graph_checks', None)
            if checks is not None:
                result.post_funcs[:0] = checks()
            runtime = getattr(self.model, 'eplb_runtime', None)
            if runtime is not None:
                runtime.after_original_sample(result, inputs.is_prompt)
            return result

    def post_process(self, metadata, output, is_prompt):
        return self.worker.model_runner._post_process_original(metadata, output, is_prompt)
