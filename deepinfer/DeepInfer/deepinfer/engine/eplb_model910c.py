"""Model interfaces around the unchanged official EPLB policy and lifecycle."""
import os

import torch
import torch.distributed as dist

# The native backend has already configured its HCCL group/allocator/queue.
# Importing the original EPLB initializes PlatformNPU via utils -> timer.
# Keep its device compatibility, while preserving these established settings.
_NATIVE_RUNTIME_KEYS = ('HCCL_BUFFSIZE', 'TASK_QUEUE_ENABLE', 'PYTORCH_NPU_ALLOC_CONF')
_NATIVE_RUNTIME_BEFORE = {key: os.environ.get(key) for key in _NATIVE_RUNTIME_KEYS}
_NATIVE_RUNTIME_DURING_IMPORT = None
try:
    from deepinfer.model_executor import eplb
    _NATIVE_RUNTIME_DURING_IMPORT = {key: os.environ.get(key) for key in _NATIVE_RUNTIME_KEYS}
finally:
    for _key, _value in _NATIVE_RUNTIME_BEFORE.items():
        if _value is None:
            os.environ.pop(_key, None)
        else:
            os.environ[_key] = _value
_NATIVE_RUNTIME_RESTORED = {key: os.environ.get(key) for key in _NATIVE_RUNTIME_KEYS}
from .eplb_w8a8_910c import W8A8ExpertShuffle, payload_fields


def requested():
    value = os.environ.get('DEEPINFER_910C_EPLB')
    if value not in (None, '0', '1'):
        raise ValueError('DEEPINFER_910C_EPLB must be 0 or 1')
    # As in the original model, load statistics exist even when the original
    # rearrangement interval is zero. Explicit zero remains an ablation.
    return value != '0'


def load_physical_experts(layer):
    """Load the official initial placement, including separate MTP padding."""
    if not requested():
        return False
    from deepinfer import envs
    group = layer.group if layer.group is not None else dist.group.WORLD
    logical = layer.config['n_routed_experts']
    minimum = envs.DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS
    if minimum < 0:
        raise ValueError('the official extra-expert lower bound must be nonnegative')
    extra = eplb.resolve_num_extra_experts(logical, minimum, group.size())
    placement = eplb.get_initial_global_physical_to_logical_map(logical, extra)
    local = len(placement) // group.size()
    for slot in range(group.rank() * local, (group.rank() + 1) * local):
        # A physical slot owns an independent parameter, even for a replica.
        layer.expert_cache[str(slot)] = layer._load_expert(f'experts.{placement[slot]}')
    layer.eplb_layout = (logical, len(placement), logical, extra, group)
    return True


class ModelEPLB:
    def __init__(self, model):
        self.layers = [*model.layers, *(model.dspark.layers if model.dspark is not None else [])]
        self.target_layers = len(model.layers)
        self.mtp_layers = len(self.layers) - self.target_layers
        self.device = model.device
        self.recording = torch.zeros((), dtype=torch.int32, device=self.device)
        self.max_query_len = 1 + (0 if model.dspark is None else model.dspark.nextn)
        self.query_lengths = torch.full((model.requests,), self.max_query_len, dtype=torch.int32, device=self.device)
        self.prompt_ends = torch.full((model.requests,), 2**24, dtype=torch.int32, device=self.device)
        self.query_lengths[-1] = 0  # Existing private scratch request.
        self.groups = []
        self.shufflers = []
        self.steps = self.migrations = 0
        self.original_worker = None
        self.history = []
        grouped = {}
        for index, layer in enumerate(self.layers):
            bank = layer.expert_bank
            weights = [getattr(bank, name) for name in payload_fields(bank)]
            layout = (*layer.eplb_layout, tuple((tuple(w.shape), w.dtype) for w in weights))
            grouped.setdefault(layout, []).append((index, layer, weights))
        with torch.device(self.device):
            for layout, rows in grouped.items():
                logical, physical, routed, extra, group = layout[:5]
                shuffle = W8A8ExpertShuffle([layer.expert_bank for _, layer, _ in rows])
                balancer = eplb.ExpertLoadBalancer(
                    [weights for _, _, weights in rows], len(rows), logical, physical,
                    group, routed, extra, is_normal_engine=False, shuffle_fn=shuffle)
                self.groups.append((balancer, [index for index, _, _ in rows]))
                self.shufflers.append(shuffle)
        self.manager = eplb.ExpertLoadBalancerManager(self.groups, len(self.layers))
        eplb.set_current(self.manager)
        for index, layer in enumerate(self.layers):
            layer.eplb_layer_state = self.manager.layer(index)
            layer.eplb_query_lengths = self.query_lengths
            layer.eplb_prompt_ends = self.prompt_ends
            layer.expert_bank.eplb_count = layer.eplb_layer_state.expert_count
            layer.expert_bank.eplb_recording = self.recording

    def reset_inputs(self):
        # Startup graph capture/native fallback must not enter real windows.
        # Keep the official windows, maps and countdown across request jobs.
        self.recording.zero_()
        self.query_lengths.fill_(self.max_query_len)
        self.query_lengths[-1] = 0
        self.prompt_ends.fill_(2**24)
        for balancer, _ in self.groups:
            balancer.expert_count.zero_()

    def validate_original_worker(self, worker):
        if self.original_worker is worker:
            return
        if self.original_worker is not None or self.steps or self.migrations:
            raise ValueError('EPLB resources already belong to an active request lifecycle')
        if eplb.current() is not self.manager:
            raise ValueError('resident EPLB manager differs from the original global owner')

    def attach_original_worker(self, worker):
        from deepinfer.engine.engine_type import EngineType
        self.validate_original_worker(worker)
        for balancer, _ in self.groups:
            balancer.configure_engine(worker.parallel_config.engine_type == EngineType.NORMAL)
        self.original_worker = worker

    def bind_original_rows(self, packet):
        """Publish actual physical query ownership before target/draft work."""
        self.query_lengths.zero_()
        self.prompt_ends.zero_()
        if packet.rows:
            slots = torch.tensor([row.state_id-1 for row in packet.rows], dtype=torch.int64, device=self.device)
            lengths = torch.tensor([row.seqlen_q for row in packet.rows], dtype=torch.int32, device=self.device)
            ends = torch.tensor([row.seqlen_k for row in packet.rows], dtype=torch.int32, device=self.device)
            self.query_lengths.index_copy_(0, slots, lengths)
            self.prompt_ends.index_copy_(0, slots, ends)

    def bind_original_device_rows(self, lengths, ends):
        """Publish original ZOS query ownership without a device-to-host read."""
        if (lengths.shape != self.query_lengths.shape or ends.shape != self.prompt_ends.shape
                or lengths.dtype != torch.int32 or ends.dtype != torch.int32
                or lengths.device != self.device or ends.device != self.device):
            raise ValueError('original device ownership differs from physical EPLB rows')
        self.query_lengths.copy_(lengths)
        self.prompt_ends.copy_(ends)
        # The final backend state is private padding, excluded from counts.
        self.query_lengths[-1].zero_()
        self.prompt_ends[-1].zero_()

    def after_original_sample(self, output, is_prompt):
        # Same step and deferred-statistics order as DeepseekForCausalLM.sample.
        self.step(is_prompt)
        def post():
            output.moe_load_balance_ratio = self.manager.take_moe_load_balance_ratio()
        output.post_funcs.append(post)

    def detach_original_worker(self, worker):
        if self.original_worker is not worker:
            raise ValueError('cannot release another original Worker EPLB owner')
        self.end_session()
        if eplb.current() is self.manager:
            eplb.destroy_current()
        self.original_worker = None

    def begin_session(self):
        for balancer, _ in self.groups:
            balancer.expert_count.zero_()
        self.recording.fill_(1)

    def end_session(self):
        self.reset_inputs()

    def step(self, is_prompt):
        before = sum(len(s.records) for s in self.shufflers)
        self.manager.step(is_prompt)
        after = sum(len(s.records) for s in self.shufflers)
        if after != before:
            # Complete map publication before another stream can submit a
            # graph. Only migration boundaries need this backend sync.
            torch.npu.synchronize()
        self.steps += 1
        self.migrations += after - before
        self.history.append((bool(is_prompt), after - before))

    def stats(self):
        return dict(enabled=True, target_layers=self.target_layers, mtp_layers=self.mtp_layers,
                    runtime_import_environment=dict(before=_NATIVE_RUNTIME_BEFORE,
                        during_import=_NATIVE_RUNTIME_DURING_IMPORT, restored=_NATIVE_RUNTIME_RESTORED,
                        current={key: os.environ.get(key) for key in _NATIVE_RUNTIME_KEYS}),
                    groups=[dict(layers=len(indices), logical=b.logical_count.shape[1],
                                 physical=b.physical_to_logical_map.shape[1],
                                 interval=b._rearrangement_interval,
                                 window=b._expert_count_window.shape[0]) for b, indices in self.groups],
                    sample_steps=self.steps, group_migrations=self.migrations,
                    step_history=self.history, migration_records=[s.records for s in self.shufflers])


def initialize(model):
    return ModelEPLB(model) if requested() else None


def active_rows(layer, state):
    """Use the actual query ownership after YOCO compaction/SP projection."""
    if not hasattr(layer, 'eplb_layer_state'):
        return None
    direct = getattr(state, 'eplb_rows', None)
    if direct is not None:
        return direct
    frame = state.decode_frame
    if frame is not None:
        slots = frame.query_requests.long()
        offsets = (torch.arange(frame.tokens, dtype=torch.int32, device=slots.device)
                   - frame.cu_query.index_select(0, slots))
        return offsets < layer.eplb_query_lengths.index_select(0, slots)
    lengths = torch.tensor(state.batch.query_lengths, dtype=torch.int64, device=layer.device)
    slots = torch.tensor(state.slots, dtype=torch.int64, device=layer.device).repeat_interleave(
        lengths, output_size=sum(state.batch.query_lengths))
    valid = state.positions < layer.eplb_prompt_ends.index_select(0, slots)
    return layer.sp.split(valid, fill=False) if state.sequence_parallel else valid


def bind_decode(model, step, metadata, metadata_zos, *, driver, root, tp_group):
    runtime = getattr(model, 'eplb_runtime', None)
    if runtime is None:
        return
    lengths = runtime.query_lengths
    lengths.zero_()
    if driver and not step.metadata.is_placeholder:
        q = (metadata_zos.seqlens_q[:len(step.metadata.rows)] if metadata_zos is not None else
             torch.tensor([m.seqlen_q for m in metadata], dtype=torch.int32, device=model.device))
        slots = torch.tensor([r.slot for r in step.metadata.rows], dtype=torch.int64, device=model.device)
        lengths.index_copy_(0, slots, q.to(torch.int32))
    dist.broadcast(lengths, src=root, group=tp_group)


