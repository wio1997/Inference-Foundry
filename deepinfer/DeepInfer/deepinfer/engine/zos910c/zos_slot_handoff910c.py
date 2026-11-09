"""Bridge official seq_id lifetime to one native fixed-address cache slot."""
from dataclasses import dataclass


@dataclass(frozen=True)
class SlotBinding:
    seq_id: int
    epoch: int

    @property
    def key(self):
        return str(self.seq_id), self.epoch


def _tensor_matches(value, target, shape):
    return (hasattr(value, "shape") and tuple(value.shape) == shape and value.dtype == target.dtype
            and value.device == target.device)


def validate_draft_snapshot(draft, values):
    """Reject incomplete or broadcastable draft payloads before any copy."""
    planes = draft.planes
    if (not isinstance(values, (tuple, list)) or len(values) != len(planes) or any(
            not _tensor_matches(value, plane.cache, (plane.resident_pages, *plane.cache.shape[1:]))
            for value, plane in zip(values, planes))):
        raise ValueError("snapshot draft state differs")


def validate_prefix(model, prompt, snapshot):
    """Validate the current native capture_prefix format before any writes."""
    ids = snapshot["ids"]
    if not 0 < len(ids) < len(prompt) or tuple(prompt[:len(ids)]) != ids:
        raise ValueError("snapshot does not match a proper prefix of the request")
    validate_snapshot(model, snapshot)


def validate_snapshot(model, snapshot):
    """Validate the complete native snapshot for every restoration entry."""
    ids = snapshot["ids"]
    if not isinstance(ids, (tuple, list)) or any(type(token) is not int or token < 0 for token in ids):
        raise ValueError("snapshot token IDs differ")
    if snapshot.get("index_format", "int8-fp16-v1") != model.cache.index_format:
        raise ValueError("snapshot index format differs")
    ratios = model.cfg["compress_ratios"]
    shapes = {}
    for name in ("swa", "compressed", "index_key", "index_scale"):
        for key, plane in getattr(model.cache, name).items():
            pages = plane.resident_pages
            if name != "swa":
                pages = min(pages, (len(ids) // ratios[key] + plane.block_size - 1) // plane.block_size)
            shapes[name, key] = plane.cache, (pages, *plane.cache.shape[1:])
    for name, key, value in snapshot["planes"]:
        target = shapes.pop((name, key), None)
        if target is None or not _tensor_matches(value, *target):
            raise ValueError("snapshot cache plane differs")
    if shapes:
        raise ValueError("snapshot cache plane missing")
    if snapshot["history"] != (list(ids) if model.histories else None):
        raise ValueError("snapshot Engram history differs")
    pending = snapshot["pending"]
    needed = {i for i in model.cfg["kv_source_layer_ids"] if ratios[i] == 2 and len(ids) % 2}
    if set(pending) != needed:
        raise ValueError("snapshot C2 pending coverage differs")
    for index, pair in pending.items():
        layer = model.layers[index]
        weights = layer.compressor_wkv_weight, layer.compressor_wgate_weight
        if len(pair) != len(weights) or any(
                tuple(value.shape) != (1, weight.shape[0]) or str(value.dtype) != "torch.float32"
                or value.device != weight.device for value, weight in zip(pair, weights)):
            raise ValueError("snapshot C2 pending payload differs")
    draft = snapshot["dspark"]
    if model.dspark is None:
        if draft is not None:
            raise ValueError("snapshot includes unexpected draft state")
    else:
        validate_draft_snapshot(model.dspark, draft)


def has_slot_reference(model, worker, slot):
    pipeline = getattr(model, "_zos_device_pipeline", None)
    if pipeline is not None:
        if getattr(pipeline, "frame", None) is None:
            return True
        frames = [getattr(pipeline, name, None) for name in ("frame", "small_frame", "compact_frame")]
        for frame in frames:
            if frame is not None and (not hasattr(frame, "slots") or slot in frame.slots):
                return True
    previous = worker.prev_step_state
    if previous is None:
        return False
    occupied = getattr(previous.sampler_output, "physical_slots", None)
    return occupied is None or slot in occupied


class SlotHandoff:
    """Owned by the Worker thread. The scheduler supplies the prompt phase."""

    def __init__(self, model, worker, slot, binding, policy=None, *, slots_per_dp):
        if (type(slot) is not int or type(slots_per_dp) is not int
                or not 0 <= slot < slots_per_dp <= len(model.next_positions)):
            raise ValueError("real native slot required")
        self.model = model
        self.worker = worker
        self.slot = slot
        self.active = binding
        self._empty = binding is None
        self._empty_reserved = False
        self.policy = policy
        self._retired = None
        self._staged = None
        self._seeded = False
        self._registered = False
        self.failed = False

    def retire(self, binding):
        if binding is None and self._empty and self.active is None and not self._empty_reserved:
            self._empty_reserved = True
            return
        if binding is None or self.active is not binding:
            raise RuntimeError("stale slot retirement")
        self._retired = self.active
        self.active = None

    def prepare(self, seq_id, prompt, snapshot, *, prompt_authorized):
        old = self._retired
        if (prompt_authorized is not True or (old is None and not self._empty_reserved) or self.active is not None
                or self._staged is not None or self.failed):
            raise RuntimeError("slot is not eligible for replacement")
        if type(seq_id) is not int or seq_id < 0 or (old is not None and seq_id == old.seq_id):
            raise ValueError("replacement needs a fresh official seq_id")
        if self.policy is not None:
            states = self.policy.inner.snapshot()
            if old is not None and old.key in states:
                raise RuntimeError("old host FIFO has not settled")
            if any(key[0] == str(seq_id) for key in states):
                raise ValueError("fresh seq_id already owned")
        if has_slot_reference(self.model, self.worker, self.slot):
            raise RuntimeError("release previous Worker and fixed-graph references before replacement")
        if not prompt:
            raise ValueError("empty replacement prompt")
        if snapshot is not None:
            validate_prefix(self.model, prompt, snapshot)
        try:
            if snapshot is None:
                self.model.next_positions[self.slot] = 0
                self.model.yoco_swa_valid_start[self.slot] = 0
                self.model.restored_prefix_slots.discard(self.slot)
                for slots in self.model.pending.values():
                    slots.pop(self.slot, None)
                if self.model.histories:
                    self.model.histories[self.slot].reset()
            else:
                self.model.restore_prefix(snapshot, self.slot)
        except Exception:
            self.failed = True
            raise
        self._staged = SlotBinding(seq_id, 0 if old is None else old.epoch + 1)
        return self._staged

    def seed(self, binding, step, confidence, *, sample_completed):
        if (self.failed or binding is None or binding is not self._staged
                or self._seeded or self._registered or sample_completed is not True):
            raise RuntimeError("staged slot needs a completed prompt sample")
        if self.policy is not None:
            self.policy.inner.register((binding.key,))
            self._registered = True
            self.policy.inner.seed_prefill((binding.key,), step, [confidence], sample_completed=True)
        self._seeded = True

    def resolve(self, binding, *, ep_ready):
        if binding is not self._staged or binding is None:
            raise RuntimeError("stale slot publication")
        if type(ep_ready) is not bool:
            raise TypeError("EP readiness must be a boolean")
        if not ep_ready:
            if self._registered:
                self.policy.retire((binding.key,))
            self.failed = True
        elif not self._seeded:
            raise RuntimeError("EP publication requires locally seeded state")
        else:
            self.active = binding
            self._empty = False
            self._empty_reserved = False
            self._retired = None
        self._staged = None
        self._seeded = False
        self._registered = False
