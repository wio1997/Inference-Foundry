"""Apply the official prompt decision to one 910C physical-slot generation."""

from dataclasses import dataclass

from .zos_slot_handoff910c import SlotBinding, has_slot_reference, validate_prefix


@dataclass(frozen=True)
class Replacement:
    binding: SlotBinding
    prompt: tuple[int, ...]
    snapshot: dict | None
    start: int


class ScheduledAdmission:
    """Keep reserved identities distinct from device-ready, published slots."""

    def __init__(self, runner, worker, gates, *, close_pipeline, on_activate=None, context_view=None):
        self.runner = runner
        self.worker = worker
        self.gates = gates
        self.close_pipeline = close_pipeline
        self.on_activate = on_activate
        self.context_view = context_view
        self.pending = {}
        self.staged = {}
        self.drains = []

    def reserve(self, slot, previous, prompt, snapshot, *, seq_id):
        if slot in self.pending or self.gates[slot].active is not previous:
            raise RuntimeError("physical slot already reserved or retired")
        if not prompt or type(seq_id) is not int or seq_id < 0 or (previous is not None and seq_id == previous.seq_id):
            raise ValueError("invalid physical slot replacement")
        if snapshot is not None:
            validate_prefix(self.runner.model, prompt, snapshot)
        start = 0 if snapshot is None else len(snapshot["ids"])
        request = Replacement(SlotBinding(seq_id, 0 if previous is None else previous.epoch + 1), tuple(prompt), snapshot, start)
        self.gates[slot].retire(previous)
        self.pending[slot] = request
        return request

    def scheduled_positions(self):
        positions = (self.context_view.logical_positions() if self.context_view is not None
                     else list(self.runner.model.next_positions))
        for slot, req in self.pending.items():
            if slot not in self.staged:
                positions[slot] = req.start
        return positions

    def prepare_step(self, step):
        """Pure local validation. A true result needs coordinated activation."""
        positions = self.scheduled_positions()
        if not step.metadata.is_prompt:
            for row in step.metadata.rows:
                gate = self.gates.get(row.slot)
                if gate is not None and (gate.active is None or
                        (row.seq_id, row.epoch) != (gate.active.seq_id, gate.active.epoch)):
                    raise RuntimeError("decode references an unpublished physical slot")
            return False

        changes = False
        for row, chunk in zip(step.metadata.rows, step.chunks, strict=True):
            req = self.pending.get(chunk.slot)
            reserved = req is not None and chunk.slot not in self.staged
            expected = req.start if reserved else positions[chunk.slot]
            if (row.slot != chunk.slot or row.seq_id != chunk.seq_id or chunk.start != expected
                    or not 0 < len(chunk.tokens) <= self.runner.model.chunk_size):
                raise ValueError("native prefill does not match the physical cursor")
            if req is not None:
                if ((row.seq_id, row.epoch) != (req.binding.seq_id, req.binding.epoch)
                        or chunk.tokens != req.prompt[chunk.start:chunk.end]):
                    raise ValueError("original prompt does not match the reserved physical generation")
                changes |= reserved
        return changes

    def activate_step(self, step):
        replacements = [c.slot for c in step.chunks if c.slot in self.pending and c.slot not in self.staged]
        if not replacements:
            raise RuntimeError("no authorized physical slot replacement")
        model = self.runner.model
        if any(has_slot_reference(model, self.worker, slot) for slot in replacements):
            self.worker.flush_inflight_reqs()
            if getattr(model, "_zos_device_pipeline", None) is not None:
                self.close_pipeline()
            self.drains.append(step.number)
        for slot in replacements:
            request = self.pending[slot]
            binding = self.gates[slot].prepare(request.binding.seq_id, request.prompt, request.snapshot,
                                                prompt_authorized=True)
            if binding != request.binding:
                raise RuntimeError("staged physical slot epoch differs")
            self.staged[slot] = binding
            if self.on_activate is not None:
                self.on_activate(slot, request)

    def complete_prompt(self, chunk, *, step, confidence, committed, ep_ready):
        binding = self.staged.get(chunk.slot)
        if binding is None or not chunk.is_last:
            return
        if not committed:
            raise RuntimeError("native slot publication requires original prompt commit")
        gate = self.gates[chunk.slot]
        gate.seed(binding, step, confidence, sample_completed=True)
        gate.resolve(binding, ep_ready=ep_ready)
        if not ep_ready:
            raise RuntimeError("physical slot publication failed across EP")
        del self.staged[chunk.slot]
        del self.pending[chunk.slot]
