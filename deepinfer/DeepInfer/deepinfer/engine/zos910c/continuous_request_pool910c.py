"""Physical readiness adapter around the original C++ waiting/running queues."""
from types import SimpleNamespace as NS

from .scheduled_worker910c import ScheduledDriver
from .zos_slot_handoff910c import SlotBinding


class PhysicalAssignments:
    def __init__(self, bindings, sequences, records):
        self.bindings = bindings
        self.sequences = sequences
        self.records = records
        self.pending = []
        self.retired = set()
        self.history = []

    def enqueue(self, seq_id):
        if seq_id in self.pending or any(b.seq_id == seq_id for b in self.bindings.values()):
            raise ValueError("request already owns or awaits a physical slot")
        self.pending.append(seq_id)
        self.sequences[seq_id].schedulable = False

    def available_slots(self):
        return [slot for slot in range(4) if slot not in self.bindings
                or self.bindings[slot].seq_id in self.retired]

    def update_readiness(self):
        # Use the original supported input-readiness gate. No physical cache
        # address is available for additional prompts until a generation retires.
        free = len(self.available_slots())
        for number, seq_id in enumerate(self.pending):
            self.sequences[seq_id].schedulable = number < free

    def project(self, metadata, outputs, bindings, positions):
        positions = list(positions)
        known = {binding.seq_id for binding in bindings.values()}
        admissions = []
        for meta in metadata:
            seq_id = meta.seq_id
            if seq_id in known:
                continue
            if not outputs.prompt_run or seq_id not in self.pending:
                raise RuntimeError("new physical generation was not selected by original prefill")
            available = self.available_slots()
            if not available:
                raise RuntimeError("original prompt exceeds ready native physical capacity")
            slot = available[0]
            previous = bindings.get(slot)
            binding = SlotBinding(seq_id, 0 if previous is None else previous.epoch + 1)
            bindings[slot] = binding
            if previous is not None:
                self.retired.remove(previous.seq_id)
            self.pending.remove(seq_id)
            positions[slot] = self.records[seq_id]["hit"]
            admissions.append((slot, seq_id, binding.epoch, positions[slot]))
            self.history.append((None if previous is None else previous.seq_id, seq_id, slot, binding.epoch))
            known.add(seq_id)
        return positions, tuple(admissions)

    def finish_step(self, retired_ids):
        self.retired.update(retired_ids)


class PersistentSchedulerPool:
    """The Scheduler, metadata pool and runtime curve survive drained jobs."""
    def __init__(self, runner, engine, eos, dp_group):
        model = runner.model
        self.mapping = [int(value) for value in model.histories[0].token_map]
        self.config = NS(max_model_len=model.max_tokens - 12, runtime_max_model_len=model.max_tokens,
            max_num_batched_tokens=model.chunk_size * 4, max_num_seqs=4,
            stop_preempted_request=True, chunked_prefill=True, engine_type=0,
            prefill_attempt_interval=2, prefill_any_attempt_interval=3)
        self.scheduler = engine.Scheduler(128, engine.BlockManager(2048), engine.BlockManager(16),
            eos, dp_group, self.config, 0, 0, 4 * model.max_tokens, None,
            self.mapping, 8, model.cfg["vocab_size"], False)
        self.scheduler.decode_perf_curve = [tuple(row) for row in model.dspark_graphs.curve]
        self.driver = ScheduledDriver(engine, self.scheduler, engine.SequenceMetadataPool(), nextn=5,
            on_commit=lambda *_: None, on_finished=lambda *_: None)
        self.sessions = 0
        self.in_use = False
        self.poisoned = False

    def begin(self, on_schedule):
        if self.in_use or self.poisoned or self.scheduler.has_unfinished_seqs() or self.driver._release:
            raise RuntimeError("persistent Scheduler has not drained its previous owner")
        self.sessions += 1
        self.in_use = True
        self.driver.on_schedule = on_schedule
        return self.driver

    def end(self):
        self.driver.release_metadata()
        if self.scheduler.has_unfinished_seqs():
            self.poisoned = True
            raise RuntimeError("persistent Scheduler still has unfinished requests")
        self.driver.on_schedule = None
        self.in_use = False
