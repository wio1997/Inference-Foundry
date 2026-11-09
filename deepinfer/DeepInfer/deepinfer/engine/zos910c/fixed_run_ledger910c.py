"""Slot bookkeeping for a fixed-four native diagnostic run."""

from dataclasses import dataclass

from deepinfer.engine.bounded_replay910c import eligible


@dataclass(frozen=True)
class StopState:
    golden: tuple[int, ...]
    remaining: tuple[int, ...]
    finished: tuple[bool, ...]


class FixedRunLedger:
    def __init__(self, runner, engine, *, eos, keep=140):
        self.runner = runner
        self.model = runner.model
        self.engine = engine
        self.eos = eos
        self.keep = keep
        self.steps = {}
        self.requests = {}
        self.stop = StopState((self.model.cfg["dspark_noise_token_id"],) * 4, (0,) * 4, (True,) * 4)
        self.results = {}
        self.trace = []
        self.admission = None
        self.publications = []

    def register(self, slot, binding, request, *, hit):
        if not 0 <= slot < 4 or binding.seq_id in self.results:
            raise ValueError("invalid native request generation")
        if slot in self.requests and not self.stop.finished[slot]:
            raise RuntimeError("cannot replace a live physical slot")
        self.requests[slot] = binding, request, hit
        self.results[binding.seq_id] = []

    def bind_step(self, step):
        if step.number in self.steps or len(self.steps) >= 2:
            raise RuntimeError("unexpected original Worker in-flight depth")
        self.steps[step.number] = step

    def step_for(self, number):
        return self.steps[number]

    def bounded_keep(self, step):
        slots = [chunk.slot for chunk in step.chunks]
        owned, hits = [], []
        for slot in slots:
            binding, request, hit = self.requests[slot]
            owned.append((binding.seq_id, request))
            hits.append(hit)
        ok = eligible(self.model, owned, slots, [c.tokens for c in step.chunks], hits,
                      internal=not slots, keep=self.keep)
        return self.keep if not self.runner._global_busy(not ok) else 0

    def decode_seed(self, step):
        if any(slot not in self.requests and (not self.stop.finished[slot] or self.stop.remaining[slot] != 0)
               for slot in range(4)):
            raise RuntimeError("unowned physical rows must remain finished with zero budget")
        return dict(golden=self.stop.golden, remaining=self.stop.remaining,
                    finished=self.stop.finished, eos_token_id=self.eos,
                    ignore_eos=[bool(self.requests[slot][1].get("ignore_eos", False))
                                if slot in self.requests else True for slot in range(4)])

    def original_stopping(self, step, seqs, rows):
        golden, remaining, finished = map(list, (self.stop.golden, self.stop.remaining, self.stop.finished))
        statuses = self.engine.SequenceStatus
        for projected, sequence, (seq_id, tokens, count, _) in zip(step.metadata.rows, seqs, rows, strict=True):
            slot = projected.slot
            binding, request, _ = self.requests[slot]
            if (binding.seq_id, binding.epoch) != (seq_id, projected.epoch) or seq_id != sequence.seq_id:
                raise RuntimeError("original Scheduler committed a different slot generation")
            if sequence.status == statuses.WAITING:
                if not step.metadata.is_prompt or count:
                    raise RuntimeError("unexpected waiting state during target commit")
                continue
            if sequence.status not in (statuses.RUNNING, statuses.FINISHED_STOPPED, statuses.FINISHED_LENGTH_CAPPED):
                raise RuntimeError("unsupported original Scheduler termination")
            if tokens:
                golden[slot] = tokens[-1]
            remaining[slot] = request["max_tokens"] - (sequence.get_len() - sequence.get_prompt_len())
            finished[slot] = sequence.status != statuses.RUNNING
            if remaining[slot] < 0:
                raise RuntimeError("original Scheduler exceeded the native output budget")
        return StopState(tuple(golden), tuple(remaining), tuple(finished))

    def publish_committed(self, step, rows, stop):
        for projected, (seq_id, tokens, _, cached) in zip(step.metadata.rows, rows, strict=True):
            if self.model.next_positions[projected.slot] != cached:
                raise RuntimeError("native KV cursor differs from original Scheduler")
            self.results[seq_id].extend(tokens)
        if step.metadata.is_prompt and self.admission is not None:
            for chunk, (_, _, count, _) in zip(step.chunks, rows, strict=True):
                if chunk.slot in self.admission.staged and chunk.is_last:
                    self.publications.append((chunk, step.number, bool(count)))
        self.stop = stop
        self.trace.append((step.number, step.metadata.is_prompt, rows))
        del self.steps[step.number]

    def finish_step(self):
        for chunk, number, committed in self.publications:
            self.admission.complete_prompt(chunk, step=number, confidence=None, committed=committed,
                                           ep_ready=True)
        self.publications.clear()
