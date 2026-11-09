"""Online file-IPC adapter; original Scheduler and Worker own execution policy."""
from pathlib import Path
from types import SimpleNamespace as NS
import json
import os
import time

import numpy as np
import torch
import torch.distributed as dist

from .zos910c.scheduled_admission910c import ScheduledAdmission
from .zos910c.cold_prefill910c import can_prefill, select_admission_dp


class ContinuousAdmission(ScheduledAdmission):
    """Paged original states share fixed native addresses, so drain before writes."""
    def __init__(self, *args, target, **kwargs):
        super().__init__(*args, **kwargs)
        self.target = target

    def prepare_step(self, step):
        changes = super().prepare_step(step)
        return changes or (step.metadata.is_prompt and self.target.pipeline is not None)

    def activate_step(self, step):
        if self.target.pipeline is not None:
            self.worker.flush_inflight_reqs()
            self.close_pipeline()
            self.drains.append(step.number)
        if any(chunk.slot in self.pending and chunk.slot not in self.staged for chunk in step.chunks):
            super().activate_step(step)


def atomic(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, separators=(",", ":")))
    temporary.replace(path)


def cached(runner, request):
    ids = request["token_ids"]
    return next((runner.prefix_cache[tuple(ids[:point])] for point in reversed(runner.prefix_points)
                 if len(ids) > point and tuple(ids[:point]) in runner.prefix_cache), None)


def eligible(runner, request):
    model = runner.model
    eos_enabled = os.environ.get("DEEPINFER_910C_ZOS_CONTINUOUS_EOS") == "1"
    return ((request.get("ignore_eos", False) or eos_enabled) and 0 < request["max_tokens"]
            and len(request["token_ids"]) + request["max_tokens"] + 12 <= model.max_tokens)


def try_run(runner, job, directory):
    requests = job["requests"]
    small = os.environ.get("DEEPINFER_910C_ZOS_CONTINUOUS_SMALL") == "1"
    cold = os.environ.get("DEEPINFER_910C_ZOS_CONTINUOUS_COLD") == "1"
    capacity = runner.slots * runner.dp_size
    valid_count = 0 < len(requests) <= capacity if small else len(requests) == capacity
    if (not job.get("continuous_admission", False) or runner.slots != 4
            or not valid_count
            or not all(eligible(runner, request) for request in requests)):
        return False
    owned = [(index, request) for index, request in enumerate(requests) if index % runner.dp_size == runner.dp_rank]
    snapshots = [cached(runner, request) for _, request in owned]
    minimum = 2 * runner.model.cfg["index_topk"]
    if runner._global_busy(any(not can_prefill(request, snapshot, cold=cold, minimum=minimum)
                               for (_, request), snapshot in zip(owned, snapshots, strict=True))):
        return False
    padding_prefix = None
    if len(owned) < runner.slots:
        # Fixed physical C2 rows require initialized addresses. Inactive rows
        # are finished padding, never synthetic original Scheduler sequences.
        padding_prefix = next((snapshot for snapshot in snapshots
                               if snapshot is not None and minimum <= len(snapshot["ids"]) <= runner.model.max_tokens - 11), None)
        if padding_prefix is None:
            padding_prefix = next((snapshot for snapshot in runner.prefix_cache.values()
                if minimum <= len(snapshot["ids"]) <= runner.model.max_tokens - 11), None)
    missing_padding = len(owned) < runner.slots and padding_prefix is None
    missing_any = runner._global_busy(missing_padding) if len(requests) < capacity else False
    if missing_any and not cold:
        return False
    if not hasattr(runner, "_scheduled_groups"):
        raise RuntimeError("original Scheduler groups were not initialized before readiness")
    if not runner.model.dspark_graphs.curve:
        # Startup capture precedes the native path's real runtime calibration.
        # Keep that existing fallback;never seed Scheduler with an invented curve.
        return False
    directory = Path(directory)
    runner._scheduled_owned = list(owned) + [None] * (runner.slots - len(owned))
    runner._scheduled_hits = [0 if snapshot is None else len(snapshot["ids"]) for snapshot in snapshots] + [0] * (runner.slots - len(owned))
    runner._scheduled_initial_owned = list(owned)
    runner._scheduled_padding_prefix = padding_prefix
    runner._scheduled_bootstrap_padding = bool(cold and missing_any)
    runner._scheduled_cold_enabled = cold
    runner._scheduled_official_yoco = os.environ.get("DEEPINFER_910C_YOCO_OFFICIAL_PREFILL") == "1"
    if runner._scheduled_official_yoco:
        from .official_yoco_metadata910c import OfficialYocoMetadata, configure_consumer
        configure_consumer(runner.model)
        if runner.official_yoco_metadata is None:
            runner.official_yoco_metadata = OfficialYocoMetadata(cp_size=runner.tp_size)
        runner.official_yoco_metadata.begin_job([request for _, request in owned])
        engine = runner.official_yoco_metadata.api
    else:
        from deepinfer._C import engine
    session = Session(runner, engine, job, directory)
    try:
        session.run(snapshots)
    finally:
        if runner._scheduled_official_yoco:
            runner.official_yoco_metadata.end_job()
        runner._scheduled_official_yoco = False
        runner._scheduled_cold_enabled = False
        session.close_events()
    return True


class Session:
    def __init__(self, runner, engine, job, directory):
        from .zos910c.continuous_request_pool910c import PersistentSchedulerPool, PhysicalAssignments
        from .zos910c.fixed_run_ledger910c import FixedRunLedger
        from .zos910c.native_scheduled_runner910c import ScheduledNativeModelRunner
        from .zos910c.native_scheduled_target910c import ScheduledNativeTarget
        from .zos910c.official_phase_core910c import NativeEventRegistry, create_worker

        self.runner, self.model, self.engine = runner, runner.model, engine
        self.job, self.directory = job, directory
        self.cold_enabled = os.environ.get("DEEPINFER_910C_ZOS_CONTINUOUS_COLD") == "1"
        self.dp_group, self.host_tp, self.control = runner._scheduled_groups
        self.is_driver = runner.tp_rank == 0
        self.started = time.monotonic()
        self.events = ((directory / f"events-dp{runner.dp_rank}.jsonl").open("a", buffering=1)
                       if self.is_driver else None)
        self.records, self.sequences, self.bindings = {}, {}, {}
        self.done, self.hits = set(), {}
        self.finish_reasons = {}
        self.inbox_cursor = 1
        self.closed = False
        self.steps = []
        self.step_wall = []
        self.trace = []
        self.staging_submissions = 0
        self.pinned = True
        self.ledger = FixedRunLedger(runner, engine, eos=job["eos_token_id"])
        self.context_view = None
        if self.cold_enabled:
            from .zos910c.cold_prefill910c import ColdContextView
            self.context_view = ColdContextView(runner, self.ledger)
        self.target = ScheduledNativeTarget(runner, engine, context_view=self.context_view,
            async_sample_inputs=os.environ.get("DEEPINFER_910C_ZOS_ASYNC_SAMPLE_INPUTS") == "1",
            multigraph=os.environ.get("DEEPINFER_910C_ZOS_OFFICIAL_MULTIGRAPH") == "1")
        self.assignments = PhysicalAssignments(self.bindings, self.sequences, self.records)
        self.pool = None
        if self.is_driver:
            self.pool = getattr(runner, "_continuous_scheduler_pool", None)
            if self.pool is None:
                self.pool = runner._continuous_scheduler_pool = PersistentSchedulerPool(
                    runner, engine, job["eos_token_id"], self.dp_group)
            driver = self.pool.begin(self.assignments.project)
        else:
            driver = None
        self.driver = driver
        registry = NativeEventRegistry()
        self.seam = ScheduledNativeModelRunner(runner, self.target, self.ledger, driver=driver,
            groups=(self.host_tp, self.control), registry=registry, scratch_token=job["requests"][0]["token_ids"][0])
        callbacks = NS(post_process_fn=self.seam.post_already_committed,
                       pre_sampler_sync_fn=None if driver is None else driver.pre_sampler_sync)
        self.worker_deferred_post = os.environ.get('DEEPINFER_910C_ZOS_WORKER_SYNC_ABLATION') != '1'
        self.worker = create_worker(self.seam, callbacks,
            NS(DEEPINFER_ENABLE_MTP_VARLEN=True, DEEPINFER_ZERO_OVERHEAD_SCHEDULING=self.worker_deferred_post),
            registry, is_driver=self.is_driver, backend=torch)
        self.seam.worker = self.worker
        self.seam.receive_step = self.receive_step

    def emit(self, event):
        if self.events is not None:
            event.setdefault("elapsed", time.monotonic() - self.started)
            self.events.write(json.dumps(event, separators=(",", ":")) + "\n")

    def close_events(self):
        if self.model.eplb_runtime is not None:self.model.eplb_runtime.end_session()
        if self.events is not None:
            self.events.close()

    def sequence(self, seq_id):
        from .zos910c.scheduled_prefill910c import initialize_prompt_cursor
        record = self.records[seq_id]
        request = record["request"]
        mapping = self.pool.mapping
        ids = np.asarray(request["token_ids"], dtype=np.int32)
        compressed = np.asarray([mapping[index] for index in ids], dtype=np.int32)
        sampling = NS(temperature=0., top_p=1., top_k=-1, seq_seed=0, top_logprobs=0,
            prompt_logprobs=False, first_n_token_no_eos=0, eos_id=self.job["eos_token_id"],
            ignore_eos=request.get("ignore_eos", False), max_tokens=request["max_tokens"])
        seq = self.engine.Sequence(seq_id, ids, compressed, None, 128, request["id"],
                                   "continuous", sampling, 0., None)
        initialize_prompt_cursor(seq, position=record["hit"], chunk_size=self.model.chunk_size,
                                 prefix_points=self.runner.prefix_points)
        self.sequences[seq_id] = seq
        self.pool.scheduler.add_sequence(seq)
        return seq

    def activate(self, slot, replacement):
        record = self.records[replacement.binding.seq_id]
        request, hit = record["request"], record["hit"]
        record.pop("snapshot", None)
        self.ledger.register(slot, replacement.binding, request, hit=hit)
        self.hits[replacement.binding.seq_id] = hit
        self.runner._scheduled_owned[slot] = (record["index"], request)
        self.runner._scheduled_hits[slot] = hit
        if self.runner._scheduled_official_yoco:
            self.runner.official_yoco_metadata.bind_slot(slot, request)
        self.emit(dict(type="start", index=record["index"], cached_tokens=hit))
        self.trace.append(dict(action="physical_activate", seq_id=replacement.binding.seq_id,
                               slot=slot, epoch=replacement.binding.epoch))

    def receive_step(self, step):
        for slot, seq_id, epoch, position in step.admissions:
            record = self.records[seq_id]
            request = record["request"]
            gate = self.gates[slot]
            old = gate.active
            snapshot = record["snapshot"] if self.cold_enabled else cached(self.runner, request)
            reserved = self.admission.reserve(slot, old, request["token_ids"], snapshot, seq_id=seq_id)
            if reserved.binding.epoch != epoch or reserved.start != position:
                raise RuntimeError("original selected generation differs across TP")
            self.bindings[slot] = reserved.binding

    def poll(self):
        message = [None]
        if dist.get_rank() == 0:
            try:
                rows = []
                while (self.directory / "admissions" / f"{self.inbox_cursor:08d}.json").exists():
                    value = json.loads((self.directory / "admissions" / f"{self.inbox_cursor:08d}.json").read_text())
                    if value["number"] != self.inbox_cursor:
                        raise RuntimeError("admission transaction order differs")
                    rows.append(value)
                    self.inbox_cursor += 1
                message[0] = (rows, (self.directory / "admission-close.json").exists(), None)
            except Exception as exc:
                message[0] = ([], False, str(exc))
        dist.broadcast_object_list(message, src=0, group=self.control, device=torch.device("cpu"))
        rows, self.closed, error = message[0]
        if error is not None:
            raise RuntimeError(f"continuous inbox failed: {error}")
        for transaction in rows:
            self.enqueue(transaction)

    def enqueue(self, transaction):
        rows = transaction["requests"]
        if not rows or len(rows) > 8:
            raise ValueError("invalid admission transaction size")
        minimum = 2 * self.model.cfg["index_topk"]
        matches = torch.zeros((len(rows), self.runner.dp_size, 2), dtype=torch.int32)
        if self.is_driver:
            for index, row in enumerate(rows):
                snapshot = cached(self.runner, row["request"])
                if eligible(self.runner, row["request"]) and can_prefill(
                        row["request"], snapshot, cold=self.cold_enabled, minimum=minimum):
                    matches[index, self.runner.dp_rank, 0] = 1
                    matches[index, self.runner.dp_rank, 1] = 0 if snapshot is None else len(snapshot["ids"])
        dist.all_reduce(matches, op=dist.ReduceOp.MAX, group=self.control)
        accepted, deferred = [], []
        next_id = self.runner._continuous_next_seq_id
        error = None
        try:
            for offset, (row, candidates) in enumerate(zip(rows, matches.tolist(), strict=True)):
                hint = row["index"] % self.runner.dp_size
                dp = select_admission_dp(candidates, hint)
                if dp is None:
                    deferred.append(row["index"])
                    continue
                seq_id = next_id + offset
                if seq_id in self.records:
                    raise ValueError("duplicate original sequence identity")
                hit = candidates[dp][1]
                snapshot = cached(self.runner, row["request"]) if dp == self.runner.dp_rank and hit else None
                if dp == self.runner.dp_rank and hit and (snapshot is None or len(snapshot["ids"]) != hit):
                    raise RuntimeError("selected cached prefix changed before admission")
                self.records[seq_id] = dict(index=row["index"], request=row["request"], hit=hit, dp=dp)
                if self.cold_enabled:
                    self.records[seq_id]["snapshot"] = snapshot
                if dp == self.runner.dp_rank and self.is_driver:
                    self.sequence(seq_id)
                    self.assignments.enqueue(seq_id)
                accepted.append(dict(index=row["index"], seq_id=seq_id, dp=dp, hit=hit))
                self.trace.append(dict(action="original_enqueue", seq_id=seq_id, index=row["index"], dp=dp,
                    step=len(self.steps), live_requests=sum(not value for value in self.ledger.stop.finished),
                    previous_worker_pending=self.worker.prev_step_state is not None))
            self.runner._continuous_next_seq_id += len(rows)
        except Exception as exc:
            error = str(exc)
        failure = torch.tensor([int(error is not None)], dtype=torch.int32)
        dist.all_reduce(failure, group=self.control)
        if failure.item():
            raise RuntimeError(f"continuous enqueue failed: {error}")
        if dist.get_rank() == 0:
            atomic(self.directory / f"admission-result-{transaction['number']:08d}.json",
                   dict(number=transaction["number"], accepted=accepted, deferred=deferred))

    def completion_reason(self, seq_id):
        # Original Scheduler stop.finished authorizes completion. This maps its
        # emitted final token/length to the HTTP reason, without rescheduling.
        from .eos_state910c import finish_reason
        request = self.records[seq_id]["request"]
        generated = self.ledger.results[seq_id]
        if not 0 < len(generated) <= request["max_tokens"]:
            raise RuntimeError("original Scheduler emitted an invalid completion length")
        reason = finish_reason(generated[-1], len(generated), request["max_tokens"],
            ignore_eos=request.get("ignore_eos", False), eos_token_id=self.job["eos_token_id"])
        if reason is None:
            raise RuntimeError("original Scheduler completion has no supported stop reason")
        return reason

    def publish(self, step, rows, stop):
        offsets = {seq_id: len(self.ledger.results[seq_id]) for seq_id, _, _, _ in rows}
        self.original_publish(step, rows, stop)
        # Original WAITING prompt chunks keep the physical decode row finished.
        # That mask is not an HTTP completion before the final prompt chunk.
        waiting_prefill = ({chunk.seq_id for chunk in step.chunks if not chunk.is_last}
                           if self.cold_enabled and step.metadata.is_prompt else ())
        for projected, (seq_id, tokens, _, _) in zip(step.metadata.rows, rows, strict=True):
            record = self.records[seq_id]
            for offset, token in enumerate(tokens, offsets[seq_id]):
                self.emit(dict(type="token", index=record["index"], token_id=token, step=offset))
            if stop.finished[projected.slot] and seq_id not in self.done and seq_id not in waiting_prefill:
                generated = self.ledger.results[seq_id]
                request = record["request"]
                reason = self.completion_reason(seq_id)
                self.finish_reasons[seq_id] = reason
                self.done.add(seq_id)
                self.emit(dict(type="done", index=record["index"], prompt_tokens=len(request["token_ids"]),
                    cached_tokens=record["hit"], output_tokens=len(generated), token_ids=generated, finish_reason=reason))

    def close_target(self):
        stats = self.target.sample_input_stats()
        self.staging_submissions += stats["submissions"]
        if stats["enabled"] and stats["submissions"]:
            self.pinned &= stats["pinned"]
        self.target.close()

    def drain(self):
        self.worker.flush_inflight_reqs()
        self.ledger.finish_step()
        if self.driver is not None:
            self.driver.release_metadata()
        self.close_target()

    def run(self, snapshots):
        from .zos910c.scheduled_worker910c import run_scheduled_step
        from .zos910c.zos_slot_handoff910c import SlotBinding, SlotHandoff, validate_prefix
        self.model.reset()
        self.runner._official_yoco_decisions = []
        self.runner._scheduled_cold_prefix_captures = 0
        self.padding_bootstrap_tokens = self.padding_bootstrap_forwards = 0
        next_id = getattr(self.runner, "_continuous_next_seq_id", 1000)
        for index, request in enumerate(self.job["requests"]):
            self.records[next_id + index] = dict(index=index, request=request,
                dp=index % self.runner.dp_size, hit=0)
        self.runner._continuous_next_seq_id = next_id + len(self.job["requests"])
        self.gates = {}
        owned = self.runner._scheduled_initial_owned
        self.initial_padding_slots = tuple(range(len(owned), self.runner.slots))
        if self.context_view is not None:
            self.context_view.initialize(self.job["requests"][0]["token_ids"])
            self.padding_bootstrap_tokens = self.context_view.bootstrap_tokens
            self.padding_bootstrap_forwards = self.context_view.bootstrap_forwards
            if self.runner._scheduled_padding_prefix is None:
                self.runner._scheduled_padding_prefix = self.context_view.padding
        for slot in self.initial_padding_slots:
            self.model.restore_prefix(self.runner._scheduled_padding_prefix, slot)
            self.gates[slot] = SlotHandoff(self.model, self.worker, slot, None, slots_per_dp=4)
        for slot, ((index, request), snapshot) in enumerate(zip(owned, snapshots, strict=True)):
            if snapshot is not None:
                validate_prefix(self.model, request["token_ids"], snapshot)
                self.model.restore_prefix(snapshot, slot)
            seq_id = next_id + index
            hit = self.model.next_positions[slot]
            self.records[seq_id]["hit"] = self.hits[seq_id] = hit
            binding = self.bindings[slot] = SlotBinding(seq_id, 0)
            self.gates[slot] = SlotHandoff(self.model, self.worker, slot, binding, slots_per_dp=4)
            self.ledger.register(slot, binding, request, hit=hit)
            if self.is_driver:
                self.sequence(seq_id)
            self.emit(dict(type="start", index=index, cached_tokens=hit))
        self.admission = self.seam.admission = self.ledger.admission = ContinuousAdmission(
            self.runner, self.worker, self.gates, target=self.target,
            close_pipeline=self.close_target, on_activate=self.activate, context_view=self.context_view)
        self.original_publish = self.ledger.publish_committed
        self.ledger.publish_committed = self.publish
        if self.model.eplb_runtime is not None:self.model.eplb_runtime.begin_session()
        dist.barrier(group=self.control)
        if self.is_driver:
            atomic(self.directory / f"admission-ready-dp{self.runner.dp_rank}.json", dict(ready=True))
        deadline = self.started + float(self.job.get("request_timeout", 3600))
        number = 0
        try:
            while time.monotonic() < deadline:
                self.poll()
                if self.is_driver:
                    self.assignments.update_readiness()
                busy = torch.tensor([int(self.is_driver and self.pool.scheduler.has_unfinished_seqs())], dtype=torch.int32)
                dist.all_reduce(busy, op=dist.ReduceOp.MAX, group=self.control)
                if not busy.item():
                    self.drain()
                    if self.closed:
                        break
                    time.sleep(.001)
                    continue
                started = time.monotonic_ns()
                schedule = dict(bindings=self.bindings, sequences=self.sequences,
                    positions=self.admission.scheduled_positions(), slots=4, chunk_size=self.model.chunk_size)
                run_scheduled_step(self.worker, self.seam, number, driver=self.driver,
                    tp_group=self.host_tp, control_group=self.control, root=self.runner.root, schedule=schedule)
                self.ledger.finish_step()
                step = self.seam.step
                self.assignments.finish_step(step.retired_ids)
                self.step_wall.append((number, step.metadata.is_prompt, started, time.monotonic_ns()))
                pending = self.worker.prev_step_state
                self.steps.append(dict(step=number, prompt=step.metadata.is_prompt,
                    rows=[(row.seq_id, row.slot, row.epoch, row.q, row.scheduled_k,
                           row.mtp_read_slot, row.mtp_write_slot) for row in step.metadata.rows],
                    sample_q=None if pending is None else pending.sampler_output.target.logical_q,
                    admissions=step.admissions))
                number += 1
            else:
                raise TimeoutError("continuous Scheduler session timed out")
            self.drain()
            if self.pool is not None:
                self.pool.end()
            self.save()
        except BaseException:
            if self.pool is not None:
                self.pool.poisoned = True
            raise

    def save(self):
        records = sorted(((seq_id, record) for seq_id, record in self.records.items()
                          if record["dp"] == self.runner.dp_rank), key=lambda item: item[1]["index"])
        for seq_id, record in records:
            if seq_id not in self.done or self.finish_reasons.get(seq_id) != self.completion_reason(seq_id):
                raise RuntimeError("continuous original request did not complete")
        stats = dict(self.model.runtime_stats(), zos_official_scheduler_used=True,
            zos_worker_deferred_post=self.worker_deferred_post,
            zos_worker_pending_decode_steps=sum(not x["prompt"] and x["sample_q"] is not None for x in self.steps),
            zos_official_scheduler_steps=len(self.steps), zos_official_scheduler_prompt_steps=sum(x["prompt"] for x in self.steps),
            zos_continuous_admission_used=True, zos_continuous_requests=len(records),
            zos_continuous_small_enabled=os.environ.get("DEEPINFER_910C_ZOS_CONTINUOUS_SMALL") == "1",
            zos_initial_real_requests=len(self.runner._scheduled_initial_owned),
            zos_initial_padding_slots=list(self.initial_padding_slots),
            zos_continuous_cold_enabled=self.cold_enabled,
            zos_cold_requests=sum(record["hit"] == 0 for _, record in records),
            zos_cold_late_requests=sum(record["hit"] == 0 and record["index"] >= len(self.job["requests"])
                                       for _, record in records),
            zos_padding_bootstrap_tokens=self.padding_bootstrap_tokens,
            zos_padding_bootstrap_forwards=self.padding_bootstrap_forwards,
            zos_cold_prefix_captures=self.runner._scheduled_cold_prefix_captures,
            zos_cold_context_view=None if self.context_view is None else self.context_view.stats(),
            zos_continuous_eos_enabled=os.environ.get("DEEPINFER_910C_ZOS_CONTINUOUS_EOS") == "1",
            zos_eos_stops=sum(self.finish_reasons[seq_id] == "stop" for seq_id, _ in records),
            zos_persistent_scheduler_sessions=0 if self.pool is None else self.pool.sessions,
            zos_async_sample_inputs=dict(enabled=self.target.async_sample_inputs,
                submissions=self.staging_submissions, pinned=self.pinned))
        stats['yoco_prefill_decisions'] = list(self.runner._official_yoco_decisions)
        stats['zos_official_multigraph']=dict(enabled=self.target.multigraph,
            records=list(self.target.graph_records),ordinary_eos_fallback='full24')
        profile = dict(step_wall=self.step_wall, phase_marks=self.seam.phase_marks,
            cpu_timer=self.worker._local_cpu_timer.samples, device_events=self.seam.device_timings,
            initial_curve=self.model.dspark_graphs.curve,
            fixed_physical_frame=None if self.target.multigraph else 24,
            graph_records=list(self.target.graph_records),
            sample_input_staging=stats["zos_async_sample_inputs"])
        rank = dist.get_rank()
        atomic(self.directory / f"scheduled-rank{rank}.json", dict(state="PASS", rank=rank,
            actual={str(seq_id): self.ledger.results[seq_id] for seq_id, _ in records},
            hits=self.hits, projected=self.steps, profile=profile, trace=self.trace,
            local_slot_drains=self.admission.drains, kv_positions=self.model.next_positions[:4]))
        if self.is_driver:
            atomic(self.directory / f"result-dp{self.runner.dp_rank}.json", dict(state="PASS",
                dp_rank=self.runner.dp_rank, internal_replica_work=False,
                request_indices=[record["index"] for _, record in records],
                cached_tokens=[record["hit"] for _, record in records],
                generated_token_ids=[self.ledger.results[seq_id] for seq_id, _ in records],
                finish_reasons=[self.finish_reasons[seq_id] for seq_id, _ in records],
                seconds=time.monotonic() - self.started, runtime_stats=stats,
                peak_npu_allocated_bytes=torch.npu.max_memory_allocated()))
        dist.barrier(group=self.control)
