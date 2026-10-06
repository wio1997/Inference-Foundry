"""Bounded observation of native RPC/commit; no alternate scheduling policy.

Loaded only by the diagnostic scheduler/worker entrypoints. Host timestamps
share CLOCK_MONOTONIC on one host; none is a device execution timestamp.
Worker copy-ready is deliberately separate from PP sampled+draft publication.
"""
from __future__ import annotations

import functools
import json
import os
from pathlib import Path
import threading
import time


class Trace:
    def __init__(self, directory, role, limit=16384):
        self.role = role
        self.limit = limit
        self.count = 0
        self.lock = threading.Lock()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(directory / f"{role}-{os.getpid()}.jsonl",
                          os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)

    def emit(self, event, **fields):
        stamp = time.monotonic_ns()
        with self.lock:
            if self.count >= self.limit:
                return
            self.count += 1
            row = dict(event=event, monotonic_ns=stamp, role=self.role,
                       pid=os.getpid(), tid=threading.get_native_id(), **fields)
            os.write(self.fd, (json.dumps(row, separators=(",", ":")) + "\n").encode())


def _wrap(obj, name, trace, fields, start=None, finish=None):
    original = getattr(obj, name)

    @functools.wraps(original)
    def call(*args, **kwargs):
        info = fields(*args, **kwargs)
        if start:
            start(*args, **kwargs)
        trace.emit(name + ".begin", **info)
        try:
            value = original(*args, **kwargs)
        except BaseException as error:
            trace.emit(name + ".error", error=type(error).__name__, **info)
            raise
        if finish:
            finish(value, *args, **kwargs)
        trace.emit(name + ".end", **info)
        return value

    setattr(obj, name, call)
    return original


def request_state(scheduler):
    """Native eligibility inputs, not a promise that KV/device state is ready."""
    return [dict(id=r.request_id, computed=r.num_computed_tokens,
                 placeholders=r.num_output_placeholders,
                 output_tokens=r.num_output_tokens,
                 next_decode_eligible_step=r.next_decode_eligible_step,
                 prefill=r.is_prefill_chunk,
                 num_tokens_with_spec=r.num_tokens_with_spec)
            for r in scheduler.running]


def install_engine(engine, trace):
    if getattr(engine, "_glm_commit_trace", False):
        raise RuntimeError("diagnostic already installed")
    engine._glm_commit_trace = True
    scheduler = engine.scheduler
    executor = engine.model_executor
    sequence = 0
    batches = {}
    active = {}

    def scheduled(output, *args, **kwargs):
        nonlocal sequence
        sequence += 1
        batches[id(output)] = sequence
        active["batch"] = sequence
        trace.emit("schedule.output", batch=sequence,
                   current_step=scheduler.current_step,
                   tokens=dict(output.num_scheduled_tokens),
                   drafts=dict(output.scheduled_spec_decode_tokens),
                   pending_grammar=output.pending_structured_output_tokens,
                   state=request_state(scheduler))

    _wrap(scheduler, "schedule", trace,
          lambda *a, **k: dict(current_step=scheduler.current_step,
                              state=request_state(scheduler)), finish=scheduled)

    def executed(future, output, *args, **kwargs):
        watch_future(future, batches[id(output)], "execute_model")

    def sampled(future, *args, **kwargs):
        watch_future(future, active["batch"], "sample_tokens")

    def watch_future(future, batch, rpc):
        # Observe the existing lazy consumer, never call result/poll/done here.
        if not hasattr(future, "get_response"):
            raise RuntimeError("unexpected native future; stop diagnostic")
        _wrap(future, "get_response", trace,
              lambda: dict(batch=batch, rpc=rpc))

    _wrap(executor, "execute_model", trace,
          lambda output, *a, **k: dict(batch=batches[id(output)]), finish=executed)
    _wrap(executor, "sample_tokens", trace,
          lambda *a, **k: dict(batch=active["batch"]), finish=sampled)

    def committed(value, output, model_output, *args, **kwargs):
        batch = batches.pop(id(output))
        trace.emit("commit.output", batch=batch,
                   state=request_state(scheduler),
                   req_ids=list(model_output.req_ids),
                   sampled=model_output.sampled_token_ids)

    _wrap(scheduler, "update_from_output", trace,
          lambda output, *a, **k: dict(batch=batches[id(output)],
                                      current_step=scheduler.current_step),
          finish=committed)

    original = engine.step_fn

    @functools.wraps(original)
    def step():
        trace.emit("step.begin", queue=len(engine.batch_queue))
        out = original()
        trace.emit("step.end", queue=len(engine.batch_queue),
                   model_executed=out[1], has_output=out[0] is not None)
        return out

    engine.step_fn = step
    trace.emit("installed", kind="engine", scheduling_changes=0,
               queue_capacity=engine.batch_queue_size,
               resource_capacity=scheduler.vllm_config.max_concurrent_batches)


def install_worker(worker, trace):
    """Wrap the already selected native worker; leave every tensor untouched."""
    sequence = 0
    current = {}
    runner = worker.model_runner

    def before_execute(output, *args, **kwargs):
        nonlocal sequence
        sequence += 1
        current.update(batch=sequence, ids=list(output.num_scheduled_tokens), rpc="execute_model")

    _wrap(worker, "execute_model", trace,
          lambda output, *a, **k: dict(batch=sequence + 1,
                                      tokens=dict(output.num_scheduled_tokens)),
          start=before_execute)

    def sampled(output, *args, **kwargs):
        if output is None or not hasattr(output, "get_output"):
            return
        info = dict(current)
        _wrap(output, "get_output", trace, lambda: info)

    _wrap(worker, "sample_tokens", trace,
          lambda *a, **k: dict(current, rpc="sample_tokens"),
          start=lambda *a, **k: current.update(rpc="sample_tokens"), finish=sampled)
    _wrap(runner, "postprocess_sampled", trace,
          lambda *a, **k: dict(current))
    if runner.speculator is not None:
        _wrap(runner.speculator, "propose", trace,
              lambda *a, **k: dict(current))
    if hasattr(runner, "_update_seq_lens_cpu"):
        _wrap(runner, "_update_seq_lens_cpu", trace,
              lambda *a, **k: dict(current))
    if runner.pp_handler is not None:
        pp = runner.pp_handler
        for name in ("get_prev_sampled_outputs", "receive", "broadcast_draft_tokens"):
            if hasattr(pp, name):
                _wrap(pp, name, trace, lambda *a, **k: dict(current))
    # Existing WorkerProc async thread is the only reply producer. Observation
    # retains its FIFO and its original AsyncOutput.get_output consumption.
    from collections import deque
    from vllm.v1.executor.multiproc_executor import WorkerProc
    pending = deque()
    original_handle = WorkerProc.handle_output
    original_enqueue = WorkerProc.enqueue_output

    @functools.wraps(original_handle)
    def handle(proc, output):
        pending.append(dict(current))
        return original_handle(proc, output)

    @functools.wraps(original_enqueue)
    def enqueue(proc, output):
        info = pending.popleft()
        trace.emit("reply.begin", **info)
        value = original_enqueue(proc, output)
        # This is an upper bound on actual MQ publication, not its exact time.
        trace.emit("reply.published_upper_bound", **info)
        return value

    WorkerProc.handle_output = handle
    WorkerProc.enqueue_output = enqueue
    trace.emit("installed", kind="worker", rank=worker.rank,
               runner=type(runner).__module__ + "." + type(runner).__name__,
               scheduling_changes=0, device_timestamps=False)


def install_engine_class(directory):
    """Called after original queue-cap install, before EngineCore sets step_fn."""
    from vllm.v1.engine.core import EngineCore
    original = EngineCore.step_with_batch_queue

    @functools.wraps(original)
    def step(self):
        if not getattr(self, "_glm_commit_trace", False):
            install_engine(self, Trace(directory, "core"))
        return original(self)

    EngineCore.step_with_batch_queue = step
