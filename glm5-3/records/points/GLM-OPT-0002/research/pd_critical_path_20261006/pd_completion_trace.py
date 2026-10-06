"""Opt-in PD host-boundary diagnostic. Never equates a CPU marker with KV ready.

Install only in a frozen diagnostic startup, after matching source_identity.json.
No imports of vLLM/torch, device synchronization, polling, scheduling changes,
future consumption, or token/prompt capture. uninstall() restores native methods.
"""
from __future__ import annotations

import functools
import json
import os
import threading
import time
from pathlib import Path


class Trace:
    def __init__(self, directory, *, host, boot_id, role, rank=None, limit=4096):
        if limit <= 0:
            raise ValueError("positive event limit required")
        self.identity = dict(host=host, boot_id=boot_id, role=role, rank=rank,
                             pid=os.getpid())
        self.limit, self.count = limit, 0
        self.failed = False
        self.lock = threading.Lock()
        self.fd = os.open(Path(directory) / f"{role}-{os.getpid()}-{time.time_ns()}.jsonl",
                          os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)

    def emit(self, event, **fields):
        """Best effort; loss/error is explicit and must invalidate a reduction."""
        stamp = time.monotonic_ns()
        with self.lock:
            if self.failed or self.count > self.limit:
                return
            if self.count == self.limit:
                event, fields = "trace.truncated", {}
            self.count += 1
            try:
                row = dict(self.identity, event=event, monotonic_ns=stamp,
                           seq=self.count, tid=threading.get_native_id(), **fields)
                raw = (json.dumps(row, separators=(",", ":")) + "\n").encode()
                # A partial write leaves malformed JSON; it is not repaired or
                # silently treated as a complete measurement.
                if os.write(self.fd, raw) != len(raw):
                    self.failed = True
            except Exception:
                self.failed = True

    def close(self):
        with self.lock:
            if self.fd is not None:
                os.close(self.fd)
                self.fd = None


class Hooks:
    """Observe existing calls once; preserve returned objects and exceptions."""
    def __init__(self, trace):
        self.trace, self.originals = trace, []

    def observe(self, event, callback):
        try:
            self.trace.emit(event, **callback())
        except Exception as error:
            # A broken field extractor must not change the native call.
            try:
                self.trace.emit("trace.extractor_error", site=event,
                                error=type(error).__name__)
            except Exception:
                pass

    def wrap(self, target, name, before=None, after=None):
        original = getattr(target, name)
        # Restore class descriptors/instance inheritance exactly, not just the
        # bound callable (important for opt-in removal and static methods).
        local = name in vars(target)
        descriptor = vars(target).get(name)

        @functools.wraps(original)
        def call(*args, **kwargs):
            if before:
                self.observe(name + ".begin", lambda: before(*args, **kwargs))
            try:
                value = original(*args, **kwargs)
            except BaseException as error:
                self.observe(name + ".error", lambda: dict(error=type(error).__name__))
                raise
            if after:
                self.observe(name + ".end", lambda: after(value, *args, **kwargs))
            return value

        setattr(target, name, call)
        self.originals.append((target, name, local, descriptor))

    def uninstall(self):
        for target, name, local, descriptor in reversed(self.originals):
            if local:
                setattr(target, name, descriptor)
            else:
                delattr(target, name)
        self.originals.clear()


def _ids(values):
    return sorted(values or ())


def install_worker(module, trace):
    """Install in a D worker process only. Tracker markers remain host markers."""
    hooks = Hooks(trace)
    bindings = {}

    def bind(worker, metadata):
        thread = worker.kv_recv_thread
        if thread is not None:
            bindings[id(thread.task_tracker)] = dict(engine=thread.local_engine_id,
                                                     contributor=worker.tp_rank)
        return dict(request_ids=_ids(metadata.requests),
                    engine=worker.engine_id if hasattr(worker, "engine_id") else None,
                    contributor=worker.tp_rank)

    hooks.wrap(module.MooncakeConnectorWorker, "start_load_kv", before=bind)
    hooks.wrap(module.KVCacheTaskTracker, "update_done_task_count",
               before=lambda tracker, request_id: dict(
                   request_id=request_id, tracker=id(tracker),
                   tracked=request_id in tracker.reqs_to_process,
                   **bindings.get(id(tracker), {})),
               after=lambda value, tracker, request_id: dict(
                   request_id=request_id, tracker=id(tracker),
                   **bindings.get(id(tracker), {})))

    def transfer_info(thread, meta):
        return dict(request_id=meta["request_id"],
                    remote_request_id=meta["remote_request_id"],
                    engine=thread.local_engine_id,
                    contributor=thread.tp_rank,
                    final_pull=meta["all_task_done"])

    hooks.wrap(module.KVCacheRecvingThread, "_transfer_kv_cache_all_groups",
               before=transfer_info,
               after=lambda value, *args: transfer_info(*args))
    hooks.wrap(module.KVCacheRecvingThread, "_reformat_pending_kv_caches",
               before=lambda thread, request_id: dict(request_id=request_id),
               after=lambda value, thread, request_id: dict(request_id=request_id,
                                                            device_ready=False))
    hooks.wrap(module.KVCacheRecvingThread, "_mark_failed_recv_request",
               before=lambda thread, request_id, blocks: dict(request_id=request_id))
    hooks.wrap(module.MooncakeConnectorWorker, "get_finished",
               after=lambda value, worker: dict(send_ids=_ids(value[0]),
                                                recv_ids=_ids(value[1]),
                                                contributor=worker.tp_rank))
    return hooks


def install_core(engine, aggregator_class, trace):
    """Observe native admission and aggregation without replacing policies."""
    hooks = Hooks(trace)
    scheduler = engine.scheduler
    batch = 0

    def begin_schedule(*args, **kwargs):
        nonlocal batch
        batch += 1
        return dict(batch=batch, running=len(scheduler.running),
                    max_running=scheduler.max_num_running_reqs,
                    max_tokens=scheduler.max_num_scheduled_tokens,
                    queue_depth=len(engine.batch_queue or ()),
                    # These counts are context; they are not an eligibility proof.
                    waiting_remote=_ids(r.request_id for r in scheduler.requests.values()
                                        if r.status.name == "WAITING_FOR_REMOTE_KVS"))

    hooks.wrap(scheduler, "schedule", before=begin_schedule,
               after=lambda out, *args, **kwargs: dict(
                   batch=batch, tokens=dict(out.num_scheduled_tokens),
                   total_tokens=out.total_num_scheduled_tokens))
    hooks.wrap(scheduler, "_update_from_kv_xfer_finished",
               before=lambda output: dict(recv_ids=_ids(output.finished_recving),
                                           invalid_blocks=_ids(output.invalid_block_ids)),
               after=lambda value, output: dict(recv_ids=_ids(output.finished_recving)))
    hooks.wrap(scheduler, "_try_promote_blocked_waiting_request",
               before=lambda request: dict(request_id=request.request_id,
                                            status=request.status.name),
               after=lambda value, request: dict(request_id=request.request_id,
                                                 promoted=bool(value),
                                                 status=request.status.name))
    hooks.wrap(scheduler, "update_from_output",
               after=lambda outputs, scheduled, model_output, *args, **kwargs: dict(
                   request_ids=list(model_output.req_ids),
                   output_request_counts={str(client): len(value.outputs)
                                  for client, value in outputs.items()}))
    # Native aggregate mutates outputs[output_rank]. Capture the input contributor
    # sets before the native call, not from the already aggregated return object.
    hooks.wrap(aggregator_class, "aggregate",
               before=lambda agg, outputs, output_rank=0: dict(
                   expected=agg._expected_finished_count,
                   contributors=[dict(index=i, recv_ids=_ids(out.kv_connector_output.finished_recving))
                                 for i, out in enumerate(outputs)
                                 if out is not None and out.kv_connector_output is not None]),
               after=lambda value, agg, outputs, output_rank=0: dict(
                   expected=agg._expected_finished_count,
                   recv_ids=_ids(value.kv_connector_output.finished_recving)
                   if value is not None and value.kv_connector_output is not None else []))
    return hooks
