#!/usr/bin/env python3
"""E33 vLLM Ascend CPU/observation telemetry.

This module is deliberately small and stdlib-only at import time.  It installs
wrappers on a fixed list of already-imported vLLM/vLLM-Ascend modules; it never
imports vLLM, torch or torch_npu itself.  The companion
``tools/e33_vllm_bootstrap/sitecustomize.py`` loads this file by path only when
``E33_DIAG_ROOT`` is set and calls :func:`install_for_module` after each target
module has finished loading.

Design constraints from the E33 task:
  * wrappers preserve return objects and exceptions exactly;
  * shared batches produce one scheduler-side record, not one per rank/request;
  * per-request speculative-decode work is recorded from the arguments of
    ``Scheduler.make_spec_decoding_stats`` and never from cumulative rank stats;
  * no SchedulerOutput object is modified;
  * snapshots are quiescent boundary observations and do not copy tensor data;
  * JSONL is buffered and flushed every N records, never per layer/kernel;
  * no automatic profiler stop on a step count.

The module is intentionally not a general instrumentation framework.  If a
hook target or method is absent, the gap is recorded as a ``missing`` entry in
an instrumentation event and in :func:`coverage_report`.
"""

from __future__ import annotations

import atexit
import functools
import inspect
import json
import os
import re
import sys
import threading
import time
import weakref
from contextvars import ContextVar

SCHEMA = "e33.vllm.telemetry/1"
FRAMEWORK = "vllm-ascend"
FLUSH_EVERY_RECORDS = 256
MAX_EVENT_LIST = 128
MAX_TENSOR_ENTRIES = 20000
MAX_SCAN_ITEMS = 256
MAX_STRING = 4000

TARGET_MODULES = (
    "vllm.v1.core.sched.scheduler",
    "vllm.v1.core.sched.output",
    "vllm.v1.engine.core",
    "vllm.config.profiler",
    "vllm_ascend.profiler.torch_npu_profiler",
    "vllm_ascend.worker.worker",
    "vllm_ascend.worker.model_runner_v1",
    "vllm_ascend.worker.v2.model_runner",
    "vllm.entrypoints.openai.completion.serving",
    "vllm.entrypoints.openai.completion.protocol",
    "vllm.entrypoints.openai.api_server",
)

_CURRENT_REQUEST = ContextVar("e33_current_request", default=None)
_STREAM_CTX = ContextVar("e33_stream_ctx", default=None)

_LOCK = threading.RLock()
_WRITER = None
_WRITER_PID = None
_WRITE_ERRORS = 0
_SEQ = 0
_ATEXIT_REGISTERED = False
_FINAL_COVERAGE_REGISTERED = False
_SOURCE_COUNTS = {}
_IDENT_CACHE = weakref.WeakKeyDictionary()
_ORDINALS = {}
_ORDINALS_LOCK = threading.Lock()
_PATCH_REPORTS = {}
_REPORTED_MODULES = set()
_PROFILER_EFFECTIVE = None
_PROFILER_PATCH_STATUS = "not_attempted"
_UNSET = object()
_WINDOW_CACHE = (0.0, "unknown", "unknown")


# ---------------------------------------------------------------------------
# JSONL writer
# ---------------------------------------------------------------------------

def _diag_root() -> str | None:
    root = os.environ.get("E33_DIAG_ROOT")
    return root.rstrip("/") if root else None


def _json_default(value):
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"bytes_len": len(value)}
    return {"unserializable_type": type(value).__name__}


def _sanitize(value, depth=0):
    """Bound unknown values before they reach json.dumps."""
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, str) and len(value) > MAX_STRING:
            return value[:MAX_STRING] + "...<truncated>"
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"bytes_len": len(value)}
    if depth >= 4:
        return {"type": type(value).__name__, "depth_limit": True}
    if isinstance(value, dict):
        items = list(value.items())[:64]
        out = {}
        for key, item in items:
            out[str(key)[:200]] = _sanitize(item, depth + 1)
        if len(value) > len(items):
            out["__truncated_keys__"] = len(value) - len(items)
        return out
    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)[:MAX_EVENT_LIST]
        out = [_sanitize(item, depth + 1) for item in items]
        if len(value) > len(items):
            out.append({"__truncated_items__": len(value) - len(items)})
        return out
    return {"type": type(value).__name__}


class _Writer:
    def __init__(self, root: str, pid: int):
        self.root = root
        self.pid = pid
        self.dir = os.path.join(root, "jsonl")
        os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, "pid-%d.jsonl" % pid)
        self.fh = open(self.path, "a", encoding="utf-8", buffering=1 << 20)
        self.pending = 0
        self.closed = False

    def write(self, record: dict) -> None:
        global _WRITE_ERRORS
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"),
                          default=_json_default)
        with _LOCK:
            if self.closed:
                return
            try:
                self.fh.write(line)
                self.fh.write("\n")
                self.pending += 1
                if self.pending >= FLUSH_EVERY_RECORDS:
                    self.fh.flush()
                    self.pending = 0
            except Exception:
                _WRITE_ERRORS += 1

    def close(self) -> None:
        with _LOCK:
            if self.closed:
                return
            self.closed = True
            try:
                self.fh.flush()
                self.fh.close()
            except Exception:
                pass


def _get_writer() -> _Writer | None:
    global _WRITER, _WRITER_PID, _ATEXIT_REGISTERED, _FINAL_COVERAGE_REGISTERED
    root = _diag_root()
    if not root:
        return None
    pid = os.getpid()
    with _LOCK:
        if _WRITER is not None and _WRITER_PID == pid and not _WRITER.closed:
            return _WRITER
        if _WRITER is not None and _WRITER_PID != pid:
            # After fork the parent's fd must not be reused by the child.
            _WRITER = None
        try:
            _WRITER = _Writer(root, pid)
            _WRITER_PID = pid
        except Exception:
            _WRITER = None
            _WRITER_PID = None
            return None
        if not _ATEXIT_REGISTERED:
            atexit.register(_close_writer)
            _ATEXIT_REGISTERED = True
        if not _FINAL_COVERAGE_REGISTERED:
            # Registered after _close_writer, so LIFO runs coverage first.
            atexit.register(_write_final_coverage)
            _FINAL_COVERAGE_REGISTERED = True
        return _WRITER


def _write_final_coverage() -> None:
    try:
        report = coverage_report()
        report["not_loaded"] = [name for name in TARGET_MODULES
                                if name not in _PATCH_REPORTS]
        report["loaded_without_hooks"] = [
            name for name, item in _PATCH_REPORTS.items() if not item.get("hooks")
        ]
        _record_event("coverage_final", "vllm.telemetry.coverage", None,
                      scope="process", work=report, sanitize_payload=False)
    except Exception:
        pass


def _close_writer() -> None:
    global _WRITER
    with _LOCK:
        if _WRITER is not None:
            _WRITER.close()
            _WRITER = None


def _flush_writer() -> None:
    """Coarse boundary flush (profile stop); never called per layer/kernel."""
    with _LOCK:
        writer = _WRITER
        if writer is None or writer.closed:
            return
        try:
            writer.fh.flush()
            writer.pending = 0
        except Exception:
            pass


def _run_id() -> tuple[str, str]:
    global _WINDOW_CACHE
    now = time.monotonic()
    if now < _WINDOW_CACHE[0]:
        return _WINDOW_CACHE[1], _WINDOW_CACHE[2]
    root = _diag_root()
    if root:
        path = os.path.join(root, "window-context.json")
        try:
            with open(path, "r", encoding="utf-8") as fh:
                context = json.load(fh)
            value = context.get("run_id")
            if isinstance(value, str) and value:
                _WINDOW_CACHE = (now + 0.2, value, "window-context.json")
                return value, "window-context.json"
        except FileNotFoundError:
            pass
        except Exception:
            _WINDOW_CACHE = (now + 0.2, "unknown", "invalid-window-context")
            return "unknown", "invalid-window-context"
    value = os.environ.get("E33_RUN_ID", "").strip()
    if value and value.lower() != "unknown":
        _WINDOW_CACHE = (now + 0.2, value, "env:E33_RUN_ID")
        return value, "env:E33_RUN_ID"
    _WINDOW_CACHE = (now + 0.2, "unknown", "unknown")
    return "unknown", "unknown"


def _next_seq() -> int:
    global _SEQ
    with _LOCK:
        _SEQ += 1
        return _SEQ


def _record_event(event: str, source: str, owner=None, runner=None, *,
                  scope: str = "process", request_id=None,
                  request_id_source: str | None = None,
                  iteration_id=None, iteration_id_source: str | None = None,
                  work=None, extra=None, exception=None,
                  start: tuple | None = None, end: tuple | None = None,
                  sanitize_payload: bool = True) -> None:
    """Append one JSONL record.  Telemetry failures never reach the caller."""
    writer = _get_writer()
    if writer is None:
        return
    try:
        run_id, run_id_source = _run_id()
        ident = _identity_for(owner, runner)
        now = _clock_triplet()
        start = start or now
        end = end or now
        rec = {
            "schema": SCHEMA,
            "event_seq": _next_seq(),
            "event": event,
            "framework": FRAMEWORK,
            "source": source,
            "scope": scope,
            "run_id": run_id,
            "run_id_source": run_id_source,
            "pid": os.getpid(),
            "native_tid": _native_tid(),
            "identity": ident,
            **ident,
            "iteration_id": iteration_id,
            "iteration_id_source": iteration_id_source,
            "request_id": request_id,
            "request_id_source": request_id_source,
            "wall_ns": start[0],
            "monotonic_ns": start[1],
            "thread_cpu_ns": start[2],
            "wall_end_ns": end[0],
            "monotonic_end_ns": end[1],
            "thread_cpu_end_ns": end[2],
            "start_ns": start[1],
            "end_ns": end[1],
            "duration_ns": end[1] - start[1],
            "thread_cpu_delta_ns": end[2] - start[2],
            "exception": _exception_dict(exception),
            "work": (_sanitize(work or {}) if sanitize_payload else (work or {})),
            "extra": (_sanitize(extra or {}) if sanitize_payload else (extra or {})),
        }
        writer.write(rec)
        with _LOCK:
            _SOURCE_COUNTS[source] = _SOURCE_COUNTS.get(source, 0) + 1
    except Exception:
        global _WRITE_ERRORS
        _WRITE_ERRORS += 1


def _clock_triplet() -> tuple[int, int, int]:
    return (time.time_ns(), time.monotonic_ns(), time.thread_time_ns())


def _native_tid() -> int:
    try:
        return threading.get_native_id()
    except Exception:
        return threading.get_ident()


def _exception_dict(exc) -> dict | None:
    if exc is None:
        return None
    try:
        message = str(exc)
    except Exception:
        message = "<unprintable>"
    if len(message) > MAX_STRING:
        message = message[:MAX_STRING] + "...<truncated>"
    return {
        "type": type(exc).__name__,
        "module": type(exc).__module__,
        "message": message,
        "is_cancelled": type(exc).__name__ in ("CancelledError", "GeneratorExit"),
    }


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def _int_or_none(value):
    try:
        if isinstance(value, bool):
            return None
        return int(value)
    except Exception:
        return None


def _group_rank(name: str) -> tuple[int | None, int | None, str | None]:
    """Read an initialized vLLM group without importing vLLM."""
    for module_name in ("vllm.distributed.parallel_state",
                        "vllm_ascend.distributed.parallel_state"):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        group = None
        for attr in (name, "get_%s" % name):
            fn = getattr(module, attr, None)
            if callable(fn):
                try:
                    group = fn()
                except Exception:
                    group = None
                if group is not None:
                    break
        if group is None:
            continue
        rank = _int_or_none(getattr(group, "rank_in_group", None))
        if rank is None:
            rank = _int_or_none(getattr(group, "rank", None))
        world = _int_or_none(getattr(group, "world_size", None))
        return rank, world, module_name
    return None, None, None


def _identity_for(owner=None, runner=None) -> dict:
    key = runner if runner is not None else owner
    if key is not None:
        try:
            cached = _IDENT_CACHE.get(key)
            if cached is not None:
                return dict(cached)
        except TypeError:
            pass
    ident = _build_identity(owner, runner)
    if key is not None:
        try:
            _IDENT_CACHE[key] = dict(ident)
        except TypeError:
            pass
    return ident


def _build_identity(owner, runner) -> dict:
    cfg = None
    if owner is not None:
        cfg = getattr(owner, "vllm_config", None)
    if cfg is None and runner is not None:
        cfg = getattr(runner, "vllm_config", None)
    pc = getattr(cfg, "parallel_config", None) if cfg is not None else None
    if pc is None and owner is not None:
        pc = getattr(owner, "parallel_config", None)
    if pc is None and runner is not None:
        pc = getattr(runner, "parallel_config", None)

    rank = _int_or_none(getattr(owner, "rank", None)) if owner is not None else None
    local_rank = _int_or_none(getattr(owner, "local_rank", None)) if owner is not None else None
    device = getattr(owner, "device", None) if owner is not None else None
    if device is None and runner is not None:
        device = getattr(runner, "device", None)

    dp_rank = _int_or_none(getattr(pc, "data_parallel_rank", None)) if pc is not None else None
    dp_index = _int_or_none(getattr(pc, "data_parallel_index", None)) if pc is not None else None
    dp_local = _int_or_none(getattr(pc, "data_parallel_rank_local", None)) if pc is not None else None
    if dp_rank is None and runner is not None:
        dp_rank = _int_or_none(getattr(runner, "dp_rank", None))

    global_rank = rank
    global_source = "owner.rank" if rank is not None else "unknown"
    if global_rank is None:
        torch_dist = sys.modules.get("torch.distributed")
        if torch_dist is not None:
            try:
                if torch_dist.is_initialized():
                    global_rank = _int_or_none(torch_dist.get_rank())
                    global_source = "torch.distributed.get_rank" if global_rank is not None else "unknown"
            except Exception:
                pass

    ep_rank, ep_world, ep_source = _group_rank("get_ep_group")
    tp_rank, tp_world, tp_source = _group_rank("get_tp_group")
    pp_rank, pp_world, pp_source = _group_rank("get_pp_group")
    dp_group_rank, dp_world, dp_source = _group_rank("get_dp_group")
    if dp_rank is None:
        dp_rank = dp_group_rank

    runner_class = None
    runner_module = None
    if runner is not None:
        runner_class = type(runner).__name__
        runner_module = type(runner).__module__
    elif owner is not None:
        model_runner = getattr(owner, "model_runner", None)
        if model_runner is not None:
            runner_class = type(model_runner).__name__
            runner_module = type(model_runner).__module__

    ident = {
        "rank": rank,
        "global_rank": global_rank,
        "global_rank_source": global_source,
        "local_rank": local_rank,
        "dp_rank": dp_rank,
        "dp_rank_source": ("parallel_config" if dp_rank is not None else "unknown"),
        "dp_index": dp_index,
        "dp_local_rank": dp_local,
        "dp_world_size": _int_or_none(getattr(pc, "data_parallel_size", None)) if pc is not None else dp_world,
        "ep_rank": ep_rank,
        "ep_world_size": ep_world,
        "ep_rank_source": ep_source,
        "tp_rank": tp_rank,
        "tp_world_size": tp_world,
        "tp_rank_source": tp_source,
        "pp_rank": pp_rank,
        "pp_world_size": pp_world,
        "pp_rank_source": pp_source,
        "dp_group_rank": dp_group_rank,
        "dp_group_world_size": dp_world,
        "dp_rank_source_group": dp_source,
        "device": None if device is None else str(device),
        "world_size": _int_or_none(getattr(pc, "world_size", None)) if pc is not None else None,
        "tensor_parallel_size": _int_or_none(getattr(pc, "tensor_parallel_size", None)) if pc is not None else None,
        "pipeline_parallel_size": _int_or_none(getattr(pc, "pipeline_parallel_size", None)) if pc is not None else None,
        "data_parallel_size": _int_or_none(getattr(pc, "data_parallel_size", None)) if pc is not None else None,
        "enable_expert_parallel": bool(getattr(pc, "enable_expert_parallel", False)) if pc is not None else None,
        "runner_class": runner_class,
        "runner_module": runner_module,
        "pid": os.getpid(),
    }
    return ident


# ---------------------------------------------------------------------------
# Span helpers
# ---------------------------------------------------------------------------

class _Span:
    def __init__(self, source, owner=None, runner=None, scope="rank",
                 request_id=None, request_id_source=None,
                 iteration_id=None, iteration_id_source=None,
                 extra=None):
        self.source = source
        self.owner = owner
        self.runner = runner
        self.scope = scope
        self.request_id = request_id
        self.request_id_source = request_id_source
        self.iteration_id = iteration_id
        self.iteration_id_source = iteration_id_source
        self.extra = extra or {}
        self.start = _clock_triplet()

    def close(self, work=None, exception=None, extra=None):
        merged_extra = dict(self.extra)
        if extra:
            merged_extra.update(extra)
        _record_event(
            "span", self.source, self.owner, self.runner,
            scope=self.scope,
            request_id=self.request_id,
            request_id_source=self.request_id_source,
            iteration_id=self.iteration_id,
            iteration_id_source=self.iteration_id_source,
            work=work,
            extra=merged_extra,
            exception=exception,
            start=self.start,
            end=_clock_triplet(),
        )


def _ordinal(owner, kind: str) -> int:
    key = (id(owner), kind)
    with _ORDINALS_LOCK:
        value = _ORDINALS.get(key, 0) + 1
        _ORDINALS[key] = value
        if len(_ORDINALS) > 4096:
            _ORDINALS.clear()
            _ORDINALS[key] = value
        return value


def _sched_iteration(owner):
    value = _int_or_none(getattr(owner, "current_step", None))
    return value, ("scheduler.current_step" if value is not None else "unavailable")


def _safe_len(value):
    try:
        return len(value)
    except Exception:
        return None


def _safe_int(value):
    try:
        return int(value)
    except Exception:
        return None


def _scheduler_output_work(scheduler_output) -> dict:
    if scheduler_output is None:
        return {}
    num_tokens = getattr(scheduler_output, "num_scheduled_tokens", None) or {}
    spec = getattr(scheduler_output, "scheduled_spec_decode_tokens", None) or {}
    request_ids = []
    try:
        request_ids = list(num_tokens.keys())[:MAX_EVENT_LIST]
    except Exception:
        pass
    spec_tokens = 0
    try:
        for value in spec.values():
            spec_tokens += len(value)
    except Exception:
        spec_tokens = None
    return {
        "total_num_scheduled_tokens": _safe_int(getattr(scheduler_output, "total_num_scheduled_tokens", None)),
        "num_scheduled_requests": _safe_len(num_tokens),
        "scheduled_request_ids": request_ids,
        "num_scheduled_requests_truncated": bool(_safe_len(num_tokens) and _safe_len(num_tokens) > len(request_ids)),
        "num_spec_decode_requests": _safe_len(spec),
        "num_spec_decode_tokens": spec_tokens,
        "num_finished_requests": _safe_len(getattr(scheduler_output, "finished_req_ids", None)),
        "num_invalid_spec_tokens": _safe_len(getattr(scheduler_output, "num_invalid_spec_tokens", None)),
        "note": "shared batch; request_ids are the batch members, work is recorded once",
    }


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------

def _request_id_from_call(self, name, args, kwargs):
    if name == "add_request" and len(args) > 1:
        request = args[1]
        return getattr(request, "request_id", None), "request.request_id"
    if name == "make_spec_decoding_stats":
        value = kwargs.get("request_id")
        if value is None and len(args) > 5:
            value = args[5]
        if value is not None:
            return value, "call.request_id"
    return None, None


def _wrap_method(cls, name, source, scope, work_fn=None,
                 request_id_fn=None, iteration_fn=None,
                 extra_fn=None, after_fn=None):
    orig = getattr(cls, name, None)
    if not callable(orig):
        return False, "method not found"
    if getattr(orig, "__e33_wrapped__", False):
        return True, "already wrapped"

    is_coro = inspect.iscoroutinefunction(orig)
    is_asyncgen = inspect.isasyncgenfunction(orig)
    if is_asyncgen:
        # Async generators need a per-__anext__ wrapper (used explicitly for
        # completion_stream_generator); a normal span around creation would
        # misattribute the whole request body.
        return False, "async generator requires dedicated wrapper"

    if is_coro:
        @functools.wraps(orig)
        async def wrapper(*args, **kwargs):
            self = args[0] if args else None
            rid, rid_src = (request_id_fn or _request_id_from_call)(self, name, args, kwargs)
            iid, iid_src = (iteration_fn or _default_iteration)(self, name)
            span = _Span(source, self, scope=scope, request_id=rid,
                         request_id_source=rid_src, iteration_id=iid,
                         iteration_id_source=iid_src)
            try:
                result = await orig(*args, **kwargs)
            except BaseException as exc:
                span.close(exception=exc)
                raise
            span.close(work=_call_work(work_fn, self, args, kwargs, result))
            if after_fn is not None:
                try:
                    after_fn(self, args, kwargs, result)
                except Exception:
                    pass
            return result
    else:
        @functools.wraps(orig)
        def wrapper(*args, **kwargs):
            self = args[0] if args else None
            rid, rid_src = (request_id_fn or _request_id_from_call)(self, name, args, kwargs)
            iid, iid_src = (iteration_fn or _default_iteration)(self, name)
            span = _Span(source, self, scope=scope, request_id=rid,
                         request_id_source=rid_src, iteration_id=iid,
                         iteration_id_source=iid_src)
            try:
                result = orig(*args, **kwargs)
            except BaseException as exc:
                span.close(exception=exc)
                raise
            span.close(work=_call_work(work_fn, self, args, kwargs, result))
            if after_fn is not None:
                try:
                    after_fn(self, args, kwargs, result)
                except Exception:
                    pass
            return result

    wrapper.__e33_wrapped__ = True
    wrapper.__e33_source__ = source
    setattr(cls, name, wrapper)
    return True, "wrapped"


def _default_iteration(self, name):
    value, source = _sched_iteration(self)
    if value is not None:
        return value, source
    return _ordinal(self, "iteration"), "local_ordinal"


def _call_work(work_fn, self, args, kwargs, result):
    if work_fn is None:
        return None
    try:
        return work_fn(self, args, kwargs, result)
    except Exception:
        return {"work_extraction_error": True}


def _work_schedule(self, args, kwargs, result):
    return _scheduler_output_work(result)


def _work_add_request(self, args, kwargs, result):
    request = args[1] if len(args) > 1 else kwargs.get("request")
    prompt_ids = getattr(request, "prompt_token_ids", None)
    return {
        "request_id": getattr(request, "request_id", None),
        "prompt_tokens": _safe_len(prompt_ids),
        "max_tokens": _safe_int(getattr(getattr(request, "sampling_params", None), "max_tokens", None)),
        "num_computed_tokens": _safe_int(getattr(request, "num_computed_tokens", None)),
        "scope_note": "one record per add_request",
    }


def _work_update_from_output(self, args, kwargs, result):
    scheduler_output = args[1] if len(args) > 1 else kwargs.get("scheduler_output")
    model_output = args[2] if len(args) > 2 else kwargs.get("model_runner_output")
    work = _scheduler_output_work(scheduler_output)
    sampled = None
    try:
        sampled = getattr(model_output, "sampled_token_ids", None)
        if sampled is not None:
            work["sampled_token_id_rows"] = _safe_len(sampled)
            work["sampled_token_count"] = sum(
                _safe_len(row) or 0 for row in list(sampled)[:MAX_EVENT_LIST]
            )
    except Exception:
        pass
    if result is not None:
        try:
            work["output_client_shards"] = _safe_len(result)
            work["output_requests"] = sum(
                _safe_len(batch) or 0 for batch in list(result.values())[:MAX_EVENT_LIST]
            )
        except Exception:
            pass
    work["note"] = "one scheduler update per step; not multiplied by worker ranks"
    return work


def _work_spec_stats(self, args, kwargs, result):
    spec_stats = args[1] if len(args) > 1 else kwargs.get("spec_decoding_stats")
    num_draft = args[2] if len(args) > 2 else kwargs.get("num_draft_tokens")
    num_accepted = args[3] if len(args) > 3 else kwargs.get("num_accepted_tokens")
    invalid = args[4] if len(args) > 4 else kwargs.get("num_invalid_spec_tokens")
    request_id = args[5] if len(args) > 5 else kwargs.get("request_id")
    invalid_for_request = None
    try:
        invalid_for_request = invalid.get(request_id, 0) if invalid else 0
    except Exception:
        invalid_for_request = None
    return {
        "request_id": request_id,
        "num_draft_tokens": _safe_int(num_draft),
        "num_accepted_tokens": _safe_int(num_accepted),
        "num_invalid_spec_tokens_for_request": _safe_int(invalid_for_request),
        "result_none": result is None,
        "result_cumulative_not_recorded": True,
        "scope_note": "one observation per make_spec_decoding_stats call; args are per-request",
    }


def _work_engine_step(self, args, kwargs, result):
    scheduler = getattr(self, "scheduler", None)
    work = {}
    outputs = result[0] if isinstance(result, tuple) and result else None
    executed = result[1] if isinstance(result, tuple) and len(result) > 1 else None
    work["model_executed"] = bool(executed) if executed is not None else None
    work["output_shards"] = _safe_len(outputs)
    work["scheduler_step"] = _safe_int(getattr(scheduler, "current_step", None))
    work["note"] = ("engine step span; nested scheduler.schedule/update spans are "
                    "decomposed separately; batch-queue APIs may return before "
                    "the queued forward finishes")
    return work


def _work_worker_execute(self, args, kwargs, result):
    scheduler_output = args[1] if len(args) > 1 else kwargs.get("scheduler_output")
    work = _scheduler_output_work(scheduler_output)
    work["result_type"] = type(result).__name__ if result is not None else None
    return work


def _work_prepare_inputs(self, args, kwargs, result):
    scheduler_output = args[1] if len(args) > 1 else kwargs.get("scheduler_output")
    work = _scheduler_output_work(scheduler_output)
    work["result_type"] = type(result).__name__ if result is not None else None
    return work


def _work_load_model(self, args, kwargs, result):
    runner = getattr(self, "model_runner", None)
    return {
        "model_memory_usage_bytes": _safe_int(getattr(runner, "model_memory_usage", None)),
        "runner_class": type(runner).__name__ if runner is not None else None,
    }


def _work_compile(self, args, kwargs, result):
    return {
        "result_type": type(result).__name__ if result is not None else None,
    }


def _profiler_config_summary(owner_or_config):
    cfg = owner_or_config
    if owner_or_config is not None and not hasattr(owner_or_config, "torch_profiler_dir"):
        cfg = getattr(owner_or_config, "profiler_config", None)
    if cfg is None:
        return None
    fields = (
        "profiler", "torch_profiler_dir", "torch_profiler_with_stack",
        "torch_profiler_with_memory", "torch_profiler_record_shapes",
        "ignore_frontend", "max_iterations", "delay_iterations",
        "warmup_iterations", "active_iterations", "wait_iterations",
        "capture_torch_profiler",
    )
    out = {}
    for field in fields:
        try:
            out[field] = _sanitize(getattr(cfg, field))
        except Exception:
            out[field] = None
    return out


def _cohort_clock_marker(owner, is_start):
    """Bracket a real MSTX boundary with host clocks for trace reconciliation."""
    from torch_npu.npu import mstx
    before = _clock_triplet()
    if is_start:
        profiler = getattr(owner, "profiler", None)
        if not getattr(profiler, "_running", False):
            raise RuntimeError("E33 native profiler did not enter running state")
        run_id, _ = _run_id()
        name = "e33_cohort:%s:pid%d" % (run_id, os.getpid())
        marker = mstx.range_start(name, domain="e33")
        if not marker:
            raise RuntimeError("E33 MSTX range_start did not return a valid id")
        owner._e33_clock_marker = (marker, name)
    else:
        state = getattr(owner, "_e33_clock_marker", None)
        if state is None:
            return
        marker, name = state
        mstx.range_end(marker, domain="e33")
        owner._e33_clock_marker = None
    after = _clock_triplet()
    _record_event("cohort_clock_anchor", "root.e33.cohort_clock_marker", owner,
                  scope="rank", start=before, end=after,
                  extra={"boundary": "start" if is_start else "end",
                         "mstx_name": name, "mstx_range_id": marker,
                         "clock_contract": "Trace boundary must lie within the host call bracket after clock mapping; export time is excluded."})


def _wrap_worker_profile(cls):
    orig = getattr(cls, "profile", None)
    if not callable(orig):
        return False, "method not found"
    if getattr(orig, "__e33_wrapped__", False):
        return True, "already wrapped"

    @functools.wraps(orig)
    def wrapper(self, is_start: bool = True, profile_prefix: str | None = None):
        if not is_start:
            _cohort_clock_marker(self, False)
        # Snapshot before start and before stop: both are quiescent boundaries.
        # A start snapshot taken after profiler.start() would pollute the trace.
        tag = "profile_start_before" if is_start else "profile_stop_before"
        _snapshot(self, tag, prefix=profile_prefix,
                  runner=getattr(self, "model_runner", None))
        span = _Span("vllm.worker.NPUWorker.profile", self, scope="rank",
                     iteration_id=None,
                     iteration_id_source="not_applicable",
                     extra={"is_start": bool(is_start),
                            "profile_prefix": profile_prefix,
                            "profiler_config": _profiler_config_summary(self),
                            "automatic_stop": False})
        try:
            result = orig(self, is_start, profile_prefix)
            if is_start:
                _cohort_clock_marker(self, True)
        except BaseException as exc:
            span.close(exception=exc)
            raise
        profiler = getattr(self, "profiler", None)
        span.close(work={
            "is_start": bool(is_start),
            "profiler_initialized": profiler is not None,
            "profiler_effective": _PROFILER_EFFECTIVE,
            "profiler_override_status": _PROFILER_PATCH_STATUS,
            "stop_is_caller_driven": not bool(is_start),
        })
        if not is_start:
            _flush_writer()
        return result

    wrapper.__e33_wrapped__ = True
    setattr(cls, "profile", wrapper)
    return True, "wrapped"


def _wrap_snapshot_after(cls, name, source, tag):
    orig = getattr(cls, name, None)
    if not callable(orig):
        return False, "method not found"
    if getattr(orig, "__e33_wrapped__", False):
        return True, "already wrapped"

    @functools.wraps(orig)
    def wrapper(*args, **kwargs):
        self = args[0] if args else None
        span = _Span(source, self, scope="rank", iteration_id=None,
                     iteration_id_source="not_applicable")
        try:
            result = orig(*args, **kwargs)
        except BaseException as exc:
            span.close(exception=exc)
            raise
        span.close(work=_work_load_model(self, args, kwargs, result)
                   if name == "load_model" else _work_compile(self, args, kwargs, result))
        try:
            _snapshot(self, tag, runner=getattr(self, "model_runner", None))
        except Exception:
            pass
        return result

    wrapper.__e33_wrapped__ = True
    setattr(cls, name, wrapper)
    return True, "wrapped"


# ---------------------------------------------------------------------------
# HTTP completion async generator + serialization
# ---------------------------------------------------------------------------

def _http_request_fields(request):
    if request is None:
        return {}
    out = {}
    for field in ("stream", "echo", "return_token_ids", "max_tokens", "n",
                  "logprobs", "prompt_logprobs", "stream_interval"):
        try:
            out[field] = _sanitize(getattr(request, field))
        except Exception:
            out[field] = None
    return out


def _wrap_completion_stream(cls):
    name = "completion_stream_generator"
    orig = getattr(cls, name, None)
    if not callable(orig):
        return False, "method not found"
    if getattr(orig, "__e33_wrapped__", False):
        return True, "already wrapped"

    try:
        sig = inspect.signature(orig)
    except Exception:
        sig = None

    @functools.wraps(orig)
    async def wrapper(*args, **kwargs):
        self = args[0] if args else None
        bound = {}
        if sig is not None:
            try:
                bound = sig.bind_partial(*args, **kwargs).arguments
            except Exception:
                bound = {}
        request = bound.get("request")
        request_id = bound.get("request_id")
        rid_source = "call.request_id"
        if request_id is None:
            request_id = getattr(request, "request_id", None)
            rid_source = "request.request_id" if request_id is not None else "unavailable"
        iteration, iteration_source = _sched_iteration(self)
        if iteration is None:
            iteration = _ordinal(self, "stream")
            iteration_source = "local_ordinal"
        inner = orig(*args, **kwargs)
        stream_ctx = {
            "request_id": request_id,
            "serialization_ns": 0,
            "serialization_events": 0,
            "chunks": 0,
            "sse_bytes": 0,
            "sum_anext_ns": 0,
            "first_anext_ns": None,
            "chunk_index": 0,
            "start": _clock_triplet(),
        }
        chunk_index = 0
        status = "completed"
        error = None
        try:
            while True:
                chunk_index += 1
                stream_ctx["chunk_index"] = chunk_index
                ser_before = stream_ctx["serialization_ns"]
                ser_events_before = stream_ctx["serialization_events"]
                ctx_token = _STREAM_CTX.set(stream_ctx)
                req_token = _CURRENT_REQUEST.set(request_id)
                span = _Span("vllm.http.sse_chunk", self, scope="http",
                             request_id=request_id,
                             request_id_source=rid_source,
                             iteration_id=iteration,
                             iteration_id_source=iteration_source,
                             extra={
                                 "chunk_index": chunk_index,
                                 "first_chunk": chunk_index == 1,
                                 "anext_duration_includes_engine_wait": True,
                                 "consumer_wait_not_included": True,
                                 "yield_suspension_not_separately_measured": True,
                                 "sse_framing_time_not_isolated": True,
                             })
                try:
                    item = await inner.__anext__()
                except StopAsyncIteration:
                    # Exhaustion is not an SSE chunk; stream_end records it.
                    break
                except BaseException as exc:
                    span.close(exception=exc)
                    raise
                finally:
                    _CURRENT_REQUEST.reset(req_token)
                    _STREAM_CTX.reset(ctx_token)
                text = item if isinstance(item, str) else str(item)
                anext_ns = time.monotonic_ns() - span.start[1]
                ser_delta = stream_ctx["serialization_ns"] - ser_before
                ser_event_delta = stream_ctx["serialization_events"] - ser_events_before
                work = {
                    "chunk_index": chunk_index,
                    "sse_chars": len(text),
                    "sse_bytes_utf8": len(text.encode("utf-8")),
                    "is_done": text == "data: [DONE]\n\n",
                    "serialization_ns_in_chunk": ser_delta,
                    "serialization_events_in_chunk": ser_event_delta,
                    "anext_total_ns": anext_ns,
                    "residual_ns_includes_upstream_wait": max(0, anext_ns - ser_delta),
                }
                span.close(work=work, extra=_http_request_fields(request))
                stream_ctx["chunks"] += 1
                stream_ctx["sse_bytes"] += work["sse_bytes_utf8"]
                stream_ctx["sum_anext_ns"] += work["anext_total_ns"]
                if stream_ctx["first_anext_ns"] is None:
                    stream_ctx["first_anext_ns"] = work["anext_total_ns"]
                yield item
        except GeneratorExit:
            status = "closed_by_consumer"
            raise
        except BaseException as exc:
            status = "error"
            error = exc
            raise
        finally:
            end = _clock_triplet()
            wall_ns = end[1] - stream_ctx["start"][1]
            _record_event(
                "stream_end", "vllm.http.completion_stream_end", self,
                scope="http", request_id=request_id,
                request_id_source=rid_source, iteration_id=iteration,
                iteration_id_source=iteration_source,
                work={
                    "status": status,
                    "chunks": stream_ctx["chunks"],
                    "sse_bytes_total": stream_ctx["sse_bytes"],
                    "sum_anext_ns": stream_ctx["sum_anext_ns"],
                    "first_anext_ns": stream_ctx["first_anext_ns"],
                    "sum_serialization_ns": stream_ctx["serialization_ns"],
                    "serialization_events": stream_ctx["serialization_events"],
                    "wall_ns_includes_consumer_wait": True,
                    "wall_ns": wall_ns,
                    "consumer_wait_ns_approx": max(0, wall_ns - stream_ctx["sum_anext_ns"]),
                },
                extra=_http_request_fields(request),
                exception=error,
                start=stream_ctx["start"],
                end=end,
            )
            try:
                await inner.aclose()
            except Exception:
                pass

    wrapper.__e33_wrapped__ = True
    setattr(cls, name, wrapper)
    return True, "wrapped"


def _payload_shape(obj):
    shape = {
        "class": type(obj).__name__,
        "json_chars": None,
        "json_bytes_utf8": None,
        "choices": 0,
        "text_chars": 0,
        "prompt_token_ids_len": 0,
        "token_ids_len": 0,
        "has_logprobs": False,
        "routed_experts_b64_chars": 0,
        "prompt_logprobs_present": False,
        "usage_prompt_tokens": None,
        "usage_completion_tokens": None,
        "payload_id": None,
    }
    try:
        shape["payload_id"] = getattr(obj, "id", None)
    except Exception:
        pass
    try:
        usage = getattr(obj, "usage", None)
        if usage is not None:
            shape["usage_prompt_tokens"] = _safe_int(getattr(usage, "prompt_tokens", None))
            shape["usage_completion_tokens"] = _safe_int(getattr(usage, "completion_tokens", None))
    except Exception:
        pass
    try:
        choices = list(getattr(obj, "choices", None) or [])[:8]
        shape["choices"] = len(choices)
        for choice in choices:
            text = getattr(choice, "text", None)
            if isinstance(text, str):
                shape["text_chars"] += len(text)
            pids = getattr(choice, "prompt_token_ids", None)
            if pids is not None:
                shape["prompt_token_ids_len"] += _safe_len(pids) or 0
            tids = getattr(choice, "token_ids", None)
            if tids is not None:
                shape["token_ids_len"] += _safe_len(tids) or 0
            if getattr(choice, "logprobs", None) is not None:
                shape["has_logprobs"] = True
            routed = getattr(choice, "routed_experts", None)
            if isinstance(routed, str):
                shape["routed_experts_b64_chars"] += len(routed)
            if getattr(choice, "prompt_logprobs", None) is not None:
                shape["prompt_logprobs_present"] = True
    except Exception:
        pass
    return shape


def _wrap_serialization(cls, name="model_dump_json"):
    orig = getattr(cls, name, None)
    if not callable(orig):
        return False, "method not found"
    if getattr(orig, "__e33_wrapped__", False):
        return True, "already wrapped"
    source = "vllm.http.json_serialize.%s" % cls.__name__

    @functools.wraps(orig)
    def wrapper(self, *args, **kwargs):
        ctx = _STREAM_CTX.get()
        ctx_request = _CURRENT_REQUEST.get()
        payload_id = None
        try:
            payload_id = getattr(self, "id", None)
        except Exception:
            pass
        request_id = ctx_request if ctx_request is not None else payload_id
        rid_source = "contextvar" if ctx_request is not None else (
            "payload.id" if payload_id is not None else "unavailable")
        span = _Span(source, None, scope="http", request_id=request_id,
                     request_id_source=rid_source, iteration_id=None,
                     iteration_id_source="not_applicable",
                     extra={"exclude_unset": bool(kwargs.get("exclude_unset", False)),
                            "exclude_none": bool(kwargs.get("exclude_none", False)),
                            "stream_chunk_index": (
                                ctx.get("chunk_index") if isinstance(ctx, dict) else None)})
        try:
            result = orig(self, *args, **kwargs)
        except BaseException as exc:
            span.close(exception=exc)
            raise
        elapsed_ns = time.monotonic_ns() - span.start[1]
        text = result if isinstance(result, str) else str(result)
        work = _payload_shape(self)
        work["json_chars"] = len(text)
        try:
            work["json_bytes_utf8"] = len(text.encode("utf-8"))
        except Exception:
            pass
        span.close(work=work)
        if ctx is not None:
            try:
                ctx["serialization_ns"] += elapsed_ns
                ctx["serialization_events"] += 1
            except Exception:
                pass
        return result

    wrapper.__e33_wrapped__ = True
    setattr(cls, name, wrapper)
    return True, "wrapped"


# ---------------------------------------------------------------------------
# Profiler override: Level0 + CPU/NPU + mstx/msprof_tx + op_attr
# ---------------------------------------------------------------------------

def _install_profiler_override(module):
    global _PROFILER_EFFECTIVE, _PROFILER_PATCH_STATUS
    cls = getattr(module, "TorchNPUProfilerWrapper", None)
    if cls is None or not hasattr(cls, "_create_profiler"):
        _PROFILER_PATCH_STATUS = "missing_wrapper"
        return False, "TorchNPUProfilerWrapper._create_profiler not found"
    if getattr(cls._create_profiler, "__e33_wrapped__", False):
        _PROFILER_PATCH_STATUS = "already_wrapped"
        return True, "already wrapped"
    original_is_static = isinstance(cls.__dict__.get("_create_profiler"), staticmethod)

    def _create_profiler(profiler_config, trace_name):
        global _PROFILER_EFFECTIVE
        torch_npu = getattr(module, "torch_npu", None)
        if torch_npu is None:
            raise RuntimeError("torch_npu is not available in the profiler module")
        if profiler_config.profiler != "torch":
            raise RuntimeError(f"Unrecognized profiler: {profiler_config.profiler}")
        if not profiler_config.torch_profiler_dir:
            raise RuntimeError("torch_profiler_dir cannot be empty.")
        # Preserve the original msmonitor-daemon exclusion when the helper exists.
        get_ascend_config = getattr(module, "get_ascend_config", None)
        if callable(get_ascend_config):
            try:
                msmonitor_use_daemon = get_ascend_config().msmonitor_use_daemon
            except Exception:
                msmonitor_use_daemon = False
            if msmonitor_use_daemon:
                raise RuntimeError(
                    "additional_config.msmonitor_use_daemon and torch profiler "
                    "cannot be enabled at the same time."
                )

        supported_exp = _supported_params(torch_npu.profiler._ExperimentalConfig)
        supported_profile = _supported_params(torch_npu.profiler.profile)
        exp_var_kwargs = bool(supported_exp.get("__var_kwargs__"))
        profile_var_kwargs = bool(supported_profile.get("__var_kwargs__"))
        requested = {
            "export_type": ["Text", "Db"],
            "profiler_level": "Level1",
            "mstx": True,
            "msprof_tx": True,
            "op_attr": True,
            "aic_metrics": "AiCoreNone",
            "l2_cache": False,
            "data_simplification": False,
            "record_op_args": False,
            "gc_detect_threshold": None,
        }
        if not exp_var_kwargs and "profiler_level" not in supported_exp:
            raise RuntimeError(
                "Installed torch_npu _ExperimentalConfig has no profiler_level; "
                "cannot force Level0"
            )
        marker_keys = [
            key for key in ("msprof_tx", "mstx")
            if key in supported_exp or exp_var_kwargs
        ]
        if not marker_keys:
            raise RuntimeError(
                "Installed torch_npu _ExperimentalConfig supports neither "
                "msprof_tx nor mstx; MSTX correlation would be absent"
            )
        exp_kwargs = {}
        unsupported_exp = []
        for key, value in requested.items():
            if key not in supported_exp and not exp_var_kwargs:
                unsupported_exp.append(key)
                continue
            if key == "export_type":
                value = [getattr(torch_npu.profiler.ExportType, item) for item in value]
            elif key == "profiler_level":
                value = getattr(torch_npu.profiler.ProfilerLevel, value)
            elif key == "aic_metrics":
                value = getattr(torch_npu.profiler.AiCMetrics, value)
            exp_kwargs[key] = value

        activities = []
        activity_names = []
        for activity_name in ("CPU", "NPU"):
            activity = getattr(torch_npu.profiler.ProfilerActivity, activity_name, None)
            if activity is not None:
                activities.append(activity)
                activity_names.append(activity_name)
        if "CPU" not in activity_names or "NPU" not in activity_names:
            raise RuntimeError(
                "torch_npu.profiler activities missing CPU or NPU; got %r"
                % (activity_names,)
            )

        handler = torch_npu.profiler.tensorboard_trace_handler(
            profiler_config.torch_profiler_dir, worker_name=trace_name
        )
        profile_requested = {
            "activities": activity_names,
            "with_stack": False,
            "profile_memory": False,
            "with_modules": False,
            "record_shapes": True,
            "schedule": None,
            "max_iterations": 0,
            "delay_iterations": 0,
            "ignore_frontend": True,
            "on_trace_ready": "tensorboard_trace_handler",
        }
        experimental_config = torch_npu.profiler._ExperimentalConfig(**exp_kwargs)
        profile_kwargs = {
            "activities": activities,
            "with_stack": False,
            "profile_memory": False,
            "with_modules": False,
            "record_shapes": True,
            "experimental_config": experimental_config,
            "on_trace_ready": handler,
        }
        unsupported_profile = []
        filtered_profile = {}
        for key, value in profile_kwargs.items():
            if key in supported_profile or profile_var_kwargs:
                filtered_profile[key] = value
            else:
                unsupported_profile.append(key)
        for required in ("activities", "experimental_config"):
            if required not in filtered_profile:
                raise RuntimeError(
                    "torch_npu.profiler.profile does not accept required kwarg %r"
                    % required
                )
        profiler = torch_npu.profiler.profile(**filtered_profile)
        _PROFILER_EFFECTIVE = {
            "requested_experimental": requested,
            "effective_experimental": {key: str(value) for key, value in exp_kwargs.items()},
            "unsupported_experimental": unsupported_exp,
            "requested_profile": profile_requested,
            "effective_profile_keys": sorted(filtered_profile),
            "unsupported_profile": unsupported_profile,
            "activities": activity_names,
            "marker_keys": marker_keys,
            "schedule": None,
            "iteration_limit": None,
            "torch_npu_version": getattr(torch_npu, "__version__", "unknown"),
            "profiler_config": _profiler_config_summary(profiler_config),
        }
        _record_event(
            "profiler_effective_config", "vllm.profiler.effective_config", None,
            scope="rank",
            work=_PROFILER_EFFECTIVE,
            extra={"trace_name": trace_name,
                   "note": "recorded from actual constructor kwargs; hardware acceptance pending"},
            sanitize_payload=False,
        )
        return profiler

    _create_profiler.__e33_wrapped__ = True
    setattr(cls, "_create_profiler",
            staticmethod(_create_profiler) if original_is_static else _create_profiler)
    _PROFILER_PATCH_STATUS = "wrapped"
    return True, "wrapped"


class _SupportedParams(dict):
    pass


def _supported_params(func):
    """Return accepted parameter names; '__var_kwargs__' means **kwargs."""
    try:
        sig = inspect.signature(func)
    except Exception:
        return _SupportedParams({"__var_kwargs__": True})
    out = _SupportedParams()
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            out["__var_kwargs__"] = True
        else:
            out[name] = True
    return out


# ---------------------------------------------------------------------------
# Quiescent snapshots
# ---------------------------------------------------------------------------

def _read_proc_status() -> dict:
    out = {}
    try:
        with open("/proc/self/status", "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                if key in ("VmRSS", "VmHWM", "VmSize", "Threads",
                           "voluntary_ctxt_switches", "nonvoluntary_ctxt_switches",
                           "Cpus_allowed_list", "Mems_allowed_list", "CapEff"):
                    out[key] = value.strip()
    except Exception as exc:
        out["error"] = str(exc)
    return out


def _read_text(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read().strip()
    except Exception:
        return None


def _cpu_snapshot() -> dict:
    out = {"proc_status": _read_proc_status()}
    try:
        out["affinity"] = sorted(os.sched_getaffinity(0))
        out["affinity_count"] = len(out["affinity"])
        out["affinity"] = out["affinity"][:512]
    except Exception as exc:
        out["affinity_error"] = str(exc)
    try:
        out["nice"] = os.nice(0)
    except Exception:
        pass
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        out["rusage_maxrss_kb"] = usage.ru_maxrss
        out["rusage_user_s"] = usage.ru_utime
        out["rusage_sys_s"] = usage.ru_stime
    except Exception:
        pass
    out["cgroup_v2"] = {
        "path": _read_text("/proc/self/cgroup"),
        "memory_current": _read_text("/sys/fs/cgroup/memory.current"),
        "memory_max": _read_text("/sys/fs/cgroup/memory.max"),
        "cpu_max": _read_text("/sys/fs/cgroup/cpu.max"),
        "cpu_stat": _read_text("/sys/fs/cgroup/cpu.stat"),
    }
    return out


def _torch_snapshot(device) -> dict:
    torch = sys.modules.get("torch")
    out = {"torch_loaded": torch is not None, "api": None, "errors": []}
    if torch is None:
        out["unavailable"] = "torch not imported in this process"
        return out
    npu = getattr(torch, "npu", None)
    cuda = getattr(torch, "cuda", None)
    api = npu if npu is not None and hasattr(npu, "memory_allocated") else cuda
    if api is None:
        out["unavailable"] = "neither torch.npu nor torch.cuda exposes memory_allocated"
        return out
    out["api"] = "torch.npu" if api is npu else "torch.cuda"
    for name, fn_name in (
        ("allocated", "memory_allocated"),
        ("reserved", "memory_reserved"),
        ("max_allocated", "max_memory_allocated"),
        ("max_reserved", "max_memory_reserved"),
    ):
        fn = getattr(api, fn_name, None)
        if not callable(fn):
            out[name] = None
            continue
        try:
            out[name] = int(fn(device))
        except Exception:
            try:
                out[name] = int(fn())
            except Exception as exc:
                out[name] = None
                out["errors"].append("%s: %s" % (fn_name, exc))
    fn = getattr(api, "mem_get_info", None)
    if callable(fn):
        try:
            info = fn(device)
            out["mem_get_info"] = [_safe_int(v) for v in info] if isinstance(info, (tuple, list)) else _sanitize(info)
            out["mem_get_info_interpretation"] = "raw tuple order from torch API"
        except Exception:
            try:
                info = fn()
                out["mem_get_info"] = [_safe_int(v) for v in info] if isinstance(info, (tuple, list)) else _sanitize(info)
            except Exception as exc:
                out["mem_get_info"] = None
                out["errors"].append("mem_get_info: %s" % exc)
    stats_fn = getattr(api, "memory_stats", None)
    if callable(stats_fn):
        try:
            try:
                stats = stats_fn(device)
            except Exception:
                stats = stats_fn()
            keys = ("allocated_bytes.all.current", "allocated_bytes.all.peak",
                    "reserved_bytes.all.current", "reserved_bytes.all.peak")
            out["memory_stats_selected"] = {key: _safe_int(stats.get(key)) for key in keys}
            out["memory_stats_key_count"] = _safe_len(stats)
        except Exception as exc:
            out["memory_stats_selected"] = None
            out["errors"].append("memory_stats: %s" % exc)
    return out


def _tensor_meta(name, tensor, kind) -> dict:
    meta = {"name": str(name)[:500], "kind": kind}
    for attr in ("shape", "stride", "dtype", "device", "requires_grad"):
        try:
            value = getattr(tensor, attr)
            if callable(value):
                value = value()
            if attr in ("shape", "stride"):
                value = list(value)[:8]
            elif attr == "dtype":
                value = str(value)
            elif attr == "device":
                value = str(value)
            meta[attr] = _sanitize(value)
        except Exception:
            meta[attr] = None
    for attr in ("numel", "element_size", "data_ptr", "nbytes"):
        try:
            value = getattr(tensor, attr)
            meta[attr] = int(value()) if callable(value) else int(value)
        except Exception:
            meta[attr] = None
    if meta.get("numel") is not None and meta.get("element_size") is not None:
        meta["nbytes"] = meta["numel"] * meta["element_size"]
    try:
        storage = tensor.untyped_storage()
        meta["storage_data_ptr"] = int(storage.data_ptr())
        meta["storage_nbytes"] = int(storage.nbytes())
    except Exception:
        meta["storage_data_ptr"] = None
        meta["storage_nbytes"] = None
    if "scale" in meta["name"].lower():
        meta["role"] = "scale"
    return meta


def _looks_like_tensor(value) -> bool:
    return (value is not None and hasattr(value, "shape") and hasattr(value, "stride")
            and hasattr(value, "data_ptr") and not isinstance(value, type))


def _collect_tensors(root, prefix, kind, out, depth=0):
    if len(out) >= MAX_TENSOR_ENTRIES:
        return
    if _looks_like_tensor(root):
        out.append(_tensor_meta(prefix, root, kind))
        return
    if depth >= 2:
        return
    if isinstance(root, dict):
        items = list(root.items())[:MAX_SCAN_ITEMS]
        for key, value in items:
            _collect_tensors(value, "%s[%s]" % (prefix, key), kind, out, depth + 1)
    elif isinstance(root, (list, tuple)):
        for index, value in enumerate(root[:MAX_SCAN_ITEMS]):
            _collect_tensors(value, "%s[%d]" % (prefix, index), kind, out, depth + 1)


def _model_inventory(runner):
    out = {"model_class": None, "model_module": None, "entries": [],
           "errors": [], "truncated": False}
    model = None
    try:
        getter = getattr(runner, "get_model", None)
        model = getter() if callable(getter) else getattr(runner, "model", None)
    except Exception as exc:
        out["errors"].append("get_model: %s" % exc)
    if model is None:
        try:
            model = getattr(runner, "model", None)
        except Exception as exc:
            out["errors"].append("runner.model: %s" % exc)
    if model is None:
        return out
    out["model_class"] = type(model).__name__
    out["model_module"] = type(model).__module__
    entries = out["entries"]
    try:
        for name, tensor in model.named_parameters():
            if len(entries) >= MAX_TENSOR_ENTRIES:
                out["truncated"] = True
                break
            entries.append(_tensor_meta(name, tensor, "param"))
    except Exception as exc:
        out["errors"].append("named_parameters: %s" % exc)
    try:
        for name, tensor in model.named_buffers():
            if len(entries) >= MAX_TENSOR_ENTRIES:
                out["truncated"] = True
                break
            entries.append(_tensor_meta(name, tensor, "buffer"))
    except Exception as exc:
        out["errors"].append("named_buffers: %s" % exc)
    return out


def _draft_inventory(runner):
    out = {"present": False, "class": None, "entries": [], "errors": []}
    drafter = getattr(runner, "drafter", None)
    if drafter is None:
        return out
    out["present"] = True
    out["class"] = type(drafter).__name__
    candidates = []
    for attr in ("model", "draft_model"):
        value = getattr(drafter, attr, None)
        if value is not None:
            candidates.append(value)
    if not candidates and hasattr(drafter, "named_parameters"):
        candidates.append(drafter)
    for index, model in enumerate(candidates[:2]):
        try:
            for name, tensor in model.named_parameters():
                if len(out["entries"]) >= MAX_TENSOR_ENTRIES:
                    break
                out["entries"].append(_tensor_meta("drafter%d.%s" % (index, name), tensor, "draft_param"))
            for name, tensor in model.named_buffers():
                if len(out["entries"]) >= MAX_TENSOR_ENTRIES:
                    break
                out["entries"].append(_tensor_meta("drafter%d.%s" % (index, name), tensor, "draft_buffer"))
        except Exception as exc:
            out["errors"].append(str(exc))
    return out


def _visible_runner_tensors(runner):
    categories = {"kv": [], "graph_comm": [], "other": []}
    errors = []
    if runner is None:
        return categories, errors, []
    for attr in ("kv_caches", "_draft_plane_cache_tensors"):
        try:
            value = getattr(runner, attr, None)
            if value is not None:
                _collect_tensors(value, attr, "kv", categories["kv"])
        except Exception as exc:
            errors.append("%s: %s" % (attr, exc))
    pattern = re.compile(r"(graph|pool|workspace|comm|hccl|symm|buffer)", re.IGNORECASE)
    try:
        items = list(vars(runner).items())
    except Exception:
        items = []
    scanned = []
    for key, value in items:
        if key in ("kv_caches", "_draft_plane_cache_tensors", "model", "drafter",
                   "requests", "input_batch"):
            continue
        if not pattern.search(key):
            continue
        scanned.append(key)
        try:
            _collect_tensors(value, key, "graph_comm", categories["graph_comm"])
        except Exception as exc:
            errors.append("%s: %s" % (key, exc))
    return categories, errors, scanned


def _summarize_tensors(entries):
    unique = {}
    counts = {}
    entry_bytes = {}
    for entry in entries:
        kind = entry.get("kind") or "unknown"
        counts[kind] = counts.get(kind, 0) + 1
        nbytes = entry.get("nbytes")
        if nbytes:
            entry_bytes[kind] = entry_bytes.get(kind, 0) + nbytes
        storage = entry.get("storage_data_ptr")
        if storage is not None:
            unique[storage] = max(unique.get(storage, 0), entry.get("storage_nbytes") or 0)
    return {
        "entries": len(entries),
        "unique_storage_entries": len(unique),
        "unique_storage_nbytes": sum(unique.values()),
        "counts_by_kind": counts,
        "entry_nbytes_by_kind_overlapping": entry_bytes,
        "entry_nbytes_sum_overlapping": sum(entry_bytes.values()),
        "note": "entry sums can double-count shared/aliased storage; unique_storage_nbytes dedupes by storage pointer",
    }


def _cmdline() -> str | None:
    try:
        with open("/proc/self/cmdline", "rb") as fh:
            raw = fh.read()
        return raw.replace(b"\x00", b" ").decode("utf-8", "replace")[:MAX_STRING]
    except Exception:
        return None


def _snapshot(owner, tag, prefix=None, runner=None):
    """Never raise into the wrapped application; report a failed snapshot."""
    try:
        _snapshot_impl(owner, tag, prefix=prefix, runner=runner)
    except Exception as exc:
        try:
            started = _clock_triplet()
            _record_event("snapshot_error", "vllm.snapshot.error", owner, runner,
                          scope="rank", iteration_id=None,
                          iteration_id_source="not_applicable",
                          work={"tag": tag,
                                "error_type": type(exc).__name__,
                                "error": str(exc)[:MAX_STRING]},
                          start=started, end=_clock_triplet())
        except Exception:
            pass


def _snapshot_impl(owner, tag, prefix=None, runner=None):
    if not _diag_root():
        return
    started = _clock_triplet()
    errors = []
    snapshot = {
        "tag": tag,
        "profile_prefix": prefix,
        "process": {
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "cmdline": _cmdline(),
            "cwd": _safe_cwd(),
        },
        "cpu": _cpu_snapshot(),
        "torch": _torch_snapshot(getattr(owner, "device", None) if owner is not None else None),
        "runner": None,
        "model": None,
        "draft": None,
        "kv": [],
        "graph_comm": [],
        "other_tensors": [],
        "scanned_graph_comm_attrs": [],
        "unknown": [
            "ascend_native_pool_bytes",
            "graph_workspace_bytes",
            "hccl_symmetric_buffers",
            "allocator_internal_fragmentation",
            "engram_offload_native_buffers",
            "device_physical_bank_placement",
        ],
        "errors": errors,
    }
    if runner is None and owner is not None:
        runner = getattr(owner, "model_runner", None)
    if runner is not None:
        snapshot["runner"] = {
            "class": type(runner).__name__,
            "module": type(runner).__module__,
            "use_v2_model_runner": _sanitize(getattr(owner, "use_v2_model_runner", None) if owner is not None else None),
            "has_execute_model": callable(getattr(runner, "execute_model", None)),
            "has_prepare_inputs": callable(getattr(runner, "prepare_inputs", None)),
            "has_private_prepare_inputs": callable(getattr(runner, "_prepare_inputs", None)),
        }
        try:
            snapshot["model"] = _model_inventory(runner)
        except Exception as exc:
            errors.append("model_inventory: %s" % exc)
        try:
            snapshot["draft"] = _draft_inventory(runner)
        except Exception as exc:
            errors.append("draft_inventory: %s" % exc)
        try:
            kv_config = getattr(runner, "kv_cache_config", None)
            if kv_config is not None:
                snapshot["kv_cache_config"] = {
                    "class": type(kv_config).__name__,
                    "num_blocks": _safe_int(getattr(kv_config, "num_blocks", None)),
                    "block_size": _safe_int(getattr(kv_config, "block_size", None)),
                    "num_groups": _safe_len(getattr(kv_config, "kv_cache_groups", None)),
                    "note": "configured capacity, not live allocation; live tensor bytes are in kv[]",
                }
        except Exception as exc:
            errors.append("kv_cache_config: %s" % exc)
        try:
            cats, cat_errors, scanned = _visible_runner_tensors(runner)
            snapshot["kv"] = cats["kv"]
            snapshot["graph_comm"] = cats["graph_comm"]
            snapshot["other_tensors"] = cats["other"]
            snapshot["scanned_graph_comm_attrs"] = scanned
            errors.extend(cat_errors)
        except Exception as exc:
            errors.append("runner_tensors: %s" % exc)
    all_entries = []
    for key in ("model", "draft"):
        if snapshot.get(key):
            all_entries.extend(snapshot[key].get("entries") or [])
    all_entries.extend(snapshot["kv"])
    all_entries.extend(snapshot["graph_comm"])
    all_entries.extend(snapshot["other_tensors"])
    snapshot["summary"] = _summarize_tensors(all_entries)
    snapshot["summary"]["truncated"] = any(
        (snapshot.get(key) or {}).get("truncated") for key in ("model", "draft")
    )
    snapshot["limits"] = {"max_tensor_entries": MAX_TENSOR_ENTRIES,
                          "max_scan_items": MAX_SCAN_ITEMS}
    end = _clock_triplet()
    _record_event("snapshot", "vllm.snapshot.%s" % tag, owner, runner,
                  scope="rank", iteration_id=None,
                  iteration_id_source="not_applicable",
                  work=snapshot, start=started, end=end,
                  sanitize_payload=False)


def _safe_cwd():
    try:
        return os.getcwd()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Module installation
# ---------------------------------------------------------------------------

def _report(module_name, module, hooks, missing, notes):
    report = {
        "module": module_name,
        "file": getattr(module, "__file__", None),
        "hooks": hooks,
        "missing": missing,
        "notes": notes,
    }
    _PATCH_REPORTS[module_name] = report
    if module_name not in _REPORTED_MODULES:
        _REPORTED_MODULES.add(module_name)
        _record_event("instrumentation", "vllm.telemetry.instrumentation", None,
                      scope="process", work=report)
    return report


def _install_scheduler(module):
    cls = getattr(module, "Scheduler", None)
    hooks, missing = [], []
    if cls is None:
        return _report(module.__name__, module, hooks, ["class Scheduler"], [])
    for name, source, scope, work, iteration in (
        ("schedule", "vllm.scheduler.schedule", "shared_batch", _work_schedule, None),
        ("add_request", "vllm.scheduler.add_request", "request", _work_add_request, None),
        ("update_from_output", "vllm.scheduler.update_from_output", "shared_batch", _work_update_from_output, None),
        ("make_spec_decoding_stats", "vllm.scheduler.make_spec_decoding_stats", "request", _work_spec_stats, None),
    ):
        ok, why = _wrap_method(cls, name, source, scope, work_fn=work,
                               iteration_fn=iteration)
        (hooks if ok else missing).append(name if ok else "%s (%s)" % (name, why))
    return _report(module.__name__, module, hooks, missing,
                   ["SchedulerOutput is read only; batch work is recorded once"])


def _install_engine_core(module):
    cls = getattr(module, "EngineCore", None)
    hooks, missing = [], []
    if cls is None:
        return _report(module.__name__, module, hooks, ["class EngineCore"], [])

    def make_step_work(api_name):
        def step_work(self, args, kwargs, result):
            work = _work_engine_step(self, args, kwargs, result)
            work["engine_api"] = api_name
            return work
        return step_work

    for name, source in (
        ("step", "vllm.engine_core.step"),
        ("step_with_batch_queue", "vllm.engine_core.step_with_batch_queue"),
    ):
        ok, why = _wrap_method(cls, name, source, "engine_step",
                               work_fn=make_step_work(name), iteration_fn=None)
        (hooks if ok else missing).append(name if ok else "%s (%s)" % (name, why))
    return _report(module.__name__, module, hooks, missing,
                   ["step_with_batch_queue is async with respect to batch queue but synchronous in its API"])


def _install_worker(module):
    cls = getattr(module, "NPUWorker", None)
    hooks, missing = [], []
    if cls is None:
        return _report(module.__name__, module, hooks, ["class NPUWorker"], [])
    ok, why = _wrap_method(cls, "execute_model", "vllm.worker.NPUWorker.execute_model",
                           "rank", work_fn=_work_worker_execute)
    (hooks if ok else missing).append("execute_model" if ok else "execute_model (%s)" % why)
    ok, why = _wrap_snapshot_after(cls, "load_model", "vllm.worker.NPUWorker.load_model",
                                   "after_load_model")
    (hooks if ok else missing).append("load_model" if ok else "load_model (%s)" % why)
    ok, why = _wrap_snapshot_after(cls, "compile_or_warm_up_model",
                                   "vllm.worker.NPUWorker.compile_or_warm_up_model",
                                   "after_compile_or_warmup")
    (hooks if ok else missing).append("compile_or_warm_up_model" if ok else
                                      "compile_or_warm_up_model (%s)" % why)
    ok, why = _wrap_worker_profile(cls)
    (hooks if ok else missing).append("profile" if ok else "profile (%s)" % why)
    return _report(module.__name__, module, hooks, missing,
                   ["profile stop is caller-driven; no automatic step-count stop",
                    "snapshots run before profiler start and before profiler stop"])


def _install_runner_module(module, class_name="NPUModelRunner", source_prefix="vllm.model_runner"):
    cls = getattr(module, class_name, None)
    hooks, missing, notes = [], [], []
    if cls is None:
        return _report(module.__name__, module, hooks, ["class %s" % class_name], [])
    ok, why = _wrap_method(cls, "execute_model", "%s.%s.execute_model" % (source_prefix, class_name),
                           "rank", work_fn=_work_worker_execute)
    (hooks if ok else missing).append("execute_model" if ok else "execute_model (%s)" % why)
    for prepare_name in ("_prepare_inputs", "prepare_inputs"):
        if not hasattr(cls, prepare_name):
            notes.append("%s absent on this runner class" % prepare_name)
            continue
        ok, why = _wrap_method(cls, prepare_name,
                               "%s.%s.%s" % (source_prefix, class_name, prepare_name),
                               "rank", work_fn=_work_prepare_inputs)
        (hooks if ok else missing).append(prepare_name if ok else "%s (%s)" % (prepare_name, why))
    notes.append("actual runner class is recorded in each span/snapshot; MRV2 may prepare inputs inside execute_model")
    return _report(module.__name__, module, hooks, missing, notes)


def _install_serving(module):
    cls = getattr(module, "OpenAIServingCompletion", None)
    hooks, missing = [], []
    if cls is None:
        return _report(module.__name__, module, hooks, ["class OpenAIServingCompletion"], [])

    def create_work(self, args, kwargs, result):
        request = args[1] if len(args) > 1 else kwargs.get("request")
        return {
            "return_type": type(result).__name__,
            "request_fields": _http_request_fields(request),
            "stream_body_deferred": True,
            "generator_creation_only": type(result).__name__,
        }

    def create_request(self, name, args, kwargs):
        request = args[1] if len(args) > 1 else kwargs.get("request")
        rid = getattr(request, "request_id", None)
        return rid, ("request.request_id" if rid is not None else "unavailable")

    ok, why = _wrap_method(cls, "create_completion", "vllm.http.create_completion",
                           "http", work_fn=create_work, request_id_fn=create_request,
                           iteration_fn=lambda self, name: (None, "not_applicable"))
    (hooks if ok else missing).append("create_completion" if ok else "create_completion (%s)" % why)
    ok, why = _wrap_completion_stream(cls)
    (hooks if ok else missing).append("completion_stream_generator" if ok else
                                      "completion_stream_generator (%s)" % why)
    return _report(module.__name__, module, hooks, missing,
                   ["SSE per-chunk timing is the await __anext__ interval; it includes engine/upstream wait",
                    "client/consumer wait is excluded from chunk spans and reported only in stream_end wall"])


def _install_protocol(module):
    hooks, missing = [], []
    for class_name in ("CompletionStreamResponse", "CompletionResponse"):
        cls = getattr(module, class_name, None)
        if cls is None:
            missing.append("class %s" % class_name)
            continue
        ok, why = _wrap_serialization(cls)
        (hooks if ok else missing).append(class_name if ok else "%s (%s)" % (class_name, why))
    return _report(module.__name__, module, hooks, missing,
                   ["serialization span records JSON chars/bytes and field lengths, not token contents"])


def install_for_module(module) -> dict:
    """Install the fixed hook set for one already-loaded target module."""
    name = getattr(module, "__name__", "")
    if name not in TARGET_MODULES:
        return {"module": name, "hooks": [], "missing": [],
                "notes": ["not a target module; untouched"]}
    if name in _PATCH_REPORTS:
        return _PATCH_REPORTS[name]
    if name == "vllm.v1.core.sched.scheduler":
        return _install_scheduler(module)
    if name == "vllm.v1.engine.core":
        return _install_engine_core(module)
    if name == "vllm_ascend.worker.worker":
        return _install_worker(module)
    if name == "vllm_ascend.worker.model_runner_v1":
        return _install_runner_module(module, "NPUModelRunner", "vllm.model_runner.mrv1")
    if name == "vllm_ascend.worker.v2.model_runner":
        return _install_runner_module(module, "NPUModelRunner", "vllm.model_runner.mrv2")
    if name == "vllm_ascend.profiler.torch_npu_profiler":
        hooks, missing = [], []
        ok, why = _install_profiler_override(module)
        (hooks if ok else missing).append("_create_profiler" if ok else "_create_profiler (%s)" % why)
        return _report(module.__name__, module, hooks, missing,
                       ["Level0, CPU+NPU, mstx/msprof_tx, op_attr requested; unsupported kwargs are filtered and reported",
                        "no schedule and no iteration truncation"])
    if name == "vllm.entrypoints.openai.completion.serving":
        return _install_serving(module)
    if name == "vllm.entrypoints.openai.completion.protocol":
        return _install_protocol(module)
    if name in ("vllm.entrypoints.openai.api_server", "vllm.v1.core.sched.output",
                "vllm.config.profiler"):
        return _report(module.__name__, module, [], [],
                       ["loaded for coverage accounting only; no hook is installed in this module"])
    return _report(module.__name__, module, [], [], ["no installer registered"])


def coverage_report() -> dict:
    return {
        "targets": list(TARGET_MODULES),
        "reports": dict(_PATCH_REPORTS),
        "source_counts": dict(sorted(_SOURCE_COUNTS.items())),
        "profiler_patch_status": _PROFILER_PATCH_STATUS,
        "profiler_effective": _PROFILER_EFFECTIVE,
        "written_pid": _WRITER_PID,
        "write_errors": _WRITE_ERRORS,
    }


def note(source: str, message: str, **extra) -> None:
    """Record a bootstrap/coverage note without touching target objects."""
    _record_event("instrumentation_note", source, None, scope="process",
                  work={"message": message}, extra=extra)

