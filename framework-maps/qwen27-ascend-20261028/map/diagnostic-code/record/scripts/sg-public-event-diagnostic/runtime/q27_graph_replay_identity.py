"""Opt-in, metadata-only NPUGraph.replay observer for the Q27 lineage trace."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from functools import wraps
import itertools
import os
import threading
import time

_INSTALL_MARKER = "_q27_replay_identity_observer_installed"
_COUNTER = itertools.count(1)
_COUNTER_LOCK = threading.Lock()
_CONTEXT_FIELDS = ("rank_id", "capture_id", "generation", "raw_path", "graph_model_ids")


def _next_call_id(pid):
    with _COUNTER_LOCK:
        number = next(_COUNTER)
    return f"{pid}:{number}"


def _read_context(self, context_lookup):
    try:
        raw = context_lookup(self)
    except BaseException as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(raw, dict):
        return None, None
    return raw, None


def _span_context(span_factory, name):
    """Return a context manager whose failures cannot interfere with replay."""
    try:
        if span_factory is not None:
            cm = span_factory(name)
        else:
            try:
                from torch.profiler import record_function
                cm = record_function(name)
            except BaseException:
                cm = nullcontext()
    except BaseException:
        cm = nullcontext()

    @contextmanager
    def safe_span():
        try:
            cm.__enter__()
        except BaseException:
            yield
            return
        try:
            yield
        except BaseException as exc:
            try:
                cm.__exit__(type(exc), exc, exc.__traceback__)
            except BaseException:
                pass
            raise
        else:
            try:
                cm.__exit__(None, None, None)
            except BaseException:
                pass
    return safe_span()


def _safe_emit(emit, payload):
    try:
        emit(payload)
    except BaseException:
        # Observability must not change graph replay behavior.
        pass


def install(npugraph_cls, context_lookup, emit, span_factory=None):
    """Wrap `npugraph_cls.replay` once and return installation metadata.

    `context_lookup(graph)` must return the root-owned CPU capture receipt for
    this exact Python graph object, or None. It must not inspect graph contents
    or tensors. `emit(event_dict)` is called synchronously at begin and end;
    failures are isolated. No capture/replay/synchronization call is added.
    """
    if not callable(context_lookup):
        raise TypeError("context_lookup must be callable")
    if not callable(emit):
        raise TypeError("emit must be callable")
    current = getattr(npugraph_cls, "replay", None)
    if not callable(current):
        raise TypeError("npugraph_cls.replay must be callable")
    if getattr(npugraph_cls, _INSTALL_MARKER, False):
        return {"status": "already_installed", "class": npugraph_cls.__qualname__}

    original = current

    @wraps(original)
    def observed_replay(self, *args, **kwargs):
        pid = os.getpid()
        tid = threading.get_ident()
        call_id = _next_call_id(pid)
        graph_id = id(self)
        context, context_error = _read_context(self, context_lookup)
        common = {"pid": pid, "tid": tid, "graph_python_id": graph_id,
                  "call_id": call_id,
                  "capture_context_found": context is not None}
        for key in _CONTEXT_FIELDS:
            common[key] = context.get(key) if context is not None else None
        if context_error is not None:
            common["capture_context_error"] = context_error
        generation = common.get("generation")
        span_name = (f"Q27 NPUGraph replay pid={pid} graph_python_id={graph_id} "
                     f"call_id={call_id} generation={generation if generation is not None else 'unknown'}")
        begin_mono = time.monotonic_ns()
        begin_wall = time.time_ns()
        _safe_emit(emit, {"event": "graph_replay_begin", **common,
                          "monotonic_ns": begin_mono, "wall_time_ns": begin_wall,
                          "span_name": span_name})
        error = None
        try:
            with _span_context(span_factory, span_name):
                return original(self, *args, **kwargs)
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            end_mono = time.monotonic_ns()
            end_wall = time.time_ns()
            _safe_emit(emit, {"event": "graph_replay_end", **common,
                              "monotonic_ns": end_mono, "wall_time_ns": end_wall,
                              "elapsed_ns": max(0, end_mono - begin_mono),
                              "error": error})

    setattr(observed_replay, "_q27_wrapped_original", original)
    setattr(npugraph_cls, "replay", observed_replay)
    setattr(npugraph_cls, _INSTALL_MARKER, True)
    return {"status": "installed", "class": npugraph_cls.__qualname__}
