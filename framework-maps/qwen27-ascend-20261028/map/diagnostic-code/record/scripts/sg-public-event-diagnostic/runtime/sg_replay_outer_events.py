"""Opt-in outer replay boundary anchors for the SGLang CUDA-graph backend.

Preparation candidate only. It does not change graph contents or read callback
results until the caller proves the profiler window is closed and drains them.
"""
from __future__ import annotations

import functools
import itertools
import os
import threading

_SERIALS = itertools.count(1)
_LOCK = threading.Lock()
_TLS = threading.local()
_PENDING = []
_DRAINED = []
_FAILED = []


def _serial():
    with _LOCK:
        return next(_SERIALS)


def _scalar(value):
    return value if value is None or isinstance(value, (bool, int, float, str)) else None


def install_replay_outer_anchors(backend_cls, *, event_factory, stream_provider,
                                 extension, emit, graph_key_provider=None, work_key_provider=None,
                                 record_scope_factory=None, graph_allowed=None):
    """Wrap both public backend replay methods; nested method calls are once-only.

    Providers must use already-available CPU metadata/current stream. This
    function performs no import, synchronization or file I/O.
    """
    if getattr(backend_cls, "_q27_outer_replay_events_installed", False):
        return []
    methods = ("replay", "replay_with_input_update")
    installed = []
    scope_factory = record_scope_factory
    for method_name in methods:
        original = getattr(backend_cls, method_name, None)
        if not callable(original):
            continue

        @functools.wraps(original)
        def wrapped(self, *args, __original=original, __method=method_name, **kwargs):
            depth = getattr(_TLS, "depth", 0)
            if depth:
                _TLS.depth = depth + 1
                try:
                    return __original(self, *args, **kwargs)
                finally:
                    _TLS.depth = depth

            if callable(graph_key_provider):
                try:
                    graph_key = graph_key_provider(self, __method, args, kwargs)
                except BaseException:
                    graph_key = None
            else:
                graph_key = _scalar(kwargs.get("shape_key", kwargs.get("graph_key")))
            graphs = getattr(self, "_graphs", None)
            graph = graphs.get(graph_key) if isinstance(graphs, dict) and graph_key in graphs else None
            graph_id = id(graph) if graph is not None else None
            if graph is None or (graph_allowed is not None and not graph_allowed(self, graph)):
                return __original(self, *args, **kwargs)
            serial = _serial()
            work_key = None
            if callable(work_key_provider):
                try:
                    work_key = _scalar(work_key_provider(self, __method, args, kwargs))
                except BaseException:
                    work_key = None
            stream = stream_provider()
            if stream is None:
                with _LOCK:
                    _FAILED.append({"status": "unknown", "reason": "current_stream_unavailable",
                                    "pid": os.getpid(), "serial": serial, "graph_python_id": graph_id})
                _TLS.depth = 1
                try:
                    return __original(self, *args, **kwargs)
                finally:
                    _TLS.depth = 0

            invocation = f"pid{os.getpid()}-replay{serial}-graph{graph_id}-shape{graph_key}"
            hold = {"pid": os.getpid(), "replay_serial": serial, "method": __method,
                    "graph_key": str(graph_key), "graph_python_id": graph_id, "graph_ref": graph,
                    "work_key": work_key, "request_id": None, "stream": stream,
                    "stream_id": None, "rows": [], "completed": False,
                    "invocation": invocation}
            local_scope_factory = scope_factory
            if local_scope_factory is None:
                import torch
                local_scope_factory = torch.profiler.record_function

            def boundary(side):
                event = event_factory()
                row = {"side": side, "event": event, "callback": None,
                       "stream": stream, "invocation": invocation}
                hold["rows"].append(row)
                event.record(stream)
                row["callback"] = extension.anchor("after_add" if side=="start" else "after_mul")
                return row

            _TLS.depth = 1
            try:
                name = invocation + ".start"
                with local_scope_factory(name):
                    boundary("start")
                try:
                    result = __original(self, *args, **kwargs)
                except BaseException:
                    hold["error"] = "backend_replay_raised"
                    raise
                name = invocation + ".end"
                with local_scope_factory(name):
                    boundary("end")
                hold["completed"] = True
                with _LOCK:
                    _PENDING.append(hold)
                return result
            except BaseException:
                if hold not in _FAILED:
                    hold["error"] = hold.get("error", "outer_replay_anchor_or_call_raised")
                    with _LOCK:
                        _FAILED.append(hold)
                raise
            finally:
                _TLS.depth = 0

        setattr(backend_cls, method_name, wrapped)
        installed.append(method_name)
    backend_cls._q27_outer_replay_events_installed = True
    return installed


def drain_outer_replay_anchors(*, stream_synchronizer=None, emit):
    """Drain and read only after profiler EOF and backend queue quiescence.

    On any drain failure references remain held in `_FAILED`; no events are
    released here because the graph may still be live.
    """
    with _LOCK:
        pending = list(_PENDING)
    if stream_synchronizer is None:
        streams = {id(row["stream"]): row["stream"] for item in pending for row in item["rows"]}
        if not streams or any(not callable(getattr(s, "synchronize", None)) for s in streams.values()):
            raise RuntimeError("saved replay streams unavailable; references retained")
        sync = lambda stream: stream.synchronize()
    else:
        streams = {id(row["stream"]): row["stream"] for item in pending for row in item["rows"]}
        sync = stream_synchronizer
    try:
        for stream in streams.values():
            sync(stream)
    except BaseException as exc:
        with _LOCK:
            _FAILED.extend(pending)
        emit("sg_replay_outer_unresolved", reason="post-EOF stream drain failed",
             error_type=type(exc).__name__, retained=len(pending))
        return {"status": "drain_failed_references_retained", "items": len(pending)}

    emitted = 0
    for item in pending:
        for row in item["rows"]:
            result = row["callback"]
            def field(name):
                return _scalar(getattr(result, name, None))
            values = {name: field(name) for name in
                      ("acl_stream_id", "task_id", "stream_rc", "task_rc", "capture_rc",
                       "capture_status", "callback_rc", "model_ri_nonnull")}
            identity_ok = (item["graph_python_id"] is not None and
                           values["model_ri_nonnull"] is False and
                           values["capture_status"] == 0 and
                           all(values[k] == 0 for k in ("stream_rc", "task_rc", "capture_rc", "callback_rc")))
            emit("sg_replay_outer_event", pid=item["pid"], replay_serial=item["replay_serial"],
                 method=item["method"], graph_key=item["graph_key"],
                 graph_python_id=item["graph_python_id"], work_key=item["work_key"],
                 request_id=item["request_id"], invocation=item["invocation"],
                 side=row["side"], **values,
                 classification="raw_outer_replay_endpoint" if identity_ok else "unknown_outer_endpoint",
                 meaning="outer replay boundary only; no per-layer or exclusive duration inferred")
            emitted += 1
    with _LOCK:
        _DRAINED.extend(pending)
        for item in pending:
            if item in _PENDING:
                _PENDING.remove(item)
    return {"status": "drained", "items": len(pending), "rows": emitted,
            "events_retained_with_graph": True}


def pending_counts():
    with _LOCK:
        return {"pending": len(_PENDING), "failed_or_unresolved": len(_FAILED)}


def release_drained_outer_events(graph_obj, *, reset_receipt):
    if reset_receipt.get("graph_python_id")!=id(graph_obj) or reset_receipt.get("reset_or_discarded")is not True:
        raise RuntimeError("exact graph reset receipt required")
    with _LOCK:
        selected=[x for x in _DRAINED if x.get("graph_ref")is graph_obj]
        if any(x.get("graph_ref")is graph_obj for x in _PENDING):
            raise RuntimeError("undrained replay events remain")
        _DRAINED[:]=[x for x in _DRAINED if x.get("graph_ref")is not graph_obj]
    return {"replay_items_released":len(selected),"outer_event_count_released":sum(len(x["rows"])for x in selected)}
