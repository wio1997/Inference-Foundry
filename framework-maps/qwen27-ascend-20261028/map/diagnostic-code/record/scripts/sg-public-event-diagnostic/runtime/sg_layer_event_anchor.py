"""Opt-in, endpoint-only SGLang graph layer task anchors.

No I/O or synchronization occurs while capture is active. The C++ extension
queues public ACL task-ID callbacks; result objects stay live until capture has
closed and the same public NPU stream has synchronized.
"""
from __future__ import annotations

import functools
import itertools
import os
import sys
import threading

_TLS = threading.local()
_FAILED_SCOPES = []
_FAILED_LOCK = threading.Lock()
_INVOCATIONS = itertools.count(1)
_INVOCATION_LOCK = threading.Lock()


def _next_capture_invocation_id():
    with _INVOCATION_LOCK:
        serial = next(_INVOCATIONS)
    return f"pid{os.getpid()}-capture{serial}"


def _default_record_scope_factory():
    try:
        import torch
        factory = getattr(getattr(torch, "profiler", None), "record_function", None)
    except BaseException as exc:
        raise RuntimeError("torch.profiler.record_function is required for explicit capture scopes") from exc
    if not callable(factory):
        raise RuntimeError("torch.profiler.record_function is unavailable; refusing unmarked capture")
    return factory


def _call_in_record_scope(factory, name, forward_fn, *args, **kwargs):
    """Run one forward in a profiler scope while preserving model exceptions."""
    manager = factory(name)
    if not callable(getattr(manager, "__enter__", None)) or not callable(getattr(manager, "__exit__", None)):
        raise TypeError("record_scope_factory must return a context manager")
    manager.__enter__()
    try:
        result = forward_fn(*args, **kwargs)
    except BaseException as forward_error:
        exc_info = sys.exc_info()
        try:
            manager.__exit__(*exc_info)
        except BaseException as scope_exit_error:
            add_note = getattr(forward_error, "add_note", None)
            if callable(add_note):
                add_note(f"Q27 profiler scope exit also failed: {scope_exit_error!r}")
        raise
    manager.__exit__(None, None, None)
    return result


def _drain_and_emit(captures, graph_key, emit, *, graph_python_id=None, graph_identity_status="unknown"):
    pending = [item for scope in captures for item in scope["pending"]]
    if not pending:
        emit("backend_capture_anchor_summary", graph_key=graph_key,
             graph_python_id=graph_python_id, graph_identity_status=graph_identity_status,
             capture_invocation_id=(captures[0].get("capture_invocation_id") if captures else None),
             forward_scopes=[{"forward_call_index": s.get("forward_call_index"),
                              "forward_scope_name": s.get("forward_scope_name"),
                              "forward_scope_kind": s.get("forward_scope_kind", "legacy_capture_scope")} for s in captures],
             forward_invocations=len(captures), raw_callback_rows=0, status1_success_rows=0,
             exact_64_layer_capture_endpoints=False,
             meaning="no callback rows; capture scope metadata only, no graph-layer identity inferred")
        return
    streams, seen = [], set()
    for item in pending:
        stream = item["stream"]
        if stream is not None and id(stream) not in seen:
            seen.add(id(stream)); streams.append(stream)
    try:
        if not streams or any(not callable(getattr(s, "synchronize", None)) for s in streams):
            raise RuntimeError("saved in-capture public stream object unavailable")
        for stream in streams: stream.synchronize()
    except BaseException as exc:
        with _FAILED_LOCK: _FAILED_SCOPES.append(captures)
        emit("layer_task_anchor_unresolved", reason="post-capture drain failed; callback objects retained",
             error=repr(exc), pending=len(pending), graph_key=graph_key)
        return
    accepted = []
    for scope in captures:
        layer_ids = []
        for seq, item in enumerate(scope["pending"]):
            result = item["callback"]
            row = {k: v for k, v in item.items() if k not in ("callback", "stream")}
            def exact_int(name):
                value = getattr(result, name, None)
                return value if isinstance(value, int) else None
            row.update({"anchor_seq": seq, "graph_key": graph_key,
                        "graph_python_id": graph_python_id,
                        "graph_identity_status": graph_identity_status,
                        "capture_invocation_id": scope.get("capture_invocation_id"),
                        "shape_key": scope.get("shape_key", graph_key),
                        "forward_scope_name": scope.get("forward_scope_name"),
                        "forward_scope_kind": scope.get("forward_scope_kind", "legacy_capture_scope"),
                        "forward_call_index": scope["forward_call_index"],
                        "producer_pid": __import__("os").getpid(),
                        "acl_stream_id": exact_int("acl_stream_id"), "task_id": exact_int("task_id"),
                        "stream_rc": exact_int("stream_rc"), "task_rc": exact_int("task_rc"),
                        "capture_rc": exact_int("capture_rc"), "capture_status": exact_int("capture_status"),
                        "model_ri_nonnull": getattr(result, "model_ri_nonnull", None) is True,
                        "callback_rc": exact_int("callback_rc")})
            ok = (graph_identity_status == "unique_backend_saved_graph_object" and
                  isinstance(graph_python_id, int) and scope["forward_call_index"] == 3 and
                  row["capture_status"] == 1 and row["model_ri_nonnull"] and
                  all(row[k] == 0 for k in ("stream_rc", "task_rc", "capture_rc", "callback_rc")))
            row["classification"] = "actual_capture_endpoint_candidate" if ok else "noncapture_or_unknown_raw_anchor"
            row["scope"] = "one exact forward_fn invocation in capture_one; endpoint only"
            if ok: accepted.append(row); layer_ids.append(item["layer_idx"])
            emit("layer_task_anchor", **row)
        scope["successful_capture_layer_count"] = len(layer_ids)
        scope["successful_capture_layer_coverage_complete"] = (scope["forward_call_index"] == 3 and
            len(layer_ids) == 64 and sorted(layer_ids) == list(range(64)))
    emit("backend_capture_anchor_summary", graph_key=graph_key,
         graph_python_id=graph_python_id, graph_identity_status=graph_identity_status,
         capture_invocation_id=(captures[0].get("capture_invocation_id") if captures else None),
         forward_scopes=[{"forward_call_index": s.get("forward_call_index"),
                          "forward_scope_name": s.get("forward_scope_name"),
                          "forward_scope_kind": s.get("forward_scope_kind", "legacy_capture_scope")} for s in captures],
         forward_invocations=len(captures), raw_callback_rows=len(pending),
         status1_success_rows=len(accepted),
         exact_64_layer_capture_endpoints=(len(accepted) == 64 and
             sorted(x["layer_idx"] for x in accepted) == list(range(64))),
         meaning="only status-1, nonnull-model-RI, all-ACL-rc-zero callbacks count; other rows remain unknown/raw")


def install_layer_anchor_set(layers, layer_types, extension, emit, *, stream_provider):
    if len(layers) != 64 or len(layer_types) != 64:
        raise ValueError("layer anchors require exactly the source-verified 64-layer model")
    for idx, layer in enumerate(layers):
        if getattr(layer, "_q27_task_anchor_wrapped", False):
            continue
        original = layer.forward

        @functools.wraps(original)
        def anchored_forward(*args, _original=original, _idx=idx, _typ=str(layer_types[idx]), **kwargs):
            result = _original(*args, **kwargs)
            active = getattr(_TLS, "capture", None)
            if active is not None:
                # The extension supports two fixed labels. They are labels only;
                # each invocation independently captures this layer endpoint.
                label = "after_add" if (_idx % 2 == 0) else "after_mul"
                try:
                    # NPUGraphRunner uses an explicit capture stream. Query the
                    # public current stream here, while each layer is executing
                    # inside that capture context, and retain that exact object.
                    stream = stream_provider()
                    callback = extension.anchor(label)
                    active["pending"].append({"layer_idx": _idx, "layer_type": _typ,
                                              "label": label, "callback": callback,
                                              "stream": stream})
                except BaseException as exc:
                    active["errors"].append({"layer_idx": _idx, "error": repr(exc)})
            return result

        layer.forward = anchored_forward
        layer._q27_task_anchor_wrapped = True
    emit("layer_task_anchor_installed", layer_count=64,
         scope="one public queued-task endpoint per Python layer invocation during graph capture")


def capture_graph_scope(original, runner, args, kwargs, *, enabled, emit):
    """Run capture; only drain/read successful, closed capture scopes."""
    if not enabled:
        out = original(runner, *args, **kwargs)
        emit("layer_task_anchor_unresolved", reason="capture wrapper ran before exact 64-layer hooks were ready",
             graph_key=str(getattr(runner, "_q27_last_graph_key", None)))
        return out
    if getattr(_TLS, "capture", None) is not None:
        raise RuntimeError("nested graph capture anchor scope is unsupported")
    graph_key = getattr(runner, "_q27_last_graph_key", None)
    capture = {"pending": [], "errors": [], "graph_key": str(graph_key) if graph_key is not None else None,
               "forward_call_index": 3}
    _TLS.capture = capture
    try:
        out = original(runner, *args, **kwargs)
    except BaseException as exc:
        _TLS.capture = None
        if capture["pending"]:
            with _FAILED_LOCK:
                _FAILED_SCOPES.append(capture)
        try:
            emit("layer_task_anchor_unresolved", reason="graph capture raised; closure unproven, callbacks retained without drain/read",
                 error=repr(exc), pending=len(capture["pending"]), graph_key=capture["graph_key"])
        except BaseException:
            pass
        raise
    _TLS.capture = None
    for error in capture["errors"]:
        emit("layer_task_anchor_unresolved", reason="callback queueing failed within layer capture",
             graph_key=capture["graph_key"], **error)
    if capture["pending"]: _drain_and_emit([capture], capture["graph_key"], emit)
    return out


def capture_backend_one_scope(original, backend, args, kwargs, *, enabled, emit,
                              record_scope_factory=None):
    """Instrument actual backend method: two warmup calls then graph forward."""
    if not enabled:
        out = original(backend, *args, **kwargs)
        emit("layer_task_anchor_unresolved", reason="backend hook ran before exact layer hooks were ready")
        return out
    arglist, kw = list(args), dict(kwargs)
    shape_key = arglist[0] if len(arglist) > 0 else kw.get("shape_key")
    forward_fn = arglist[1] if len(arglist) > 1 else kw.get("forward_fn")
    if not callable(forward_fn) or shape_key is None:
        return original(backend, *args, **kwargs)
    graph_key = str(shape_key)
    capture_invocation_id = _next_capture_invocation_id()
    if record_scope_factory is None:
        try:
            record_scope_factory = _default_record_scope_factory()
        except BaseException as exc:
            try:
                emit("layer_task_anchor_unresolved", reason="record_function unavailable; refusing unmarked capture",
                     graph_key=graph_key, capture_invocation_id=capture_invocation_id, error=repr(exc))
            except BaseException:
                pass
            raise
    if not callable(record_scope_factory):
        raise TypeError("record_scope_factory must be callable")
    captures, calls = [], {"count": 0}

    @functools.wraps(forward_fn)
    def tracked_forward(*fargs, **fkwargs):
        calls["count"] += 1
        call_index = calls["count"]
        scope_kind = "warmup" if call_index in (1, 2) else ("graph_capture_forward" if call_index == 3 else "unknown_forward")
        scope_name = (f"q27.sglang.capture_invocation={capture_invocation_id}"
                      f".shape_key={graph_key}.forward_call_index={call_index}.kind={scope_kind}")
        scope = {"pending": [], "errors": [], "graph_key": graph_key,
                 "shape_key": graph_key, "capture_invocation_id": capture_invocation_id,
                 "forward_scope_name": scope_name, "forward_scope_kind": scope_kind,
                 "forward_call_index": call_index}
        captures.append(scope)
        _TLS.capture = scope
        try: return _call_in_record_scope(record_scope_factory, scope_name, forward_fn, *fargs, **fkwargs)
        finally: _TLS.capture = None

    if len(arglist) > 1: arglist[1] = tracked_forward
    else: kw["forward_fn"] = tracked_forward
    try:
        out = original(backend, *arglist, **kw)
    except BaseException as exc:
        _TLS.capture = None
        if any(x["pending"] for x in captures):
            with _FAILED_LOCK: _FAILED_SCOPES.append(captures)
        try:
            emit("layer_task_anchor_unresolved", reason="backend capture_one raised; closure unproven, callbacks retained without drain/read",
                 error=repr(exc), forward_calls=calls["count"],
                 pending=sum(len(x["pending"]) for x in captures), graph_key=graph_key,
                 capture_invocation_id=capture_invocation_id,
                 forward_scopes=[{"forward_call_index": s["forward_call_index"],
                                  "forward_scope_name": s["forward_scope_name"],
                                  "forward_scope_kind": s["forward_scope_kind"]} for s in captures])
        except BaseException: pass
        raise
    _TLS.capture = None
    if calls["count"] != 3:
        emit("layer_task_anchor_unresolved", reason="frozen backend call count changed; preserve rows without success assumption",
             forward_calls=calls["count"], expected_source_calls=3, graph_key=graph_key,
             capture_invocation_id=capture_invocation_id,
             forward_scopes=[{"forward_call_index": s["forward_call_index"],
                              "forward_scope_name": s["forward_scope_name"],
                              "forward_scope_kind": s["forward_scope_kind"]} for s in captures])
    graph_python_id = None
    identity_status = "unknown_backend_graph_dict_or_shape_key_missing"
    graph_dict = getattr(backend, "_graphs", None)
    try:
        if isinstance(graph_dict, dict) and shape_key in graph_dict:
            graph_obj = graph_dict[shape_key]
            if graph_obj is not None:
                object_keys = [k for k, value in graph_dict.items() if value is graph_obj]
                graph_python_id = id(graph_obj)
                identity_status = ("unique_backend_saved_graph_object" if len(object_keys) == 1
                                   else "ambiguous_backend_graph_object_reused_by_multiple_shape_keys")
    except BaseException:
        graph_python_id = None
        identity_status = "unknown_backend_graph_lookup_failed"
    _drain_and_emit(captures, graph_key, emit, graph_python_id=graph_python_id,
                    graph_identity_status=identity_status)
    return out


def _normal_key(row, aliases):
    normalized = {"".join(ch.lower() for ch in str(k) if ch.isalnum()): v
                  for k, v in row.items()}
    result = {}
    for out_key, candidates in aliases.items():
        matches = [normalized[k] for k in candidates if k in normalized]
        if len(matches) != 1 or not isinstance(matches[0], int):
            return None
        result[out_key] = matches[0]
    return result


def bridge_anchor_to_graph(anchor, capture_rows, graph_rows, capture_receipts):
    """Join one endpoint only through unique CANN physical→model stream bridge.

    capture_rows must be decoded 64-byte capture_stream_info records from the
    exact same rank/profile capture. A duplicate bridge or graph key is unknown.
    """
    required_rc = ("stream_rc", "task_rc", "capture_rc", "callback_rc")
    if any(anchor.get(k) != 0 for k in required_rc) or anchor.get("capture_status") != 1 or not anchor.get("model_ri_nonnull"):
        return {"status": "unknown", "reason": "anchor callback did not satisfy strict successful-capture gate"}
    if not isinstance(anchor.get("graph_key"), str) or not anchor["graph_key"]:
        return {"status": "unknown", "reason": "anchor lacks exact captured graph_key"}
    graph_id = anchor.get("graph_python_id")
    if not isinstance(graph_id, int) or anchor.get("graph_identity_status") != "unique_backend_saved_graph_object":
        return {"status": "unknown", "reason": "backend graph object ID is missing or ambiguous"}
    if not isinstance(anchor.get("pid"), int) or not isinstance(anchor.get("rank_id"), int):
        return {"status": "unknown", "reason": "anchor lacks explicit PID/rank for capture receipt binding"}
    receipt_matches = [r for r in capture_receipts
                       if r.get("graph_python_id") == graph_id and r.get("pid") == anchor["pid"]
                       and r.get("rank_id") == anchor["rank_id"]
                       and r.get("capture_end_status") == "success"
                       and r.get("dump_status") == "success" and r.get("raw_exists") is True]
    if len(receipt_matches) != 1:
        return {"status": "unknown", "reason": "graph Python ID has no unique successful capture receipt",
                "receipt_candidates": len(receipt_matches)}
    receipt = receipt_matches[0]
    allowed_models = {str(x) for x in receipt.get("graph_model_ids", [])}
    if not allowed_models:
        return {"status": "unknown", "reason": "capture receipt has no graph_model_ids"}
    if not isinstance(anchor.get("acl_stream_id"), int) or not isinstance(anchor.get("task_id"), int):
        return {"status": "unknown", "reason": "anchor lacks exact integer ACL stream/task id"}
    bridge = [r for r in capture_rows if r.get("original_stream_id") == anchor["acl_stream_id"]
              and str(r.get("model_id")) in allowed_models
              and isinstance(r.get("model_stream_id"), int)
              and isinstance(r.get("model_id"), int)
              and isinstance(r.get("device_id"), int)]
    if len(bridge) != 1:
        return {"status": "unknown", "reason": "physical stream has no unique capture_stream_info bridge",
                "bridge_candidates": len(bridge)}
    b = bridge[0]
    nodes = []
    for row in graph_rows:
        k = _normal_key(row, {"model_id": ("modelid",), "stream_id": ("streamid",), "task_id": ("taskid",)})
        if k and (k["model_id"], k["stream_id"], k["task_id"]) == (b["model_id"], b["model_stream_id"], anchor["task_id"]):
            nodes.append(row)
    if len(nodes) != 1:
        return {"status": "unknown", "reason": "exact bridged graph task key is absent or duplicated",
                "graph_key_candidates": len(nodes), "model_id": b["model_id"],
                "model_stream_id": b["model_stream_id"], "task_id": anchor["task_id"]}
    return {"status": "exact_endpoint", "physical_acl_stream_id": anchor["acl_stream_id"],
            "graph_python_id": graph_id, "capture_receipt_model_ids": sorted(allowed_models),
            "capture_receipt_path": receipt.get("receipt_path"),
            "profiler_device_id": b["device_id"], "model_id": b["model_id"],
            "model_stream_id": b["model_stream_id"], "capture_stream_info_status": b.get("capture_status"),
            "task_id": anchor["task_id"],
            "graph_node": nodes[0], "scope": "single endpoint only; no inference for intervening tasks or operator timing"}


# Public-event layer-marker candidate (gen44). Event wrappers are allocated at
# capture_one entry, before graph capture. Their ACL handles remain lazy until
# the third forward's first record(stream), as established by gen43 source.
_GRAPH_EVENT_HOLDS = {}
_GRAPH_EVENT_HOLDS_LOCK = threading.Lock()
_FAILED_EVENT_SCOPES = []


def _default_public_event_factory():
    import torch
    return torch.npu.Event(enable_timing=False)


def install_layer_event_anchor_set(layers, layer_types, extension, emit, *, stream_provider):
    if len(layers) != 64 or len(layer_types) != 64:
        raise ValueError("layer event markers require exactly 64 source-verified layers")
    for idx, layer in enumerate(layers):
        if getattr(layer, "_q27_event_anchor_wrapped", False):
            continue
        original = layer.forward

        @functools.wraps(original)
        def wrapped(*args, _original=original, _idx=idx, _typ=str(layer_types[idx]), **kwargs):
            active = getattr(_TLS, "capture", None)
            if active is None or active.get("forward_call_index") != 3:
                return _original(*args, **kwargs)
            pair = active["events_by_layer"][_idx]
            outputs = []
            # Boundary markers bracket the Python layer invocation. Each event
            # and queued anchor uses the actual public stream inside this call.
            for side, event, label in (("pre", pair[0], "after_add" if _idx % 2 == 0 else "after_mul"),):
                stream = stream_provider()
                event.record(stream)
                callback = extension.anchor(label)
                marker_id = f"{active['capture_invocation_id']}:layer{_idx}:{side}"
                item = {"layer_idx": _idx, "layer_type": _typ, "marker_side": side,
                        "marker_id": marker_id, "label": label, "event": event,
                        "callback": callback, "stream": stream}
                active["pending"].append(item); outputs.append(item)
            try:
                result = _original(*args, **kwargs)
            except BaseException:
                raise
            side, event, label = "post", pair[1], "after_add" if _idx % 2 == 0 else "after_mul"
            stream = stream_provider()
            event.record(stream)
            callback = extension.anchor(label)
            marker_id = f"{active['capture_invocation_id']}:layer{_idx}:{side}"
            item = {"layer_idx": _idx, "layer_type": _typ, "marker_side": side,
                    "marker_id": marker_id, "label": label, "event": event,
                    "callback": callback, "stream": stream}
            active["pending"].append(item)
            return result

        layer.forward = wrapped
        layer._q27_event_anchor_wrapped = True
    emit("layer_public_event_markers_installed", layer_count=64,
         marker_count_per_capture=128, scope="pre/post Python layer boundary; endpoint markers only")


def _event_row(item, scope, shape_key, graph_obj, identity_status):
    result = item["callback"]
    def val(name):
        v = getattr(result, name, None)
        return v if isinstance(v, int) else None
    return {"layer_idx": item["layer_idx"], "layer_type": item["layer_type"],
            "marker_side": item["marker_side"], "marker_id": item["marker_id"],
            "shape_key": str(shape_key), "capture_invocation_id": scope["capture_invocation_id"],
            "forward_call_index": 3, "graph_python_id": id(graph_obj) if graph_obj is not None else None,
            "graph_identity_status": identity_status, "acl_stream_id": val("acl_stream_id"),
            "task_id": val("task_id"), "stream_rc": val("stream_rc"), "task_rc": val("task_rc"),
            "capture_rc": val("capture_rc"), "capture_status": val("capture_status"),
            "callback_rc": val("callback_rc"),
            "model_ri_nonnull": getattr(result, "model_ri_nonnull", None) is True,
            "classification": ("capture_endpoint_candidate" if
                identity_status == "unique_backend_saved_graph_object" and val("capture_status") == 1 and
                getattr(result, "model_ri_nonnull", None) is True and
                all(val(k) == 0 for k in ("stream_rc", "task_rc", "capture_rc", "callback_rc"))
                else "raw_or_unknown_marker"),
            "meaning": "public event brackets one Python layer boundary; no interior task assignment or exclusive duration"}


def capture_backend_one_with_public_events(original, backend, args, kwargs, *, emit,
        event_factory=None, record_scope_factory=None, expected_shape_key=None):
    arglist, kw = list(args), dict(kwargs)
    shape_key = arglist[0] if arglist else kw.get("shape_key")
    forward_fn = arglist[1] if len(arglist) > 1 else kw.get("forward_fn")
    if not callable(forward_fn) or shape_key is None:
        return original(backend, *args, **kwargs)
    shape_text = str(shape_key)
    if not isinstance(expected_shape_key, str) or not expected_shape_key:
        emit("layer_public_event_unresolved", reason="exact bucket-1 shape-key allowlist is required; events disabled",
             shape_key=shape_text)
        return original(backend, *args, **kwargs)
    if shape_text != expected_shape_key:
        emit("layer_public_event_unresolved", reason="shape key did not match explicit bucket-1 allowlist",
             shape_key=shape_text, expected_shape_key=expected_shape_key)
        return original(backend, *args, **kwargs)
    if event_factory is None: event_factory = _default_public_event_factory
    # 128 wrapper objects are allocated before original capture begins. The
    # underlying ACL event remains lazy until record(stream), per gen43 source.
    events_by_layer = [[event_factory(), event_factory()] for _ in range(64)]
    arglist[1] = forward_fn
    calls = {"n": 0}; invocation = _next_capture_invocation_id(); captures = []
    if record_scope_factory is None: record_scope_factory = _default_record_scope_factory()
    @functools.wraps(forward_fn)
    def tracked(*fargs, **fkwargs):
        calls["n"] += 1; idx = calls["n"]
        if idx != 3:
            return _call_in_record_scope(record_scope_factory,
                f"q27.sglang.event_capture={invocation}.shape={shape_text}.forward={idx}.kind=warmup",
                forward_fn, *fargs, **fkwargs)
        scope = {"pending": [], "events_by_layer": events_by_layer,
                 "capture_invocation_id": invocation, "forward_call_index": 3}
        captures.append(scope); _TLS.capture = scope
        try:
            return _call_in_record_scope(record_scope_factory,
                f"q27.sglang.event_capture={invocation}.shape={shape_text}.forward=3.kind=graph_capture",
                forward_fn, *fargs, **fkwargs)
        finally: _TLS.capture = None
    arglist[1] = tracked
    try:
        out = original(backend, *arglist, **kw)
    except BaseException:
        _TLS.capture = None
        retained = {"events_by_layer": events_by_layer, "captures": captures,
                    "shape_key": shape_text, "invocation": invocation,
                    "reason": "capture_one raised; graph closure unproven; no event drain/read"}
        _FAILED_EVENT_SCOPES.append(retained)
        try: emit("layer_public_event_unresolved", reason=retained["reason"],
                  forward_calls=calls["n"], capture_invocation_id=invocation,
                  pending_markers=sum(len(x["pending"]) for x in captures))
        except BaseException: pass
        raise
    if calls["n"] != 3:
        _FAILED_EVENT_SCOPES.append({"events_by_layer": events_by_layer, "captures": captures,
                                     "shape_key": shape_text, "invocation": invocation,
                                     "reason": "source forward call count changed"})
        emit("layer_public_event_unresolved", reason="source forward call count changed; retained without drain",
             expected_calls=3, actual_calls=calls["n"], capture_invocation_id=invocation)
        return out
    graph_dict = getattr(backend, "_graphs", None); graph_obj = None; identity = "unknown"
    if isinstance(graph_dict, dict) and shape_key in graph_dict:
        graph_obj = graph_dict[shape_key]
        keys = [key for key, value in graph_dict.items() if value is graph_obj]
        identity = "unique_backend_saved_graph_object" if graph_obj is not None and len(keys) == 1 else "ambiguous_or_missing_graph_object"
    if graph_obj is None or identity != "unique_backend_saved_graph_object" or len(captures) != 1:
        _FAILED_EVENT_SCOPES.append({"events_by_layer": events_by_layer, "captures": captures,
                                     "shape_key": shape_text, "invocation": invocation,
                                     "reason": identity})
        emit("layer_public_event_unresolved", reason=identity, graph_python_id=id(graph_obj) if graph_obj else None,
             forward_calls=calls["n"], capture_scopes=len(captures))
        return out
    pending = captures[0]["pending"]
    streams = {id(x["stream"]): x["stream"] for x in pending if x.get("stream") is not None}
    try:
        if len(pending) != 128 or not streams or any(not callable(getattr(s, "synchronize", None)) for s in streams.values()):
            raise RuntimeError("expected 128 markers and saved public streams before drain")
        for stream in streams.values(): stream.synchronize()
    except BaseException as exc:
        _FAILED_EVENT_SCOPES.append({"events_by_layer": events_by_layer, "captures": captures,
                                     "graph": graph_obj, "shape_key": shape_text, "invocation": invocation,
                                     "reason": "post-capture stream drain failed", "error": repr(exc)})
        emit("layer_public_event_unresolved", reason="post-capture drain failed; graph/event refs retained",
             error=repr(exc), pending_markers=len(pending), graph_python_id=id(graph_obj))
        return out
    with _GRAPH_EVENT_HOLDS_LOCK:
        _GRAPH_EVENT_HOLDS[(id(backend), id(graph_obj))] = {"backend": backend, "graph": graph_obj,
            "events_by_layer": events_by_layer, "shape_key": shape_text, "capture_invocation_id": invocation}
    rows = [_event_row(x, captures[0], shape_key, graph_obj, identity) for x in pending]
    arm_owned_graph_cleanup(backend, graph_obj, emit)
    for row in rows: emit("layer_public_event_marker", **row)
    emit("layer_public_event_summary", graph_key=shape_text, graph_python_id=id(graph_obj),
         graph_identity_status=identity, capture_invocation_id=invocation,
         forward_calls=calls["n"], marker_rows=len(rows), expected_marker_rows=128,
         capture_endpoint_rows=sum(r["classification"] == "capture_endpoint_candidate" for r in rows),
         lifetime="events strongly held with exact backend graph object until explicit graph reset/discard and post-reset release")
    return out


def release_graph_event_holds(backend, graph_obj, *, reset_receipt):
    """Release only after caller proves this exact graph was reset/discarded."""
    if not isinstance(reset_receipt, dict) or reset_receipt.get("graph_python_id") != id(graph_obj) or reset_receipt.get("reset_or_discarded") is not True:
        raise RuntimeError("explicit exact-graph reset/discard receipt required before releasing event wrappers")
    key = (id(backend), id(graph_obj))
    with _GRAPH_EVENT_HOLDS_LOCK:
        held = _GRAPH_EVENT_HOLDS.get(key)
        if held is None or held.get("graph") is not graph_obj:
            raise RuntimeError("no exact graph event hold exists")
        del _GRAPH_EVENT_HOLDS[key]
    return {"released": True, "graph_python_id": id(graph_obj), "event_count": 128}


def arm_owned_graph_cleanup(backend, graph_obj, emit):
    """Diagnostic only: controller requests cleanup after its final EOF/idle check."""
    from pathlib import Path
    import json, time
    request = os.environ.get("Q27_EVENT_CLEANUP_REQUEST")
    if not request:
        raise RuntimeError("owned cleanup request path is required")
    key=(id(backend),id(graph_obj))
    with _GRAPH_EVENT_HOLDS_LOCK:
        held=_GRAPH_EVENT_HOLDS[key]
        if held.get("cleanup_armed"):return
        held["cleanup_armed"]=True
    def worker():
        try:
            while True:
                p=Path(request)
                if p.exists():
                    row=json.loads(p.read_text())
                    if row.get("task_id")!="Q27-FULL-MAP-20261025" or os.getpid() not in row.get("worker_pids",[]):
                        raise RuntimeError("cleanup request identity mismatch")
                    break
                time.sleep(1)
            import torch
            torch.npu.set_device(backend._device_id)
            backend._update_executor.shutdown(wait=True, cancel_futures=True)
            torch.npu.synchronize()
            import sg_replay_outer_events
            outer=sg_replay_outer_events.drain_outer_replay_anchors(emit=emit)
            if outer["status"]!="drained":raise RuntimeError("outer event drain failed")
            emit("sg_replay_outer_summary",**outer)
            graph_obj.reset()
            torch.npu.synchronize()
            keys=[k for k,v in backend._graphs.items() if v is graph_obj]
            if len(keys)!=1:raise RuntimeError("cleanup exact graph key became ambiguous")
            shape=keys[0]
            del backend._graphs[shape]
            backend._outputs.pop(shape,None)
            receipt={"graph_python_id":id(graph_obj),"reset_or_discarded":True,
                     "device_drained_before_and_after_reset":True}
            outer_released=sg_replay_outer_events.release_drained_outer_events(graph_obj,reset_receipt=receipt)
            released=release_graph_event_holds(backend,graph_obj,reset_receipt=receipt)
            torch.npu.synchronize()
            emit("layer_public_event_cleanup", **receipt, events_released=released["event_count"],outer_events_released=outer_released,
                 status="reset_drained_and_events_released", serving_after_cleanup_allowed=False)
        except BaseException as exc:
            emit("layer_public_event_cleanup", status="failed_preserve_holds", error=repr(exc),
                 graph_python_id=id(graph_obj))
    threading.Thread(target=worker,name="q27-owned-graph-cleanup",daemon=True).start()
