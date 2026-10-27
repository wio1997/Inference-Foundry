"""Opt-in root-owned vLLM full-model FX layer-event diagnostic glue.

No torch/NPU import at module import. Root supplies explicit drain/emit/device
providers and owns replay fences plus shutdown cleanup.
"""
from __future__ import annotations
import functools
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import itertools
import json
import os
import re
import sys
import threading
from pathlib import Path

from fx_layer_event_injector import insert_layer_endpoint_markers

COMPILER_TARGETS = {
    "npugraph_ex.npu_fx_compiler", "torch_npu.dynamo.npugraph_ex.npu_fx_compiler",
    "npugraph_ex._acl_concrete_graph.fx2acl_converter",
    "torch_npu.dynamo.npugraph_ex._acl_concrete_graph.fx2acl_converter",
}
ACL_TARGET = "vllm_ascend.compilation.acl_graph"
_NS = "q27_vllm_layer_diag"
_LOCK = threading.RLock()
_TLS = threading.local()
_INSTALLED = set()
_COUNTER = itertools.count(1)
_EVENTS = {}                 # tag -> strong torch.npu.Event reference
_EVENT_HISTORY = {}          # tag -> replaced compile/runtime Events, held to reset
_COMPILE_ROWS = []
_GRAPHS = {}                  # id(graph) -> graph-bound strong refs
_ANCHOR_EXTENSION = None
_EMIT = None
_DRAIN = None
_TARGET_MODE = None
_DEVICE = None
_INSTALL_CONFIG = None


def anchor_extension():
    """The single loaded gen53 extension, shared with root replay fences."""
    if _ANCHOR_EXTENSION is None:
        raise RuntimeError("anchor extension is not loaded")
    return _ANCHOR_EXTENSION


def load_anchor_extension(path, expected_sha256):
    """Verify/import the Py3.12 gen78 annotated RI bridge under its exact init name."""
    global _ANCHOR_EXTENSION
    p = Path(path)
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise RuntimeError("anchor extension SHA mismatch")
    name = "q27_vllm_task_anchor_tag_root78"
    spec = importlib.util.spec_from_file_location(name, str(p))
    if spec is None or spec.loader is None:
        raise ImportError("could not load gen53 anchor extension")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    _ANCHOR_EXTENSION = module
    return module


def _marker_records(rows, capture_id):
    return [r for r in rows if r.get("capture_id") == capture_id]


def _capture_wrapper(cls, *, target_mode, drain_callbacks, emit, device_provider=None):
    key = (cls, "__call__")
    if key in _INSTALLED or not callable(getattr(cls, "__call__", None)):
        return False
    original = cls.__call__
    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        # Use the runtime's already-established forward context; no imports or
        # tensor inspection are done to classify the owner.
        from vllm.forward_context import get_forward_context
        fc = get_forward_context()
        bd = fc.batch_descriptor
        mode = fc.cudagraph_runtime_mode
        entries = getattr(self, "concrete_aclgraph_entries", None)
        entry_before = entries.get(bd) if isinstance(entries, dict) else None
        if (mode != target_mode or mode != getattr(self, "runtime_mode", None)
                or (entry_before is not None and entry_before.aclgraph is not None)):
            return original(self, *args, **kwargs)
        capture_id = f"pid{os.getpid()}-capture{next(_COUNTER)}"
        scope = {"capture_id": capture_id, "phase": "aclgraph_capture",
                 "capture_state": True, "rows": [], "batch_descriptor": bd,
                 "wrapper": self, "entry_before": entry_before}
        stack = list(getattr(_TLS, "capture_stack", []))
        _TLS.capture_stack = stack + [scope]
        try:
            result = original(self, *args, **kwargs)
        except BaseException:
            scope["status"] = "original_call_raised"
            _COMPILE_ROWS.append({"status": scope["status"], "capture_id": capture_id,
                                  "callback_rows": scope["rows"]})
            raise
        finally:
            _TLS.capture_stack = stack
        entries_after = getattr(self, "concrete_aclgraph_entries", None)
        entry = entries_after.get(bd) if isinstance(entries_after, dict) else None
        graph = getattr(entry, "aclgraph", None) if entry is not None else None
        if entry is None or graph is None or (entry_before is not None and entry is not entry_before):
            _COMPILE_ROWS.append({"status": "unknown_owner_or_graph_not_created",
                                  "capture_id": capture_id, "callback_rows": scope["rows"]})
            return result
        # Keep graph and all tag events alive before any callback extraction.
        rows = scope["rows"]
        tags = sorted({r.get("marker_id", "") + ":event" for r in rows
                       if isinstance(r.get("marker_id"), str)})
        observed_events = {r["marker_id"] + ":event": r["event_ref"] for r in rows
                           if isinstance(r.get("marker_id"), str) and r.get("event_ref") is not None}
        # On incomplete callback evidence retain all event refs until root has
        # exact ownership/reset proof; successful gates bind only this set.
        events = dict(observed_events) if len(observed_events) == 128 else dict(_EVENTS)
        event_tags = tuple(sorted(set(tags)))
        event_history = {tag: tuple(_EVENT_HISTORY.get(tag, ())) for tag in event_tags}
        record = {"graph": graph, "wrapper": self, "entry": entry,
                  "graph_key": bd, "device": device_provider(fc) if device_provider else None,
                  "events": events, "event_tags": event_tags,
                  "event_history": event_history, "capture_id": capture_id,
                  "callback_rows": rows}
        with _LOCK:
            _GRAPHS[id(graph)] = record
        try:
            # Supplied root callback performs its approved saved-stream/device
            # drain, then returns decoded rows; this wrapper does no sync itself.
            decoded = drain_callbacks(rows)
            if not isinstance(decoded, list):
                decoded = []
            valid, reason = _validate_capture_rows(decoded, capture_id)
            if valid and any((row["marker_id"] + ":event") not in events for row in decoded):
                valid, reason = False, "capture_event_ref_missing"
            # Callback drain is complete. Query borrowed RI handles now, while
            # this exact owner graph remains strongly held and before reset.
            ri_queries = (_query_capture_ri(_ANCHOR_EXTENSION, decoded,
                           capture_id=capture_id, graph=graph,
                           device_id=record["device"]) if valid else [])
            record["ri_queries"] = ri_queries
            by_marker = {q.get("marker_id"): q for q in ri_queries}
            status = "exact_capture_marker_set" if valid else "unknown_capture_marker_set"
            summary = {"status": status, "reason": reason, "capture_id": capture_id,
                       "graph_python_id": id(graph), "marker_count": len(decoded),
                       "expected_marker_count": 128,
                       "batch_descriptor": _safe_scalar(bd), "device": record["device"]}
            if valid:
                for row in decoded:
                    marker_id = row["marker_id"]
                    tag = marker_id + ":event"
                    layer_match = re.search(r":layer(\d+):(pre|post)$", marker_id)
                    safe_row = {k: v for k, v in row.items()
                                if k not in {"callback", "result", "event", "event_ref", "stream_ref"}}
                    emit("layer_public_event_marker", **{
                        "capture_invocation_id": capture_id, "capture_id": capture_id,
                        "graph_python_id": id(graph), "batch_descriptor": _safe_scalar(bd),
                        "device": record["device"], "event_ref_tag": tag,
                        "graph_identity_status": "unique_backend_saved_graph_object",
                        "classification": "capture_endpoint_candidate",
                        "producer_pid": os.getpid(),
                        "layer_idx": int(layer_match.group(1)) if layer_match else None,
                        "marker_side": layer_match.group(2) if layer_match else None,
                        "callback_nonnull": row.get("callback_nonnull"),
                        "callback_capture_status": row.get("callback_capture_status"),
                        "model_ri_query": by_marker.get(marker_id, {
                            "status": "unknown_query_missing_for_marker",
                            "marker_id": marker_id, "capture_id": capture_id,
                            "graph_python_id": id(graph), "device_id": record["device"]}),
                        **safe_row})
                if valid:
                    summary["status"] = "exact_capture_marker_set"
            summary.update({"capture_invocation_id": capture_id,
                            "marker_rows": len(decoded),
                            "capture_endpoint_rows": len(decoded) if valid else 0,
                            "ri_query_rows": len(ri_queries),
                            "ri_single_stream_candidate_rows": sum(
                                q.get("ri_bridge_status") == "candidate_singleton_pair_requires_same_profile_match"
                                for q in ri_queries),
                            "ri_queries_require_same_profile_match": True,
                            "graph_identity_status": "unique_backend_saved_graph_object"})
            emit("layer_public_event_summary", **summary)
            _COMPILE_ROWS.append(summary)
        except BaseException as exc:
            # Retain the graph/events on all post-capture failures.
            _COMPILE_ROWS.append({"status": "capture_rows_drain_or_emit_failed",
                                  "capture_id": capture_id, "graph_python_id": id(graph),
                                  "error_type": type(exc).__name__})
            raise
        return result
    wrapped._q27_vllm_fx_owner_wrapped = True
    cls.__call__ = wrapped
    _INSTALLED.add(key)
    return True


def _safe_scalar(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return None


def _validate_capture_rows(rows, capture_id):
    if not isinstance(rows, list) or len(rows) != 128:
        return False, "callback_row_count_not_128"
    expected = {f"layer{i:02d}:{edge}" for i in range(64) for edge in ("pre", "post")}
    seen = set()
    for row in rows:
        if not isinstance(row, dict): return False, "callback_row_not_object"
        if row.get("capture_id") != capture_id: return False, "capture_id_mismatch"
        marker = row.get("marker_id")
        # Prefix includes PID/compile serial and is deliberately treated opaque.
        suffix = marker.rsplit(":", 2)[-2:] if isinstance(marker, str) else []
        if len(suffix) != 2 or suffix[0] not in {f"layer{i:02d}" for i in range(64)} or suffix[1] not in {"pre", "post"}:
            return False, "marker_identity_malformed"
        ident = ":".join(suffix)
        if ident not in expected or ident in seen: return False, "marker_missing_or_duplicate"
        if (row.get("callback_capture_status") != 1
                or row.get("callback_nonnull") is not True
                or ("model_ri_nonnull" in row and row.get("model_ri_nonnull") is not True)):
            return False, "callback_not_successful_capture"
        return_codes = row.get("callback_return_codes")
        expected_rcs = {"callback_rc", "capture_rc", "stream_rc", "task_rc"}
        if not isinstance(return_codes, dict) or set(return_codes) != expected_rcs or any(rc != 0 for rc in return_codes.values()):
            return False, "callback_api_return_codes_not_all_zero"
        if row.get("event_ref") is None or row.get("stream_ref") is None:
            return False, "event_or_capture_stream_reference_missing"
        seen.add(ident)
    return seen == expected, None if seen == expected else "marker_set_incomplete"


def _query_capture_ri(extension, rows, *, capture_id, graph, device_id):
    """Query each drained marker's exact callback result before graph reset.

    Full stream lists and API results are retained. Singleton rows are only
    candidates for validation against the same raw profile; ID domains are not
    assumed equivalent, and multi-stream rows remain ambiguous.
    """
    queries = []
    seen = set()
    for row in rows:
        marker = row.get("marker_id") if isinstance(row, dict) else None
        base = {"anchor_id": marker, "marker_id": marker,
                "capture_id": capture_id, "graph_python_id": id(graph),
                "device_id": device_id}
        if (not isinstance(marker, str) or not marker or marker in seen
                or row.get("capture_id") != capture_id):
            queries.append({**base, "status": "unknown_marker_identity_or_capture_mismatch"})
            continue
        seen.add(marker)
        anchor_result = row.get("callback")
        if anchor_result is None:
            queries.append({**base, "status": "unknown_callback_result_missing"})
            continue
        try:
            result = extension.query_model_ri(anchor_result)
            if not isinstance(result, dict):
                queries.append({**base, "status": "unknown_query_result_not_object"})
                continue
            query = {**base, **result}
            streams = query.get("model_streams")
            if (query.get("model_ri_get_id_rc") != 0
                    or query.get("get_streams_rc") != 0
                    or not isinstance(streams, list) or not streams
                    or any(s.get("stream_get_id_rc") != 0
                           or s.get("model_stream_id") is None for s in streams)):
                query["ri_bridge_status"] = "unknown_query_or_stream_list"
            elif len(streams) != 1:
                query["ri_bridge_status"] = "unknown_multiple_model_streams"
            elif query.get("physical_stream_id") is None:
                query["ri_bridge_status"] = "unknown_physical_stream_id"
            else:
                query["ri_bridge_status"] = "candidate_singleton_pair_requires_same_profile_match"
                query["model_stream_id_candidate"] = streams[0]["model_stream_id"]
                query["no_integer_domain_equivalence_assumed"] = True
            query["task_annotation_status"] = (
                "unknown_extend_info_rc_or_task_id_changed"
                if query.get("extend_info_rc") not in (None, 0)
                   or query.get("task_after_rc") not in (None, 0)
                   or query.get("annotation_task_id_unchanged") is False
                else ("annotated_exact_last_task_candidate"
                      if query.get("extend_info_rc") == 0
                         and query.get("annotation_task_id_unchanged") is True
                      else "not_requested_or_api_not_reported"))
            queries.append(query)
        except Exception as exc:
            queries.append({**base, "status": "unknown_ri_query_exception",
                            "error_type": type(exc).__name__})
    return queries


def _install_compiler(module, torch, create_event):
    name = getattr(module, "__name__", "")
    if name not in COMPILER_TARGETS: return []
    installed = []
    cls = getattr(module, "AclConcreteGraph", None)
    if cls is None or (cls, "save_fx_graph") in _INSTALLED: return installed
    original = getattr(cls, "save_fx_graph", None)
    if not callable(original): return installed
    @functools.wraps(original)
    def save_fx_graph(self, graph_module, *args, **kwargs):
        serial = next(_COUNTER)
        prefix = f"pid{os.getpid()}-compile{serial}"
        tagged_op = torch.ops.npugraph_ex.tagged_event_record.default
        anchor_op = torch.ops.q27_vllm_layer_diag.anchor.default
        result = insert_layer_endpoint_markers(self, graph_module, prefix=prefix,
                    create_event_by_name=create_event,
                    tagged_event_record_op=tagged_op, anchor_op=anchor_op)
        _COMPILE_ROWS.append({"status": result.get("status"), "phase": "save_fx_graph",
                              "pid": os.getpid(), "compile_serial": serial,
                              "graph_module_id": id(graph_module),
                              "marker_count": result.get("tagged_event_record_count", 0),
                              "reason": result.get("reason")})
        if result.get("status") == "unknown":
            raise RuntimeError("Q27 FX layer markers failed closed: " + str(result.get("reason")))
        return original(self, graph_module, *args, **kwargs)
    cls.save_fx_graph = save_fx_graph
    _INSTALLED.add((cls, "save_fx_graph"))
    installed.append(f"{name}.AclConcreteGraph.save_fx_graph")
    return installed


def _install_acl_graph(module, *, target_mode, drain_callbacks, emit, device_provider=None):
    if getattr(module, "__name__", None) != ACL_TARGET: return []
    cls = getattr(module, "ACLGraphWrapper", None)
    if cls is None: return []
    return ["ACLGraphWrapper.__call__"] if _capture_wrapper(
        cls, target_mode=target_mode, drain_callbacks=drain_callbacks,
        emit=emit, device_provider=device_provider) else []


def _anchor_impl(extension):
    def phase_provider():
        stack = getattr(_TLS, "capture_stack", [])
        if stack:
            scope = stack[-1]
            return {"phase": "aclgraph_capture", "capture_state": True,
                    "capture_id": scope["capture_id"], "rows": scope["rows"]}
        return {"phase": "outside_capture", "capture_state": False,
                "capture_id": None, "rows": None}
    def record_function(label):
        import torch
        return torch.profiler.record_function(label)
    def wrapped(enum, marker_id, annotation_json):
        context = phase_provider()
        if enum not in {"after_add", "after_mul"} or not isinstance(marker_id, str) or not marker_id:
            raise ValueError("invalid anchor enum or marker identity")
        if (not isinstance(annotation_json, str)
                or len(annotation_json.encode("utf-8")) > 4096
                or "q27_marker_id" not in annotation_json
                or marker_id not in annotation_json):
            raise ValueError("invalid precomputed task annotation JSON")
        with record_function(marker_id):
            # The JSON literal is prepared during FX insertion, before capture.
            result = extension.anchor(enum, annotation_json)
        torch = sys.modules.get("torch")
        npu = getattr(torch, "npu", None) if torch is not None else None
        current_stream = getattr(npu, "current_stream", None) if npu is not None else None
        stream_ref = current_stream() if callable(current_stream) else None
        tag = marker_id + ":event"
        graph_pass = sys.modules.get("npugraph_ex._acl_concrete_graph.graph_pass")
        registry = getattr(graph_pass, "_GLOBAL_SCOPE_TAG_TO_EVENT", None)
        event_ref = registry.get(tag) if isinstance(registry, dict) else None
        if event_ref is not None:
            with _LOCK:
                history = _EVENT_HISTORY.setdefault(tag, [])
                if not any(ref is event_ref for ref in history):
                    history.append(event_ref)
                _EVENTS[tag] = event_ref
        row = {"marker_id": marker_id, "capture_id": context["capture_id"],
               "capture_scope_active": bool(context["capture_state"]),
               "callback": result, "event_ref": event_ref,
               "stream_ref": stream_ref}
        if context["rows"] is None:
            _COMPILE_ROWS.append({"status": "outside_capture_anchor_unknown",
                                  "marker_id": marker_id, "callback": result})
        else:
            context["rows"].append(row)
        return None
    return wrapped


def register_anchor_ops(torch, extension):
    """Register event op (installed package) and custom side-effect anchor."""
    importlib.import_module("npugraph_ex.ops._tagged_event")
    library = torch.library.Library(_NS, "DEF")
    library.define("anchor(str anchor_enum, str marker_id, str annotation_json) -> ()")
    op = getattr(getattr(torch.ops, _NS), "anchor").default
    from torch.fx.node import has_side_effect
    has_side_effect(op)
    library.impl("anchor", _anchor_impl(extension), "CompositeExplicitAutograd")
    library.impl("anchor", lambda enum, marker_id, annotation_json: None, "Meta")
    return library, op


def held_graphs():
    """Return graph-bound strong refs for root-owned fences/cleanup."""
    with _LOCK:
        return [dict(row) for row in _GRAPHS.values()]


def drain_captured_rows(graph):
    """Return retained raw callbacks for exact graph; caller decides when drained."""
    with _LOCK:
        row = _GRAPHS.get(id(graph))
        if row is None or row["graph"] is not graph:
            raise KeyError("graph is not held by this diagnostic")
        return list(row["callback_rows"])


def release_graph_after_reset(graph, receipt):
    """Release only after caller proves same graph reset/discard and device drains."""
    required = (isinstance(receipt, dict)
                and receipt.get("graph_python_id") == id(graph)
                and receipt.get("reset_or_discarded") is True
                and receipt.get("device_drained_before_and_after_reset") is True)
    if not required:
        return {"released": False, "reason": "reset_receipt_not_proven"}
    with _LOCK:
        row = _GRAPHS.get(id(graph))
        if row is None or row["graph"] is not graph:
            return {"released": False, "reason": "exact_graph_not_held"}
        graph_pass = sys.modules.get("npugraph_ex._acl_concrete_graph.graph_pass")
        registry = getattr(graph_pass, "_GLOBAL_SCOPE_TAG_TO_EVENT", None)
        for tag in row.get("event_tags", ()):
            _EVENTS.pop(tag, None)
            _EVENT_HISTORY.pop(tag, None)
            if isinstance(registry, dict):
                registry.pop(tag, None)
        _GRAPHS.pop(id(graph), None)
    return {"released": True, "graph_python_id": id(graph), "event_count": len(row["events"])}


def install(torch, extension, *, target_mode, drain_callbacks, emit,
            device_provider=None, module_names=()):
    """Install after ops registration; root retains returned library object."""
    global _ANCHOR_EXTENSION, _EMIT, _DRAIN, _TARGET_MODE, _DEVICE, _INSTALL_CONFIG
    _ANCHOR_EXTENSION, _EMIT, _DRAIN, _TARGET_MODE, _DEVICE = extension, emit, drain_callbacks, target_mode, device_provider
    library, anchor_op = register_anchor_ops(torch, extension)
    def create_event(tag):
        graph_pass = importlib.import_module("npugraph_ex._acl_concrete_graph.graph_pass")
        graph_pass._create_event_by_name(tag)
        event = graph_pass._GLOBAL_SCOPE_TAG_TO_EVENT.get(tag)
        if event is None:
            raise RuntimeError("tagged event registry did not retain created event")
        _EVENTS[tag] = event
        _EVENT_HISTORY.setdefault(tag, []).append(event)
        return event
    _INSTALL_CONFIG = {"torch": torch, "extension": extension,
                       "create_event": create_event, "target_mode": target_mode,
                       "drain_callbacks": drain_callbacks, "emit": emit,
                       "device_provider": device_provider}
    targets = set(COMPILER_TARGETS) | {ACL_TARGET}
    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname not in targets: return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec and spec.loader and not isinstance(spec.loader, Loader): spec.loader = Loader(spec.loader)
            return spec
    class Loader(importlib.abc.Loader):
        def __init__(self, wrapped): self.wrapped = wrapped
        def create_module(self, spec):
            fn = getattr(self.wrapped, "create_module", None)
            return fn(spec) if fn else None
        def exec_module(self, module):
            self.wrapped.exec_module(module)
            install_module(module)
    if not any(type(item).__name__ == "Finder" and getattr(item, "_q27_vllm_events", False) for item in sys.meta_path):
        finder = Finder(); finder._q27_vllm_events = True
        sys.meta_path.insert(0, finder)
    installed = []
    for name in targets | set(module_names):
        module = sys.modules.get(name)
        if module is not None: installed.extend(install_module(module))
    return {"library": library, "anchor_op": anchor_op, "installed": installed,
            "anchor_extension": extension}


def register_import_finder(torch, extension, *, target_mode, drain_callbacks, emit,
                           device_provider=None, module_names=()):
    """Root bootstrap entry; call before target imports with explicit runtime providers."""
    return install(torch, extension, target_mode=target_mode,
                   drain_callbacks=drain_callbacks, emit=emit,
                   device_provider=device_provider, module_names=module_names)


def install_module(module):
    """Patch a just-loaded or already-loaded target module; root may wrap this."""
    config = _INSTALL_CONFIG
    if config is None:
        return []
    name = getattr(module, "__name__", "")
    if name in COMPILER_TARGETS:
        return _install_compiler(module, config["torch"], config["create_event"])
    if name == ACL_TARGET:
        return _install_acl_graph(module, target_mode=config["target_mode"],
                  drain_callbacks=config["drain_callbacks"], emit=config["emit"],
                  device_provider=config["device_provider"])
    return []
