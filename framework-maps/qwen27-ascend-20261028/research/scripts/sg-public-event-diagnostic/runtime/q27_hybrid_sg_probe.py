"""Opt-in CPU-only observer for SGLang hybrid radix/Mamba COW restore.

Call install() after importing the exact runtime classes. No tensor is copied,
no device value is read, and no synchronization is added. Numeric tensor IDs are
emitted only when the supplied index object is explicitly CPU-resident.
"""
from __future__ import annotations
import os
import time
import sys
from functools import wraps

_INSTALLED = set()


def _metadata(value):
    """Return safe metadata and CPU values only; never call .cpu()/.item()."""
    if value is None:
        return {"ids": None, "device": None, "shape": None, "count": 0}
    if type(value) is int:
        return {"ids": [value], "device": "python", "shape": [], "count": 1}
    if isinstance(value, (list, tuple)) and all(type(x) is int for x in value):
        return {"ids": list(value), "device": "python", "shape": [len(value)], "count": len(value)}
    device_obj = getattr(value, "device", None)
    device = str(device_obj) if device_obj is not None else None
    device_type = getattr(device_obj, "type", None)
    shape_obj = getattr(value, "shape", None)
    shape = list(shape_obj) if shape_obj is not None else None
    try:
        count = int(value.numel())
    except Exception:
        try:
            count = len(value)
        except Exception:
            count = None
    ids = None
    if device_type == "cpu" and callable(getattr(value, "tolist", None)):
        # CPU-only metadata read. This does not copy tensor data or synchronize a device.
        try:
            raw = value.tolist()
            flat = raw if isinstance(raw, list) else [raw]
            if all(type(x) is int for x in flat):
                ids = flat
        except Exception:
            ids = None
    return {"ids": ids, "device": device, "shape": shape, "count": count}


def _emit(emit_fn, event, fields):
    row = {"event": event, "pid": os.getpid(), "mono_ns": time.monotonic_ns(),
           "wall_ns": time.time_ns()}
    row.update(fields)
    try:
        emit_fn(row)
    except Exception:
        # Diagnostics must not alter serving behavior if their sink fails.
        pass


def _observe(emit_fn, event, fields_factory):
    """Build and emit diagnostic metadata without affecting serving behavior."""
    try:
        fields = fields_factory()
        _emit(emit_fn, event, fields)
    except Exception:
        pass


class _SafeSpan:
    """A record_function-like context manager whose own failures are inert."""
    def __init__(self, factory, name):
        self.factory, self.name, self.context = factory, name, None

    def __enter__(self):
        try:
            candidate = self.factory(self.name) if self.factory is not None else None
            candidate.__enter__()
            self.context = candidate
        except Exception:
            self.context = None
        return None

    def __exit__(self, exc_type, exc, tb):
        if self.context is not None:
            try:
                self.context.__exit__(exc_type, exc, tb)
            except Exception:
                pass
        # Do not allow a diagnostic span to suppress or replace original exceptions.
        return False


def _safe_len(value):
    try:
        return len(value)
    except Exception:
        return None


def _node_id(node):
    value = getattr(node, "id", None)
    return value if type(value) in (int, str) else None


def _rid(req):
    value = getattr(req, "rid", None)
    return value if isinstance(value, (str, int)) else None


def _patch(cls, method_name, factory):
    original = getattr(cls, method_name, None)
    if original is None:
        raise AttributeError(f"{cls.__name__}.{method_name} missing")
    marker = f"_q27_hybrid_restore_{method_name}"
    if getattr(original, marker, False):
        return
    wrapped = factory(original)
    setattr(wrapped, marker, True)
    setattr(cls, method_name, wrapped)


def install(*, mamba_component_cls, schedule_batch_cls, model_runner_cls,
            hybrid_pool_cls, emit, checkpoint_pool_cls=None, span_factory=None):
    """Patch narrow methods; return installed method names; repeat-safe per class.

    Expected classes are imported by the deployment-side bootstrap from the
    active SGLang interpreter. The caller owns the JSONL sink and its lifecycle.
    """
    key = (mamba_component_cls, schedule_batch_cls, model_runner_cls,
           hybrid_pool_cls, checkpoint_pool_cls)
    if key in _INSTALLED:
        return []
    names = []

    def component_factory(original):
        @wraps(original)
        def wrapped(self, params, result):
            out = original(self, params, result)
            def fields():
                req = getattr(params, "req", None)
                node = getattr(out, "best_match_node", None)
                kv = getattr(req, "kv", None)
                src = getattr(kv, "mamba_cow_src_index", None)
                dst = getattr(kv, "mamba_pool_idx", None)
                if src is None and dst is None:
                    return None
                return {"request_id": _rid(req), "tree_node_id": _node_id(node),
                        "matched_prefix_len": _safe_len(getattr(out, "device_indices", None)),
                        "source_state_index": _metadata(src), "destination_state_index": _metadata(dst),
                        "planned": src is not None and dst is not None, "index_values_cpu_only": True}
            def emit_if_present():
                value = fields()
                if value is not None:
                    _emit(emit, "mamba_prefix_cow_planned", value)
            try:
                emit_if_present()
            except Exception:
                pass
            return out
        return wrapped

    _patch(mamba_component_cls, "finalize_match_result_in_cache", component_factory)
    names.append("MambaComponent.finalize_match_result_in_cache")

    def schedule_factory(original):
        @wraps(original)
        def wrapped(self, reqs):
            planned = []
            try:
                for req in reqs:
                    kv = getattr(req, "kv", None)
                    src = getattr(kv, "mamba_cow_src_index", None)
                    if src is not None:
                        dst = getattr(kv, "mamba_pool_idx", None)
                        planned.append({"request_id": _rid(req),
                                        "source_state_index": _metadata(src),
                                        "destination_state_index": _metadata(dst)})
            except Exception:
                planned = []
            try:
                result = original(self, reqs)
            except BaseException as exc:
                if planned:
                    _observe(emit, "mamba_cow_batch_collect_exception",
                             lambda: {"planned": planned, "exception_type": type(exc).__name__})
                raise
            if planned:
                _observe(emit, "mamba_cow_batch_collect", lambda: {
                    "planned": planned, "batch_src_indices": _metadata(getattr(self, "mamba_cow_src_indices", None)),
                    "batch_dst_indices": _metadata(getattr(self, "mamba_cow_dst_indices", None)),
                    "note": "scheduler metadata assembled; device copy not yet proven"})
            return result
        return wrapped

    _patch(schedule_batch_cls, "_collect_deferred_mamba_cow_and_clear", schedule_factory)
    names.append("ScheduleBatch._collect_deferred_mamba_cow_and_clear")

    def runner_factory(original):
        @wraps(original)
        def wrapped(self, forward_batch, *args, **kwargs):
            try:
                src = getattr(forward_batch, "mamba_cow_src_indices", None)
                dst = getattr(forward_batch, "mamba_cow_dst_indices", None)
                src_meta, dst_meta = _metadata(src), _metadata(dst)
                active = bool(src_meta["count"])
                fields = {"source_state_indices": src_meta, "destination_state_indices": dst_meta,
                          "forward_mode": str(getattr(forward_batch, "forward_mode", None)),
                          "note": "execution window; completion requires device trace"}
            except Exception:
                active, fields = False, {}
            if active:
                _observe(emit, "mamba_cow_execute_window_begin", lambda: fields)
            try:
                result = original(self, forward_batch, *args, **kwargs)
            except BaseException as exc:
                if active:
                    _observe(emit, "mamba_cow_execute_window_exception",
                             lambda: {**fields, "exception_type": type(exc).__name__})
                raise
            if active:
                _observe(emit, "mamba_cow_execute_window_return", lambda: fields)
            return result
        return wrapped

    _patch(model_runner_cls, "_maybe_execute_deferred_mamba_cow_and_clear", runner_factory)
    names.append("ModelRunner._maybe_execute_deferred_mamba_cow_and_clear")

    def pool_factory(original):
        @wraps(original)
        def wrapped(self, src_indices, dst_indices, *args, **kwargs):
            try:
                fields = {"source_state_indices_physical": _metadata(src_indices),
                          "destination_state_indices_physical": _metadata(dst_indices),
                          "copy_path": "HybridReqToTokenPool.copy_mamba_state"}
            except Exception:
                fields = {"copy_path": "HybridReqToTokenPool.copy_mamba_state", "metadata_status": "unavailable"}
            _observe(emit, "mamba_state_copy_call_begin", lambda: fields)
            try:
                with _SafeSpan(span_factory, "Q27 HybridReqToTokenPool.copy_mamba_state"):
                    result = original(self, src_indices, dst_indices, *args, **kwargs)
            except BaseException as exc:
                _observe(emit, "mamba_state_copy_call_exception",
                         lambda: {**fields, "exception_type": type(exc).__name__})
                raise
            _observe(emit, "mamba_state_copy_call_return", lambda: fields)
            return result
        return wrapped

    _patch(hybrid_pool_cls, "copy_mamba_state", pool_factory)
    names.append("HybridReqToTokenPool.copy_mamba_state")

    if checkpoint_pool_cls is not None:
        def checkpoint_factory(original):
            @wraps(original)
            def wrapped(self, active_pool, ckpt_slots, active_slots, *args, **kwargs):
                try:
                    fields = {"source_checkpoint_slots": _metadata(ckpt_slots),
                              "destination_active_slots": _metadata(active_slots),
                              "copy_path": "MambaCheckpointPool.load_to_active"}
                except Exception:
                    fields = {"copy_path": "MambaCheckpointPool.load_to_active", "metadata_status": "unavailable"}
                _observe(emit, "mamba_checkpoint_load_call_begin", lambda: fields)
                try:
                    result = original(self, active_pool, ckpt_slots, active_slots, *args, **kwargs)
                except BaseException as exc:
                    _observe(emit, "mamba_checkpoint_load_call_exception",
                             lambda: {**fields, "exception_type": type(exc).__name__})
                    raise
                _observe(emit, "mamba_checkpoint_load_call_return", lambda: fields)
                return result
            return wrapped
        _patch(checkpoint_pool_cls, "load_to_active", checkpoint_factory)
        names.append("MambaCheckpointPool.load_to_active")

    _INSTALLED.add(key)
    return names
