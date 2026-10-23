#!/usr/bin/env python3
"""E33 DeepInfer diagnostic primitives.

Three independent pieces, all task-local:

1. ``E33Telemetry`` -- buffered host-side CPU event JSONL for one rank/run.
   Every record carries framework/rank/pid/native_tid, time.time_ns,
   perf_counter_ns, thread_time_ns, run_id, event_name, request_id or
   iteration_id, an explicit start/end pair, source and exception.  Writing is
   buffered; nothing here synchronizes an NPU.

2. ``install_cpu_timer_probe`` -- wraps the *instance method*
   ``deepinfer.timer.cpu_timer.record`` while preserving its context-manager
   semantics (including ``enable=False`` and ``record_name``).  The original
   context manager is always entered and exited, and an instrumentation error
   never replaces the original exception.

3. ``CohortProfiler`` -- torch_npu profiler with no active-step schedule.  It
   records CPU+NPU at Level0 with mstx/msprof_tx/op_attr until an explicit
   ``stop()``.  There is deliberately no ``step()`` API.

``collect_layout_snapshot`` records tensor/storage metadata and device/CPU
memory counters only.  It never reads tensor values, never copies weights to
CPU and never claims a physical HBM bank layout.

Local API self-test (no torch, no NPU):
    python3 tools/e33_di_telemetry.py --selftest
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import re
import resource
import sys
import threading
import time
import traceback
from pathlib import Path

SCHEMA = "e33-di-event/1.0"
FRAMEWORK = "deepinfer"
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + ("%06d" % (time.time_ns() % 1000000))


def safe_name(value: str, what: str) -> str:
    text = str(value or "").strip()
    if not _SAFE_NAME.match(text):
        raise ValueError("%s must match %s, got %r" % (what, _SAFE_NAME.pattern, value))
    return text


def atomic_write_json(path, payload) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp.%d" % os.getpid())
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, target)
    return str(target)


def clock_snapshot() -> dict:
    return {
        "time_time_ns": time.time_ns(),
        "perf_counter_ns": time.perf_counter_ns(),
        "thread_time_ns": time.thread_time_ns(),
        "monotonic_ns": time.monotonic_ns(),
    }


def _exception_info(exc) -> dict | None:
    if exc is None:
        return None
    try:
        return {
            "type": type(exc).__name__,
            "message": str(exc)[:4000],
            "traceback": "".join(traceback.format_exception(
                type(exc), exc, exc.__traceback__))[:12000],
        }
    except Exception:  # noqa: BLE001 - diagnostics only
        return {"type": "unknown", "message": "exception formatting failed"}


# ---------------------------------------------------------------------------
# buffered event log
# ---------------------------------------------------------------------------
class E33Span:
    """One host span.  ``end`` is idempotent and never raises."""

    def __init__(self, telemetry, event_name: str, start: dict, **fields) -> None:
        self.telemetry = telemetry
        self.event_name = event_name
        self.start = start
        self.fields = fields
        self.ended = False

    def end(self, exception=None, extra: dict | None = None,
            end: dict | None = None) -> None:
        if self.ended:
            return
        self.ended = True
        try:
            self.telemetry.emit(self.event_name, start=self.start,
                                end=end or clock_snapshot(),
                                exception=exception,
                                extra=dict(self.fields.get("extra") or {}, **(extra or {})),
                                **{key: value for key, value in self.fields.items()
                                   if key != "extra"})
        except BaseException:  # noqa: BLE001 - instrumentation must not break the caller
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.end(exc)
        return False


class _NullSpan(E33Span):
    def __init__(self) -> None:  # noqa: D107
        self.telemetry = None
        self.event_name = ""
        self.start = {}
        self.fields = {}
        self.ended = False

    def end(self, *args, **kwargs) -> None:
        self.ended = True


class E33Telemetry:
    """Thread-safe buffered JSONL writer for one rank (or the parent, rank=-1)."""

    def __init__(self, *, rank: int, out_dir, run_id: str, case_id: str = "",
                 phase: str = "", framework: str = FRAMEWORK,
                 role: str = "worker", source: str = "",
                 flush_events: int = 512, flush_interval_s: float = 1.0) -> None:
        self.rank = int(rank)
        self.run_id = str(run_id)
        self.case_id = str(case_id)
        self.phase = str(phase)
        self.framework = str(framework)
        self.role = role
        self.default_source = source
        self.flush_events = max(1, int(flush_events))
        self.flush_interval_ns = int(max(0.0, float(flush_interval_s)) * 1e9)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.event_path = self.out_dir / ("events-%s.jsonl" % safe_name(phase or "run", "phase"))
        self._lock = threading.RLock()
        self._fh = open(self.event_path, "a", buffering=1 << 20)
        self._pending = []
        self._events = 0
        self._bytes = 0
        self._last_flush_ns = time.monotonic_ns()
        self._closed = False
        self._errors = []
        self._pid = os.getpid()

    # -- span API ----------------------------------------------------------
    def begin(self, event_name: str, *, request_id=None, iteration_id=None,
              source: str | None = None, extra: dict | None = None) -> E33Span:
        try:
            return E33Span(
                self, str(event_name), clock_snapshot(),
                request_id=None if request_id is None else str(request_id),
                iteration_id=None if iteration_id is None else int(iteration_id),
                source=source or self.default_source or self.role,
                extra=extra or {})
        except BaseException:  # noqa: BLE001
            return _NullSpan()

    def emit(self, event_name: str, *, start: dict | None = None,
             end: dict | None = None, request_id=None, iteration_id=None,
             source: str | None = None, exception=None,
             extra: dict | None = None) -> None:
        """Append one event.  All failures are contained in ``self._errors``."""
        try:
            start = start or clock_snapshot()
            end = end or start
            record = {
                "schema": SCHEMA,
                "framework": self.framework,
                "role": self.role,
                "rank": self.rank,
                "pid": self._pid,
                "native_tid": threading.get_native_id(),
                "thread_name": threading.current_thread().name,
                "run_id": self.run_id,
                "case_id": self.case_id,
                "phase": self.phase,
                "event_name": str(event_name),
                "request_id": None if request_id is None else str(request_id),
                "iteration_id": None if iteration_id is None else int(iteration_id),
                "source": source or self.default_source or self.role,
                "clock_domain": "host_cpu",
                "device_time": False,
                "time_time_ns_start": int(start["time_time_ns"]),
                "time_time_ns_end": int(end["time_time_ns"]),
                "perf_counter_ns_start": int(start["perf_counter_ns"]),
                "perf_counter_ns_end": int(end["perf_counter_ns"]),
                "thread_time_ns_start": int(start["thread_time_ns"]),
                "thread_time_ns_end": int(end["thread_time_ns"]),
                "thread_time_ns": int(end["thread_time_ns"]) - int(start["thread_time_ns"]),
                "start": {key: int(value) for key, value in start.items()},
                "end": {key: int(value) for key, value in end.items()},
                "exception": _exception_info(exception),
                "extra": extra or {},
            }
            line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            with self._lock:
                if self._closed:
                    return
                self._pending.append(line)
                self._events += 1
                self._bytes += len(line)
                now = time.monotonic_ns()
                if (len(self._pending) >= self.flush_events
                        or now - self._last_flush_ns >= self.flush_interval_ns):
                    self._flush_locked(now)
        except BaseException as exc:  # noqa: BLE001
            try:
                with self._lock:
                    if len(self._errors) < 50:
                        self._errors.append(repr(exc))
            except BaseException:  # noqa: BLE001
                pass

    def _flush_locked(self, now_ns: int | None = None) -> None:
        if self._pending and not self._closed:
            self._fh.write("".join(self._pending))
            self._pending = []
        self._last_flush_ns = now_ns if now_ns is not None else time.monotonic_ns()

    def flush(self) -> None:
        try:
            with self._lock:
                self._flush_locked()
                if not self._closed:
                    self._fh.flush()
        except BaseException:  # noqa: BLE001
            pass

    def close(self) -> dict:
        try:
            with self._lock:
                if not self._closed:
                    self._flush_locked()
                    self._fh.flush()
                    self._fh.close()
                    self._closed = True
        except BaseException as exc:  # noqa: BLE001
            try:
                with self._lock:
                    self._errors.append(repr(exc))
                    self._closed = True
            except BaseException:  # noqa: BLE001
                pass
        return self.stats()

    def stats(self) -> dict:
        try:
            size = self.event_path.stat().st_size
        except OSError:
            size = 0
        with self._lock:
            return {
                "path": str(self.event_path),
                "events": self._events,
                "bytes_reported": self._bytes,
                "bytes_on_disk": size,
                "pending": len(self._pending),
                "closed": self._closed,
                "errors": list(self._errors),
                "flush_events": self.flush_events,
                "flush_interval_s": self.flush_interval_ns / 1e9,
            }


# ---------------------------------------------------------------------------
# cpu_timer.record probe
# ---------------------------------------------------------------------------
def install_cpu_timer_probe(telemetry: E33Telemetry, timer=None,
                            install_id: str = "e33") -> bool:
    """Wrap ``timer.record`` on the instance; return True if newly installed.

    The original record is always used as the context manager.  If this probe
    is already installed it only retargets the telemetry sink, so repeated
    cohort starts do not stack wrappers.
    """
    if timer is None:
        from deepinfer.timer import cpu_timer as timer  # noqa: PLC0415
    flag = "_%s_probe_installed" % install_id
    tel_attr = "_%s_telemetry" % install_id
    if getattr(timer, flag, False):
        setattr(timer, tel_attr, telemetry)
        return False
    original = timer.record

    @contextlib.contextmanager
    def probed_record(name, enable: bool = True, record_name=None):
        tel = getattr(timer, tel_attr, None)
        span = None
        if tel is not None:
            try:
                span = tel.begin(
                    str(name), source="deepinfer.cpu_timer.record",
                    extra={"timer": "cpu_timer", "enable": bool(enable),
                           "record_name": record_name})
            except BaseException:  # noqa: BLE001
                span = None
        try:
            with original(name, enable=enable, record_name=record_name):
                yield
        except BaseException as exc:
            if span is not None:
                span.end(exc)
            raise
        else:
            if span is not None:
                span.end()

    timer.record = probed_record
    setattr(timer, "_%s_original_record" % install_id, original)
    setattr(timer, flag, True)
    setattr(timer, tel_attr, telemetry)
    return True


def set_cpu_timer_telemetry(telemetry, timer=None, install_id: str = "e33") -> None:
    if timer is None:
        from deepinfer.timer import cpu_timer as timer  # noqa: PLC0415
    if getattr(timer, "_%s_probe_installed" % install_id, False):
        setattr(timer, "_%s_telemetry" % install_id, telemetry)


# ---------------------------------------------------------------------------
# cohort profiler (no schedule, explicit stop)
# ---------------------------------------------------------------------------
_TRACE_FILE_PATTERNS = {
    "kernel_details": ("kernel_details*.csv", "kernel_details*.csv.gz"),
    "op_summary": ("op_summary*.csv", "op_summary*.csv.gz"),
    "op_statistic": ("op_statistic*.csv", "op_statistic*.csv.gz"),
    "trace_view": ("trace_view.json",),
    "step_trace": ("step_trace*.json",),
    "profiler_info": ("profiler_info.json",),
    "mstx": ("*mstx*.csv",),
    "msprof_tx": ("*msprof_tx*.csv",),
}


def discover_profile_files(root) -> dict:
    base = Path(root)
    found = {}
    files = []
    total = 0
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        total += size
        files.append({"path": str(path), "bytes": size,
                      "relative": str(path.relative_to(base))})
    for key, patterns in _TRACE_FILE_PATTERNS.items():
        if isinstance(patterns, str):
            patterns = (patterns,)
        matches = set()
        for pattern in patterns:
            matches.update(str(path) for path in base.rglob(pattern)
                           if path.is_file())
        found[key] = sorted(matches)
    raw_dirs = sorted({str(path) for path in base.rglob("*_ascend_pt") if path.is_dir()})
    trace_present = bool(found["trace_view"] or found["kernel_details"]
                         or found["op_summary"])
    return {
        "out_dir": str(base),
        "files": files[:20000],
        "file_count": len(files),
        "total_bytes": total,
        "truncated_file_list": len(files) > 20000,
        "raw_dirs": raw_dirs,
        **found,
        "trace_present": trace_present,
        "csv_present": bool(found["kernel_details"] or found["op_summary"]),
        "raw_present": bool(raw_dirs or found["profiler_info"]),
    }


class CohortProfiler:
    """Whole-cohort torch_npu profiler; records until ``stop``.

    No ``schedule=`` is passed and no ``step()`` is ever called.  If the
    installed API requires a schedule, ``start`` fails closed instead of
    silently truncating the cohort.
    """

    def __init__(self, *, out_dir, run_id: str, case_id: str, phase: str,
                 rank: int, device: int, identity: dict | None = None,
                 profiler_level: str = "Level1", op_attr: bool = True,
                 worker_name: str | None = None) -> None:
        self.out_dir = Path(out_dir)
        self.run_id = str(run_id)
        self.case_id = str(case_id)
        self.phase = str(phase)
        self.rank = int(rank)
        self.device = int(device)
        self.identity = identity
        self.profiler_level = profiler_level
        self.op_attr = bool(op_attr)
        self.worker_name = worker_name or ("rank%02d-e33-%s" % (rank, safe_name(phase, "phase")))
        self.mstx_label = "e33_cohort:%s" % safe_name(phase, "phase")
        self.mstx_domain = "e33"
        self._prof = None
        self._mstx = None
        self._mstx_range = None
        self.mstx_range_id = None
        self.mstx_stopped_range_id = None
        self.started = False
        self.stopped = False
        self.started_wall_ns = None
        self.stopped_wall_ns = None
        self.started_perf_counter_ns = None
        self.stopped_perf_counter_ns = None
        self.started_thread_time_ns = None
        self.stopped_thread_time_ns = None
        self.errors = []
        self.api = {}
        self.export = {}
        self.manifest_path = self.out_dir / "profile_manifest.json"

    def _error(self, where: str, exc) -> None:
        self.errors.append({"where": where, "error": repr(exc),
                            "traceback": traceback.format_exc()[:8000]})

    def start(self) -> dict:
        if self.started:
            raise RuntimeError("cohort profiler already started")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        import inspect  # noqa: PLC0415
        import importlib.metadata  # noqa: PLC0415
        import torch  # noqa: PLC0415
        import torch_npu  # noqa: PLC0415
        from torch_npu.profiler import (  # noqa: PLC0415
            ProfilerActivity, _ExperimentalConfig, profile, tensorboard_trace_handler,
        )
        from torch_npu.npu import mstx  # noqa: PLC0415

        level = getattr(torch_npu.profiler.ProfilerLevel, self.profiler_level)
        wanted_exp = {"profiler_level": level, "mstx": True, "msprof_tx": True,
                      "op_attr": self.op_attr, "data_simplification": False,
                      "export_type": [torch_npu.profiler.ExportType.Text, torch_npu.profiler.ExportType.Db],
                      "aic_metrics": torch_npu.profiler.AiCMetrics.AiCoreNone}
        exp_params = inspect.signature(_ExperimentalConfig.__init__).parameters
        missing_exp = [key for key in ("mstx", "msprof_tx")
                       if key not in exp_params]
        if missing_exp:
            raise RuntimeError("installed _ExperimentalConfig lacks %r; supported=%r"
                               % (missing_exp, sorted(exp_params)))
        exp_missing_optional = [key for key in ("op_attr", "data_simplification")
                                if key not in exp_params]
        if "op_attr" in exp_missing_optional:
            self.errors.append({
                "where": "experimental_config",
                "error": "installed _ExperimentalConfig has no op_attr; "
                         "the requested op-attribute capture is incomplete",
            })
        applied_exp = {key: value for key, value in wanted_exp.items()
                       if key in exp_params}
        config = _ExperimentalConfig(**applied_exp)

        handler_kwargs = {"dir_name": str(self.out_dir),
                          "worker_name": self.worker_name,
                          "analyse_flag": True, "async_mode": False}
        try:
            handler_params = inspect.signature(
                tensorboard_trace_handler).parameters
            handler_missing = [key for key in ("analyse_flag", "async_mode")
                               if key not in handler_params
                               and not any(p.kind == p.VAR_KEYWORD
                                           for p in handler_params.values())]
            if handler_missing:
                self.errors.append({
                    "where": "handler",
                    "error": "tensorboard_trace_handler lacks %r; trace may not be analysed"
                             % handler_missing})
            handler = tensorboard_trace_handler(
                **{key: value for key, value in handler_kwargs.items()
                   if key in handler_params
                   or any(p.kind == p.VAR_KEYWORD for p in handler_params.values())})
        except BaseException as exc:  # noqa: BLE001
            self._error("handler", exc)
            raise

        profile_params = inspect.signature(profile).parameters
        if "schedule" in profile_params:
            param = profile_params["schedule"]
            if param.default is inspect.Parameter.empty and param.kind not in (
                    inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                raise RuntimeError(
                    "installed torch_npu.profiler.profile requires schedule; "
                    "the E33 whole-cohort no-schedule contract is unavailable")
        profile_kwargs = {
            "activities": [ProfilerActivity.CPU, ProfilerActivity.NPU],
            "on_trace_ready": handler,
            "record_shapes": True,
            "with_stack": False,
            "experimental_config": config,
        }
        profile_kwargs = {key: value for key, value in profile_kwargs.items()
                          if key in profile_params}
        self.api = {
            "profile": "torch_npu.profiler.profile",
            "torch_npu_version": getattr(torch_npu, "__version__", "unknown"),
            "torch_version": getattr(torch, "__version__", "unknown"),
            "activities": ["CPU", "NPU"],
            "schedule": {"mode": "none", "explicit_stop_required": True,
                         "step_calls": 0},
            "record_shapes": True,
            "with_stack": False,
            "mstx_range": {"label": self.mstx_label, "domain": self.mstx_domain},
            "handler": {"dir_name": str(self.out_dir),
                        "worker_name": self.worker_name,
                        "analyse_flag": True, "async_mode": False},
            "experimental_config_requested": {
                "profiler_level": self.profiler_level, "mstx": True,
                "msprof_tx": True, "op_attr": self.op_attr,
                "data_simplification": False, "export_type": ["Text", "Db"],
                "aic_metrics": "AiCoreNone"},
            "experimental_config_applied": {
                key: str(value) for key, value in applied_exp.items()},
            "experimental_config_missing_optional": exp_missing_optional,
        }
        try:
            self._prof = profile(**profile_kwargs)
            self._prof.start()
        except BaseException as exc:  # noqa: BLE001
            self._error("profile_start", exc)
            raise
        self._mstx = mstx
        self.started = True
        self.started_wall_ns = time.time_ns()
        self.started_perf_counter_ns = time.perf_counter_ns()
        self.started_thread_time_ns = time.thread_time_ns()
        try:
            self.mstx_start_before = {"wall_ns": time.time_ns(), "perf_counter_ns": time.perf_counter_ns()}
            self._mstx_range = self._mstx.range_start(
                self.mstx_label, domain=self.mstx_domain)
            self.mstx_start_after = {"wall_ns": time.time_ns(), "perf_counter_ns": time.perf_counter_ns()}
            self.mstx_range_id = self._mstx_range
            if not self._mstx_range:
                raise RuntimeError("E33 MSTX range_start returned an invalid range id")
        except BaseException as exc:  # noqa: BLE001
            self._error("mstx_range_start", exc)
        return self.status()

    def status(self) -> dict:
        return {
            "rank": self.rank,
            "device": self.device,
            "run_id": self.run_id,
            "case_id": self.case_id,
            "phase": self.phase,
            "out_dir": str(self.out_dir),
            "started": self.started,
            "stopped": self.stopped,
            "started_wall_ns": self.started_wall_ns,
            "stopped_wall_ns": self.stopped_wall_ns,
            "started_perf_counter_ns": self.started_perf_counter_ns,
            "stopped_perf_counter_ns": self.stopped_perf_counter_ns,
            "started_thread_time_ns": self.started_thread_time_ns,
            "stopped_thread_time_ns": self.stopped_thread_time_ns,
            "manifest": str(self.manifest_path),
            "api": self.api,
            "mstx": {"range_label": self.mstx_label,
                     "domain": self.mstx_domain,
                     "start_call_bracket": [getattr(self, "mstx_start_before", None), getattr(self, "mstx_start_after", None)],
                     "end_call_bracket": [getattr(self, "mstx_end_before", None), getattr(self, "mstx_end_after", None)],
                     "started_range_id": self.mstx_range_id,
                     "stopped_range_id": self.mstx_stopped_range_id},
            "errors": list(self.errors),
            "export": {key: value for key, value in self.export.items()
                       if key != "files"},
            "qualified": None,
            "qualification": "deferred_to_main_controller",
        }

    def stop(self, *, stopped_reason: str = "explicit_stop") -> dict:
        if not self.started:
            raise RuntimeError("cohort profiler is not started")
        if self.stopped:
            return self.status()
        if self._mstx_range is not None:
            try:
                self.mstx_end_before = {"wall_ns": time.time_ns(), "perf_counter_ns": time.perf_counter_ns()}
                self._mstx.range_end(self._mstx_range, domain=self.mstx_domain)
                self.mstx_end_after = {"wall_ns": time.time_ns(), "perf_counter_ns": time.perf_counter_ns()}
                self.mstx_stopped_range_id = self._mstx_range
            except BaseException as exc:  # noqa: BLE001
                self._error("mstx_range_end", exc)
            self._mstx_range = None
        try:
            self._prof.stop()
        except BaseException as exc:  # noqa: BLE001
            self._error("profile_stop", exc)
        finally:
            self._prof = None
            self.stopped = True
            self.stopped_wall_ns = time.time_ns()
            self.stopped_perf_counter_ns = time.perf_counter_ns()
            self.stopped_thread_time_ns = time.thread_time_ns()
        self.export = discover_profile_files(self.out_dir)
        self.export["ok"] = bool(self.export.get("trace_present")
                                 and not self.errors)
        export_errors = [error for error in self.errors
                         if error.get("where", "").startswith("profile")
                         or error.get("where") in ("handler", "mstx_range_end",
                                                   "experimental_config")]
        self.export["errors"] = export_errors
        manifest = {
            "schema_version": "e33-cohort/1.0",
            "synthetic": False,
            "run_id": self.run_id,
            "case_id": self.case_id,
            "phase": self.phase,
            "rank": self.rank,
            "device": self.device,
            "out_dir": str(self.out_dir),
            "started_wall_ns": self.started_wall_ns,
            "stopped_wall_ns": self.stopped_wall_ns,
            "stopped": self.stopped,
            "stop_complete": self.stopped and not any(
                error.get("where") == "profile_stop" for error in self.errors),
            "stopped_reason": stopped_reason,
            "schedule": {"mode": "none", "explicit_stop_required": True,
                         "step_calls": 0,
                         "note": "torch_npu profiler recorded continuously until stop; "
                                 "no ProfilerStep schedule was used"},
            "steady_state": {
                "present": False,
                "ranges": [],
                "note": "E33 records the whole explicit cohort; it is not a "
                        "ProfilerStep steady window and must not be qualified as one",
            },
            "mstx": {
                "range_label": self.mstx_label,
                "domain": self.mstx_domain,
                "start_call_bracket": [getattr(self, "mstx_start_before", None), getattr(self, "mstx_start_after", None)],
                "end_call_bracket": [getattr(self, "mstx_end_before", None), getattr(self, "mstx_end_after", None)],
                "started_range_id": self.mstx_range_id,
                "stopped_range_id": self.mstx_stopped_range_id,
                "note": "the exported trace should contain this range around the "
                        "whole explicit cohort; it is not a steady-state window",
            },
            "api": self.api,
            "identity": self.identity,
            "clock": {
                "host_clock_domain": "time.time_ns / perf_counter_ns / thread_time_ns",
                "started_wall_ns": self.started_wall_ns,
                "stopped_wall_ns": self.stopped_wall_ns,
                "started_perf_counter_ns": self.started_perf_counter_ns,
                "stopped_perf_counter_ns": self.stopped_perf_counter_ns,
                "started_thread_time_ns": self.started_thread_time_ns,
                "stopped_thread_time_ns": self.stopped_thread_time_ns,
                "device_clock_mapping": None,
                "clock_mapping_error_bound_ns": None,
                "note": "host clocks and NPU trace clocks are not aligned here",
            },
            "export": self.export,
            "errors": self.errors,
            "qualified": None,
            "qualification": "deferred_to_main_controller",
            "qualification_note": "no trace/CSV rows or a missing rank never makes "
                                  "this manifest qualified; the main controller decides",
            "limits": [
                "no active-step schedule: the trace covers the whole explicit "
                "start-to-stop interval and grows with cohort duration",
                "host clocks and device trace clocks are not aligned; no unique "
                "device critical path is claimed here",
                "native EngramBuffer allocation may expose only capacity, not a "
                "torch storage; unknown metadata stays unknown",
                "fused/native subregions may be invisible in the exported trace",
            ],
            "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        try:
            atomic_write_json(self.manifest_path, manifest)
        except BaseException as exc:  # noqa: BLE001 - return the failure in status
            self._error("manifest_write", exc)
        return self.status()


# ---------------------------------------------------------------------------
# memory / layout snapshot
# ---------------------------------------------------------------------------
def _first_attr(obj, names):
    for name in names:
        try:
            value = getattr(obj, name)
        except BaseException:  # noqa: BLE001
            continue
        if value is not None:
            return value
    return None


def _device_memory() -> dict:
    result = {
        "source": None,
        "total_bytes": None,
        "free_bytes": None,
        "used_bytes": None,
        "torch_allocated_bytes": None,
        "torch_reserved_bytes": None,
        "torch_peak_allocated_bytes": None,
        "torch_peak_reserved_bytes": None,
        "torch_memory_stats": None,
        "errors": [],
    }
    try:
        import torch  # noqa: PLC0415
        try:
            import torch_npu  # noqa: F401,PLC0415
        except BaseException:  # noqa: BLE001
            pass
        acc = None
        source = None
        for candidate in (getattr(torch, "npu", None), getattr(torch, "cuda", None)):
            if candidate is not None and hasattr(candidate, "mem_get_info"):
                acc, source = candidate, ("npu" if candidate is getattr(torch, "npu", None)
                                          else "cuda")
                break
        if acc is None:
            result["errors"].append("no torch.npu/torch.cuda mem_get_info")
            return result
        result["source"] = source
        info = acc.mem_get_info()
        if isinstance(info, (tuple, list)) and len(info) >= 2:
            free, total = int(info[0]), int(info[1])
            result["free_bytes"] = free
            result["total_bytes"] = total
            result["used_bytes"] = total - free
        for key, names in (
                ("torch_allocated_bytes", ("memory_allocated",)),
                ("torch_reserved_bytes", ("memory_reserved",)),
                ("torch_peak_allocated_bytes", ("max_memory_allocated",)),
                ("torch_peak_reserved_bytes", ("max_memory_reserved",))):
            func = _first_attr(acc, names)
            if func is None:
                continue
            try:
                result[key] = int(func())
            except BaseException as exc:  # noqa: BLE001
                result["errors"].append("%s: %r" % (names[0], exc))
        stats = _first_attr(acc, ("memory_stats",))
        if stats is not None:
            try:
                raw = stats()
                result["torch_memory_stats"] = {
                    str(key): (int(value) if isinstance(value, (int, float))
                               else str(value))
                    for key, value in dict(raw).items()
                }
            except BaseException as exc:  # noqa: BLE001
                result["errors"].append("memory_stats: %r" % (exc,))
    except BaseException as exc:  # noqa: BLE001
        result["errors"].append(repr(exc))
    return result


def _cpu_memory() -> dict:
    result = {
        "rss_bytes": None,
        "peak_rss_bytes": None,
        "vm_hwm_bytes": None,
        "affinity": None,
        "affinity_count": None,
        "cpus_allowed_list": None,
        "mems_allowed_list": None,
        "cgroup_memory_current_bytes": None,
        "cgroup_memory_max_bytes": None,
        "pinned_memory_bytes": None,
        "pinned_memory_note": "not exposed by a stable API here; kept unknown",
        "errors": [],
    }
    try:
        with open("/proc/self/status", "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                value = value.strip()
                if key == "VmRSS":
                    result["rss_bytes"] = int(value.split()[0]) * 1024
                elif key == "VmHWM":
                    result["vm_hwm_bytes"] = int(value.split()[0]) * 1024
                elif key == "Cpus_allowed_list":
                    result["cpus_allowed_list"] = value
                elif key == "Mems_allowed_list":
                    result["mems_allowed_list"] = value
    except OSError as exc:
        result["errors"].append("proc_status: %r" % (exc,))
    try:
        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        result["peak_rss_bytes"] = peak if sys.platform == "darwin" else peak * 1024
    except BaseException as exc:  # noqa: BLE001
        result["errors"].append("rusage: %r" % (exc,))
    get_affinity = getattr(os, "sched_getaffinity", None)
    if get_affinity is not None:
        try:
            allowed = sorted(int(cpu) for cpu in get_affinity(0))
            result["affinity"] = allowed
            result["affinity_count"] = len(allowed)
        except BaseException as exc:  # noqa: BLE001
            result["errors"].append("affinity: %r" % (exc,))
    for key, path in (
            ("cgroup_memory_current_bytes", "/sys/fs/cgroup/memory.current"),
            ("cgroup_memory_max_bytes", "/sys/fs/cgroup/memory.max")):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                text = fh.read().strip()
            result[key] = None if text == "max" else int(text)
        except (OSError, ValueError):
            pass
    return result


def _tensor_entry(t, category: str, name: str) -> dict | None:
    try:
        import torch  # noqa: PLC0415
        if not torch.is_tensor(t):
            return None
        shape = [int(value) for value in t.shape]
        stride = [int(value) for value in t.stride()]
        dtype = str(t.dtype)
        device = str(t.device)
        numel = int(t.numel())
        element_size = int(t.element_size())
        tensor_nbytes = numel * element_size
        storage = t.untyped_storage()
        ptr = int(storage.data_ptr())
        storage_nbytes = int(storage.nbytes())
        storage_offset = int(t.storage_offset())
    except BaseException:  # noqa: BLE001
        return None
    return {
        "category": category,
        "name": name,
        "shape": shape,
        "dtype": dtype,
        "stride": stride,
        "device": device,
        "numel": numel,
        "element_size": element_size,
        "tensor_nbytes": tensor_nbytes,
        "storage_ptr": ptr,
        "storage_ptr_hex": "0x%x" % ptr if ptr else "0x0",
        "storage_nbytes": storage_nbytes,
        "storage_offset": storage_offset,
        "tensor_values_read": False,
    }


def _shallow_tensors(obj, category: str, prefix: str, limit: int = 256) -> list:
    entries = []
    try:
        values = vars(obj)
    except BaseException:  # noqa: BLE001
        return entries
    for name, value in list(values.items())[:limit]:
        if name.startswith("__"):
            continue
        try:
            import torch  # noqa: PLC0415
            if torch.is_tensor(value):
                entry = _tensor_entry(value, category, "%s.%s" % (prefix, name))
                if entry is not None:
                    entries.append(entry)
            elif isinstance(value, (list, tuple)) and len(value) <= 64:
                for index, item in enumerate(value):
                    if torch.is_tensor(item):
                        entry = _tensor_entry(item, category,
                                              "%s.%s[%d]" % (prefix, name, index))
                        if entry is not None:
                            entries.append(entry)
        except BaseException:  # noqa: BLE001
            continue
    return entries


def _collect_model_tensors(model, max_modules: int, max_tensors: int) -> tuple:
    entries = []
    seen = set()
    visited_modules = 0
    truncated = False
    try:
        named_modules = model.named_modules()
    except BaseException:  # noqa: BLE001
        return entries, 0, True
    for module_name, module in named_modules:
        visited_modules += 1
        if visited_modules > max_modules or len(entries) >= max_tensors:
            truncated = True
            break
        try:
            params = list(module.named_parameters(recurse=False, remove_duplicate=True))
        except TypeError:
            params = list(module.named_parameters(recurse=False))
        except BaseException:  # noqa: BLE001
            params = []
        try:
            buffers = list(module.named_buffers(recurse=False))
        except BaseException:  # noqa: BLE001
            buffers = []
        for param_name, tensor in params:
            if id(tensor) in seen:
                continue
            seen.add(id(tensor))
            if len(entries) >= max_tensors:
                truncated = True
                break
            full = ("%s.%s" % (module_name, param_name)).lstrip(".")
            category = "scales" if any(token in param_name.lower() for token in
                                       ("scale", "_sf", "dequant")) else "weights"
            entry = _tensor_entry(tensor, category, full)
            if entry is not None:
                entries.append(entry)
        for buffer_name, tensor in buffers:
            if id(tensor) in seen:
                continue
            seen.add(id(tensor))
            if len(entries) >= max_tensors:
                truncated = True
                break
            full = ("%s.%s" % (module_name, buffer_name)).lstrip(".")
            category = ("scales" if any(token in buffer_name.lower() for token in
                                        ("scale", "_sf", "dequant"))
                        else "model_buffers")
            entry = _tensor_entry(tensor, category, full)
            if entry is not None:
                entries.append(entry)
    return entries, visited_modules, truncated


def _collect_known_tensors(engine, max_tensors: int) -> tuple:
    entries = []
    notes = []
    driver = getattr(engine, "driver_worker", None)
    if driver is None:
        notes.append("engine.driver_worker missing")
        return entries, notes
    cache_engine = getattr(driver, "cache_engine", None)
    if cache_engine is not None:
        for name in ("gpu_blocks", "gpu_states"):
            entry = _tensor_entry(getattr(cache_engine, name, None),
                                  "state_cache" if name == "gpu_states" else "kv_cache",
                                  "cache_engine.%s" % name)
            if entry is not None:
                entries.append(entry)
        cache_list = getattr(cache_engine, "cache_list", None) or []
        for index, cache in enumerate(list(cache_list)[:512]):
            for field in ("ica_cache", "ca_cache", "swa_cache"):
                entry = _tensor_entry(
                    getattr(cache, field, None), "kv_cache",
                    "cache_engine.cache_list[%d].%s" % (index, field))
                if entry is not None:
                    entries.append(entry)
            for field in ("ica_state", "ca_state"):
                entry = _tensor_entry(
                    getattr(cache, field, None), "state_cache",
                    "cache_engine.cache_list[%d].%s" % (index, field))
                if entry is not None:
                    entries.append(entry)
        notes.append("kvcache_views=%d" % len(cache_list))
    else:
        notes.append("driver.cache_engine missing")
    copy_state = getattr(engine, "copy_state", None)
    if copy_state is not None:
        try:
            values = list(copy_state) if isinstance(copy_state, (list, tuple)) else [copy_state]
        except BaseException:  # noqa: BLE001
            values = []
        for index, tensor in enumerate(values[:512]):
            entry = _tensor_entry(tensor, "state_copy", "engine.copy_state[%d]" % index)
            if entry is not None:
                entries.append(entry)
    try:
        from deepinfer.model_executor.parallel_utils.userbuffer import (  # noqa: PLC0415
            userbuffer,
        )
        entries.extend(_shallow_tensors(userbuffer, "comm_buffer", "userbuffer", limit=64))
        for index, spec in enumerate(getattr(userbuffer, "_sym_buffers", {}).values()):
            buffer = _first_attr(spec, ("buffer",))
            entry = _tensor_entry(buffer, "comm_symm_buffer",
                                  "userbuffer.symm[%d].buffer" % index)
            if entry is not None:
                entries.append(entry)
            else:
                entries.extend(_shallow_tensors(
                    spec, "comm_symm_buffer", "userbuffer.symm[%d]" % index, limit=64))
        engram = getattr(userbuffer, "engram", None)
        if engram is not None:
            entries.extend(_shallow_tensors(
                engram, "engram_buffer", "userbuffer.engram", limit=128))
            gpu_bytes = _first_attr(engram, ("num_gpu_bytes",))
            rdma_bytes = _first_attr(engram, ("num_rdma_storage_bytes",
                                              "num_rdma_storage_size"))
            if gpu_bytes is not None:
                notes.append("engram_num_gpu_bytes=%s (reported capacity, not a "
                             "reachable storage sum)" % gpu_bytes)
            notes.append(
                "engram buffer metadata limitation: num_gpu_bytes=%r "
                "num_rdma_storage_bytes=%r are capacities; the live allocation is "
                "owned by the native EngramBuffer/HCCL runtime and exposes no "
                "Python torch storage, so shape/dtype/stride/storage_ptr/"
                "storage_nbytes are unknown, not zero"
                % (gpu_bytes, rdma_bytes))
        ep = getattr(userbuffer, "ep", None)
        if ep is not None:
            entries.extend(_shallow_tensors(ep, "comm_buffer", "userbuffer.ep", limit=128))
            reported = _first_attr(ep, ("num_bytes",))
            if reported is not None:
                notes.append("ep_num_bytes=%s (reported capacity, not a reachable "
                             "storage sum; native allocation may have no torch "
                             "storage metadata)" % reported)
        cp = getattr(userbuffer, "cp", None)
        if cp is not None:
            entries.extend(_shallow_tensors(cp, "comm_buffer", "userbuffer.cp", limit=128))
    except BaseException as exc:  # noqa: BLE001
        notes.append("userbuffer inspect failed: %r" % (exc,))
    try:
        from deepinfer.model_executor.models.v4_modules.engram import (  # noqa: PLC0415
            CompressedTokenizer,
        )
        entry = _tensor_entry(getattr(CompressedTokenizer, "lookup_table_gpu", None),
                              "engram_lookup", "CompressedTokenizer.lookup_table_gpu")
        if entry is not None:
            entries.append(entry)
    except BaseException as exc:  # noqa: BLE001
        notes.append("CompressedTokenizer inspect failed: %r" % (exc,))
    runners = []
    model_runner = getattr(driver, "model_runner", None)
    if model_runner is not None:
        for attr in ("cuda_graph_runner", "dflash_cuda_graph_runner"):
            runner = getattr(model_runner, attr, None)
            if runner is not None:
                runners.append((attr, runner))
    for attr, runner in runners:
        graph_count = None
        try:
            graphs = getattr(runner, "graphs", None)
            graph_count = len(graphs) if graphs is not None else None
        except BaseException:  # noqa: BLE001
            pass
        notes.append("graph_runner.%s graphs=%r" % (attr, graph_count))
        notes.append("graph private pools are not separately exposed by torch; "
                     "they remain inside torch reserved/unknown, not a disjoint category")
        entries.extend(_shallow_tensors(runner, "graph_pool",
                                        "model_runner.%s" % attr, limit=128))
    return entries[:max_tensors], notes


def _dedupe_storages(entries: list) -> dict:
    """Storage-level ledger; a shared storage is counted once, never per view."""
    storages = {}
    zero_ptr = []
    for entry in entries:
        ptr = int(entry.get("storage_ptr") or 0)
        nbytes = int(entry.get("storage_nbytes") or 0)
        if ptr == 0 or nbytes == 0:
            zero_ptr.append(entry.get("name"))
            continue
        key = (ptr, nbytes)
        rec = storages.setdefault(key, {
            "devices": [],
            "storage_ptr": ptr,
            "storage_ptr_hex": entry.get("storage_ptr_hex"),
            "storage_nbytes": nbytes,
            "categories": [],
            "names": [],
            "tensor_count": 0,
        })
        device = entry.get("device")
        if device is not None and device not in rec["devices"]:
            rec["devices"].append(device)
        if entry.get("category") not in rec["categories"]:
            rec["categories"].append(entry.get("category"))
        rec["tensor_count"] += 1
        if len(rec["names"]) < 8:
            rec["names"].append(entry.get("name"))
    first_category_bytes = {}
    first_category_storages = {}
    cross_alias_bytes = 0
    alias_storages = 0
    cross_device_storages = 0
    for rec in storages.values():
        categories = sorted(rec["categories"], key=lambda value: str(value))
        owner = categories[0] if categories else "unknown"
        first_category_bytes[owner] = first_category_bytes.get(owner, 0) + rec["storage_nbytes"]
        first_category_storages[owner] = first_category_storages.get(owner, 0) + 1
        if rec["tensor_count"] > 1:
            alias_storages += 1
        if len(categories) > 1:
            cross_alias_bytes += rec["storage_nbytes"]
        if len(rec["devices"]) > 1:
            cross_device_storages += 1
    unique_bytes = sum(rec["storage_nbytes"] for rec in storages.values())
    by_device_type = {}
    for rec in storages.values():
        types = sorted({str(device).split(":", 1)[0]
                        for device in rec["devices"] if device})
        if len(types) == 1:
            group = types[0]
        elif not types:
            group = "unknown"
        else:
            group = "mixed"
        by_device_type[group] = by_device_type.get(group, 0) + rec["storage_nbytes"]
    accelerator_types = {"npu", "cuda", "xpu", "hpu"}
    accelerator_bytes = sum(value for key, value in by_device_type.items()
                            if key in accelerator_types)
    cpu_bytes = by_device_type.get("cpu", 0)
    other_bytes = unique_bytes - accelerator_bytes - cpu_bytes
    span_bytes = sum(int(entry.get("tensor_nbytes") or 0) for entry in entries)
    largest = sorted(storages.values(), key=lambda rec: rec["storage_nbytes"],
                     reverse=True)[:500]
    return {
        "storage_count": len(storages),
        "unique_storage_bytes": unique_bytes,
        "accelerator_unique_storage_bytes": accelerator_bytes,
        "cpu_unique_storage_bytes": cpu_bytes,
        "other_or_mixed_unique_storage_bytes": other_bytes,
        "unique_storage_bytes_by_device_type": by_device_type,
        "tensor_span_bytes_summed_per_view": span_bytes,
        "view_double_counted_bytes": max(0, span_bytes - unique_bytes),
        "unique_storage_bytes_by_first_category": first_category_bytes,
        "storage_count_by_first_category": first_category_storages,
        "alias_storage_count": alias_storages,
        "cross_category_alias_bytes": cross_alias_bytes,
        "cross_device_alias_storages": cross_device_storages,
        "largest_storages": largest,
        "zero_or_meta_storage_entries": len(zero_ptr),
        "reconciliation_note": "unique_storage_bytes counts each (storage_ptr, nbytes) "
                                "once, independent of device-string aliases; only "
                                "accelerator_unique_storage_bytes is compared with torch "
                                "accelerator allocated bytes; per-view tensor_nbytes must "
                                "not be summed as disjoint memory",
    }


def collect_layout_snapshot(engine, *, rank: int, phase: str, run_id: str,
                            case_id: str, out_dir, identity: dict | None = None,
                            label: str | None = None,
                            max_modules: int = 20000,
                            max_tensors: int = 30000) -> dict:
    """Collect a quiescent, bounded metadata snapshot.  Never reads values."""
    started = time.time_ns()
    driver = getattr(engine, "driver_worker", None)
    model_runner = getattr(driver, "model_runner", None) if driver is not None else None
    model = getattr(model_runner, "model", None) if model_runner is not None else None
    entries = []
    notes = []
    visited_modules = 0
    truncated = False
    # Reserve part of the bounded budget for KV/state/communication pools so a
    # large parameter tree cannot crowd them out.
    known_budget = min(max_tensors, max(4096, max_tensors // 4))
    known, known_notes = _collect_known_tensors(engine, max_tensors=known_budget)
    entries.extend(known)
    notes.extend(known_notes)
    if model is not None:
        model_entries, visited_modules, truncated = _collect_model_tensors(
            model, max_modules=max_modules,
            max_tensors=max(0, max_tensors - len(entries)))
        entries.extend(model_entries)
        if len(model_entries) >= max(0, max_tensors - known_budget):
            truncated = True
    else:
        notes.append("engine.driver_worker.model_runner.model missing")
    if len(entries) > max_tensors:
        entries = entries[:max_tensors]
        truncated = True
    deduped = _dedupe_storages(entries)
    device = _device_memory()
    cpu = _cpu_memory()
    cache_engine = getattr(driver, "cache_engine", None) if driver is not None else None
    cache_layout = {
        "kvcache_layout": getattr(cache_engine, "kvcache_layout", None),
        "state_size_to_disk": getattr(cache_engine, "state_size_to_disk", None),
        "num_gpu_blocks": getattr(cache_engine, "num_gpu_blocks", None),
        "num_layers": getattr(cache_engine, "num_layers", None),
        "max_num_seqs": getattr(cache_engine, "max_num_seqs", None),
    }
    cache_config = getattr(cache_engine, "cache_config", None) if cache_engine else None
    for name in ("block_size", "num_gpu_blocks", "num_state_seqs",
                 "bytes_per_block", "bytes_of_state_per_seqs"):
        if cache_config is not None:
            cache_layout["cache_config.%s" % name] = getattr(cache_config, name, None)
    if cache_engine is not None and getattr(cache_engine, "kvcache_layout", None):
        try:
            cache_layout["layout_total_bytes"] = sum(
                int(item[2]) for item in cache_engine.kvcache_layout)
            cache_layout["layout_entry_count"] = len(cache_engine.kvcache_layout)
        except BaseException:  # noqa: BLE001
            pass
    torch_allocated = device.get("torch_allocated_bytes")
    unique_bytes = deduped["unique_storage_bytes"]
    accelerator_bytes = deduped["accelerator_unique_storage_bytes"]
    device_used = device.get("used_bytes")
    torch_reserved = device.get("torch_reserved_bytes")
    payload = {
        "schema_version": "e33-layout-snapshot/1.0",
        "synthetic": False,
        "framework": FRAMEWORK,
        "rank": rank,
        "phase": phase,
        "label": label or phase,
        "run_id": run_id,
        "case_id": case_id,
        "pid": os.getpid(),
        "host": platform.node(),
        "clock": {
            "started_time_time_ns": started,
            "finished_time_time_ns": time.time_ns(),
            "started_perf_counter_ns": time.perf_counter_ns(),
            "device_clock_mapping": None,
            "clock_mapping_error_bound_ns": None,
            "note": "host CPU clocks only; no reconstructed device timeline",
        },
        "identity": identity,
        "quiescent": True,
        "tensor_values_read": False,
        "tensor_metadata_bounded": {
            "max_modules": max_modules,
            "max_tensors": max_tensors,
            "visited_modules": visited_modules,
            "retained_tensor_entries": len(entries),
            "truncated": bool(truncated or len(entries) >= max_tensors),
        },
        "tensors": entries,
        "storage_ledger": deduped,
        "cache_layout": cache_layout,
        "device_memory": device,
        "cpu_memory": cpu,
        "reconciliation": {
            "device_used_bytes": device_used,
            "torch_allocated_bytes": torch_allocated,
            "torch_reserved_bytes": torch_reserved,
            "reachable_unique_storage_bytes": accelerator_bytes,
            "reachable_unique_storage_bytes_all_devices": unique_bytes,
            "cpu_unique_storage_bytes": deduped["cpu_unique_storage_bytes"],
            "reachable_minus_torch_allocated_bytes": (
                None if torch_allocated is None else accelerator_bytes - torch_allocated),
            "unreachable_torch_allocated_estimate_bytes": (
                None if torch_allocated is None
                else max(0, torch_allocated - accelerator_bytes)),
            "native_or_driver_bytes_unknown": (
                None if device_used is None or torch_reserved is None
                else max(0, device_used - torch_reserved)),
            "unknown_is_not_zero": True,
            "physical_hbm_bank_layout": "unknown",
            "note": "device total/free/used come from the installed memory API; "
                    "allocator arenas, native/driver allocations and other processes "
                    "cannot be resolved from these counters. No HBM bank mapping is claimed.",
        },
        "notes": notes,
        "errors": device.get("errors", []) + cpu.get("errors", []),
        "physical_hbm_bank_layout": "unknown",
    }
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    stamp = utc_stamp()
    path = target / ("layout-%s-%s.json" % (safe_name(label or phase, "label"), stamp))
    atomic_write_json(path, payload)
    return {
        "path": str(path),
        "synthetic": False,
        "tensor_entries": len(entries),
        "truncated": payload["tensor_metadata_bounded"]["truncated"],
        "unique_storage_bytes": accelerator_bytes,
        "unique_storage_bytes_all_devices": unique_bytes,
        "cpu_unique_storage_bytes": deduped["cpu_unique_storage_bytes"],
        "device_used_bytes": device_used,
        "device_free_bytes": device.get("free_bytes"),
        "device_total_bytes": device.get("total_bytes"),
        "torch_allocated_bytes": torch_allocated,
        "torch_reserved_bytes": torch_reserved,
        "torch_peak_allocated_bytes": device.get("torch_peak_allocated_bytes"),
        "native_or_driver_bytes_unknown": payload["reconciliation"][
            "native_or_driver_bytes_unknown"],
        "cpu_rss_bytes": cpu.get("rss_bytes"),
        "affinity_count": cpu.get("affinity_count"),
        "errors": payload["errors"],
    }


# ---------------------------------------------------------------------------
# local self-test (no torch, no NPU)
# ---------------------------------------------------------------------------
def _selftest() -> int:
    import tempfile

    checks = []

    def check(name, condition, detail=""):
        checks.append({"name": name, "pass": bool(condition), "detail": str(detail)})

    root = Path(tempfile.mkdtemp(prefix="e33-telemetry-selftest-"))
    tel = E33Telemetry(rank=0, out_dir=root / "rank00", run_id="selftest",
                       case_id="local", phase="selftest", source="selftest")
    with tel.begin("outer", request_id="req-1", iteration_id=7,
                   extra={"label": "outer"}) as outer:
        with tel.begin("inner", request_id="req-1", iteration_id=7):
            time.sleep(0.001)
    tel.emit("instant", request_id="req-1", extra={"value": 1})
    stats = tel.close()
    rows = [json.loads(line) for line in Path(stats["path"]).read_text().splitlines()]
    check("event-rows", len(rows) == 3, "rows=%d" % len(rows))
    required = {
        "framework", "rank", "pid", "native_tid", "time_time_ns_start",
        "time_time_ns_end", "perf_counter_ns_start", "perf_counter_ns_end",
        "thread_time_ns", "run_id", "event_name", "source", "start", "end",
        "request_id", "iteration_id", "exception", "device_time",
    }
    check("required-keys", all(required <= set(row) for row in rows),
          "missing=%r" % sorted(required - set(rows[0])))
    outer_row = [row for row in rows if row["event_name"] == "outer"][0]
    inner_row = [row for row in rows if row["event_name"] == "inner"][0]
    check("nested-spans", outer_row["perf_counter_ns_start"]
          <= inner_row["perf_counter_ns_start"]
          <= inner_row["perf_counter_ns_end"]
          <= outer_row["perf_counter_ns_end"], "nested host spans")
    check("host-not-device", all(row["device_time"] is False for row in rows),
          "device_time flags")

    class StubTimer:
        def __init__(self):
            self.calls = []

        @contextlib.contextmanager
        def record(self, name, enable=True, record_name=None):
            self.calls.append(("enter", name, bool(enable), record_name))
            try:
                yield
            finally:
                self.calls.append(("exit", name, bool(enable), record_name))

    timer = StubTimer()
    tel2 = E33Telemetry(rank=1, out_dir=root / "rank01", run_id="selftest",
                        case_id="local", phase="timer", source="selftest")
    installed = install_cpu_timer_probe(tel2, timer=timer)
    check("timer-installed", installed is True, "install")
    with timer.record("schedule:cpp"):
        pass
    with timer.record("disabled", enable=False, record_name="alias"):
        pass
    with timer.record("schedule:post"):
        raise_key = False
        try:
            with timer.record("body"):
                raise KeyError("original-body-error")
        except KeyError as exc:
            raise_key = str(exc) == "'original-body-error'"
    tel2.close()
    timer_rows = [json.loads(line) for line in
                  Path(tel2.event_path).read_text().splitlines()]
    check("timer-rows", len(timer_rows) == 4, "rows=%d" % len(timer_rows))
    check("timer-names", {row["event_name"] for row in timer_rows}
          == {"schedule:cpp", "disabled", "schedule:post", "body"},
          sorted({row["event_name"] for row in timer_rows}))
    check("timer-disable-semantics",
          ("enter", "disabled", False, "alias") in timer.calls
          and ("exit", "disabled", False, "alias") in timer.calls, "calls")
    check("timer-original-exception", raise_key is True,
          "KeyError propagated from the original body")
    body_row = [row for row in timer_rows if row["event_name"] == "body"][0]
    check("timer-exception-recorded", (body_row.get("exception") or {}).get("type") == "KeyError",
          body_row.get("exception"))

    class ExplodingTelemetry:
        def begin(self, *args, **kwargs):
            raise RuntimeError("telemetry begin failed")

        def emit(self, *args, **kwargs):
            raise RuntimeError("telemetry emit failed")

    timer2 = StubTimer()
    install_cpu_timer_probe(ExplodingTelemetry(), timer=timer2)
    try:
        with timer2.record("must-still-run"):
            raise ValueError("original-value")
    except ValueError as exc:
        check("telemetry-begin-failure-keeps-original", str(exc) == "original-value",
              repr(exc))
    else:
        check("telemetry-begin-failure-keeps-original", False, "no exception")

    class EndExplodingTelemetry(E33Telemetry):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.fail_end = True

        def emit(self, *args, **kwargs):
            if self.fail_end:
                raise RuntimeError("telemetry emit failed")
            return super().emit(*args, **kwargs)

    timer3 = StubTimer()
    tel3 = EndExplodingTelemetry(rank=2, out_dir=root / "rank02", run_id="selftest",
                                 case_id="local", phase="endfail", source="selftest")
    install_cpu_timer_probe(tel3, timer=timer3)
    try:
        with timer3.record("end-fails"):
            raise LookupError("original-end")
    except LookupError as exc:
        check("telemetry-end-failure-keeps-original", str(exc) == "original-end", repr(exc))
    else:
        check("telemetry-end-failure-keeps-original", False, "no exception")

    entries = [
        {"category": "weights", "name": "w0", "device": "npu:0",
         "storage_ptr": 100, "storage_ptr_hex": "0x64", "storage_nbytes": 4096,
         "tensor_nbytes": 2048},
        {"category": "weights", "name": "w0.view", "device": "npu:0",
         "storage_ptr": 100, "storage_ptr_hex": "0x64", "storage_nbytes": 4096,
         "tensor_nbytes": 1024},
        {"category": "kv_cache", "name": "kv", "device": "npu:0",
         "storage_ptr": 200, "storage_ptr_hex": "0xc8", "storage_nbytes": 8192,
         "tensor_nbytes": 8192},
        {"category": "state_copy", "name": "cpu_copy", "device": "cpu",
         "storage_ptr": 300, "storage_ptr_hex": "0x12c", "storage_nbytes": 512,
         "tensor_nbytes": 512},
    ]
    ledger = _dedupe_storages(entries)
    check("storage-unique", ledger["unique_storage_bytes"] == 12800,
          ledger["unique_storage_bytes"])
    check("storage-accelerator-cpu-split",
          ledger["accelerator_unique_storage_bytes"] == 12288
          and ledger["cpu_unique_storage_bytes"] == 512,
          {"accelerator": ledger["accelerator_unique_storage_bytes"],
           "cpu": ledger["cpu_unique_storage_bytes"]})
    check("storage-alias-not-added", ledger["tensor_span_bytes_summed_per_view"] == 11776,
          ledger["tensor_span_bytes_summed_per_view"])
    check("storage-alias-count", ledger["alias_storage_count"] == 1,
          ledger["alias_storage_count"])

    export_dir = root / "fake-export"
    (export_dir / "ASCEND_PROFILER_OUTPUT").mkdir(parents=True)
    (export_dir / "ASCEND_PROFILER_OUTPUT" / "kernel_details.csv").write_text(
        "Device ID,Name\n0,matmul\n")
    (export_dir / "ASCEND_PROFILER_OUTPUT" / "trace_view.json").write_text("[]\n")
    files = discover_profile_files(export_dir)
    check("trace-discovered", files["trace_present"] and files["csv_present"],
          "%d files" % files["file_count"])
    empty = discover_profile_files(root / "empty-export")
    check("missing-trace-not-ok", empty["trace_present"] is False, empty["trace_present"])

    ok = all(item["pass"] for item in checks)
    print(json.dumps({"ok": ok, "selftest_root": str(root), "checks": checks},
                     indent=2, sort_keys=True))
    return 0 if ok else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true",
                        help="run the local, hardware-free API checks")
    args = parser.parse_args(argv)
    if args.selftest:
        return _selftest()
    parser.error("choose --selftest")
    return 2


if __name__ == "__main__":
    sys.exit(main())
