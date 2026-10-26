"""B-side (SGLang) real-forward window observer.

The native SchedulerProfilerManager is used unchanged for start; the leader
stop mirrors its exact legacy NPU stop sequence without the TP-group barrier,
because only TP rank 0 owns a profiler in this diagnostic.  The barrier is a
trace-export synchronisation, not model computation, and is disclosed in the
run record.  No tensor is read and no scheduler/model function body changes.
"""
from __future__ import annotations

import contextlib
import gc
import importlib
import inspect
import os
import pathlib

import r234_common as C

TARGET_MODULE = "sglang.srt.managers.scheduler"
TARGET_CLASS = "Scheduler"
TARGET_METHOD = "run_batch"
PROFILER_MODULE = "sglang.srt.managers.scheduler_components.profiler_manager"

DEFAULT_PINS = {
    "sglang/srt/managers/scheduler.py":
        "bf67744cd55fb77b6b210774c56ed6c7ce4da6e1228e3208d3d61718a1044978",
    "sglang/srt/managers/scheduler_components/profiler_manager.py":
        "a962ab8283b468d6c3f10e53527613ea79569490cfdef340f3c22ffbee6bdcc3",
}

_STATE = {"installed": False, "reason": None, "identity": None}
_SELF_SHA = None


def _self_sha():
    global _SELF_SHA
    if _SELF_SHA is None:
        _SELF_SHA = C.sha256_file(__file__)
    return _SELF_SHA


def _env_pins():
    raw = os.environ.get("R234_SGLANG_SOURCE_PINS_JSON")
    return dict(DEFAULT_PINS) if not raw else dict(__import__("json").loads(raw))


def _install_path():
    return pathlib.Path(os.environ["R234_PROFILE_ROOT"]) / (
        "observer-install-" + str(os.getpid()) + ".json")


def _write_install(obj):
    try:
        C.atomic_json(_install_path(), obj)
    except BaseException:
        pass


def _binding(module):
    path = inspect.getfile(module)
    return {"module": module.__name__, "file": path, "sha256": C.sha256_file(path)}


def install(module=None):
    if _STATE["installed"]:
        return _STATE
    try:
        if module is None:
            module = importlib.import_module(TARGET_MODULE)
        pins = _env_pins()
        observed = {}
        for rel, expected in pins.items():
            if rel.startswith("sglang/srt/managers/scheduler.py"):
                path = inspect.getfile(module)
            else:
                path = inspect.getfile(importlib.import_module(PROFILER_MODULE))
            actual = C.sha256_file(path)
            if actual != expected:
                raise RuntimeError("native source SHA mismatch " + rel + " " + actual)
            observed[rel] = actual
        scheduler_cls = getattr(module, TARGET_CLASS)
        original = getattr(scheduler_cls, TARGET_METHOD)
        if getattr(scheduler_cls, "__r234_window_observer__", False):
            _STATE.update(installed=True, reason="already_installed")
            return _STATE
        envs = importlib.import_module("sglang.srt.environ").envs
        if envs.SGLANG_PROFILE_V2.get():
            raise RuntimeError("SGLANG_PROFILE_V2 must be false for the legacy native window path")
        profiler_module = importlib.import_module(PROFILER_MODULE)
        manager_cls = profiler_module.SchedulerProfilerManager

        def run_batch(self, batch, *args, **kwargs):
            return _run(self, original, batch, args, kwargs, manager_cls)

        run_batch.__name__ = original.__name__
        run_batch.__doc__ = original.__doc__
        run_batch.__wrapped__ = original
        setattr(scheduler_cls, TARGET_METHOD, run_batch)
        scheduler_cls.__r234_window_observer__ = True
        identity = {"target": _binding(module),
                    "profiler_target": _binding(profiler_module),
                    "pins": observed, "method": TARGET_METHOD,
                    "observer_file": __file__, "observer_sha256": _self_sha()}
        _STATE.update(installed=True, reason=None, identity=identity)
        _write_install({"schema": C.SCHEMA_STATUS, "framework": "sglang",
                        "installed": True, **identity})
        return _STATE
    except BaseException as exc:
        _STATE.update(installed=False, reason=repr(exc))
        _write_install({"schema": C.SCHEMA_STATUS, "framework": "sglang",
                        "installed": False, "reason": repr(exc)})
        return _STATE


def _parallel():
    return importlib.import_module("sglang.srt.runtime_context").get_parallel()


def _leader_identity():
    """Return the sampled TP0 identity or None.

    The R201 deployment is TP16 DP1, so the sampled leader is TP rank 0.  The
    namespace still records dp_rank + global rank; this keeps collector/status
    names unambiguous if a future run samples another rank.
    """
    try:
        parallel = _parallel()
        tp_rank = int(parallel.tp_rank)
        dp_rank = int(parallel.dp_rank or 0)
        world_rank = getattr(parallel, "world_rank", None)
        rank = int(world_rank) if world_rank is not None else tp_rank
        tp_size = int(parallel.tp_size)
    except BaseException:
        return None
    if tp_rank != 0 or dp_rank != 0:
        return None
    return {"dp_rank": dp_rank, "rank": rank, "tp_rank": tp_rank,
            "tp_size": tp_size, "namespace": C.rank_namespace(dp_rank, rank)}


def _forward_mode(batch):
    mode = getattr(batch, "forward_mode", None)
    if mode is None:
        return "other"
    try:
        if mode.is_prefill():
            return "prefill"
        if mode.is_decode():
            return "decode"
    except BaseException:
        return "other"
    return "other"


def _host_metadata(batch):
    reqs = list(getattr(batch, "reqs", []) or [])
    if not reqs:
        return None
    mode = _forward_mode(batch)
    prefix_lens = getattr(batch, "prefix_lens", None)
    extend_lens = getattr(batch, "extend_lens", None)
    if prefix_lens is None:
        prefix_lens = [len(getattr(r, "prefix_indices", []) or []) for r in reqs]
    if extend_lens is None:
        out = []
        for r in reqs:
            rng = getattr(r, "extend_range", None)
            out.append(int(getattr(rng, "length", 1 if mode == "decode" else 0)))
        extend_lens = out
    rows = []
    for r, prefix, extend in zip(reqs, prefix_lens, extend_lens):
        prefix, extend = int(prefix), int(extend)
        rows.append({"rid": str(getattr(r, "rid", "")),
                     "context_start": prefix, "context_end": prefix + extend,
                     "query_tokens": extend})
    query = getattr(batch, "extend_num_tokens", None)
    if type(query) is not int:
        query = sum(row["query_tokens"] for row in rows)
    return {"mode": mode, "active_requests": len(rows),
            "batch_requests": len(reqs), "query_tokens": int(query),
            "requests": rows}


def _meta_record(worker, meta, index, identity):
    parallel = _parallel()
    return {
        "schema": C.SCHEMA_META, "framework": "sglang",
        "node": int(os.environ.get("R234_PROFILE_NODE", "0")),
        "rank": identity["rank"],
        "local_rank": int(getattr(parallel, "tp_rank", -1)),
        "dp_rank": identity["dp_rank"],
        "tp_rank": identity["tp_rank"],
        "tp_size": identity["tp_size"],
        "rank_namespace": identity["namespace"],
        "forward_index": index, "engine_forward_ct": int(getattr(worker, "forward_ct", -1)) + 1,
        "capture": None, "capture_slot": None,
        "observer_file": __file__, "observer_sha256": _self_sha(),
        **meta,
    }


def _leader_start(worker, manager_cls, batch, window_name, identity):
    manager = worker.profiler_manager
    root = pathlib.Path(os.environ["R234_PROFILE_ROOT"]) / identity["namespace"] / window_name
    root.mkdir(parents=True, exist_ok=True)
    minimum = C.env_int("R234_PROFILE_MIN_FREE_BYTES", 8 * 1024 ** 3)
    if C.free_bytes(root) < minimum:
        raise RuntimeError("profile filesystem below free-byte floor")
    if manager.profile_in_progress:
        raise RuntimeError("native profiler already in progress")
    manager.torch_profiler_output_dir = root
    manager.profiler_activities = ["CPU", "GPU"]
    manager.torch_profiler_with_stack = False
    manager.torch_profiler_record_shapes = False
    manager.profile_id = "r234B-" + identity["namespace"] + "-" + window_name
    manager.profile_prefix = ""
    manager.merge_profiles = False
    manager.detailed_annotations = False
    manager.profile_by_stage = False
    manager.profiler_start_forward_ct = None
    manager.profiler_target_forward_ct = None
    C.synchronize_profile_boundary()
    result = manager._start_profile(stage=getattr(batch, "forward_mode", None))
    if result is not None and not getattr(result, "success", False):
        raise RuntimeError("native SGLang profiler start failed: "
                           + repr(getattr(result, "message", result)))
    if manager.torch_profiler is None or not manager.profile_in_progress:
        raise RuntimeError("native SGLang profiler did not become active")
    manager.torch_profiler = C.WindowedProfiler(
        manager.torch_profiler, identity["namespace"], window_name)
    return manager, root


def _leader_stop(manager):
    """Exact legacy NPU stop sequence minus the TP-group export barrier."""
    manager.torch_profiler_output_dir.mkdir(parents=True, exist_ok=True)
    manager.torch_profiler.stop()
    manager.torch_profiler = None
    gc.collect()
    manager.profile_in_progress = False
    manager.profiler_start_forward_ct = None
    manager.profiler_target_forward_ct = None


def _run(worker, original, batch, args, kwargs, manager_cls):
    arm = os.environ.get("R234_PROFILE_ARM_FILE")
    if arm and not pathlib.Path(arm).exists():
        return original(worker, batch, *args, **kwargs)
    identity = _leader_identity()
    if identity is None:
        return original(worker, batch, *args, **kwargs)
    obs = getattr(worker, "_r234_observer", None)
    if obs is None:
        try:
            obs = _Observer(worker, identity)
            worker._r234_observer = obs
        except BaseException:
            worker._r234_observer = False
            return original(worker, batch, *args, **kwargs)
    if obs is False:
        return original(worker, batch, *args, **kwargs)
    return obs.step(original, batch, args, kwargs, manager_cls)


class _Observer:
    def __init__(self, worker, identity):
        self.worker = worker
        self.identity = dict(identity)
        self.root = pathlib.Path(os.environ["R234_PROFILE_ROOT"])
        self.local = int(identity["tp_rank"])
        self.dp_rank = int(identity["dp_rank"])
        self.rank = int(identity["rank"])
        self.namespace = identity["namespace"]
        self.meta_path = self.root / C.metadata_filename(self.namespace)
        self.status_path = self.root / C.status_filename(self.namespace)
        self.windows = C.parse_windows(os.environ.get("R234_PROFILE_WINDOWS"))
        self.max_forwards = C.env_int("R234_PROFILE_MAX_FORWARDS", 6)
        self.max_meta = C.env_int("R234_PROFILE_MAX_METADATA_FORWARDS", 64)
        self.policy = C.WindowPolicy(self.windows, self.max_forwards,
                                          C.env_int("R234_PROFILE_EXPECT_ACTIVE", 1))
        self.index = 0
        self.manager = None
        self.profiler_dir = None
        self.active_window = None
        self.expected = 0
        self.captured = []
        self.error = None
        self._started = 0
        self._stopped = 0
        self.install_identity = _STATE.get("identity")
        self._write_status()

    def _write_status(self, **extra):
        status = {
            "schema": C.SCHEMA_STATUS, "framework": "sglang",
            "node": int(os.environ.get("R234_PROFILE_NODE", "0")),
            "rank": self.rank,
            "dp_rank": self.dp_rank,
            "local_rank": self.local,
            "rank_namespace": self.namespace,
            "leader": True,
            "installed": bool(_STATE.get("installed")),
            "install_identity": self.install_identity,
            "chained_sitecustomize": os.environ.get("R234_CHAINED_SITECUSTOMIZE") or None,
            "forward_index": self.index,
            "profile_sessions_started": self._started,
            "profile_sessions_stopped": self._stopped,
            "stop_barrier_omitted": True,
            "stop_barrier_scope": "trace export synchronisation only; no compute/numerics",
            "policy": self.policy.status(), "error": self.error,
        }
        status.update(extra)
        try:
            C.atomic_json(self.status_path, status)
        except BaseException:
            pass

    def _arm(self, batch, manager_cls, meta):
        if getattr(self, "stopping_failed", False):
            self.policy.abort("profiler_stop_failed")
            self._write_status()
            return
        try:
            candidate = self.policy.begin(meta)
            if candidate is None:
                return
            window, expected = candidate
            self.manager, self.profiler_dir = _leader_start(
                self.worker, manager_cls, batch, window["name"], self.identity)
            self.active_window, self.expected = window, expected
            self.captured = []
            self._started += 1
        except BaseException as exc:
            self.error = "profiler arm: " + repr(exc)
            self.manager = None
            self.active_window = None
            self.expected = 0
            self.captured = []
            self.policy.abort("profiler_arm_failed")
        self._write_status()

    def _finish(self):
        size = 0
        try:
            _leader_stop(self.manager)
            self._stopped += 1
        except BaseException as exc:
            self.error = "profiler stop: " + repr(exc)
            self.stopping_failed = True
        try:
            size = C.dir_bytes(self.profiler_dir) if self.profiler_dir else 0
        except BaseException:
            pass
        result = self.policy.finish()
        result["profile_dir"] = str(self.profiler_dir)
        result["profile_dir_bytes"] = size
        limit = C.env_int("R234_PROFILE_MAX_WINDOW_BYTES", 4 * 1024 ** 3)
        if size > limit:
            result["valid"] = False
            result["reason"] = "profile_window_bytes_exceeded"
            self.policy.captured_forwards = self.policy.max_forwards
        self.active_window = None
        self.manager = None
        self.expected = 0
        self.captured = []
        self._write_status()

    def step(self, original, batch, args, kwargs, manager_cls):
        self.index += 1
        meta = None
        record = None
        try:
            meta = _host_metadata(batch)
            if meta is not None:
                self.policy.note_batch(meta)
            record = _meta_record(self.worker, meta, self.index, self.identity) if meta else None
        except BaseException as exc:
            self.error = "host_metadata: " + repr(exc)
        if self.expected and len(self.captured) >= self.expected:
            try:
                self._finish()
            except BaseException as exc:
                self.error = "defensive_finish: " + repr(exc)
        if self.manager is None and meta is not None and meta.get("mode") == "prefill":
            self._arm(batch, manager_cls, meta)
        capturing = self.manager is not None and self.active_window is not None
        if capturing and record is None:
            self.error = "missing_scheduler_metadata_during_capture"
            try:
                _leader_stop(self.manager)
                self._stopped += 1
            except BaseException:
                pass
            self.policy.abort("missing_scheduler_metadata")
            self.manager = None
            self.expected = 0
            self.captured = []
            self._write_status()
            capturing = False
        if capturing and record is not None and not self.policy.compatible(meta):
            self.error = "capture_shape_or_request_changed"
            try:
                _leader_stop(self.manager)
                self._stopped += 1
            except BaseException:
                pass
            self.policy.abort("capture_shape_or_request_changed")
            self.manager = None
            self.expected = 0
            self.captured = []
            self._write_status()
            capturing = False
        if capturing and record is not None:
            record["capture"] = self.active_window["name"]
            record["capture_slot"] = len(self.captured) + 1
        started = C.now_ns()
        context = contextlib.nullcontext()
        if capturing:
            try:
                context = importlib.import_module("torch").profiler.record_function(
                    "r234_forward_iter_" + self.namespace + "_" + str(self.index))
            except BaseException:
                context = contextlib.nullcontext()
        try:
            with context:
                result = original(self.worker, batch, *args, **kwargs)
        except BaseException as native_exc:
            self.error = "native_forward_exception: " + repr(native_exc)
            try:
                if self.manager is not None:
                    _leader_stop(self.manager)
                    self._stopped += 1
                self.policy.abort("native_forward_exception")
            except BaseException:
                pass
            self._write_status()
            raise
        ended = C.now_ns()
        if record is not None:
            record["wall_start_monotonic_ns"] = started
            record["wall_end_monotonic_ns"] = ended
            record["wall_ns"] = ended - started
            if capturing or self.index <= self.max_meta:
                try:
                    C.append_jsonl(self.meta_path, record)
                except BaseException as exc:
                    self.error = "metadata append: " + repr(exc)
        if capturing:
            try:
                self.captured.append(record)
                self.policy.record(record)
                if len(self.captured) >= self.expected:
                    self._finish()
            except BaseException as exc:
                self.error = "capture_finish: " + repr(exc)
                try:
                    if self.manager is not None:
                        _leader_stop(self.manager)
                        self._stopped += 1
                except BaseException:
                    pass
                self._write_status()
        self._write_status()
        return result
