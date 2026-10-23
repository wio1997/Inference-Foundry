#!/usr/bin/env python3
"""DeepInfer native-DP8 independent-request adapter, E33 full-cohort diagnostics.

This is the E27 adapter with one narrow change: profiling is no longer armed
by a scheduler step window.  A loopback-only control plane starts one
continuous torch_npu recording per rank when the whole DP8 group is globally
idle, and stops/exports it only after the client cohort has finished and every
rank queue is drained.  Scheduling, prefix retention/fork, sampling and HTTP
response fields are unchanged.

Design
------
Eight long-lived rank processes each own one DeepInfer LLM/LLMEngine.  A
request is added to exactly one rank (`LLMEngine.add_request`); the other
ranks keep calling the native `has_unfinished_requests()` / `step()` /
`flush_inflight_reqs()` collectives and execute the scheduler's placeholder
or EP work.  Requests are never mirrored.

The per-rank loop is the original `LLM._run_engine` loop turned into a
service loop:

    drain local inbox -> has_unfinished_requests() (global all_reduce)
      -> step() while unfinished, else flush_inflight_reqs()
      -> clone() every RequestOutput before the next step releases it

A wake broadcast keeps the eight ranks entering the collective loop
together after a globally idle period.  Idle ranks never exit early.

E33 control plane
-----------------
The HTTP parent validates loopback, global idle, drained queues and the task
root, broadcasts one ``e33_start``/``e33_stop``/``e33_snapshot`` command to
all eight inboxes, releases the cluster lock, then waits for eight rank acks.
Worker command handling never enters a new collective.  The profiler has no
active-step schedule and no ``step()`` calls; it records until the explicit
stop/export at global idle.  There is no automatic 4-step window.

Modes
-----
  --serve           HTTP server (OpenAI-completions SSE / native endpoints)
  --offline-diag    one independent job per rank from a JSON file, no HTTP
  --mock            CPU-only routing/protocol self-test (no deepinfer import)
  --e33-mock-check  local end-to-end mock of the E33 control protocol

Protocol notes
--------------
* `/v1/completions` accepts `prompt_token_ids` (exact IDs) and returns
  `token_ids` plus usage.  Unsupported sampling fields are rejected, never
  silently dropped.
* `/readyz` is 200 once all ranks are ready and carries honest
  inflight/running/waiting counters for drain polling.
* Every event carries `assigned_dp_rank` and `output_origin_step_id`.
* `/e33/profile/start|stop`, `/e33/snapshot` and `/e33/status` are accepted
  only from a loopback peer.  start/stop/snapshot bodies carry
  `run_id`, `case_id`, `phase`, `out_dir`; out_dir must be below the task root.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import ipaddress
import json
import multiprocessing as mp
import os
import queue
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DEVICES = 8
E33_TASK_ROOT = os.environ.get("E33_TASK_ROOT", "/work/performance-20261021")
E33_START_TIMEOUT_S = float(os.environ.get("E33_START_TIMEOUT_S", "900"))
E33_STOP_TIMEOUT_S = float(os.environ.get("E33_STOP_TIMEOUT_S", "3600"))
E33_SNAPSHOT_TIMEOUT_S = float(os.environ.get("E33_SNAPSHOT_TIMEOUT_S", "600"))
_E33_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))
from e33_di_telemetry import (  # noqa: E402
    CohortProfiler,
    E33Telemetry,
    atomic_write_json,
    collect_layout_snapshot,
    install_cpu_timer_probe,
    set_cpu_timer_telemetry,
)

# E33 contains no step-window profiler controller.  The legacy --profile-*
# arguments are still parsed for launcher compatibility, but --profile-dir is
# rejected and full recording is controlled only by the /e33 endpoints.


# Startup compile-only DeepGEMM readiness, populated by --gemm-warm-manifest.
# Defaults are permissive so mock/offline modes and older launches are not
# blocked; a serve launch that enables the warmer replaces this dict.
_STARTUP_CACHE: dict = {"enabled": False, "coverage_ok": True}

# Normal-loop startup warm, populated by --startup-warm.  The service must not
# publish HTTP readiness until this reports completed so the first public
# request does not pay the native module/graph/placeholder initialization cost.
_STARTUP_WARM: dict = {"enabled": False, "completed": False,
                       "reason": "not_run"}

# Compile-only sparse-MQA metadata readiness, populated by
# --sparse-mqa-warm-manifest.  It covers the small-Q metadata specializations
# that the dense GEMM manifest does not include.
_STARTUP_SPARSE: dict = {"enabled": False, "coverage_ok": True,
                         "reason": "not_run"}


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass
class ProfilePlan:
    """Phase-gated worker profiling plan.

    Windows are armed by the parent when a request whose id is in
    ``request_ids`` is dispatched, then started inside the rank worker from
    scheduler-observed state (``decode_step`` plus ``post_schedule`` phase),
    never from HTTP arrival.  Each window is exported by its own
    torch_npu.profiler session in the worker process.
    """
    out_dir: str
    run_id: str
    case_id: str = ""
    baseline_run_id: str = ""
    request_ids: tuple[str, ...] = ()
    arm_after_dispatched: int = 0
    mode: str = "prompt"          # prompt | decode | both
    steps: int = 8
    prompt_windows: int = 4
    prompt_interval: int = 100
    prompt_trigger_moduli: tuple[int, ...] = (100,)
    prompt_trigger_k_fracs: tuple[float, ...] = ()
    prompt_trigger_requires_local: bool = False
    arm_lead: int = 2
    initial_prompt_window: bool = False
    immediate_prompt_window: bool = False
    decode_trigger: int = 4
    decode_windows: int = 1
    profiler_level: str = "Level1"
    record_shapes: bool = False
    with_stack: bool = False

    def __post_init__(self) -> None:
        if self.mode not in ("prompt", "decode", "both"):
            raise ValueError("profile mode must be prompt, decode or both")
        if self.steps < 1:
            raise ValueError("profile steps must be >= 1")
        if self.prompt_interval < 1 or self.arm_lead < 0:
            raise ValueError("bad prompt interval/lead")
        if not self.prompt_trigger_moduli:
            raise ValueError("prompt_trigger_moduli must not be empty")
        if any(int(m) < 1 for m in self.prompt_trigger_moduli):
            raise ValueError("prompt_trigger_moduli must be positive")
        if any(not 0.0 < float(f) < 1.0 for f in self.prompt_trigger_k_fracs):
            raise ValueError("prompt_trigger_k_fracs must be in (0, 1)")
        if not self.out_dir:
            raise ValueError("profile out_dir is required")
        if not self.run_id:
            raise ValueError("profile run_id is required")

    def public(self) -> dict:
        d = self.__dict__.copy()
        d["request_ids"] = list(self.request_ids)
        return d


@dataclass
class ClusterConfig:
    model: str = "/data/wio/DeepSeek-V4.1-Flash-1017"
    ep_size: int = 8
    cp_size: int = 1
    dp_size: int = 8
    max_model_len: int = 4096
    max_num_batched_tokens: int = 256
    max_num_seqs: int = 1
    gpu_memory_utilization: float = 0.85
    sampler_force_topk: int = 4096
    zos: bool = False
    nextn: int = 0
    mtp_varlen: bool = False
    enforce_eager: bool = True
    prefill_attempt_interval: int | None = None
    prefill_any_attempt_interval: int | None = None
    jit_cache_dir: str | None = None
    tilelang_cache_dir: str | None = None
    master_port: int = 23333
    trace_steps: bool = False
    start_timeout_s: float = 1800.0
    request_timeout_s: float = 600.0
    label: str = "deepinfer-native-dp8"
    profile: ProfilePlan | None = None
    diag_requests_path: str | None = None
    e33_task_root: str = E33_TASK_ROOT

    def public(self) -> dict:
        d = self.__dict__.copy()
        if self.profile is not None:
            d["profile"] = self.profile.public()
        return d


@dataclass
class Job:
    request_id: str
    prompt_token_ids: list[int]
    max_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    ignore_eos: bool = False
    seed: int | None = None
    rank: int | None = None
    arrival_ts: float = field(default_factory=time.time)
    meta: dict = field(default_factory=dict)


class E33ControlBusy(RuntimeError):
    """A request arrived while the parent holds the E33 control gate."""


class E33ControlError(RuntimeError):
    """Control-plane rejection; carries an HTTP status and JSON detail."""

    def __init__(self, message: str, *, status: int = 409, detail=None) -> None:
        super().__init__(message)
        self.status = int(status)
        self.detail = detail


def _is_loopback_addr(addr) -> bool:
    try:
        parsed = ipaddress.ip_address(str(addr).split("%", 1)[0])
        if isinstance(parsed, ipaddress.IPv6Address) and parsed.ipv4_mapped is not None:
            parsed = parsed.ipv4_mapped
        return bool(parsed.is_loopback)
    except ValueError:
        return False


def _e33_validate_name(value, what: str) -> str:
    text = str(value or "").strip()
    if not _E33_NAME.match(text):
        raise ValueError("%s must match %s, got %r" % (what, _E33_NAME.pattern, value))
    return text


def _e33_validate_out_dir(raw, task_root) -> Path:
    """Absolute directory strictly below the task root, after symlink resolution."""
    if not raw:
        raise ValueError("out_dir is required")
    path = Path(str(raw))
    if not path.is_absolute():
        raise ValueError("out_dir must be absolute, got %r" % raw)
    root = Path(task_root).resolve()
    resolved = path.resolve(strict=False)
    if resolved == root or root not in resolved.parents:
        raise ValueError("out_dir must be below task root %s, got %s"
                         % (root, resolved))
    return resolved


def _e33_context(body: dict, task_root, *, require_all: bool = True) -> dict:
    if not isinstance(body, dict):
        raise ValueError("body must be a JSON object")
    fields = {}
    for name in ("run_id", "case_id", "phase"):
        value = body.get(name)
        if require_all or value not in (None, ""):
            fields[name] = _e33_validate_name(value, name)
        else:
            fields[name] = ""
    out_dir = body.get("out_dir")
    if require_all or out_dir not in (None, ""):
        fields["out_dir"] = str(_e33_validate_out_dir(out_dir, task_root))
    else:
        fields["out_dir"] = ""
    return fields


def _e33_snapshot_label(phase: str) -> str:
    return "%s-%s" % (phase, time.strftime("%Y%m%dT%H%M%S", time.gmtime()))


def _e33_cohort_dir(rank_dir: Path, ctx: dict) -> Path:
    return Path(rank_dir) / ("cohort-%s-%s" % (ctx["run_id"], ctx["phase"]))



# ---------------------------------------------------------------------------
# per-rank worker (real DeepInfer)
# ---------------------------------------------------------------------------
def _configure_worker_env(cfg: ClusterConfig, rank: int) -> None:
    # Must happen before deepinfer/torch are imported.
    os.environ["DEEPINFER_ACCELERATOR"] = "ascend"
    os.environ["DEEPINFER_ZERO_OVERHEAD_SCHEDULING"] = "1" if cfg.zos else "0"
    os.environ["DEEPINFER_USE_NEXTN_DECODE"] = str(cfg.nextn)
    os.environ["DEEPINFER_ENABLE_MTP_VARLEN"] = "1" if cfg.mtp_varlen else "0"
    os.environ["DEEPINFER_ENFORCE_EAGER"] = "1" if cfg.enforce_eager else "0"
    os.environ["DEEPINFER_DFLASH_ENFORCE_EAGER"] = "1" if cfg.enforce_eager else "0"
    os.environ["DEEPINFER_SAMPLER_FORCE_TOPK"] = str(cfg.sampler_force_topk)
    os.environ["DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS"] = "0"
    os.environ["DEEPINFER_EXPERT_COUNT_WINDOW_SIZE"] = "3"
    os.environ["DEEPINFER_USE_MULTIMODAL_POOL"] = "0"
    os.environ["DEEPINFER_ENGRAM_FORCE_GPU_STORAGE"] = "1"
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ.setdefault("TASK_QUEUE_ENABLE", "0")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(cfg.master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(cfg.dp_size * cfg.cp_size)
    if cfg.jit_cache_dir:
        # Explicit diagnostic cache root; overrides the per-library caches.
        os.environ["DEEPINFER_JIT_CACHE_DIR"] = cfg.jit_cache_dir
    elif cfg.tilelang_cache_dir:
        os.environ.setdefault("TILELANG_CACHE_DIR", cfg.tilelang_cache_dir)
    if cfg.prefill_attempt_interval is not None:
        os.environ["DEEPINFER_PREFILL_ATTEMPT_INTERVAL"] = str(cfg.prefill_attempt_interval)
    if cfg.prefill_any_attempt_interval is not None:
        os.environ["DEEPINFER_PREFILL_ANY_ATTEMPT_INTERVAL"] = str(cfg.prefill_any_attempt_interval)


def _build_llm(cfg: ClusterConfig, rank: int):
    root = Path("/work")
    sys.path[:0] = [str(root / "src/DeepInfer/examples"), str(root / "src/DeepInfer")]
    from _config import create_llm, setup_common_env

    setup_common_env(
        model_name=cfg.model,
        sampler_force_topk=cfg.sampler_force_topk,
        set_override_experts=False,
        enable_eplb=False,
    )
    llm = create_llm(
        cfg.model,
        rank,
        ep_size=cfg.ep_size,
        cp_size=cfg.cp_size,
        data_parallel_size=cfg.dp_size,
        max_model_len=cfg.max_model_len,
        max_num_batched_tokens=cfg.max_num_batched_tokens,
        max_num_seqs=cfg.max_num_seqs,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        lookahead_tokens=0,
    )
    return llm


def _send(events: mp.Queue, payload: dict) -> None:
    payload.setdefault("ts", time.time())
    try:
        events.put(payload, block=False)
    except queue.Full:  # pragma: no cover - queue is unbounded in practice
        pass


def _sha256_file(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _profile_identity(rank: int, cfg: ClusterConfig) -> dict:
    """Identity recorded by the worker that actually owns the device."""
    import importlib.metadata
    import torch
    import torch_npu
    import deepinfer

    package_dir = Path(deepinfer.__file__).resolve().parent
    libs = sorted(str(p) for p in (package_dir / "_C").glob("*.so"))
    adapter = Path(__file__).resolve()
    logical = int(torch.npu.current_device())
    physical = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    if not physical:
        physical = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    visible_ids = []
    for item in str(physical).split(","):
        item = item.strip()
        if item.isdigit():
            visible_ids.append(int(item))
    if len(visible_ids) == 1:
        # The usual launcher exposes exactly one device per rank; the process
        # logical index is 0 for every rank, so it cannot identify the card.
        physical_device_id = visible_ids[0]
    elif 0 <= logical < len(visible_ids):
        # If all devices are visible, CUDA_VISIBLE_DEVICES maps logical->physical.
        physical_device_id = visible_ids[logical]
    else:
        physical_device_id = None
    return {
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "rank": rank,
        "logical_device": logical,
        "physical_card": physical,
        "visible_device_ids": visible_ids,
        "physical_device_id": physical_device_id,
        "dp_rank": rank,
        "ep_rank": rank,
        "cp_rank": 0,
        "world_size": cfg.dp_size * cfg.cp_size,
        "ep_size": cfg.ep_size,
        "cp_size": cfg.cp_size,
        "torch_npu_version": getattr(torch_npu, "__version__", "unknown"),
        "torch_version": getattr(torch, "__version__", "unknown"),
        "deepinfer_version": getattr(deepinfer, "__version__", "unknown"),
        "deepinfer_package": str(package_dir),
        "deepinfer_libs": libs,
        "deepinfer_lib_sha256": {p: _sha256_file(p) for p in libs},
        "adapter_path": str(adapter),
        "adapter_sha256": _sha256_file(adapter),
        "python": sys.version.split()[0],
        "env": {
            key: os.environ.get(key)
            for key in (
                "ASCEND_RT_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
                "RANK", "WORLD_SIZE",
                "DEEPINFER_ZERO_OVERHEAD_SCHEDULING", "DEEPINFER_USE_NEXTN_DECODE",
                "DEEPINFER_ENFORCE_EAGER", "DEEPINFER_SAMPLER_FORCE_TOPK",
                "DEEPINFER_PREFILL_ATTEMPT_INTERVAL",
                "DEEPINFER_PREFILL_ANY_ATTEMPT_INTERVAL",
            )
        },
    }


# ---------------------------------------------------------------------------
# E33 worker-side cohort runtime
# ---------------------------------------------------------------------------
def _wrap_method_span(runtime: "E33RankRuntime", obj, attr: str) -> None:
    """Wrap one Python method with a telemetry span; never changes its result."""
    if obj is None or not hasattr(obj, attr):
        return
    original = getattr(obj, attr)
    if getattr(original, "_e33_span_wrapped", False):
        return

    def wrapped(*args, **kwargs):
        span = runtime.begin_event(attr, source="py_method")
        try:
            result = original(*args, **kwargs)
        except BaseException as exc:
            runtime.end_event(span, exc)
            raise
        runtime.end_event(span)
        return result

    wrapped._e33_span_wrapped = True
    wrapped._e33_original = original
    wrapped.__name__ = getattr(original, "__name__", attr)
    setattr(obj, attr, wrapped)


class E33RankRuntime:
    """Explicit start/stop cohort control inside one real rank worker.

    Command handling is synchronous in the worker loop, at a point where the
    worker owns no inference step.  It emits an ack and returns; it never
    enters a new collective.  The parent is responsible for broadcasting the
    same command to all eight workers while the group is globally idle.
    """

    ACK_KINDS = {
        "e33_start": "e33_start_ack",
        "e33_stop": "e33_stop_ack",
        "e33_snapshot": "e33_snapshot_ack",
    }

    def __init__(self, rank: int, cfg: ClusterConfig, events, engine,
                 idle_fn, stats: dict) -> None:
        self.rank = rank
        self.cfg = cfg
        self.events = events
        self.engine = engine
        self.idle_fn = idle_fn
        self.stats = stats
        self.active: dict | None = None
        self.last: dict | None = None
        self.identity: dict | None = None
        self._method_probes_installed = False
        self._spec_seen: set[int] = set()

    # -- acks --------------------------------------------------------------
    def _send_ack(self, kind: str, op_id, ok: bool, **fields) -> None:
        payload = {"kind": kind, "op_id": op_id, "rank": self.rank,
                   "ok": bool(ok),
                   "host": {
                       "pid": os.getpid(),
                       "native_tid": threading.get_native_id(),
                       "time_time_ns": time.time_ns(),
                       "perf_counter_ns": time.perf_counter_ns(),
                       "thread_time_ns": time.thread_time_ns(),
                   },
                   "worker_steps": int(self.stats.get("steps", 0))}
        payload.update(fields)
        _send(self.events, payload)

    def handle_cmd(self, msg: dict) -> bool:
        cmd = msg.get("cmd")
        if cmd not in self.ACK_KINDS:
            return False
        op_id = msg.get("op_id")
        try:
            if cmd == "e33_start":
                fields = self._start(msg)
            elif cmd == "e33_stop":
                fields = self._stop(msg)
            else:
                fields = self._snapshot(msg)
            self._send_ack(self.ACK_KINDS[cmd], op_id, True, **fields)
        except BaseException as exc:  # noqa: BLE001 - a failed command must ack
            tb = traceback.format_exc()
            print("[e33-rank%d] %s failed: %s" % (self.rank, cmd, tb),
                  file=sys.stderr, flush=True)
            self._send_ack("e33_error", op_id, False, where=cmd,
                           error=tb, state=self._state(),
                           identity=self.identity,
                           last=self.last)
        return True

    def _state(self) -> str:
        if self.active is None:
            return "stopped" if self.last is not None else "none"
        return self.active.get("state", "collecting")

    def _require_idle(self) -> None:
        try:
            idle = bool(self.idle_fn())
        except BaseException as exc:  # noqa: BLE001
            raise RuntimeError("worker idle check failed: %r" % (exc,))
        if not idle:
            raise RuntimeError("worker is not locally idle")

    def _snapshot_safe(self, ctx: dict, telemetry: E33Telemetry,
                       label: str, identity: dict | None) -> dict:
        rank_dir = Path(ctx["out_dir"]) / ("rank%02d" % self.rank)
        span = telemetry.begin("layout_snapshot", source="worker.snapshot",
                               extra={"label": label})
        try:
            summary = collect_layout_snapshot(
                self.engine, rank=self.rank, phase=ctx["phase"], label=label,
                run_id=ctx["run_id"], case_id=ctx["case_id"],
                identity=identity, out_dir=rank_dir / "snapshots")
        except BaseException as exc:  # noqa: BLE001 - snapshot is diagnostic
            telemetry.emit("layout_snapshot_error", source="worker.snapshot",
                           exception=exc, extra={"label": label})
            summary = {"path": None, "error": traceback.format_exc()[:8000],
                       "synthetic": False}
        span.end(extra={"path": summary.get("path"),
                        "tensor_entries": summary.get("tensor_entries"),
                        "truncated": summary.get("truncated")})
        return summary

    def _install_method_probes(self) -> None:
        if self._method_probes_installed:
            return
        driver = getattr(self.engine, "driver_worker", None)
        runner = getattr(driver, "model_runner", None) if driver is not None else None
        for attr in ("_prepare_input", "prepare_sampler_inputs", "execute_main_model",
                     "execute_sample"):
            _wrap_method_span(self, runner, attr)
        self._method_probes_installed = True

    def _start(self, msg: dict) -> dict:
        self._require_idle()
        if self.active is not None:
            raise RuntimeError("cohort already active on rank %d" % self.rank)
        self._spec_seen.clear()
        ctx = _e33_context(msg, self.cfg.e33_task_root, require_all=True)
        rank_dir = Path(ctx["out_dir"]) / ("rank%02d" % self.rank)
        rank_dir.mkdir(parents=True, exist_ok=True)
        telemetry = E33Telemetry(
            rank=self.rank, out_dir=rank_dir, run_id=ctx["run_id"],
            case_id=ctx["case_id"], phase=ctx["phase"], role="worker",
            source="worker")
        profiler = None
        try:
            identity = _profile_identity(self.rank, self.cfg)
            identity["e33_role"] = "worker"
            identity["e33_task_root"] = str(Path(self.cfg.e33_task_root).resolve())
            self.identity = identity
            telemetry.emit("rank_start_request", source="worker.control",
                           extra={"context": ctx,
                                  "worker_steps": int(self.stats.get("steps", 0))})
            before = self._snapshot_safe(ctx, telemetry, "cohort_start", identity)
            device = int(identity["logical_device"])
            cohort_dir = _e33_cohort_dir(rank_dir, ctx)
            profiler = CohortProfiler(
                out_dir=cohort_dir, run_id=ctx["run_id"],
                case_id=ctx["case_id"], phase=ctx["phase"], rank=self.rank,
                device=device, identity=identity)
            status = profiler.start()
            install_cpu_timer_probe(telemetry)
            self._install_method_probes()
            telemetry.emit("rank_profiler_start", source="worker.profiler",
                           extra={"profiler": status, "snapshot": before,
                                  "probe": "cpu_timer.record(instance)"})
            self.active = {
                "state": "collecting",
                "context": ctx,
                "telemetry": telemetry,
                "profiler": profiler,
                "identity": identity,
                "snapshot_start": before,
                "started_wall_ns": time.time_ns(),
                "op_id": msg.get("op_id"),
            }
            return {
                "state": "collecting",
                "identity": identity,
                "profiler": profiler.status(),
                "snapshot": before,
                "telemetry": telemetry.stats(),
                "context": ctx,
                "device": device,
            }
        except BaseException:
            if profiler is not None and profiler.started:
                try:
                    profiler.stop(stopped_reason="start_failure_cleanup")
                except BaseException:  # noqa: BLE001
                    pass
            try:
                set_cpu_timer_telemetry(None)
            except BaseException:  # noqa: BLE001
                pass
            telemetry.emit("rank_start_failed", source="worker.control",
                           exception=sys.exc_info()[1])
            telemetry.close()
            raise

    def _stop(self, msg: dict) -> dict:
        self._require_idle()
        supplied = _e33_context(msg, self.cfg.e33_task_root, require_all=True)
        if self.active is None:
            last = self.last
            if last and last.get("context"):
                for name in ("run_id", "case_id", "phase", "out_dir"):
                    if str(supplied[name]) != str(last["context"][name]):
                        raise RuntimeError(
                            "stop context %s=%r does not match stopped %r"
                            % (name, supplied[name], last["context"][name]))
                profiler = last.get("profiler") or {}
                export = profiler.get("export") or {}
                return {
                    "state": "stopped",
                    "already_stopped": True,
                    "identity": last.get("identity"),
                    "profiler": profiler,
                    "manifest": last.get("manifest"),
                    "snapshot": last.get("snapshot_stop"),
                    "snapshot_start": last.get("snapshot_start"),
                    "telemetry": last.get("telemetry") or {},
                    "context": last["context"],
                    "export_ok": bool(export.get("ok")),
                    "trace_present": bool(export.get("trace_present")),
                    "errors": profiler.get("errors") or [],
                }
            raise RuntimeError("cohort is not active on rank %d" % self.rank)
        session = self.active
        ctx = session["context"]
        for name in ("run_id", "case_id", "phase", "out_dir"):
            if str(supplied[name]) != str(ctx[name]):
                raise RuntimeError("stop context %s=%r does not match active %r"
                                   % (name, supplied[name], ctx[name]))
        telemetry: E33Telemetry = session["telemetry"]
        profiler: CohortProfiler = session["profiler"]
        status = profiler.stop(stopped_reason="explicit_parent_stop")
        after = self._snapshot_safe(ctx, telemetry, "cohort_stop",
                                    session["identity"])
        telemetry.emit("rank_profiler_stop", source="worker.profiler",
                       extra={"profiler": status, "snapshot": after})
        set_cpu_timer_telemetry(None)
        stats = telemetry.close()
        session["state"] = "stopped"
        self.last = {
            "context": ctx,
            "identity": session["identity"],
            "manifest": str(profiler.manifest_path),
            "profiler": status,
            "snapshot_start": session.get("snapshot_start"),
            "snapshot_stop": after,
            "telemetry": stats,
            "stopped_wall_ns": time.time_ns(),
        }
        self.active = None
        return {
            "state": "stopped",
            "identity": session["identity"],
            "profiler": status,
            "manifest": str(profiler.manifest_path),
            "snapshot": after,
            "snapshot_start": session.get("snapshot_start"),
            "telemetry": stats,
            "context": ctx,
            "export_ok": bool(status.get("export", {}).get("ok")),
            "trace_present": bool(status.get("export", {}).get("trace_present")),
            "errors": status.get("errors", []),
        }

    def _snapshot(self, msg: dict) -> dict:
        self._require_idle()
        if self.active is not None:
            session = self.active
            ctx = session["context"]
            telemetry = session["telemetry"]
            identity = session["identity"]
            own_telemetry = False
        else:
            ctx = _e33_context(msg, self.cfg.e33_task_root, require_all=True)
            rank_dir = Path(ctx["out_dir"]) / ("rank%02d" % self.rank)
            telemetry = E33Telemetry(
                rank=self.rank, out_dir=rank_dir, run_id=ctx["run_id"],
                case_id=ctx["case_id"], phase=ctx["phase"], role="worker",
                source="worker")
            identity = _profile_identity(self.rank, self.cfg)
            own_telemetry = True
        label = _e33_snapshot_label(ctx["phase"])
        summary = self._snapshot_safe(ctx, telemetry, label, identity)
        if own_telemetry:
            telemetry.close()
        return {"state": self._state(), "snapshot": summary,
                "identity": identity, "context": ctx}

    # -- inference-side observations --------------------------------------
    def begin_event(self, event_name: str, *, request_id=None,
                    iteration_id=None, source: str | None = None,
                    extra: dict | None = None):
        session = self.active
        if session is None:
            return None
        try:
            return session["telemetry"].begin(
                event_name, request_id=request_id, iteration_id=iteration_id,
                source=source or "worker", extra=extra)
        except BaseException:  # noqa: BLE001
            return None

    def end_event(self, span, exc=None, extra: dict | None = None) -> None:
        if span is not None:
            try:
                span.end(exc, extra=extra)
            except BaseException:  # noqa: BLE001
                pass

    def note_post_schedule(self, meta: dict, iteration_id: int,
                           started_ns: int | None = None, span=None) -> None:
        session = self.active
        if session is None:
            return
        extra = {
            "host_span_is_device_time": False,
            "observed_after_original_post_schedule": True,
            "prompt_run": bool(meta.get("prompt_run")),
            "is_empty": bool(meta.get("is_empty")),
            "decode_step": int(meta.get("decode_step") or 0),
            "num_batched_tokens": int(meta.get("num_batched_tokens") or 0),
            "scheduled_detail": list(meta.get("scheduled_detail") or []),
            "scheduled": list(meta.get("scheduled") or []),
            "finished": list(meta.get("finished") or []),
            "max_seqlen_k": int(meta.get("max_seqlen_k") or 0),
            "min_seqlen_k": int(meta.get("min_seqlen_k") or 0),
            "sum_seqlen_qxk": int(meta.get("sum_seqlen_qxk") or 0),
            "num_cached_tokens_gpu": int(meta.get("num_cached_tokens_gpu") or 0),
            "host_span_start_perf_counter_ns": started_ns,
        }
        if span is not None:
            self.end_event(span, extra=extra)
            return
        try:
            session["telemetry"].emit(
                "post_schedule", iteration_id=iteration_id,
                source="engine.post_schedule", extra=extra)
        except BaseException:  # noqa: BLE001
            pass

    def note_spec_counters(self, generated: int, verified: int,
                           iteration_id: int, scheduled: list) -> None:
        """One batch total per rank+iteration; never copy it onto each request."""
        session = self.active
        if session is None:
            return
        iteration_id = int(iteration_id)
        if iteration_id in self._spec_seen:
            return
        self._spec_seen.add(iteration_id)
        if len(self._spec_seen) > 8192:
            self._spec_seen.clear()
            self._spec_seen.add(iteration_id)
        try:
            session["telemetry"].emit(
                "spec_counters", iteration_id=iteration_id,
                source="engine.stats_logger.log_step",
                extra={
                    "scope": "rank_iteration_batch_total",
                    "not_per_request": True,
                    "num_generated_tokens": int(generated),
                    "num_verified_tokens": int(verified),
                    "scheduled_request_count": len(scheduled),
                    "scheduled_request_ids": list(scheduled),
                    "engine_generate_step": int(getattr(self.engine,
                                                         "generate_step", -1)),
                    "worker_loop_step": int(self.stats.get("steps", -1)),
                    "host_span_is_device_time": False,
                })
        except BaseException:  # noqa: BLE001
            pass

    def note_step(self, span, step_type: str, meta: dict, duration_ns: int,
                  exc=None) -> None:
        self.end_event(span, exc, extra={
            "step_type": step_type,
            "host_span_is_device_time": False,
            "scheduled_detail": list(meta.get("scheduled_detail") or []),
            "num_batched_tokens": int(meta.get("num_batched_tokens") or 0),
            "prompt_run": bool(meta.get("prompt_run")),
            "is_empty": bool(meta.get("is_empty")),
            "scheduled_query_tokens": dict(meta.get("scheduled_query_tokens") or {}),
            "step_duration_ns": int(duration_ns),
        })

    def shutdown(self) -> None:
        session = self.active
        if session is None:
            return
        try:
            status = session["profiler"].stop(stopped_reason="worker_shutdown")
            session["telemetry"].emit("rank_profiler_stop", source="worker.shutdown",
                                      extra={"profiler": status})
            set_cpu_timer_telemetry(None)
        except BaseException:  # noqa: BLE001
            pass
        finally:
            try:
                session["telemetry"].close()
            except BaseException:  # noqa: BLE001
                pass
            session["state"] = "stopped_on_shutdown"
            self.last = {"context": session.get("context"),
                         "profiler": session["profiler"].status(),
                         "shutdown": True}
            self.active = None


class MockE33Rank:
    """Protocol double for local checks; clearly synthetic, never qualified."""

    ACK_KINDS = E33RankRuntime.ACK_KINDS

    def __init__(self, rank: int, cfg: ClusterConfig, events, stats: dict,
                 idle_fn) -> None:
        self.rank = rank
        self.cfg = cfg
        self.events = events
        self.stats = stats
        self.idle_fn = idle_fn
        self.active = None
        self.last = None
        self.identity = None

    def _ack(self, kind, op_id, ok, **fields):
        payload = {"kind": kind, "op_id": op_id, "rank": self.rank,
                   "ok": bool(ok),
                   "host": {"pid": os.getpid(),
                            "native_tid": threading.get_native_id(),
                            "time_time_ns": time.time_ns(),
                            "perf_counter_ns": time.perf_counter_ns(),
                            "thread_time_ns": time.thread_time_ns()},
                   "worker_steps": int(self.stats.get("steps", 0))}
        payload.update(fields)
        _send(self.events, payload)

    def handle_cmd(self, msg: dict) -> bool:
        cmd = msg.get("cmd")
        if cmd not in self.ACK_KINDS:
            return False
        op_id = msg.get("op_id")
        try:
            if not self.idle_fn():
                raise RuntimeError("mock worker is not locally idle")
            if cmd == "e33_start":
                fields = self._start(msg)
            elif cmd == "e33_stop":
                fields = self._stop(msg)
            else:
                fields = self._snapshot(msg)
            self._ack(self.ACK_KINDS[cmd], op_id, True, **fields)
        except BaseException as exc:  # noqa: BLE001
            self._ack("e33_error", op_id, False, where=cmd,
                      error=traceback.format_exc()[:8000],
                      identity=self.identity, last=self.last)
        return True

    def _identity(self):
        return {"pid": os.getpid(), "rank": self.rank,
                "logical_device": 0,
                "physical_card": str(self.rank),
                "visible_device_ids": [self.rank],
                "physical_device_id": self.rank,
                "dp_rank": self.rank, "ep_rank": self.rank, "cp_rank": 0,
                "world_size": DEVICES, "ep_size": 8, "cp_size": 1,
                "synthetic": True, "mock": True,
                "e33_task_root": str(Path(self.cfg.e33_task_root).resolve()),
                "adapter_path": str(Path(__file__).resolve())}

    def _start(self, msg):
        if self.active is not None:
            raise RuntimeError("mock cohort already active")
        ctx = _e33_context(msg, self.cfg.e33_task_root, require_all=True)
        rank_dir = Path(ctx["out_dir"]) / ("rank%02d" % self.rank)
        rank_dir.mkdir(parents=True, exist_ok=True)
        telemetry = E33Telemetry(rank=self.rank, out_dir=rank_dir,
                                 run_id=ctx["run_id"], case_id=ctx["case_id"],
                                 phase=ctx["phase"], role="worker",
                                 source="mock")
        self.identity = self._identity()
        telemetry.emit("rank_profiler_start", source="mock.profiler",
                       extra={"synthetic": True,
                              "schedule": {"mode": "none", "step_calls": 0}})
        self.active = {"context": ctx, "telemetry": telemetry,
                       "identity": self.identity,
                       "steps_at_start": int(self.stats.get("steps", 0)),
                       "state": "collecting", "op_id": msg.get("op_id")}
        return {"state": "collecting", "identity": self.identity,
                "context": ctx, "synthetic": True,
                "profiler": {"started": True, "stopped": False,
                             "schedule": {"mode": "none", "step_calls": 0},
                             "out_dir": str(_e33_cohort_dir(rank_dir, ctx)),
                             "qualified": None,
                             "qualification": "deferred_to_main_controller"},
                "telemetry": telemetry.stats()}

    def _stop(self, msg):
        supplied = _e33_context(msg, self.cfg.e33_task_root, require_all=True)
        if self.active is None:
            last = self.last
            if last and last.get("context"):
                for name in ("run_id", "case_id", "phase", "out_dir"):
                    if str(supplied[name]) != str(last["context"][name]):
                        raise RuntimeError("mock stop context mismatch for %s" % name)
                manifest = last.get("profiler") or {}
                export = manifest.get("export") or {}
                return {"state": "stopped", "already_stopped": True,
                        "identity": last.get("identity"),
                        "profiler": manifest,
                        "manifest": last.get("manifest"),
                        "telemetry": last.get("telemetry") or {},
                        "context": last["context"],
                        "export_ok": bool(export.get("ok")),
                        "trace_present": bool(export.get("trace_present")),
                        "errors": [], "synthetic": True}
            raise RuntimeError("mock cohort is not active")
        ctx = self.active["context"]
        for name in ("run_id", "case_id", "phase", "out_dir"):
            if str(supplied[name]) != str(ctx[name]):
                raise RuntimeError("mock stop context mismatch for %s" % name)
        rank_dir = Path(ctx["out_dir"]) / ("rank%02d" % self.rank)
        cohort_dir = _e33_cohort_dir(rank_dir, ctx)
        output_dir = cohort_dir / "ASCEND_PROFILER_OUTPUT"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "kernel_details.csv").write_text(
            "Device ID,Name,Start Time(us),Duration(us)\n"
            "%d,mock_kernel,1.0,1.0\n" % self.rank)
        (output_dir / "trace_view.json").write_text(
            json.dumps([{"name": "mock", "ph": "X", "ts": 1.0, "dur": 1.0}]) + "\n")
        telemetry: E33Telemetry = self.active["telemetry"]
        steps_at_stop = int(self.stats.get("steps", 0))
        telemetry.emit("rank_profiler_stop", source="mock.profiler",
                       extra={"steps_at_stop": steps_at_stop,
                              "synthetic": True, "step_calls": 0})
        telemetry_stats = telemetry.close()
        manifest = {
            "schema_version": "e33-cohort/1.0",
            "synthetic": True,
            "run_id": ctx["run_id"], "case_id": ctx["case_id"],
            "phase": ctx["phase"], "rank": self.rank, "device": self.rank,
            "out_dir": str(cohort_dir),
            "started_wall_ns": None, "stopped_wall_ns": time.time_ns(),
            "stopped": True, "stop_complete": True,
            "stopped_reason": "explicit_parent_stop",
            "schedule": {"mode": "none", "explicit_stop_required": True,
                         "step_calls": 0},
            "profiler_step_calls": 0,
            "engine_steps_at_start": self.active.get("steps_at_start"),
            "engine_steps_at_stop": steps_at_stop,
            "identity": self.active["identity"],
            "export": {"ok": True, "trace_present": True,
                       "kernel_details": [str(output_dir / "kernel_details.csv")],
                       "trace_view": [str(output_dir / "trace_view.json")]},
            "errors": [],
            "qualified": None,
            "qualification": "deferred_to_main_controller",
            "qualification_note": "synthetic mock export; never hardware evidence",
        }
        manifest_path = cohort_dir / "profile_manifest.json"
        atomic_write_json(manifest_path, manifest)
        self.last = {"context": ctx, "identity": self.active["identity"],
                     "manifest": str(manifest_path), "profiler": manifest,
                     "telemetry": telemetry_stats,
                     "stopped_wall_ns": time.time_ns()}
        self.active = None
        return {"state": "stopped", "identity": self.last["identity"],
                "profiler": manifest, "manifest": str(manifest_path),
                "telemetry": telemetry_stats, "context": ctx,
                "export_ok": True, "trace_present": True, "errors": [],
                "synthetic": True}

    def _snapshot(self, msg):
        ctx = (self.active["context"] if self.active is not None
               else _e33_context(msg, self.cfg.e33_task_root, require_all=True))
        rank_dir = Path(ctx["out_dir"]) / ("rank%02d" % self.rank)
        label = _e33_snapshot_label(ctx["phase"])
        path = rank_dir / "snapshots" / ("layout-%s.json" % label)
        atomic_write_json(path, {
            "schema_version": "e33-layout-snapshot/1.0",
            "synthetic": True, "mock": True, "rank": self.rank,
            "phase": ctx["phase"], "label": label, "run_id": ctx["run_id"],
            "case_id": ctx["case_id"], "device_memory": {"source": "mock"},
            "cpu_memory": {"affinity_count": None},
            "tensor_values_read": False,
            "physical_hbm_bank_layout": "unknown",
            "note": "synthetic mock snapshot; never hardware evidence",
        })
        return {"state": "collecting" if self.active else "stopped",
                "snapshot": {"path": str(path), "synthetic": True,
                             "errors": []},
                "identity": self.identity, "context": ctx}


# ---------------------------------------------------------------------------
# opt-in in-worker MegaMoE replay (disabled unless DI_REPLAY_SPEC is set)
# ---------------------------------------------------------------------------
_REPLAY_CONTROLLER = None


def _tensor_meta(t) -> dict:
    return {
        "shape": [int(x) for x in t.shape],
        "dtype": str(t.dtype),
        "device": str(t.device),
        "stride": [int(x) for x in t.stride()],
        "numel": int(t.numel()),
        "nbytes": int(t.numel() * t.element_size()),
    }


def _tensor_sha256(t) -> str:
    import torch
    cpu = t.detach().to("cpu").contiguous()
    h = hashlib.sha256()
    h.update(bytes(cpu.view(torch.uint8).numpy().tobytes()))
    return h.hexdigest()


def _find_mega_modules(root) -> tuple[list, int]:
    """Find model modules that own a MegaMoE callable, without touching weights."""
    import torch
    import torch.nn as nn
    seen: set[int] = set()
    stack = [(root, 0)]
    scanned = 0
    found: list = []
    while stack and scanned < 300000:
        obj, depth = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        scanned += 1
        try:
            if hasattr(obj, "mega_moe_func") and hasattr(obj, "layer_idx"):
                found.append(obj)
                continue
        except Exception:  # noqa: BLE001
            continue
        if depth >= 30:
            continue
        if isinstance(obj, (str, bytes, bytearray, int, float, bool, type(None))):
            continue
        if torch.is_tensor(obj):
            continue
        try:
            values = vars(obj).values()
        except TypeError:
            values = ()
        for value in values:
            if isinstance(value, (str, bytes, bytearray, int, float, bool, type(None))):
                continue
            if torch.is_tensor(value):
                continue
            if isinstance(value, (list, tuple, set, frozenset)):
                stack.extend((v, depth + 1) for v in value)
            elif isinstance(value, dict):
                stack.extend((v, depth + 1) for v in value.values())
            elif isinstance(value, nn.Module) or hasattr(value, "__dict__"):
                stack.append((value, depth + 1))
    return found, scanned


class MegaMoEReplayController:
    """Capture and time real MegaMoE calls inside the rank worker.

    The replay uses the live module weights, live symmetric buffers and the
    exact call-site tensors.  It is deliberately opt-in: without
    ``DI_REPLAY_SPEC`` no model call is wrapped and no extra NPU work runs.
    """

    def __init__(self, specs: list[dict], rank: int, cfg: ClusterConfig) -> None:
        self.specs = [dict(s) for s in specs]
        for s in self.specs:
            s.setdefault("warmup", 3)
            s.setdefault("repeats", 10)
            s["done"] = False
        self.rank = rank
        self.cfg = cfg
        self.step = 0
        self.meta: dict = {}
        self.call_index = 0
        self.layers: list[int] = []
        out = os.environ.get("DI_REPLAY_OUT") or "/work/performance-20261021/replays"
        self.out_root = Path(out)
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.installed_at = time.time()

    def begin_engine_step(self, step: int) -> None:
        self.step = int(step)
        self.call_index = 0
        # New step: calls at each layer start again.
        self._step_calls = 0

    def note_scheduler(self, meta: dict) -> None:
        self.meta = dict(meta)

    def note_call(self, layer_idx: int) -> int:
        # layer order is deterministic inside one engine step; the counter is
        # only evidence, matching uses the bound module's layer_idx.
        self.call_index += 1
        return self.call_index

    def should_fire(self, layer_idx: int) -> bool:
        for spec in self.specs:
            if spec["done"]:
                continue
            if int(spec.get("layer", -1)) != int(layer_idx):
                continue
            if spec.get("engine_step") is not None and int(spec["engine_step"]) != self.step:
                continue
            phase = spec.get("phase")
            if phase == "prompt" and not self.meta.get("prompt_run"):
                continue
            if phase == "decode" and (self.meta.get("prompt_run")
                                      or self.meta.get("is_empty")):
                continue
            if spec.get("prompt_len") is not None:
                detail = list(self.meta.get("scheduled_detail") or [])
                if not detail or int(detail[0].get("prompt_len") or 0) != int(spec["prompt_len"]):
                    continue
            k_len = int(self.meta.get("max_seqlen_k") or 0)
            if spec.get("min_k") is not None and k_len < int(spec["min_k"]):
                continue
            if spec.get("max_k") is not None and k_len > int(spec["max_k"]):
                continue
            if spec.get("decode_step") is not None and int(
                    self.meta.get("decode_step") or 0) != int(spec["decode_step"]):
                continue
            return True
        return False

    def _mark(self, layer_idx: int) -> dict:
        for spec in self.specs:
            if (not spec["done"] and int(spec.get("layer", -1)) == int(layer_idx)
                    and self.should_fire(layer_idx)):
                spec["done"] = True
                return spec
        raise RuntimeError("replay fire without matching spec")

    def replay(self, orig, layer_idx: int, args, kwargs):
        import torch

        spec = self._mark(layer_idx)
        label = str(spec.get("label") or ("l%02d-%s" % (layer_idx, self.step)))
        out = args[0]
        sym = args[3]
        num_tokens = int(out.shape[0])
        warmup = int(spec.get("warmup", 3))
        repeats = int(spec.get("repeats", 10))
        t_start = time.perf_counter()
        torch.npu.synchronize()

        x = sym.x[:num_tokens]
        x_sf = sym.x_sf[:num_tokens]
        topk_idx = sym.topk_idx[:num_tokens]
        topk_weights = sym.topk_weights[:num_tokens]
        capture = {
            "x": x.detach().to("cpu").clone(),
            "x_sf": x_sf.detach().to("cpu").clone(),
            "topk_idx": topk_idx.detach().to("cpu").clone(),
            "topk_weights": topk_weights.detach().to("cpu").clone(),
        }
        input_sha = {name: _tensor_sha256(t) for name, t in capture.items()}
        route_hist = torch.bincount(
            topk_idx.detach().reshape(-1).to(torch.int64),
            minlength=int(getattr(sym, "num_experts", 0)),
        ).to("cpu").tolist()
        stats = kwargs.get("cumulative_local_expert_recv_stats")
        stats_backup = stats.detach().clone() if torch.is_tensor(stats) else None
        x_backup = x.detach().clone()
        xsf_backup = x_sf.detach().clone()
        topk_backup = topk_idx.detach().clone()
        weights_backup = topk_weights.detach().clone()

        scratch = torch.empty_like(out)
        meta = {
            "schema_version": "1.0",
            "label": label,
            "rank": self.rank,
            "pid": os.getpid(),
            "adapter_sha256": _sha256_file(Path(__file__).resolve()),
            "installed_at": self.installed_at,
            "spec": {k: v for k, v in spec.items()},
            "engine_step": self.step,
            "layer_idx": layer_idx,
            "call_index_in_step": self.call_index,
            "scheduler": {
                "prompt_run": bool(self.meta.get("prompt_run")),
                "decode_step": int(self.meta.get("decode_step") or 0),
                "num_batched_tokens": int(self.meta.get("num_batched_tokens") or 0),
                "max_seqlen_k": int(self.meta.get("max_seqlen_k") or 0),
                "min_seqlen_k": int(self.meta.get("min_seqlen_k") or 0),
                "sum_seqlen_qxk": int(self.meta.get("sum_seqlen_qxk") or 0),
                "scheduled_detail": list(self.meta.get("scheduled_detail") or []),
            },
            "effective_tokens_local": num_tokens,
            "backend": {
                "callable": str(getattr(orig, "__module__", "")) + "." + str(
                    getattr(orig, "__name__", type(orig).__name__)),
                "mma_type": getattr(sym, "mma_type", None),
                "activation": getattr(sym, "activation", None),
                "ep_size": int(self.cfg.ep_size),
                "num_experts": int(getattr(sym, "num_experts", 0)),
                "num_max_tokens_per_rank": int(getattr(sym, "num_max_tokens_per_rank", 0)),
                "num_topk": int(getattr(sym, "num_topk", 0)),
                "hidden": int(getattr(sym, "hidden", 0)),
                "intermediate_hidden": int(getattr(sym, "intermediate_hidden", 0)),
                "num_shared_experts": int(getattr(sym, "num_shared_experts", 0)),
            },
            "tensors": {
                "out": _tensor_meta(out),
                "scratch_out": _tensor_meta(scratch),
                "x": _tensor_meta(x),
                "x_sf": _tensor_meta(x_sf),
                "topk_idx": _tensor_meta(topk_idx),
                "topk_weights": _tensor_meta(topk_weights),
                "l1_weight": [_tensor_meta(t) for t in args[1]] if isinstance(args[1], (tuple, list)) else _tensor_meta(args[1]),
                "l2_weight": [_tensor_meta(t) for t in args[2]] if isinstance(args[2], (tuple, list)) else _tensor_meta(args[2]),
            },
            "input_sha256": input_sha,
            "route_hist_physical": route_hist,
            "route_unique_experts": int(sum(1 for v in route_hist if v)),
            "weights": {
                "l1_shapes": [list(t.shape) for t in args[1]] if isinstance(args[1], (tuple, list)) else list(args[1].shape),
                "l2_shapes": [list(t.shape) for t in args[2]] if isinstance(args[2], (tuple, list)) else list(args[2].shape),
                "note": "live repacked FP8/FP4 buffers stayed in the resident worker; weights were not copied",
            },
            "timing": {},
            "validity": {},
            "errors": [],
        }

        def call(dst):
            return orig(dst, *args[1:], **kwargs)

        try:
            for _ in range(warmup):
                call(scratch)
            torch.npu.synchronize()
            isolated = []
            events = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                ev0 = ev1 = None
                try:
                    ev0 = torch.npu.Event(enable_timing=True)
                    ev1 = torch.npu.Event(enable_timing=True)
                    ev0.record()
                except Exception as exc:  # noqa: BLE001
                    meta["errors"].append({"where": "event_create", "error": repr(exc)})
                call(scratch)
                try:
                    if ev1 is not None:
                        ev1.record()
                    torch.npu.synchronize()
                finally:
                    t1 = time.perf_counter()
                row = {"wall_ms": (t1 - t0) * 1000.0}
                if ev0 is not None and ev1 is not None:
                    try:
                        row["event_ms"] = float(ev0.elapsed_time(ev1))
                    except Exception as exc:  # noqa: BLE001
                        meta["errors"].append({"where": "event_elapsed", "error": repr(exc)})
                isolated.append(row)
            # Back-to-back queueing, closer to a resident steady-state layer.
            t0 = time.perf_counter()
            for _ in range(repeats):
                call(scratch)
            torch.npu.synchronize()
            t1 = time.perf_counter()
            pipelined_ms = (t1 - t0) * 1000.0 / float(repeats)
            # Restore the exact pre-call inputs and stats, then run the real call.
            if stats_backup is not None:
                stats.copy_(stats_backup)
            x.copy_(x_backup)
            x_sf.copy_(xsf_backup)
            topk_idx.copy_(topk_backup)
            topk_weights.copy_(weights_backup)
            torch.npu.synchronize()
            call(out)
            torch.npu.synchronize()
            diff = (scratch.float() - out.float()).abs()
            meta["validity"] = {
                "same_inputs_after_replay": {
                    "x": _tensor_sha256(x) == input_sha["x"],
                    "x_sf": _tensor_sha256(x_sf) == input_sha["x_sf"],
                    "topk_idx": _tensor_sha256(topk_idx) == input_sha["topk_idx"],
                    "topk_weights": _tensor_sha256(topk_weights) == input_sha["topk_weights"],
                },
                "replay_vs_real_max_abs": float(diff.max().item()),
                "replay_vs_real_mean_abs": float(diff.mean().item()),
                "replay_vs_real_exact_frac": float((scratch == out).float().mean().item()),
                "replay_out_sha256": _tensor_sha256(scratch),
                "real_out_sha256": _tensor_sha256(out),
            }
            meta["timing"] = {
                "warmup": warmup,
                "repeats": repeats,
                "isolated": isolated,
                "isolated_wall_ms_mean": sum(r["wall_ms"] for r in isolated) / len(isolated),
                "isolated_wall_ms_min": min(r["wall_ms"] for r in isolated),
                "isolated_wall_ms_max": max(r["wall_ms"] for r in isolated),
                "isolated_event_ms_mean": (
                    sum(r["event_ms"] for r in isolated if "event_ms" in r)
                    / max(1, sum(1 for r in isolated if "event_ms" in r))),
                "pipelined_wall_ms_per_call": pipelined_ms,
                "capture_and_restore_wall_ms": (time.perf_counter() - t_start) * 1000.0,
            }
        except BaseException as exc:  # noqa: BLE001
            meta["errors"].append({"where": "replay", "error": repr(exc),
                                   "traceback": traceback.format_exc()[-4000:]})
            if stats_backup is not None:
                try:
                    stats.copy_(stats_backup)
                except Exception:  # noqa: BLE001
                    pass
        finally:
            dest = self.out_root / label
            dest.mkdir(parents=True, exist_ok=True)
            try:
                torch.save(capture, dest / ("rank%02d-capture.pt" % self.rank))
            except Exception as exc:  # noqa: BLE001
                meta["errors"].append({"where": "capture_save", "error": repr(exc)})
            (dest / ("rank%02d.json" % self.rank)).write_text(
                json.dumps(meta, indent=2, sort_keys=True) + "\n")
        return out


def _install_replay_hooks(llm, rank: int, cfg: ClusterConfig) -> None:
    spec_json = os.environ.get("DI_REPLAY_SPEC")
    if not spec_json:
        return
    try:
        parsed = json.loads(spec_json)
    except Exception as exc:  # noqa: BLE001
        print("[replay-spec-invalid] rank=%d %r" % (rank, exc),
              file=sys.stderr, flush=True)
        return
    specs = parsed if isinstance(parsed, list) else [parsed]
    ctl = MegaMoEReplayController(specs, rank, cfg)

    global _REPLAY_CONTROLLER

    found: dict[int, object] = {}
    modules, scanned = _find_mega_modules(llm)
    for module in modules:
        try:
            layer = int(module.layer_idx)
        except Exception:  # noqa: BLE001
            continue
        if layer in found:
            continue
        orig = module.mega_moe_func

        def make_wrapper(original, layer_idx):
            def wrapped(*args, **kwargs):
                controller = _REPLAY_CONTROLLER
                if controller is None:
                    return original(*args, **kwargs)
                controller.note_call(layer_idx)
                if controller.should_fire(layer_idx):
                    return controller.replay(original, layer_idx, args, kwargs)
                return original(*args, **kwargs)
            return wrapped

        module.mega_moe_func = make_wrapper(orig, layer)
        found[layer] = module
    _REPLAY_CONTROLLER = ctl
    ctl.layers = sorted(found)
    manifest = {
        "rank": rank,
        "pid": os.getpid(),
        "specs": specs,
        "layers": ctl.layers,
        "scanned_objects": scanned,
        "adapter_sha256": _sha256_file(Path(__file__).resolve()),
        "written_at": time.time(),
    }
    ctl.out_root.mkdir(parents=True, exist_ok=True)
    (ctl.out_root / ("install-rank%02d.json" % rank)).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print("[replay-installed] rank=%d layers=%d out=%s" % (
        rank, len(ctl.layers), ctl.out_root), file=sys.stderr, flush=True)


def real_worker(rank: int, cfg: ClusterConfig, inbox: mp.Queue, events: mp.Queue,
                idle_barrier, stop_barrier, stop_votes) -> None:
    try:
        _configure_worker_env(cfg, rank)
        t0 = time.monotonic()
        llm = _build_llm(cfg, rank)
        _install_replay_hooks(llm, rank, cfg)
        engine = llm.llm_engine
        from e27_prefix_cache import install as _install_e27_prefix_cache
        _install_e27_prefix_cache(engine, rank)
        _send(events, {
            "kind": "ready", "rank": rank,
            "load_seconds": time.monotonic() - t0,
            "resolved": {
                "zos": os.environ.get("DEEPINFER_ZERO_OVERHEAD_SCHEDULING"),
                "nextn": os.environ.get("DEEPINFER_USE_NEXTN_DECODE"),
                "mtp_varlen": os.environ.get("DEEPINFER_ENABLE_MTP_VARLEN"),
                "enforce_eager": os.environ.get("DEEPINFER_ENFORCE_EAGER"),
                "force_topk": os.environ.get("DEEPINFER_SAMPLER_FORCE_TOPK"),
                "prefill_attempt_interval": os.environ.get("DEEPINFER_PREFILL_ATTEMPT_INTERVAL", "default:70"),
                "prefill_any_attempt_interval": os.environ.get("DEEPINFER_PREFILL_ANY_ATTEMPT_INTERVAL", "default:100"),
                "max_num_seqs": cfg.max_num_seqs,
                "max_num_batched_tokens": cfg.max_num_batched_tokens,
                "max_model_len": cfg.max_model_len,
                "chunked_prefill": True,
            },
        })
    except BaseException:
        tb = traceback.format_exc()
        print("[worker-build-failed] rank=%d\n%s" % (rank, tb),
              file=sys.stderr, flush=True)
        _send(events, {"kind": "error", "rank": rank, "where": "build",
                       "error": tb})
        time.sleep(0.5)
        raise SystemExit(1)

    from deepinfer._C.engine import RequestOutput
    from deepinfer.sampling_params import SamplingParams

    stats = {"steps": 0, "placeholder_steps": 0, "prompt_steps": 0,
             "decode_steps": 0, "tokens_emitted": 0, "flushes": 0}
    last_meta = {"is_empty": None, "prompt_run": None, "decode_step": None,
                 "scheduled": [], "num_batched_tokens": 0,
                 "max_seqlen_k": 0, "min_seqlen_k": 0, "sum_seqlen_qxk": 0,
                 "num_cached_tokens_gpu": 0, "scheduled_detail": []}
    first_seen: dict[str, bool] = {}
    local_active: set[str] = set()
    phase_records: dict[str, dict] = {}

    e33 = E33RankRuntime(
        rank, cfg, events, engine,
        idle_fn=lambda: not local_active,
        stats=stats)

    # --- E12 service-path counters (no device synchronizations) ------------
    graph_stats = {"main_replays": 0, "dflash_replays": 0,
                   "main_graph_sizes": {}, "dflash_graph_sizes": {}}

    def install_graph_counters() -> None:
        try:
            from deepinfer.worker.cuda_graph_runner import CudaGraphRunner

            original_replay = CudaGraphRunner.replay

            def counted_replay(self, data):
                graph_stats["main_replays"] += 1
                size = None
                for bucket, candidate in (self.graphs or {}).items():
                    if candidate is data:
                        size = bucket
                        break
                key = str(size)
                sizes = graph_stats["main_graph_sizes"]
                sizes[key] = sizes.get(key, 0) + 1
                return original_replay(self, data)

            CudaGraphRunner.replay = counted_replay

            from deepinfer.worker.dflash_cuda_graph_runner import DFlashCudaGraphRunner

            original_dflash_replay = DFlashCudaGraphRunner.replay
            original_select = DFlashCudaGraphRunner.select

            def counted_select(self, req_cnt):
                size, data = original_select(self, req_cnt)
                sizes = graph_stats["dflash_graph_sizes"]
                key = str(size)
                sizes[key] = sizes.get(key, 0) + 1
                return size, data

            def counted_dflash_replay(self, decode_input):
                graph_stats["dflash_replays"] += 1
                return original_dflash_replay(self, decode_input)

            DFlashCudaGraphRunner.select = counted_select
            DFlashCudaGraphRunner.replay = counted_dflash_replay
            print("[graph-counters] rank=%d installed" % rank,
                  file=sys.stderr, flush=True)
        except BaseException as exc:  # noqa: BLE001 - diagnostics only
            print("[graph-counters] rank=%d unavailable %r" % (rank, exc),
                  file=sys.stderr, flush=True)

    def install_spec_counters() -> None:
        logger = getattr(engine, "stats_logger", None)
        if logger is None or not hasattr(logger, "log_step"):
            print("[spec-counters] rank=%d no stats_logger" % rank,
                  file=sys.stderr, flush=True)
            return
        original_log_step = logger.log_step

        def counted_log_step(step_stats):
            try:
                generated = getattr(step_stats, "num_generated_tokens", None)
                verified = getattr(step_stats, "num_verified_tokens", None)
                if generated is not None:
                    gen = sum(int(v) for v in generated)
                    ver = (sum(int(v) for v in verified)
                           if verified is not None else 0)
                    # These are batch totals.  Record them once for this
                    # rank+iteration; never add the whole batch to each
                    # scheduled request (the known E28 accounting bug).
                    e33.note_spec_counters(
                        gen, ver, int(stats["steps"]) + 1,
                        list(last_meta.get("scheduled") or []))
            except BaseException:  # noqa: BLE001 - diagnostics only
                pass
            return original_log_step(step_stats)

        logger.log_step = counted_log_step
        print("[spec-counters] rank=%d installed" % rank,
              file=sys.stderr, flush=True)

    install_graph_counters()
    install_spec_counters()

    def phase_report(rid: str, reason: str) -> None:
        rec = phase_records.pop(rid, None)
        if rec is None or rec.get("reported"):
            return
        baseline = rec.pop("graph_baseline", None)
        if baseline is not None:
            rec["graph_main_replays"] = (
                graph_stats["main_replays"] - baseline["main"])
            rec["graph_dflash_replays"] = (
                graph_stats["dflash_replays"] - baseline["dflash"])
            rec["graph_main_sizes"] = {
                key: value - baseline["main_sizes"].get(key, 0)
                for key, value in graph_stats["main_graph_sizes"].items()
                if value - baseline["main_sizes"].get(key, 0) > 0}
            rec["graph_dflash_sizes"] = {
                key: value - baseline["dflash_sizes"].get(key, 0)
                for key, value in graph_stats["dflash_graph_sizes"].items()
                if value - baseline["dflash_sizes"].get(key, 0) > 0}
        rec["reported"] = True
        rec["report_reason"] = reason
        rec["reported_wall"] = time.time()
        _send(events, {"kind": "request_phase", "rank": rank, "record": rec})

    # Retain the actual scheduler object through ZOS post-processing.  A
    # missed identity stays unknown; origin is never inferred as delivery-1.
    scheduled_origins: dict[int, tuple[object, int]] = {}
    output_origins: dict[int, int | None] = {}
    orig_process_model_outputs = engine._process_model_outputs

    def traced_process_model_outputs(output, scheduler_outputs):
        entry = scheduled_origins.pop(id(scheduler_outputs), None)
        origin = (entry[1] if entry is not None
                  and entry[0] is scheduler_outputs else None)
        span = e33.begin_event("process_model_outputs",
                               iteration_id=origin,
                               source="engine._process_model_outputs")
        try:
            raw, log_stats_func = orig_process_model_outputs(
                output, scheduler_outputs)
        except BaseException as exc:  # noqa: BLE001 - preserve the original failure
            e33.end_event(span, exc)
            raise
        if raw is not None:
            for pointer in raw:
                if len(output_origins) >= 4096:
                    output_origins.pop(next(iter(output_origins)), None)
                output_origins[int(pointer)] = origin
        e33.end_event(span)
        return raw, log_stats_func

    engine._process_model_outputs = traced_process_model_outputs
    orig_post_schedule = engine.post_schedule

    def post_schedule(scheduler_outputs):
        origin_step = int(stats["steps"]) + 1
        # Bound diagnostic retention if a binding ever changes object identity.
        for key, (_, recorded_step) in tuple(scheduled_origins.items()):
            if recorded_step < origin_step - 4:
                scheduled_origins.pop(key, None)
        scheduled_origins[id(scheduler_outputs)] = (
            scheduler_outputs, origin_step)
        detail = []
        for seq in scheduler_outputs.scheduled_seqs:
            try:
                detail.append({
                    "request_id": str(seq.request_id),
                    "prompt_len": int(seq.get_prompt_len()),
                    "full_len": int(seq.get_len()),
                    "num_cached_tokens": int(seq.num_cached_tokens),
                })
            except Exception:  # noqa: BLE001 - metadata is diagnostic only
                detail.append({"request_id": str(getattr(seq, "request_id", ""))})
        last_meta.update({
            "is_empty": bool(scheduler_outputs.is_empty()),
            "prompt_run": bool(scheduler_outputs.prompt_run),
            "num_batched_tokens": int(scheduler_outputs.num_batched_tokens),
            "max_seqlen_k": int(scheduler_outputs.max_seqlen_k),
            "min_seqlen_k": int(scheduler_outputs.min_seqlen_k),
            "sum_seqlen_qxk": int(scheduler_outputs.sum_seqlen_qxk),
            "num_cached_tokens_gpu": int(scheduler_outputs.num_cached_tokens_gpu),
            "scheduled_detail": detail,
            "scheduled": [s.request_id for s in scheduler_outputs.scheduled_seqs],
            "finished": [s.request_id for s in scheduler_outputs.finished_seqs],
            "decode_step": int(engine.scheduler.decode_step),
        })
        if _REPLAY_CONTROLLER is not None:
            _REPLAY_CONTROLLER.note_scheduler(last_meta)
        span = e33.begin_event("post_schedule", iteration_id=origin_step,
                               source="engine.post_schedule")
        started_ns = time.perf_counter_ns()
        try:
            orig_post_schedule(scheduler_outputs)
        except BaseException as exc:  # noqa: BLE001 - preserve the original failure
            e33.end_event(span, exc)
            raise
        e33.note_post_schedule(last_meta, origin_step, started_ns=started_ns,
                               span=span)

    engine.post_schedule = post_schedule

    # Read the exact per-sequence Q lengths before native workers consume the
    # pooled metadata. Copy integers only; never retain the borrowed objects.
    original_execute_async_step = engine._execute_model_async_step

    def traced_execute_async_step(seq_metadata_list, scheduler_outputs):
        last_meta["scheduled_query_tokens"] = {
            str(meta.request_id): int(meta.seqlen_q)
            for meta in seq_metadata_list
        }
        span = e33.begin_event("execute_model_async_step",
                               iteration_id=int(stats["steps"]) + 1,
                               source="engine._execute_model_async_step")
        try:
            result = original_execute_async_step(seq_metadata_list, scheduler_outputs)
        except BaseException as exc:  # noqa: BLE001
            e33.end_event(span, exc)
            raise
        e33.end_event(span)
        return result

    engine._execute_model_async_step = traced_execute_async_step

    def handle_outputs(raw_outputs, step_id: int) -> None:
        if raw_outputs is None:
            return
        for raw in raw_outputs:
            origin_step = output_origins.pop(int(raw), None)
            out = RequestOutput.clone(raw)
            rid = out.request_id
            first = rid not in first_seen
            first_seen[rid] = True
            stats["tokens_emitted"] += 1
            rec = phase_records.get(rid)
            now = time.time()
            drafted = getattr(out, "draft_tok", None)
            accepted = getattr(out, "accept_tok", None)
            if rec is not None:
                # Official examples/offline_inference.py counts native rounds
                # on each delivered golden RequestOutput.  draft_tok is
                # verified candidates + the bonus token; accept_tok is the
                # accepted subset + bonus.  Values precede output-cap/EOS
                # filtering, so they are reported separately from usage.
                if drafted is None or accepted is None:
                    rec["native_output_counters_available"] = False
                else:
                    drafted, accepted = int(drafted), int(accepted)
                    rec["native_output_counters_available"] = True
                    rec["native_draft_tok_last"] = drafted
                    rec["native_accept_tok_last"] = accepted
                    if drafted > 0:
                        for key, value in (
                            ("native_reported_rounds", 1),
                            ("native_verified_candidates_plus_bonus", drafted),
                            ("native_accepted_candidates_plus_bonus", accepted),
                        ):
                            rec[key] = rec.get(key, 0) + value
                        rec["native_verified_draft_tokens"] = max(
                            0, rec["native_verified_candidates_plus_bonus"]
                            - rec["native_reported_rounds"])
                        rec["native_accepted_draft_tokens"] = max(
                            0, rec["native_accepted_candidates_plus_bonus"]
                            - rec["native_reported_rounds"])
                        goldens = rec.setdefault("native_golden_outputs", [])
                        if len(goldens) < 256:
                            goldens.append({
                                "num_output_tokens": int(out.num_output_tokens),
                                "draft_tok": drafted,
                                "accept_tok": accepted,
                                "origin_step": origin_step,
                                "delivery_step": step_id,
                                "postprocess_output_wall": now,
                            })
                rec["last_delivery_step"] = step_id
                rec["last_delivery_wall"] = now
                if first and rec.get("first_delivery_step") is None:
                    rec["first_delivery_step"] = step_id
                    rec["first_delivery_wall"] = now
                    if rec.get("last_prompt_end_wall") is not None:
                        rec["first_delivery_after_last_prompt_s"] = (
                            now - rec["last_prompt_end_wall"])
            payload = {
                "kind": "first_token" if first else "token",
                "rank": rank, "request_id": rid,
                "token_id": int(out.token_id),
                "finished": bool(out.finished),
                "finish_reason": str(out.finish_reason) if out.finished else None,
                "num_output_tokens": int(out.num_output_tokens),
                "prompt_len": int(out.prompt_len),
                "delivery_step_id": step_id,
                "output_origin_step_id": origin_step,
                "origin_known": origin_step is not None,
                "native_draft_tok": int(drafted or 0),
                "native_accept_tok": int(accepted or 0),
                "step": dict(last_meta),
                "stats": dict(stats),
            }
            _send(events, payload)
            if out.finished:
                rec = phase_records.get(rid)
                if rec is not None:
                    rec["finished"] = True
                    rec["finished_wall"] = now
                    rec["output_tokens"] = int(out.num_output_tokens)
                    if rec.get("first_delivery_wall") is not None:
                        rec["server_ttft_s"] = (rec["first_delivery_wall"]
                                                - rec["added_wall"])
                    if rec.get("first_delivery_wall") is not None:
                        rec["server_e2e_s"] = now - rec["added_wall"]
                local_active.discard(rid)
                phase_report(rid, "finished")

    def add_job(job: dict) -> None:
        rid = job["request_id"]
        if rid in local_active:
            _send(events, {"kind": "add_error", "rank": rank, "request_id": rid,
                           "error": "request id is already active on this rank"})
            return
        # A later window may reuse the same logical/request id.  Do not let
        # first-token classification or the phase map leak from the old one.
        first_seen.pop(rid, None)
        phase_records.pop(rid, None)
        params = SamplingParams(
            temperature=float(job.get("temperature", 0.0)),
            top_p=float(job.get("top_p", 1.0)),
            top_k=int(job.get("top_k", -1)),
            ignore_eos=bool(job.get("ignore_eos", False)),
            max_tokens=int(job["max_tokens"]),
            seq_seed=job.get("seed"),
        )
        import numpy as np
        ids = np.asarray(job["prompt_token_ids"], dtype=np.int32)
        local_active.add(rid)
        add_span = e33.begin_event("engine_add_request", request_id=rid,
                                   source="engine.add_request",
                                   extra={"prompt_len": int(len(ids)),
                                          "max_tokens": int(job["max_tokens"])})
        try:
            seq = engine.add_request(
                user_id=str(job.get("user", "adapter")),
                request_id=rid,
                sampling_params=params,
                prompt_token_ids=ids,
                arrival_time=job.get("arrival_time"),
            )
        except BaseException as exc:  # noqa: BLE001
            e33.end_event(add_span, exc)
            local_active.discard(rid)
            _send(events, {"kind": "add_error", "rank": rank, "request_id": rid,
                           "error": traceback.format_exc()})
            return
        if seq is None:
            e33.end_event(add_span, extra={"result": "none"})
            local_active.discard(rid)
            _send(events, {"kind": "add_error", "rank": rank, "request_id": rid,
                           "error": "add_request returned None (invalid token ids?)"})
            return
        e33.end_event(add_span, extra={
            "cache_hit_tokens": int(seq.cached_hit_tokens),
            "decode_step_at_add": int(engine.scheduler.decode_step)})
        phase_records[rid] = {
            "request_id": rid,
            "rank": rank,
            "logical_id": job.get("logical_id") or job.get("user"),
            "prompt_len": int(len(ids)),
            "max_tokens": int(job["max_tokens"]),
            "http_t0": job.get("http_t0"),
            "added_wall": time.time(),
            "add_step": int(stats["steps"]),
            "decode_step_at_add": int(engine.scheduler.decode_step),
            "placeholder_steps": 0,
            "placeholder_wall_s": 0.0,
            "inter_chunk_placeholder_steps": 0,
            "inter_chunk_placeholder_wall_s": 0.0,
            "prompt_steps": [],
            "decode_steps": [],
            "first_prompt_step": None,
            "first_prompt_wall": None,
            "first_prompt_start_wall": None,
            "first_prompt_end_wall": None,
            "last_prompt_end_wall": None,
            "first_delivery_step": None,
            "first_delivery_wall": None,
            "last_delivery_step": None,
            "finished": False,
            "graph_baseline": {
                "main": graph_stats["main_replays"],
                "dflash": graph_stats["dflash_replays"],
                "main_sizes": dict(graph_stats["main_graph_sizes"]),
                "dflash_sizes": dict(graph_stats["dflash_graph_sizes"]),
            },
        }
        _send(events, {
            "kind": "added", "rank": rank, "request_id": rid,
            "cache_hit_tokens": int(seq.cached_hit_tokens),
            "resolved_sampling": {
                "temperature": params.temperature,
                "top_p": params.top_p,
                "top_k": params.top_k,
                "ignore_eos": params.ignore_eos,
                "max_tokens": params.max_tokens,
                "seq_seed": params.seq_seed,
                "force_topk": int(os.environ.get("DEEPINFER_SAMPLER_FORCE_TOPK", "0")),
            },
            "add_step": stats["steps"],
            "decode_step_at_add": int(engine.scheduler.decode_step),
        })

    def abort_job(rid: str) -> None:
        try:
            n, toks = engine.abort_request(rid)
        except BaseException as exc:  # noqa: BLE001
            _send(events, {"kind": "abort_error", "rank": rank, "request_id": rid,
                           "error": repr(exc)})
            return
        local_active.discard(rid)
        _send(events, {"kind": "aborted", "rank": rank, "request_id": rid,
                       "aborted_sequences": n, "aborted_tokens": toks,
                       "step": stats["steps"]})
        phase_report(rid, "abort")

    stop_requested = False
    stop_vote_cast = False

    def note_stop() -> None:
        nonlocal stop_requested, stop_vote_cast
        stop_requested = True
        if not stop_vote_cast:
            stop_vote_cast = True
            with stop_votes.get_lock():
                stop_votes.value += 1

    prev_type = None
    current_step_span = None
    try:
        while True:
            msgs = []
            while True:
                try:
                    msgs.append(inbox.get_nowait())
                except queue.Empty:
                    break
            for msg in msgs:
                if e33.handle_cmd(msg):
                    continue
                cmd = msg.get("cmd")
                if cmd == "generate":
                    add_job(msg["job"])
                elif cmd == "abort":
                    abort_job(msg["request_id"])
                elif cmd == "wake":
                    pass
                elif cmd == "stop":
                    note_stop()

            unfinished = engine.has_unfinished_requests()
            if not unfinished:
                raw = engine.flush_inflight_reqs()
                stats["flushes"] += 1
                handle_outputs(raw, stats["steps"])
                # All eight ranks must finish the flush collective before any
                # of them can re-enter has_unfinished_requests.  Without this
                # barrier a rank that receives a new job during the idle
                # transition could enter the next all_reduce while a peer is
                # still inside flush, mixing collective operations.
                idle_barrier.wait()
                # A rank may have received stop while a peer has not.  Only
                # exit when all eight have voted and all are past the common
                # idle point; otherwise one rank would close while a peer is
                # entering the next has_unfinished_requests collective.
                if stop_requested and stop_votes.value >= DEVICES:
                    _send(events, {"kind": "stopping", "rank": rank,
                                   "step": stats["steps"]})
                    break
                _send(events, {
                    "kind": "idle", "rank": rank, "step": stats["steps"],
                    "stats": dict(stats), "active": sorted(local_active),
                    "decode_step": int(engine.scheduler.decode_step),
                })
                if stop_requested and stop_votes.value < DEVICES:
                    # A stop was consumed before its peers.  Re-enter the
                    # collective loop instead of blocking on an empty inbox;
                    # peers with queued stop messages will join the next
                    # has_unfinished_requests all_reduce.
                    continue
                # Block until the parent wakes the whole group.  The parent
                # only does that after queueing a job, so all eight ranks
                # enter the collective `has_unfinished_requests` together.
                # E33 control commands are consumed here too; handling one
                # never enters a new collective, and `continue` goes back to
                # the top so every rank reaches the same all_reduce.
                msg = inbox.get()
                if e33.handle_cmd(msg):
                    continue
                cmd = msg.get("cmd")
                if cmd == "generate":
                    add_job(msg["job"])
                elif cmd == "abort":
                    abort_job(msg["request_id"])
                elif cmd == "stop":
                    note_stop()
                continue

            if _REPLAY_CONTROLLER is not None:
                _REPLAY_CONTROLLER.begin_engine_step(int(stats["steps"]) + 1)
            step_iteration = int(stats["steps"]) + 1
            step_span = e33.begin_event("engine_step", iteration_id=step_iteration,
                                        source="worker.loop")
            current_step_span = step_span
            step_t0 = time.perf_counter()
            step_start_wall = time.time()
            try:
                raw = engine.step()
            except BaseException as exc:  # noqa: BLE001 - preserve the original failure
                e33.end_event(step_span, exc, extra={"step_type": "exception",
                                                     "host_span_is_device_time": False})
                raise
            step_dur = time.perf_counter() - step_t0
            step_end_wall = step_start_wall + step_dur
            stats["steps"] += 1
            stats["graph_main_replays"] = graph_stats["main_replays"]
            stats["graph_dflash_replays"] = graph_stats["dflash_replays"]
            if last_meta["is_empty"]:
                stats["placeholder_steps"] += 1
                step_type = "placeholder"
            elif last_meta["prompt_run"]:
                stats["prompt_steps"] += 1
                step_type = "prompt"
            else:
                stats["decode_steps"] += 1
                step_type = "decode"
            if local_active:
                scheduled_now = set(last_meta.get("scheduled") or [])
                for rid in list(local_active):
                    rec = phase_records.get(rid)
                    if rec is None or rec.get("finished"):
                        continue
                    if step_type == "placeholder":
                        if rec["first_prompt_step"] is None:
                            rec["placeholder_steps"] += 1
                            rec["placeholder_wall_s"] += step_dur
                        else:
                            rec["inter_chunk_placeholder_steps"] += 1
                            rec["inter_chunk_placeholder_wall_s"] += step_dur
                    elif step_type == "prompt" and rid in scheduled_now:
                        rec["prompt_steps"].append({
                            "step": stats["steps"],
                            "start_wall": step_start_wall,
                            "end_wall": step_end_wall,
                            "duration_s": step_dur,
                            "decode_step": int(last_meta.get("decode_step") or 0),
                            "num_batched_tokens": int(
                                last_meta.get("scheduled_query_tokens", {}).get(rid, -1)),
                            "batch_tokens_total": int(last_meta.get("num_batched_tokens") or 0),
                            "token_count_source": "native SequenceMetadata.seqlen_q",
                        })
                        if rec["first_prompt_step"] is None:
                            rec["first_prompt_step"] = stats["steps"]
                            rec["first_prompt_start_wall"] = step_start_wall
                            rec["first_prompt_wall"] = step_start_wall
                            rec["first_prompt_end_wall"] = step_end_wall
                        rec["last_prompt_end_wall"] = step_end_wall
                    elif step_type == "decode" and rid in scheduled_now:
                        if len(rec["decode_steps"]) < 2048:
                            rec["decode_steps"].append({
                                "step": stats["steps"],
                                "duration_s": step_dur,
                            })
            handle_outputs(raw, stats["steps"])
            e33.note_step(step_span, step_type, last_meta,
                          int(step_dur * 1e9))
            current_step_span = None
            if (cfg.trace_steps or step_type != prev_type
                    or stats["steps"] % 64 == 0):
                _send(events, {
                    "kind": "step", "rank": rank, "step": stats["steps"],
                    "type": step_type, "meta": dict(last_meta),
                    "stats": dict(stats),
                })
            prev_type = step_type
    except BaseException as exc:
        e33.end_event(current_step_span, exc)
        current_step_span = None
        tb = traceback.format_exc()
        print("[worker-loop-failed] rank=%d\n%s" % (rank, tb),
              file=sys.stderr, flush=True)
        _send(events, {"kind": "error", "rank": rank, "where": "loop",
                       "error": tb})
    finally:
        # Stop/export an active cohort before llm.close() releases the model
        # and cache structures used by the manifest snapshot.
        e33.shutdown()
        try:
            llm.close()
        except BaseException:
            _send(events, {"kind": "error", "rank": rank, "where": "close",
                           "error": traceback.format_exc()})
        for rid in list(phase_records):
            phase_report(rid, "worker_close")
        _send(events, {"kind": "closed", "rank": rank, "stats": dict(stats)})


# ---------------------------------------------------------------------------
# CPU mock worker (same message protocol; no deepinfer/torch)
# ---------------------------------------------------------------------------
def mock_worker(rank: int, cfg: ClusterConfig, inbox: mp.Queue, events: mp.Queue,
                global_active, lock, stop_votes) -> None:
    """Deterministic fake rank for protocol tests.

    Every request produces `max_tokens` tokens, one per step.  A rank with no
    local request still emits a placeholder step while any rank is active, so
    the parent sees the same collective-step shape as the real engine.
    """
    _send(events, {"kind": "ready", "rank": rank, "load_seconds": 0.0,
                   "resolved": {"mock": True, "zos": cfg.zos, "nextn": cfg.nextn}})
    stats = {"steps": 0, "placeholder_steps": 0, "prompt_steps": 0,
             "decode_steps": 0, "tokens_emitted": 0, "flushes": 0}
    local: dict[str, dict] = {}
    mock_e33 = MockE33Rank(
        rank, cfg, events, stats,
        idle_fn=lambda: (not local and global_active.value == 0))
    stop_requested = False
    stop_vote_cast = False

    def note_stop() -> None:
        nonlocal stop_requested, stop_vote_cast
        stop_requested = True
        if not stop_vote_cast:
            stop_vote_cast = True
            with stop_votes.get_lock():
                stop_votes.value += 1

    prev_type = None
    try:
        while True:
            msgs = []
            while True:
                try:
                    msgs.append(inbox.get_nowait())
                except queue.Empty:
                    break
            for msg in msgs:
                if mock_e33.handle_cmd(msg):
                    continue
                cmd = msg.get("cmd")
                if cmd == "generate":
                    job = msg["job"]
                    local[job["request_id"]] = {
                        "job": job, "n": 0, "max_tokens": job["max_tokens"],
                        "first": True,
                    }
                    with lock:
                        global_active.value += 1
                    _send(events, {"kind": "added", "rank": rank,
                                   "request_id": job["request_id"],
                                   "resolved_sampling": {"mock": True},
                                   "add_step": stats["steps"],
                                   "decode_step_at_add": stats["steps"]})
                elif cmd == "abort":
                    rid = msg["request_id"]
                    if rid in local:
                        del local[rid]
                        with lock:
                            global_active.value -= 1
                    _send(events, {"kind": "aborted", "rank": rank,
                                   "request_id": rid, "step": stats["steps"]})
                elif cmd == "wake":
                    pass
                elif cmd == "stop":
                    note_stop()

            with lock:
                active = global_active.value > 0
            if not active:
                if stop_requested and stop_votes.value >= DEVICES:
                    _send(events, {"kind": "stopping", "rank": rank,
                                   "step": stats["steps"]})
                    break
                _send(events, {"kind": "idle", "rank": rank, "step": stats["steps"],
                               "stats": dict(stats), "active": sorted(local),
                               "decode_step": stats["steps"]})
                if stop_requested and stop_votes.value < DEVICES:
                    time.sleep(0.01)
                    continue
                msg = inbox.get()
                if mock_e33.handle_cmd(msg):
                    continue
                cmd = msg.get("cmd")
                if cmd == "generate":
                    job = msg["job"]
                    local[job["request_id"]] = {
                        "job": job, "n": 0, "max_tokens": job["max_tokens"],
                        "first": True,
                    }
                    with lock:
                        global_active.value += 1
                    _send(events, {"kind": "added", "rank": rank,
                                   "request_id": job["request_id"],
                                   "resolved_sampling": {"mock": True},
                                   "add_step": stats["steps"],
                                   "decode_step_at_add": stats["steps"]})
                elif cmd == "abort":
                    _send(events, {"kind": "aborted", "rank": rank,
                                   "request_id": msg["request_id"], "step": stats["steps"]})
                elif cmd == "stop":
                    note_stop()
                continue

            stats["steps"] += 1
            if local:
                stats["decode_steps"] += 1
                for rid in list(local):
                    st = local[rid]
                    st["n"] += 1
                    stats["tokens_emitted"] += 1
                    finished = st["n"] >= st["max_tokens"]
                    _send(events, {
                        "kind": "first_token" if st["first"] else "token",
                        "rank": rank, "request_id": rid,
                        "token_id": 1000 + (st["n"] % 100),
                        "finished": finished,
                        "finish_reason": "length" if finished else None,
                        "num_output_tokens": st["n"],
                        "prompt_len": len(st["job"]["prompt_token_ids"]),
                        "delivery_step_id": stats["steps"],
                        "output_origin_step_id": stats["steps"],
                        "origin_known": True,
                        "step": {"is_empty": False, "prompt_run": st["first"],
                                 "decode_step": stats["steps"],
                                 "scheduled": [rid], "num_batched_tokens": 1},
                        "stats": dict(stats),
                    })
                    st["first"] = False
                    if finished:
                        del local[rid]
                        with lock:
                            global_active.value -= 1
            else:
                stats["placeholder_steps"] += 1
                _send(events, {
                    "kind": "step", "rank": rank, "step": stats["steps"],
                    "meta": {"is_empty": True, "prompt_run": False,
                             "decode_step": stats["steps"], "scheduled": [],
                             "num_batched_tokens": 0},
                    "stats": dict(stats),
                })
            time.sleep(0.002)
    except BaseException:
        _send(events, {"kind": "error", "rank": rank, "where": "mock",
                       "error": traceback.format_exc()})
    finally:
        _send(events, {"kind": "closed", "rank": rank, "stats": dict(stats)})


# ---------------------------------------------------------------------------
# request state in the parent
# ---------------------------------------------------------------------------
class RequestState:
    def __init__(self, job: Job):
        self.job = job
        self.request_id = job.request_id
        self.rank: int | None = job.rank
        self.assigned_at: float | None = None
        self.added_at: float | None = None
        self.first_at: float | None = None
        self.terminal_at: float | None = None
        self.token_ids: list[int] = []
        self.chunks: list[dict] = []
        self.finish_reason: str | None = None
        self.error: str | None = None
        self.cancelled = False
        self.done = threading.Event()
        self.cond = threading.Condition()
        self.resolved_sampling: dict | None = None
        self.decode_step_at_add: int | None = None
        self.origin_steps: list = []
        self.delivery_steps: list[int] = []
        self.native_usage = {
            "reported_rounds": 0,
            "verified_candidates_plus_bonus": 0,
            "accepted_candidates_plus_bonus": 0,
            "verified_draft_tokens": 0,
            "accepted_draft_tokens": 0,
        }
        self.native_golden_outputs: list[dict] = []
        self.closed_at: float | None = None

    def add_chunk(self, token_id: int, origin_step, delivery_step: int,
                  origin_known: bool) -> None:
        with self.cond:
            self.token_ids.append(token_id)
            self.origin_steps.append(origin_step)
            self.delivery_steps.append(delivery_step)
            if self.first_at is None:
                self.first_at = time.time()
            self.chunks.append({"token_id": token_id, "origin_step": origin_step,
                                "delivery_step": delivery_step,
                                "origin_known": origin_known})
            self.cond.notify_all()

    def finish(self, reason: str) -> None:
        with self.cond:
            if self.terminal_at is not None:
                return
            self.finish_reason = reason
            self.terminal_at = time.time()
            self.done.set()
            self.cond.notify_all()


# ---------------------------------------------------------------------------
# cluster
# ---------------------------------------------------------------------------
class Cluster:
    def __init__(self, cfg: ClusterConfig, mock: bool = False):
        self.cfg = cfg
        self.mock = mock
        self.ctx = mp.get_context("spawn")
        self.events: mp.Queue = self.ctx.Queue()
        self.inboxes: list[mp.Queue] = [self.ctx.Queue() for _ in range(DEVICES)]
        self.workers: list[mp.Process] = []
        self.lock = threading.RLock()
        self.ready = [False] * DEVICES
        self.rank_ready_at: list[float | None] = [None] * DEVICES
        self.closed = [False] * DEVICES
        self.close_times: dict[int, float] = {}
        self.stop_sent_times: dict[int, float] = {}
        self.stopping: list[int] = []
        self.ready_event = threading.Event()
        self.failed_event = threading.Event()
        self.fatal_detail: dict | None = None
        self.idle_ready = threading.Event()
        self.idle = [False] * DEVICES
        self.server_idle = False
        self.idle_epoch = 0
        self.requests: dict[str, RequestState] = {}
        self.pending_dispatch: list[RequestState] = []
        self.rank_active: list[set[str]] = [set() for _ in range(DEVICES)]
        self.rank_stats: list[dict] = [{} for _ in range(DEVICES)]
        self.rank_last_step: list[dict] = [{} for _ in range(DEVICES)]
        self.rank_resolved: list[dict] = [{} for _ in range(DEVICES)]
        self.next_rank = 0
        self.profile_status: list[dict] = [{} for _ in range(DEVICES)]
        self.profile_armed_ids: set[str] = set()
        self.profile_dispatch_count = 0
        self.request_phases: dict[str, dict] = {}
        self.fatal: str | None = None
        self.diagnostics: list[dict] = []
        self._diag_path: Path | None = None
        # E33 control plane.  All mutations happen under self.lock; the ack
        # condition shares that RLock and is waited on only after releasing it.
        self.e33_task_root = Path(cfg.e33_task_root).resolve()
        self.e33_busy = False
        self.e33_session: dict | None = None
        self.e33_last_session: dict | None = None
        self.e33_op: dict | None = None
        self.e33_acks: dict[int, dict] = {}
        self.e33_ack_cond = threading.Condition(self.lock)
        self.e33_rank_status: list[dict] = [{} for _ in range(DEVICES)]
        self.e33_errors: list[dict] = []
        self.e33_telemetry: E33Telemetry | None = None
        self.http_inflight = 0
        self.http_requests_total = 0
        self._router = threading.Thread(target=self._route, daemon=True)
        self._stop = False
        if self.mock:
            self._global_active = self.ctx.Value("i", 0)
            self._global_lock = self.ctx.Lock()
        self._idle_barrier = self.ctx.Barrier(DEVICES)
        self._stop_barrier = self.ctx.Barrier(DEVICES)
        self._stop_votes = self.ctx.Value("i", 0)
        self._loop_barrier = self.ctx.Barrier(DEVICES)

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> bool:
        for rank in range(DEVICES):
            if self.mock:
                args = (rank, self.cfg, self.inboxes[rank], self.events,
                        self._global_active, self._global_lock,
                        self._stop_votes)
                target = mock_worker
            else:
                args = (rank, self.cfg, self.inboxes[rank], self.events,
                        self._idle_barrier, self._stop_barrier, self._stop_votes)
                target = real_worker
            proc = self.ctx.Process(target=target, args=args,
                                    name=f"dp8-rank{rank}")
            proc.start()
            self.workers.append(proc)
        self._router.start()
        deadline = time.monotonic() + self.cfg.start_timeout_s
        while True:
            if self.ready_event.wait(0.2):
                break
            if self.failed_event.is_set():
                break
            dead = [i for i, proc in enumerate(self.workers)
                    if proc.exitcode is not None]
            if dead:
                self.fatal = "worker(s) exited during startup: %r" % dead
                self.fatal_detail = self.fatal_detail or {
                    "where": "startup_exit", "ranks": dead}
                break
            if time.monotonic() >= deadline:
                self.fatal = "startup timeout waiting for 8 ranks"
                break
        if self.fatal is not None:
            print("[fatal] %s detail=%r" % (self.fatal, self.fatal_detail),
                  file=sys.stderr, flush=True)
            return False
        # Wait for the initial global-idle report so the HTTP server never
        # dispatches during the first flush/barrier transition.
        idle_deadline = time.monotonic() + self.cfg.start_timeout_s
        while not self.idle_ready.wait(0.2):
            if self.failed_event.is_set() or time.monotonic() >= idle_deadline:
                break
        return self.fatal is None

    def _route(self) -> None:
        while not self._stop:
            try:
                ev = self.events.get(timeout=0.5)
            except queue.Empty:
                continue
            kind = ev.get("kind")
            rank = ev.get("rank")
            if kind == "ready":
                with self.lock:
                    self.ready[rank] = True
                    if self.rank_ready_at[rank] is None:
                        self.rank_ready_at[rank] = time.time()
                    self.rank_resolved[rank] = ev.get("resolved", {})
                    if all(self.ready):
                        self.ready_event.set()
            elif kind == "added":
                with self.lock:
                    st = self.requests.get(ev["request_id"])
                    if st is not None:
                        st.added_at = time.time()
                        st.resolved_sampling = ev.get("resolved_sampling")
                        st.decode_step_at_add = ev.get("decode_step_at_add")
                        st.cache_hit_tokens = int(ev.get("cache_hit_tokens") or 0)
                    if rank is not None:
                        self.rank_active[rank].add(ev["request_id"])
                    self.idle[rank] = False
                    self.server_idle = False
            elif kind in ("first_token", "token"):
                rid = ev["request_id"]
                with self.lock:
                    st = self.requests.get(rid)
                    if st is not None and not st.cancelled:
                        st.add_chunk(int(ev["token_id"]),
                                     ev.get("output_origin_step_id"),
                                     int(ev.get("delivery_step_id", ev.get("origin_step_id"))),
                                     bool(ev.get("origin_known", False)))
                        drafted = int(ev.get("native_draft_tok") or 0)
                        accepted = int(ev.get("native_accept_tok") or 0)
                        if drafted > 0:
                            nu = st.native_usage
                            nu["reported_rounds"] += 1
                            nu["verified_candidates_plus_bonus"] += drafted
                            nu["accepted_candidates_plus_bonus"] += accepted
                            nu["verified_draft_tokens"] = max(
                                0, nu["verified_candidates_plus_bonus"]
                                - nu["reported_rounds"])
                            nu["accepted_draft_tokens"] = max(
                                0, nu["accepted_candidates_plus_bonus"]
                                - nu["reported_rounds"])
                            if len(st.native_golden_outputs) < 256:
                                st.native_golden_outputs.append({
                                    "token_id": int(ev["token_id"]),
                                    "draft_tok": drafted,
                                    "accept_tok": accepted,
                                    "origin_step": ev.get("output_origin_step_id"),
                                    "delivery_step": ev.get("delivery_step_id"),
                                })
                        if ev.get("finished"):
                            st.finish(ev.get("finish_reason") or "stop")
                if ev.get("finished"):
                    self._on_terminal(rid, ev.get("finish_reason") or "stop")
            elif kind == "add_error":
                rid = ev["request_id"]
                with self.lock:
                    st = self.requests.get(rid)
                    if st is not None:
                        st.error = ev.get("error")
                        st.finish("error")
                self._on_terminal(rid, "error")
            elif kind == "aborted":
                rid = ev["request_id"]
                with self.lock:
                    st = self.requests.get(rid)
                    if st is not None:
                        st.cancelled = True
                        st.finish("abort")
                self._on_terminal(rid, "abort")
            elif kind == "idle":
                with self.lock:
                    self.idle[rank] = True
                    self.rank_stats[rank] = ev.get("stats", {})
                    self.rank_last_step[rank] = {
                        "step": ev.get("step"), "active": ev.get("active", []),
                        "decode_step": ev.get("decode_step"),
                    }
                    self._mark_idle_locked()
            elif kind == "step":
                with self.lock:
                    self.rank_stats[rank] = ev.get("stats", {})
                    self.rank_last_step[rank] = {
                        "step": ev.get("step"), "meta": ev.get("meta", {}),
                    }
                    if self._diag_path is not None:
                        self.diagnostics.append(ev)
            elif kind == "profile_ready":
                with self.lock:
                    self.profile_status[rank] = {
                        "rank": rank, "plan": ev.get("plan"), "armed": False}
            elif kind == "profile_armed":
                with self.lock:
                    self.profile_status[rank].update({
                        "rank": rank, "armed": True,
                        "request_ids": ev.get("request_ids", [])})
            elif kind == "profile_status":
                with self.lock:
                    self.profile_status[rank] = ev.get("status", {})
            elif kind == "request_phase":
                rec = ev.get("record") or {}
                rid = rec.get("request_id") or ("rank%02d-unknown" % (rank or 0))
                with self.lock:
                    self.request_phases[rid] = rec
                path = self.cfg.diag_requests_path
                if path:
                    try:
                        Path(path).parent.mkdir(parents=True, exist_ok=True)
                        with open(path, "a") as fh:
                            fh.write(json.dumps(
                                {"captured_wall": time.time(), **rec},
                                sort_keys=True) + "\n")
                    except OSError:
                        pass
            elif kind in ("e33_start_ack", "e33_stop_ack", "e33_snapshot_ack",
                          "e33_error"):
                if not isinstance(rank, int) or not 0 <= rank < DEVICES:
                    with self.lock:
                        self.e33_errors.append({
                            "where": "router", "error": "invalid rank in E33 ack",
                            "kind": kind, "rank": rank, "wall": time.time()})
                    continue
                with self.lock:
                    self.e33_rank_status[rank] = {
                        "rank": rank,
                        "last_kind": kind,
                        "op_id": ev.get("op_id"),
                        "ok": ev.get("ok"),
                        "state": ev.get("state"),
                        "identity": ev.get("identity") or self.e33_rank_status[rank].get("identity"),
                        "profiler": ev.get("profiler") or self.e33_rank_status[rank].get("profiler"),
                        "snapshot": ev.get("snapshot") or self.e33_rank_status[rank].get("snapshot"),
                        "telemetry": ev.get("telemetry") or self.e33_rank_status[rank].get("telemetry"),
                        "manifest": ev.get("manifest"),
                        "errors": ev.get("errors") or [],
                        "where": ev.get("where"),
                        "error": ev.get("error"),
                        "worker_steps": ev.get("worker_steps"),
                        "host": ev.get("host"),
                        "updated_wall": time.time(),
                    }
                    if (self.e33_op is not None
                            and ev.get("op_id") == self.e33_op.get("op_id")):
                        self.e33_acks[rank] = ev
                        self.e33_ack_cond.notify_all()
            elif kind == "error":
                detail = {"rank": rank, "where": ev.get("where"),
                          "error": ev.get("error", "worker error")}
                print("[fatal] rank=%r where=%r\n%s"
                      % (rank, detail["where"], detail["error"]),
                      file=sys.stderr, flush=True)
                with self.lock:
                    self.fatal = detail["error"]
                    self.fatal_detail = detail
                self.failed_event.set()
                self.ready_event.set()
            elif kind == "closed":
                with self.lock:
                    self.closed[rank] = True
                    self.close_times[rank] = time.time()
            elif kind == "stopping":
                with self.lock:
                    self.stopping.append(rank)
            if self._diag_path is not None and kind != "step":
                self.diagnostics.append(ev)

    def _mark_idle_locked(self) -> None:
        # The last terminal event can arrive after the last rank's idle
        # report (events from different ranks are not ordered).  Recompute
        # quiescence on every terminal as well, otherwise /readyz can stay
        # non-idle forever after a batch.
        if all(self.idle) and not any(self.rank_active) and not self.pending_dispatch:
            if not self.server_idle:
                self.server_idle = True
                self.idle_epoch += 1
            self.idle_ready.set()

    def _on_terminal(self, rid: str, reason: str) -> None:
        with self.lock:
            st = self.requests.get(rid)
            if st is not None:
                st.finish(reason)
                if st.rank is not None:
                    self.rank_active[st.rank].discard(rid)
            self._dispatch_locked()
            self._mark_idle_locked()

    # -- dispatch ----------------------------------------------------------
    def _maybe_arm_profile_locked(self, states: list["RequestState"]) -> None:
        """Arm every rank when a planned request is dispatched.

        The command reaches non-owning ranks too, so placeholder prompt/decode
        work is captured by the same phase trigger on all eight workers.
        """
        plan = self.cfg.profile
        if plan is None:
            return
        hits = []
        for st in states:
            self.profile_dispatch_count += 1
            if st.request_id in plan.request_ids:
                hits.append(st.request_id)
            elif (plan.arm_after_dispatched
                  and self.profile_dispatch_count == plan.arm_after_dispatched):
                hits.append(st.request_id)
        hits = [rid for rid in hits if rid not in self.profile_armed_ids]
        if not hits:
            return
        self.profile_armed_ids.update(hits)
        for inbox in self.inboxes:
            inbox.put({"cmd": "profile_arm", "request_ids": hits})

    def _pick_rank_locked(self) -> int | None:
        for offset in range(DEVICES):
            rank = (self.next_rank + offset) % DEVICES
            if len(self.rank_active[rank]) < self.cfg.max_num_seqs:
                self.next_rank = (rank + 1) % DEVICES
                return rank
        return None

    def _dispatch_locked(self) -> None:
        while self.pending_dispatch:
            st = self.pending_dispatch[0]
            if st.job.rank is not None:
                rank = st.job.rank
                if len(self.rank_active[rank]) >= self.cfg.max_num_seqs:
                    return
            else:
                rank = self._pick_rank_locked()
                if rank is None:
                    return
            self.pending_dispatch.pop(0)
            st.rank = rank
            st.assigned_at = time.time()
            self.rank_active[rank].add(st.request_id)
            job = {
                "request_id": st.request_id,
                "prompt_token_ids": st.job.prompt_token_ids,
                "max_tokens": st.job.max_tokens,
                "temperature": st.job.temperature,
                "top_p": st.job.top_p,
                "top_k": st.job.top_k,
                "ignore_eos": st.job.ignore_eos,
                "seed": st.job.seed,
                "user": st.job.meta.get("user", "adapter"),
                "logical_id": st.job.meta.get("logical_id"),
                "http_t0": st.job.meta.get("http_t0"),
                "arrival_time": st.job.arrival_ts,
            }
            self._maybe_arm_profile_locked([st])
            self.inboxes[rank].put({"cmd": "generate", "job": job})
            # Always wake the group: harmless when it is already stepping and
            # required when it is blocked in the idle handshake.
            for inbox in self.inboxes:
                inbox.put({"cmd": "wake"})
            self.server_idle = False
            self.idle = [False] * DEVICES

    def submit(self, job: Job) -> RequestState:
        st = RequestState(job)
        with self.lock:
            if self.e33_busy:
                raise E33ControlBusy("E33 control operation in flight; retry")
            self.requests[st.request_id] = st
            self.pending_dispatch.append(st)
            self._dispatch_locked()
        return st

    def submit_many(self, jobs: list[Job]) -> list[RequestState]:
        """Assign a group of jobs to free ranks, then wake the group once.

        Used by the native batch endpoint so the first schedule sees the whole
        batch (the clean C8 dense case).  Jobs that do not fit are queued.
        """
        states = []
        with self.lock:
            if self.e33_busy:
                raise E33ControlBusy("E33 control operation in flight; retry")
            for job in jobs:
                st = RequestState(job)
                self.requests[st.request_id] = st
                rank = job.rank if job.rank is not None else self._pick_rank_locked()
                if rank is None:
                    self.pending_dispatch.append(st)
                    states.append(st)
                    continue
                st.rank = rank
                st.assigned_at = time.time()
                self.rank_active[rank].add(st.request_id)
                self.inboxes[rank].put({"cmd": "generate", "job": {
                    "request_id": st.request_id,
                    "prompt_token_ids": st.job.prompt_token_ids,
                    "max_tokens": st.job.max_tokens,
                    "temperature": st.job.temperature,
                    "top_p": st.job.top_p,
                    "top_k": st.job.top_k,
                    "ignore_eos": st.job.ignore_eos,
                    "seed": st.job.seed,
                    "user": st.job.meta.get("user", "adapter"),
                    "arrival_time": st.job.arrival_ts,
                }})
                states.append(st)
            self._maybe_arm_profile_locked(states)
            for inbox in self.inboxes:
                inbox.put({"cmd": "wake"})
            self.server_idle = False
            self.idle = [False] * DEVICES
        return states

    def submit_batch(self, jobs: list[Job]) -> list[RequestState]:
        """Queue a whole batch before waking the group.

        Used by the offline diagnostic so all eight ranks enter the first
        collective schedule with their request already present.
        """
        states = []
        with self.lock:
            if self.e33_busy:
                raise E33ControlBusy("E33 control operation in flight; retry")
            for job in jobs:
                st = RequestState(job)
                self.requests[st.request_id] = st
                rank = job.rank
                st.rank = rank
                st.assigned_at = time.time()
                self.rank_active[rank].add(st.request_id)
                payload = {
                    "request_id": st.request_id,
                    "prompt_token_ids": st.job.prompt_token_ids,
                    "max_tokens": st.job.max_tokens,
                    "temperature": st.job.temperature,
                    "top_p": st.job.top_p,
                    "top_k": st.job.top_k,
                    "ignore_eos": st.job.ignore_eos,
                    "seed": st.job.seed,
                    "user": st.job.meta.get("user", "adapter"),
                    "arrival_time": st.job.arrival_ts,
                }
                self.inboxes[rank].put({"cmd": "generate", "job": payload})
                states.append(st)
            self._maybe_arm_profile_locked(states)
            for inbox in self.inboxes:
                inbox.put({"cmd": "wake"})
            self.server_idle = False
            self.idle = [False] * DEVICES
        return states

    def abort(self, rid: str) -> bool:
        with self.lock:
            st = self.requests.get(rid)
            if st is None:
                return False
            if st.done.is_set():
                return False
            st.cancelled = True
            rank = st.rank
        if rank is not None:
            self.inboxes[rank].put({"cmd": "abort", "request_id": rid})
        return True

    # -- E33 control plane -------------------------------------------------
    def http_enter(self) -> None:
        with self.lock:
            self.http_inflight += 1
            self.http_requests_total += 1

    def http_leave(self) -> None:
        with self.lock:
            self.http_inflight = max(0, self.http_inflight - 1)

    def _e33_idle_locked(self, ignore_busy: bool = False) -> dict:
        active = sorted({rid for ranks in self.rank_active for rid in ranks})
        pending = len(self.pending_dispatch)
        not_done = sorted(rid for rid, st in self.requests.items()
                          if not st.done.is_set())
        idle_ranks = [rank for rank in range(DEVICES) if not self.idle[rank]]
        ranks_ready = sum(self.ready)
        ok = (not self.fatal
              and ranks_ready == DEVICES
              and not active
              and pending == 0
              and not not_done
              and self.server_idle
              and not idle_ranks
              and self.http_inflight == 0
              and (ignore_busy or not self.e33_busy))
        return {
            "ok": bool(ok),
            "fatal": self.fatal,
            "ranks_ready": ranks_ready,
            "server_idle": self.server_idle,
            "idle_epoch": self.idle_epoch,
            "active_requests": active,
            "pending_dispatch": pending,
            "requests_not_done": not_done[:64],
            "requests_not_done_count": len(not_done),
            "ranks_not_idle": idle_ranks,
            "http_inflight": self.http_inflight,
            "e33_busy": self.e33_busy,
            "observed_wall_ns": time.time_ns(),
        }

    def _e33_new_op_locked(self, kind: str, ctx: dict) -> str:
        op_id = uuid.uuid4().hex
        self.e33_op = {
            "op_id": op_id,
            "kind": kind,
            "context": dict(ctx),
            "started_wall_ns": time.time_ns(),
        }
        self.e33_acks = {}
        self.e33_busy = True
        return op_id

    def _e33_broadcast_locked(self, cmd: str, op_id: str, ctx: dict) -> list:
        payload = {"cmd": cmd, "op_id": op_id}
        payload.update(ctx)
        errors = []
        for rank, inbox in enumerate(self.inboxes):
            try:
                inbox.put(payload)
            except (OSError, ValueError) as exc:
                errors.append({"rank": rank, "error": repr(exc)})
        if errors:
            self.e33_errors.append({"where": "broadcast:" + cmd,
                                    "errors": errors, "wall": time.time()})
        return errors

    def _e33_wait_acks(self, op_id: str, timeout_s: float) -> dict:
        deadline = time.monotonic() + max(1.0, float(timeout_s))
        with self.e33_ack_cond:
            while True:
                if self.fatal:
                    raise E33ControlError(
                        "fatal server state during E33 control: %s" % self.fatal,
                        status=503, detail={"fatal": self.fatal,
                                            "acks": dict(self.e33_acks)})
                if len(self.e33_acks) >= DEVICES:
                    return dict(self.e33_acks)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    missing = [rank for rank in range(DEVICES)
                               if rank not in self.e33_acks]
                    raise E33ControlError(
                        "E33 %s ack timeout; missing ranks %r"
                        % (self.e33_op.get("kind") if self.e33_op else "?",
                           missing),
                        status=504,
                        detail={"missing_ranks": missing,
                                "acks": dict(self.e33_acks),
                                "timeout_s": timeout_s})
                self.e33_ack_cond.wait(min(remaining, 0.5))

    def _e33_rank_rows_locked(self, acks: dict, expected_kind: str) -> list:
        rows = []
        for rank in range(DEVICES):
            ack = acks.get(rank) or {}
            identity = ack.get("identity") or {}
            rows.append({
                "rank": rank,
                "ack_kind": ack.get("kind"),
                "ok": ack.get("ok"),
                "worker_pid_expected": (self.workers[rank].pid
                                        if rank < len(self.workers) else None),
                "pid": identity.get("pid"),
                "identity_rank": identity.get("rank"),
                "logical_device": identity.get("logical_device"),
                "physical_card": identity.get("physical_card"),
                "visible_device_ids": identity.get("visible_device_ids"),
                "physical_device_id": identity.get("physical_device_id"),
                "synthetic": identity.get("synthetic"),
                "identity": identity,
                "state": ack.get("state"),
                "profiler": ack.get("profiler"),
                "snapshot": ack.get("snapshot"),
                "telemetry": ack.get("telemetry"),
                "manifest": ack.get("manifest"),
                "export_ok": ack.get("export_ok"),
                "trace_present": ack.get("trace_present"),
                "errors": ack.get("errors") or [],
                "where": ack.get("where"),
                "error": ack.get("error"),
                "worker_steps": ack.get("worker_steps"),
                "host": ack.get("host"),
                "expected_kind": expected_kind,
            })
        return rows

    def _e33_validate_start_rows(self, acks: dict) -> tuple:
        rows = self._e33_rank_rows_locked(acks, "e33_start_ack")
        problems = []
        physical_devices = []
        logical_devices = []
        pids = []
        for row in rows:
            rank = row["rank"]
            if rank >= len(self.workers) or not self.workers[rank].is_alive():
                problems.append("rank%d worker process is not alive" % rank)
            if row["ack_kind"] != "e33_start_ack" or row["ok"] is not True:
                problems.append("rank%d ack failed: %r %r"
                                % (rank, row["ack_kind"], row["error"]))
                continue
            identity = row["identity"]
            if not identity:
                problems.append("rank%d has no identity" % rank)
                continue
            if (row.get("profiler") or {}).get("started") is not True:
                problems.append("rank%d profiler did not report started" % rank)
            if row.get("state") != "collecting":
                problems.append("rank%d state=%r, expected collecting"
                                % (rank, row.get("state")))
            if int(identity.get("rank", -1)) != rank:
                problems.append("rank%d identity rank=%r" % (rank, identity.get("rank")))
            if int(identity.get("world_size", -1)) != DEVICES:
                problems.append("rank%d world_size=%r, expected %d"
                                % (rank, identity.get("world_size"), DEVICES))
            if int(identity.get("ep_size", -1)) != int(self.cfg.ep_size):
                problems.append("rank%d ep_size=%r, expected %r"
                                % (rank, identity.get("ep_size"),
                                   self.cfg.ep_size))
            if int(identity.get("cp_size", -1)) != 1:
                problems.append("rank%d cp_size=%r, expected 1"
                                % (rank, identity.get("cp_size")))
            if str(identity.get("e33_task_root")) != str(self.e33_task_root):
                problems.append("rank%d e33_task_root=%r != parent %r"
                                % (rank, identity.get("e33_task_root"),
                                   str(self.e33_task_root)))
            if row["pid"] is None:
                problems.append("rank%d has no pid" % rank)
            else:
                pids.append(int(row["pid"]))
                expected = row["worker_pid_expected"]
                if expected is not None and int(row["pid"]) != int(expected):
                    problems.append("rank%d pid %r != parent process pid %r"
                                    % (rank, row["pid"], expected))
            if row["logical_device"] is None:
                problems.append("rank%d has no logical_device" % rank)
            else:
                logical_devices.append(int(row["logical_device"]))
            if row["physical_device_id"] is None:
                problems.append(
                    "rank%d has no physical_device_id from "
                    "ASCEND_RT_VISIBLE_DEVICES/CUDA_VISIBLE_DEVICES (%r)"
                    % (rank, row.get("physical_card")))
            else:
                physical_devices.append(int(row["physical_device_id"]))
        if len(physical_devices) == DEVICES and len(set(physical_devices)) != DEVICES:
            problems.append("physical devices are not distinct: %r"
                            % sorted(physical_devices))
        if len(pids) == DEVICES and len(set(pids)) != DEVICES:
            problems.append("worker pids are not distinct: %r" % sorted(pids))
        if not self.mock and any(row["synthetic"] for row in rows):
            problems.append("at least one rank reports synthetic identity")
        return rows, problems, physical_devices, logical_devices

    def _e33_cleanup_failed_start(self, ctx: dict) -> dict:
        cleanup_id = uuid.uuid4().hex
        with self.lock:
            self.e33_op = {"op_id": cleanup_id, "kind": "cleanup_start",
                           "context": dict(ctx),
                           "started_wall_ns": time.time_ns()}
            self.e33_acks = {}
            self._e33_broadcast_locked("e33_stop", cleanup_id, ctx)
        try:
            acks = self._e33_wait_acks(cleanup_id, min(600.0, E33_STOP_TIMEOUT_S))
            return {"ok": all(ack.get("ok") for ack in acks.values()),
                    "acks": {rank: {"ok": ack.get("ok"),
                                    "kind": ack.get("kind"),
                                    "error": ack.get("error")}
                             for rank, ack in acks.items()}}
        except E33ControlError as exc:
            return {"ok": False, "error": str(exc), "detail": exc.detail}
        finally:
            with self.lock:
                self.e33_busy = False
                self.e33_op = None
                self.e33_acks = {}

    def _e33_set_stopped_session(self, session: dict, rows: list,
                                 manifest_path: str, idle: dict,
                                 errors: list, acks_complete: bool) -> dict:
        trace_ranks = [row["rank"] for row in rows if row.get("trace_present")]
        export_complete = (len(trace_ranks) == DEVICES
                           and all(row.get("ok") is True for row in rows)
                           and not errors)
        stopped = {
            **session,
            "state": "stopped" if acks_complete else "stop_incomplete",
            "stopped_wall_ns": time.time_ns(),
            "global_idle_at_stop": idle,
            "ranks": rows,
            "manifest": manifest_path,
            "export": {
                "complete": export_complete,
                "trace_present_ranks": trace_ranks,
                "missing_trace_ranks": [rank for rank in range(DEVICES)
                                        if rank not in trace_ranks],
                "errors": errors,
            },
            "qualified": None,
            "qualification": "deferred_to_main_controller",
        }
        return stopped

    def e33_start(self, body: dict) -> dict:
        ctx = _e33_context(body, self.e33_task_root, require_all=True)
        with self.lock:
            if self.e33_busy:
                raise E33ControlError("another E33 control operation is in flight")
            session = self.e33_session
            if session is not None and session.get("state") in (
                    "active", "stop_incomplete"):
                raise E33ControlError(
                    "cohort state is %r; stop it before another start"
                    % session.get("state"),
                    detail={"session": self._e33_session_summary_locked()})
            idle = self._e33_idle_locked()
            if not idle["ok"]:
                raise E33ControlError(
                    "E33 start requires the whole DP8 group globally idle",
                    detail=idle)
            Path(ctx["out_dir"]).mkdir(parents=True, exist_ok=True)
            op_id = self._e33_new_op_locked("start", ctx)
            self._e33_broadcast_locked("e33_start", op_id, ctx)
        try:
            acks = self._e33_wait_acks(op_id, E33_START_TIMEOUT_S)
            rows, problems, physical_devices, logical_devices = (
                self._e33_validate_start_rows(acks))
            if problems:
                raise E33ControlError(
                    "E33 start rejected: " + "; ".join(problems[:12]),
                    status=502,
                    detail={"problems": problems,
                            "physical_devices": physical_devices,
                            "logical_devices": logical_devices,
                            "acks": acks})
        except BaseException as exc:
            cleanup = self._e33_cleanup_failed_start(ctx)
            with self.lock:
                self.e33_errors.append({
                    "where": "start", "error": repr(exc),
                    "detail": getattr(exc, "detail", None),
                    "cleanup": cleanup, "wall": time.time(),
                })
                self.e33_errors = self.e33_errors[-50:]
            if isinstance(exc, E33ControlError):
                raise
            raise E33ControlError("E33 start failed: %r" % (exc,), status=502,
                                  detail={"cleanup": cleanup}) from exc
        started_wall_ns = time.time_ns()
        session = {
            "state": "active",
            "op_id": op_id,
            **ctx,
            "started_wall_ns": started_wall_ns,
            "global_idle_at_start": idle,
            "ranks": rows,
            "devices": sorted(physical_devices),
            "physical_devices": sorted(physical_devices),
            "logical_devices": sorted(logical_devices),
            "identity_distinct_devices": (len(set(physical_devices)) == DEVICES),
            "config": self.cfg.public(),
            "qualified": None,
            "qualification": "deferred_to_main_controller",
        }
        start_payload = {
            "ok": True,
            "op_id": op_id,
            **ctx,
            "started_wall_ns": started_wall_ns,
            "global_idle_at_start": idle,
            "ack_count": len(rows),
            "devices": sorted(physical_devices),
            "physical_devices": sorted(physical_devices),
            "logical_devices": sorted(logical_devices),
            "ranks": rows,
            "qualified": None,
            "qualification": "deferred_to_main_controller",
            "note": "start succeeded only after all eight workers acknowledged "
                    "their real device/pid identity at global idle",
        }
        manifest_path = str(Path(ctx["out_dir"]) / "start.json")
        start_payload["manifest"] = manifest_path
        try:
            atomic_write_json(manifest_path, start_payload)
        except BaseException as exc:
            cleanup = self._e33_cleanup_failed_start(ctx)
            with self.lock:
                self.e33_busy = False
                self.e33_op = None
            raise E33ControlError("cannot write start manifest: %r" % (exc,),
                                  status=500, detail={"cleanup": cleanup}) from exc
        try:
            telemetry = E33Telemetry(
                rank=-1, out_dir=ctx["out_dir"], run_id=ctx["run_id"],
                case_id=ctx["case_id"], phase=ctx["phase"], role="parent",
                source="parent")
        except BaseException as exc:
            cleanup = self._e33_cleanup_failed_start(ctx)
            with self.lock:
                self.e33_busy = False
                self.e33_op = None
            raise E33ControlError("cannot open parent telemetry: %r" % (exc,),
                                  status=500, detail={"cleanup": cleanup}) from exc
        with self.lock:
            self.e33_telemetry = telemetry
            self.e33_session = session
            self.e33_busy = False
            self.e33_op = None
            self.e33_acks = {}
            telemetry.emit("cohort_start", source="parent.control",
                           extra={"op_id": op_id,
                                  "global_idle": idle,
                                  "ack_count": len(rows),
                                  "devices": sorted(physical_devices),
                                  "manifest": manifest_path})
        return start_payload

    def e33_stop(self, body: dict) -> dict:
        ctx = _e33_context(body, self.e33_task_root, require_all=True)
        with self.lock:
            if self.e33_busy:
                raise E33ControlError("another E33 control operation is in flight")
            session = self.e33_session
            if session is None or session.get("state") not in (
                    "active", "stop_incomplete"):
                raise E33ControlError("no active E33 cohort to stop",
                                      detail={"session": self._e33_session_summary_locked()})
            for name in ("run_id", "case_id", "phase", "out_dir"):
                if str(ctx[name]) != str(session[name]):
                    raise E33ControlError(
                        "stop %s=%r does not match active cohort %r"
                        % (name, ctx[name], session[name]))
            idle = self._e33_idle_locked()
            if not idle["ok"]:
                raise E33ControlError(
                    "E33 stop requires all client requests finished and all "
                    "rank queues drained", detail=idle)
            op_id = self._e33_new_op_locked("stop", ctx)
            self._e33_broadcast_locked("e33_stop", op_id, ctx)
        timeout_error = None
        try:
            acks = self._e33_wait_acks(op_id, E33_STOP_TIMEOUT_S)
        except E33ControlError as exc:
            timeout_error = exc
            acks = dict(exc.detail.get("acks", {})) if exc.detail else {}
        rows = self._e33_rank_rows_locked(acks, "e33_stop_ack")
        missing = [rank for rank in range(DEVICES) if rank not in acks]
        errors = []
        if timeout_error is not None:
            errors.append({"where": "ack_timeout", "missing_ranks": missing,
                           "error": str(timeout_error)})
        for row in rows:
            for item in row.get("errors") or []:
                errors.append({"rank": row["rank"], "error": item})
        if any(row.get("ack_kind") != "e33_stop_ack" or row.get("ok") is not True
               for row in rows):
            errors.append({"where": "rank_ack",
                           "detail": [{"rank": row["rank"],
                                       "kind": row.get("ack_kind"),
                                       "ok": row.get("ok"),
                                       "error": row.get("error")}
                                      for row in rows
                                      if row.get("ok") is not True]})
        manifest_path = str(Path(ctx["out_dir"]) / "cohort_manifest.json")
        incomplete_acks = (timeout_error is not None or bool(missing)
                           or any(row.get("ok") is not True for row in rows))
        stopped = self._e33_set_stopped_session(
            session, rows, manifest_path, idle, errors,
            acks_complete=not incomplete_acks)
        with self.lock:
            parent_telemetry = self.e33_telemetry
            if not incomplete_acks:
                self.e33_telemetry = None
        telemetry_stats = None
        if parent_telemetry is not None:
            parent_telemetry.emit(
                "cohort_stop_attempt" if incomplete_acks else "cohort_stop",
                source="parent.control",
                extra={"op_id": op_id, "global_idle": idle,
                       "incomplete_acks": incomplete_acks,
                       "missing_ranks": missing,
                       "errors": errors, "manifest": manifest_path})
            if incomplete_acks:
                parent_telemetry.flush()
                telemetry_stats = parent_telemetry.stats()
            else:
                telemetry_stats = parent_telemetry.close()
        manifest = {
            "schema_version": "e33-cohort-manifest/1.0",
            "framework": "deepinfer",
            "synthetic": bool(self.mock),
            "run_id": ctx["run_id"],
            "case_id": ctx["case_id"],
            "phase": ctx["phase"],
            "out_dir": ctx["out_dir"],
            "op_id": op_id,
            "started_wall_ns": session.get("started_wall_ns"),
            "stopped_wall_ns": stopped.get("stopped_wall_ns"),
            "session": {key: value for key, value in stopped.items()
                        if key not in ("ranks", "config", "global_idle_at_start",
                                       "global_idle_at_stop")},
            "global_idle_at_start": session.get("global_idle_at_start"),
            "global_idle_at_stop": idle,
            "ranks": rows,
            "ack_count": len(acks),
            "missing_ranks": missing,
            "export": stopped.get("export"),
            "parent_telemetry": telemetry_stats,
            "errors": errors,
            "clock": {
                "host_clock_domain": "time.time_ns / perf_counter_ns",
                "device_clock_mapping": None,
                "clock_mapping_error_bound_ns": None,
                "note": "host and NPU trace clocks are not aligned here",
            },
            "qualified": None,
            "qualification": "deferred_to_main_controller",
            "main_controller_validation_required": [
                "read every rank manifest and confirm raw/CSV/trace presence",
                "correlate host events with device trace intervals",
                "check trace drops, profiler buffers and graph visibility",
                "compare observer effect against a matched no-profiler window",
                "run independent trace/cohort checks before calling anything qualified",
            ],
            "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        try:
            atomic_write_json(manifest_path, manifest)
            manifest_error = None
        except BaseException as exc:  # noqa: BLE001
            manifest_error = repr(exc)
            with self.lock:
                self.e33_errors.append({"where": "stop_manifest",
                                        "error": manifest_error,
                                        "wall": time.time()})
        with self.lock:
            stopped["manifest_error"] = manifest_error
            self.e33_session = stopped
            self.e33_last_session = stopped
            self.e33_busy = False
            self.e33_op = None
            self.e33_acks = {}
        response = {
            "ok": not incomplete_acks and manifest_error is None,
            "op_id": op_id,
            **ctx,
            "stopped_wall_ns": stopped.get("stopped_wall_ns"),
            "ack_count": len(acks),
            "missing_ranks": missing,
            "state": stopped.get("state"),
            "export_complete": bool(stopped.get("export", {}).get("complete")),
            "missing_trace_ranks": stopped.get("export", {}).get(
                "missing_trace_ranks"),
            "manifest": manifest_path,
            "manifest_error": manifest_error,
            "ranks": rows,
            "parent_telemetry": telemetry_stats,
            "errors": errors,
            "qualified": None,
            "qualification": "deferred_to_main_controller",
        }
        if timeout_error is not None:
            raise E33ControlError(
                "E33 stop ack timeout; partial manifest written",
                status=504, detail=response)
        if incomplete_acks:
            raise E33ControlError(
                "E33 stop failed on one or more ranks; retry after they are idle",
                status=502, detail=response)
        return response

    def e33_snapshot(self, body: dict) -> dict:
        with self.lock:
            if self.e33_busy:
                raise E33ControlError("another E33 control operation is in flight")
            session = self.e33_session
            if session is not None and session.get("state") in (
                    "active", "stop_incomplete"):
                ctx = {name: session[name] for name in
                       ("run_id", "case_id", "phase", "out_dir")}
                if body and any(body.get(name) not in (None, "") for name in ctx):
                    supplied = _e33_context(body, self.e33_task_root,
                                            require_all=True)
                    for name in ctx:
                        if str(supplied[name]) != str(ctx[name]):
                            raise E33ControlError(
                                "snapshot %s does not match active cohort" % name)
            else:
                ctx = _e33_context(body, self.e33_task_root, require_all=True)
            idle = self._e33_idle_locked()
            if not idle["ok"]:
                raise E33ControlError(
                    "E33 snapshot requires global idle", detail=idle)
            Path(ctx["out_dir"]).mkdir(parents=True, exist_ok=True)
            op_id = self._e33_new_op_locked("snapshot", ctx)
            self._e33_broadcast_locked("e33_snapshot", op_id, ctx)
        try:
            acks = self._e33_wait_acks(op_id, E33_SNAPSHOT_TIMEOUT_S)
        except BaseException as exc:
            with self.lock:
                self.e33_busy = False
                self.e33_op = None
                self.e33_errors.append({"where": "snapshot", "error": repr(exc),
                                        "wall": time.time()})
            raise
        rows = self._e33_rank_rows_locked(acks, "e33_snapshot_ack")
        problems = [row for row in rows
                    if row.get("ack_kind") != "e33_snapshot_ack"
                    or row.get("ok") is not True]
        if problems:
            with self.lock:
                self.e33_busy = False
                self.e33_op = None
            raise E33ControlError("E33 snapshot failed on ranks %r"
                                  % [row["rank"] for row in problems],
                                  status=502,
                                  detail={"ranks": problems})
        manifest_path = str(Path(ctx["out_dir"]) / "snapshot_manifest.json")
        payload = {
            "schema_version": "e33-snapshot-manifest/1.0",
            "framework": "deepinfer",
            "synthetic": bool(self.mock),
            "op_id": op_id,
            **ctx,
            "taken_wall_ns": time.time_ns(),
            "global_idle": idle,
            "ranks": rows,
            "ack_count": len(rows),
            "qualified": None,
            "qualification": "deferred_to_main_controller",
        }
        try:
            atomic_write_json(manifest_path, payload)
            manifest_error = None
        except BaseException as exc:  # noqa: BLE001
            manifest_error = repr(exc)
        with self.lock:
            self.e33_busy = False
            self.e33_op = None
            self.e33_acks = {}
        return {"ok": True, "op_id": op_id, **ctx,
                "ack_count": len(rows), "global_idle": idle,
                "manifest": manifest_path, "manifest_error": manifest_error,
                "ranks": rows, "qualified": None,
                "qualification": "deferred_to_main_controller"}

    def _e33_session_summary_locked(self) -> dict | None:
        session = self.e33_session
        if session is None:
            return None
        return {
            "state": session.get("state"),
            "op_id": session.get("op_id"),
            "run_id": session.get("run_id"),
            "case_id": session.get("case_id"),
            "phase": session.get("phase"),
            "out_dir": session.get("out_dir"),
            "started_wall_ns": session.get("started_wall_ns"),
            "stopped_wall_ns": session.get("stopped_wall_ns"),
            "manifest": session.get("manifest"),
            "export": session.get("export"),
        }

    def e33_status(self) -> dict:
        with self.lock:
            idle = self._e33_idle_locked()
            rows = []
            for rank in range(DEVICES):
                raw = self.e33_rank_status[rank] or {}
                identity = raw.get("identity") or {}
                proc = self.workers[rank] if rank < len(self.workers) else None
                profiler = raw.get("profiler") or {}
                telemetry = raw.get("telemetry") or {}
                rows.append({
                    "rank": rank,
                    "parent_worker_pid": proc.pid if proc is not None else None,
                    "worker_alive": bool(proc is not None and proc.is_alive()),
                    "worker_exitcode": proc.exitcode if proc is not None else None,
                    "last_kind": raw.get("last_kind"),
                    "op_id": raw.get("op_id"),
                    "ok": raw.get("ok"),
                    "state": raw.get("state"),
                    "pid": identity.get("pid"),
                    "logical_device": identity.get("logical_device"),
                    "physical_card": identity.get("physical_card"),
                    "physical_device_id": identity.get("physical_device_id"),
                    "visible_device_ids": identity.get("visible_device_ids"),
                    "synthetic": identity.get("synthetic"),
                    "identity_summary": {
                        key: identity.get(key) for key in
                        ("pid", "rank", "logical_device", "physical_card",
                         "visible_device_ids", "physical_device_id",
                         "world_size", "ep_size", "cp_size",
                         "torch_npu_version", "deepinfer_version")
                    } if identity else None,
                    "profiler": {
                        "started": profiler.get("started"),
                        "stopped": profiler.get("stopped"),
                        "out_dir": profiler.get("out_dir"),
                        "manifest": profiler.get("manifest"),
                        "trace_present": (profiler.get("export") or {}).get(
                            "trace_present"),
                        "errors": len(profiler.get("errors") or []),
                        "qualified": None,
                    } if profiler else None,
                    "snapshot": raw.get("snapshot"),
                    "telemetry": telemetry,
                    "manifest": raw.get("manifest"),
                    "errors": raw.get("errors") or [],
                    "where": raw.get("where"),
                    "error": raw.get("error"),
                    "updated_wall": raw.get("updated_wall"),
                    "live_stats": self.rank_stats[rank],
                    "last_step": self.rank_last_step[rank],
                })
            telemetry_stats = (self.e33_telemetry.stats()
                               if self.e33_telemetry is not None else None)
            return {
                "framework": "deepinfer",
                "role": "parent",
                "run_id": (self.e33_session or {}).get("run_id"),
                "case_id": (self.e33_session or {}).get("case_id"),
                "phase": (self.e33_session or {}).get("phase"),
                "state": (self.e33_session or {}).get("state", "none"),
                "busy": self.e33_busy,
                "op": copy.deepcopy(self.e33_op),
                "session": self._e33_session_summary_locked(),
                "global_idle": idle,
                "http_inflight": self.http_inflight,
                "http_requests_total": self.http_requests_total,
                "ranks": rows,
                "parent_telemetry": telemetry_stats,
                "errors": list(self.e33_errors),
                "qualified": None,
                "qualification": "deferred_to_main_controller",
                "control_endpoints": [
                    "GET /e33/status", "POST /e33/profile/start",
                    "POST /e33/profile/stop", "POST /e33/snapshot",
                ],
                "loopback_only": True,
            }

    def status(self) -> dict:
        with self.lock:
            running = sum(len(s) for s in self.rank_active)
            waiting = len(self.pending_dispatch)
            cache = dict(_STARTUP_CACHE)
            sparse = copy.deepcopy(_STARTUP_SPARSE)
            sparse_ok = ((not sparse.get("enabled"))
                         or sparse.get("coverage_ok") is True)
            warm = copy.deepcopy(_STARTUP_WARM)
            warm_ok = (not warm.get("enabled")) or warm.get("completed") is True
            ready = (all(self.ready) and cache.get("coverage_ok", True)
                     and sparse_ok and warm_ok)
            ready_order = [r for r in sorted(
                range(DEVICES),
                key=lambda i: (self.rank_ready_at[i] is None,
                               self.rank_ready_at[i] or 0.0))
                if self.ready[r]]
            return {
                "ready": ready,
                "ranks_ready": sum(self.ready),
                "gemm_cache": cache,
                "sparse_mqa_cache": sparse,
                "startup_warm": warm,
                "rank_ready_at": list(self.rank_ready_at),
                "ready_order": ready_order,
                "server_idle": self.server_idle,
                "idle_epoch": self.idle_epoch,
                "inflight": running + waiting,
                "inflight_requests": running + waiting,
                "running": running,
                "waiting": waiting,
                "active_requests": running + waiting,
                "ranks": [
                    {
                        "rank": r,
                        "ready": self.ready[r],
                        "ready_at": self.rank_ready_at[r],
                        "active": sorted(self.rank_active[r]),
                        "stats": self.rank_stats[r],
                        "last_step": self.rank_last_step[r],
                        "resolved": self.rank_resolved[r],
                    }
                    for r in range(DEVICES)
                ],
                "fatal": self.fatal,
                "profile_status": self.profile_status,
                "http_inflight": self.http_inflight,
                "e33_busy": self.e33_busy,
                "e33_state": (self.e33_session or {}).get("state", "none"),
                "e33_session": self._e33_session_summary_locked(),
                "config": self.cfg.public(),
            }

    def wait_idle(self, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                if self.server_idle and not any(self.rank_active) and not self.pending_dispatch:
                    return True
            if self.fatal:
                return False
            time.sleep(0.02)
        return False

    def shutdown(self, stagger_ms: float = 0.0, force: bool = False) -> None:
        if not force:
            with self.lock:
                active = [rid for rid, st in self.requests.items()
                          if not st.done.is_set()]
            for rid in active:
                self.abort(rid)
            # Give the group a chance to flush the aborts before stopping it.
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                with self.lock:
                    if self.server_idle and not any(self.rank_active):
                        break
                time.sleep(0.05)
        for rank, inbox in enumerate(self.inboxes):
            try:
                inbox.put({"cmd": "stop"})
            except (OSError, ValueError):
                pass
            self.stop_sent_times[rank] = time.time()
            if stagger_ms:
                time.sleep(stagger_ms / 1000.0)
        for proc in self.workers:
            proc.join(timeout=5 if force else 60)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=5)
        self._stop = True
        self._router.join(timeout=5)
        with self.lock:
            parent_telemetry = self.e33_telemetry
            self.e33_telemetry = None
        if parent_telemetry is not None:
            parent_telemetry.close()

    # -- diagnostics -------------------------------------------------------
    def diag_start(self, path: str) -> None:
        self._diag_path = Path(path)
        self._diag_path.parent.mkdir(parents=True, exist_ok=True)
        self._diag_path.write_text("")

    def diag_flush(self) -> None:
        if self._diag_path is None:
            return
        with self.lock:
            rows = self.diagnostics
            self.diagnostics = []
        with self._diag_path.open("a") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# tokenizer helper (parent, CPU only)
# ---------------------------------------------------------------------------
class Tokenizer:
    def __init__(self, model: str):
        self._tok = None
        self._model = model
        self._lock = threading.Lock()

    def get(self):
        with self._lock:
            if self._tok is None:
                from transformers import AutoTokenizer
                self._tok = AutoTokenizer.from_pretrained(
                    self._model, trust_remote_code=True, local_files_only=True)
            return self._tok

    def encode(self, text: str) -> list[int]:
        return list(self.get().encode(text, add_special_tokens=False))

    def decode(self, ids: list[int]) -> str:
        return self.get().decode(ids, skip_special_tokens=False)


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------
SUPPORTED_BODY_KEYS = {
    "model", "prompt", "prompt_token_ids", "input_ids", "max_tokens",
    "temperature", "top_p", "top_k", "ignore_eos", "seed", "stream",
    "stream_options", "request_id", "user", "return_token_ids", "n",
    "assigned_dp_rank",
    "best_of", "stop", "echo", "suffix", "logprobs", "min_tokens",
    "repetition_penalty", "presence_penalty", "frequency_penalty",
    "logit_bias", "response_format", "stop_token_ids",
}


def validate_completion_body(body: dict) -> tuple[dict, list[int]]:
    unknown = sorted(set(body) - SUPPORTED_BODY_KEYS)
    if unknown:
        raise ValueError(f"unsupported fields: {unknown}")
    for key, allowed in (
        ("n", (1,)), ("best_of", (None, 1)), ("echo", (False, None)),
        ("suffix", (None,)), ("logprobs", (None,)), ("min_tokens", (None, 0)),
        ("repetition_penalty", (None, 1.0)), ("presence_penalty", (None, 0.0)),
        ("frequency_penalty", (None, 0.0)), ("logit_bias", (None,)),
        ("response_format", (None,)), ("stop_token_ids", (None, [])),
    ):
        if key in body and body[key] not in allowed:
            raise ValueError(f"unsupported value for {key}: {body[key]!r}")
    stop = body.get("stop")
    if stop not in (None, [], ""):
        raise ValueError("stop sequences are not implemented by the DeepInfer core; "
                         "refusing to drop them silently")
    ids = body.get("prompt_token_ids", body.get("input_ids"))
    if ids is not None:
        if not isinstance(ids, list) or not ids or not all(
                isinstance(i, int) and not isinstance(i, bool) and i >= 0 for i in ids):
            raise ValueError("prompt_token_ids must be a non-empty list of ints")
    return body, ids


class AdapterState:
    def __init__(self, cfg: ClusterConfig, cluster: Cluster):
        self.cfg = cfg
        self.cluster = cluster
        self.tokenizer = Tokenizer(cfg.model)
        self.request_timeout_s = cfg.request_timeout_s


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "DeepInferDP8Adapter/0.1"

    @property
    def state(self) -> AdapterState:
        return self.server.adapter_state  # type: ignore[attr-defined]

    def log_message(self, fmt, *args):
        sys.stderr.write("[http] %s - %s\n" % (self.address_string(), fmt % args))

    def _json(self, code: int, payload: dict) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _is_loopback(self) -> bool:
        return _is_loopback_addr(self.client_address[0])

    def _e33_json(self, code: int, payload: dict) -> None:
        payload = dict(payload)
        payload.setdefault("loopback_only", True)
        payload.setdefault("client", self.client_address[0])
        self._json(code, payload)

    def _telemetry(self):
        return self.state.cluster.e33_telemetry

    def _tel_emit(self, event_name: str, *, request_id=None, start=None,
                  exception=None, extra=None):
        telemetry = self._telemetry()
        if telemetry is None:
            return
        try:
            telemetry.emit(event_name, request_id=request_id, start=start,
                           source="parent.http", exception=exception,
                           extra=extra or {})
        except BaseException:  # noqa: BLE001
            pass

    def _tel_span(self, event_name: str, *, request_id=None, extra=None):
        telemetry = self._telemetry()
        if telemetry is None:
            return None
        try:
            return telemetry.begin(event_name, request_id=request_id,
                                   source="parent.http", extra=extra or {})
        except BaseException:  # noqa: BLE001
            return None

    def _tel_end(self, span, exc=None, extra=None):
        if span is not None:
            try:
                span.end(exc, extra=extra or {})
            except BaseException:  # noqa: BLE001
                pass

    @staticmethod
    def _now_clock() -> dict:
        return {"time_time_ns": time.time_ns(),
                "perf_counter_ns": time.perf_counter_ns(),
                "thread_time_ns": time.thread_time_ns(),
                "monotonic_ns": time.monotonic_ns()}

    def _e33_post(self, kind: str):
        if not self._is_loopback():
            return self._json(403, {"error": "E33 control is loopback-only",
                                    "client": self.client_address[0]})
        try:
            body = self._read_body()
        except Exception as exc:  # noqa: BLE001
            return self._e33_json(400, {"error": "invalid JSON body: %r" % (exc,)})
        try:
            cluster = self.state.cluster
            if kind == "start":
                result = cluster.e33_start(body)
            elif kind == "stop":
                result = cluster.e33_stop(body)
            else:
                result = cluster.e33_snapshot(body)
        except E33ControlError as exc:
            return self._e33_json(exc.status, {
                "error": str(exc), "detail": exc.detail,
                "qualified": None,
                "qualification": "deferred_to_main_controller"})
        except (ValueError, TypeError) as exc:
            return self._e33_json(400, {"error": str(exc)})
        except E33ControlBusy as exc:
            return self._e33_json(409, {"error": str(exc)})
        except BaseException as exc:  # noqa: BLE001
            return self._e33_json(500, {"error": repr(exc),
                                        "traceback": traceback.format_exc()[:8000],
                                        "qualified": None,
                                        "qualification": "deferred_to_main_controller"})
        return self._e33_json(200, result)

    def do_GET(self):
        if self.path == "/readyz":
            status = self.state.cluster.status()
            code = 200 if status["ready"] else 503
            return self._json(code, status)
        if self.path == "/status":
            return self._json(200, self.state.cluster.status())
        if self.path == "/profile/status":
            return self._json(200, {"profile_status": self.state.cluster.profile_status})
        if self.path == "/e33/status":
            if not self._is_loopback():
                return self._json(403, {"error": "E33 control is loopback-only",
                                        "client": self.client_address[0]})
            return self._e33_json(200, self.state.cluster.e33_status())
        if self.path == "/diag/requests":
            with self.state.cluster.lock:
                rows = list(self.state.cluster.request_phases.values())
            return self._json(200, {"requests": rows, "count": len(rows)})
        if self.path == "/server_info":
            status = self.state.cluster.status()
            return self._json(200, {
                "framework": "deepinfer",
                "lane": "native-dp8",
                "config": status["config"],
                "ranks_ready": status["ranks_ready"],
                "ranks": [r["resolved"] for r in status["ranks"]],
            })
        return self._json(404, {"error": "unknown endpoint"})

    def do_POST(self):
        if self.path == "/shutdown":
            self._json(200, {"shutting_down": True})
            threading.Thread(target=self.state.cluster.shutdown, daemon=True).start()
            return
        if self.path.startswith("/abort/"):
            rid = self.path[len("/abort/"):]
            ok = self.state.cluster.abort(rid)
            return self._json(200 if ok else 404, {"aborted": ok, "request_id": rid})
        if self.path == "/v1/completions":
            return self._completions()
        if self.path == "/generate":
            return self._native_generate()
        if self.path == "/e33/profile/start":
            return self._e33_post("start")
        if self.path == "/e33/profile/stop":
            return self._e33_post("stop")
        if self.path == "/e33/snapshot":
            return self._e33_post("snapshot")
        return self._json(404, {"error": "unknown endpoint"})

    def _read_body(self) -> dict:
        size = int(self.headers.get("Content-Length", 0))
        if size <= 0 or size > 8 * 1024 * 1024:
            raise ValueError("invalid request size")
        return json.loads(self.rfile.read(size))

    def _completions(self):
        cluster = self.state.cluster
        cluster.http_enter()
        try:
            return self._completions_impl()
        finally:
            cluster.http_leave()

    def _completions_impl(self):
        parse_start = self._now_clock()
        try:
            body = self._read_body()
            body, ids = validate_completion_body(body)
        except Exception as exc:  # noqa: BLE001
            self._tel_emit("http.parse", start=parse_start, exception=exc,
                           extra={"path": "/v1/completions"})
            return self._json(400, {"error": str(exc)})
        if ids is None:
            prompt = body.get("prompt")
            if isinstance(prompt, list):
                if not prompt or not all(
                        isinstance(i, int) and not isinstance(i, bool) and i >= 0
                        for i in prompt):
                    self._tel_emit("http.parse", start=parse_start,
                                   exception=ValueError("prompt list token ids"),
                                   extra={"path": "/v1/completions"})
                    return self._json(400, {"error": "prompt list must contain token ids"})
                ids = list(prompt)
            elif isinstance(prompt, str) and prompt:
                try:
                    ids = self.state.tokenizer.encode(prompt)
                except Exception as exc:  # noqa: BLE001
                    self._tel_emit("http.parse", start=parse_start,
                                   exception=exc,
                                   extra={"path": "/v1/completions"})
                    return self._json(500, {"error": f"tokenizer failed: {exc!r}"})
            else:
                self._tel_emit("http.parse", start=parse_start,
                               exception=ValueError("missing prompt"),
                               extra={"path": "/v1/completions"})
                return self._json(400, {"error": "provide prompt or prompt_token_ids"})
        max_tokens = int(body.get("max_tokens", 16))
        if max_tokens < 1:
            return self._json(400, {"error": "max_tokens must be >= 1"})
        if len(ids) + max_tokens > self.state.cfg.max_model_len:
            return self._json(400, {
                "error": f"context exceeded: {len(ids)} + {max_tokens} > "
                         f"{self.state.cfg.max_model_len}"})
        req_id = str(body.get("request_id") or f"cmpl-{uuid.uuid4().hex[:16]}")
        self._tel_emit("http.parse", request_id=req_id, start=parse_start,
                       extra={"path": "/v1/completions",
                              "prompt_tokens": len(ids),
                              "max_tokens": max_tokens,
                              "prompt_ids_provided": bool(
                                  body.get("prompt_token_ids") or body.get("input_ids")),
                              "content_length": int(
                                  self.headers.get("Content-Length", 0) or 0)})
        job = Job(
            request_id=req_id, prompt_token_ids=list(ids), max_tokens=max_tokens,
            temperature=float(body.get("temperature", 0.0)),
            top_p=float(body.get("top_p", 1.0)),
            top_k=int(body.get("top_k", -1)),
            ignore_eos=bool(body.get("ignore_eos", False)),
            seed=body.get("seed"),
            rank=body.get("assigned_dp_rank"),
            meta={"user": body.get("user", "adapter"),
                  "logical_id": body.get("user"),
                  "http_t0": time.time(),
                  "request_body": {k: v for k, v in body.items()
                                   if k not in ("prompt", "prompt_token_ids", "input_ids")}},
        )
        stream = bool(body.get("stream", False))
        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        t0 = time.time()
        admission = self._tel_span("http.admission", request_id=req_id,
                                   extra={"stream": stream})
        try:
            st = self.state.cluster.submit(job)
        except E33ControlBusy as exc:
            self._tel_end(admission, exc)
            return self._json(409, {"error": str(exc), "request_id": req_id})
        except BaseException as exc:  # noqa: BLE001
            self._tel_end(admission, exc)
            raise
        self._tel_end(admission, extra={
            "assigned_dp_rank": st.rank,
            "assigned_at": st.assigned_at,
            "queue_s": ((st.assigned_at - t0) if st.assigned_at else None),
            "http_inflight": self.state.cluster.http_inflight,
        })
        if stream:
            return self._stream_completion(st, include_usage, t0)
        return self._wait_completion(st, t0)

    def _seq_result(self, st: RequestState, t0: float) -> dict:
        try:
            text = self.state.tokenizer.decode(st.token_ids) if st.token_ids else ""
        except Exception:
            text = ""
        usage = {
            "prompt_tokens": len(st.job.prompt_token_ids),
            "completion_tokens": len(st.token_ids),
            "total_tokens": len(st.job.prompt_token_ids) + len(st.token_ids),
            "prompt_tokens_details": {"cached_tokens": getattr(st, "cache_hit_tokens", 0)},
        }
        return {
            "id": st.request_id,
            "request_id": st.request_id,
            "object": "text_completion",
            "model": self.state.cfg.model,
            "created": int(t0),
            "choices": [{
                "index": 0, "text": text, "token_ids": list(st.token_ids),
                "finish_reason": st.finish_reason,
                "logprobs": None,
            }],
            "usage": usage,
            "assigned_dp_rank": st.rank,
            "resolved_sampling": st.resolved_sampling,
            "decode_step_at_add": st.decode_step_at_add,
            "origin_step_ids": st.origin_steps,
            "delivery_step_ids": st.delivery_steps,
            "origin_known": bool(st.origin_steps) and all(
                value is not None for value in st.origin_steps),
            "native_usage": {**st.native_usage,
                             "emitted_tokens_raw": len(st.token_ids),
                             "definition": "RequestOutput.draft_tok/accept_tok on golden outputs; before output-cap/EOS filtering"},
            "native_golden_outputs": st.native_golden_outputs,
            "timing": {
                "submitted_at": t0,
                "assigned_at": st.assigned_at,
                "added_at": st.added_at,
                "first_token_at": st.first_at,
                "terminal_at": st.terminal_at,
                "ttft_s": (st.first_at - t0) if st.first_at else None,
                "e2e_s": (st.terminal_at - t0) if st.terminal_at else None,
                "queue_s": (st.assigned_at - t0) if st.assigned_at else None,
            },
            "server_idle_at_response": self.state.cluster.status()["server_idle"],
            "error": st.error,
        }

    def _wait_completion(self, st: RequestState, t0: float):
        span = self._tel_span("http.response", request_id=st.request_id,
                              extra={"stream": False})
        if not st.done.wait(self.state.request_timeout_s):
            self.state.cluster.abort(st.request_id)
            self._tel_end(span, extra={"status": 504, "error": "timeout"})
            return self._json(504, {"error": "request timeout", "request_id": st.request_id,
                                    "assigned_dp_rank": st.rank})
        if st.finish_reason == "abort" and not st.token_ids:
            self._tel_end(span, extra={"status": 499, "error": "aborted"})
            return self._json(499, {"error": "request aborted", "request_id": st.request_id})
        self._tel_end(span, extra={
            "status": 200,
            "tokens": len(st.token_ids),
            "ttft_s": ((st.first_at - t0) if st.first_at else None),
            "e2e_s": ((st.terminal_at - t0) if st.terminal_at else None),
            "finish_reason": st.finish_reason,
        })
        return self._json(200, self._seq_result(st, t0))

    def _stream_completion(self, st: RequestState, include_usage: bool, t0: float):
        response_span = self._tel_span("http.response", request_id=st.request_id,
                                       extra={"stream": True})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        def _write_sse(payload: bytes, kind: str = "frame") -> None:
            span = self._tel_span("sse.write", request_id=st.request_id,
                                  extra={"kind": kind, "bytes": len(payload)})
            try:
                self.wfile.write(("%X\r\n" % len(payload)).encode() + payload + b"\r\n")
                self.wfile.flush()
            except BaseException as exc:  # noqa: BLE001
                self._tel_end(span, exc)
                raise
            self._tel_end(span)

        def _serialize(payload: dict, kind: str) -> bytes:
            span = self._tel_span("sse.serialize", request_id=st.request_id,
                                  extra={"kind": kind})
            try:
                text = json.dumps(payload, ensure_ascii=False)
            except BaseException as exc:  # noqa: BLE001
                self._tel_end(span, exc)
                raise
            self._tel_end(span, extra={"json_bytes": len(text)})
            return ("data: " + text + "\n\n").encode()

        idx = 0
        sent_ids: list[int] = []
        sent_text = ""
        first_chunk = True
        frames = 0
        try:
            while True:
                with st.cond:
                    while idx >= len(st.chunks) and not st.done.is_set():
                        st.cond.wait(timeout=1.0)
                    chunks = st.chunks[idx:]
                    idx = len(st.chunks)
                    done = st.done.is_set()
                if chunks:
                    sent_ids.extend(c["token_id"] for c in chunks)
                    try:
                        text = self.state.tokenizer.decode(sent_ids)
                    except Exception:
                        text = ""
                    delta = text[len(sent_text):] if text.startswith(sent_text) else ""
                    sent_text = text
                    payload = {
                        "id": st.request_id, "object": "text_completion",
                        "created": int(t0), "model": self.state.cfg.model,
                        "choices": [{
                            "index": 0, "text": delta, "token_ids":
                                [c["token_id"] for c in chunks],
                            "finish_reason": None, "logprobs": None,
                        }],
                        "usage": None,
                    }
                    if first_chunk:
                        payload["assigned_dp_rank"] = st.rank
                        payload["decode_step_at_add"] = st.decode_step_at_add
                        payload["resolved_sampling"] = st.resolved_sampling
                        first_chunk = False
                    _write_sse(_serialize(payload, "chunk"), "chunk")
                    frames += 1
                if done and idx >= len(st.chunks):
                    break
            final = {
                "id": st.request_id, "object": "text_completion",
                "created": int(t0), "model": self.state.cfg.model,
                "choices": [{
                    "index": 0, "text": "", "token_ids": [],
                    "finish_reason": st.finish_reason, "logprobs": None,
                }],
                "usage": None if not include_usage else {
                    "prompt_tokens": len(st.job.prompt_token_ids),
                    "completion_tokens": len(sent_ids),
                    "total_tokens": len(st.job.prompt_token_ids) + len(sent_ids),
                    "prompt_tokens_details": {"cached_tokens": getattr(st, "cache_hit_tokens", 0)},
                },
                "assigned_dp_rank": st.rank,
                "decode_step_at_add": st.decode_step_at_add,
                "resolved_sampling": st.resolved_sampling,
                "origin_step_ids": st.origin_steps,
                "delivery_step_ids": st.delivery_steps,
                "origin_known": bool(st.origin_steps) and all(
                    value is not None for value in st.origin_steps),
                "native_usage": {**st.native_usage,
                                 "emitted_tokens_raw": len(sent_ids),
                                 "definition": "RequestOutput.draft_tok/accept_tok on golden outputs; before output-cap/EOS filtering"},
                "timing": {
                    "submitted_at": t0,
                    "assigned_at": st.assigned_at,
                    "added_at": st.added_at,
                    "first_token_at": st.first_at,
                    "terminal_at": st.terminal_at,
                    "ttft_s": (st.first_at - t0) if st.first_at else None,
                    "e2e_s": (st.terminal_at - t0) if st.terminal_at else None,
                    "queue_s": (st.assigned_at - t0) if st.assigned_at else None,
                },
                "usage_emitted": {
                    "prompt_tokens": len(st.job.prompt_token_ids),
                    "completion_tokens": len(sent_ids),
                },
                "server_idle_at_response": self.state.cluster.status()["server_idle"],
            }
            _write_sse(_serialize(final, "final"), "final")
            _write_sse(b"data: [DONE]\n\n", "done")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError) as exc:
            self._tel_end(response_span, exc, extra={"status": "client_disconnect",
                                                     "frames": frames})
            self.state.cluster.abort(st.request_id)
        except BaseException as exc:  # noqa: BLE001
            self._tel_end(response_span, exc, extra={"frames": frames})
            raise
        else:
            self._tel_end(response_span, extra={
                "status": 200,
                "frames": frames,
                "tokens": len(sent_ids),
                "finish_reason": st.finish_reason,
                "ttft_s": ((st.first_at - t0) if st.first_at else None),
                "e2e_s": ((st.terminal_at - t0) if st.terminal_at else None),
            })

    def _native_generate(self):
        cluster = self.state.cluster
        cluster.http_enter()
        try:
            return self._native_generate_impl()
        finally:
            cluster.http_leave()

    def _native_generate_impl(self):
        """Batch of independent requests; one job per rank, never mirrored."""
        parse_start = self._now_clock()
        try:
            body = self._read_body()
            rows = body.get("requests")
            if not isinstance(rows, list) or not rows:
                raise ValueError("requests must be a non-empty list")
        except Exception as exc:  # noqa: BLE001
            self._tel_emit("http.parse", start=parse_start, exception=exc,
                           extra={"path": "/generate"})
            return self._json(400, {"error": str(exc)})
        jobs = []
        t0 = time.time()
        for row in rows:
            try:
                row, ids = validate_completion_body(dict(row))
            except Exception as exc:  # noqa: BLE001
                self._tel_emit("http.parse", start=parse_start, exception=exc,
                               extra={"path": "/generate"})
                return self._json(400, {"error": str(exc)})
            if ids is None:
                return self._json(400, {"error": "native /generate requires prompt_token_ids"})
            max_tokens = int(row.get("max_tokens", 16))
            rid = str(row.get("request_id") or f"gen-{uuid.uuid4().hex[:16]}")
            jobs.append(Job(request_id=rid, prompt_token_ids=list(ids),
                            max_tokens=max_tokens,
                            temperature=float(row.get("temperature", 0.0)),
                            top_p=float(row.get("top_p", 1.0)),
                            top_k=int(row.get("top_k", -1)),
                            ignore_eos=bool(row.get("ignore_eos", False)),
                            seed=row.get("seed"),
                            rank=row.get("assigned_dp_rank"),
                            meta={"user": "native"}))
        request_ids = [job.request_id for job in jobs]
        self._tel_emit("http.parse", start=parse_start,
                       extra={"path": "/generate",
                              "request_ids": request_ids,
                              "content_length": int(
                                  self.headers.get("Content-Length", 0) or 0)})
        admission = self._tel_span("http.admission",
                                   extra={"batch": True,
                                          "request_ids": request_ids})
        try:
            states = [(st, t0) for st in self.state.cluster.submit_many(jobs)]
        except E33ControlBusy as exc:
            self._tel_end(admission, exc)
            return self._json(409, {"error": str(exc)})
        except BaseException as exc:  # noqa: BLE001
            self._tel_end(admission, exc)
            raise
        self._tel_end(admission, extra={
            "assigned": {st.request_id: st.rank for st, _ in states}})
        response_span = self._tel_span("http.response", extra={"batch": True})
        results = []
        for st, start in states:
            if not st.done.wait(self.state.request_timeout_s):
                self.state.cluster.abort(st.request_id)
                results.append({"request_id": st.request_id, "error": "timeout"})
                continue
            results.append(self._seq_result(st, start))
        self._tel_end(response_span, extra={
            "status": 200,
            "requests": len(results),
            "errors": sum(1 for item in results if item.get("error")),
        })
        return self._json(200, {"results": results,
                                "server_idle": self.state.cluster.status()["server_idle"]})


# ---------------------------------------------------------------------------
# offline diagnostic
# ---------------------------------------------------------------------------
def run_offline_diag(cfg: ClusterConfig, jobs_path: str, out_path: str) -> int:
    rows = json.loads(Path(jobs_path).read_text())
    if not isinstance(rows, list) or not 1 <= len(rows) <= DEVICES:
        raise SystemExit("offline diag needs 1..8 jobs")
    cluster = Cluster(cfg)
    cluster.diag_start(out_path)
    if not cluster.start():
        print(json.dumps({"error": "startup failed",
                          "fatal": cluster.fatal,
                          "status": cluster.status()}, indent=2))
        cluster.diag_flush()
        cluster.shutdown()
        return 2
    print(json.dumps({"ready": True, "resolved": cluster.rank_resolved}, indent=2))
    t0 = time.time()
    jobs = []
    seen_ranks = set()
    for idx, row in enumerate(rows):
        rank = int(row.get("rank", idx))
        if rank in seen_ranks or not 0 <= rank < DEVICES:
            raise SystemExit(f"invalid/duplicate rank {rank}")
        seen_ranks.add(rank)
        jobs.append(Job(
            request_id=row.get("request_id", f"diag-r{rank}"),
            prompt_token_ids=row["prompt_token_ids"],
            max_tokens=int(row["max_tokens"]),
            temperature=float(row.get("temperature", 0.0)),
            top_p=float(row.get("top_p", 1.0)),
            top_k=int(row.get("top_k", -1)),
            ignore_eos=bool(row.get("ignore_eos", False)),
            seed=row.get("seed", 1),
            rank=rank, meta={"user": "offline-diag"},
        ))
    states = cluster.submit_batch(jobs)
    deadline = time.time() + cfg.request_timeout_s
    while time.time() < deadline:
        if all(st.done.is_set() for st in states):
            break
        if cluster.fatal:
            break
        time.sleep(0.05)
    idle = cluster.wait_idle(timeout=60)
    summary = {
        "config": cfg.public(),
        "idle_verified": idle,
        "server_idle": cluster.server_idle,
        "requests": [
            {
                "request_id": st.request_id,
                "assigned_dp_rank": st.rank,
                "assigned_at": st.assigned_at,
                "added_at": st.added_at,
                "first_at": st.first_at,
                "terminal_at": st.terminal_at,
                "ttft_s": (st.first_at - t0) if st.first_at else None,
                "e2e_s": (st.terminal_at - t0) if st.terminal_at else None,
                "finish_reason": st.finish_reason,
                "token_ids": st.token_ids,
                "origin_step_ids": st.origin_steps,
                "delivery_step_ids": st.delivery_steps,
                "origin_known": not cfg.zos,
                "decode_step_at_add": st.decode_step_at_add,
                "resolved_sampling": st.resolved_sampling,
                "done": st.done.is_set(),
            }
            for st in states
        ],
        "ranks": cluster.status()["ranks"],
    }
    Path(out_path).with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({"summary": str(Path(out_path).with_suffix('.summary.json')),
                      "idle_verified": idle,
                      "finished": sum(1 for s in states if s.done.is_set())},
                     indent=2))
    cluster.diag_flush()
    cluster.shutdown()
    return 0 if idle and all(s.done.is_set() for s in states) else 3


# ---------------------------------------------------------------------------
# startup GEMM cache coverage
# ---------------------------------------------------------------------------
def run_startup_gemm_warm(args) -> dict:
    """Run the compile-only helper before any rank binds its model context.

    The helper is a separate process: it binds one NPU context, compiles the
    declared M<=max_m families into the persistent JIT cache, verifies every
    call is a hit, then exits and releases the context before the model group
    starts.  A failure is preserved in /status; it is never reported ready.
    """
    global _STARTUP_CACHE
    if args.mock:
        _STARTUP_CACHE = {"enabled": False, "coverage_ok": True,
                          "reason": "mock backend"}
        return _STARTUP_CACHE
    if not args.gemm_warm_manifest:
        _STARTUP_CACHE = {"enabled": False, "coverage_ok": True,
                          "reason": "no --gemm-warm-manifest"}
        return _STARTUP_CACHE
    helper = Path(args.gemm_warm_helper or
                  Path(__file__).with_name("e14_gemm_precompile.py")).resolve()
    manifest = Path(args.gemm_warm_manifest).resolve()
    report_path = Path(args.gemm_warm_report or
                       "/work/runtime-20261021/e14-gemm-readiness.json")
    started = time.time()
    base = {"enabled": True, "coverage_ok": False,
            "manifest": str(manifest), "helper": str(helper),
            "report": str(report_path)}
    if not helper.is_file() or not manifest.is_file():
        base["error"] = "helper or manifest missing"
        _STARTUP_CACHE = base
        return base
    cache_dir = args.gemm_warm_cache or os.environ.get(
        "DG_JIT_CACHE_DIR", "/work/runtime-20261021/dg-jit")
    cmd = [sys.executable, str(helper),
           "--manifest", str(manifest),
           "--cache-dir", cache_dir,
           "--out", str(report_path),
           "--max-m", str(args.gemm_warm_max_m),
           "--workers", str(args.gemm_warm_workers),
           "--adapter-source", str(Path(__file__).resolve())]
    env = os.environ.copy()
    env["DG_JIT_CACHE_DIR"] = cache_dir
    print(json.dumps({"stage": "gemm_warm_start", "cmd": cmd,
                      "cache_dir": cache_dir}), flush=True)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=args.gemm_warm_timeout, env=env)
    except subprocess.TimeoutExpired as exc:
        base.update({"error": "timeout after %ss" % args.gemm_warm_timeout,
                     "stdout_tail": (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
                     "elapsed_s": round(time.time() - started, 3)})
        _STARTUP_CACHE = base
        print(json.dumps({"stage": "gemm_warm_timeout", **_STARTUP_CACHE}),
              flush=True)
        return base
    report = None
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text())
        except (OSError, ValueError) as exc:
            base["report_parse_error"] = repr(exc)
    manifest_hash = _sha256_file(manifest)
    adapter_hash = _sha256_file(Path(__file__).resolve())
    stale = []
    if report is not None:
        if report.get("manifest_sha256") != manifest_hash:
            stale.append("manifest_sha256")
        if report.get("adapter_sha256") != adapter_hash:
            stale.append("adapter_sha256")
        if int(report.get("max_m") or 0) != int(args.gemm_warm_max_m):
            stale.append("max_m")
        if not report.get("cache", {}).get("fingerprint_persisted", False):
            stale.append("cache_fingerprint")
    base.update({
        "returncode": proc.returncode,
        "elapsed_s": round(time.time() - started, 3),
        "coverage_ok": (bool((report or {}).get("coverage_ok", False))
                        and not stale and proc.returncode == 0),
        "manifest_sha256": manifest_hash,
        "adapter_sha256": adapter_hash,
        "stale_reasons": stale,
        "report_sha256": _sha256_file(report_path),
        "stdout_tail": (proc.stdout or "")[-2000:],
        "stderr_tail": (proc.stderr or "")[-2000:],
    })
    if report:
        base["summary"] = {
            "cache_dirs": report.get("cache", {}).get("dirs_after_verify"),
            "new_dirs_compile": report.get("cache", {}).get("new_dirs_compile"),
            "verify_slow_calls": report.get("verification", {}).get("slow_calls"),
            "verify_new_dirs": len(report.get("verification", {}).get("new_dirs") or []),
            "remaining_misses": report.get("remaining_misses"),
            "total_elapsed_s": report.get("total_elapsed_s"),
            "max_m": report.get("max_m"),
            "families": len(report.get("declared_covered_input_classes", {}).get("families") or []),
        }
    else:
        base["error"] = "report missing"
    _STARTUP_CACHE = base
    print(json.dumps({"stage": "gemm_warm_done", **_STARTUP_CACHE}), flush=True)
    return base


def run_startup_sparse_mqa_warm(args) -> dict:
    """Compile-only sparse-MQA metadata coverage before any rank starts.

    The helper uses the installed dry-run path
    (``compile_without_load``) through the public metadata API, so it never
    launches a kernel.  A failed or stale report never reports ready.
    """
    global _STARTUP_SPARSE
    if args.mock:
        _STARTUP_SPARSE = {"enabled": False, "coverage_ok": True,
                           "reason": "mock backend"}
        return _STARTUP_SPARSE
    if not args.sparse_mqa_warm_manifest:
        _STARTUP_SPARSE = {"enabled": False, "coverage_ok": True,
                           "reason": "no --sparse-mqa-warm-manifest"}
        return _STARTUP_SPARSE
    helper = Path(args.sparse_mqa_warm_helper or Path(__file__).with_name(
        "e15_sparse_mqa_metadata_precompile.py")).resolve()
    manifest = Path(args.sparse_mqa_warm_manifest).resolve()
    report_path = Path(args.sparse_mqa_warm_report or
                       "/work/runtime-20261021/e15-sparse-mqa-readiness.json")
    started = time.time()
    base = {"enabled": True, "coverage_ok": False,
            "manifest": str(manifest), "helper": str(helper),
            "report": str(report_path)}
    if not helper.is_file() or not manifest.is_file():
        base["error"] = "helper or manifest missing"
        _STARTUP_SPARSE = base
        return base
    cache_dir = args.sparse_mqa_warm_cache or os.environ.get(
        "DG_JIT_CACHE_DIR", "/work/runtime-20261021/dg-jit")
    cmd = [sys.executable, str(helper),
           "--manifest", str(manifest),
           "--cache-dir", cache_dir,
           "--out", str(report_path),
           "--max-q", str(args.sparse_mqa_warm_max_q),
           "--workers", str(args.sparse_mqa_warm_workers),
           "--adapter-source", str(Path(__file__).resolve())]
    env = os.environ.copy()
    env["DG_JIT_CACHE_DIR"] = cache_dir
    print(json.dumps({"stage": "sparse_mqa_warm_start", "cmd": cmd,
                      "cache_dir": cache_dir}), flush=True)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=args.sparse_mqa_warm_timeout, env=env)
    except subprocess.TimeoutExpired as exc:
        base.update({"error": "timeout after %ss" % args.sparse_mqa_warm_timeout,
                     "stdout_tail": (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
                     "elapsed_s": round(time.time() - started, 3)})
        _STARTUP_SPARSE = base
        print(json.dumps({"stage": "sparse_mqa_warm_timeout", **_STARTUP_SPARSE}),
              flush=True)
        return base
    report = None
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text())
        except (OSError, ValueError) as exc:
            base["report_parse_error"] = repr(exc)
    manifest_hash = _sha256_file(manifest)
    helper_hash = _sha256_file(helper)
    adapter_hash = _sha256_file(Path(__file__).resolve())
    stale = []
    if report is not None:
        if report.get("manifest_sha256") != manifest_hash:
            stale.append("manifest_sha256")
        if report.get("helper_sha256") != helper_hash:
            stale.append("helper_sha256")
        if report.get("adapter_sha256") != adapter_hash:
            stale.append("adapter_sha256")
        if int(report.get("max_q") or 0) != int(args.sparse_mqa_warm_max_q):
            stale.append("max_q")
        if not report.get("cache", {}).get("fingerprint_persisted", False):
            stale.append("cache_fingerprint")
        if report.get("verification", {}).get("new_dirs"):
            stale.append("verify_new_dirs")
    base.update({
        "returncode": proc.returncode,
        "elapsed_s": round(time.time() - started, 3),
        "coverage_ok": (bool((report or {}).get("coverage_ok", False))
                        and not stale and proc.returncode == 0),
        "manifest_sha256": manifest_hash,
        "helper_sha256": helper_hash,
        "adapter_sha256": adapter_hash,
        "stale_reasons": stale,
        "report_sha256": _sha256_file(report_path),
        "stdout_tail": (proc.stdout or "")[-2000:],
        "stderr_tail": (proc.stderr or "")[-2000:],
    })
    if report:
        base["summary"] = {
            "cache_dirs": report.get("cache", {}).get("dirs_after_verify"),
            "new_dirs_compile": report.get("cache", {}).get("new_dirs_compile"),
            "new_signatures": len(report.get("cache", {}).get("new_signatures") or []),
            "signatures_total": len(report.get("cache", {}).get("signatures_after") or []),
            "verify_slow_calls": report.get("verification", {}).get("slow_calls"),
            "verify_new_dirs": len(report.get("verification", {}).get("new_dirs") or []),
            "remaining_misses": report.get("remaining_misses"),
            "total_elapsed_s": report.get("total_elapsed_s"),
            "max_q": report.get("max_q"),
        }
    else:
        base["error"] = "report missing"
    _STARTUP_SPARSE = base
    print(json.dumps({"stage": "sparse_mqa_warm_done", **_STARTUP_SPARSE}),
          flush=True)
    return base


def _sha256_file(path: Path) -> str | None:
    try:
        import hashlib
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


# ---------------------------------------------------------------------------
# normal-loop startup warm
# ---------------------------------------------------------------------------
def _cache_snapshot() -> dict:
    """Count persistent DG/TileLang/EP cache entries before/after a warm."""
    def _dirs(root: Path) -> set:
        try:
            return {p.name for p in root.iterdir() if p.is_dir()}
        except OSError:
            return set()

    def _files(root: Path) -> set:
        found = set()
        try:
            for dirpath, _dirnames, filenames in os.walk(root):
                base = Path(dirpath)
                for name in filenames:
                    found.add(str(Path(base, name).relative_to(root)))
        except OSError:
            pass
        return found

    dg_root = Path(os.environ.get(
        "DG_JIT_CACHE_DIR", "/work/runtime-20261021/dg-jit"))
    dg_commit_root = dg_root / "cache"
    if not dg_commit_root.is_dir():
        dg_commit_root = dg_root
    tl_root = Path(os.environ.get(
        "TILELANG_CACHE_DIR", "/work/runtime-20261021/tilelang-cache"))
    ep_root = Path(os.environ.get(
        "EP_JIT_CACHE_DIR", "/work/runtime-20261021/ep-jit"))
    return {
        "dg_commit_root": str(dg_commit_root),
        "dg_dirs": _dirs(dg_commit_root),
        "tilelang_root": str(tl_root),
        "tilelang_files": _files(tl_root),
        "ep_root": str(ep_root),
        "ep_files": _files(ep_root),
    }


def _cache_delta(before: dict, after: dict) -> dict:
    def _row(before_set: set, after_set: set) -> dict:
        new = sorted(after_set - before_set)
        gone = sorted(before_set - after_set)
        return {"before": len(before_set), "after": len(after_set),
                "new_count": len(new), "removed_count": len(gone),
                "new": new[:200], "removed": gone[:50]}

    return {
        "dg_commit_root": after.get("dg_commit_root"),
        "dg_dirs": _row(before.get("dg_dirs") or set(),
                        after.get("dg_dirs") or set()),
        "tilelang_root": after.get("tilelang_root"),
        "tilelang_files": _row(before.get("tilelang_files") or set(),
                               after.get("tilelang_files") or set()),
        "ep_root": after.get("ep_root"),
        "ep_files": _row(before.get("ep_files") or set(),
                         after.get("ep_files") or set()),
    }


def _rank_stats_snapshot(cluster: "Cluster") -> list[dict]:
    with cluster.lock:
        return [copy.deepcopy(cluster.rank_stats[r]) for r in range(DEVICES)]


def _warm_counter_deltas(before: list[dict], after: list[dict]) -> list[dict]:
    keys = ("steps", "placeholder_steps", "prompt_steps", "decode_steps",
            "tokens_emitted", "flushes", "graph_main_replays",
            "graph_dflash_replays")
    rows = []
    for rank in range(DEVICES):
        row = {"rank": rank}
        for key in keys:
            row[key] = int(after[rank].get(key) or 0) - int(
                before[rank].get(key) or 0)
        rows.append(row)
    return rows


def _warm_case_rows(states: list["RequestState"]) -> list[dict]:
    rows = []
    for st in states:
        rows.append({
            "request_id": st.request_id,
            "rank": st.rank,
            "prompt_tokens": len(st.job.prompt_token_ids),
            "max_tokens": st.job.max_tokens,
            "temperature": st.job.temperature,
            "ignore_eos": st.job.ignore_eos,
            "seed": st.job.seed,
            "assigned_at": st.assigned_at,
            "added_at": st.added_at,
            "first_token_at": st.first_at,
            "terminal_at": st.terminal_at,
            "ttft_s": (st.first_at - st.job.arrival_ts) if st.first_at else None,
            "e2e_s": ((st.terminal_at - st.job.arrival_ts)
                      if st.terminal_at else None),
            "finish_reason": st.finish_reason,
            "error": st.error,
            "output_tokens": len(st.token_ids),
            "token_ids": list(st.token_ids),
            "native_usage": copy.deepcopy(st.native_usage),
        })
    return rows


def _warm_wait(cluster: "Cluster", states: list["RequestState"],
               deadline: float) -> str | None:
    """Wait for terminal states and a clean eight-rank idle group."""
    while True:
        if cluster.fatal:
            return "worker fatal: %s" % cluster.fatal
        if all(st.done.is_set() for st in states):
            break
        if time.monotonic() >= deadline:
            pending = [st.request_id for st in states if not st.done.is_set()]
            return "timeout waiting for warm requests: %r" % pending
        time.sleep(0.02)
    bad = [{"request_id": st.request_id, "error": st.error}
           for st in states if st.error]
    if bad:
        return "warm request error: %r" % bad
    if not cluster.wait_idle(max(0.1, deadline - time.monotonic())):
        return "timeout waiting for global idle after warm"
    status = cluster.status()
    if (not status.get("server_idle") or status.get("inflight")
            or status.get("running") or status.get("waiting")):
        return "group not idle after warm: server_idle=%r inflight=%r " \
               "running=%r waiting=%r" % (
                   status.get("server_idle"), status.get("inflight"),
                   status.get("running"), status.get("waiting"))
    active = {str(r["rank"]): r["active"] for r in status.get("ranks", [])
              if r.get("active")}
    if active:
        return "rank active set not empty after warm: %r" % active
    return None


def _write_startup_warm_report(path: Path, payload: dict) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
        return True
    except OSError as exc:
        print(json.dumps({"stage": "startup_warm_report_error",
                          "path": str(path), "error": repr(exc)}), flush=True)
        return False


def run_startup_normal_warm(adapter: "AdapterState", args) -> bool:
    """Exercise the real C1-sparse and C8 batch serving paths before ready.

    The two requests are fixed internal text, use ignore_eos with a small
    output cap, and are reported separately from public traffic.  A failure
    leaves readiness unpublished; the caller tears the rank group down.
    """
    global _STARTUP_WARM
    started_mono = time.monotonic()
    started_wall = time.time()
    nonce = "%s-%s" % (time.strftime("%Y%m%dT%H%M%S", time.localtime(started_wall)),
                       uuid.uuid4().hex[:6])
    prompts_path = Path(args.startup_warm_prompts or Path(__file__).with_name(
        "e15_warm_prompts.json")).resolve()
    report_path = Path(args.startup_warm_report or
                       "/work/runtime-20261021/e15-startup-warm.json")
    max_tokens = int(args.startup_warm_max_tokens)
    timeout_s = float(args.startup_warm_timeout)
    base = {
        "schema_version": "1.0",
        "experiment": "E15",
        "enabled": True,
        "completed": False,
        "label": adapter.cfg.label,
        "nonce": nonce,
        "started_at": started_wall,
        "adapter_path": str(Path(__file__).resolve()),
        "adapter_sha256": _sha256_file(Path(__file__).resolve()),
        "prompts_path": str(prompts_path),
        "prompts_sha256": _sha256_file(prompts_path),
        "max_tokens_per_request": max_tokens,
        "timeout_s": timeout_s,
        "report_path": str(report_path),
        "note": "internal startup warm; not a user request and not part of "
                "formal performance measurements",
    }
    # Publish the honest in-progress state so status() cannot report ready
    # while the two real warm requests are still running.
    _STARTUP_WARM = copy.deepcopy(base)

    def finish(error: str | None) -> bool:
        global _STARTUP_WARM
        base["completed"] = error is None
        base["error"] = error
        base["completed_at"] = time.time()
        base["elapsed_s"] = round(time.monotonic() - started_mono, 6)
        _STARTUP_WARM = copy.deepcopy(base)
        _write_startup_warm_report(report_path, _STARTUP_WARM)
        print(json.dumps({
            "stage": "startup_warm_done" if error is None
                     else "startup_warm_failed",
            "nonce": nonce,
            "completed": base["completed"],
            "error": error,
            "elapsed_s": base["elapsed_s"],
            "cases": {k: (len(v) if isinstance(v, list) else v)
                      for k, v in (base.get("cases") or {}).items()},
            "cache_delta": base.get("cache"),
            "rank_counter_deltas": base.get("rank_counter_deltas"),
        }), flush=True)
        return error is None

    try:
        prompts = json.loads(prompts_path.read_text())
        c1_text = str(prompts["c1_text"])
        c8_text = str(prompts["c8_text"])
    except (OSError, ValueError, KeyError) as exc:
        return finish("prompt file: %r" % exc)
    try:
        c1_ids = adapter.tokenizer.encode(c1_text)
        c8_ids = adapter.tokenizer.encode(c8_text)
        adapter.tokenizer.decode(c1_ids[:1])
    except BaseException as exc:  # noqa: BLE001
        return finish("tokenizer: %r" % exc)

    budget = int(adapter.cfg.max_num_batched_tokens)
    if not c1_ids or not c8_ids:
        return finish("warm prompt encoded to zero tokens")
    if len(c1_ids) > budget or len(c8_ids) > budget:
        return finish("warm prompt exceeds the one-chunk budget: c1=%d "
                      "c8=%d budget=%d" % (len(c1_ids), len(c8_ids), budget))
    if (len(c1_ids) + max_tokens > adapter.cfg.max_model_len
            or len(c8_ids) + max_tokens > adapter.cfg.max_model_len):
        return finish("warm prompt plus output exceeds max_model_len")
    if max_tokens < 1:
        return finish("startup warm max tokens must be >= 1")

    base["prompts"] = {
        "c1_tokens": len(c1_ids),
        "c8_tokens": len(c8_ids),
        "c1_ids_sha256": hashlib.sha256(
            json.dumps(c1_ids).encode()).hexdigest(),
        "c8_ids_sha256": hashlib.sha256(
            json.dumps(c8_ids).encode()).hexdigest(),
    }
    cluster = adapter.cluster
    deadline = started_mono + timeout_s
    cache_before = _cache_snapshot()
    stats_before = _rank_stats_snapshot(cluster)
    cases: dict = {}
    base["cases"] = cases

    c1_job = Job(
        request_id="e15-warm-%s-c1" % nonce,
        prompt_token_ids=list(c1_ids),
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        ignore_eos=True,
        seed=15000,
        rank=0,
        meta={"user": "internal-startup-warm",
              "logical_id": "e15-warm-c1"},
    )
    phase_t0 = time.monotonic()
    c1_state = cluster.submit(c1_job)
    error = _warm_wait(cluster, [c1_state], deadline)
    cases["c1"] = _warm_case_rows([c1_state])
    cases["c1_elapsed_s"] = round(time.monotonic() - phase_t0, 6)
    if error is not None:
        if not c1_state.done.is_set():
            cluster.abort(c1_state.request_id)
        base["cache"] = _cache_delta(cache_before, _cache_snapshot())
        base["rank_counter_deltas"] = _warm_counter_deltas(
            stats_before, _rank_stats_snapshot(cluster))
        return finish("C1 sparse warm: %s" % error)

    c8_jobs = []
    for rank in range(DEVICES):
        c8_jobs.append(Job(
            request_id="e15-warm-%s-c8-r%d" % (nonce, rank),
            prompt_token_ids=list(c8_ids),
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            ignore_eos=True,
            seed=15100 + rank,
            rank=rank,
            meta={"user": "internal-startup-warm",
                  "logical_id": "e15-warm-c8-r%d" % rank},
        ))
    phase_t0 = time.monotonic()
    c8_states = cluster.submit_batch(c8_jobs)
    error = _warm_wait(cluster, c8_states, deadline)
    cases["c8"] = _warm_case_rows(c8_states)
    cases["c8_elapsed_s"] = round(time.monotonic() - phase_t0, 6)
    cache_after = _cache_snapshot()
    stats_after = _rank_stats_snapshot(cluster)
    base["cache"] = _cache_delta(cache_before, cache_after)
    deltas = _warm_counter_deltas(stats_before, stats_after)
    base["rank_counter_deltas"] = deltas
    if error is not None:
        for st in c8_states:
            if not st.done.is_set():
                cluster.abort(st.request_id)
        return finish("C8 batch warm: %s" % error)

    short = [row["rank"] for row in deltas
             if row["prompt_steps"] < 1 or row["decode_steps"] < 1
             or row["tokens_emitted"] < min(max_tokens, 8)]
    if short:
        return finish("ranks did not exercise real prompt/decode: %r" % short)

    status = cluster.status()
    ready_at = [value for value in status.get("rank_ready_at") or []
                if value is not None]
    base["ready"] = {
        "rank_ready_at": status.get("rank_ready_at"),
        "ready_order": status.get("ready_order"),
        "all_rank_ready_at": max(ready_at) if ready_at else None,
        "warm_completed_at": time.time(),
        "server_idle": status.get("server_idle"),
        "inflight": status.get("inflight"),
        "running": status.get("running"),
        "waiting": status.get("waiting"),
        "all_ranks_ready": bool(status.get("ranks_ready") == DEVICES),
    }
    return finish(None)


# ---------------------------------------------------------------------------
# local E33 control-protocol check (mock only; never hardware evidence)
# ---------------------------------------------------------------------------
def run_e33_mock_check(cfg: ClusterConfig) -> int:
    import tempfile
    import urllib.error
    import urllib.request

    root = Path(tempfile.mkdtemp(prefix="e33-mock-check-"))
    mock_cfg = copy.copy(cfg)
    mock_cfg.mock = True
    mock_cfg.e33_task_root = str(root)
    mock_cfg.request_timeout_s = 120.0
    checks = []

    def check(name: str, ok: bool, detail="") -> None:
        checks.append({"name": name, "pass": bool(ok), "detail": str(detail)})

    cluster = Cluster(mock_cfg, mock=True)
    httpd = None
    server_thread = None
    exit_code = 1
    try:
        if not cluster.start():
            check("cluster-start", False, cluster.fatal)
            raise SystemExit(1)
        check("cluster-start", True, "8 mock ranks ready")
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        httpd.daemon_threads = True
        httpd.adapter_state = AdapterState(mock_cfg, cluster)
        server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        server_thread.start()
        base = "http://127.0.0.1:%d" % httpd.server_address[1]

        def call(method: str, path: str, body=None, timeout=120.0):
            data = None if body is None else json.dumps(body).encode()
            request = urllib.request.Request(
                base + path, data=data, method=method,
                headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    raw = response.read()
                    if path == "/v1/completions" and body and body.get("stream"):
                        return response.status, raw.decode(errors="replace")
                    return response.status, json.loads(raw)
            except urllib.error.HTTPError as exc:
                raw = exc.read()
                try:
                    return exc.code, json.loads(raw)
                except ValueError:
                    return exc.code, raw.decode(errors="replace")

        def wait_idle(timeout=30.0) -> dict:
            deadline = time.monotonic() + timeout
            last = {}
            while time.monotonic() < deadline:
                code, status = call("GET", "/e33/status")
                last = status
                if code == 200 and (status.get("global_idle") or {}).get("ok"):
                    return status
                time.sleep(0.02)
            return last

        def wait_inflight(timeout=10.0) -> bool:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                code, status = call("GET", "/status")
                if code == 200 and int(status.get("inflight") or 0) > 0:
                    return True
                time.sleep(0.005)
            return False

        def contexts(tag: str) -> dict:
            return {"run_id": "E33-MOCK-%s" % tag, "case_id": "MOCK-C32-N64",
                    "phase": "mock_%s" % tag,
                    "out_dir": str(root / "profiles" / ("run-%s" % tag))}

        code, status = call("GET", "/e33/status")
        check("status-endpoint", code == 200
              and (status.get("global_idle") or {}).get("ranks_ready") == DEVICES,
              {"code": code, "state": status.get("state")})
        check("loopback-helper",
              _is_loopback_addr("127.0.0.1") and _is_loopback_addr("::1")
              and _is_loopback_addr("::ffff:127.0.0.1")
              and not _is_loopback_addr("10.1.2.3"), "address classification")

        outside = {"run_id": "E33-MOCK-OUTSIDE", "case_id": "MOCK",
                   "phase": "outside", "out_dir": "/tmp/e33-outside-root"}
        code, body = call("POST", "/e33/profile/start", outside)
        check("start-rejects-outside-root", code == 400, {"code": code,
                                                          "error": body.get("error")})

        ctx = contexts("a")
        code, start = call("POST", "/e33/profile/start", ctx)
        check("start-8-rank-acks", code == 200
              and start.get("ack_count") == DEVICES
              and len(start.get("ranks") or []) == DEVICES,
              {"code": code, "ack_count": start.get("ack_count")})
        check("start-devices-distinct",
              sorted(start.get("devices") or []) == list(range(DEVICES)),
              start.get("devices"))
        check("start-uses-physical-device-not-only-logical",
              len(set(start.get("logical_devices") or [])) <= 1
              and len(set(start.get("devices") or [])) == DEVICES,
              {"logical_devices": start.get("logical_devices"),
               "physical_devices": start.get("devices")})
        check("start-pids-distinct",
              len({row.get("pid") for row in start.get("ranks") or []}) == DEVICES,
              [row.get("pid") for row in start.get("ranks") or []])
        check("start-synthetic-labelled", all(
            row.get("synthetic") is True for row in start.get("ranks") or []),
            "mock identities are synthetic")
        check("start-not-qualified", start.get("qualified") is None,
              start.get("qualification"))

        code, again = call("POST", "/e33/profile/start", ctx)
        check("start-rejects-active-cohort", code == 409,
              {"code": code, "error": again.get("error")})

        stream_result = {}

        def stream_request():
            stream_result["result"] = call(
                "POST", "/v1/completions",
                {"prompt_token_ids": [1, 2, 3, 4], "max_tokens": 12,
                 "request_id": "mock-stream-1", "stream": True,
                 "stream_options": {"include_usage": True}}, timeout=120.0)

        stream_thread = threading.Thread(target=stream_request, daemon=True)
        stream_thread.start()
        check("client-inflight-observed", wait_inflight(10.0), "inflight > 0")
        code, stop_busy = call("POST", "/e33/profile/stop", ctx)
        check("stop-rejects-active-client", code == 409
              and "drained" in str(stop_busy.get("error")),
              {"code": code, "error": stop_busy.get("error")})
        stream_thread.join(timeout=120)
        stream_code, stream_text = stream_result.get("result", (None, ""))
        check("stream-request-completed", stream_code == 200
              and "data: [DONE]" in stream_text
              and '"token_ids"' in stream_text,
              {"code": stream_code, "bytes": len(stream_text)})
        idle = wait_idle(30.0)
        check("idle-after-client", bool((idle.get("global_idle") or {}).get("ok")),
              idle.get("global_idle"))

        code, snapshot = call("POST", "/e33/snapshot", ctx)
        check("snapshot-8-rank-acks", code == 200
              and snapshot.get("ack_count") == DEVICES,
              {"code": code, "ack_count": snapshot.get("ack_count")})
        check("snapshot-manifest", bool(snapshot.get("manifest"))
              and Path(snapshot["manifest"]).is_file(),
              snapshot.get("manifest"))

        code, active_status = call("GET", "/e33/status")
        code2, server_status = call("GET", "/status")
        steps_seen = max([int((row.get("stats") or {}).get("steps") or 0)
                          for row in (server_status.get("ranks") or [])] or [0])
        stopped_early = any(row.get("last_kind") == "e33_stop_ack"
                            for row in active_status.get("ranks") or [])
        check("no-auto-stop-after-four-steps",
              code == 200 and active_status.get("state") == "active"
              and steps_seen >= 8 and not stopped_early,
              {"state": active_status.get("state"), "steps_seen": steps_seen,
               "stop_ack_seen": stopped_early})

        code, stop = call("POST", "/e33/profile/stop", ctx)
        check("stop-8-rank-acks", code == 200
              and stop.get("ack_count") == DEVICES,
              {"code": code, "ack_count": stop.get("ack_count"),
               "error": stop.get("error")})
        check("stop-export-labelled-synthetic-not-qualified",
              stop.get("qualified") is None
              and stop.get("export_complete") is True,
              {"qualified": stop.get("qualified"),
               "export_complete": stop.get("export_complete")})
        manifest_path = stop.get("manifest")
        check("stop-cohort-manifest", bool(manifest_path)
              and Path(manifest_path).is_file(), manifest_path)
        rank_manifests = [row.get("manifest") for row in stop.get("ranks") or []]
        check("stop-rank-manifests", len(rank_manifests) == DEVICES
              and all(path and Path(path).is_file() for path in rank_manifests),
              rank_manifests)
        if rank_manifests and rank_manifests[0]:
            rank_manifest = json.loads(Path(rank_manifests[0]).read_text())
            check("no-profiler-step-calls",
                  rank_manifest.get("profiler_step_calls") == 0
                  and (rank_manifest.get("schedule") or {}).get("mode") == "none"
                  and (rank_manifest.get("schedule") or {}).get("step_calls") == 0,
                  rank_manifest.get("schedule"))
            check("engine-steps-at-stop-exceed-four",
                  int(rank_manifest.get("engine_steps_at_stop") or 0) >= 8,
                  rank_manifest.get("engine_steps_at_stop"))
            check("rank-manifest-synthetic",
                  rank_manifest.get("synthetic") is True,
                  rank_manifest.get("synthetic"))

        required = {"framework", "rank", "pid", "native_tid",
                    "time_time_ns_start", "time_time_ns_end",
                    "perf_counter_ns_start", "perf_counter_ns_end",
                    "thread_time_ns", "run_id", "event_name", "source",
                    "start", "end", "exception", "device_time"}
        parent_path = Path(ctx["out_dir"]) / ("events-%s.jsonl" % ctx["phase"])
        parent_rows = ([json.loads(line) for line in
                        parent_path.read_text().splitlines()] if parent_path.is_file()
                       else [])
        parent_names = {row.get("event_name") for row in parent_rows}
        check("parent-events-required-fields",
              bool(parent_rows) and all(required <= set(row) for row in parent_rows),
              parent_path)
        check("parent-http-events",
              {"http.parse", "http.admission", "http.response"} <= parent_names,
              sorted(name for name in parent_names if name and name.startswith("http")))
        check("parent-sse-events",
              {"sse.serialize", "sse.write"} <= parent_names,
              sorted(name for name in parent_names if name and name.startswith("sse")))
        check("parent-control-events",
              {"cohort_start", "cohort_stop"} <= parent_names,
              sorted(name for name in parent_names if name and name.startswith("cohort")))
        rank_path = Path(ctx["out_dir"]) / "rank00" / ("events-%s.jsonl" % ctx["phase"])
        rank_rows = ([json.loads(line) for line in
                      rank_path.read_text().splitlines()] if rank_path.is_file()
                     else [])
        rank_names = {row.get("event_name") for row in rank_rows}
        check("rank-events-required-fields",
              bool(rank_rows) and all(required <= set(row) for row in rank_rows),
              rank_path)
        check("rank-control-events",
              {"rank_profiler_start", "rank_profiler_stop"} <= rank_names,
              sorted(name for name in rank_names if name and name.startswith("rank_")))

        ctx2 = contexts("b")
        plain_result = {}

        def plain_request():
            plain_result["result"] = call(
                "POST", "/v1/completions",
                {"prompt_token_ids": [5, 6, 7, 8], "max_tokens": 20,
                 "request_id": "mock-plain-2"}, timeout=120.0)

        plain_thread = threading.Thread(target=plain_request, daemon=True)
        plain_thread.start()
        check("second-client-inflight", wait_inflight(10.0), "inflight > 0")
        code, busy = call("POST", "/e33/profile/start", ctx2)
        check("start-rejects-non-idle-group", code == 409,
              {"code": code, "error": busy.get("error")})
        plain_thread.join(timeout=120)
        plain_code, plain_body = plain_result.get("result", (None, {}))
        check("plain-request-completed", plain_code == 200
              and isinstance(plain_body, dict)
              and bool((plain_body.get("choices") or [{}])[0].get("token_ids")),
              {"code": plain_code})
        wait_idle(30.0)
        code, start2 = call("POST", "/e33/profile/start", ctx2)
        check("second-start-after-drain", code == 200
              and start2.get("ack_count") == DEVICES,
              {"code": code, "error": start2.get("error")})
        code, stop2 = call("POST", "/e33/profile/stop", ctx2)
        check("second-stop", code == 200 and stop2.get("ack_count") == DEVICES,
              {"code": code, "error": stop2.get("error")})

        outside_snapshot = {"run_id": "E33-MOCK-OUTSIDE", "case_id": "MOCK",
                            "phase": "outside",
                            "out_dir": "/tmp/e33-outside-snapshot"}
        code, body = call("POST", "/e33/snapshot", outside_snapshot)
        check("snapshot-rejects-outside-root", code == 400,
              {"code": code, "error": body.get("error")})

        exit_code = 0 if all(item["pass"] for item in checks) else 1
    except SystemExit:
        exit_code = 1
    except BaseException as exc:  # noqa: BLE001
        checks.append({"name": "mock-check-exception", "pass": False,
                       "detail": traceback.format_exc()[:8000]})
        exit_code = 1
    finally:
        if httpd is not None:
            httpd.shutdown()
            if server_thread is not None:
                server_thread.join(timeout=5)
        cluster.shutdown(force=True)
    result = {
        "tool": "e33-mock-check",
        "synthetic": True,
        "hardware_executed": False,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "root": str(root),
        "checks": checks,
        "ok": exit_code == 0 and all(item["pass"] for item in checks),
        "note": "CPU mock protocol check only; it cannot qualify a real capture",
    }
    out_path = root / "mock-check.json"
    try:
        atomic_write_json(out_path, result)
        result["path"] = str(out_path)
    except BaseException:  # noqa: BLE001
        pass
    print(json.dumps(result, indent=2, sort_keys=True))
    return exit_code


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    global _STARTUP_WARM
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--offline-diag", action="store_true")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--e33-mock-check", action="store_true",
                    help="run the local CPU-only E33 control protocol check")
    ap.add_argument("--e33-task-root", default=E33_TASK_ROOT,
                    help="only /e33 out_dir values below this root are accepted")
    ap.add_argument("--port", type=int, default=18121)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--jobs-json")
    ap.add_argument("--out", default="offline_diag.jsonl")
    ap.add_argument("--model", default=ClusterConfig.model)
    ap.add_argument("--force-topk", type=int, default=4096)
    ap.add_argument("--zos", action="store_true")
    ap.add_argument("--nextn", type=int, default=0)
    ap.add_argument("--mtp-varlen", action="store_true",
                    help="enable DSpark confidence-scheduled variable-length verification")
    ap.add_argument("--no-enforce-eager", dest="enforce_eager", action="store_false",
                    default=True, help="allow decode/DSpark graph capture")
    ap.add_argument("--prefill-attempt-interval", type=int, default=None)
    ap.add_argument("--prefill-any-attempt-interval", type=int, default=None)
    ap.add_argument("--max-num-seqs", type=int, default=1)
    ap.add_argument("--max-num-batched-tokens", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--trace-steps", action="store_true", default=False)
    ap.add_argument("--no-trace-steps", dest="trace_steps", action="store_false")
    ap.add_argument("--label", default="deepinfer-native-dp8")
    ap.add_argument("--instance-lock", default=None,
                    help="container-scoped flock held for the whole parent life; "
                         "prevents a second model group on the same experiment")
    ap.add_argument("--gemm-warm-manifest", default=None,
                    help="compile-only GEMM family manifest; run before rank start")
    ap.add_argument("--gemm-warm-helper",
                    default=None,
                    help="precompile helper path (default: next to this adapter)")
    ap.add_argument("--gemm-warm-report", default=None,
                    help="persistent readiness manifest path")
    ap.add_argument("--gemm-warm-cache", default=None,
                    help="DG_JIT_CACHE_DIR (default: process env)")
    ap.add_argument("--gemm-warm-max-m", type=int, default=512)
    ap.add_argument("--gemm-warm-workers", type=int, default=10)
    ap.add_argument("--gemm-warm-timeout", type=int, default=900)
    ap.add_argument("--sparse-mqa-warm-manifest", default=None,
                    help="compile-only sparse-MQA metadata manifest; run before "
                         "any rank starts")
    ap.add_argument("--sparse-mqa-warm-helper", default=None,
                    help="sparse-MQA precompile helper path (default: next to "
                         "this adapter)")
    ap.add_argument("--sparse-mqa-warm-report",
                    default="/work/runtime-20261021/e15-sparse-mqa-readiness.json",
                    help="persistent sparse-MQA readiness report")
    ap.add_argument("--sparse-mqa-warm-cache", default=None,
                    help="DG_JIT_CACHE_DIR (default: process env)")
    ap.add_argument("--sparse-mqa-warm-max-q", type=int, default=512)
    ap.add_argument("--sparse-mqa-warm-workers", type=int, default=8)
    ap.add_argument("--sparse-mqa-warm-timeout", type=int, default=600)
    ap.add_argument("--startup-warm", dest="startup_warm", action="store_true",
                    default=True,
                    help="run bounded real C1-sparse and C8 startup requests "
                         "before publishing HTTP readiness")
    ap.add_argument("--no-startup-warm", dest="startup_warm",
                    action="store_false",
                    help="skip the normal-loop startup warm (diagnostics only)")
    ap.add_argument("--startup-warm-prompts", default=None,
                    help="internal warm prompt JSON (default: e15_warm_prompts.json "
                         "next to this adapter)")
    ap.add_argument("--startup-warm-report",
                    default="/work/runtime-20261021/e15-startup-warm.json",
                    help="persistent startup warm evidence/readiness report")
    ap.add_argument("--startup-warm-timeout", type=float, default=240.0)
    ap.add_argument("--startup-warm-max-tokens", type=int, default=16)
    ap.add_argument("--diag-requests-path", default=None,
                    help="append per-request scheduler/latency phase records here")
    ap.add_argument("--profile-dir", default=None,
                    help="enable worker-side phase-gated profiling under this root")
    ap.add_argument("--profile-run-id", default=None)
    ap.add_argument("--profile-case-id", default="")
    ap.add_argument("--profile-baseline-run-id", default="")
    ap.add_argument("--profile-request-ids", default="",
                    help="comma-separated exact request ids that arm profiling")
    ap.add_argument("--profile-arm-after-dispatched", type=int, default=0,
                    help="arm when this many requests have been dispatched "
                         "(count includes warmups)")
    ap.add_argument("--profile-mode", choices=("prompt", "decode", "both"),
                    default="prompt")
    ap.add_argument("--profile-steps", type=int, default=8)
    ap.add_argument("--profile-prompt-windows", type=int, default=4)
    ap.add_argument("--profile-prompt-interval", type=int, default=100)
    ap.add_argument("--profile-prompt-trigger-moduli", default="100",
                    help="comma-separated decode_step moduli for prompt windows "
                         "(ANY=100 for C1; 70,100 when every rank has work)")
    ap.add_argument("--profile-prompt-trigger-k-fracs", default="",
                    help="comma-separated prompt-length fractions; start a "
                         "prompt window when the scheduler-reported max_seqlen_k "
                         "reaches that fraction of the scheduled prompt length")
    ap.add_argument("--profile-prompt-trigger-requires-local", action="store_true",
                    default=False)
    ap.add_argument("--profile-arm-lead", type=int, default=2)
    ap.add_argument("--profile-initial-prompt-window", action="store_true",
                    default=False)
    ap.add_argument("--profile-immediate-prompt-window", action="store_true",
                    default=False,
                    help="start the first prompt window at the first step after "
                         "arming (for ALL=ANY=1 where the first step may already prefill)")
    ap.add_argument("--profile-decode-trigger", type=int, default=4)
    ap.add_argument("--profile-decode-windows", type=int, default=1)
    ap.add_argument("--profile-level", default="Level1")
    ap.add_argument("--profile-record-shapes", action="store_true", default=False)
    ap.add_argument("--profile-with-stack", action="store_true", default=False)
    args = ap.parse_args()

    profile_plan = None
    if args.profile_dir:
        raise SystemExit(
            "E33 disables the legacy --profile-dir step-window controller; "
            "start a whole-cohort recording with POST /e33/profile/start and "
            "stop it explicitly")

    cfg = ClusterConfig(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        sampler_force_topk=args.force_topk,
        zos=args.zos,
        nextn=args.nextn,
        mtp_varlen=args.mtp_varlen,
        enforce_eager=args.enforce_eager,
        prefill_attempt_interval=args.prefill_attempt_interval,
        prefill_any_attempt_interval=args.prefill_any_attempt_interval,
        trace_steps=args.trace_steps,
        label=args.label,
        profile=profile_plan,
        diag_requests_path=args.diag_requests_path,
        e33_task_root=args.e33_task_root,
    )
    if args.e33_mock_check:
        return run_e33_mock_check(cfg)
    lock_fd = None
    if not args.mock:
        lock_path = args.instance_lock or ("/tmp/deepinfer-dp8-%d.lock" % args.port)
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.lseek(lock_fd, 0, os.SEEK_SET)
            holder = os.read(lock_fd, 300).decode(errors="replace").strip()
            print(json.dumps({"error": "instance lock busy",
                              "lock": lock_path, "holder": holder,
                              "detail": repr(exc)}, indent=2), flush=True)
            return 3
        os.ftruncate(lock_fd, 0)
        os.write(lock_fd, ("%d %.3f %s\n" % (os.getpid(), time.time(),
                                             args.label)).encode())
    if args.serve and not args.mock:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.bind((args.host, args.port))
        except OSError as exc:
            print(json.dumps({"error": "port already in use",
                              "host": args.host, "port": args.port,
                              "detail": repr(exc)}, indent=2), flush=True)
            return 3
        finally:
            probe.close()

    if args.offline_diag:
        if not args.jobs_json:
            raise SystemExit("--offline-diag requires --jobs-json")
        return run_offline_diag(cfg, args.jobs_json, args.out)

    if args.serve and not args.mock:
        run_startup_gemm_warm(args)
        sparse_warm = run_startup_sparse_mqa_warm(args)
        if sparse_warm.get("enabled") and not sparse_warm.get("coverage_ok"):
            print(json.dumps({
                "error": "startup sparse-MQA compile-only warm failed",
                "sparse_mqa_cache": dict(_STARTUP_SPARSE),
            }, indent=2), flush=True)
            return 2

    cluster = Cluster(cfg, mock=args.mock)
    if not cluster.start():
        print(json.dumps({"error": "startup failed", "fatal": cluster.fatal,
                          "detail": cluster.fatal_detail,
                          "status": cluster.status()}, indent=2), flush=True)
        cluster.shutdown(force=True)
        return 2
    adapter = AdapterState(cfg, cluster)
    if not args.mock:
        tokenizer_t0 = time.perf_counter()
        adapter.tokenizer.get()
        # Exercise the first decode before publishing ready or accepting a
        # request.  The lazy get() imports Transformers and loads the HF
        # tokenizer; _stream_completion decodes before writing its first
        # token-ID frame, so leaving this cold makes the first SSE frame wait
        # on an unrelated CPU load rather than model work.
        adapter.tokenizer.decode([1])
        print(json.dumps({
            "stage": "http_tokenizer_ready",
            "load_and_first_decode_s": round(
                time.perf_counter() - tokenizer_t0, 6),
        }), flush=True)
    if args.serve and not args.mock:
        if args.startup_warm:
            warm_ok = run_startup_normal_warm(adapter, args)
        else:
            _STARTUP_WARM = {"enabled": False, "completed": False,
                             "reason": "disabled by --no-startup-warm"}
            warm_ok = True
        if not warm_ok:
            print(json.dumps({
                "error": "startup normal warm failed",
                "startup_warm": dict(_STARTUP_WARM),
            }, indent=2), flush=True)
            cluster.shutdown()
            return 2
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    if _STARTUP_WARM.get("enabled") and _STARTUP_WARM.get("completed"):
        _STARTUP_WARM["http_listen_at"] = time.time()
        _STARTUP_WARM["public_ready_at"] = _STARTUP_WARM["http_listen_at"]
        report_path = _STARTUP_WARM.get("report_path")
        if report_path:
            _write_startup_warm_report(Path(report_path), _STARTUP_WARM)
    httpd.daemon_threads = True
    httpd.adapter_state = adapter  # type: ignore[attr-defined]
    print(json.dumps({
        "ready": bool(cluster.status()["ready"]),
        "framework": "deepinfer", "lane": "native-dp8",
        "port": args.port, "resolved": cluster.rank_resolved,
        "gemm_cache": dict(_STARTUP_CACHE),
        "sparse_mqa_cache": dict(_STARTUP_SPARSE),
        "config": cfg.public(),
    }), flush=True)

    def _term(signum, frame):
        threading.Thread(target=cluster.shutdown, daemon=True).start()
        httpd.shutdown()

    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)
    try:
        httpd.serve_forever(poll_interval=0.2)
    finally:
        cluster.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
