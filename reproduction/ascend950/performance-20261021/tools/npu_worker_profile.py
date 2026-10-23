#!/usr/bin/env python3
"""Worker-level Ascend NPU profiling harness and trace validator (v2).

Two independent layers:

1. WorkerProfiler (imported by the actual DeepInfer/vLLM worker process):
   wraps torch_npu.profiler around the scheduler step loop, marks a steady-state
   window with mstx, and writes profile_manifest.json.  It must run inside the
   process that owns the device, never around the HTTP parent.

2. Trace checker (CLI):
   verifies that a produced trace has the target rank/device, a steady-state
   window *found in the trace itself*, kernels inside that window, a complete
   stop marker, and an unprofiled baseline Run with the same execution and
   workload fingerprint.

Qualification is deliberately conservative:
   * a manifest-only steady declaration is NOT enough;
   * if the installed export format cannot be parsed, the verdict is
     structural_only / manual verification required;
   * synthetic traces are never qualified;
   * qualification also requires --accept-export-format (the operator confirms
     the parsed format matches the installed exporter);
   * a cohort must cover ranks 0..7 with distinct devices.

Static API check (no NPU initialization):
    python3 npu_worker_profile.py --check-api

Synthetic self-test (no NPU, clearly labelled):
    python3 npu_worker_profile.py --demo-trace /tmp/demo-trace --demo-ranks 8
    python3 npu_worker_profile.py --check-cohort /tmp/demo-trace --expect-ranks 8
"""
from __future__ import annotations

import argparse
import ast
import csv
import importlib.metadata
import inspect
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

REQUIRED_PROFILE_KWARGS = ["activities", "schedule", "on_trace_ready", "experimental_config"]
REQUIRED_SCHEDULE_ARGS = ["wait", "active", "warmup", "repeat", "skip_first", "skip_first_wait"]
REQUIRED_EXPERIMENTAL_ARGS = ["profiler_level", "aic_metrics", "mstx", "msprof_tx",
                              "data_simplification", "export_type"]

COMPUTE_EXCLUDE = re.compile(
    r"(memcpy|memset|memcpy_async|stream|notify|wait|event|barrier|sync|h2d|d2h|"
    r"record|range|marker|flush|launch)", re.I)
COMM_PATTERN = re.compile(
    r"(hccl|allreduce|all_reduce|allgather|all_gather|reducescatter|reduce_scatter|"
    r"alltoall|all_to_all|dispatch|combine|send|recv|broadcast)", re.I)
MARKER_FILE_PATTERNS = ("*msprof_tx*.csv", "*mstx*.csv", "*msprof_tx*.json")


# ---------------------------------------------------------------------------
# Static API check
# ---------------------------------------------------------------------------

def _signature(node: ast.AST) -> List[str]:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return []
    args = node.args
    names = [arg.arg for arg in list(getattr(args, "posonlyargs", [])) + list(args.args)]
    names += [arg.arg for arg in args.kwonlyargs]
    return names


def _find_def(tree: ast.AST, name: str) -> Optional[ast.AST]:
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name:
            if isinstance(node, ast.ClassDef):
                for sub in node.body:
                    if isinstance(sub, ast.FunctionDef) and sub.name == "__init__":
                        return sub
                return node
            return node
    return None


def check_api() -> Dict[str, Any]:
    import importlib.util
    try:
        spec = importlib.util.find_spec("torch_npu")
    except Exception:  # noqa: BLE001
        spec = None
    locations = list(getattr(spec, "submodule_search_locations", None) or [])
    package_dir = Path(locations[0]).resolve() if locations else None
    result: Dict[str, Any] = {
        "python": sys.executable,
        "torch_npu_version": None,
        "package_dir": str(package_dir) if package_dir else None,
        "ok": False,
        "signatures": {},
        "missing": [],
    }
    try:
        result["torch_npu_version"] = importlib.metadata.version("torch_npu")
    except Exception:  # noqa: BLE001
        pass
    if package_dir is None:
        result["missing"].append("torch_npu package not resolvable")
        return result
    profiler_dir = package_dir / "profiler"
    files = {
        "profile": profiler_dir / "profiler.py",
        "schedule": profiler_dir / "scheduler.py",
        "experimental": profiler_dir / "experimental_config.py",
        "init": profiler_dir / "__init__.py",
        "mstx": package_dir / "npu" / "mstx.py",
    }
    for label, path in files.items():
        if not path.is_file():
            result["missing"].append("missing %s" % path)
            continue
        with open(path) as fh:
            tree = ast.parse(fh.read(), filename=str(path))
        if label == "init":
            aliases = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        aliases[alias.asname or alias.name] = alias.name
            result["signatures"]["exports_schedule_alias"] = aliases.get("schedule")
            result["signatures"]["exports_ProfilerActivity"] = "from torch_npu._C" in path.read_text()
            continue
        if label == "mstx":
            result["signatures"]["mstx_methods"] = {
                sub.name: _signature(sub)
                for sub in ast.walk(tree)
                if isinstance(sub, ast.FunctionDef) and sub.name in ("range_start", "range_end", "mstx_range")
            }
            continue
        names = {"profile": "profile", "schedule": "Schedule",
                 "experimental": "_ExperimentalConfig"}[label]
        node = _find_def(tree, names)
        if node is None:
            result["missing"].append("%s not found in %s" % (names, path))
            result["signatures"][names] = None
        else:
            result["signatures"][names] = _signature(node)

    required = {
        "profile": REQUIRED_PROFILE_KWARGS,
        "Schedule": REQUIRED_SCHEDULE_ARGS,
        "_ExperimentalConfig": REQUIRED_EXPERIMENTAL_ARGS,
    }
    for name, wanted in required.items():
        got = result["signatures"].get(name)
        if got is None:
            result["missing"].append(name + " signature unavailable")
            continue
        for arg in wanted:
            if arg not in got:
                result["missing"].append("%s.%s missing" % (name, arg))
    if result["signatures"].get("exports_schedule_alias") != "Schedule":
        result["missing"].append("torch_npu.profiler does not export schedule=Schedule")
    mstx = result["signatures"].get("mstx_methods") or {}
    for method in ("range_start", "range_end"):
        if method not in mstx:
            result["missing"].append("torch_npu.npu.mstx.%s missing" % method)
    result["ok"] = not result["missing"]
    result["required_kwargs"] = required
    return result


# ---------------------------------------------------------------------------
# Worker-side profiler
# ---------------------------------------------------------------------------

class WorkerProfiler:
    """Wrap torch_npu.profiler around one worker's step loop.

    Usage inside a worker process:
        prof = WorkerProfiler(out_dir, run_id, rank, device, baseline_run_id, case_id,
                              wait=2, warmup=1, active=3, repeat=1)
        prof.start()
        for step in range(...):
            ...
            prof.step()
        prof.stop()
    """

    def __init__(self, out_dir: str, run_id: str, rank: int, device: int,
                 baseline_run_id: str = "", case_id: str = "",
                 wait: int = 2, active: int = 3, warmup: int = 1, repeat: int = 1,
                 skip_first: int = 0, profiler_level: str = "Level1",
                 record_shapes: bool = False, with_stack: bool = False,
                 synthetic: bool = False, worker_name: str | None = None) -> None:
        self.out_dir = str(out_dir)
        self.run_id = run_id
        self.rank = rank
        self.device = device
        self.baseline_run_id = baseline_run_id
        self.case_id = case_id
        self.wait = wait
        self.active = active
        self.warmup = warmup
        self.repeat = repeat
        self.skip_first = skip_first
        self.profiler_level = profiler_level
        self.record_shapes = record_shapes
        self.with_stack = with_stack
        self.synthetic = synthetic
        self.worker_name = worker_name or ("rank%02d" % rank)
        self.steps = 0
        self.steady_ranges: List[Dict[str, Any]] = []
        self._prof = None
        self._mstx = None
        self._steady_id = 0
        self._steady_wall_ns = 0
        self._started_wall_ns = 0
        self._api_used: Dict[str, Any] = {}

    def start(self) -> None:
        import torch_npu
        from torch_npu.profiler import (
            ProfilerActivity, _ExperimentalConfig, profile, schedule,
            tensorboard_trace_handler,
        )
        from torch_npu.npu import mstx

        level = getattr(torch_npu.profiler.ProfilerLevel, self.profiler_level)
        experimental_kwargs = {
            "profiler_level": level,
            "mstx": True,
            "msprof_tx": True,
            "data_simplification": True,
        }
        supported = inspect.signature(_ExperimentalConfig.__init__).parameters
        if not ({"mstx", "msprof_tx"} & set(supported)):
            raise RuntimeError("Installed profiler cannot enable MSTX ranges")
        experimental_kwargs = {key: value for key, value in experimental_kwargs.items()
                               if key in supported}
        config = _ExperimentalConfig(**experimental_kwargs)
        self._mstx = mstx
        self._prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
            schedule=schedule(wait=self.wait, active=self.active, warmup=self.warmup,
                              repeat=self.repeat, skip_first=self.skip_first),
            on_trace_ready=tensorboard_trace_handler(
                dir_name=self.out_dir,
                worker_name=self.worker_name,
                analyse_flag=True,
                async_mode=False),
            record_shapes=self.record_shapes,
            with_stack=self.with_stack,
            experimental_config=config,
        )
        self._api_used = {
            "profile": "torch_npu.profiler.profile",
            "activities": ["CPU", "NPU"],
            "schedule": {"wait": self.wait, "active": self.active, "warmup": self.warmup,
                         "repeat": self.repeat, "skip_first": self.skip_first,
                         "skip_first_wait": 0},
            "handler": {"dir_name": self.out_dir, "worker_name": self.worker_name,
                        "analyse_flag": True, "async_mode": False},
            "experimental_config": {key: str(value) for key, value in experimental_kwargs.items()},
            "torch_npu_version": getattr(torch_npu, "__version__", "unknown"),
        }
        self._prof.start()
        self._started_wall_ns = time.time_ns()

    def begin_steady(self, label: str) -> None:
        if self._mstx is None:
            raise RuntimeError("profiler is not started")
        self._steady_wall_ns = time.time_ns()
        self._steady_id = self._mstx.range_start("steady_state:%s" % label, domain="bench")

    def end_steady(self) -> None:
        if self._mstx is None or not self._steady_id:
            raise RuntimeError("steady range was not started")
        self._mstx.range_end(self._steady_id, domain="bench")
        self.steady_ranges.append({
            "label": "steady_state",
            "start_wall_ns": self._steady_wall_ns,
            "end_wall_ns": time.time_ns(),
            "mstx_range_id": self._steady_id,
            "steps": self.steps,
            "message": "steady_state",
            "domain": "bench",
        })
        self._steady_id = 0

    def step(self) -> None:
        if self._prof is None:
            raise RuntimeError("profiler is not started")
        self._prof.step()
        self.steps += 1

    def stop(self) -> Dict[str, Any]:
        if self._prof is None:
            raise RuntimeError("profiler is not started")
        self._prof.stop()
        self._prof = None
        stopped_wall_ns = time.time_ns()
        manifest = {
            "schema_version": "2.0",
            "synthetic": self.synthetic,
            "run_id": self.run_id,
            "baseline_run_id": self.baseline_run_id,
            "case_id": self.case_id,
            "rank": self.rank,
            "device": self.device,
            "out_dir": self.out_dir,
            "started_wall_ns": self._started_wall_ns,
            "stopped_wall_ns": stopped_wall_ns,
            "stopped": True,
            "stop_complete": True,
            "steps": self.steps,
            "steady_state": {
                "present": bool(self.steady_ranges),
                "ranges": self.steady_ranges,
                "domain": "bench",
                "note": "wall-clock + mstx range id; the trace-side marker is verified separately",
            },
            "api": self._api_used,
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        target = Path(self.out_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "profile_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return manifest


# ---------------------------------------------------------------------------
# Trace validation
# ---------------------------------------------------------------------------

NAME_COLUMNS = ("Name", "Kernel Name", "Op Name", "name")
DEVICE_COLUMNS = ("Device ID", "NPU ID", "DeviceId", "device_id", "Device")
START_COLUMNS = ("Start Time(us)", "Start Time", "Start Time(ns)", "start_time")
DURATION_COLUMNS = ("Duration(us)", "Duration", "Duration(ns)", "duration")
MARKER_MSG_COLUMNS = ("Message", "message", "Name", "Event Name", "Range Name")
MARKER_START_COLUMNS = ("Start Time(us)", "Start Time", "Start Time(ns)", "Timestamp(us)")
MARKER_END_COLUMNS = ("End Time(us)", "End Time", "End Time(ns)", "Duration(us)", "Duration")


def _pick(header: List[str], candidates: Tuple[str, ...]) -> Optional[str]:
    stripped = {column.strip(): column for column in header}
    for candidate in candidates:
        if candidate in stripped:
            return stripped[candidate]
    for column in header:
        for candidate in candidates:
            if candidate.lower() in column.lower():
                return column
    return None


def _to_us(value: float, column: str) -> float:
    return value / 1000.0 if "ns" in (column or "").lower() else value


def _scan_csv(path: Path, limit: int) -> Dict[str, Any]:
    info = {"path": str(path), "rows": 0, "devices": [], "names": {}, "compute": 0,
            "communication": 0, "bookkeeping": 0, "start_min": None, "start_max": None,
            "duration_sum": 0.0}
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        name_col = _pick(header, NAME_COLUMNS)
        device_col = _pick(header, DEVICE_COLUMNS)
        start_col = _pick(header, START_COLUMNS)
        duration_col = _pick(header, DURATION_COLUMNS)
        info["columns"] = {"name": name_col, "device": device_col,
                           "start": start_col, "duration": duration_col}
        if name_col is None:
            info["error"] = "no kernel-name column"
            return info
        devices = set()
        starts = []
        for row in reader:
            if info["rows"] >= limit:
                break
            info["rows"] += 1
            name = (row.get(name_col) or "").strip()
            info["names"][name] = info["names"].get(name, 0) + 1
            if COMM_PATTERN.search(name):
                info["communication"] += 1
            elif COMPUTE_EXCLUDE.search(name):
                info["bookkeeping"] += 1
            else:
                info["compute"] += 1
            if device_col and row.get(device_col):
                devices.add(str(row[device_col]).strip())
            if start_col and row.get(start_col):
                try:
                    starts.append(_to_us(float(row[start_col]), start_col))
                except ValueError:
                    pass
            if duration_col and row.get(duration_col):
                try:
                    info["duration_sum"] += float(row[duration_col])
                except ValueError:
                    pass
    info["devices"] = sorted(devices)
    info["start_us"] = starts
    info["start_min"] = min(starts) if starts else None
    info["start_max"] = max(starts) if starts else None
    info["distinct_ops"] = len(info["names"])
    top = sorted(info["names"].items(), key=lambda item: item[1], reverse=True)[:10]
    info["top_ops"] = [{"name": name, "count": count} for name, count in top]
    info["names"] = None
    return info


def _events_from_trace_view(path: Path, stage: str) -> Tuple[List[Dict[str, Any]], int]:
    """Read MSTX/ProfilerStep ranges from an Ascend trace_view.json.

    The Text export writes kernel_details.csv plus trace_view.json; MSTX
    ranges appear in the chrome-trace file, not necessarily as a CSV.  We
    accept X events, B/E pairs keyed by (pid, tid), and instant events.
    """
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return [], 0
    if isinstance(payload, dict):
        payload = payload.get("traceEvents") or []
    if not isinstance(payload, list):
        return [], 0
    rows: List[Dict[str, Any]] = []
    profiler_steps = 0
    open_ranges: Dict[Tuple[Any, Any], List[Dict[str, Any]]] = {}
    for event in payload:
        if not isinstance(event, dict):
            continue
        name = str(event.get("name") or "")
        if "ProfilerStep#" in name:
            profiler_steps += 1
        if stage not in name:
            continue
        ts = event.get("ts")
        if ts is None:
            continue
        try:
            ts = float(ts)
        except (TypeError, ValueError):
            continue
        key = (event.get("pid"), event.get("tid"))
        phase_char = event.get("ph")
        if phase_char == "X":
            dur = event.get("dur")
            end = ts + float(dur) if dur is not None else None
            rows.append({"file": str(path), "message": name, "start_us": ts,
                         "end_us": end, "device": event.get("device")})
        elif phase_char == "B":
            open_ranges.setdefault(key, []).append({"name": name, "start": ts})
        elif phase_char == "E":
            stack = open_ranges.get(key) or []
            for index in range(len(stack) - 1, -1, -1):
                if stage in stack[index]["name"]:
                    opened = stack.pop(index)
                    rows.append({"file": str(path), "message": opened["name"],
                                 "start_us": opened["start"], "end_us": ts,
                                 "device": event.get("device")})
                    break
        elif phase_char == "i":
            rows.append({"file": str(path), "message": name, "start_us": ts,
                         "end_us": ts, "device": event.get("device")})
    return rows, profiler_steps


def _marker_windows(trace_dir: Path, stage: str) -> Dict[str, Any]:
    """Find steady_state markers inside the exported trace, not in our manifest."""
    files = []
    for pattern in MARKER_FILE_PATTERNS:
        files.extend(trace_dir.rglob(pattern))
    files = sorted(set(files))
    rows = []
    files_scanned = []
    trace_views = sorted(set(trace_dir.rglob("trace_view.json")))
    profiler_steps = 0
    for path in trace_views:
        view_rows, steps = _events_from_trace_view(path, stage)
        files_scanned.append(str(path))
        rows.extend(view_rows)
        profiler_steps += steps
    for path in files:
        if path.suffix == ".json":
            continue
        files_scanned.append(str(path))
        try:
            with path.open(newline="") as fh:
                reader = csv.DictReader(fh)
                header = reader.fieldnames or []
                msg_col = _pick(header, MARKER_MSG_COLUMNS)
                start_col = _pick(header, MARKER_START_COLUMNS)
                end_col = _pick(header, MARKER_END_COLUMNS)
                device_col = _pick(header, DEVICE_COLUMNS)
                if msg_col is None or start_col is None:
                    continue
                for index, row in enumerate(reader):
                    if index > 200000:
                        break
                    message = (row.get(msg_col) or "").strip()
                    if stage not in message:
                        continue
                    try:
                        start = _to_us(float(row[start_col]), start_col)
                    except (TypeError, ValueError):
                        continue
                    end = None
                    if end_col and row.get(end_col):
                        try:
                            end = _to_us(float(row[end_col]), end_col)
                        except (TypeError, ValueError):
                            end = None
                    if end is not None and "Duration" in end_col:
                        end = start + end
                    rows.append({"file": str(path), "message": message,
                                 "start_us": start, "end_us": end,
                                 "device": (row.get(device_col) if device_col else None)})
        except (OSError, csv.Error):
            continue
    start = min((row["start_us"] for row in rows), default=None)
    ends = [row["end_us"] for row in rows if row["end_us"] is not None]
    end = max(ends) if ends else None
    window = {"start_us": start, "end_us": end} if rows and start is not None and end is not None else None
    return {"files_scanned": files_scanned, "markers": rows, "window": window,
            "profiler_step_markers": profiler_steps,
            "format_assumption": "Ascend Text export: trace_view.json chrome events "
                                 "(MSTX ranges + ProfilerStep#n) and/or msprof_tx CSV"}


def _kernels_in_window(scans: List[Dict[str, Any]], start: float, end: float,
                       tolerance: float = 1.0) -> int:
    count = 0
    for scan in scans:
        for value in scan.get("start_us") or []:
            if start - tolerance <= value <= end + tolerance:
                count += 1
    return count


def check_trace(trace_dir: Path, manifest_path: Optional[Path], rank: Optional[int],
                device: Optional[int], expect_ops: List[str], min_steps: int = 3,
                profile_run: Optional[Path] = None,
                baseline_run: Optional[Path] = None,
                require_real: bool = False,
                require_link: bool = True,
                accept_export_format: bool = False,
                stage: str = "steady_state",
                row_limit: int = 200000) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    def record(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    trace_dir = Path(trace_dir)
    manifest: Dict[str, Any] = {}
    if manifest_path is None:
        candidate = trace_dir / "profile_manifest.json"
        if candidate.is_file():
            manifest_path = candidate
    if manifest_path and Path(manifest_path).is_file():
        manifest = json.loads(Path(manifest_path).read_text())
    record("manifest_present", bool(manifest), str(manifest_path or "not found"))
    record("manifest_synthetic_explicit",
           isinstance(manifest.get("synthetic"), bool),
           "synthetic=%r" % manifest.get("synthetic"))

    snapshot = (manifest.get("steady_state") or {})
    record("steady_manifest_present", bool(snapshot.get("present") and snapshot.get("ranges")),
           "manifest ranges=%d (not sufficient alone)" % len(snapshot.get("ranges") or []))

    csv_paths = sorted(set(trace_dir.rglob("kernel_details*.csv")) |
                       set(trace_dir.rglob("op_summary*.csv")))
    record("trace_files_present", bool(csv_paths), "%d kernel csv file(s)" % len(csv_paths))
    scans = [_scan_csv(path, row_limit) for path in csv_paths]
    rows = sum(scan["rows"] for scan in scans)
    compute = sum(scan["compute"] for scan in scans)
    communication = sum(scan["communication"] for scan in scans)
    bookkeeping = sum(scan["bookkeeping"] for scan in scans)
    devices = sorted({value for scan in scans for value in scan["devices"]})
    record("kernel_rows", rows > 0, "rows=%d" % rows)
    record("actual_ops", (compute + communication) > 0,
           "compute=%d communication=%d bookkeeping=%d" % (compute, communication, bookkeeping))
    record("device_column", any(scan.get("columns", {}).get("device") for scan in scans),
           "columns=%r" % [scan.get("columns", {}).get("device") for scan in scans])
    if device is not None:
        record("device_present", str(device) in devices, "target=%r found=%r" % (device, devices))
    if rank is not None:
        record("rank_matches_manifest", manifest.get("rank") == rank,
               "target=%r manifest=%r" % (rank, manifest.get("rank")))
        name_hit = any(re.search(r"rank[_-]?0*%d(?!\d)" % rank, str(path), re.I) for path in csv_paths)
        record("rank_in_trace_path", name_hit, "paths=%r" % [path.name for path in csv_paths[:3]])

    steps = int(manifest.get("steps") or 0)
    record("stopped_complete",
           manifest.get("stopped") is True and manifest.get("stop_complete") is True
           and manifest.get("stopped_wall_ns") is not None,
           "stopped=%r stop_complete=%r" % (manifest.get("stopped"), manifest.get("stop_complete")))
    record("steps_recorded", steps >= min_steps, "steps=%d min=%d" % (steps, min_steps))

    markers = _marker_windows(trace_dir, stage)
    record("mstx_marker_in_trace", bool(markers["markers"]),
           "files=%r markers=%d" % (markers["files_scanned"], len(markers["markers"])))
    kernels_in_window = 0
    if markers.get("window"):
        kernels_in_window = _kernels_in_window(scans, markers["window"]["start_us"],
                                               markers["window"]["end_us"])
    record("kernels_in_steady_window", kernels_in_window > 0,
           "kernels=%d window=%r" % (kernels_in_window, markers.get("window")))
    record("kernel_coverage_of_steps", kernels_in_window >= min_steps,
           "kernels_in_window=%d steps=%d" % (kernels_in_window, steps))

    records = manifest.get("profiler_step_records")
    if records:
        engine_steps = [int(item.get("engine_step", -1)) for item in records]
        contiguous = (engine_steps == list(range(engine_steps[0], engine_steps[0] + len(records)))
                      if engine_steps else False)
        record("worker_step_records", len(records) >= min_steps,
               "records=%d phases=%r" % (len(records),
                                         sorted({item.get("step_type") for item in records})))
        record("worker_steps_contiguous", contiguous, "engine_steps=%r" % engine_steps)
        record("phase_steps_in_trace",
               markers.get("profiler_step_markers", 0) >= len(records),
               "ProfilerStep markers=%d records=%d"
               % (markers.get("profiler_step_markers", 0), len(records)))
        record("window_label_matches_manifest",
               bool(manifest.get("window_label")),
               "window_label=%r" % manifest.get("window_label"))

    for wanted in expect_ops:
        hits = 0
        for path in csv_paths:
            with path.open(newline="") as fh:
                reader = csv.DictReader(fh)
                header = reader.fieldnames or []
                name_col = _pick(header, NAME_COLUMNS)
                if name_col is None:
                    continue
                for index, row in enumerate(reader):
                    if index >= row_limit:
                        break
                    if wanted.lower() in (row.get(name_col) or "").lower():
                        hits += 1
        record("expected_op:%s" % wanted, hits > 0, "hits=%d" % hits)

    link_ok = False
    link_detail = "not checked"
    if profile_run is not None and baseline_run is not None:
        from dinfer_bench import gates, model
        try:
            baseline = model.load_run(Path(baseline_run), model.load_cases())
            profile = model.load_run(Path(profile_run), model.load_cases())
            link = gates.check_profile_link(profile, baseline)
            link_ok = link["ok"]
            link_detail = "violations=%r" % [item["code"] for item in link["violations"]]
        except Exception as exc:  # noqa: BLE001
            link_detail = "load error: %r" % exc
    elif not require_link:
        link_ok = True
        link_detail = "link check disabled for structural tool test"
    record("baseline_link", link_ok, link_detail)

    if require_real:
        record("not_synthetic", manifest.get("synthetic") is False,
               "synthetic must be explicitly false, got %r" % manifest.get("synthetic"))
    record("export_format_accepted", accept_export_format,
           "operator must confirm the parsed marker format matches the installed exporter")

    structural_ok = all(item["pass"] for item in checks
                        if item["check"] != "export_format_accepted")
    qualified = bool(structural_ok
                     and manifest.get("synthetic") is False
                     and accept_export_format
                     and link_ok
                     and markers["markers"]
                     and kernels_in_window > 0)
    reasons = [item["check"] for item in checks if not item["pass"]]
    if manifest.get("synthetic") is not False:
        reasons.append("synthetic_not_false")
    if not accept_export_format:
        reasons.append("export_format_not_accepted")
    if not link_ok:
        reasons.append("baseline_link_failed")
    return {
        "ok": structural_ok,
        "qualified": qualified,
        "verdict": "qualified" if qualified else "structural_only",
        "qualification_reasons": sorted(set(reasons)),
        "synthetic_fixture": manifest.get("synthetic") is True,
        "trace_dir": str(trace_dir),
        "manifest": str(manifest_path) if manifest_path else None,
        "checks": checks,
        "markers": markers,
        "stats": {
            "csv_files": len(csv_paths),
            "rows": rows,
            "compute_ops": compute,
            "communication_ops": communication,
            "bookkeeping_rows": bookkeeping,
            "devices": devices,
            "steps": steps,
            "kernels_in_steady_window": kernels_in_window,
            "scans": [{key: value for key, value in scan.items()
                       if key not in ("names", "start_us")} for scan in scans],
        },
        "limits": [
            "Host and device clocks are not aligned here; the steady window is taken from marker rows in the trace and kernel start times in the same CSV unit, not from a reconstructed device timeline.",
            "The marker CSV format is an assumption until the operator accepts it for the installed exporter.",
            "kernel rows are Ascend kernel rows; op_summary is only a fallback.",
        ],
    }


def check_cohort(parent: Path, expect_ranks: int = 8, stage: str = "steady_state",
                 require_real: bool = False, require_link: bool = True,
                 accept_export_format: bool = False,
                 window: Optional[str] = None) -> Dict[str, Any]:
    parent = Path(parent)
    candidates = []
    for manifest in sorted(parent.rglob("profile_manifest.json")):
        directory = manifest.parent
        if directory == parent:
            continue
        data = json.loads(manifest.read_text())
        if window is not None and data.get("window_label") != window:
            continue
        candidates.append({"dir": directory, "manifest": data})
    if window is None and any(item["manifest"].get("window_label")
                              for item in candidates):
        labels = sorted({item["manifest"].get("window_label") for item in candidates})
        if len(labels) > 1:
            raise ValueError("cohort has multiple windows %r; pass --window" % labels)
    manifests = candidates
    ranks = [item["manifest"].get("rank") for item in manifests]
    devices = [item["manifest"].get("device") for item in manifests]
    checks = []
    checks.append({"check": "cohort_rank_count", "pass": len(manifests) == expect_ranks,
                   "detail": "found=%d expected=%d" % (len(manifests), expect_ranks)})
    checks.append({"check": "cohort_ranks_complete",
                   "pass": sorted(r for r in ranks if isinstance(r, int)) == list(range(expect_ranks)),
                   "detail": "ranks=%r" % sorted(r for r in ranks if isinstance(r, int))})
    checks.append({"check": "cohort_devices_distinct",
                   "pass": len(set(devices)) == len(devices) and len(devices) == expect_ranks,
                   "detail": "devices=%r" % devices})
    per_rank = []
    for item in manifests:
        manifest = item["manifest"]
        result = check_trace(item["dir"], item["dir"] / "profile_manifest.json",
                             manifest.get("rank"), manifest.get("device"), [],
                             require_real=require_real, require_link=require_link,
                             accept_export_format=accept_export_format, stage=stage)
        per_rank.append({"rank": manifest.get("rank"), "device": manifest.get("device"),
                         "ok": result["ok"], "qualified": result["qualified"],
                         "reasons": result["qualification_reasons"],
                         "kernels_in_steady_window": result["stats"]["kernels_in_steady_window"]})
    checks.append({"check": "cohort_all_ranks_structural", "pass": all(r["ok"] for r in per_rank),
                   "detail": "%d rank(s) failed" % sum(1 for r in per_rank if not r["ok"])})
    qualified = (not any(not c["pass"] for c in checks)
                 and len(per_rank) == expect_ranks
                 and all(r["qualified"] for r in per_rank))
    return {
        "ok": not any(not c["pass"] for c in checks),
        "qualified": qualified,
        "verdict": "qualified" if qualified else "structural_only",
        "expect_ranks": expect_ranks,
        "window": window,
        "checks": checks,
        "ranks": per_rank,
        "limits": ["A cohort verdict is only as strong as each rank's trace-side marker and kernel window."],
    }


# ---------------------------------------------------------------------------
# Synthetic trace fixture (tool test only)
# ---------------------------------------------------------------------------

def make_demo_trace(out_dir: Path, ranks: int = 8) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_template = [
        ("matmul_fp8", "AI_CORE", 12.5),
        ("flash_mla_prefill", "AI_CORE", 40.0),
        ("allreduce_tp", "COMM", 8.0),
        ("memcpy_h2d", "MEMCPY", 1.5),
        ("moe_dispatch", "AI_CORE", 22.0),
        ("matmul_fp8", "AI_CORE", 12.4),
        ("flash_mla_decode", "AI_CORE", 9.5),
        ("allgather_engram", "COMM", 3.0),
    ]
    for rank in range(ranks):
        rank_dir = out_dir / ("rank%02d_20261021_demo_ascend_pt" % rank)
        data_dir = rank_dir / "ASCEND_PROFILER_OUTPUT"
        data_dir.mkdir(parents=True, exist_ok=True)
        (rank_dir / "profiler_info.json").write_text(json.dumps({
            "rank_id": rank, "device_id": rank, "synthetic": True,
            "note": "synthetic fixture for the trace checker; not an NPU measurement",
        }, indent=2) + "\n")
        start = 1000.0 + rank * 10
        with (data_dir / "kernel_details.csv").open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["Device ID", "Name", "Type", "Start Time(us)", "Duration(us)"])
            for iteration in range(4):
                for position, (name, kind, duration) in enumerate(rows_template):
                    writer.writerow([rank, name, kind,
                                     start + iteration * 100 + position, duration])
        with (data_dir / "msprof_tx.csv").open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["Device ID", "Message", "Start Time(us)", "End Time(us)"])
            writer.writerow([rank, "steady_state:decode", start + 100, start + 350])
    manifest = {
        "schema_version": "2.0",
        "synthetic": True,
        "run_id": "E01-PROFILE-DEMO",
        "baseline_run_id": "E01-BASELINE-DEMO",
        "case_id": "D01",
        "steps": 4,
        "started_wall_ns": time.time_ns(),
        "stopped_wall_ns": time.time_ns(),
        "stopped": True,
        "stop_complete": True,
        "steady_state": {"present": True, "domain": "bench", "ranges": [
            {"label": "steady_state", "start_wall_ns": time.time_ns(),
             "end_wall_ns": time.time_ns(), "mstx_range_id": 42, "steps": 4}]},
        "api": {"note": "synthetic"},
    }
    for rank in range(ranks):
        item = dict(manifest)
        item["rank"] = rank
        item["device"] = rank
        item["out_dir"] = str(out_dir / ("rank%02d_20261021_demo_ascend_pt" % rank))
        (out_dir / ("rank%02d_20261021_demo_ascend_pt" % rank) / "profile_manifest.json").write_text(
            json.dumps(item, indent=2) + "\n")
    (out_dir / "profile_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check-api", action="store_true")
    parser.add_argument("--check-trace", default=None)
    parser.add_argument("--check-cohort", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--expect-op", action="append", default=[])
    parser.add_argument("--min-steps", type=int, default=3)
    parser.add_argument("--profile-run", default=None)
    parser.add_argument("--baseline-run", default=None)
    parser.add_argument("--require-real", action="store_true")
    parser.add_argument("--no-require-link", dest="require_link", action="store_false", default=True)
    parser.add_argument("--accept-export-format", action="store_true",
                        help="operator confirms the parsed marker CSV matches the installed exporter")
    parser.add_argument("--stage", default="steady_state")
    parser.add_argument("--expect-ranks", type=int, default=8)
    parser.add_argument("--window", default=None,
                        help="window_label to select when a cohort has several windows")
    parser.add_argument("--require-qualified", action="store_true",
                        help="exit non-zero unless the trace/cohort is qualified")
    parser.add_argument("--demo-trace", default=None)
    parser.add_argument("--demo-ranks", type=int, default=8)
    args = parser.parse_args()

    if args.check_api:
        print(json.dumps(check_api(), indent=2, sort_keys=True))
        return 0
    if args.demo_trace:
        print(str(make_demo_trace(Path(args.demo_trace), args.demo_ranks)))
        return 0
    if args.check_trace:
        result = check_trace(Path(args.check_trace),
                             Path(args.manifest) if args.manifest else None,
                             args.rank, args.device, args.expect_op, args.min_steps,
                             Path(args.profile_run) if args.profile_run else None,
                             Path(args.baseline_run) if args.baseline_run else None,
                             args.require_real, args.require_link,
                             args.accept_export_format, args.stage)
        print(json.dumps(result, indent=2, sort_keys=True))
        if args.require_qualified and not result["qualified"]:
            return 1
        return 0 if result["ok"] else 1
    if args.check_cohort:
        result = check_cohort(Path(args.check_cohort), args.expect_ranks, args.stage,
                              args.require_real, args.require_link,
                              args.accept_export_format, args.window)
        print(json.dumps(result, indent=2, sort_keys=True))
        if args.require_qualified and not result["qualified"]:
            return 1
        return 0 if result["ok"] else 1
    parser.error("choose --check-api, --check-trace, --check-cohort or --demo-trace")
    return 2


if __name__ == "__main__":
    sys.exit(main())
