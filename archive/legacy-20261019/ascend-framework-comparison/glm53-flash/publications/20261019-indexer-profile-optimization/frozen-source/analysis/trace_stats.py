"""Native torch_npu Text-export statistics for R247.

The analyzer reads only files produced by the native profiler plus the
observer's host-metadata JSONL.  It never invents a forward or a context.
Kernel durations are unioned as intervals; overlapping work is never summed
into a claimed critical-path share.  Native WAIT/NOTIFY/SQE tasks are control,
not compute, and are reported separately.
"""
from __future__ import annotations

import csv
import gzip
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "observers/common"))
import r234_common as C


def _num(value):
    if value is None or value == "":
        return 0.0
    return float(str(value).strip())


CONTROL_TASK_RE = re.compile(r"(^|[^A-Z0-9])(WAIT|NOTIFY|SQE)([^A-Z0-9]|$)")


def is_control_task(name, task_type=""):
    """Native runtime control tasks are never device compute."""
    text = (str(name or "") + " " + str(task_type or "")).upper()
    return bool(CONTROL_TASK_RE.search(text))


def union_intervals(intervals):
    """Return the normalized, merged intervals of one input side."""
    rows = sorted((max(0.0, lo), max(0.0, hi)) for lo, hi in intervals if hi > lo)
    merged = []
    for lo, hi in rows:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def union_us(intervals):
    return sum(hi - lo for lo, hi in union_intervals(intervals))


def intersection_us(a, b):
    """Measure the intersection of two interval sets after unioning each side.

    Intersecting raw lists assumes each side is already disjoint; overlapping
    or nested intervals inside one side would otherwise be counted more than
    once (e.g. [(0,10),(5,15)] against [(0,20)] would report 20 us, not 15 us).
    Both sides are therefore unioned first.
    """
    a = union_intervals(a)
    b = union_intervals(b)
    i = j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if hi > lo:
            total += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def clipped(intervals, lo, hi):
    out = []
    for a, b in intervals:
        l, h = max(a, lo), min(b, hi)
        if h > l:
            out.append((l, h))
    return out


def _open(path):
    path = pathlib.Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def load_trace(path):
    with _open(path) as stream:
        data = json.load(stream)
    return data if isinstance(data, list) else data.get("traceEvents", [])


def process_maps(events):
    names, threads = {}, {}
    for event in events:
        if event.get("ph") != "M":
            continue
        pid, tid = event.get("pid"), event.get("tid")
        if event.get("name") == "process_name":
            names[pid] = (event.get("args") or {}).get("name", "")
        elif event.get("name") == "thread_name":
            threads[(pid, tid)] = (event.get("args") or {}).get("name", "")
    return names, threads


def device_kernels(events, names):
    rows = []
    for event in events:
        pid = event.get("pid")
        name = names.get(pid, "")
        if "Ascend Hardware" not in name:
            continue
        if event.get("ph") != "X" or event.get("dur") is None:
            continue
        op = event.get("name") or ""
        if op in ("Computing", "Communication", "Communication(Not Overlapped)",
                  "Free", "Stage", "Bubble", "Preparing"):
            continue
        if is_control_task(op):
            continue
        ts = _num(event.get("ts"))
        dur = _num(event.get("dur"))
        if dur > 0:
            rows.append((ts, ts + dur, op, int(pid)))
    return rows


def cpu_ops(events, names):
    rows = []
    for event in events:
        if event.get("ph") != "X" or event.get("dur") is None:
            continue
        if event.get("cat") != "cpu_op":
            continue
        ts = _num(event.get("ts"))
        dur = _num(event.get("dur"))
        if dur > 0:
            rows.append((ts, ts + dur, event.get("name") or "", event.get("pid")))
    return rows


def forward_annotations(events):
    rows = []
    for event in events:
        if event.get("ph") != "X" or event.get("dur") is None:
            continue
        name = event.get("name") or ""
        match = re.fullmatch(r"r234_forward_iter_(?:dp\d+-rank\d+_)?(\d+)", name)
        if match:
            ts = _num(event.get("ts"))
            rows.append({"forward_index": int(match.group(1)), "ts": ts,
                         "dur": _num(event.get("dur"))})
    return sorted(rows, key=lambda row: row["ts"])


def read_operator_details(path):
    rows = []
    with pathlib.Path(path).open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            name = row.get("Name") or ""
            rows.append({"name": name,
                         "host_self_us": _num(row.get("Host Self Duration(us)")),
                         "host_total_us": _num(row.get("Host Total Duration(us)")),
                         "device_self_us": _num(row.get("Device Self Duration(us)")),
                         "device_total_us": _num(row.get("Device Total Duration(us)"))})
    return rows


def read_kernel_details(path):
    rows = []
    with pathlib.Path(path).open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            name = row.get("Name") or ""
            task_type = row.get("Type") or ""
            rows.append({"name": name,
                         "type": task_type,
                         "accelerator_core": row.get("Accelerator Core") or "",
                         "duration_us": _num(row.get("Duration(us)")),
                         "wait_us": _num(row.get("Wait Time(us)")),
                         "control": is_control_task(name, task_type)})
    return rows


def read_step_time(path):
    rows = []
    with pathlib.Path(path).open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            rows.append({key: row.get(key) for key in row})
    return rows


def rank_of(path):
    text = str(path)
    match = re.search(r"(?:^|/)(dp\d+-rank\d+)(?:/|$)", text)
    if match:
        return match.group(1)
    match = re.search(r"(?:^|/)(dp|tp)(\d+)(?:/|$)", text)
    if match:
        return match.group(1) + match.group(2)
    return "unknown"


def window_of(path):
    text = str(path)
    for name in ("early", "mid", "late"):
        if "/" + name + "/" in text:
            return name
    return "unknown"


def load_metadata(root):
    """Index metadata by (rank namespace, forward index).

    New files are ``forward-metadata-dp<D>-rank<R>.jsonl`` so DP0/DP1 leaders
    with the same local_rank cannot alias.  The legacy ``rank<N>`` name is
    still parsed for old synthetic fixtures, never for a new actual run.
    """
    records = {}
    for path in sorted(pathlib.Path(root).rglob("forward-metadata-*.jsonl")):
        namespace = path.stem.removeprefix("forward-metadata-")
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            records[(namespace, int(row.get("forward_index", -1)))] = row
    return records


def category_rows(operator_rows, kernel_rows, rules):
    cats = {}
    for row in operator_rows:
        cat = C.classify(row["name"], rules)
        slot = cats.setdefault(cat, {"category": cat, "operator_calls": 0,
                                     "operator_device_self_us": 0.0,
                                     "operator_host_self_us": 0.0,
                                     "kernel_duration_us": 0.0, "kernel_calls": 0})
        slot["operator_calls"] += 1
        slot["operator_device_self_us"] += row["device_self_us"]
        slot["operator_host_self_us"] += row["host_self_us"]
    for row in kernel_rows:
        if row.get("control"):
            continue
        cat = C.classify(row["name"], rules)
        slot = cats.setdefault(cat, {"category": cat, "operator_calls": 0,
                                     "operator_device_self_us": 0.0,
                                     "operator_host_self_us": 0.0,
                                     "kernel_duration_us": 0.0, "kernel_calls": 0})
        slot["kernel_calls"] += 1
        slot["kernel_duration_us"] += row["duration_us"]
    return sorted(cats.values(), key=lambda row: (-row["operator_device_self_us"],
                                                  -row["kernel_duration_us"]))


def analyze_capture(directory, rules, metadata):
    directory = pathlib.Path(directory)
    out = {"capture_dir": str(directory), "rank": rank_of(directory),
           "window": window_of(directory), "status": "PENDING", "errors": []}
    info_paths = sorted(directory.glob("profiler_info*.json"))
    info_path = info_paths[0] if len(info_paths) == 1 else None
    trace_path = directory / "ASCEND_PROFILER_OUTPUT/trace_view.json"
    op_path = directory / "ASCEND_PROFILER_OUTPUT/operator_details.csv"
    kernel_path = directory / "ASCEND_PROFILER_OUTPUT/kernel_details.csv"
    step_path = directory / "ASCEND_PROFILER_OUTPUT/step_trace_time.csv"
    if info_path is not None:
        info = json.loads(info_path.read_text())
        config = info.get("config", {})
        common_cfg = config.get("common_config", {})
        exp_cfg = config.get("experimental_config", {})
        activities = common_cfg.get("activities") or []
        out["profiler_info"] = {
            "activities": activities,
            "record_shapes": common_cfg.get("record_shapes"),
            "profile_memory": common_cfg.get("profile_memory"),
            "with_stack": common_cfg.get("with_stack"),
            "profiler_level": exp_cfg.get("_profiler_level"),
            "data_simplification": exp_cfg.get("_data_simplification"),
            "export_type": exp_cfg.get("_export_type"),
        }
        out["profiler_config_ok"] = bool(
            any("CPU" in str(a) for a in activities)
            and any("NPU" in str(a) for a in activities)
            and common_cfg.get("record_shapes") is False
            and common_cfg.get("profile_memory") is False
            and common_cfg.get("with_stack") is False
            and str(exp_cfg.get("_profiler_level")) == "Level1")
        if not out["profiler_config_ok"]:
            out["errors"].append("profiler_config_not_level1_cpu_npu_no_shapes_no_memory_no_stack")
    else:
        out["errors"].append("profiler_info_missing_or_ambiguous")
        out["profiler_info_files"] = [str(path) for path in info_paths]
    operator_rows = read_operator_details(op_path) if op_path.is_file() else []
    all_kernel_rows = read_kernel_details(kernel_path) if kernel_path.is_file() else []
    control_rows = [row for row in all_kernel_rows if row.get("control")]
    kernel_rows = [row for row in all_kernel_rows if not row.get("control")]
    if not op_path.is_file():
        out["errors"].append("operator_details_missing")
    if not kernel_path.is_file():
        out["errors"].append("kernel_details_missing")
    out["operator_rows"] = len(operator_rows)
    out["kernel_rows"] = len(kernel_rows)
    out["control_task_rows"] = len(control_rows)
    out["control_task_duration_us"] = sum(row["duration_us"] for row in control_rows)
    out["control_task_policy"] = "WAIT/NOTIFY/SQE are control, excluded from compute and categories"
    out["operator_all"] = operator_rows
    out["kernel_all"] = kernel_rows
    out["operator_top"] = sorted(
        operator_rows, key=lambda row: -row["device_self_us"])[:20]
    out["kernel_top"] = sorted(
        kernel_rows, key=lambda row: -row["duration_us"])[:20]
    out["category_summary"] = category_rows(operator_rows, kernel_rows, rules)
    out["step_trace_time"] = read_step_time(step_path) if step_path.is_file() else []
    if not trace_path.is_file():
        out["errors"].append("trace_view_missing; union/overlap NOT_COMPUTED")
        out["status"] = "PARTIAL"
        return out
    events = load_trace(trace_path)
    names, _threads = process_maps(events)
    kernels = device_kernels(events, names)
    cpu = cpu_ops(events, names)
    annotations = forward_annotations(events)
    out["annotations"] = len(annotations)
    out["forwards"] = []
    if len(annotations) != 2:
        out["errors"].append("expected_two_forward_annotations")
    for annotation in annotations:
        lo, hi = annotation["ts"], annotation["ts"] + annotation["dur"]
        c = clipped([(a, b) for a, b, _n, _p in cpu], lo, hi)
        meta = metadata.get((out["rank"], annotation["forward_index"]))
        out["forwards"].append({
            "forward_index": annotation["forward_index"],
            "wall_us": annotation["dur"],
            "instruction_span_us": annotation["dur"],
            "cpu_op_union_us": union_us(c),
            "cpu_unattributed_instruction_us": max(0.0, annotation["dur"] - union_us(c)),
            "cpu_launch_gap_us": None,
            "npu_kernel_union_us": None,
            "npu_kernel_sum_us": None,
            "hccl_union_us": None,
            "category_union_us": None,
            "device_attribution": "unavailable_without_native_flow_correlation",
            "metadata": meta,
        })
    marker_name = "r234_capture_window_" + out["rank"] + "_" + out["window"]
    spans = [e for e in events if e.get("ph") == "X" and e.get("name") == marker_name]
    if len(spans) != 1:
        out["errors"].append("device_drained_capture_boundary_missing_or_duplicate")
    else:
        lo = _num(spans[0].get("ts")); hi = lo + _num(spans[0].get("dur"))
        if hi <= lo or any(a["ts"] < lo or a["ts"] + a["dur"] > hi for a in annotations):
            out["errors"].append("capture_boundary_does_not_enclose_forward_annotations")
        k = clipped([(a, b) for a, b, _n, _p in kernels], lo, hi)
        if not k:
            out["errors"].append("capture_has_no_device_kernels")
        crossing = [(a, b) for a, b, _n, _p in kernels
                    if a < hi and b > lo and (a < lo - 1.0 or b > hi + 1.0)]
        if crossing:
            out["errors"].append("kernel_crosses_device_drained_capture_boundary")
        hccl = clipped([(a, b) for a, b, n, _p in kernels
                        if C.classify(n, rules) == "communication"], lo, hi)
        other = clipped([(a, b) for a, b, n, _p in kernels
                         if C.classify(n, rules) != "communication"], lo, hi)
        categories = {cat: union_us(clipped(
            [(a, b) for a, b, n, _p in kernels if C.classify(n, rules) == cat], lo, hi))
            for cat in {row["category"] for row in out["category_summary"]}}
        elapsed = hi - lo
        out["capture_totals"] = {
            "forward_count": len(annotations),
            "window_elapsed_us": elapsed,
            "boundary": "device_synchronized_capture_annotation",
            "CPU_enqueue_instruction_wall_us": sum(a["dur"] for a in annotations),
            "cpu_unattributed_instruction_us": sum(f["cpu_unattributed_instruction_us"] for f in out["forwards"]),
            "cpu_launch_gap_us": None,
            "npu_kernel_union_us": union_us(k),
            "npu_kernel_sum_us": sum(b - a for a, b in k),
            "npu_idle_within_capture_us": max(0.0, elapsed - union_us(k)),
            "npu_union_share": union_us(k) / elapsed if elapsed > 0 else None,
            "hccl_union_us": union_us(hccl),
            "hccl_overlap_with_non_hccl_us": intersection_us(hccl, other),
            "category_union_us": categories,
            "category_share_upper_bound": {cat: value / elapsed if elapsed > 0 else None
                                             for cat, value in categories.items()},
            "per_forward_device_attribution": None,
            "per_forward_values_are_amortized_only": True,
        }
    out["status"] = "PARTIAL" if out["errors"] else "OK"
    return out


def discover(artifacts):
    root = pathlib.Path(artifacts)
    dirs = set()
    for name in ("ASCEND_PROFILER_OUTPUT/trace_view.json",
                 "ASCEND_PROFILER_OUTPUT/operator_details.csv"):
        for path in root.rglob(name):
            dirs.add(path.parents[1])
    return sorted(dirs)


def summarize(captures):
    windows, all_ops, all_kernels = {}, {}, {}
    for capture in captures:
        for row in capture.get("operator_all", []):
            slot = all_ops.setdefault(row["name"], {"name": row["name"], "device_self_us": 0.0,
                                                    "host_self_us": 0.0, "device_self_us_max_capture": 0.0})
            slot["device_self_us"] += row["device_self_us"]
            slot["host_self_us"] += row["host_self_us"]
            slot["device_self_us_max_capture"] = max(slot["device_self_us_max_capture"], row["device_self_us"])
        for row in capture.get("kernel_all", []):
            slot = all_kernels.setdefault(row["name"], {"name": row["name"], "duration_us": 0.0, "calls": 0})
            slot["duration_us"] += row["duration_us"]; slot["calls"] += 1
        key = (capture.get("window", "unknown"), capture.get("rank", "unknown"))
        bucket = windows.setdefault(key, {"window": key[0], "rank": key[1], "captures": 0,
            "forwards": 0, "window_elapsed_us": 0.0, "CPU_enqueue_instruction_wall_us": 0.0,
            "npu_union_us": 0.0, "npu_sum_us": 0.0, "cpu_unattributed_instruction_us": 0.0,
            "cpu_gap_us": None, "hccl_union_us": 0.0, "category_union_us": {}})
        total = capture.get("capture_totals")
        if not total:
            continue
        bucket["captures"] += 1; bucket["forwards"] += total["forward_count"]
        for dst, src in [("window_elapsed_us", "window_elapsed_us"),
            ("CPU_enqueue_instruction_wall_us", "CPU_enqueue_instruction_wall_us"),
            ("npu_union_us", "npu_kernel_union_us"), ("npu_sum_us", "npu_kernel_sum_us"),
            ("hccl_union_us", "hccl_union_us"),
            ("cpu_unattributed_instruction_us", "cpu_unattributed_instruction_us")]:
            bucket[dst] += total[src]
        for cat, value in total["category_union_us"].items():
            bucket["category_union_us"][cat] = bucket["category_union_us"].get(cat, 0.0) + value
    for bucket in windows.values():
        elapsed = bucket["window_elapsed_us"]
        bucket["npu_union_share"] = bucket["npu_union_us"] / elapsed if elapsed else None
        bucket["hccl_union_share_upper_bound"] = bucket["hccl_union_us"] / elapsed if elapsed else None
        bucket["category_share_upper_bound"] = {k: v / elapsed if elapsed else None
                                                 for k, v in bucket["category_union_us"].items()}
    return {"windows": [windows[k] for k in sorted(windows)],
        "top_operators_by_device_self_us": sorted(all_ops.values(), key=lambda r: -r["device_self_us"])[:30],
        "top_kernels_by_summed_duration_us": sorted(all_kernels.values(), key=lambda r: -r["duration_us"])[:30],
        "overlap_policy": "Occupancy uses complete device-drained capture elapsed time. Category unions overlap and are not additive or proven critical paths.",
        "unknown_policy": "Unknown categories retained; no per-forward device ownership inferred from CPU launch timestamps."}


def analyze(artifacts, windows):
    root = pathlib.Path(artifacts)
    rules = json.loads((pathlib.Path(__file__).resolve().parent / "categories.json").read_text())["rules"]
    metadata = load_metadata(root)
    captures = [analyze_capture(directory, rules, metadata) for directory in discover(root)]
    if not captures:
        return {"status": "PENDING", "reason": "no_native_profile_output",
                "artifacts": str(root), "captures": [], "summary": None,
                "windows_requested": windows}
    bounds = {w["name"]: w for w in windows}
    bad = []
    for capture in captures:
        reason = None
        if capture.get("status") != "OK":
            reason = "capture_not_ok"
        elif len(capture.get("forwards", [])) != 2:
            reason = "not_two_forwards"
        else:
            window = bounds.get(capture.get("window"))
            for row in capture["forwards"]:
                meta = row.get("metadata")
                if not meta or not meta.get("capture"):
                    reason = "metadata_missing"
                    break
                if window is None:
                    reason = "window_not_requested"
                    break
                reqs = [r for r in meta.get("requests", []) if r.get("query_tokens", 0) > 0]
                if len(reqs) != 1:
                    reason = "active_request_count_not_native_single_row"
                    break
                if meta.get("rank_namespace") not in (None, capture.get("rank")):
                    reason = "metadata_rank_namespace_mismatch"
                    break
                if "lo" in window and "hi" in window and any(
                        not (window["lo"] <= r.get("context_end", -1) <= window["hi"])
                        for r in reqs):
                    reason = "context_outside_requested_window"
                    break
        if reason:
            capture["window_proof_error"] = reason
            bad.append(capture)
    return {
        "status": "PENDING_WINDOW_PROOF" if bad else "COMPUTED",
        "artifacts": str(root),
        "captures": captures,
        "summary": summarize(captures),
        "windows_requested": windows,
        "incomplete_captures": [capture["capture_dir"] for capture in bad],
        "claims": {
            "throughput": None,
            "formal_baseline": None,
            "critical_path": "not claimed; overlapping kernels are not additive",
            "forward_boundary": "A annotation is NPUWorker.execute_model; B annotation is Scheduler.run_batch (includes scheduler/PD pre-step). Device occupancy uses an outer capture marker drained before start and before end. Individual CPU forward annotations report enqueue instruction spans; asynchronous per-forward device ownership is unavailable without native flow correlation. These boundaries are disclosed, not treated as identical.",
            "eliminable_share": "per-category interval union / complete observed capture elapsed; diagnostic occupancy upper bound only, not whole-request/C32 or a proven critical path",
            "whole_machine_share_upper_bound": None,
            "whole_machine_share_reason": "D forward, queueing, PD handoff and the remaining prefill chunks are not in this profile; a P-forward share must not be multiplied into a whole-request claim",
        },
    }
