#!/usr/bin/env python3
"""Independent read-only diagnostic of R247 native whole-model captures.

Adapted from the main-control ``checks/analyze-recovered-native-profile.py``
parser without rewriting its raw-file handling.  Native WAIT/NOTIFY/SQE tasks
are control, not compute.  CSV sums are reported separately from interval
unions and never treated as critical shares.  A missing/timed-out CANN export
is ``PARTIAL``, never a fake PASS.
"""
import collections
import csv
import hashlib
import json
import pathlib
import sys

A = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(A))
import trace_stats as T

root = pathlib.Path(sys.argv[1])
metadata_root = pathlib.Path(sys.argv[2])
output = pathlib.Path(sys.argv[3])
rules = json.loads((A / "categories.json").read_text())["rules"]
metadata = T.load_metadata(metadata_root)
captures = []
for directory in T.discover(root):
    trace = directory / "ASCEND_PROFILER_OUTPUT/trace_view.json"
    if not trace.is_file():
        continue
    if not (directory / "ASCEND_PROFILER_OUTPUT/kernel_details.csv").is_file():
        captures.append({"directory": str(directory), "window": T.window_of(directory),
                         "status": "NATIVE_KERNEL_CSV_MISSING",
                         "original_partial_export_retained": True})
        continue
    trace_valid = True
    trace_error = None
    try:
        events = T.load_trace(trace)
    except json.JSONDecodeError as e:
        trace_valid = False
        trace_error = str(e)
        source = trace.read_text()
        decoder = json.JSONDecoder()
        position = source.index("[") + 1
        events = []
        while position < len(source):
            while position < len(source) and source[position] in " \t\r\n,":
                position += 1
            if position >= len(source) or source[position] == "]":
                break
            try:
                event, position = decoder.raw_decode(source, position)
            except ValueError:
                break
            if isinstance(event, dict):
                events.append(event)
    names, threads = T.process_maps(events)
    native_csv = T.read_kernel_details(directory / "ASCEND_PROFILER_OUTPUT/kernel_details.csv")
    with (directory / "ASCEND_PROFILER_OUTPUT/kernel_details.csv").open() as stream:
        timed_csv = list(csv.DictReader(stream))
    infofiles = list(directory.glob("profiler_info*.json"))
    info = json.loads(infofiles[0].read_text()) if len(infofiles) == 1 else None
    config = (info or {}).get("config", {})
    common = config.get("common_config", {})
    exp = config.get("experimental_config", {})
    config_ok = (any("CPU" in str(x) for x in common.get("activities", []))
                 and any("NPU" in str(x) for x in common.get("activities", []))
                 and common.get("record_shapes") is False
                 and common.get("with_stack") is False
                 and common.get("profile_memory") is False
                 and exp.get("_profiler_level") == "Level1")
    window = T.window_of(directory)
    rank = T.rank_of(directory)
    markers = [e for e in events if e.get("ph") == "X" and
               e.get("name") == "r234_capture_window_" + rank + "_" + window]
    if len(markers) != 1:
        captures.append({"directory": str(directory), "window": window,
                         "status": "CAPTURE_MARKER_ABSENT_OR_DUPLICATE"})
        continue
    lo = float(markers[0]["ts"])
    hi = lo + float(markers[0]["dur"])
    # Device intervals come from the native hardware CSV on both sides.  CPU
    # enqueue spans are never used to own a device forward.  WAIT/NOTIFY/SQE
    # are runtime control tasks and are kept out of every compute union.
    hardware = []
    core = []
    control = []
    task_types = collections.Counter()
    for e in timed_csv:
        n = e["Name"]
        typ = str(e.get("Type", ""))
        accel = str(e.get("Accelerator Core", ""))
        begin = float(e["Start Time(us)"])
        end = begin + float(e["Duration(us)"])
        if end <= begin or begin >= hi or end <= lo:
            continue
        task_types[typ] += 1
        if T.is_control_task(n, typ):
            control.append((begin, end, n, typ))
            continue
        hardware.append((begin, end, n, accel))
        if "CORE" in accel or accel in ("MIX_AIC", "MIX_AIV"):
            core.append((begin, end, n, accel))
    intervals = [(a, b) for a, b, _, _ in hardware]
    clipped = T.clipped(intervals, lo, hi)
    core_intervals = T.clipped([(a, b) for a, b, _, _ in core], lo, hi)
    categories = {}
    for category in {T.C.classify(n, rules) for _, _, n, _ in core}:
        cat_int = T.clipped([(a, b) for a, b, n, _ in core
                             if T.C.classify(n, rules) == category], lo, hi)
        categories[category] = T.union_us(cat_int)
    byname = collections.defaultdict(lambda: {"duration_us": 0.0, "calls": 0})
    for row in native_csv:
        if row.get("control"):
            continue
        byname[row["name"]]["duration_us"] += row["duration_us"]
        byname[row["name"]]["calls"] += 1
    kernel_top = sorted([dict(name=n, **v) for n, v in byname.items()],
                        key=lambda x: -x["duration_us"])[:25]
    annotations = T.forward_annotations(events)
    forwards = [dict(x, metadata=metadata.get((rank, x["forward_index"])))
                for x in annotations]
    crossing = [dict(name=n, task_type=typ, before_us=max(0, lo - a),
                     after_us=max(0, b - hi))
                for a, b, n, typ in hardware if a < lo or b > hi]
    ops_path = directory / "ASCEND_PROFILER_OUTPUT/operator_details.csv"
    ops = T.read_operator_details(ops_path) if ops_path.is_file() else []
    host = collections.defaultdict(float)
    for row in ops:
        host[row["name"]] += row["host_self_us"]
    merged = T.union_intervals(clipped)
    gaps = []
    previous = lo
    for a, b in merged:
        if a > previous:
            gaps.append((previous, a))
        previous = b
    if previous < hi:
        gaps.append((previous, hi))
    status = ("TWO_FORWARD_NATIVE_CSV_WITH_UNMERGED_CPU_PREFIX" if not trace_valid
              else "COMPLETE_TWO_FORWARD_RAW_DIAGNOSTIC")
    if not (len(forwards) == 2 and config_ok and all(f["metadata"] for f in forwards)):
        status = "PARTIAL"
    row = {"directory": str(directory), "rank": rank, "window": window,
           "status": status,
           "trace_JSON_valid": trace_valid,
           "original_trace_JSON_error": trace_error,
           "CPU_records_read_without_repair": len(events),
           "device_time_source": ("Original native kernel_details.csv Start Time(us) and "
                                  "Duration(us); outer capture marker drained before start "
                                  "and after torch.npu.synchronize before stop"),
           "cpu_enqueue_is_device_forward": False,
           "per_forward_device_attribution": None,
           "native_profiler_config_valid": config_ok,
           "profiler_info_file": str(infofiles[0]) if len(infofiles) == 1 else None,
           "profiler_info_files": [str(path) for path in infofiles],
           "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
           "capture_elapsed_us": hi - lo,
           "forwards": forwards,
           "native_CSV_kernels": len(native_csv) - sum(1 for r in native_csv if r.get("control")),
           "native_CSV_control_tasks": sum(1 for r in native_csv if r.get("control")),
           "native_task_types": dict(task_types),
           "control_task_count": len(control),
           "control_task_duration_sum_us": sum(b - a for a, b, _, _ in control),
           "control_task_policy": "WAIT/NOTIFY/SQE excluded from compute, categories and occupancy",
           "native_CSV_span_us": ([min(float(e["Start Time(us)"]) for e in timed_csv),
                                  max(float(e["Start Time(us)"]) + float(e["Duration(us)"])
                                      for e in timed_csv)] if timed_csv else [None, None]),
           "native_kernel_busy_union_us": T.union_us(clipped),
           "core_compute_union_us": T.union_us(core_intervals),
           "core_category_union_us": categories,
           "core_category_occupancy_upper_bound": {k: v / (hi - lo) for k, v in categories.items()},
           "native_kernels_crossing_CPU_capture_boundary": crossing,
           "largest_observed_native_kernel_gaps_us": [
               {"start_from_capture_us": a - lo, "duration_us": b - a}
               for a, b in sorted(gaps, key=lambda x: x[0] - x[1])[:8]],
           "kernel_duration_sums_not_critical_path": kernel_top,
           "host_self_duration_top": sorted(
               [{"name": k, "host_self_us": v} for k, v in host.items()],
               key=lambda x: -x["host_self_us"])[:15],
           "causal_attribution": ("No Python call stack or compiler trace; host enqueue gaps "
                                  "and copy waits require a targeted follow-up and are not "
                                  "assigned to a device forward.")}
    captures.append(row)
accepted = {"COMPLETE_TWO_FORWARD_RAW_DIAGNOSTIC",
            "TWO_FORWARD_NATIVE_CSV_WITH_UNMERGED_CPU_PREFIX"}
complete = len(captures) == 3 and all(row.get("status") in accepted for row in captures)
out = {"project_id": "glm53-flash", "unit": "r247",
       "kind": "R247_B_WHOLE_MODEL_P_FORWARD_DIAGNOSTIC",
       "source_analyzer_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
       "status": "COMPUTED" if complete else "PARTIAL",
       "expected_captures": 3,
       "captures_complete": complete,
       "captures": captures,
       "formal_requests": 0,
       "throughput": None,
       "whole_machine_C32_share": None,
       "original_R238_acceptance_not_changed": True,
       "control_task_policy": ("WAIT/NOTIFY/SQE are runtime control, not compute; "
                               "they are reported separately."),
       "cpu_policy": ("CPU enqueue/instruction spans are host-side only; no per-forward "
                      "device ownership is inferred from CPU timestamps."),
       "device_time_source": ("Native kernel_details.csv intervals inside a capture marker "
                              "that is drained before start and synchronizes the device "
                              "before stop."),
       "export_policy": ("A missing kernel CSV, missing marker or absent native info file "
                         "makes the capture PARTIAL; a collection timeout can never PASS."),
       "limits": ["One sampled leader rank only (TP0).",
                  "Native CSV duration sums are not a critical path.",
                  "Boundary clock crossings are retained, not silently discarded."]}
output.parent.mkdir(exist_ok=True, parents=True)
output.write_text(json.dumps(out, indent=2) + "\n")
for row in captures:
    print(json.dumps({k: row.get(k) for k in [
        "window", "status", "capture_elapsed_us", "core_compute_union_us",
        "core_category_union_us", "control_task_duration_sum_us",
        "native_kernels_crossing_CPU_capture_boundary",
        "kernel_duration_sums_not_critical_path"]})[:4200])
