#!/usr/bin/env python3
"""Summarize an Extreme Runtime-only torch_npu profile across TP ranks.

The report keeps wall-clock scope durations separate from device interval
unions.  Overlapping streams are unioned, so communication/compute totals are
not incorrectly added together as if they were serial.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


SCOPE_PREFIX = "extreme::"
COMM_WORDS = ("hcom", "allreduce", "all_reduce", "allgather", "all_gather",
              "reducescatter", "reduce_scatter", "alltoall", "all_to_all",
              "broadcast", "send", "recv")
SYNC_WORDS = ("synchronize", "wait", "event")


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[round((len(values) - 1) * fraction)]


def stats_us(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "median_ms": statistics.median(values) / 1000 if values else None,
        "p90_ms": percentile(values, 0.9) / 1000 if values else None,
        "max_ms": max(values) / 1000 if values else None,
    }


def union_us(intervals: list[tuple[float, float]]) -> float:
    rows = sorted((a, b) for a, b in intervals if b > a)
    if not rows:
        return 0.0
    left, right = rows[0]
    total = 0.0
    for start, end in rows[1:]:
        if start > right:
            total += right - left
            left, right = start, end
        else:
            right = max(right, end)
    return total + right - left


def is_comm(name: str, kind: str) -> bool:
    text = f"{name} {kind}".lower()
    return kind.lower().startswith("hcom_") or any(x in text for x in COMM_WORDS)


def analyze_rank(rank_dir: Path) -> dict:
    outputs = list(rank_dir.glob("*_ascend_pt/ASCEND_PROFILER_OUTPUT"))
    if len(outputs) != 1:
        raise RuntimeError(f"expected one parsed profile under {rank_dir}, got {len(outputs)}")
    output = outputs[0]
    trace = json.loads((output / "trace_view.json").read_text())

    scopes: dict[str, list[tuple[float, float]]] = defaultdict(list)
    cpu_events: list[tuple[float, float, str]] = []
    for event in trace:
        if event.get("ph") != "X" or event.get("cat") != "cpu_op":
            continue
        start = float(event["ts"])
        end = start + float(event.get("dur", 0))
        name = event.get("name", "")
        cpu_events.append((start, end, name))
        if name.startswith(SCOPE_PREFIX):
            scopes[name].append((start, end))
    cycles = sorted(scopes.get("extreme::cycle", []))
    if len(cycles) != 8:
        raise RuntimeError(f"{rank_dir.name}: expected 8 cycles, got {len(cycles)}")

    kernels = []
    with (output / "kernel_details.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                start = float(row["Start Time(us)"].strip())
                duration = float(row["Duration(us)"])
            except (KeyError, TypeError, ValueError):
                continue
            kernels.append((start, start + duration, row.get("Name", ""), row.get("Type", "")))

    per_cycle = []
    for cycle_index, (begin, end) in enumerate(cycles):
        clipped = []
        for start, stop, name, kind in kernels:
            if stop <= begin or start >= end:
                continue
            clipped.append((max(start, begin), min(stop, end), name, kind))
        comm = [(a, b) for a, b, name, kind in clipped if is_comm(name, kind)]
        compute = [(a, b) for a, b, name, kind in clipped if not is_comm(name, kind)]
        all_intervals = [(a, b) for a, b, _, _ in clipped]
        sync_events = [(a, b, n) for a, b, n in cpu_events
                       if a >= begin and b <= end and not n.startswith(SCOPE_PREFIX)
                       and any(word in n.lower() for word in SYNC_WORDS)]
        child_cpu = [(max(a, begin), min(b, end)) for a, b, n in cpu_events
                     if not n.startswith(SCOPE_PREFIX) and b > begin and a < end]
        per_cycle.append({
            "index": cycle_index,
            "host_wall_us": end - begin,
            "device_union_us": union_us(all_intervals),
            "compute_union_us": union_us(compute),
            "communication_union_us": union_us(comm),
            "host_cpu_op_union_us": union_us(child_cpu),
            "host_uncovered_gap_us": (end - begin) - union_us(child_cpu),
            "sync_event_sum_us": sum(b - a for a, b, _ in sync_events),
            "kernel_count": len(clipped),
            "communication_kernel_count": len(comm),
        })

    scope_durations = {name: [end - start for start, end in rows]
                       for name, rows in scopes.items()}
    scope_device = {}
    for name, rows in scopes.items():
        measurements = []
        for begin, end in rows:
            selected = [(max(start, begin), min(stop, end), kernel_name, kind)
                        for start, stop, kernel_name, kind in kernels
                        if stop > begin and start < end]
            measurements.append({
                "all_union_us": union_us([(a, b) for a, b, _, _ in selected]),
                "compute_union_us": union_us([(a, b) for a, b, n, k in selected
                                               if not is_comm(n, k)]),
                "communication_union_us": union_us([(a, b) for a, b, n, k in selected
                                                     if is_comm(n, k)]),
            })
        scope_device[name] = measurements
    top_cpu = Counter()
    for start, end, name in cpu_events:
        if name.startswith(SCOPE_PREFIX):
            continue
        if any(start >= a and end <= b for a, b in cycles):
            top_cpu[name] += end - start
    top_kernel = Counter()
    top_comm = Counter()
    for start, end, name, kind in kernels:
        if any(end > a and start < b for a, b in cycles):
            top_kernel[name] += end - start
            if is_comm(name, kind):
                top_comm[name] += end - start
    return {
        "rank": rank_dir.name,
        "scope_durations_us": scope_durations,
        "scope_device": scope_device,
        "cycles": per_cycle,
        "top_cpu_sum_ms": [(name, value / 1000) for name, value in top_cpu.most_common(20)],
        "top_kernel_sum_ms_overlaps": [(name, value / 1000) for name, value in top_kernel.most_common(20)],
        "top_communication_sum_ms_overlaps": [(name, value / 1000) for name, value in top_comm.most_common(20)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    ranks = [analyze_rank(path) for path in sorted(args.root.glob("rank*"))]
    scope_values: dict[str, list[float]] = defaultdict(list)
    steady_scope_values: dict[str, list[float]] = defaultdict(list)
    for rank in ranks:
        for name, values in rank["scope_durations_us"].items():
            scope_values[name].extend(values)
            steady_scope_values[name].extend(values[1:])
    scope_device_fields = ("all_union_us", "compute_union_us", "communication_union_us")
    scope_device_steady = {}
    for name in sorted({name for rank in ranks for name in rank["scope_device"]}):
        scope_device_steady[name] = {
            field: stats_us([
                item[field]
                for rank in ranks
                for item in rank["scope_device"].get(name, [])[1:]
            ])
            for field in scope_device_fields
        }
    cycle_fields = ("host_wall_us", "device_union_us", "compute_union_us",
                    "communication_union_us", "host_uncovered_gap_us",
                    "sync_event_sum_us")
    report = {
        "method": {
            "scope": "record_function CPU wall intervals",
            "device": "kernel intervals clipped to cycle and unioned across streams",
            "steady_state": "cycles 1..7; cycle 0 excluded",
            "warning": "compute and communication unions may overlap and must not be summed",
        },
        "rank_count": len(ranks),
        "scope_all_cycles": {name: stats_us(values) for name, values in sorted(scope_values.items())},
        "scope_steady_cycles": {name: stats_us(values) for name, values in sorted(steady_scope_values.items())},
        "scope_device_steady_cycles": scope_device_steady,
        "cycle_steady": {
            field: stats_us([cycle[field] for rank in ranks for cycle in rank["cycles"][1:]])
            for field in cycle_fields
        },
        "ranks": ranks,
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(json.dumps({
        "rank_count": report["rank_count"],
        "scope_steady_cycles": report["scope_steady_cycles"],
        "scope_device_steady_cycles": report["scope_device_steady_cycles"],
        "cycle_steady": report["cycle_steady"],
    }, indent=2))


if __name__ == "__main__":
    main()
