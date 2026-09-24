#!/usr/bin/env python3
"""Summarize synchronized target graph windows for Loop038 Run107."""
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path("/data/wio/Inference_Foundry/evidence/20260924_loop038_cycle/run107")
COMM = ("hcom", "allreduce", "all_reduce", "allgather", "all_gather",
        "reducescatter", "reduce_scatter", "alltoall", "all_to_all")
def union(intervals):
    intervals = sorted((a, b) for a, b in intervals if b > a)
    if not intervals:
        return 0.0
    start, end = intervals[0]
    total = 0.0
    for a, b in intervals[1:]:
        if a > end:
            total += end - start
            start, end = a, b
        else:
            end = max(end, b)
    return total + end - start

ranks = []
for rank in range(8):
    paths = list((ROOT / "profile").glob(
        f"rank{rank}_*ascend_pt/ASCEND_PROFILER_OUTPUT"))
    assert len(paths) == 1, (rank, paths)
    p = paths[0]
    trace = json.loads((p / "trace_view.json").read_text())
    scopes = sorted((float(e["ts"]), float(e["ts"]) + float(e["dur"]))
                    for e in trace if e.get("ph") == "X"
                    and e.get("name") == "extreme::target")
    assert len(scopes) == 2, (rank, scopes)
    with (p / "kernel_details.csv").open(newline="") as handle:
        kernels = [(float(r["Start Time(us)"].strip()),
                    float(r["Start Time(us)"].strip()) +
                    float(r["Duration(us)"]), r["Name"], r["Type"])
                   for r in csv.DictReader(handle)]
    cycles = []
    for cycle, (start, end) in enumerate(scopes, 64):
        selected = [(max(a, start), min(b, end), name, kind)
                    for a, b, name, kind in kernels
                    if b > start and a < end]
        compute = [(a, b) for a, b, name, kind in selected
                   if not any(w in (name + kind).lower() for w in COMM)]
        comm = [(a, b) for a, b, name, kind in selected
                if any(w in (name + kind).lower() for w in COMM)]
        all_intervals = [(a, b) for a, b, _, _ in selected]
        counts = Counter()
        sums = Counter()
        for a, b, name, _ in selected:
            counts[name] += 1
            sums[name] += b - a
        total_u = union(all_intervals)
        compute_u = union(compute)
        comm_u = union(comm)
        cycles.append({
            "cycle": cycle,
            "host_scope_ms": (end - start) / 1000,
            "device_union_ms": total_u / 1000,
            "compute_union_ms": compute_u / 1000,
            "communication_union_ms": comm_u / 1000,
            "compute_communication_overlap_ms":
                (compute_u + comm_u - total_u) / 1000,
            "host_scope_minus_device_union_ms":
                (end - start - total_u) / 1000,
            "kernel_count": len(selected),
            "grouped_matmul_sum_ms":
                sum(duration for name, duration in sums.items()
                    if "GroupedMatmul" in name) / 1000,
            "grouped_matmul_count":
                sum(count for name, count in counts.items()
                    if "GroupedMatmul" in name),
            "top_kernel_sums_ms": [[name, duration / 1000]
                                   for name, duration in sums.most_common(15)],
        })
    ranks.append({"rank": rank, "cycles": cycles})
rows = [c for r in ranks for c in r["cycles"] if c["kernel_count"] == 2836]
assert len(rows) == 15
excluded = [{"rank": r["rank"], "cycle": c["cycle"], "kernel_count": c["kernel_count"]}
            for r in ranks for c in r["cycles"] if c["kernel_count"] != 2836]
fields = ("host_scope_ms", "device_union_ms", "compute_union_ms",
          "communication_union_ms", "compute_communication_overlap_ms",
          "host_scope_minus_device_union_ms", "grouped_matmul_sum_ms",
          "grouped_matmul_count", "kernel_count")
summary = {"run": "run107", "rank_count": 8, "cycles_per_rank": 2,
           "valid_target_windows": len(rows),
           "excluded_windows": excluded,
           "caveat": "Only cycles 64-65 are synchronized and instrumented. "
                     "Synchronization changes stage timing and may serialize "
                     "work; values are diagnostic, not formal E2E. Kernel "
                     "sums across streams overlap and are not additive."}
for field in fields:
    values = [r[field] for r in rows]
    summary[field] = {"median": statistics.median(values),
                      "min": min(values), "max": max(values)}
summary["ranks"] = ranks
(ROOT / "target_window.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({k: v for k, v in summary.items() if k != "ranks"}, indent=2))
