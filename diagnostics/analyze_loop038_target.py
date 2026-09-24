#!/usr/bin/env python3
"""Analyze bounded Loop038 profiles without equating host scopes to device work."""
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path("/data/wio/Inference_Foundry/evidence/20260924_loop038_cycle/run106")
def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))
def median(xs):
    return statistics.median(xs)

ranks = []
for rank in range(8):
    paths = list((ROOT / "profile").glob(
        f"rank{rank}_*ascend_pt/ASCEND_PROFILER_OUTPUT"))
    assert len(paths) == 1, (rank, paths)
    p = paths[0]
    ops = read_csv(p / "operator_details.csv")
    kernels = read_csv(p / "kernel_details.csv")
    steps = read_csv(p / "step_trace_time.csv")
    target_indices = [i for i, row in enumerate(ops)
                      if row["Name"] == "extreme::target"]
    assert len(target_indices) == 2, (rank, target_indices)
    targets = []
    for cycle, start in enumerate(target_indices, 64):
        end = next(i for i in range(start + 1, len(ops))
                   if ops[i]["Name"] == "extreme::acceptance")
        rows = ops[start:end]
        waits = [float(row["Device Total Duration(us)"]) / 1000
                 for row in rows if row["Name"] == "wait_event"]
        gather = [float(row["Device Total Duration(us)"]) / 1000
                  for row in rows if row["Name"] == "HcclAllGather"]
        targets.append({
            "cycle": cycle,
            "host_scope_ms": float(rows[0]["Host Total Duration(us)"]) / 1000,
            "attributed_device_total_ms":
                float(rows[0]["Device Total Duration(us)"]) / 1000,
            "largest_nested_wait_event_ms": max(waits),
            "nested_hccl_allgather_sum_ms": sum(gather),
        })
    grouped = [row for row in kernels if "GroupedMatmul" in row["Name"]]
    top = Counter()
    for row in kernels:
        top[row["Name"]] += float(row["Duration(us)"])
    step = steps[0]
    ranks.append({
        "rank": rank, "targets": targets,
        "whole_trace_grouped_matmul_count": len(grouped),
        "whole_trace_grouped_matmul_sum_ms":
            sum(float(row["Duration(us)"]) for row in grouped) / 1000,
        "whole_trace_compute_ms": float(step["Computing"]) / 1000,
        "whole_trace_communication_not_overlapped_ms":
            float(step["Communication(Not Overlapped)"]) / 1000,
        "whole_trace_top_kernel_sum_ms":
            [[name, duration / 1000] for name, duration in top.most_common(12)],
    })
assert all(r["whole_trace_grouped_matmul_count"] == 184 for r in ranks)
all_targets = [target for rank in ranks for target in rank["targets"]]
summary = {
    "run": "run106",
    "profile_rank_count": 8,
    "target_scope_count_per_rank": 2,
    "target_attributed_device_total_median_ms":
        median(t["attributed_device_total_ms"] for t in all_targets),
    "target_attributed_device_total_min_ms":
        min(t["attributed_device_total_ms"] for t in all_targets),
    "target_attributed_device_total_max_ms":
        max(t["attributed_device_total_ms"] for t in all_targets),
    "target_host_scope_median_ms":
        median(t["host_scope_ms"] for t in all_targets),
    "nested_wait_event_median_ms":
        median(t["largest_nested_wait_event_ms"] for t in all_targets),
    "nested_hccl_allgather_median_ms":
        median(t["nested_hccl_allgather_sum_ms"] for t in all_targets),
    "whole_trace_grouped_matmul_sum_median_ms":
        median(r["whole_trace_grouped_matmul_sum_ms"] for r in ranks),
    "whole_trace_compute_median_ms":
        median(r["whole_trace_compute_ms"] for r in ranks),
    "whole_trace_communication_not_overlapped_median_ms":
        median(r["whole_trace_communication_not_overlapped_ms"]
               for r in ranks),
    "interpretation": (
        "Target CPU scopes only cover graph enqueue; timestamp clipping loses "
        "asynchronous graph kernels. Device Total Duration is profiler "
        "attribution, not a device interval union. Nested wait_event "
        "exposes preceding graph completion, not all-gather device latency. "
        "Whole-trace kernel sums include target and proposer and are not "
        "a target-only optimization bound."
    ),
    "ranks": ranks,
}
(ROOT / "attribution.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({k: v for k, v in summary.items() if k != "ranks"}, indent=2))
