#!/usr/bin/env python3
"""Distill full-service cold msprof Compressor tasks without summing them into TTFT."""
import csv
import glob
import json
import statistics
from pathlib import Path

root = Path("/data/wio/Inference_Foundry")
source = Path(glob.glob(str(root / "evidence/20260920_diagnostic/app_profile_cold2/raw/*/mindstudio_profiler_output/op_summary_*.csv"))[0])
out = root / "evidence/20260920_loop017_compressor/profile_clusters.json"
out.parent.mkdir(parents=True, exist_ok=True)
shapes = ("8096,4096;1024,4096", "8096,4096;512,4096", "8096,4096;256,4096")
records = {s: [] for s in shapes}
with source.open() as f:
    for row in csv.DictReader(f):
        if row["OP Type"] != "Compressor":
            continue
        for shape in shapes:
            if shape in row["Input Shapes"]:
                records[shape].append((float(row["Task Start Time(us)"].strip()), float(row["Task Duration(us)"]), float(row["aiv_time(us)"])))
                break
result = {"source": str(source), "cluster_gap_threshold_us": 500000, "shapes": {}}
for shape, tasks in records.items():
    tasks.sort()
    clusters = []
    for task in tasks:
        if not clusters or task[0] - clusters[-1][-1][0] > 500000:
            clusters.append([])
        clusters[-1].append(task)
    result["shapes"][shape] = {
        "count": len(tasks),
        "median_task_us": statistics.median(x[1] for x in tasks),
        "median_aiv_us": statistics.median(x[2] for x in tasks),
        "clusters": [
            {"count": len(c), "span_ms": (c[-1][0]-c[0][0])/1000,
             "sum_task_ms": sum(x[1] for x in c)/1000,
             "median_task_us": statistics.median(x[1] for x in c)}
            for c in clusters
        ]
    }
out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result["shapes"], indent=2))
