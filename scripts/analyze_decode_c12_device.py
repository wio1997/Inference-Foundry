"""Clip eight-rank torch-NPU kernel timelines to the c12 profiled request window."""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

root = Path("/data/wio/Inference_Foundry/evidence/20260920_decode_c12_profile")
phase = json.loads((root / "run4/phase_times.json").read_text())
t0 = pd.Timestamp(phase["profile_started_utc"]).timestamp() * 1e6
t1 = pd.Timestamp(phase["profiled_request_done_utc"]).timestamp() * 1e6
paths = sorted(glob.glob(str(root / "torch_raw/dp0_pp0_tp*_dcp0_ep*_rank*_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv")))
if len(paths) != 8:
    raise RuntimeError(f"expected eight rank CSVs, found {len(paths)}")


def union(starts, ends):
    if len(starts) == 0:
        return 0.0, 0.0
    ix = np.argsort(starts)
    starts, ends = starts[ix], ends[ix]
    left = right = starts[0]
    total = longest = 0.0
    for start, end in zip(starts[1:], ends[1:]):
        if start > right:
            total += right - left
            longest = max(longest, start - right)
            left, right = start, end
        else:
            right = max(right, end)
    return total + right - left, longest


out = {"profile_start_utc": phase["profile_started_utc"], "request_done_utc": phase["profiled_request_done_utc"],
       "wall_s": (t1 - t0) / 1e6, "ranks": {}}
for path in paths:
    rank = int(Path(path).parts[-3].split("_rank")[1].split("_")[0])
    d = pd.read_csv(path, usecols=["Task ID", "Type", "Start Time(us)", "Duration(us)"])
    d = d.dropna(subset=["Task ID", "Type", "Start Time(us)", "Duration(us)"])
    x = d[(d["Start Time(us)"] + d["Duration(us)"] > t0) & (d["Start Time(us)"] < t1)].copy()
    x["start"] = np.maximum(x["Start Time(us)"], t0)
    x["end"] = np.minimum(x["Start Time(us)"] + x["Duration(us)"], t1)
    x["category"] = np.where(x["Type"].str.startswith("hcom_"), "communication", "compute_or_copy")
    result = {"events": len(x), "categories": {}, "top_types_sum_s_overlaps": {}}
    for label, frame in [("all", x), *list(x.groupby("category"))]:
        busy, gap = union(frame["start"].to_numpy(), frame["end"].to_numpy())
        result["categories"][label] = {"events": len(frame), "union_busy_s": busy / 1e6,
                                        "longest_intra_event_gap_ms": gap / 1e3,
                                        "sum_duration_s_overlaps": float((frame["end"] - frame["start"]).sum() / 1e6)}
    result["top_types_sum_s_overlaps"] = (
        x.assign(clipped_duration=x["end"] - x["start"]).groupby("Type")["clipped_duration"]
        .sum().sort_values(ascending=False).head(20) / 1e6
    ).to_dict()
    out["ranks"][str(rank)] = result
target = root / "run4/device_c12_summary.json"
target.write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps({"wall_s": out["wall_s"], "ranks": {k: v["categories"] for k, v in out["ranks"].items()}}, indent=2))
