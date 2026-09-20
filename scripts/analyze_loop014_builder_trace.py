"""Pair per-builder timings with pure-decode prepare steps in the c12 sample."""
import ast
import bisect
import csv
import glob
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

root = Path("/data/wio/Inference_Foundry/evidence/20260920_loop014_prepare_trace")
bench = json.loads((root / "run2/decode_c12.json").read_text())
lo = min(r["start"] for r in bench["requests"]) * 1e9
hi = max(r["end"] for r in bench["requests"]) * 1e9
out = {"bench_summary": bench["summary"], "monotonic_window_ns": [lo, hi],
       "filter": "builder call starts within pure-decode prepare step of measured c12 window",
       "ranks": {}}


def summarize(values):
    return {"count": len(values), "sum_ms": sum(values), "mean_ms": statistics.mean(values),
            "median_ms": statistics.median(values),
            "p90_ms": sorted(values)[int((len(values) - 1) * .9)],
            "max_ms": max(values)} if values else None


for path in sorted(glob.glob(str(root / "builder_raw2/rank*_pid*.txt"))):
    rank = int(Path(path).name.split("_")[0][4:])
    stage_path = glob.glob(str(root / f"raw2/rank{rank}_pid*.csv"))[0]
    steps = []
    with open(stage_path) as handle:
        for row in csv.reader(handle):
            if len(row) != 10:
                continue
            start, end = int(row[0]), int(row[6])
            if lo <= start < hi and int(row[9]) == 0:
                steps.append((start, end))
    steps.sort()
    starts = [s for s, _ in steps]
    totals = defaultdict(list)
    builder_names = Counter()
    group_names = Counter()
    matched = 0
    for line in Path(path).read_text().splitlines():
        parts = line.split(",", 6)
        if len(parts) != 7:
            continue
        start, dcp_start, dcp_end, end, num_reqs, num_tokens = map(int, parts[:6])
        ix = bisect.bisect_right(starts, start) - 1
        if ix < 0 or not (start <= steps[ix][1] and end <= steps[ix][1]):
            continue
        builders = ast.literal_eval(parts[6])
        matched += 1
        total_ms = (end - start) / 1e6
        dcp_ms = (dcp_end - dcp_start) / 1e6
        builder_ms = sum(b[3] for b in builders) / 1e6
        totals["metadata_total_ms"].append(total_ms)
        totals["dcp_ms"].append(dcp_ms)
        totals["builders_ms"].append(builder_ms)
        totals["residual_ms"].append(total_ms - dcp_ms - builder_ms)
        totals["num_builders"].append(len(builders))
        for gid, aid, name, duration in builders:
            builder_names[name] += 1
            group_names[f"{gid}:{aid}:{name}"] += 1
            totals[f"builder:{gid}:{aid}:{name}"].append(duration / 1e6)
    out["ranks"][str(rank)] = {
        "pure_decode_steps": len(steps), "matched_builder_steps": matched,
        "builder_name_counts": dict(builder_names), "group_builder_counts": dict(group_names),
        "stages": {k: summarize(v) for k, v in sorted(totals.items())},
    }
(root / "run2/builder_summary.json").write_text(json.dumps(out, indent=2) + "\n")
for rank, result in out["ranks"].items():
    print(rank, result["matched_builder_steps"],
          {k: round(v["median_ms"], 3) for k, v in result["stages"].items() if not k.startswith("builder:")})
    if rank == "0":
        print("rank0 builder medians:",
              {k: round(v["median_ms"], 3) for k, v in result["stages"].items() if k.startswith("builder:")})
