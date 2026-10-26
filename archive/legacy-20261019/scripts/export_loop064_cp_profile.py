#!/usr/bin/env python3
"""Export torch_npu cycle-window profiles after a Loop064 CP service stops."""
import argparse
from pathlib import Path
from torch_npu.profiler.profiler import analyse

parser = argparse.ArgumentParser()
parser.add_argument("--profile-root", required=True)
parser.add_argument("--latest-per-rank", action="store_true")
args = parser.parse_args()
root = Path(args.profile_root)
paths = sorted(root.glob("rank*_ascend_pt"))
if args.latest_per_rank:
    latest = {}
    for path in paths:
        rank = path.name.split("_", 1)[0]
        latest[rank] = path
    paths = sorted(latest.values())
if not paths:
    raise SystemExit("No rank profile directories")
print("directories", len(paths), flush=True)
for path in paths:
    if (path / "ASCEND_PROFILER_OUTPUT" / "trace_view.json").exists():
        print("EXPORTED", path.name, flush=True)
        continue
    print("ANALYZE", path.name, flush=True)
    analyse(str(path))
