"""Stream TP0 host scope events during the c12 profiled request window."""
import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path

import ijson
import pandas as pd

root = Path("/data/wio/Inference_Foundry/evidence/20260920_decode_c12_profile")
phase = json.loads((root / "run4/phase_times.json").read_text())
t0 = pd.Timestamp(phase["profile_started_utc"]).timestamp() * 1e6
t1 = pd.Timestamp(phase["profiled_request_done_utc"]).timestamp() * 1e6
paths = glob.glob(str(root / "torch_raw/dp0_pp0_tp0*/ASCEND_PROFILER_OUTPUT/trace_view.json"))
if len(paths) != 1:
    raise RuntimeError(paths)
scopes = {"prepare input", "forward", "post process", "sample_token", "draft_token", "async_state_update"}
records = defaultdict(list)
count = 0
with open(paths[0], "rb") as handle:
    for event in ijson.items(handle, "item"):
        count += 1
        if event.get("ph") != "X" or event.get("cat") != "cpu_op":
            continue
        name = event.get("name")
        if name not in scopes:
            continue
        ts = float(event["ts"])
        if t0 <= ts < t1:
            records[name].append(float(event["dur"]) / 1000)
out = {"trace": paths[0], "events_scanned": count, "wall_s": (t1 - t0) / 1e6, "scopes": {}}
for name, values in sorted(records.items()):
    out["scopes"][name] = {"count": len(values), "sum_ms": sum(values),
                            "mean_ms": statistics.mean(values), "median_ms": statistics.median(values),
                            "p90_ms": sorted(values)[int((len(values) - 1) * .9)],
                            "max_ms": max(values)}
(root / "run4/host_scope_c12_summary.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
