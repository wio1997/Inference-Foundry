"""Stream a representative TP0 draft and prepare scope without loading the trace."""
import glob
import json
from collections import Counter
from pathlib import Path

import ijson
import pandas as pd

root = Path("/data/wio/Inference_Foundry/evidence/20260920_decode_c12_profile")
marks = json.loads((root / "run4/phase_times.json").read_text())
start = pd.Timestamp(marks["profile_started_utc"]).timestamp() * 1e6
end = pd.Timestamp(marks["profiled_request_done_utc"]).timestamp() * 1e6
path = glob.glob(str(root / "torch_raw/dp0_pp0_tp0*/ASCEND_PROFILER_OUTPUT/trace_view.json"))[0]
wanted = {"draft_token", "prepare input"}
scopes = {name: [] for name in wanted}
with open(path, "rb") as handle:
    for event in ijson.items(handle, "item"):
        if event.get("ph") == "X" and event.get("cat") == "cpu_op" and event.get("name") in wanted:
            ts = float(event["ts"])
            if start <= ts < end:
                scopes[event["name"]].append((float(event["dur"]), ts, event["tid"]))
chosen = {name: sorted(values)[len(values) // 2] for name, values in scopes.items()}
children = {name: [] for name in chosen}
with open(path, "rb") as handle:
    for event in ijson.items(handle, "item"):
        if event.get("ph") != "X" or event.get("cat") != "cpu_op":
            continue
        ts = float(event["ts"])
        duration = float(event["dur"])
        if duration < 100:
            continue
        for name, (span, lo, tid) in chosen.items():
            if event.get("tid") == tid and lo <= ts and ts + duration <= lo + span and not (
                event.get("name") == name and ts == lo
            ):
                children[name].append({"name": event["name"], "start_ms": (ts - lo) / 1000,
                                       "duration_ms": duration / 1000})
out = {"trace": path, "representatives": {}}
for name, (span, lo, tid) in chosen.items():
    total = Counter()
    for item in children[name]:
        total[item["name"]] += item["duration_ms"]
    out["representatives"][name] = {
        "scope_duration_ms": span / 1000, "tid": tid,
        "top_child_duration_sums_ms": total.most_common(30),
        "top_individual_children": sorted(children[name], key=lambda item: item["duration_ms"], reverse=True)[:30],
    }
(root / "run4/host_children_c12_summary.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out["representatives"], indent=2)[:5000])
