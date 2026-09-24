#!/usr/bin/env python3
"""Check first collective peer wait against unsynchronized Run106 host stages."""
import json
import statistics
from pathlib import Path

ROOT = Path("/data/wio/Inference_Foundry")
TRACES = ROOT / "evidence/20260924_loop038_cycle/run106/profile"
OUT = ROOT / "evidence/20260925_loop039_comm/run118/unsynced_arrival.json"
NAMES = ("extreme::prepare_target", "extreme::derived_target_metadata",
         "extreme::target", "extreme::proposer")
rows = []
for rank in range(8):
    p = next(TRACES.glob(f"rank{rank}_*/ASCEND_PROFILER_OUTPUT/trace_view.json"))
    xs = json.loads(p.read_text())
    scopes = {name: sorted((e for e in xs if e.get("name") == name), key=lambda e: float(e["ts"]))
              for name in NAMES}
    assert all(len(scopes[name]) == 2 for name in NAMES), rank
    for cycle in (0, 1):
        op_name = f"hcom_reduceScatter__503_0_{cycle+1}"
        matches = [e for e in xs if e.get("name") == op_name]
        assert len(matches) == 1, (rank, cycle)
        op = matches[0]
        row = {"rank": rank, "cycle": cycle,
               "first_collective_name": op_name,
               "first_collective_count": op.get("args", {}).get("count"),
               "first_collective_dtype": op.get("args", {}).get("data_type"),
               "first_collective_start_us": float(op["ts"]),
               "first_collective_end_us": float(op["ts"]) + float(op["dur"]),
               "first_collective_duration_ms": float(op["dur"]) / 1000}
        for name in NAMES:
            label = name.split("::")[1]
            e = scopes[name][cycle]
            row[f"{label}_start_us"] = float(e["ts"])
            row[f"{label}_end_us"] = float(e["ts"]) + float(e["dur"])
            row[f"{label}_host_duration_ms"] = float(e["dur"]) / 1000
        row["prepare_to_target_ms"] = (row["target_start_us"] - row["prepare_target_start_us"]) / 1000
        row["target_to_first_collective_ms"] = (row["first_collective_start_us"] - row["target_start_us"]) / 1000
        rows.append(row)
by_cycle = []
for cycle in (0, 1):
    z = [x for x in rows if x["cycle"] == cycle]
    def spread(key):
        v = [x[key] for x in z]
        return (max(v) - min(v)) / 1000
    by_cycle.append({
        "cycle": cycle,
        "prepare_start_skew_ms": spread("prepare_target_start_us"),
        "target_start_skew_ms": spread("target_start_us"),
        "first_collective_start_skew_ms": spread("first_collective_start_us"),
        "first_collective_end_skew_ms": spread("first_collective_end_us"),
        "first_collective_duration_median_ms": statistics.median(x["first_collective_duration_ms"] for x in z),
        "first_collective_duration_min_ms": min(x["first_collective_duration_ms"] for x in z),
        "first_collective_duration_max_ms": max(x["first_collective_duration_ms"] for x in z),
        "prepare_to_target_median_ms": statistics.median(x["prepare_to_target_ms"] for x in z),
        "target_to_first_collective_median_ms": statistics.median(x["target_to_first_collective_ms"] for x in z),
        "proposer_host_duration_median_ms": statistics.median(x["proposer_host_duration_ms"] for x in z),
    })
out = {
    "run": "run118",
    "source": "Loop038 Run106 unsynchronized two-cycle profiler; graph replay is async so target CPU scope does not clip device DAG",
    "by_cycle": by_cycle,
    "rows": rows,
    "interpretation": "Peer-wait signature persists without target synchronization. Rank arrival skew is present at prepare_target entry, before the target graph starts; first reduce-scatter ends align across ranks. The first collective duration is largely a symptom of upstream rank scheduling/skew, not an independently removable HCCL transfer duration.",
    "limits": ["Only two profiled cycles", "Graph replay is asynchronous: CPU scopes cannot establish the full target device interval", "Earlier cycle rank skew source is not yet localized", "No formal E2E inference"],
}
OUT.write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(by_cycle, indent=2))
