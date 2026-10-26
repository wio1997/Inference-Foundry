#!/usr/bin/env python3
"""Analyze first target collective arrival skew in synchronized Run107 traces."""
import json
import statistics
from pathlib import Path

ROOT = Path("/data/wio/Inference_Foundry")
TRACE_ROOT = ROOT / "evidence/20260924_loop038_cycle/run107/profile"
OUT = ROOT / "evidence/20260925_loop039_comm/run117/arrival_audit.json"

def union_ms(intervals):
    intervals = sorted(intervals)
    total = 0.0
    end = float("-inf")
    for a, b in intervals:
        if b > end:
            total += b - max(a, end)
            end = b
    return total / 1000

rows = []
for rank in range(8):
    path = next(TRACE_ROOT.glob(f"rank{rank}_*/ASCEND_PROFILER_OUTPUT/trace_view.json"))
    events = json.loads(path.read_text())
    scopes = sorted((e for e in events if e.get("name") == "extreme::target"), key=lambda e: float(e["ts"]))
    assert len(scopes) == 2, (rank, len(scopes))
    for cycle, scope in enumerate(scopes):
        a = float(scope["ts"]); b = a + float(scope["dur"])
        first = [e for e in events if e.get("name", "").startswith("hcom_reduceScatter__503_0_")
                 and a <= float(e.get("ts", 0)) < b]
        assert len(first) == 1, (rank, cycle, len(first))
        op = first[0]; start = float(op["ts"]); finish = start + float(op["dur"])
        comm = [e for e in events if e.get("ph") == "X" and e.get("name", "").startswith("hcom_")
                and float(e.get("ts", 0)) < b and float(e.get("ts", 0)) + float(e.get("dur", 0)) > a]
        intervals = [(max(a, float(e["ts"])), min(b, float(e["ts"]) + float(e["dur"]))) for e in comm]
        without_first = [(u, v) for u, v in intervals if abs(u - start) > .01 or abs(v - finish) > .01]
        union = union_ms(intervals)
        rows.append({
            "rank": rank, "cycle": cycle, "target_scope_start_us": a,
            "first_reduce_scatter_start_us": start, "first_reduce_scatter_end_us": finish,
            "first_reduce_scatter_duration_ms": float(op["dur"]) / 1000,
            "first_reduce_scatter_count": op.get("args", {}).get("count"),
            "first_reduce_scatter_dtype": op.get("args", {}).get("data_type"),
            "communication_union_ms": union,
            "first_reduce_scatter_union_contribution_ms": union - union_ms(without_first),
        })
by_cycle = []
for cycle in (0, 1):
    xs = [r for r in rows if r["cycle"] == cycle]
    def spread(k):
        v = [x[k] for x in xs]
        return (max(v) - min(v)) / 1000
    by_cycle.append({
        "cycle": cycle,
        "target_scope_start_skew_ms": spread("target_scope_start_us"),
        "first_reduce_scatter_start_skew_ms": spread("first_reduce_scatter_start_us"),
        "first_reduce_scatter_end_skew_ms": spread("first_reduce_scatter_end_us"),
        "first_reduce_scatter_duration_median_ms": statistics.median(x["first_reduce_scatter_duration_ms"] for x in xs),
        "first_reduce_scatter_duration_min_ms": min(x["first_reduce_scatter_duration_ms"] for x in xs),
        "first_reduce_scatter_duration_max_ms": max(x["first_reduce_scatter_duration_ms"] for x in xs),
        "communication_union_median_ms": statistics.median(x["communication_union_ms"] for x in xs),
        "first_reduce_scatter_union_contribution_median_ms": statistics.median(x["first_reduce_scatter_union_contribution_ms"] for x in xs),
    })
out = {
    "run": "run117",
    "source": "Loop038 Run107 synchronized target profile; cycle index 0/1 corresponds to cycles64/65",
    "op": "first hcom_reduceScatter__503_0_*",
    "all_counts": sorted(set(r["first_reduce_scatter_count"] for r in rows)),
    "all_dtypes": sorted(set(r["first_reduce_scatter_dtype"] for r in rows)),
    "by_cycle": by_cycle,
    "rank_windows": rows,
    "interpretation": "Near-common collective end timestamps and inverse start-duration relation indicate early ranks wait for late ranks inside the first collective. This wait contributes to measured communication union; the 10.2475 ms uncovered communication estimate cannot be treated as intrinsic transfer cost.",
    "limits": ["Run107 synchronized target diagnostic can create/amplify rank-arrival skew", "No unsynchronized causal attribution yet", "Collective duration may include both peer wait and real transfer", "No formal E2E inference"],
}
OUT.write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(by_cycle, indent=2))
