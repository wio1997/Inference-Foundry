#!/usr/bin/env python3
"""Cross-check stable Run98 stages against formal Run99 cohort wall."""
import json
import statistics
from pathlib import Path

ROOT = Path("/data/wio/Inference_Foundry")
DAG = ROOT / "evidence/20260924_loop036_metadata/run98/dag"
FORMAL = ROOT / "evidence/20260924_loop036_metadata/run99/runtime"
OUT = ROOT / "evidence/20260925_loop039_priority/run119/steady_priority.json"
profiles = [json.loads(p.read_text()) for p in sorted(DAG.glob("rank*.json"))]
assert [x["rank"] for x in profiles] == list(range(8))
assert all(x["cycles"] >= 256 for x in profiles)
indices = range(64, 256)
def summary(xs):
    return {"median": statistics.median(xs), "min": min(xs), "max": max(xs)}
runtime_keys = profiles[0]["runtime_stage_ms"][0].keys()
dspark_keys = profiles[0]["dspark_stage_ms"][0].keys()
runtime = {k: summary([r["runtime_stage_ms"][i][k] for r in profiles for i in indices])
           for k in runtime_keys}
dspark = {k: summary([r["dspark_stage_ms"][i][k] for r in profiles for i in indices])
          for k in dspark_keys}
formal_rows = []
for p in FORMAL.glob("rank*_cohort*.json"):
    x = json.loads(p.read_text())
    if x.get("pass") and x.get("cycles", 0) > 0:
        formal_rows.append({"rank": x["rank"], "cohort": x["cohort"],
                            "cycles": x["cycles"], "wall_seconds": x["wall_seconds"],
                            "wall_ms_per_cycle": 1000 * x["wall_seconds"] / x["cycles"]})
bench = json.loads((ROOT / "evidence/20260924_loop036_metadata/run98/bench.json").read_text())["summary"]
out = {
    "run": "run119",
    "steady_source": "Loop036 Run98 DAG event timing, cycles64-255, eight ranks",
    "sample_count": len(profiles) * len(indices),
    "runtime_stage_ms": runtime,
    "dspark_stage_ms": dspark,
    "run98_diagnostic_bench": {k: bench[k] for k in ("n","success","max_tokens","output_tps")},
    "formal_source": "Loop036 Run99 rank cohort wall; use for consistency only",
    "formal_cohort_count": len(formal_rows),
    "formal_wall_ms_per_cycle": summary([x["wall_ms_per_cycle"] for x in formal_rows]),
    "formal_cohorts": formal_rows,
    "decision": "The stable target (~46.6 ms) dominates proposer (~6.4 ms). Run106/107 first-collective duration is largely peer wait caused by rank arrival skew during profiled cycles, and cannot be promoted to a ~10 ms intrinsic communication removal opportunity. Keep GMM within target as the next bounded candidate, while seeking an exact semantics-valid substitution and measuring its effect on whole target stage.",
    "limits": ["Run98 DAG event measurements are diagnostic, not formal E2E", "Stage medians cannot be added without dependency/overlap proof", "Run99 cohort wall is not a component attribution", "GMM sum ~9.97 ms comes from synchronized Run107 and is an upper screen, not realizable gain"],
}
OUT.write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps({"runtime_target":runtime["target"],"runtime_proposer":runtime["proposer"],"dspark_model":dspark["model"],"formal_wall":out["formal_wall_ms_per_cycle"]},indent=2))
