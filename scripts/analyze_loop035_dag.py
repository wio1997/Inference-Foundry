#!/usr/bin/env python3
import json
import statistics
from pathlib import Path

root = Path("/data/wio/Inference_Foundry/evidence/20260924_loop035_diagnostic/run87")
rows = [json.loads(p.read_text()) for p in sorted((root / "dag").glob("rank*.json"))]
cohorts = [json.loads(p.read_text()) for p in sorted((root / "runtime").glob("rank*_cohort1.json"))]
assert len(rows) == len(cohorts) == 8
assert all(row["cycles"] == len(row["runtime_stage_ms"]) == len(row["dspark_stage_ms"]) for row in rows)
assert all(c["pass"] and c["generated_output_counts"] == [1024] * 12 for c in cohorts)
windows = [(1, 8), (8, 64), (64, 128), (128, 256), (256, 1024)]
summary = {"run": "run87", "rank_count": 8,
           "cycles_by_rank": {str(r["rank"]): r["cycles"] for r in rows},
           "stage_median_ms": {}, "windows": {}}
for kind, field in (("runtime", "runtime_stage_ms"), ("dspark", "dspark_stage_ms")):
    labels = sorted(set().union(*(r[field][0].keys() for r in rows)))
    summary["stage_median_ms"][kind] = {
        k: statistics.median([cyc[k] for r in rows for cyc in r[field][1:] if k in cyc])
        for k in labels
    }
    for start, end in windows:
        wkey = f"{start}-{end-1}"
        dst = summary["windows"].setdefault(wkey, {})
        dst[kind] = {
            k: statistics.median([r[field][i][k] for r in rows for i in range(start, min(end, r["cycles"])) if k in r[field][i]])
            for k in labels
            if any(r["cycles"] > start for r in rows)
        }
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
