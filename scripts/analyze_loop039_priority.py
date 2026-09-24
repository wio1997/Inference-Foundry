#!/usr/bin/env python3
"""Recompute target-stage opportunity bounds from existing synchronized windows."""
import json
import statistics
from pathlib import Path

ROOT = Path("/data/wio/Inference_Foundry")
trace = json.loads((ROOT / "evidence/20260924_loop038_cycle/run107/target_window.json").read_text())
shape = json.loads((ROOT / "evidence/20260925_loop039_gmm/run115/shape_summary.json").read_text())
excluded = {(x["rank"], x["cycle"]) for x in trace["excluded_windows"]}
windows = []
for rank in trace["ranks"]:
    for cycle in rank["cycles"]:
        if (rank["rank"], cycle["cycle"]) in excluded:
            continue
        windows.append({
            "rank": rank["rank"],
            "cycle": cycle["cycle"],
            "device_union_ms": cycle["device_union_ms"],
            "compute_union_ms": cycle["compute_union_ms"],
            "communication_union_ms": cycle["communication_union_ms"],
            "compute_communication_overlap_ms": cycle["compute_communication_overlap_ms"],
            "gmm_kernel_sum_ms": cycle["grouped_matmul_sum_ms"],
            "communication_uncovered_by_compute_ms": cycle["communication_union_ms"] - cycle["compute_communication_overlap_ms"],
            "non_gmm_compute_union_lower_bound_ms": cycle["compute_union_ms"] - cycle["grouped_matmul_sum_ms"],
        })
if len(windows) != 15 or not shape["all_rank_target_shape_observed"]:
    raise SystemExit("input evidence mismatch")
def stats(key):
    xs = [w[key] for w in windows]
    return {"median": statistics.median(xs), "min": min(xs), "max": max(xs)}
out = {
    "run": "run116",
    "source_profile": "Loop038 Run107 synchronized target windows",
    "source_shape": "Loop039 Run115 eight-rank shape envelope",
    "valid_windows": len(windows),
    "gmm_kernel_sum_ms": stats("gmm_kernel_sum_ms"),
    "communication_uncovered_by_compute_ms": stats("communication_uncovered_by_compute_ms"),
    "non_gmm_compute_union_lower_bound_ms": stats("non_gmm_compute_union_lower_bound_ms"),
    "device_union_ms": stats("device_union_ms"),
    "target_shape": shape["rank_records"][0]["target_96x6_shape"],
    "decision": "GMM is a bounded candidate, not proven highest-value. Require a semantics-valid per-channel W4A8 alternative and same-state eight-rank A/B/A. Check actual group counts, output/acceptance parity, GMM interval union and full target stage before formal E2E. If no candidate can plausibly save >=5 ms/cycle or gain does not reach target stage, pivot to communication/non-GMM path.",
    "limits": ["Synchronized diagnostic profile, not formal E2E", "Kernel sums overlap and are not stage reductions", "Run115 static shapes do not contain live expert counts", "Communication uncovered time can include rank arrival/synchronization wait"],
    "windows": windows,
}
p = ROOT / "evidence/20260925_loop039_gmm/run116/priority_audit.json"
p.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
print(json.dumps({k: out[k] for k in ("valid_windows", "gmm_kernel_sum_ms", "communication_uncovered_by_compute_ms", "non_gmm_compute_union_lower_bound_ms")}))
