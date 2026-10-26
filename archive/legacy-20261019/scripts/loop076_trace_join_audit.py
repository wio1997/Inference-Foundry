#!/usr/bin/env python3
"""Read-only audit of whether saved FULL Graph traces close a same-cycle DAG."""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "evidence/20260926_loop060_resource/run246/profile"
ROUTES = ROOT / "evidence/20260925_loop039_gmm/run121/counts"
STAGES = ("extreme::target", "extreme::acceptance", "extreme::proposer", "extreme::dspark_model", "extreme::draft_commit")


def f(row, key):
    try:
        return float(row.get(key, "").strip())
    except (ValueError, AttributeError):
        return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ranks = []
    for rank in range(8):
        dirs = sorted(PROFILE.glob(f"rank{rank}_*/ASCEND_PROFILER_OUTPUT"))
        assert len(dirs) == 5, (rank, len(dirs))
        folder = dirs[-1]
        trace_path = folder / "trace_view.json"
        csv_path = folder / "kernel_details.csv"
        events = json.loads(trace_path.read_text())
        kernels = list(csv.DictReader(csv_path.open(newline="")))
        stages = {}
        for name in STAGES:
            scopes = sorted((e for e in events if e.get("cat") == "cpu_op" and e.get("name") == name), key=lambda e: float(e["ts"]))
            assert len(scopes) == 2, (rank, name, len(scopes))
            stages[name] = [{"host_start_us": float(e["ts"]), "host_end_us": float(e["ts"]) + float(e["dur"]),
                             "host_duration_ms": float(e["dur"]) / 1000} for e in scopes]
        target = []
        for idx, scope in enumerate(stages["extreme::target"]):
            selected = [row for row in kernels if scope["host_start_us"] <= f(row, "Start Time(us)") < scope["host_end_us"]]
            gmm1 = [row for row in selected if row["Name"].startswith("aclnnGroupedMatmulSwigluQuantWeightNzV2_")]
            gmm2 = [row for row in selected if row["Name"].startswith("aclnnGroupedMatmulWeightNz_")]
            assert len(gmm1) == len(gmm2) == 43, (rank, idx, len(gmm1), len(gmm2))
            assert all(f(gmm2[i], "Start Time(us)") < f(gmm2[i + 1], "Start Time(us)") for i in range(42))
            target.append({
                "target_index_in_capture": idx,
                "gmm1_native_tasks": len(gmm1),
                "gmm2_native_tasks": len(gmm2),
                "gmm2_ordinal21": {
                    "native_start_us": f(gmm2[21], "Start Time(us)"),
                    "native_duration_us_profiled": f(gmm2[21], "Duration(us)"),
                    "aic_read_KB": f(gmm2[21], "aic_read_main_memory_datas(KB)"),
                    "aiv_read_KB": f(gmm2[21], "aiv_read_main_memory_datas(KB)"),
                    "aic_GM_to_L1_KB": f(gmm2[21], "aic_GM_to_L1_datas(KB)"),
                    "input_shapes": gmm2[21].get("Input Shapes"),
                    "input_types": gmm2[21].get("Input Data Types"),
                    "input_formats": gmm2[21].get("Input Formats"),
                },
                "gmm2_total_AIC_read_GB": sum(f(r, "aic_read_main_memory_datas(KB)") for r in gmm2) * 1024 / 1e9,
                "gmm2_total_AIV_read_GB": sum(f(r, "aiv_read_main_memory_datas(KB)") for r in gmm2) * 1024 / 1e9,
            })
        route_path = ROUTES / f"rank{rank}_cycle64.json"
        route = json.loads(route_path.read_text())
        assert route["rank"] == rank and route["cycle"] == 64
        counts = route["rows"][64]["counts"]
        ranks.append({
            "rank": rank,
            "run246_latest_capture_folder": str(folder.relative_to(ROOT)),
            "run246_csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            "run121_route_path": str(route_path.relative_to(ROOT)),
            "run121_route_ordinal64_count_sha256": hashlib.sha256(json.dumps(counts).encode()).hexdigest(),
            "run121_route_ordinal64_tokens": sum(counts),
            "stages_host": stages,
            "target_native": target,
            "host_interstage_gaps_us": {
                "target0_end_to_proposer0_start": stages["extreme::proposer"][0]["host_start_us"] - stages["extreme::target"][0]["host_end_us"],
                "proposer0_end_to_target1_start": stages["extreme::target"][1]["host_start_us"] - stages["extreme::proposer"][0]["host_end_us"],
            },
        })
    out = {
        "status": "saved_trace_partial_join_only",
        "ranks": ranks,
        "gmm2_ordinal21_profiled_duration_us_range_by_target": [
            [min(row["target_native"][i]["gmm2_ordinal21"]["native_duration_us_profiled"] for row in ranks),
             max(row["target_native"][i]["gmm2_ordinal21"]["native_duration_us_profiled"] for row in ranks)]
            for i in range(2)
        ],
        "what_closes": [
            "Run246 latest profiler capture has two original Target and proposer Host scopes per rank and exactly 43 ordered native GMM1/GMM2 tasks per Target scope.",
            "Kernel MemoryAccess, native start/duration and Stage Host order can be indexed together within the same Run246 capture.",
        ],
        "missing_join": [
            "Run121 route counts were captured in a separate service run; matching cycle64 ordinal64 labels do not prove identical routed counts or state in Run246.",
            "Run246 trace has no same-cycle per-GMM route count or packed-weight address/physical cache-hit ledger, and no explicit device-ready-to-downstream-join correlation for Target versus DSpark.",
            "Profiler perturbs absolute service; Host scopes and kernel sums are neither native critical-path cost nor compulsory traffic.",
            "Only two profiled Target cycles per capture; not accepted formal Run99 trajectory.",
        ],
        "next_minimal_capture": "On original all8 FULL Graph, record same-cycle per-rank Target GMM ordinal route fingerprint and pointers plus existing Level1 task/counter and Target/DSpark downstream device join markers; keep two-cycle capture and compare no-profiler control. Do not inject synchronization into hot path.",
        "bound_effect": "No numeric Resource/Hardware, Scheduling or Product promotion; identifies exact missing source-route/device-join observables.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"status": out["status"], "ranks": len(ranks), "gmm2_ordinal21_profiled_duration_us_range_by_target": out["gmm2_ordinal21_profiled_duration_us_range_by_target"]}))


if __name__ == "__main__":
    main()
