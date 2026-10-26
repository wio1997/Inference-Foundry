#!/usr/bin/env python3
"""Compare two saved Level1 GMM2 bank traces without using profiled timing as service."""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "evidence/20260927_loop075_bound"
NATIVE = "aclnnGroupedMatmulWeightNz_GroupedMatmul_GroupedMatmul"
COUNTERS = [
    "aic_read_main_memory_datas(KB)",
    "aiv_read_main_memory_datas(KB)",
    "aic_write_main_memory_datas(KB)",
    "aiv_write_main_memory_datas(KB)",
    "aic_GM_to_L1_datas(KB)",
    "aiv_GM_to_UB_datas(KB)",
    "aic_total_cycles",
    "aiv_total_cycles",
    "Duration(us)",
]


def extract(run, expected_banks):
    base = ROOT / PREFIX / f"run{run}"
    request_path = base / "profile_request.json"
    request = json.loads(request_path.read_text())
    assert request["status"] == "profile_completed"
    assert request["rank"] == 4 and request["route_ordinal"] == 64
    assert request["weight_banks"] == expected_banks
    assert request["graph_captured_calls_per_replay"] == 16
    assert request["profile_active_replays"] == 2
    assert request["post_replay_all_outputs_finite"]
    paths = list((base / "profile").glob("**/kernel_details.csv"))
    assert len(paths) == 1, paths
    csv_path = paths[0]
    rows = list(csv.DictReader(csv_path.open(newline="")))
    assert all(row["Name"] == NATIVE for row in rows)
    # Profiler starts NPU collection partway through a warmup replay. The
    # final 32 rows are exactly two complete 0..15 Task-ID sequences.
    assert len(rows) >= 32
    complete = rows[-32:]
    task_ids = [int(row["Task ID"]) for row in complete]
    assert task_ids == list(range(16)) * 2, task_ids
    assert len({row["Model ID"] for row in complete}) == 1
    stat = {}
    for name in COUNTERS:
        values = [float(row[name]) for row in complete]
        stat[name] = {"median": statistics.median(values), "min": min(values), "max": max(values)}
    return {
        "run": f"run{run}",
        "weight_banks": expected_banks,
        "weight_bank_bytes": request["weight_bank_bytes"],
        "route_count_sha256": request["route_count_sha256"],
        "profile_csv": str(csv_path.relative_to(ROOT)),
        "profile_csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "native_rows_total": len(rows),
        "initial_partial_replay_rows": len(rows) - 32,
        "verified_complete_replay_native_rows": 32,
        "complete_replay_task_ids": task_ids,
        "counter_medians": stat,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    a, b = extract(358, 1), extract(359, 8)
    assert a["route_count_sha256"] == b["route_count_sha256"]
    delta = {}
    for key in COUNTERS:
        va = a["counter_medians"][key]["median"]
        vb = b["counter_medians"][key]["median"]
        delta[key] = {"bank1": va, "bank8": vb, "bank8_vs_bank1_percent": 100 * (vb / va - 1) if va else None}
    out = {
        "status": "verified_two_complete_graph_replays_each_condition",
        "scope": "onecard rank4 Run121 cycle64 ordinal64 GMM2, 16 captured calls/replay; synthetic zero data/weights",
        "conditions": [a, b],
        "median_counter_comparison": delta,
        "interpretation": [
            "Native task IDs 0..15 repeat twice after an initial partial profiler window, confirming at least two complete graph replays per condition.",
            "AIC main-memory read and GM-to-L1 data counters are unchanged; AIV read changes slightly. Rotating weights increases time/cycles under profiler and unprofiled all8 Run354-356, but this counter does not prove extra HBM bytes.",
            "Profiler reports acl-to-npu flow-event parse error in both conditions. Kernel rows are present, but Host/NPU correlation is not usable; profiler timing is not substituted for unprofiled Graph service.",
            "Only one rank and one layer shape have Level1 counters. This calibrates a conditional resource-service input, not compulsory traffic, full Target service, a scheduling bound, or Product TPS.",
        ],
        "bound_effect": {
            "algorithm_resource": "no change: compulsory work/traffic incomplete",
            "hardware_resource": "one conditional GMM2 repeated-versus-rotating-weight service sensitivity; counter-visible byte volume nearly unchanged; wider cache/locality/contended-capacity uncertainty remains",
            "scheduling_execution": "no change: isolated GMM Graph has no Target/DSpark/HCCL dependency DAG",
            "product_e2e": "no numeric upper bound or Current-to-Bound Gap update",
        },
    }
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"status": out["status"], "output": str(dest), "native_rows": [a["native_rows_total"], b["native_rows_total"]]}))


if __name__ == "__main__":
    main()
