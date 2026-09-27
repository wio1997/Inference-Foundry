#!/usr/bin/env python3
"""Gate all8/one-card/all8 GMM Graph capacity without E2E extrapolation."""
import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "evidence/20260927_loop075_bound"


def read(run):
    folder = BASE / run
    summary = json.loads((folder / "summary.json").read_text())
    rows = [json.loads((folder / f"rank{rank}.json").read_text()) for rank in range(summary["ranks"])]
    return summary, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    a, ar = read("run349")
    b, br = read("run350")
    a2, a2r = read("run351")
    assert a["ranks"] == a2["ranks"] == 8 and b["ranks"] == 1
    assert all(s["graph_ops_per_replay"] == 16 and s["samples_per_rank_case"] == 30 for s in (a, b, a2))
    assert all(s["cases"][case]["all_rank_overlap_ms"] > 25 for s in (a, a2) for case in ("gmm1", "gmm2"))
    for rank in range(8):
        for key in ("route_tokens", "active_experts", "w1_shape", "w2_shape", "weight_format"):
            assert ar[rank][key] == a2r[rank][key]
    for key in ("route_tokens", "active_experts", "w1_shape", "w2_shape", "weight_format"):
        assert ar[0][key] == br[0][key]
    source_ordinals = []
    for rank in range(8):
        snap = json.loads((ROOT / f"evidence/20260925_loop039_gmm/run121/counts/rank{rank}_cycle64.json").read_text())
        routes = snap["rows"][43:]
        selected = min(routes, key=lambda row: abs(sum(x > 0 for x in row["counts"]) - 15) + abs(sum(row["counts"]) - 69) / 10)
        assert sum(selected["counts"]) == ar[rank]["route_tokens"]
        assert sum(x > 0 for x in selected["counts"]) == ar[rank]["active_experts"]
        source_ordinals.append(selected["ordinal"])
    cases = {}
    for case in ("gmm1", "gmm2"):
        old = ar[0]["cases"][case]["median_us"]
        single = br[0]["cases"][case]["median_us"]
        new = a2r[0]["cases"][case]["median_us"]
        medians = [row["cases"][case]["median_us"] for row in a2r]
        cases[case] = {
            "rank0_A_all8_us": old,
            "rank0_B_single_us": single,
            "rank0_A2_all8_us": new,
            "rank0_A_to_A2_drift_percent": 100 * (new / old - 1),
            "rank0_single_vs_all8_midpoint_percent": 100 * (single / ((old + new) / 2) - 1),
            "all8_A2_rank_median_us_range": [min(medians), max(medians)],
            "all8_A2_median_of_rank_medians_us": statistics.median(medians),
        }
    counter1 = json.loads((ROOT / "evidence/20260925_loop044_target/run148/counter_analysis.json").read_text())
    counter2 = json.loads((ROOT / "evidence/20260925_loop044_target/run150/counter_analysis.json").read_text())
    output = {
        "status": "valid_isolated_graph_shape_service_ABA_no_product_bound",
        "contract": "8x910B3, CANN9.1, driver26.0.rc1; per-rank independently selected Run121 count vectors from different ordinals, synthetic zero data/weights, GMM1/GMM2 W4A8 product shapes",
        "selected_source_ordinal_by_rank": source_ordinals,
        "samples": "30 Graph replays x 16 same-shape operators per rank and case; device event span divided by 16",
        "all8_host_window_overlap_not_per_replay_device_proof": {run: {case: s["cases"][case]["all_rank_overlap_ms"] for case in ("gmm1", "gmm2")}
                         for run, s in (("run349", a), ("run351", a2))},
        "cases": cases,
        "prior_one_card_Level1_kernel_us_different_method": {"gmm1_run148": counter1["kernel_duration_us"]["median"],
                                                        "gmm2_run150": counter2["kernel_duration_us"]["median"]},
        "pilot_run348": "80 direct event-wrapped operators/rank with Host enqueue in each event span; excluded from hardware service calibration",
        "interpretation": "Eight Host windows overlap under independent repeated-weight Graph service. Rank0 A/B/A2 differences are of the same order as cross-run drift; statistical equivalence is not established. This is a timing-only service sample, not a coherent real-layer eight-rank route, sustained FULL Graph capacity or compulsory Product work.",
        "remaining": ["one same-layer same-cycle all8 route and multiple independent weight banks to test cache transfer", "short Graph replay native task/counter and output check", "FULL Graph target/draft simultaneous AIC/AIV/HBM/HCCL service and contention", "complete necessary per-cycle arithmetic and traffic", "real product DAG and external output coupling"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"status": output["status"], "cases": cases}))


if __name__ == "__main__":
    main()
