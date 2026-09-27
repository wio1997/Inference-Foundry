#!/usr/bin/env python3
"""Same-layer all8 repeated/rotating/repeated GMM Graph bank comparison."""
import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "evidence/20260927_loop075_bound"


def load(run):
    folder = BASE / run
    summary = json.loads((folder / "summary.json").read_text())
    rows = [json.loads((folder / f"rank{rank}.json").read_text()) for rank in range(8)]
    return summary, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    a, ar = load("run354")
    b, br = load("run355")
    a2, a2r = load("run356")
    assert [s["weight_banks"] for s in (a, b, a2)] == [1, 8, 1]
    assert all(s["route_ordinal"] == 64 and s["ranks"] == 8 and s["graph_ops_per_replay"] == 16 for s in (a, b, a2))
    assert all(s["cases"][case]["all_rank_overlap_ms"] > 20 for s in (a, b, a2) for case in ("gmm1", "gmm2"))
    for rank in range(8):
        for key in ("route_ordinal", "route_counts", "route_tokens", "active_experts", "w1_shape", "w2_shape", "weight_format"):
            assert ar[rank][key] == br[rank][key] == a2r[rank][key]
        assert ar[rank]["weight_bank_bytes_per_rank"] * 8 == br[rank]["weight_bank_bytes_per_rank"]
        assert ar[rank]["weight_bank_bytes_per_rank"] == a2r[rank]["weight_bank_bytes_per_rank"]
        for source in (ar, br, a2r):
            assert all(source[rank]["cases"][case]["post_replay_all_captured_outputs_finite"] for case in ("gmm1", "gmm2"))
    cases = {}
    for case in ("gmm1", "gmm2"):
        rows = []
        for rank in range(8):
            old, rotate, new = [source[rank]["cases"][case]["median_us"] for source in (ar, br, a2r)]
            rows.append({"rank": rank, "route_tokens": ar[rank]["route_tokens"],
                         "active_experts": ar[rank]["active_experts"],
                         "single_bank_A_us": old, "eight_bank_B_us": rotate,
                         "single_bank_A2_us": new,
                         "B_minus_A_midpoint_us": rotate - (old + new) / 2,
                         "B_vs_A_midpoint_percent": 100 * (rotate / ((old + new) / 2) - 1),
                         "B_slower_than_both_controls": rotate > max(old, new)})
        cases[case] = {"ranks": rows,
                       "median_B_vs_A_midpoint_percent": statistics.median(r["B_vs_A_midpoint_percent"] for r in rows),
                       "strict_B_slower_ranks": sum(r["B_slower_than_both_controls"] for r in rows),
                       "max_rank_A_us": max(r["single_bank_A_us"] for r in rows),
                       "max_rank_B_us": max(r["eight_bank_B_us"] for r in rows),
                       "max_rank_A2_us": max(r["single_bank_A2_us"] for r in rows)}
    output = {"status": "valid_same_layer_bank_A_B_A2_timing_only",
              "source": "Run121 cycle64 ordinal64 all8 route counts; Run354 bank1 / Run355 bank8 / Run356 bank1",
              "route_ordinal": 64,
              "weight_bank_bytes_per_rank": {"single": ar[0]["weight_bank_bytes_per_rank"],
                                             "eight": br[0]["weight_bank_bytes_per_rank"]},
              "all8_host_window_overlap_ms": {run: {case: s["cases"][case]["all_rank_overlap_ms"] for case in ("gmm1", "gmm2")}
                                              for run, s in (("run354", a), ("run355", b), ("run356", a2))},
              "cases": cases,
              "limits": ["synthetic zero data/weights and isolated GMM Graph; no Target/Draft/HCCL overlap",
                         "bank rotation has eight independent physical tensors and two visits per bank per replay, but no MemoryAccess counter yet",
                         "device event per-op includes amortized Graph submission; native task count not trace-verified",
                         "no Product E2E or strict Resource/Hardware lower bound follows"],
              "next_gate": "short Level1 MemoryAccess trace for same rank/ordinal bank1 and bank8 to verify replay native op count and traffic without using profiler timing as service"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"status": output["status"], "summary": {k: {j: v[j] for j in ("median_B_vs_A_midpoint_percent", "strict_B_slower_ranks", "max_rank_A_us", "max_rank_B_us", "max_rank_A2_us")} for k, v in cases.items()}}))


if __name__ == "__main__":
    main()
