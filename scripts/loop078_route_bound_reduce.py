#!/usr/bin/env python3
"""Reduce validated per-token route to same-run conditional work/weight sets."""
import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHT_BYTES = 12_582_912
PAIR_GFLOP = 2 * (4096 * 4096 + 2048 * 4096) / 1e9


def lower_unique(hist, useful):
    if useful == 0:
        return 0
    capacity = sorted((min(n, useful) for n in hist.values()), reverse=True)
    covered = 0
    for k, count in enumerate(capacity, 1):
        covered += count
        if covered >= 6 * useful:
            return k
    raise ValueError("insufficient route histogram capacity")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capture-dir", type=Path, required=True)
    p.add_argument("--validated", type=Path, required=True)
    p.add_argument("--post-count", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    assert int(args.post_count.read_text()) == 60
    valid = json.loads(args.validated.read_text())
    assert valid["valid"] and len(valid["cohorts"]) == 5
    rows = []
    for cohort in valid["cohorts"]:
        c = cohort["cohort"]
        raw = json.loads((args.capture_dir / f"rank0_cohort{c}.json").read_text())
        assert raw["cohort"] == c
        for cyc in (64, 65):
            info = cohort["cycles"][str(cyc)]
            useful = info["useful_rows"]
            target = sorted(raw["records"][str(cyc)]["target"], key=lambda x:x["ordinal"])
            draft = sorted(raw["records"][str(cyc)]["draft"], key=lambda x:x["ordinal"])
            assert len(target) == 43 and len(draft) == 3
            conditional_per_layer = info["models"]["target"]["retained_experts_per_layer"]
            assert len(conditional_per_layer) == 43
            lower = upper = current = conditional = 0
            for i, layer in enumerate(target):
                ids = layer["ids"]
                assert len(ids) == 96
                hist = Counter(ex for row in ids for ex in row)
                l = lower_unique(hist, useful)
                u = min(len(hist), 6 * useful)
                e = conditional_per_layer[i]
                assert l <= e <= u
                lower += l; upper += u; current += len(hist); conditional += e
            assert current == sum(info["models"]["target"]["active_experts_per_layer"])
            q = info["models"]["draft"]["rows"]
            draft_active = sum(info["models"]["draft"]["active_experts_per_layer"])
            assert q in (84, 96)
            assert all(len(layer["ids"]) == q for layer in draft)
            rows.append({
                "cohort":c, "cycle":cyc, "runtime_clipped_useful_rows":useful,
                "target_current_candidate_rows":96,
                "target_current_standard_GMM_GFLOP_TP8":96*6*43*PAIR_GFLOP,
                "target_clairvoyant_retained_standard_GMM_GFLOP_TP8":useful*6*43*PAIR_GFLOP,
                "target_expert_layer_pair_counts":{"histogram_lower":lower,"conditional_slot_major_selected":conditional,
                    "histogram_upper":upper,"current_active":current},
                "target_packed_weight_set_GB_TP8":{"histogram_lower":lower*WEIGHT_BYTES/1e9,
                    "conditional_slot_major_selected":conditional*WEIGHT_BYTES/1e9,
                    "histogram_upper":upper*WEIGHT_BYTES/1e9,
                    "current_active":current*WEIGHT_BYTES/1e9},
                "draft_actual_rows":q,
                "draft_current_standard_GMM_GFLOP_TP8_if_same_expert_dims":q*6*3*PAIR_GFLOP,
                "draft_current_active_expert_layer_pairs":draft_active,
                "draft_selected_packed_weight_set_GB_TP8_if_same_expert_format":draft_active*WEIGHT_BYTES/1e9,
            })
    assert len(rows) == 10
    out = {
        "status":"same_run_per_token_route_conditional_resource_numerator",
        "rows":rows,
        "weight_bytes_per_expert_layer_pair":WEIGHT_BYTES,
        "standard_GFLOP_per_routed_expert_token_pair":PAIR_GFLOP,
        "scope":"Validated current original-route snapshots, Runtime-clipped selected rows. Selected retained set is conditional on unverified slot-major row identity across all 43 Target layers, plus a clairvoyant rejection boundary and ordinary top6 expert algorithm; it is not an executable schedule, compulsory HBM physical traffic or latency floor. Draft same-dimension footprint is conditional on expert format parity. Diagnostic timing excluded.",
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps({"status":out["status"],"cycles":len(rows)}))


if __name__ == "__main__":
    main()
