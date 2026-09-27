#!/usr/bin/env python3
"""Add Run403 same-trajectory resource numerators without inferring a ceiling."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIOR = "evidence/20260927_loop078_bound/run402/bound_calibration_v3_8.json"
NUM = "evidence/20260927_loop078_bound/run405/resource_numerator.json"
GATE = "evidence/20260927_loop078_bound/run404/gates.json"
VALID = "evidence/20260927_loop078_bound/run404/validation.json"
REVIEW = "evidence/20260927_loop078_bound/run405/astra_review.md"


def load(path):
    return json.loads((ROOT / path).read_text())


def span(rows, *keys):
    vals = [row for row in rows]
    for key in keys:
        vals = [v[key] for v in vals]
    vals.sort()
    return {"min": vals[0], "median": (vals[4] + vals[5]) / 2, "max": vals[-1]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    model, num, gate, valid = [load(path) for path in (PRIOR, NUM, GATE, VALID)]
    review = (ROOT / REVIEW).read_text()
    assert "conditional on the slot-major candidate-row layout" in review
    assert num["status"] == "same_run_per_token_route_conditional_resource_numerator"
    assert len(num["rows"]) == 10
    assert gate["status"] == "clean_frozen_diagnostic_gate" and gate["server_post_count"] == 60 and gate["source_sha_restored"] and valid["valid"]
    assert all(row["target_current_candidate_rows"] == 96 and row["draft_actual_rows"] == 84 for row in num["rows"])
    rows = num["rows"]
    model["bound_ladder"]["algorithm_resource"]["run403_same_trajectory_route_numerator"] = {
        "target_current_standard_GMM_GFLOP_TP8_cycle": span(rows, "target_current_standard_GMM_GFLOP_TP8"),
        "target_clairvoyant_retained_standard_GMM_GFLOP_TP8_cycle": span(rows, "target_clairvoyant_retained_standard_GMM_GFLOP_TP8"),
        "draft_current_standard_GMM_GFLOP_TP8_cycle_if_same_expert_dims": span(rows, "draft_current_standard_GMM_GFLOP_TP8_if_same_expert_dims"),
        "scope": "Current Target arithmetic exact under observed 96 rows/top6/43 layer standard 2MNK equivalent; retained path assumes perfect foresight of rejection, ordinary top6 evaluation, and unproven all-layer slot-major row identity. Draft arithmetic additionally assumes Target expert dimensions. None is full compulsory work.",
        "source": [NUM, GATE, VALID, REVIEW],
    }
    model["bound_ladder"]["hardware_resource"]["run403_same_trajectory_selected_packed_weight_set_GB_TP8_cycle"] = {
        "target_current_active": span(rows, "target_packed_weight_set_GB_TP8", "current_active"),
        "target_histogram_retained_lower": span(rows, "target_packed_weight_set_GB_TP8", "histogram_lower"),
        "target_slot_major_selected_conditional": span(rows, "target_packed_weight_set_GB_TP8", "conditional_slot_major_selected"),
        "target_histogram_retained_upper": span(rows, "target_packed_weight_set_GB_TP8", "histogram_upper"),
        "draft_selected_if_same_expert_format": span(rows, "draft_selected_packed_weight_set_GB_TP8_if_same_expert_format"),
        "scope": "Packed selected storage sets, excluding scales and other tensors. Conditional retained selection requires unproven row identity and clairvoyant rejection; histogram bounds use row-count feasibility. None is compulsory physical HBM bytes, minimum traffic, or latency floor.",
        "source": [NUM, GATE, VALID, REVIEW],
    }
    model["bound_ladder"]["algorithm_resource"]["missing"].append("prove per-layer Target route row identity across FlashComm1, DSA CP, chunk/gather; resolve retained causal closure and full Draft work")
    model["bound_ladder"]["hardware_resource"]["missing"].append("convert route-selected packed storage sets to compulsory physical HBM with scale/cache/residency evidence under concurrency")
    model["next_measurement"]["priority"] = "validate route row identity with source-plus-live labels and close cross-cycle D2H read-before-overwrite; then measure compulsory-vs-current HBM for exact selected routes"
    model["next_measurement"]["specific_gate"] = "Run405 independent review; selected row labels before and after all active FlashComm1/DSA CP transforms; Run401 same-device copy completion versus next overwrite with A0-B-A1"
    model["input_paths"].extend([PRIOR, NUM, GATE, VALID, REVIEW])
    assert model["bound_ladder"]["product_e2e"]["finite_tps_upper_bound"] is None
    assert all(model["bound_ladder"][key]["latency_floor_s"] is None for key in ("algorithm_resource", "hardware_resource", "scheduling_execution"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "ok", "same_trajectory_cycles": len(rows), "finite_product_ceiling": None}))


if __name__ == "__main__":
    main()
