#!/usr/bin/env python3
"""Admit Run477's no-getter local observation without promoting a Bound."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from extreme_bound_calibration_v3_16_rev2 import require_null_endpoints

ROOT = Path(__file__).resolve().parents[1]
PRIOR = "evidence/20260927_loop079_identity/run471/bound_calibration_v3_19.json"
INTERVALS = "evidence/20260927_loop079_identity/run477/intervals.json"
VALIDATION = "evidence/20260927_loop079_identity/run472/b_clean2/validation.json"
FINAL = "evidence/20260927_loop079_identity/run472/b_clean2/final_admission.json"
REVIEW = "evidence/20260927_loop079_identity/run480/astra_clean_b_review.md"
HARDWARE = "evidence/20260927_loop079_identity/run479/astra_minimal_compulsory_review.md"
HASHES = {
    PRIOR: "c87b8fea01c5a742d977a5241bd8d29df7f2658731fc7f5c20cc346711bea829",
    INTERVALS: "f0731bedfd3385442c7ae9cee33af40294b9866374d3fefa5b32bdd14f0ca68a",
    VALIDATION: "f2cac0d8b4853810d4c542ab234b8a0f6c00e164ae1b70914e0ab672fad0995d",
    FINAL: "356de5f539cd53090e26a35809fb49d5080fda3541f9bd15294c682957cb8e1f",
    REVIEW: "b5fef38a78940c2b8ea4f8e381a90d80431725181cca4fcd2e1fe22583f335d0",
    HARDWARE: "d5ff6050ab0679886ecf5676f63d0a883c7c6e28960ef260899c5925679ee985",
}


def build():
    for name, expected in HASHES.items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"evidence hash mismatch: {name}")
    model = json.loads((ROOT / PRIOR).read_text())
    intervals = json.loads((ROOT / INTERVALS).read_text())
    admitted = json.loads((ROOT / VALIDATION).read_text())
    final = json.loads((ROOT / FINAL).read_text())
    review = (ROOT / REVIEW).read_text()
    hardware = (ROOT / HARDWARE).read_text()
    if model["model_revision"] != "V3.19" or model["current"]["accepted_formal_tps"] != 571.681:
        raise ValueError("prior model/current changed")
    if admitted.get("valid") is not True or final.get("valid") is not True:
        raise ValueError("Run477 admission not complete")
    if intervals["all_slices"]["P_C1_ms"]["n"] != 40 or not (0.25 < intervals["all_slices"]["P_C1_ms"]["median_ms"] < 0.30):
        raise ValueError("Run477 interval scale/cardinality changed")
    if "PASS for post-cleanup diagnostic admission" not in review or "No fully certified positive W_minus / matching C_plus pair" not in hardware:
        raise ValueError("independent review scope changed")
    if any(model["proof_dag"]["certified"].values()):
        raise ValueError("unexpected proof certification")

    model["model_revision"] = "V3.20"
    scheduling = model["bound_ladder"]["scheduling_execution"]
    if scheduling["terminal_logits_local_slice"]["status"] != "queue_drained_diagnostic_only":
        raise ValueError("old intervention column missing")
    scheduling["terminal_logits_no_getter_local_slice"] = {
        "status": "admitted_instrumented_local_current_only",
        "source": [INTERVALS, VALIDATION, FINAL, REVIEW],
        "scope": "separate Run477 exact60 diagnostic; selected cycle64, five cohorts x eight ranks; same-device local P/J/G/C0/C1; direct logical stream fields remove known raw getter queue drain",
        "intervals_ms": intervals["all_slices"],
        "cohort_cluster_count": 5,
        "all_rank_common_clock_makespan_ms": None,
        "g_c0_accuracy_floor_resolved": False,
        "matched_A0_B_A1_controls": False,
        "timing_transfer_to_uninstrumented_current": False,
        "necessary_duration_floor_ms": None,
        "removable_e2e_ms": None,
        "old_new_median_difference_causal": False,
        "missing": [
            "unmarked original-schedule exposure and probe-overhead calibration",
            "earlier Target FULL replay/hidden-output producer and existing Host wait critical-path placement",
            "matched reset/trajectory controls before original-schedule gap ranking",
            "legal same-state intervention plus repeated correct formal E2E before savings claim",
        ],
    }
    scheduling["run477_current_trajectory"] = {
        "status": "diagnostic_current_only",
        "cohort_cycles": [341, 315, 305, 307, 293],
        "cohort_useful_tokens": [12288] * 5,
        "scope": "Current fixed cohorts; first four are warmup, fifth is the diagnostic bench; neither formal Run99 trajectory nor a scheduling floor",
        "source": [REVIEW],
    }
    certificate = model["certificate_graph"]["scheduling_execution"]["terminal_logits_join"]
    certificate["status"] = "lineage_admitted_no_getter_timing_instrumented"
    certificate["evidence"].extend([INTERVALS, FINAL, REVIEW])
    certificate["missing"] = list(scheduling["terminal_logits_no_getter_local_slice"]["missing"])
    model["certificate_graph"]["algorithm_resource"]["strict_necessary_work_subset_W_minus"]["evidence"].append(HARDWARE)
    model["certificate_graph"]["hardware_resource"]["matching_true_capacity_upper_C_plus"]["evidence"].append(HARDWARE)
    model["next_measurement"]["priority"] = (
        "Repair Run478's Target FULL replay/hidden-output frontier source and admission gates, then acquire a separate selected-cycle diagnostic. "
        "In parallel, bind one fresh retained wo_a group to the timed external output and seek an authoritative exact-board C_plus certificate; "
        "do not infer a finite Hardware or Scheduling endpoint from measured attained rates or probe intervals."
    )
    model["next_measurement"]["specific_gate"] = (
        "Run481 preflight blockers must close before Target service launch: safe non-sync replay and disabled capture branches, "
        "frozen12-slot remaining relation, full marker/native/tree/Runtime lineage, reviewed controller and post-cleanup final admission. "
        "R1 completion remains conditional on actual captured producer membership."
    )
    model["input_paths"].extend([INTERVALS, VALIDATION, FINAL, REVIEW, HARDWARE])
    require_null_endpoints(model)
    if any(model["proof_dag"]["certified"].values()):
        raise ValueError("diagnostic evidence cannot certify a Bound proof node")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "ok", "revision": "V3.20", "finite_endpoints": 0}))


if __name__ == "__main__":
    main()
