#!/usr/bin/env python3
"""V3.30: admit Run558 selected Scheduling structure without a finite ceiling."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from extreme_bound_calibration_v3_29 import build as build_prior
from extreme_bound_calibration_v3_16_rev2 import require_null_endpoints


ROOT = Path(__file__).resolve().parents[1]
PRIOR = "evidence/20260927_loop080_bound/run547/bound_calibration_v3_29.json"
REDUCTION = "evidence/20260927_loop080_bound/run558/frontier_reduction.json"
ADMISSION = "evidence/20260927_loop080_bound/run558/final_admission.json"
SERVER = "evidence/20260927_loop080_bound/run558/server_admission.json"
REDUCER = "scripts/loop080_frontier_reduce.py"
HASHES = {
    PRIOR: "b2da7fde8ed8779fdc691984f9df72d7007dc62ef62a82cdd64456d2b8cc1154",
    REDUCTION: "ebfce3c35bc262fd40563df18055577573938f6254e4420551ddcc8ce6ecc792",
    ADMISSION: "37eaaa9b36703dff35eed78e2879c500f004187ed886e3bef749f683f4af7f0f",
    SERVER: "6e24a62ff962ee870e83e148a9aaa6a6259c6ffdc626d5093def8e5de8bdb428",
    REDUCER: "a3a939c5436109a2bdbb3827f32802de12cb6a6361db52bd403c6774855338eb",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def build():
    for name, expected in HASHES.items():
        got = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        require(got == expected, f"pinned input changed: {name}: {got}")
    model = build_prior()
    require(json.dumps(model, ensure_ascii=False, indent=2) + "\n" ==
            (ROOT / PRIOR).read_text(), "V3.29 generator/output mismatch")
    require(model["model_revision"] == "V3.29", "unexpected prior revision")
    reduction = json.loads((ROOT / REDUCTION).read_text())
    admission = json.loads((ROOT / ADMISSION).read_text())
    server = json.loads((ROOT / SERVER).read_text())
    require(reduction["status"] == "selected_frontier_reduced_conditional" and
            admission["status"] == "selected_frontier_diagnostic_integrity_admitted" and
            server["status"] == "server_two_phase_admitted" and
            admission["source_restored"] is True and
            admission["service_stopped"] is True and len(admission["files"]) == 16,
            "Run558 admission/scope changed")
    require(all(value is None for value in reduction["bound_endpoints"].values()) and
            reduction["current_formal_tps"] == 571.681,
            "unproved Run558 endpoint")
    first = reduction["selected_predecessor"]
    brackets = first["rank_local_history_completion_brackets"]
    require(first["slot"] == 5 and
            first["full_1024_runtime_history_crossing_cycle_rank0"] == 188 and
            len(brackets) == 8 and
            {x["rank"] for x in brackets} == set(range(8)) and
            all(x["full_1024_runtime_history_crossing_cycle"] == 188 and
                x["before_cycle"] == 184 and x["after_cycle"] == 192
                for x in brackets), "selected output history changed")
    target = reduction["successor"]["first_target_current_stream_by_rank"]
    seeds = reduction["successor"]["branch_calls_by_rank"]
    require(len(target) == len(seeds) == 8 and
            {x["rank"] for x in target} == set(range(8)) and
            {x["rank"] for x in seeds} == set(range(8)),
            "selected successor all8 incomplete")
    for row in seeds:
        calls = row["matched_calls"]
        require(any(x["input_branch"] == "async_no_common" and
                    x["prompt_upload"] and 83 in x["scheduled"].values()
                    for x in calls), "cached residual path not observed")
        require(any(x["input_branch"] == "async_sample_scatter" and
                    x["draft_scatter_possible"] for x in calls),
                "later async draft branch not observed")
    client = reduction["c12_observed_client"]
    require(client["unique_permit_parent"] is False and
            client["counterfactual_early_release_legal"] is None,
            "client permit relation overpromoted")

    model["model_revision"] = "V3.30"
    model["bound_semantics_v3_30"] = {
        "status": "admitted_selected_frontier_structural_calibration_only",
        "sources": [REDUCTION, ADMISSION, SERVER, REDUCER],
        "current_formal_tps": 571.681,
        "selected_predecessor": {
            "cohort": 5,
            "total_cycles_rank0": 299,
            "first_full_1024_slot": first["slot"],
            "history_crossing_cycle_all8": 188,
            "current_stream_history_marker_cycles": [184, 192],
            "marker_bracket_width_ms_range": [
                min(x["bracket_width_ms"] for x in brackets),
                max(x["bracket_width_ms"] for x in brackets),
            ],
            "scope": "Same-run instrumented Runtime count and current-stream history-copy completion bracket. No early publication, API/client completion, slot/KV retirement, or wall-time saving.",
        },
        "selected_successor": {
            "cohort": 6,
            "observed_first_residual_prefill_tokens": 83,
            "actual_host_branch_sequence": ["async_no_common", "async_sample_scatter"],
            "runtime_step_prepare_current_stream_ms_range": [
                min(x["before_prepare_to_before_target_current_stream_ms"] for x in target),
                max(x["before_prepare_to_before_target_current_stream_ms"] for x in target),
            ],
            "runtime_first_target_to_acceptance_current_stream_ms_range": [
                min(x["before_target_to_after_acceptance_current_stream_ms"] for x in target),
                max(x["before_target_to_after_acceptance_current_stream_ms"] for x in target),
            ],
            "scope": "Actual Host branch and rank-local marked Runtime step; the intervals do not measure the complete earlier ordinary residual-prefill/seed path. Unjoined producers may overlap or cause waits/contention inside them; no private-stream/HCCL/KV completion or all8 makespan is certified.",
        },
        "c12_release": {
            "eligible_parent_set_only": True,
            "unique_permit_parent_certified": False,
            "counterfactual_early_release_legal": None,
        },
        "proof_impact": {
            "closed": ["same-run all8 selected history count crossing", "rank-local sparse current-stream brackets", "successor actual Host branch", "client eligible release sets"],
            "open": ["safe per-slot output publication and state/KV lifetime", "legal actual c12 release while cohort continues", "residual prefill/seed/Target producer joins", "matched unmarked Current control", "all8 mixed resource contention", "strict W-minus and C-plus/B"],
            "algorithm_resource_tps_ceiling": None,
            "hardware_resource_tps_ceiling": None,
            "scheduling_execution_tps_ceiling": None,
            "product_e2e_tps_ceiling": None,
            "distance_from_formal_current_to_credible_limit_tps": None,
        },
    }
    model["bound_ladder"]["scheduling_execution"]["run558_selected_frontier"] = {
        "source": REDUCTION,
        "history_crossing_cycle": 188,
        "cohort_cycles": 299,
        "current_stream_marker_cycles": [184, 192],
        "safe_early_publication": None,
        "legal_successor_release": None,
        "mixed_resource_feasible_schedule": None,
        "scope": "Instrumented structural witness; 111 remaining current cycles are not an attainable saving or necessary Scheduling floor.",
    }
    model["next_measurement"]["priority"] = (
        "First prove or refute one same-generation per-slot normal API completion and actual c12 permit release at a new run's dynamically observed complete crossing while its original cohort continues, retaining its KV/state until safe retirement; verify exact output and matched control. Do not launch overlapping successor NPU work in this feasibility witness. Conditional all8 mixed residual83/seed/Target resource service may be measured independently; Product schedule promotion requires legal release and dependency joins. Independently seek a genuine exact-board C-plus/B certificate and fresh retained W-minus for the declared execution class."
    )
    model["input_paths"].extend(HASHES)
    require_null_endpoints(model)
    prior_semantics = model["bound_semantics_v3_28"]
    host_semantics = model["bound_semantics_v3_29"]
    frontier_semantics = model["bound_semantics_v3_30"]
    unproved = [
        prior_semantics["architecture_class"]["unrestricted_semantic_equivalence"]["positive_measured_window_Target_W_minus"],
        prior_semantics["architecture_class"]["declared_online_inference"]["fresh_required_F_layer_group"],
        prior_semantics["resource_relaxation"]["attainable_C_plus_B"],
        prior_semantics["resource_relaxation"]["finite_architecture_floor_s"],
        prior_semantics["scheduling_execution"]["necessary_path_floor_s"],
        prior_semantics["scheduling_execution"]["feasible_resource_contended_schedule_s"],
        prior_semantics["product_e2e"]["strict_tps_ceiling"],
        prior_semantics["product_e2e"]["conditional_predictive_interval_tps"],
        prior_semantics["product_e2e"]["distance_from_current_to_credible_limit_tps"],
        model["bound_ladder"]["scheduling_execution"]["run537_architecture_variant_review"]["all8_mixed_service_frontier"],
        model["bound_ladder"]["algorithm_resource"]["run538_semantic_witness_limit"]["formal_F_layer_group"],
        *[host_semantics["proof_impact"][key] for key in (
            "algorithm_resource_latency_floor_s", "hardware_resource_latency_floor_s",
            "scheduling_execution_latency_floor_s", "product_e2e_tps_ceiling",
            "distance_from_formal_current_571_681_to_limit_tps")],
        *[model["bound_ladder"]["scheduling_execution"]["run542_546_host_lineage"][key]
          for key in ("cached_residual_device_ready", "all8_mixed_service_frontier",
                      "feasible_alternative_schedule")],
    ]
    unproved.extend(frontier_semantics["proof_impact"][key] for key in (
        "algorithm_resource_tps_ceiling", "hardware_resource_tps_ceiling",
        "scheduling_execution_tps_ceiling", "product_e2e_tps_ceiling",
        "distance_from_formal_current_to_credible_limit_tps"))
    unproved.extend([frontier_semantics["c12_release"]["counterfactual_early_release_legal"],
        *[model["bound_ladder"]["scheduling_execution"]["run558_selected_frontier"][key]
          for key in ("safe_early_publication", "legal_successor_release",
                      "mixed_resource_feasible_schedule")]])
    require(all(value is None for value in unproved),
            "V3.30 unproved endpoint filled without evidence")
    require(not any(model["proof_dag"]["certified"].values()),
            "selected frontier cannot certify strict proof DAG")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    model = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "ok", "revision": "V3.30", "finite_endpoints": 0}))


if __name__ == "__main__":
    main()
