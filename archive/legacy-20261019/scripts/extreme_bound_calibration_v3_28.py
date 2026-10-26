#!/usr/bin/env python3
"""V3.28: separate strict resource/schedule proofs from empirical costed schedules."""
import argparse
import hashlib
import json
from pathlib import Path

from extreme_bound_calibration_v3_27 import build as build_prior
from extreme_bound_calibration_v3_16_rev2 import require_null_endpoints

ROOT = Path(__file__).resolve().parents[1]
PRIOR = "evidence/20260927_loop079_identity/run530/bound_calibration_v3_27.json"
ASTRA = "evidence/20260927_loop080_bound/run537/astra_bound_review.md"
WITNESS = "evidence/20260927_loop079_identity/run538/target_semantic_witness_feasibility.md"
HASHES = {
    PRIOR: "9803af4ae2e312a2a0962a564826ca5a3a7801bb5bb2832bc3f8f48a94366adf",
    ASTRA: "b3c9cc97244e277c6f5a677f6d4eb74e13e2d48eba4027da29f007251dcf4495",
    WITNESS: "551e44873ea2dd4d9dee47b1c4fdc04d9497acce1fe24dfac3715bf038fddab7",
}


def build():
    for path, expected in HASHES.items():
        actual = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"pinned input changed: {path}: {actual}")
    model = build_prior()
    if json.dumps(model, ensure_ascii=False, indent=2) + "\n" != (ROOT / PRIOR).read_text():
        raise ValueError("V3.27 generator/output mismatch")
    if model["model_revision"] != "V3.27":
        raise ValueError("unexpected prior revision")
    model["model_revision"] = "V3.28"
    model["bound_semantics_v3_28"] = {
        "status": "formalism_recorded_endpoints_unproved",
        "frozen_product_output_tokens": 49152,
        "architecture_class": {
            "unrestricted_semantic_equivalence": {
                "pre_window_full_output_reuse_policy": None,
                "positive_measured_window_Target_W_minus": None,
                "scope": "Observed Current Target calls do not establish unavoidable fresh work when warmup completed the same inputs to1024.",
            },
            "declared_online_inference": {
                "full_output_memoization_excluded": True,
                "partial_intermediate_reuse_policy": None,
                "fresh_required_F_layer_group": None,
                "scope": "Useful conditional engineering class; must remain separately labeled from unrestricted Product contract.",
            },
        },
        "resource_relaxation": {
            "legal_variant": "v includes representation, placement, recomputation, storage/reuse, speculation and collective schedule",
            "capacity_certificate": "service_k(T) <= C_plus_k*T+B_k with exact-board, same-scope evidence",
            "load_floor": "max(0,W_minus_k(v)-B_k)/C_plus_k",
            "variant_floor": "L(v)=max(valid_resource_load_floors,proved_necessary_path_floors,release_completion_floors)",
            "architecture_floor": "L_A=inf_{v in legal class A} L(v), or a common relaxation valid for every v",
            "combine_rule": "Do not sum independent resource/path floors or minima from incompatible variants; maintain compute/traffic/communication/storage Pareto variants.",
            "attainable_C_plus_B": None,
            "finite_architecture_floor_s": None,
        },
        "scheduling_execution": {
            "required_input": "semantic/resource DAG with proved data and lifetime edges, rank joins, resource sharing and peak live storage",
            "c12_release_rule": "successor admission is endogenous to previous external completion plus client overhead; do not freeze Current submit times",
            "current_19_node_ledger_role": "certification workflow, not an executable semantic/resource DAG",
            "conditional_cycle_figures": [512, 1011, 1015, 1118, 1122],
            "conditional_cycle_figures_are_product_time_floors": False,
            "necessary_path_floor_s": None,
            "feasible_resource_contended_schedule_s": None,
            "missing": ["same-trajectory preparation/cache/seed/KV device readiness", "fresh semantic Target key and legal reuse", "all8 mixed compute/HBM/HCCL service and contention", "publication/client drain and endogenous release"],
        },
        "product_e2e": {
            "strict_tps_ceiling": None,
            "conditional_predictive_interval_tps": None,
            "formal_current_tps": 571.681,
            "distance_from_current_to_credible_limit_tps": None,
            "interval_rule": "A proved positive full Product floor L gives TPS ceiling49152/L; a costed feasible schedule and measured E2E give a separately labeled achieved/predictive point.",
        },
        "next_acquisition_scope": "Run529-style 48+48 Host lineage closes output accounting and release graph only; follow with selected all8 mixed readiness/service frontier and strict C_plus/B work in parallel.",
        "sources": [ASTRA, WITNESS],
    }
    model["bound_ladder"]["scheduling_execution"]["run537_architecture_variant_review"] = {
        "source": ASTRA,
        "status": "review_pass_formalism_open_measurement",
        "endogenous_c12_successor_release": True,
        "current_graph_boundaries_compulsory": False,
        "incompatible_variant_minima_may_be_added": False,
        "all8_mixed_service_frontier": None,
    }
    model["bound_ladder"]["algorithm_resource"]["run538_semantic_witness_limit"] = {
        "source": WITNESS,
        "first_retained_argmax_source_conditional": True,
        "in_window_fresh_required_Target_row_certified": False,
        "formal_F_layer_group": None,
    }
    model["next_measurement"]["priority"] = (
        "Admit one same-run warmup48+measured48 Host/client output and release ledger, labeled diagnostic. "
        "Acquire missing cache/prefill/seed/KV device-ready and selected fresh semantic Target keys next, "
        "then choose one all8 resource-contended readiness/service frontier; pursue strict exact-board C-plus/B in parallel."
    )
    model["input_paths"].extend(HASHES)
    require_null_endpoints(model)
    v28 = model["bound_semantics_v3_28"]
    new_unproved = [
        v28["architecture_class"]["unrestricted_semantic_equivalence"]["positive_measured_window_Target_W_minus"],
        v28["architecture_class"]["declared_online_inference"]["fresh_required_F_layer_group"],
        v28["resource_relaxation"]["attainable_C_plus_B"],
        v28["resource_relaxation"]["finite_architecture_floor_s"],
        v28["scheduling_execution"]["necessary_path_floor_s"],
        v28["scheduling_execution"]["feasible_resource_contended_schedule_s"],
        v28["product_e2e"]["strict_tps_ceiling"],
        v28["product_e2e"]["conditional_predictive_interval_tps"],
        v28["product_e2e"]["distance_from_current_to_credible_limit_tps"],
        model["bound_ladder"]["scheduling_execution"]["run537_architecture_variant_review"]["all8_mixed_service_frontier"],
        model["bound_ladder"]["algorithm_resource"]["run538_semantic_witness_limit"]["formal_F_layer_group"],
    ]
    if any(value is not None for value in new_unproved):
        raise ValueError("V3.28 unproved endpoint filled without new evidence")
    if any(model["proof_dag"]["certified"].values()):
        raise ValueError("review cannot certify a finite bound")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    model = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "ok", "revision": "V3.28", "finite_endpoints": 0}))


if __name__ == "__main__":
    main()
