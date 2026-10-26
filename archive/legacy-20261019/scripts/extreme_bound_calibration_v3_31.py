#!/usr/bin/env python3
"""V3.31: active fixed-DSpark7-work execution-time Bound semantics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from extreme_bound_calibration_v3_30 import build as build_prior
from extreme_bound_calibration_v3_16_rev2 import require_null_endpoints


ROOT = Path(__file__).resolve().parents[1]
PRIOR = "evidence/20260927_loop080_bound/run559/bound_calibration_v3_30.json"
PRIOR_SHA = "3957eeaf32dada2c58619dc4f25736b7f03571acad738751aaca32aadd2092ee"


def need(ok, message):
    if not ok:
        raise ValueError(message)


def build():
    need(hashlib.sha256((ROOT / PRIOR).read_bytes()).hexdigest() == PRIOR_SHA,
         "V3.30 output pin changed")
    model = build_prior()
    need(json.dumps(model, ensure_ascii=False, indent=2) + "\n" ==
         (ROOT / PRIOR).read_text(), "V3.30 generator/output mismatch")
    need(model["model_revision"] == "V3.30" and
         model["current"]["accepted_formal_tps"] == 571.681,
         "prior revision or formal Current changed")

    model["model_revision"] = "V3.31"
    model["bound_semantics_v3_31"] = {
        "active_scope": "fixed_DSpark7_algorithm_acceptance_output_and_model_work_execution_time",
        "trajectory_ledger_W0": {
            "formal_product_W0": None,
            "required_identity": [
                "exact request IDs and initial model/KV state",
                "same logical Target and DSpark evaluations, semantic inputs and model work",
                "per-slot per-cycle acceptance and count arrays",
                "retained external token IDs and final output semantics",
                "KV/state transitions and collective payload dependencies",
            ],
            "current_physical_call_shape_layout_census": "calibration observation, not a future launch/layout invariant",
            "run558_diagnostic_trajectory": "admitted full48+48 raw output/count/release ledger, with sparse frontier events only for measured cohort5/6",
            "run558_to_run99_formal_trajectory_transfer": False,
            "scope": "A fixed-work conditional Bound must bind one specific trace or a declared workload distribution. Run558's diagnostic acceptance cannot silently replace the historical Run99 formal trace.",
        },
        "resource_hardware": {
            "workload": "compulsory computation, HBM and communication for the same W0 under declared legal execution class; current materialization and repeated reads are not assumed necessary",
            "latency_floor_formula": "max_k max(0, W0_minus[k] - B[k]) / C_plus[k]",
            "formal_W0_compulsory_vector": None,
            "exact_board_cumulative_C_plus_B": None,
            "fixed_work_latency_floor_s": None,
            "scope": "Capacity must upper-bound cumulative service for the matching 8x910B3 class; attained microbenchmark rate is calibration, not strict C_plus.",
        },
        "scheduling_execution": {
            "legal_change": ["Graph/replay boundaries", "fusion, layout and materialization",
                             "asynchrony and overlap", "pipeline and space-for-time",
                             "persistent/device-resident orchestration",
                             "publication timing with state/KV ownership"],
            "invariants": ["same W0 Target/DSpark work", "same acceptance/count trajectory",
                           "same output semantics", "data and resource dependencies"],
            "fixed_work_dependency_DAG": None,
            "attainable_all8_mixed_service": None,
            "feasible_fixed_work_schedule_s": None,
            "scope": "Minimize makespan for fixed W0; historical fewer-cycle relaxations do not reduce W0 in this active class.",
        },
        "product_e2e": {
            "closed_loop_dependency": "completion -> actual client release -> new arrival -> admission -> residual prefill/seed/KV -> Target/DSpark -> output drain",
            "candidate_arrivals_must_be_regenerated": True,
            "legal_per_slot_early_publication": None,
            "fixed_work_product_latency_floor_s": None,
            "fixed_work_product_tps_interval": None,
            "distance_from_formal_current_to_credible_limit_tps": None,
        },
        "reference_only_not_active_optimization_target": [
            "speculative acceptance improvement or altered draft/target algorithm",
            "perfect-hit 512-cycle cardinality floor",
            "fixed-duration 1011/1015 and zero-cost FIFO 1118/1122 conditional cycle scenarios",
        ],
        "next_bound_test": "Use Run343 fence, Run344 ghost-publication protocol and uninstalled Run345 segmented-serving draft to test one dynamically completed slot's same-W0 normal API completion and actual c12 release while original cohort/KV remain. Compare all8 per-cycle acceptance/count, same logical Target/DSpark evaluations, outputs and state/KV to matched control; verify Graph/storage ownership locally for this ghost witness. Do not constrain future fusion/layout/Graph implementations to current physical calls. No successor NPU overlap in this legality witness. Measure conditional mixed all8 service separately.",
    }
    model["bound_ladder"]["scheduling_execution"]["active_fixed_work_scope_v3_31"] = {
        "W0_formal_trace_bound": None,
        "run558_structural_prior_only": True,
        "acceptance_or_cycle_reduction_as_objective": False,
        "fixed_work_feasible_schedule_s": None,
    }
    model["next_measurement"]["priority"] = model["bound_semantics_v3_31"]["next_bound_test"]
    model["input_paths"].append(PRIOR)
    require_null_endpoints(model)
    v = model["bound_semantics_v3_31"]
    expected_null = [
        v["trajectory_ledger_W0"]["formal_product_W0"],
        *[v["resource_hardware"][k] for k in (
            "formal_W0_compulsory_vector", "exact_board_cumulative_C_plus_B",
            "fixed_work_latency_floor_s")],
        *[v["scheduling_execution"][k] for k in (
            "fixed_work_dependency_DAG", "attainable_all8_mixed_service",
            "feasible_fixed_work_schedule_s")],
        *[v["product_e2e"][k] for k in (
            "legal_per_slot_early_publication", "fixed_work_product_latency_floor_s",
            "fixed_work_product_tps_interval",
            "distance_from_formal_current_to_credible_limit_tps")],
        model["bound_ladder"]["scheduling_execution"]["active_fixed_work_scope_v3_31"]["W0_formal_trace_bound"],
        model["bound_ladder"]["scheduling_execution"]["active_fixed_work_scope_v3_31"]["fixed_work_feasible_schedule_s"],
    ]
    need(all(value is None for value in expected_null), "V3.31 unproved endpoint filled")
    need(not any(model["proof_dag"]["certified"].values()),
         "fixed-work scope alone cannot certify a Bound")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    model = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "ok", "revision": "V3.31", "finite_endpoints": 0}))


if __name__ == "__main__":
    main()
