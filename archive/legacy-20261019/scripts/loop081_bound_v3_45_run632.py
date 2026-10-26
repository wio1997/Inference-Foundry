#!/usr/bin/env python3
"""Run632: promote Run628–631 scope-controlled Product accounting to V3.45.

Numerical Resource/Scheduling/Product bounds stay null. This revision adds
evidence coverage and fixed-output saving hurdles, not a synthetic ceiling.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path("/data/wio/Inference_Foundry")
INPUTS = {
    "v3_44": ("evidence/20260928_loop081_bound/run625/bound_calibration_v3_44.json", "09900472c880c86695b6c6ad8eaae14145106af8001bd1ba3149ebe66034b180"),
    "sentinel": ("evidence/20260928_loop081_bound/run628/astra_sentinel_review.md", "3a97b243e2716e0600ff38b80606aa54a07a625dc59b36e57d42bfea198b6eb8"),
    "priority": ("evidence/20260928_loop081_bound/run629/astra_bound_priority_review.md", "db604ade2264e8dbba6e05f61fcd3e5a2e1f0d7d2b4c51731021c017ec150da6"),
    "hurdle": ("evidence/20260928_loop081_bound/run630/product_hurdle.json", "63c8e7690136c3a74f4471b2cadd2d33f5d5cd4fb47e8ffafcb159cdb2ec7620"),
    "hurdle_review": ("evidence/20260928_loop081_bound/run630/astra_product_hurdle_review.md", "4f80ae9d957289930c93796b394823392de2a6130f298ccbbb2e4a0018926fb0"),
    "coverage": ("evidence/20260928_loop081_bound/run631/product_coverage.json", "ceafd884c8df096bd07e80d9f37a13a378879ac93b9c95b29c87981c261abee2"),
    "coverage_review": ("evidence/20260928_loop081_bound/run631/astra_product_coverage_review.md", "b28b8121571bb505f96096bd51257d6caf5ec9fd8925431938d1a9aa9ddc7142"),
}
OUT = ROOT / "evidence/20260928_loop081_bound/run632/bound_calibration_v3_45.json"


def read(name: str) -> dict:
    path, digest = INPUTS[name]
    data = (ROOT / path).read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise AssertionError(f"source drift {name}")
    if path.endswith(".json"):
        return json.loads(data)
    return {}


def validate_no_bound(model: dict) -> None:
    """Reject numerical endpoint promotion anywhere in the inherited model."""
    endpoint_keys = {
        "floor_s", "ceiling_tps", "finite_tps_upper_bound",
        "unresolved_gap_tps", "numeric_current_to_credible_limit_gap",
        "conditional_engineering_interval_tps", "product_e2e_tps_interval",
        "strict_outer_tps_ceiling_value",
        "matching_C_plus_or_B", "maximum_capacity_bound",
        "all_legal_engine_capacity_upper", "exact_board_aggregate_capacity_upper",
        "exact_board_C_plus", "attainable_C_plus_B",
        "exact_board_cumulative_C_plus_B", "certified_exact_board_C_plus_B",
        "exact_board_certified_cumulative_C_plus_B",
        "certified_exact_board_cumulative_C_plus_B",
    }

    def walk(value, path=""):
        if isinstance(value, dict):
            if path == "proof_dag.certified":
                return  # Certificate flags are checked explicitly below.
            for key, child in value.items():
                where = f"{path}.{key}" if path else key
                numerical_endpoint = path != "proof_dag.nodes" and (
                    key in endpoint_keys or key.endswith("_floor_s")
                    or key.endswith("_floor_ms")
                    or key.endswith("_ceiling_tps")
                    or key.endswith("_tps_ceiling")
                    or key.endswith("_tps_upper_bound")
                    or key.endswith("_tps_interval")
                    or key.endswith("_interval_tps")
                )
                if numerical_endpoint and child is not None:
                    raise AssertionError(f"unsupported endpoint {where}")
                walk(child, where)
        elif isinstance(value, list):
            for i, child in enumerate(value):
                walk(child, f"{path}[{i}]")

    walk(model)
    flags = model["proof_dag"]["certified"]
    if any(v is not False for v in flags.values()):
        raise AssertionError("unproved proof DAG certificate")
    if model["current"]["accepted_formal_tps"] != 571.681:
        raise AssertionError("Current drift")


def main() -> None:
    model = read("v3_44")
    validate_no_bound(model)
    for name in ("sentinel", "priority", "hurdle_review", "coverage_review"):
        read(name)
    hurdle = read("hurdle")
    coverage = read("coverage")
    assert model["current"]["accepted_formal_tps"] == hurdle["formal_current_tps"] == 571.681
    assert hurdle["paired_median_repeat"] == 3 and hurdle["paired_median_cycles"] == 1206
    assert [r["run"] for r in coverage["runs"]] == [602, 606]
    assert all(r["product_output_ids"] == 49152 for r in coverage["runs"])
    assert all(v is None for v in hurdle["bound_endpoints"].values())
    assert all(coverage[k] is None for k in (
        "strict_resource_floor_s", "strict_scheduling_floor_s",
        "strict_product_tps_ceiling", "numeric_current_to_credible_limit_gap"))

    model["model_revision"] = "v3.45_fixed_work_product_coverage_and_sensitivity_no_strict_bound"
    model["bound_semantics_v3_45"] = {
        "active_objective": "same DSpark7 acceptance/cycles/output/model work; minimize full Product execution time on 8x910B3 DP1TP8",
        "source_pins": {name: {"path": path, "sha256": digest} for name, (path, digest) in INPUTS.items()},
        "formal_current": {
            "tps": 571.681,
            "paired_repeat_scope": hurdle["paired_repeats"],
            "paired_median_repeat": hurdle["paired_median_repeat"],
            "median_exact_client_wall_s": hurdle["paired_median_exact_client_wall_s"],
            "paired_median_cycles": hurdle["paired_median_cycles"],
            "run99_scope_rule": "T-R is paired arithmetic only: no saved absolute server interval/request-ID join, no disjoint stage complement or removable Host claim",
            "sample_range": hurdle["sample_range"],
        },
        "fixed_output_net_saving_hurdles": hurdle["scenario_landmark_hurdles"],
        "hurdle_status": "counterfactual required E2E-exposed wall saving; four scenario landmarks are not ceilings, achieved savings, or stopping targets",
        "diagnostic_product_coverage": [
            {
                "run": r["run"],
                "client_outer_clock_span_s": r["client_outer_clock_span_s"],
                "client_measured_wall_monotonic_span_s": r["client_measured_wall_monotonic_span_s"],
                "cohort_first_recorded_to_first_own_s": [
                    c["host_marker_durations_s"]["first_recorded_to_first_own"]
                    for c in r["cohorts"]],
                "cohort_execute_label_counts": [c["execute_record_label_presence_all8"] for c in r["cohorts"]],
                "host_envelope_union": {
                    k: r["union_coverage"][k] for k in (
                        "covered_by_any_host_marker_envelope_ns",
                        "uncovered_by_these_envelopes_ns",
                        "multiple_envelopes_active_ns")},
            } for r in coverage["runs"]
        ],
        "diagnostic_scope_rule": "Run602/606 are separate observer-perturbed W0s. Cohort6-8 files start with previous-cohort execute markers; wide Host envelope union is not compute/device occupancy, necessary work, an attainable saving, or a Run99 timing transfer.",
        "resource_hardware": {
            "necessary_full_work_and_traffic": None,
            "certified_exact_board_cumulative_C_plus_B": None,
            "strict_floor_s": None,
            "attained_mixed_service_rule": "Run579/580 joint GMM+HCCL service shows competition; max(isolated GMM,HCCL) is not demonstrated. Fixture service cannot replace universal capacity or production compulsory work.",
        },
        "scheduling_execution": {
            "strict_floor_s": None,
            "source_partial_dag": "Run625 82 nodes/106 edges untimed; 18 metadata ancestry roots unbound",
            "typed_kv_packet_priority": "conditional supporting branch, selected-row lineage cannot yet rank global exposed saving",
            "sentinel": "Run628 conditional source no-row-write for [-1,31]; actual loaded path remains UNKNOWN",
            "unknown": "same-W0 all8 device producer completion, existing wait/consumer readiness, cross-cycle tails, resource contention and observer-off timing exposure",
        },
        "product_e2e": {
            "strict_ceiling_tps": None,
            "conditional_engineering_interval_tps": None,
            "numeric_current_to_credible_limit_gap": None,
            "next_action": "Use current same-W0 offline Product coverage to specify the smallest low-overhead all8/client phase completion packet with causal device-ready markers and OFF/ON/OFF transfer; rank it against mixed-service Resource uncertainty before live run. Keep formal repeated E2E as intervention judge.",
        },
        "independent_review": "Astra High Run629 PIVOT and Run630/631 SCOPED PASS; V3.45 independent review still required",
    }
    validate_no_bound(model)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(model, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"revision": model["model_revision"], "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
