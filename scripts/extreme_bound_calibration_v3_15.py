#!/usr/bin/env python3
"""Fail-closed typed proof DAG for future numeric Bound promotion."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIOR = "evidence/20260927_loop079_identity/run434/bound_calibration_v3_14.json"
REVIEW = "evidence/20260927_loop079_identity/run434/astra_review.md"
CONTRACT = "DeepSeek V4 W4A8; 8x910B3 DP1TP8 DSpark7; warm-cache 48x32K->1024 c12; 49152 externally counted output tokens"


def obligation(kind, status, requires=(), alternatives=(), evidence=(), missing=()):
    return {"kind": kind, "evidence_status": status, "requires_all": list(requires),
            "requires_any": list(alternatives), "evidence": list(evidence),
            "missing": list(missing), "contract": CONTRACT}


def validate_dag(nodes):
    seen = set()
    active = set()

    def visit(name):
        if name not in nodes:
            raise ValueError(f"unknown proof dependency: {name}")
        if name in active:
            raise ValueError(f"proof dependency cycle: {name}")
        if name in seen:
            return
        active.add(name)
        n = nodes[name]
        if n["contract"] != CONTRACT:
            raise ValueError(f"scope mismatch: {name}")
        for dep in n["requires_all"] + n["requires_any"]:
            visit(dep)
        active.remove(name)
        seen.add(name)

    for name in nodes:
        visit(name)
    resolved = {}

    def certified(name):
        if name in resolved:
            return resolved[name]
        n = nodes[name]
        result = (n["evidence_status"] == "certified"
                  and all(certified(dep) for dep in n["requires_all"])
                  and (not n["requires_any"] or any(certified(dep) for dep in n["requires_any"])))
        resolved[name] = result
        return result

    for name in nodes:
        certified(name)
    return resolved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = json.loads((ROOT / PRIOR).read_text())
    review = (ROOT / REVIEW).read_text()
    assert "human-reviewed obligation ledger" in review
    model["next_measurement"]["decision_rule"] = (
        "Strict outer TPS ceiling: positive unavoidable in-window W_minus and a matching genuine aggregate "
        "C_plus upper cap under the same legal algorithm/output/time scope suffice, with N=49152. "
        "A Scheduling lower-latency endpoint additionally needs necessary dependency edges and valid "
        "node-duration lower bounds, not full feasible overlap. Engineering/Aggressive attainable "
        "performance needs a legal resource/storage-feasible implementation and repeated unchanged-contract "
        "formal E2E; measured mixed service supports feasibility but is not C_plus.")
    model["certificate_graph"]["kind"] = "human_reviewed_obligation_ledger"
    graph_node = model["certificate_graph"]["algorithm_resource"]["all43_current_graph_row_identity"]
    graph_node["missing"].append("exact graph all-slot sensitivity; Run433 payload/tolerance does not inherit Run432 k mutation check")
    scheduling = model["certificate_graph"]["scheduling_execution"]["typed_all8_critical_path"]
    scheduling["missing"].extend([
        "per-node duration class: observed Current versus proven lower service versus attainable estimate",
        "common clock domain and uncertainty envelope",
        "measured timed-window arrival/handoff/publication boundary and preexisting reusable work",
    ])
    product = model["certificate_graph"]["product_e2e"]
    product["strict_outer_ceiling"]["scope_checks"] = [
        "same legal algorithm and correctness/output contract",
        "same 49152-output timed window and no pre-window reusable work in W_minus",
        "matching precision/sparsity/operation units and all permitted engines in C_plus",
        "positive finite W_minus and C_plus with conservative uncertainty",
    ]
    product["scheduling_lower_latency"] = {
        "endpoint_type": "strict_relaxed_latency_floor_s", "numeric_s": None,
        "requires": ["necessary typed dependency path", "valid task-duration lower bounds",
                     "same-window timing/clock scope"],
    }
    product["scheduling_aware_interval"]["endpoint_type"] = "attainable_implementation_estimate"
    product["engineering_achievable_interval"]["endpoint_type"] = "repeated_formal_feasibility_plus_predictive_model"

    nodes = {
        "current_row_native_sample": obligation("observation", "sampled", evidence=[
            "evidence/20260927_loop079_identity/run432/result.json",
            "evidence/20260927_loop079_identity/run433/result.json"]),
        "all43_current_graph_identity": obligation("work_numerator_candidate", "open",
            evidence=["evidence/20260927_loop079_identity/run432/result.json"], missing=[
                "all43 selected branch/kwargs/group binding and remaining native contracts"]),
        "handoff_device_freshness": obligation("work_numerator_candidate", "open",
            missing=["D_i(H_run), same-window freshness and reusable output lineage"]),
        "retained_route_W_candidate": obligation("work_numerator_candidate", "open",
            requires=["all43_current_graph_identity", "handoff_device_freshness"],
            missing=["algorithm-class compulsory membership; no clairvoyant rejection shortcut"]),
        "one_projection_W_candidate": obligation("work_numerator_candidate", "open",
            missing=["fresh required BF16 projection with consumer and in-window witness"]),
        "positive_unavoidable_W_minus": obligation("strict_work", "open",
            alternatives=["retained_route_W_candidate", "one_projection_W_candidate"],
            missing=["positive minimum under declared legal implementation class"]),
        "matching_true_C_plus": obligation("strict_capacity_upper", "open",
            missing=["910B3 board/bin max clock+issue/boost or authoritative matching upper cap"]),
        "strict_scope_units_join": obligation("scope_check", "open",
            missing=["same class, 49152-output timed interval, precision, sparsity and aggregate engines"]),
        "strict_outer_tps_ceiling": obligation("strict_product_bound", "open", requires=[
            "positive_unavoidable_W_minus", "matching_true_C_plus", "strict_scope_units_join"]),
        "typed_necessary_path": obligation("scheduling_lower", "open",
            missing=["all8 Graph/HCCL/Draft/KV/Host dependency edges and storage lifetimes"]),
        "necessary_node_duration_lower": obligation("scheduling_lower", "open",
            missing=["task service lower durations with clock/error and resource class"]),
        "scheduling_window_join": obligation("scope_check", "open",
            missing=["same Product arrival/handoff/publication and pre-window work ownership"]),
        "scheduling_relaxed_latency_floor": obligation("strict_scheduling_bound", "open", requires=[
            "typed_necessary_path", "necessary_node_duration_lower", "scheduling_window_join"]),
        "legal_resource_storage_schedule": obligation("attainable_design", "open",
            missing=["resource-capacity contention, overlap, storage and all-rank feasibility"]),
        "attainable_node_service": obligation("attainable_design", "open",
            missing=["matching fixed-shape FULL Graph compute/HBM/HCCL under concurrent load"]),
        "formal_correct_repeats": obligation("attainable_measurement", "open",
            missing=["correct unchanged-contract repeated formal E2E for the proposed schedule"]),
        "engineering_attainable_endpoint": obligation("attainable_product", "open", requires=[
            "legal_resource_storage_schedule", "attainable_node_service", "formal_correct_repeats"]),
    }
    resolved = validate_dag(nodes)
    if (resolved["strict_outer_tps_ceiling"] or resolved["scheduling_relaxed_latency_floor"]
            or resolved["engineering_attainable_endpoint"]):
        raise ValueError("unexpected numeric Bound certificate from open obligations")
    if model["bound_ladder"]["product_e2e"]["finite_tps_upper_bound"] is not None:
        raise ValueError("numeric Product ceiling cannot be inherited without proof DAG certificate")
    model["proof_dag"] = {"schema": 1, "contract": CONTRACT,
                          "timed_external_output_count": 49152,
                          "nodes": nodes, "certified": resolved,
                          "promotion_rule": "Numeric endpoint forbidden unless corresponding node certifies and matching positive finite values/units are independently checked."}
    model["next_measurement"]["priority"] = (
        "For the current retained-route W candidate, close actual all43 graph/branch/kwargs row identity; "
        "a separate one-projection W candidate can yield a looser strict ceiling without all43. "
        "In parallel bind matching true C_plus. Scheduling duration/edge evidence and legal overlap "
        "must keep strict lower bounds separate from attainable estimates.")
    model["input_paths"].extend([PRIOR, REVIEW])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "ok", "dag_nodes": len(nodes),
                      "strict_certified": resolved["strict_outer_tps_ceiling"],
                      "scheduling_certified": resolved["scheduling_relaxed_latency_floor"],
                      "engineering_certified": resolved["engineering_attainable_endpoint"]}))


if __name__ == "__main__":
    main()
