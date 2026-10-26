#!/usr/bin/env python3
"""Scope-preserving Bound ledger update after Run437/440 and HCCL Test.

The generator deliberately cannot emit a finite endpoint. Numeric promotion
needs an independently checked work/capacity or necessary-path payload.
"""
import argparse
import json
from pathlib import Path

from extreme_bound_calibration_v3_15 import validate_dag, obligation

ROOT = Path(__file__).resolve().parents[1]
PRIOR = "evidence/20260927_loop079_identity/run435/bound_calibration_v3_15.json"
VALIDATION = "evidence/20260927_loop079_identity/run437/validation.json"
REVIEW = "evidence/20260927_loop079_identity/run440/astra_run437_review.md"
HCCL = "evidence/20260927_loop079_identity/run441/findings.md"
DECISION = "evidence/20260927_loop079_identity/run438/astra_bound_gap_review.md"
DESIGN = "evidence/20260927_loop079_identity/run439/scheduling_slice_design.md"
NUMERIC_ENDPOINT_KEYS = {"numeric_tps", "numeric_s", "latency_floor_s", "finite_tps_upper_bound"}


def require_null_endpoints(tree, prefix=""):
    if isinstance(tree, dict):
        for key, value in tree.items():
            path = f"{prefix}.{key}" if prefix else key
            if key in NUMERIC_ENDPOINT_KEYS and value is not None:
                raise ValueError(f"uncertified numeric endpoint inherited at {path}")
            require_null_endpoints(value, path)
    elif isinstance(tree, list):
        for index, value in enumerate(tree):
            require_null_endpoints(value, f"{prefix}[{index}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = json.loads((ROOT / PRIOR).read_text())
    v = json.loads((ROOT / VALIDATION).read_text())
    review = (ROOT / REVIEW).read_text()
    hccl = (ROOT / HCCL).read_text()
    decision = (ROOT / DECISION).read_text()
    design = (ROOT / DESIGN).read_text()
    if not (v["diagnostic_valid"] and not v["all43_row_identity_certificate"]
            and len(v["cohorts"]) == 5):
        raise ValueError("Run437 scoped diagnostic gate failed")
    if not all(c["certificates"]["full_graph_input_binding_valid"]
               and c["certificates"]["cp_value_map_valid"]
               and c["certificates"]["all43_row_composition"] == "CONDITIONAL_NATIVE"
               for c in v["cohorts"]):
        raise ValueError("Run437 cohort certificate scope changed")
    for snippet, content in (("ACCEPT Run437", review),
                             ("179.45", hccl),
                             ("one positive", decision),
                             ("native HCCL completion", design)):
        if snippet not in content:
            raise ValueError(f"review/design anchor missing: {snippet}")

    resource = model["certificate_graph"]["algorithm_resource"]
    row = resource["all43_current_graph_row_identity"]
    row["evidence"].extend([VALIDATION, REVIEW])
    row["missing"] = [
        "per-layer selected branch and actual layer-to-CP consumer binding",
        "FlashComm/DSA/embedding/residual/native row composition and graph all-slot semantics",
        "loaded native binary/tiling and GMM/scale row association",
    ]
    row["bound_effect"] = (
        "Run437 closes selected FULL explicit input-storage and CP value-prefix provenance, "
        "but all43 retained-route numerator remains conditional")
    model["certificate_graph"]["hardware_resource"]["isolated_terminal_allgather"] = {
        "status": "attained_isolated_observation", "evidence": [HCCL],
        "payload_per_rank_bytes": 3102720,
        "reported_host_inclusive_mean_us_three_processes": [221.77, 179.45, 205.78],
        "scope": "CANN9.1 HCCL Test bfp16 8 ranks, 10 warmup/30 measured, correctness pass; not mixed Graph or C_plus",
        "bound_effect": "no strict or Product endpoint promotion",
    }
    model["certificate_graph"]["scheduling_execution"]["terminal_logits_join"] = {
        "status": "design_only", "evidence": [DESIGN],
        "missing": ["installed native HCCL-to-consumer-stream completion contract",
                    "original-path P/J/G/C0/C1 correlated events",
                    "same-policy A0-B-A1 overhead controls"],
        "bound_effect": "no Current exposed interval or necessary-path duration floor yet",
    }

    nodes = model["proof_dag"]["nodes"]
    nodes["selected_full_graph_input_binding_sample"] = obligation(
        "observation", "sampled", evidence=[VALIDATION, REVIEW], missing=[
            "implicit contexts and all43 layer/native row composition remain open"])
    nodes["isolated_terminal_allgather_service_sample"] = obligation(
        "observation", "sampled", evidence=[HCCL], missing=[
            "matching mixed contention and original-path consumer join"])
    nodes["all43_current_graph_identity"]["evidence"].extend([VALIDATION, REVIEW])
    nodes["all43_current_graph_identity"]["missing"] = [
        "per-layer actual branch/layer-to-CP binding and native row composition",
        "exact graph all-slot semantics, GMM/scale association and loaded binary provenance",
    ]
    nodes["typed_necessary_path"]["missing"] = [
        "selected necessary dependency path and relevant cross-rank/storage edges; complete all8 graph only if claimed endpoint uses it",
    ]
    nodes["legal_resource_storage_schedule"]["missing"] = [
        "feasible resource/storage schedule with all-rank contention and overlap, independently of a necessary-path relaxation",
    ]
    resolved = validate_dag(nodes)
    if any(resolved.values()):
        raise ValueError("Run437/441 observations cannot certify a Bound proof node")
    model["proof_dag"]["certified"] = resolved
    model["proof_dag"]["scope_note"] = (
        "Run437/441 are sampled observations. The DAG verifies structure/status only; "
        "a future numeric certificate requires evidence truth, units, rank aggregation and uncertainty checks.")
    model["next_measurement"]["priority"] = (
        "For useful Scheduling interval, verify exact installed native logits HCCL join, then acquire one "
        "original-path terminal logits→AllGather→argmax slice with matched A0-B-A1 controls. "
        "For first loose strict outer ceiling, independently certify one fresh required BF16 projection "
        "in the Product window and an exact-board aggregate C_plus. Harvest Run437's first unresolved "
        "layer/native row transform only if retained-route numerator remains the selected Resource path.")
    model["input_paths"].extend([VALIDATION, REVIEW, HCCL, DECISION, DESIGN])
    require_null_endpoints(model["bound_ladder"], "bound_ladder")
    require_null_endpoints(model["certificate_graph"], "certificate_graph")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "ok", "proof_nodes": len(nodes),
                      "finite_endpoints": 0, "run437_scoped": True,
                      "hccl_attained_isolated": True}))


if __name__ == "__main__":
    main()
