#!/usr/bin/env python3
"""Separate numerical ceilings from tested and missing proof obligations."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIOR = "evidence/20260927_loop079_identity/run429/bound_calibration_v3_13.json"
NATIVE = "evidence/20260927_loop079_identity/run430/astra_native_row_contract_review.md"
EAGER = "evidence/20260927_loop079_identity/run432/result.json"
EAGER_REVIEW = "evidence/20260927_loop079_identity/run432/astra_review.md"
MUTATION = "evidence/20260927_loop079_identity/run432/cpu_mutation_oracle.json"
GRAPH = "evidence/20260927_loop079_identity/run433/result.json"
GRAPH_REVIEW = "evidence/20260927_loop079_identity/run433/astra_review.md"
OEM = "evidence/20260927_loop079_identity/run431/hardware_identity.md"


def read(path):
    return json.loads((ROOT / path).read_text())


def node(status, evidence, missing, effect):
    return {"status": status, "evidence": evidence, "missing": missing,
            "bound_effect": effect}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = read(PRIOR)
    eager, mutation, graph = read(EAGER), read(MUTATION), read(GRAPH)
    assert eager["status"] == graph["status"] == mutation["status"] == "PASS" or (
        eager["status"] == graph["status"] == "pass" and mutation["status"] == "PASS")
    assert len(eager["cases"]) == len(graph["cases"]) == 3
    assert [c["valid_routes"] for c in graph["cases"]] == [85, 69, 85]
    assert mutation["k_swap_numeric_cases"] == 1440
    assert sum(c["same_token_expert_swaps_rejected"] for c in mutation["cases"]) == 64
    assert "CONDITIONAL_NATIVE" in (ROOT / NATIVE).read_text()
    assert "V2 follow-up" in (ROOT / EAGER_REVIEW).read_text()
    assert "SAMPLED_FIXED_STORAGE_GENERATION_REFRESH" in (ROOT / GRAPH_REVIEW).read_text()
    assert "S900K3" in (ROOT / OEM).read_text()

    model["certificate_graph"] = {
        "schema": 1,
        "current_formal": node("measured", ["Run99 formal"], [],
                               "571.681 tok/s median; fastest valid single sample 612.962 is not a repeatability guarantee"),
        "algorithm_resource": {
            "host_commit_consume_yield": node("scoped_closed_run427", [
                "evidence/20260927_loop079_identity/run427/analysis.json",
                "evidence/20260927_loop079_identity/run428/astra_timeline_pre_review.md"],
                ["device-completed D_i(H_run)", "literal token publication", "same-run formal join"],
                "Narrows Host ownership/publication order only; not fresh compulsory work"),
            "moe_native_eager": node("sampled_pass", [EAGER, EAGER_REVIEW, MUTATION],
                ["loaded tiling/object provenance", "all local EP ranges and branches", "quantized scale/GMM pairing"],
                "Tested 4x2 and actual96x6 quant1 partial-EP token/expert gather, abs/mask/unpermute; no full-model W_minus"),
            "all43_current_graph_row_identity": node("open_conditional_native", [NATIVE, GRAPH, GRAPH_REVIEW],
                ["actual FULL entry and keyword tensor binding", "all43 layer branch/group/EP-map evidence",
                 "attention/rotary/GMM/collective native contracts", "target_logits_indices and retained-row join"],
                "Run433 only proves synchronized isolated fixed-storage graph refresh; retained route numerator remains conditional"),
            "strict_necessary_work_subset_W_minus": node("open", [
                "evidence/20260927_loop079_identity/run423/astra_minimal_ceiling_review.md"],
                ["one fresh in-window required subset for stated algorithm class and matching arithmetic units"],
                "No strict numerical Resource/Product ceiling numerator"),
        },
        "hardware_resource": {
            "installed_identity": node("scoped_closed", [OEM],
                ["maximum compute bin/clock/issue certificate", "same-host HBM/HCCL maximum and practical mixed capacity"],
                "Eight910B3 on Wuzhou S900K3 are identified; OEM marketing rates are not C_plus"),
            "matching_true_capacity_upper_C_plus": node("open", [OEM],
                ["work-matched maximum across allowed engines, boost/error envelope and SKU binding"],
                "No valid W_minus/C_plus numeric ceiling"),
            "practical_attainable_mixed_capacity": node("open", [
                "evidence/20260927_loop079_identity/run423/astra_minimal_ceiling_review.md"],
                ["same-shape compute/HBM/HCCL under joint contention and graph schedule"],
                "No Engineering resource endpoint"),
        },
        "scheduling_execution": {
            "typed_all8_critical_path": node("partial_source_and_host_order", [
                "evidence/20260927_loop078_bound/run400/dependency_ledger.json", GRAPH, GRAPH_REVIEW],
                ["device Graph/HCCL stream completion and all-rank joins", "actual Draft consumers and KV/state reuse",
                 "resource-constrained overlap feasibility"],
                "No numeric scheduling floor or removable Current gap"),
        },
        "product_e2e": {
            "strict_outer_ceiling": {"requires": ["strict_necessary_work_subset_W_minus",
                "matching_true_capacity_upper_C_plus"], "numeric_tps": None},
            "engineering_achievable_interval": {"requires": ["complete legal execution variant",
                "practical_attainable_mixed_capacity", "repeated frozen formal E2E"], "numeric_tps": None},
            "scheduling_aware_interval": {"requires": ["typed_all8_critical_path",
                "resource-constrained overlap feasibility", "formal calibration"], "numeric_tps": None},
        },
    }
    model["bound_ladder"]["algorithm_resource"]["run430_433_native_scope"] = {
        "eager_tested": "sampled token/expert gather and abs/masked unpermute for 4x2 and actual96x6 partial EP; 64 expert and1440 k pair mutation checks pass",
        "graph_tested": "single isolated96x6 partial-EP synchronized fixed-storage three-generation replay, valid routes85→69→85",
        "not_proved": "production all43 selected FULL Graph row identity, loaded native binary/tiling, exact graph all-slot sensitivity, full model compulsory work",
        "source": [NATIVE, EAGER, EAGER_REVIEW, MUTATION, GRAPH, GRAPH_REVIEW],
    }
    model["bound_ladder"]["hardware_resource"]["run431_installed_platform"] = {
        "host": "Wuzhou S900K3", "card": "IT21HMDC_Bin6", "chip": "8x910B3",
        "configured_AI_Core_MHz": 1800, "AI_Core_count_per_chip": 20,
        "maximum_capacity_bound": None, "source": OEM,
    }
    model["next_measurement"]["priority"] = (
        "First join actual all43 FULL Graph branch/kwargs/group/CP/logits-index identity to the now tested "
        "partial-EP native mapping; in parallel seek board-matched maximum C_plus. Avoid repeating Host "
        "ledger or interpreting isolated graph pass as Product bound.")
    model["next_measurement"]["specific_gate"] = (
        "A reversible Host metadata plus selected-cycle device snapshot of actual graph entry, keyword "
        "input storage, all43 branch/rank maps, target_logits_indices and CP bindings; exact60 "
        "frozen diagnostic and independent row-composition validator. If a native boundary remains opaque, "
        "name it and test that one boundary with sentinels.")
    model["input_paths"].extend([PRIOR, NATIVE, EAGER, EAGER_REVIEW, MUTATION,
                                 GRAPH, GRAPH_REVIEW, OEM])
    assert model["bound_ladder"]["product_e2e"]["finite_tps_upper_bound"] is None
    assert all(model["bound_ladder"][k]["latency_floor_s"] is None for k in
               ("algorithm_resource", "hardware_resource", "scheduling_execution"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "ok", "current_tps": 571.681,
                      "finite_product_ceiling": None,
                      "next_gate": "actual_all43_graph_identity"}))


if __name__ == "__main__":
    main()
