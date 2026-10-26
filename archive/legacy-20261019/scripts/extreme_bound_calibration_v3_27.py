#!/usr/bin/env python3
"""Expose warmup/reuse legal-class gates for any formal W-minus multiplier."""
import argparse
import hashlib
import json
from pathlib import Path

from extreme_bound_calibration_v3_26 import build as build_prior
from extreme_bound_calibration_v3_16_rev2 import require_null_endpoints

ROOT = Path(__file__).resolve().parents[1]
PRIOR = "evidence/20260927_loop079_identity/run522/bound_calibration_v3_26.json"
DESIGN = "evidence/20260927_loop079_identity/run526/formal_ledger_source_preflight.md"
FORMAL_RUNNER = "scripts/run_loop036_static_e2e.sh"
CLIENT = "scripts/bench.py"
HASHES = {
    PRIOR: "1021f169313b378bbcde3213bb94fbe33051cd435fdb4654bcff41b055e56809",
    DESIGN: "beee89de51d6a315b69a0100cd7c342bd7135bac4726a234214763a65226987a",
    FORMAL_RUNNER: "bbf015e8388b9e734980a6957975c8e457cd7cf249a1b1cda3fbf0c912085a5d",
    CLIENT: "fcc584fa62dd933523c6c4c18a58e57c6561be554451d4d11a528b50dbff9526",
}


def build():
    for path, expected in HASHES.items():
        actual = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"pinned input changed: {path}: {actual}")
    model = build_prior()
    if json.dumps(model, ensure_ascii=False, indent=2) + "\n" != (ROOT / PRIOR).read_text():
        raise ValueError("V3.26 generator/output mismatch")
    runner = (ROOT / FORMAL_RUNNER).read_text()
    if runner.count("--out \"${OUT}/warmup.json\"") != 1 or runner.count("--max-tokens 1024") != 2:
        raise ValueError("formal warmup/measured protocol changed")
    if model["model_revision"] != "V3.26" or model["current"]["accepted_formal_tps"] != 571.681:
        raise ValueError("prior model changed")
    if any(model["proof_dag"]["certified"].values()):
        raise ValueError("prior proof state changed")
    model["model_revision"] = "V3.27"
    model["bound_ladder"]["algorithm_resource"]["pre_window_reuse_class_gate"] = {
        "status": "open_contract_interpretation_and_freshness",
        "source": [DESIGN, FORMAL_RUNNER, CLIENT],
        "formal_protocol_warmup_same_48_prompts_and_max_1024": True,
        "class_ladder": {
            "unrestricted_semantic_equivalence": {
                "full_result_memoization_policy": None,
                "positive_Target_conventional_W_minus": None,
                "caveat": "If arbitrary pre-window full-result memoization is legal, an observed measured-pass Target launch does not prove Target arithmetic compulsory across all architectures.",
            },
            "online_model_inference": {
                "full_result_memoization": "excluded_as_declared_class_assumption",
                "prefix_KV_and_allowed_intermediate_reuse": "must_be_specified",
                "fresh_semantic_F": None,
                "universal_dense_2MNK_work": None,
            },
            "ordinary_dense_BF16_projection": {
                "inherits": "online_model_inference",
                "additional_exclusions": "factorization, decision-only shortcuts, exact sparsity and alternate representations as in Run521",
                "per_fresh_required_Target_row_ops_if_all43x8_needed": 2885681152,
                "formal_F_by_layer_group": None,
                "formal_in_window_W_minus_ops": None,
            },
        },
        "warmup_measured_semantic_key_dedup_status": "unobserved",
        "scope": "Analysis classes only; no change to frozen serving/product contract or admission of a new Bound.",
    }
    model["certificate_graph"]["algorithm_resource"]["run526_pre_window_reuse_gate"] = {
        "status": "unresolved_legal_class_and_cross_warmup_identity",
        "evidence": [DESIGN, FORMAL_RUNNER],
        "missing": [
            "declare legal full-result and intermediate reuse for the Product inference claim",
            "same-run warmup/measured semantic keys, actual pre-window residency and phase lineage",
            "fresh required formal F[layer,group] under the chosen class",
        ],
        "bound_effect": "no positive universal Target work or finite Hardware/Product endpoint from observed current launches",
    }
    model["proof_dag"]["nodes"]["positive_unavoidable_W_minus"]["evidence"].append(DESIGN)
    model["proof_dag"]["nodes"]["positive_unavoidable_W_minus"]["missing"].append(
        "declare whether identical warmup full-output memoization is legal; deduplicate semantic keys across warmup and measured phases"
    )
    model["next_measurement"]["priority"] = (
        "One instrumented full48 warmup plus full48 measured ledger to join raw external output, Scheduler G, Runtime q/R, fresh required Target semantic keys and arrival/prefill/seed/publication boundaries. "
        "Keep its TPS diagnostic-only. Use the resulting same-trajectory frontier for all8 mixed-service/overlap calibration; pursue exact-board cumulative C-plus/B documentary evidence in parallel."
    )
    model["next_measurement"]["specific_gate"] = (
        "Preserve both phase identities and actual warmup reuse; do not infer F from new request IDs, measured native launches or 49152 output tokens alone. "
        "Attribute ordinary dense work only within an explicit legal execution class, and distinguish Host submission from device-ready and raw token ID from text representation."
    )
    model["input_paths"].extend(HASHES)
    require_null_endpoints(model)
    if any(model["proof_dag"]["certified"].values()):
        raise ValueError("warmup class gate cannot certify Bound")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    model = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "ok", "revision": "V3.27", "finite_endpoints": 0}))


if __name__ == "__main__":
    main()
