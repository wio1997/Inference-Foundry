#!/usr/bin/env python3
"""Record Run421 terminal-output accounting without inventing a finite bound."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIOR = "evidence/20260927_loop078_bound/run418/bound_calibration_v3_11.json"
LEDGER = "evidence/20260927_loop079_identity/run421/analysis.json"
REVIEW = "evidence/20260927_loop079_identity/run422/astra_ledger_review.md"
RESOURCE = "evidence/20260927_loop079_identity/run423/astra_minimal_ceiling_review.md"
IDENTITY = "evidence/20260927_loop079_identity/run424/identity_capture_plan.md"


def read_json(path):
    return json.loads((ROOT / path).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = read_json(PRIOR)
    ledger = read_json(LEDGER)
    review = (ROOT / REVIEW).read_text()
    resource = (ROOT / RESOURCE).read_text()
    identity = (ROOT / IDENTITY).read_text()
    assert ledger["status"] == "positive_prior_generated_requires_api_publication_ledger"
    assert ledger["scheduler_ledger_rows"] == ledger["client_requests"] == ledger["http_posts"] == 60
    assert ledger["rank_cohort_reports"] == 40
    assert ledger["scheduler_generated_before_bulk_total"] == 584
    assert ledger["runtime_incoming_bulk_total"] == 61440
    assert ledger["scheduler_admitted_from_bulk_total"] == 60856
    assert ledger["runtime_incoming_bulk_total"] - ledger["scheduler_admitted_from_bulk_total"] == 584
    assert ledger["external_pre_handoff_p_i_verified"] is False
    assert "g_before" in review and "p_i" in review
    assert "W^- / C^+" in resource or "W^-" in resource
    assert "CONDITIONAL_NATIVE" in identity

    algorithm = model["bound_ladder"]["algorithm_resource"]
    algorithm["run421_terminal_bulk_token_accounting"] = {
        "scope": "Clean frozen 48+12 diagnostic, five cohorts, 60 unique requests, all8 FULL Graph Runtime reports; not formal Run99 or historical Run403 trajectory.",
        "scheduler_generated_before_terminal_bulk_total": 584,
        "runtime_incoming_bulk_total": 61440,
        "scheduler_admitted_from_bulk_total": 60856,
        "runtime_output_clipped_by_scheduler_total": 584,
        "per_cohort": ledger["cohort_accounting"],
        "handoff_time_already_generated_g_i_H": None,
        "handoff_time_API_published_p_i_H": None,
        "interpretation": "Terminal Scheduler g_before is neither handoff-time generated g_i_H nor handoff-time API-published p_i_H. It proves 1024 Runtime-retained tokens/request cannot be equated to 1024 externally newly admitted tokens/request for this diagnostic.",
        "source": [LEDGER, REVIEW],
    }
    algorithm["missing"].append("request-correlated handoff-time generated/queued state g_i_H and API publication p_i_H; no Run421 count transfer to Run99 or Run403")
    algorithm["run424_router_identity_gate"] = {
        "status": "conditional_native",
        "need": "actual all43 per-layer branch, keyword graph address binding, Runtime CP query geometry, target_logits_indices and native row-order contracts",
        "source": IDENTITY,
    }

    hardware = model["bound_ladder"]["hardware_resource"]
    hardware["run423_minimal_strict_ceiling_review"] = {
        "sufficient_theorem": "For a proved positive necessary in-window subset W_minus with matching genuine aggregate capacity upper cap C_plus, T_product >= W_minus/C_plus and TPS_product <= 49152*C_plus/W_minus; omitted work and ideal overlap loosen but do not invalidate this relaxation.",
        "candidate": "One fresh BF16 wo_a group-row, 2*4096*1024 = 8388608 conventional operations, conditional on dense evaluation and a consumer/freshness witness.",
        "missing": "910B3 SKU/configuration BF16 maximum-rate binding with clock/error envelope, and subset necessity for declared algorithm class; the official Atlas800T A2 options are not yet bound to this host.",
        "numeric_ceiling_promoted": False,
        "source": RESOURCE,
    }
    product = model["bound_ladder"]["product_e2e"]
    product["run421_published_output_ledger"] = {
        "terminal_scheduler_count_known": True,
        "handoff_generated_count_known": False,
        "handoff_published_count_known": False,
        "formal_current_changed": False,
        "source": [LEDGER, REVIEW],
    }
    assert product["finite_tps_upper_bound"] is None
    assert product["demonstrated_current_tps"] == 571.681
    assert all(model["bound_ladder"][name]["latency_floor_s"] is None for name in
               ("algorithm_resource", "hardware_resource", "scheduling_execution"))
    model["next_measurement"]["priority"] = "Close handoff-time already-generated g_i_H and API-published p_i_H against terminal Scheduler ledger; preserve all43 row-identity source/graph/native proof and the minimal W-minus/C-plus hardware certificate as parallel Bound tracks."
    model["next_measurement"]["specific_gate"] = "A clean exact60-POST run with request-correlated ModelRunner handoff token-state/position and Scheduler/OutputProcessor/API publication ordinals on a shared monotonic clock; no hot-path NPU sync. Confirm normal lifecycle, no alternate output route, and exact per-request clipping."
    model["input_paths"].extend([PRIOR, LEDGER, REVIEW, RESOURCE, IDENTITY])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "ok", "terminal_prior_generated": 584,
                      "external_p_i_H": None, "finite_product_ceiling": None}))


if __name__ == "__main__":
    main()
