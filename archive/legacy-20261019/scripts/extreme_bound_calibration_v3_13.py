#!/usr/bin/env python3
"""Add reviewed Run427 Host cutoffs without promoting a numeric TPS ceiling."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIOR = "evidence/20260927_loop079_identity/run425/bound_calibration_v3_12.json"
LEDGER = "evidence/20260927_loop079_identity/run427/analysis.json"
REVIEW = "evidence/20260927_loop079_identity/run428/astra_timeline_pre_review.md"
DESIGN = "evidence/20260927_loop079_identity/run426/astra_handoff_publication_plan.md"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = json.loads((ROOT / PRIOR).read_text())
    ledger = json.loads((ROOT / LEDGER).read_text())
    review = (ROOT / REVIEW).read_text()
    assert ledger["run_id"] == "LOOP079-RUN427"
    assert ledger["status"] == "host_timeline_valid_pending_independent_review"
    assert ledger["http_posts"] == ledger["request_generations"] == 60
    assert ledger["runtime_rank_cohort_reports"] == 40
    assert all(ledger["event_counts"].get(k) == 376 for k in (
        "scheduler_append", "output_processor_receive",
        "output_processor_queue_submitted", "api_consume", "api_generator_yield"))
    assert ledger["scheduler_terminal_prebulk_total"] == 796
    assert ledger["scheduler_committed_at_global_H_total_interval"] == [795, 796]
    assert ledger["api_raw_consumed_at_global_H_total_interval"] == [674, 796]
    assert ledger["api_generator_yield_count_at_global_H_interval"] == [213, 268]
    assert ledger["requests_with_confirmed_reasoning_delta_yield_before_H"] == 50
    assert len(ledger["per_request"]) == 60
    assert "Latest reducer revision rechecked" in review
    assert "H_probe is not H_run" in review
    assert "213" in review and "376" in review

    scope = ("Run427 exact60 48+12 diagnostic; five separate cohort H_probe "
             "envelopes, all8 FULL Graph; Host-only instrumentation perturbs timing. "
             "No transfer to Run99, Run403, or Run421 trajectories.")
    algorithm = model["bound_ladder"]["algorithm_resource"]
    algorithm["run427_host_cutoff_accounting"] = {
        "scope": scope,
        "H_definition": "per-rank Host probe before FixedCohortServing.run; not exact run entry or device-ready time",
        "scheduler_committed_G_at_cohort_H_probe_sum_interval": [795, 796],
        "terminal_scheduler_prior_G": 796,
        "api_raw_consumed_A_at_cohort_H_probe_sum_interval": [674, 796],
        "chat_generator_generated_output_yields_Y_at_cohort_H_probe_sum_interval": [213, 268],
        "requests_with_certified_nonempty_reasoning_delta_yields_before_H_probe_at_least": 50,
        "certified_nonempty_reasoning_delta_yields_before_H_probe_at_least": 213,
        "raw_token_literal_publication_p_i_H": None,
        "device_completed_tokens_D_i_H": None,
        "exact_H_run": None,
        "interpretation": "S→R→Q→A ordered len/SHA256 and causal chain proves a scoped Host consumption cap; successor markers prove completed generator yields. These are not ASGI send/client receipt or a necessary device-work numerator.",
        "source": [LEDGER, REVIEW, DESIGN],
    }
    algorithm["missing"].append(
        "Run427 device-completed D_i(H_run), exact H_run, literal raw-token publication p_i(H), ASGI/client receipt remain unmeasured; no cross-run transfer")
    scheduling = model["bound_ladder"]["scheduling_execution"]
    scheduling["run427_host_boundary_dependency"] = {
        "observed": "For this run, Scheduler append→OutputProcessor receive→queue submit→Chat consume matches 376 ordered segments. Generator yield completion has successor-based [213,268] envelope.",
        "not_proven": "No device-completion/Graph critical path, overlap saving, Host submission saving, or resource-constrained execution schedule follows from these markers.",
        "source": [LEDGER, REVIEW],
    }
    product = model["bound_ladder"]["product_e2e"]
    product["run427_host_to_product_boundary"] = {
        "generator_reasoning_chunks_completed_before_H_probe_at_least": 213,
        "requests_with_such_chunks_at_least": 50,
        "asgi_send_or_client_receipt_timed": False,
        "formal_current_changed": False,
        "finite_bound_promoted": False,
        "source": [LEDGER, REVIEW],
    }
    assert model["bound_ladder"]["hardware_resource"]["latency_floor_s"] is None
    assert all(model["bound_ladder"][name]["latency_floor_s"] is None for name in
               ("algorithm_resource", "hardware_resource", "scheduling_execution"))
    assert product["finite_tps_upper_bound"] is None
    assert product["demonstrated_current_tps"] == 571.681
    model["next_measurement"]["priority"] = (
        "Close all43 current native-router row identity and same-trajectory compulsory work; "
        "in parallel, bind a genuine 910B3 matching resource capacity upper and probe "
        "the device/Host execution boundary only where it changes a Bound certificate.")
    model["next_measurement"]["specific_gate"] = (
        "Current installed CANN native row-order/branch source plus selected FULL Graph "
        "keyword tensor/address/shape identity for all43 routers; retain source/clock "
        "versioning. For a finite ceiling, pair a strictly necessary same-run W_minus "
        "with true aggregate C_plus or a valid resource-constrained DAG."
    )
    model["input_paths"].extend([PRIOR, LEDGER, REVIEW, DESIGN])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "ok", "G_H_probe": [795,796],
                      "A_H_probe": [674,796], "Y_H_probe": [213,268],
                      "finite_product_ceiling": None}))


if __name__ == "__main__":
    main()
