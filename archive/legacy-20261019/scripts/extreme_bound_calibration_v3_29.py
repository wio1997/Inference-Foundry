#!/usr/bin/env python3
"""V3.29: admit the same-run Host output lineage without promoting it to a bound."""
import argparse
import hashlib
import json
from pathlib import Path

from extreme_bound_calibration_v3_28 import build as build_prior
from extreme_bound_calibration_v3_16_rev2 import require_null_endpoints

ROOT = Path(__file__).resolve().parents[1]
PRIOR = "evidence/20260927_loop080_bound/run540/bound_calibration_v3_28.json"
METRICS = "evidence/20260927_loop079_identity/run546/host_ledger_metrics.json"
REVIEW = "evidence/20260927_loop079_identity/run544/astra_posthoc_review.md"
HASHES = {
    PRIOR: "f0384d6bbf5655d33d14c1302aad21583cf5716125b88ed0dd4ef387a1a8cf3b",
    METRICS: "0c86ec1d23aae9e2cebf28d0555f05d711b56ab62cfdbf78c9651a41444763a6",
    REVIEW: "b5ad77f36a5abe779612d1c174a5d8e3dbe5ded398d1e13b4e067939797b5d01",
    "evidence/20260927_loop080_bound/run545/astra_bound_review.md": "1244ceec0a04c6c6143eaa9288b23138c40bc75cec94df34abdfa6b88a0a6473",
}


def build():
    for path, expected in HASHES.items():
        actual = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"pinned input changed: {path}: {actual}")
    model = build_prior()
    if json.dumps(model, ensure_ascii=False, indent=2) + "\n" != (ROOT / PRIOR).read_text():
        raise ValueError("V3.28 generator/output mismatch")
    if model["model_revision"] != "V3.28":
        raise ValueError("unexpected prior revision")
    metric = json.loads((ROOT / METRICS).read_text())
    if metric["status"] != "posthoc_host_lineage_metrics_conditional_on_run543_admission":
        raise ValueError("lineage not admitted")
    m = metric["phases"]["measured"]
    w = metric["phases"]["warmup"]
    if (m["request_count"], w["request_count"]) != (48, 48):
        raise ValueError("not frozen full 48+48")
    if m["runtime_retained_tokens"] != 49152 or m["ordinary_admitted_tokens"] + m["runtime_terminal_admitted_tokens"] != 49152:
        raise ValueError("output accounting does not close")
    if m["runtime_sampled_q_before_remaining_clip"] < m["runtime_retained_tokens"]:
        raise ValueError("sampled q cannot be smaller than retained R")
    if not m["all48_first_runtime_token_is_in_bulk_admitted_prefix"]:
        raise ValueError("first Runtime token lineage incomplete")
    if m["output_processor_prefill_stats_repr"]["cached_prompt_tokens"]["min"] != 32768:
        raise ValueError("measured cache statistic changed")
    if metric["paired_warmup_measured_server_raw_outputs"]["identical_full_1024_sequences"] != 0:
        raise ValueError("paired raw output fact changed")
    model["model_revision"] = "V3.29"
    model["bound_semantics_v3_29"] = {
        "status": "same_run_host_lineage_admitted_conditional_bounds_still_open",
        "source_run": "Run542 controller failed at original validator serialization; Run543 replay and Run544 independent posthoc review admitted diagnostic lineage",
        "source": [METRICS, REVIEW, "evidence/20260927_loop080_bound/run545/astra_bound_review.md"],
        "clock_scope": "server/client CLOCK_MONOTONIC comparisons gated on common boot ID, time namespace and zero offsets; controller clock not joined",
        "measured_output_accounting": {
            "client_required_output_tokens": 49152,
            "scheduler_ordinary_before_terminal_bulk_G": m["ordinary_admitted_tokens"],
            "runtime_terminal_bulk_admitted": m["runtime_terminal_admitted_tokens"],
            "runtime_retained_R": m["runtime_retained_tokens"],
            "runtime_sampled_q_before_remaining_clip": m["runtime_sampled_q_before_remaining_clip"],
            "sampled_q_minus_R": m["runtime_q_minus_retained_R"],
            "first_runtime_token_retained_all48": True,
            "first_runtime_token_server_raw_ordinal_zero_based_range": m["first_runtime_token_server_full_raw_ordinal_zero_based_range"],
            "rank0_cohort_cycles": m["rank0_cohort_cycles"],
            "rank0_cohort_cycle_sum": m["rank0_cohort_cycle_sum"],
            "ordinary_G_at_rank0_handoff": 400,
            "ordinary_G_after_handoff_before_terminal_bulk": 1,
            "scope": "Current instrumented same-run output lineage; G at terminal prebulk is401, G at rank0 handoff is400. q and cycles are observed work, not compulsory Algorithm/Resource work or a schedule floor.",
        },
        "measured_prefill_stats_repr": {
            "cached_prompt_tokens_per_request": 32768,
            "computed_prompt_tokens_total": m["output_processor_prefill_stats_repr"]["computed_prompt_tokens"]["sum"],
            "computed_prompt_tokens_per_request_range": [m["output_processor_prefill_stats_repr"]["computed_prompt_tokens"]["min"], m["output_processor_prefill_stats_repr"]["computed_prompt_tokens"]["max"]],
            "warmup_cached_prompt_tokens_total": w["output_processor_prefill_stats_repr"]["cached_prompt_tokens"]["sum"],
            "warmup_computed_prompt_tokens_total": w["output_processor_prefill_stats_repr"]["computed_prompt_tokens"]["sum"],
            "scope": "OutputProcessor source repr only. Cache lookup/residual computation, seed, KV and device readiness remain unobserved.",
        },
        "instrumented_host_spans": {
            "client_start_to_output_add_median_ms": m["client_start_to_output_add_ms"]["median"],
            "output_add_to_rank0_handoff_median_ms": m["output_add_to_rank0_handoff_ms"]["median"],
            "rank0_handoff_to_done_median_ms": m["rank0_handoff_to_rank0_done_ms"]["median"],
            "rank0_done_to_client_end_median_ms": m["rank0_done_to_client_end_ms"]["median"],
            "scope": "Inclusive, overlapping current Host observations. No sum or subtraction is a necessary critical path or removable cost.",
        },
        "paired_raw_output": metric["paired_warmup_measured_server_raw_outputs"],
        "conditional_same_trajectory_cycle_relaxations": {
            "remaining_bulk_slot_load_ceil": 508,
            "per_request_ideal8_acceptance_slot_load_ceil": 510,
            "four_fixed_cohort_ideal8_chain_sum": 512,
            "observed_q_duration_clipped_bulk_arbitrary_slot_load_ceil": 1035,
            "observed_q_duration_clipped_bulk_fixed_cohort_chain_sum": 1206,
            "executed_cycle_sum": 1218,
            "scope": "Run545 independent count arithmetic with observed current q and optimistic zero-cost G. These are conditional cycle-domain relaxations, not Product time lower bounds, alternative feasible schedules or removable cycles.",
        },
        "proof_impact": {
            "closed": ["same-run full48 raw-output lineage", "prebulk versus terminal bulk useful output arithmetic", "client/server Host monotonic scope"],
            "open": ["legal memoization and fresh Target witness", "prefill/seed/KV device-ready and successor release", "same-run all8 mixed compute/HBM/HCCL contention", "strict C_plus/B and compulsory work/traffic", "feasible alternative schedule and formal E2E replication"],
            "algorithm_resource_latency_floor_s": None,
            "hardware_resource_latency_floor_s": None,
            "scheduling_execution_latency_floor_s": None,
            "product_e2e_tps_ceiling": None,
            "distance_from_formal_current_571_681_to_limit_tps": None,
        },
    }
    model["bound_ladder"]["algorithm_resource"]["run542_546_same_run_output_lineage"] = {
        "source": METRICS,
        "ordinary_G": m["ordinary_admitted_tokens"],
        "terminal_bulk_admitted": m["runtime_terminal_admitted_tokens"],
        "sampled_q": m["runtime_sampled_q_before_remaining_clip"],
        "required_external_output": 49152,
        "fresh_required_Target_F_certified": False,
        "scope": "Observed Current trajectory, not model compulsory work or arbitrary architecture numerator.",
    }
    model["bound_ladder"]["scheduling_execution"]["run542_546_host_lineage"] = {
        "source": METRICS,
        "clock_scope_gated": True,
        "rank0_cycle_sum": m["rank0_cohort_cycle_sum"],
        "cached_residual_device_ready": None,
        "all8_mixed_service_frontier": None,
        "feasible_alternative_schedule": None,
        "ordinary_G_at_rank0_handoff": 400,
        "terminal_prebulk_G": 401,
        "scope": "Instrumented current Host event chronology only; overlap and resource contention unmeasured.",
    }
    model["next_measurement"]["priority"] = (
        "Use Run542–546 admitted same-run output lineage to capture per-cohort cache/residual prefill, seed and KV device-ready cuts "
        "and successor release around actual handoff. Then sample all8 mixed compute/HBM/HCCL frontier with real readiness, "
        "while separately obtaining a matching certified exact-board cumulative-capacity C-plus/B envelope and completing compulsory work/traffic. Keep Current and diagnostic TPS apart."
    )
    model["input_paths"].extend(HASHES)
    require_null_endpoints(model)
    prior_semantics = model["bound_semantics_v3_28"]
    new_semantics = model["bound_semantics_v3_29"]
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
        new_semantics["proof_impact"]["algorithm_resource_latency_floor_s"],
        new_semantics["proof_impact"]["hardware_resource_latency_floor_s"],
        new_semantics["proof_impact"]["scheduling_execution_latency_floor_s"],
        new_semantics["proof_impact"]["product_e2e_tps_ceiling"],
        new_semantics["proof_impact"]["distance_from_formal_current_571_681_to_limit_tps"],
        model["bound_ladder"]["scheduling_execution"]["run542_546_host_lineage"]["cached_residual_device_ready"],
        model["bound_ladder"]["scheduling_execution"]["run542_546_host_lineage"]["all8_mixed_service_frontier"],
        model["bound_ladder"]["scheduling_execution"]["run542_546_host_lineage"]["feasible_alternative_schedule"],
    ]
    if any(value is not None for value in unproved):
        raise ValueError("V3.29 unproved endpoint filled without new evidence")
    if any(model["proof_dag"]["certified"].values()):
        raise ValueError("Host lineage cannot certify a finite bound")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    model = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "ok", "revision": "V3.29", "finite_endpoints": 0}))


if __name__ == "__main__":
    main()
