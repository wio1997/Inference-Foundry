#!/usr/bin/env python3
"""Evidence-pinned calibration of the frozen Extreme product bounds.

The conditional refill screens are sensitivity analyses. Missing compulsory
work, capacity, or dependency costs remain null instead of becoming a TPS.
"""
import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dual = read("evidence/20260926_loop064_cp/dual_bound_v2.json")
    fifo = read("evidence/20260926_loop074_refill/run330/counterfactual.json")
    fixed = read("evidence/20260926_loop074_refill/run329/counterfactual.json")
    granular = read("evidence/20260926_loop074_refill/run331/granularity.json")
    prep = read("evidence/20260926_loop074_refill/run340/analysis.json")
    hccl = read("evidence/20260926_loop074_refill/run339/analysis.json")
    acceptance239 = read("evidence/20260926_loop074_refill/run347/acceptance_reanalysis.json")
    gmm352 = read("evidence/20260927_loop075_bound/run352/analysis.json")
    banks357 = read("evidence/20260927_loop075_bound/run357/analysis.json")
    prof360 = read("evidence/20260927_loop075_bound/run360/analysis.json")
    binding365 = read("evidence/20260927_loop075_bound/run365/summary.json")
    assert dual["current_formal_tps"] == 571.681
    assert fixed["observed_cohort_wave_cycles"] == fifo["observed_cohort_wave_cycles"] == 1203
    assert fifo["instant_completion_zero_incremental_cost_fifo_cycles"] == 1118
    assert fixed["fixed_duration_work_floor_cycles"] == 1015
    assert fifo["instant_completion_fixed_duration_floor_cycles"] == 1011
    assert len(hccl["cases"]) == 5
    assert acceptance239["aggregate"]["all_rank_count_parity"]
    assert gmm352["status"] == "valid_isolated_graph_shape_service_ABA_no_product_bound"
    assert banks357["status"] == "valid_same_layer_bank_A_B_A2_timing_only"
    assert prof360["status"] == "verified_two_complete_graph_replays_each_condition"
    assert binding365["banks"] == 8 and binding365["all_bank_active_prefix_match"]
    assert binding365["active_prefix_max_abs_diff"] == 0.0
    assert sum(sum(row["immediate_completion_cycles"]) for row in fifo["cohorts"]) == 12122

    formal = dual["current_formal_observed_wall_envelope"]
    observed_cycles = [sum(c["cycles"] for c in row["cohorts"]) for row in formal]
    current = {
        "accepted_formal_tps": dual["current_formal_tps"],
        "implied_wall_s_for_49152_tokens": 49152 / dual["current_formal_tps"],
        "formal_repeat_client_wall_s": [row["client_wall_s"] for row in formal],
        "formal_repeat_cycles": observed_cycles,
        "source": "Run99 formal; dual_bound_v2 current_formal_observed_wall_envelope",
        "confidence": "high for measured Current, not an upper bound",
    }

    # These are three different relaxations of the *Run287 diagnostic* trace.
    # They neither share Run99 acceptance nor include the new 32K transaction.
    cycle_screen = {
        "source": "Run287 original warmup48 all-rank trajectory; Run329/330/331 offline replay",
        "observed_cycles": 1203,
        "slot_busy_cycles": fixed["observed_occupied_slot_cycles"],
        "slot_free_cycles": fixed["observed_free_slot_cycles"],
        "slot_free_fraction": fixed["observed_free_slot_fraction"],
        "fixed_duration_relaxation_cycles": {
            "instant_completion": 1011,
            "host_mirror_release": 1015,
            "derivation_instant": "ceil(12122 occupied slot-cycles / 12 slots) = 1011",
            "derivation_host_mirror": "ceil(12170 occupied slot-cycles / 12 slots) = 1015",
            "scope": "capacity relaxation for fixed Run287 request durations, not an achievable schedule",
        },
        "zero_incremental_cost_fifo_cycles": {
            "instant_completion": 1118,
            "host_mirror_release": fixed["fifo_per_slot_delay_cycles"]["0"]["cycles"],
            "scope": "constructed FIFO schedule under zero incremental cost, not a universal lower bound",
        },
        "batched_refill_host_mirror_cycles": {
            k: granular["granularity"][k]["cycles"] for k in ("2", "3", "4", "6", "12")
        },
        "not_product_bound": [
            "new prompts arrive only after early HTTP stream completion",
            "32K prefix and residual prefill, DSpark seed, KV/state transaction omitted",
            "resource contention and changed acceptance/duration omitted",
            "Run287 diagnostic trajectory is not Run99 formal trajectory",
        ],
    }
    assert math.ceil(cycle_screen["slot_busy_cycles"] / 12) == 1015

    prep_total = prep["sum_s"]
    break_even = []
    for row in prep["rows"]:
        saved = row["conditional_saved_cycles"]
        low_ms, high_ms = row["cycle_wall_reference_ms_range"]
        # With this cross-run reference only, all incremental preparation and
        # publication exposed to the Product path must fit this time budget.
        budget = [saved * low_ms / 1000, saved * high_ms / 1000]
        break_even.append({
            "refill_batch_size": row["batch_size"],
            "conditional_saved_cycles": saved,
            "screening_gross_wall_budget_s": budget,
            "run332_later_three_serial_prepare_s": prep_total,
            "required_hidden_prepare_fraction_if_same_work": [
                max(0, 1 - budget[1] / prep_total),
                max(0, 1 - budget[0] / prep_total),
            ],
            "scope": "cross-run break-even sensitivity, not attainable overlap or TPS",
        })

    inventory = dual["partial_resource_inventory"]
    output = {
        "schema_version": 3,
        "contract": dual["contract"],
        "current": current,
        "bound_ladder": {
            "algorithm_resource": {
                "latency_floor_s": None,
                "compulsory_work_complete": False,
                "current_runtime_cardinality_min_cycles": 512,
                "cardinality_scope": "four current zero-output handoff cohorts; if handoff/seed changes, recompute from real remaining external tokens and already generated output",
                "run239_same_trace_acceptance": acceptance239["aggregate"],
                "known_partial": inventory["canonical_matmul_arithmetic_Gflop_per_rank_target_cycle"],
                "missing": ["full Target and DSpark FLOPs at actual active shapes", "unique necessary packed weights/activation/KV/metadata bytes", "prefix-cache-conditioned 32K prompt and residual prefill work", "necessary TP8 link bytes and collective order", "acceptance-dependent executed work"],
                "confidence": "low; incomplete numerator",
            },
            "hardware_resource": {
                "latency_floor_s": None,
                "attainable_910B3_capacity_complete": False,
                "current_full_graph_counter_GB_per_rank_cycle": inventory["current_FULL_Graph_counter_GB_per_rank_target_cycle"],
                "isolated_hccl_actual_payload_us": {k: v["median_us"] for k, v in hccl["cases"].items()},
                "isolated_hccl_scope": "Run339 five warmed padded88 prefill collective shapes; not a mixed decode chain or physical link floor",
                "all8_GMM_graph_isolated_attained_service_us": {
                    name: row["all8_A2_rank_median_us_range"] for name, row in gmm352["cases"].items()
                },
                "all8_GMM_scope": "Run349/350/351 synthetic Graph A/B/A each rank selected a different Run121 ordinal layer; repeat one weight, 16 captured same-shape calls/replay. Host windows overlap, device per-replay overlap unproved. No FULL Graph mixed contention or physical peak.",
                "same_layer_weight_bank_sensitivity": {
                    "source": "Run354/355/356 ordinal64 all8 A=1/B=8/A2=1 independent packed-weight banks, 16 calls/replay; Run357 analysis",
                    "gmm1_median_B_vs_A_midpoint_percent": banks357["cases"]["gmm1"]["median_B_vs_A_midpoint_percent"],
                    "gmm1_strict_slower_ranks": banks357["cases"]["gmm1"]["strict_B_slower_ranks"],
                    "gmm2_median_B_vs_A_midpoint_percent": banks357["cases"]["gmm2"]["median_B_vs_A_midpoint_percent"],
                    "gmm2_strict_slower_ranks": banks357["cases"]["gmm2"]["strict_B_slower_ranks"],
                    "scope": "isolated synthetic zero-input Graph service; does not enclose actual Runtime GMM cost, compulsory traffic, or Product path. Distinct bank pointers and separate Run365 nonzero eight-bank active-prefix Graph/eager binding validated.",
                },
                "onecard_GMM2_bank_memoryaccess": {
                    "source": "Run358/359 Level1 rank4 ordinal64; Run360 verified final two 16-task native replay groups per condition",
                    "aic_read_main_memory_KB_per_native_task": prof360["median_counter_comparison"]["aic_read_main_memory_datas(KB)"],
                    "aic_GM_to_L1_KB_per_native_task": prof360["median_counter_comparison"]["aic_GM_to_L1_datas(KB)"],
                    "aiv_read_main_memory_KB_per_native_task": prof360["median_counter_comparison"]["aiv_read_main_memory_datas(KB)"],
                    "scope": "AIC byte counters unchanged despite timing sensitivity; no L2 hit/physical HBM attribution. Profiler Host/NPU flow parse failed and profiler timing is excluded from attained service.",
                },
                "missing": ["same-shape concurrent FULL Graph AIC/AIV/HBM capacity and cache residency", "mixed 264-collective service and topology link bytes", "rank arrival and compute/communication contention"],
                "confidence": "low; isolated HCCL timings are attainable samples, not a physical floor",
            },
            "scheduling_execution": {
                "latency_floor_s": None,
                "conditional_cycle_screen": cycle_screen,
                "refill_break_even_sensitivity": break_even,
                "missing": ["same-state legal-arrival/prefill/seed/first-Target timeline", "all-rank resource-constrained DAG costs including GMM weight residency", "actual concurrent resource contention", "cycle duration and useful-token changes under refill"],
                "confidence": "low for attainable Product; medium for offline cycle arithmetic",
            },
            "product_e2e": {
                "finite_tps_upper_bound": None,
                "demonstrated_current_tps": dual["current_formal_tps"],
                "unresolved_gap_tps": None,
                "missing": ["legal arrival and output publication trajectory", "prefill/seed/decode overlap", "repeatable corrected E2E intervention", "complete hardware and scheduling bounds"],
                "confidence": "high for Current; no defensible finite upper interval endpoint yet",
            },
        },
        "next_measurement": {
            "priority": "pin full-workload mandatory-work and same-path concurrent capacity before promoting a numeric bound",
            "specific_gate": "all-eight rank-cycle stage/traffic ledger with Target, DSpark, KV, HCCL payload and overlap, plus actual-arrival-to-first-new-Target only if refill is tested",
            "decision_rule": "numeric Product bound requires a legal resource-constrained DAG and unchanged-contract corrected E2E calibration",
        },
        "input_paths": [
            "evidence/20260926_loop064_cp/dual_bound_v2.json",
            "evidence/20260926_loop074_refill/run329/counterfactual.json",
            "evidence/20260926_loop074_refill/run330/counterfactual.json",
            "evidence/20260926_loop074_refill/run331/granularity.json",
            "evidence/20260926_loop074_refill/run339/analysis.json",
            "evidence/20260926_loop074_refill/run340/analysis.json",
            "evidence/20260926_loop074_refill/run347/acceptance_reanalysis.json",
            "evidence/20260927_loop075_bound/run352/analysis.json",
            "evidence/20260927_loop075_bound/run357/analysis.json",
            "evidence/20260927_loop075_bound/run360/analysis.json",
            "evidence/20260927_loop075_bound/run365/summary.json",
        ],
    }
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "ok", "output": str(dest), "formal_cycles": observed_cycles, "conditional_refill_cycles": cycle_screen["zero_incremental_cost_fifo_cycles"]}))


if __name__ == "__main__":
    main()
