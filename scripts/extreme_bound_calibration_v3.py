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
    joint368 = read("evidence/20260927_loop076_bound/run368/analysis.json")
    dense370 = read("evidence/20260927_loop076_bound/run370/analysis.json")
    dspark372 = read("evidence/20260927_loop076_bound/run372/analysis.json")
    markov376 = read("evidence/20260927_loop077_bound/run376/analysis.json")
    hccl378 = read("evidence/20260927_loop077_bound/run378/ledger.json")
    joined380 = read("evidence/20260927_loop077_bound/run380/analysis.json")
    hccl_graph = [read(f"evidence/20260927_loop077_bound/run{run}/chain.json") for run in (382, 383, 384)]
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
    assert joint368["status"] == "same_run_same_cycle_route_counter_join_instrumented"
    assert joint368["summary"]["samples"] == 16
    assert dense370["summary"]["valid_rank_cycles"] == 16
    assert dspark372["status"] == "DSpark_proposer_partial_GEMM_work_census_corrected_device_tail"
    assert dspark372["summary"]["rank_cycles"] == 16
    assert markov376["summary"]["samples"] == 16
    assert hccl378["summary"]["rank_cycles_identical"] == 16
    assert joined380["summary"]["rank_cycles"] == 80
    assert all(x["result_checks_passed"] and x["collectives_per_chain"] == 265 for x in hccl_graph)
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
    gmm_work = joint368["summary"]["all43_standard_GMM_Gflop_per_rank_cycle_range"]
    dense_work = dense370["summary"]["sum_standard_Gflop_median"]
    compressor_work = inventory["canonical_matmul_arithmetic_Gflop_per_rank_target_cycle"]["compressor_fixed_shapes"]
    draft_dense_work = dspark372["summary"]["dense_standard_Gflop_partial_median"]
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
                "same_cycle_target_route_useful_output_diagnostic": {
                    "source": "Run375/380 no Level1 profiler, selected cycle64/65 across five 12-request cohorts and all8 ranks",
                    "TP8_target_GMM_standard_Gflop_each_cycle": joined380["summary"]["TP8_target_GMM_standard_Gflop_each_cycle"],
                    "useful_tokens_by_cohort_cycle64_65": joined380["summary"]["useful_clipped_tokens_cycle64_65_by_cohort"],
                    "TP8_target_GMM_Gflop_per_useful_token_range": joined380["summary"]["TP8_target_GMM_Gflop_per_useful_token_range"],
                    "scope": "Current fixed padded Target work and clipped Runtime useful output on ten diagnostic cycles; route-stack operation adds selected-cycle device work. Not compulsory model FLOPs or formal Run99 trajectory.",
                },
                "dspark_partial_dense_standard_GEMM_Gflop_per_rank_cycle": {"observed": draft_dense_work, "source": "Run372 corrected device-tail census from Run246 original FULL Graph", "scope": "18 quant + 16 plain + 3 transpose; not compulsory or total draft work; three GMM1/GMM2 routed counts, attention/KV/state omitted"},
                "same_trajectory_standard_GEMM_equivalent_partial_Gflop_per_rank_cycle": {
                    "GMM_route_dependent_range": gmm_work,
                    "fixed_dense_quant_plain_transpose": dense_work,
                    "compressor_fixed_shape_separate_source": compressor_work,
                    "sum_partial_range_cross_source": [gmm_work[0] + dense_work + compressor_work, gmm_work[1] + dense_work + compressor_work],
                    "GMM_active_packed_weight_footprint_GB_range": joint368["summary"]["all43_active_packed_weight_GB_per_rank_cycle_range"],
                    "source": "Run368 same-run same-cycle original FULL Graph live route for 43 GMM pairs; Run370 fixed-shape dense families from prior Run246; Run256 Compressor arithmetic",
                    "scope": "current-route standard GEMM-equivalent arithmetic and storage footprint; not complete compulsory work, not a strict algorithmic arithmetic or HBM-read minimum. Cross-source sum assumes frozen fixed dense/Compressor shapes.",
                },
                "missing": ["full Target and DSpark FLOPs at actual active shapes", "unique necessary packed weights/activation/KV/metadata bytes", "prefix-cache-conditioned 32K prompt and residual prefill work", "necessary TP8 link bytes and collective order", "acceptance-dependent executed work"],
                "confidence": "low; incomplete numerator",
            },
            "hardware_resource": {
                "latency_floor_s": None,
                "attainable_910B3_capacity_complete": False,
                "current_full_graph_counter_GB_per_rank_cycle": inventory["current_FULL_Graph_counter_GB_per_rank_target_cycle"],
                "same_trajectory_GMM_counter_to_active_packed_weight_ratio": {
                    "AIC_plus_AIV_read_range": joint368["summary"]["all43_AIC_AIV_read_to_active_packed_weight_ratio_range"],
                    "median": joint368["summary"]["all43_AIC_AIV_read_to_active_packed_weight_ratio_median"],
                    "same_ordinal_active_expert_read_correlations": joint368["summary"]["same_ordinal_active_expert_to_AIC_read_pearson_688_pairs"],
                    "scope": "Run368 matched route/task Level1 current traffic; non-weight reads included, cache/physical HBM unresolved. Replaces cross-run 1.092 ratio for same-trajectory interpretation.",
                },
                "isolated_hccl_actual_payload_us": {k: v["median_us"] for k, v in hccl["cases"].items()},
                "exact_order_265_HCCL_graph_service_attained_ms": {
                    "independent_process_medians": [x["latest_rank_device_median_ms"] for x in hccl_graph],
                    "source": "Run378 allrank reported task order; Run381 four-class Graph semantic gate; Run382-384 ordered-signature synthetic Graph replay with final reused buffer check per signature",
                    "scope": "Isolated synthetic Graph with conditional ABI task kind/count/dtype/order, final reused buffer check per signature, no per-operation native identity proof or rank-synchronized makespan; no model compute/HBM or producer arrivals. Attained service only, not physical floor or exposed Product cost.",
                },
                "target_hccl_reported_payload_bytes_per_rank_cycle": hccl378["summary"]["reported_chunk_bytes_sum"],
                "target_hccl_conditional_API_input_bytes_per_rank_cycle": hccl378["summary"]["conditional_API_input_bytes_sum"],
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
                "missing": ["same-shape concurrent FULL Graph AIC/AIV/HBM capacity and cache residency", "physical topology link bytes and mixed compute-contended 265-collective service", "rank arrival and compute/communication contention"],
                "confidence": "low; isolated HCCL timings are attainable samples, not a physical floor",
            },
            "scheduling_execution": {
                "latency_floor_s": None,
                "conditional_cycle_screen": cycle_screen,
                "current_DSpark_Markov_feedback": {
                    "source": "Run372/376 corrected device-tail and llm_base_proposer.py:1422-1436",
                    "seven_bias_task_span_us_profiled_range": markov376["summary"]["span_us_range"],
                    "seven_bias_task_read_counter_GB_median": markov376["summary"]["seven_task_read_counter_GB_median"],
                    "dependency": "draft[idx] -> embedding -> bias -> add -> argmax -> draft[idx+1] under current algorithm",
                    "scope": "Serial feedback in current proposer; profiled span not a floor, and alternate space-for-time algorithms may change work/storage.",
                },
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
            "specific_gate": "unperturbed all-rank full-cycle Target/acceptance/state/Draft/next-Target device join, direct decode HCCL ABI/physical bytes, Draft GMM/KV/prefill compulsory work and mixed compute-HBM-HCCL capacity; legal arrival-to-first-Target if refill is tested",
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
            "evidence/20260927_loop076_bound/run368/analysis.json",
            "evidence/20260927_loop076_bound/run370/analysis.json",
            "evidence/20260927_loop076_bound/run372/analysis.json",
            "evidence/20260927_loop077_bound/run376/analysis.json",
            "evidence/20260927_loop077_bound/run378/ledger.json",
            "evidence/20260927_loop077_bound/run380/analysis.json",
            "evidence/20260927_loop077_bound/run382/chain.json",
            "evidence/20260927_loop077_bound/run383/chain.json",
            "evidence/20260927_loop077_bound/run384/chain.json",
        ],
    }
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "ok", "output": str(dest), "formal_cycles": observed_cycles, "conditional_refill_cycles": cycle_screen["zero_incremental_cost_fifo_cycles"]}))


if __name__ == "__main__":
    main()
