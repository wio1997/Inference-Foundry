#!/usr/bin/env python3
"""Qualify local count-copy ordering and minimum sufficient strict-ceiling gate."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIOR = "evidence/20260927_loop078_bound/run415/bound_calibration_v3_10.json"
LOCAL = "evidence/20260927_loop078_bound/run412/local_b_analysis.json"
REVIEW = "evidence/20260927_loop078_bound/run412/astra_b_review.md"
BOUND_REVIEW = "evidence/20260927_loop078_bound/run416/astra_next_bound_review.md"
FAILURE = "evidence/20260927_loop078_bound/run411/host_process_census_before.txt"
A1 = "evidence/20260927_loop078_bound/run417/gate.json"
ANALYSIS = "evidence/20260927_loop078_bound/run412/analysis.json"


def load(path):
    return json.loads((ROOT / path).read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    m = load(PRIOR)
    local = load(LOCAL)
    a1 = load(A1)
    analysis = load(ANALYSIS)
    assert a1["arm"] == "A1" and a1["http_posts"] == 60 and a1["lifecycle_closed"]
    assert analysis["status"] == "INCONCLUSIVE" and not analysis["control_gate_passed"]
    assert local["rank_cohort_pairs"] == 40
    assert local["status"] == "accepted_instrumented_same_device_local_read_before_overwrite"
    assert "INCONCLUSIVE" in (ROOT / REVIEW).read_text()
    assert "W^- / C^+" in (ROOT / BOUND_REVIEW).read_text()
    assert "vllm-ascend26-dsv4f-w4a8 885.6GiB / 1006GiB 9756" in (ROOT / FAILURE).read_text()
    schedule = m["bound_ladder"]["scheduling_execution"]
    schedule["run410_count_copy_local_observation"] = {
        "same_device_rank_cohort_pairs": 40,
        "R_DONE64_to_W_PRE65_direct_margin_ms": local["direct_margin_ms"],
        "first_host_numeric_consumers": local["first_host_numeric_consumers"],
        "later_draft_metadata_consumer": local["later_draft_metadata_consumer"],
        "source_ordering_proof": False,
        "original_schedule_extrapolation": None,
        "A0_B_A1_control_note": "Run411 A1 startup OOM before POST; Run417 retry passed frozen diagnostic after dedicated-container reset. Environment comparability and per-cycle acceptance parity are absent; no marker-overhead coefficient.",
        "scope": "Instrumented normal unparked generation64 to65 on five 12-request cohorts and eight ranks; not a removable latency or a universal edge.",
        "source": [LOCAL, REVIEW, FAILURE, A1, ANALYSIS],
    }
    schedule["missing"].append("actual fixed-branch Draft/DSA numeric consumer, parking/terminal and all8 Graph/HCCL joins; original-schedule A/A/B overhead remains uncalibrated")
    product = m["bound_ladder"]["product_e2e"]
    product["strict_outer_ceiling_sufficient_gate"] = {
        "formula": "T_star >= max_i(W_i_minus/C_i_plus) > 0; TPS_star <= 49152/T_star",
        "requirements": "At least one proven positive necessary work or byte subset W_minus for an explicitly declared execution class, and a matching justified upper capacity cap C_plus. Serial path floors may be added only with proven dependence and nonoverlap.",
        "status": "No currently proved matched W_minus/C_plus pair; no finite TPS endpoint promoted.",
        "distinction": "An attainable engineering interval additionally requires a legal schedule/implementation, storage feasibility, mixed service evidence and Product calibration.",
        "source": BOUND_REVIEW,
    }
    assert product["finite_tps_upper_bound"] is None
    assert all(m["bound_ladder"][key]["latency_floor_s"] is None for key in ("algorithm_resource", "hardware_resource", "scheduling_execution"))
    m["next_measurement"]["priority"] = "source-and-runtime row identity plus Target/Draft format and external output ledger; then qualify physical transfer/cache counters and capacity, then all8 typed execution joins"
    m["next_measurement"]["specific_gate"] = "Run407 branch/Graph certificate bound to selected replay; one real DSA A2A native peer/byte/path export capability; original fixed-branch Draft consumer and parking/terminal dependency"
    m["input_paths"].extend([PRIOR, LOCAL, REVIEW, BOUND_REVIEW, FAILURE, A1, ANALYSIS])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "ok", "local_pairs": 40, "finite_product_ceiling": None}))


if __name__ == "__main__":
    main()
