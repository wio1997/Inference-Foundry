#!/usr/bin/env python3
"""Reconcile saved Run239 cycle acceptance with Runtime-retained outputs."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "evidence/20260926_loop059_boundary/run239"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    bench = json.loads((BASE / "measured48.json").read_text())
    assert bench["summary"]["success"] == 48 and bench["summary"]["fail"] == 0
    cohorts = []
    all_durations = []
    for cid in range(5, 9):
        rows = [json.loads((BASE / f"boundary/rank{rank}_cohort{cid}.json").read_text()) for rank in range(8)]
        counts = rows[0]["counts_cpu"]
        digest = hashlib.sha256(json.dumps(counts, separators=(",", ":")).encode()).hexdigest()
        for rank, row in enumerate(rows):
            assert row["rank"] == rank and row["cohort"] == cid
            assert row["counts_cpu"] == counts, f"acceptance rank mismatch cohort{cid} rank{rank}"
            assert len(row["counts_cpu"]) > 0
        assert all(len(row) == 12 and all(isinstance(x, int) and 0 <= x <= 8 for x in row) for row in counts)
        progress = [0] * 12
        completion = [None] * 12
        retained_per_cycle = []
        raw_per_cycle = []
        active_before_cycle = []
        for index, accepted in enumerate(counts, 1):
            active_before_cycle.append(sum(p < 1024 for p in progress))
            retained = 0
            for slot, count in enumerate(accepted):
                remaining = max(0, 1024 - progress[slot])
                credited = min(count, remaining)
                progress[slot] += credited
                retained += credited
                if progress[slot] == 1024 and completion[slot] is None:
                    completion[slot] = index
            retained_per_cycle.append(retained)
            raw_per_cycle.append(sum(accepted))
        assert progress == [1024] * 12 and all(v is not None for v in completion)
        assert sum(retained_per_cycle) == 12288
        # The serving Host progress mirror trails actual sampled output by one
        # cycle. Keep the observed trailing cycle; do not credit its removal.
        assert len(counts) - max(completion) == 1
        all_durations.extend(completion)
        by_rank_wall_s = [(row["serve_end_ns"] - row["serve_start_ns"]) / 1e9 for row in rows]
        cohorts.append({
            "cohort": cid,
            "cycles": len(counts),
            "count_matrix_sha256": digest,
            "all_rank_count_exact": True,
            "completion_cycle_by_slot_1_based": completion,
            "slot_busy_cycles": sum(completion),
            "slot_capacity_term_cycles_fixed_duration": math.ceil(sum(completion) / 12),
            "request_serial_chain_term_cycles_fixed_duration": max(completion),
            "fixed_cohort_lower_relaxation_cycles": max(math.ceil(sum(completion) / 12), max(completion)),
            "earliest_completion_cycle": min(completion),
            "latest_completion_cycle": max(completion),
            "cycles_after_all_1024_runtime_retained_tokens": len(counts) - max(completion),
            "raw_staged_tokens": sum(raw_per_cycle),
            "runtime_retained_output_tokens": sum(retained_per_cycle),
            "raw_minus_runtime_retained": sum(raw_per_cycle) - sum(retained_per_cycle),
            "mean_runtime_retained_tokens_per_cycle": sum(retained_per_cycle) / len(counts),
            "mean_runtime_retained_tokens_per_unfinished_slot_cycle": sum(retained_per_cycle) / sum(active_before_cycle),
            "max_rank_serve_wall_s": max(by_rank_wall_s),
            "min_rank_serve_wall_s": min(by_rank_wall_s),
            "source": f"Run239 measured boundary/rank0..7_cohort{cid}.json",
        })
    current_cycles = sum(row["cycles"] for row in cohorts)
    cross_cohort_capacity = math.ceil(sum(all_durations) / 12)
    fixed_cohort_capacity = sum(row["slot_capacity_term_cycles_fixed_duration"] for row in cohorts)
    fixed_cohort_serial = sum(row["request_serial_chain_term_cycles_fixed_duration"] for row in cohorts)
    fixed_cohort_relaxation = sum(row["fixed_cohort_lower_relaxation_cycles"] for row in cohorts)
    output = {
        "status": "saved_trace_reanalysis_no_npu",
        "scope": "Run239 instrumented 48-request measured pass, not accepted formal Run99; fixed 12-slot current handoff",
        "cohorts": cohorts,
        "aggregate": {
            "runtime_retained_output_tokens": 49152,
            "external_client_output_tokens_from_bench": 49152,
            "handoff_prior_published_tokens_by_request": None,
            "observed_target_cycles": current_cycles,
            "observed_mean_runtime_retained_tokens_per_cycle": 49152 / current_cycles,
            "fixed_cohort_slot_capacity_term_cycles": fixed_cohort_capacity,
            "fixed_cohort_request_serial_chain_term_cycles": fixed_cohort_serial,
            "fixed_cohort_duration_lower_relaxation_cycles": fixed_cohort_relaxation,
            "arbitrary_refill_slot_capacity_term_cycles": cross_cohort_capacity,
            "arbitrary_refill_request_serial_chain_term_cycles": max(all_durations),
            "arbitrary_refill_duration_lower_relaxation_cycles": max(cross_cohort_capacity, max(all_durations)),
            "current_handoff_cardinality_relaxation_cycles": 512,
            "relaxation_scope": "fixed cohorts = sum_c max(max_i observed duration, ceil(sum_i duration/12)); arbitrary refill = max(max_i duration, ceil(sum_all duration/12)). Durations fixed from Run239 only; arrival/preparation/contention omitted; neither is an attainable schedule or Product bound",
            "all_rank_count_parity": True,
            "all_48_client_length_exact": True,
        },
        "limits": [
            "The 1024-per-slot retained Runtime suffix cannot be equated with extra external client output without each Request's actual pre-handoff published token count.",
            "Observed acceptance and completion durations would change under refill or a different execution architecture.",
            "Device resource cost, prefill/seed, real request arrival and client publication are absent from these cycle relaxations.",
            "Instrumented Run239 wall and throughput are diagnostic, not formal Run99 or a hardware capacity measurement.",
        ],
    }
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"status": output["status"], **output["aggregate"]}))


if __name__ == "__main__":
    main()
