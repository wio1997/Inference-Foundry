#!/usr/bin/env python3
"""Conditional formal-workload replay for DeepSeek Extreme P0.

This is a calibrated counterfactual calculator, not an identified hardware
upper bound. All scenario savings are assumed *exposed* on the E2E path.
"""
import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN99 = ROOT / "evidence/20260924_loop036_metadata/run99"
OUTPUT_TOKENS = 48 * 1024
COHORTS = ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16))
SCENARIOS = {
    "engineering_conservative": dict(target_ms=1.0, other_decode_ms=0.0, prefill_s_per_cohort=0.05),
    "engineering_optimistic": dict(target_ms=3.0, other_decode_ms=0.5, prefill_s_per_cohort=0.20),
    "aggressive_conservative": dict(target_ms=4.0, other_decode_ms=0.5, prefill_s_per_cohort=0.20),
    "aggressive_optimistic": dict(target_ms=8.0, other_decode_ms=1.5, prefill_s_per_cohort=0.60),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    formal = json.loads((RUN99 / "summary.json").read_text())
    assert formal["pass"] and len(formal["runs"]) == 3
    current = []
    for run_index, cohort_ids in enumerate(COHORTS):
        cohort_rows = []
        for cid in cohort_ids:
            path = RUN99 / f"runtime/rank0_cohort{cid}.json"
            row = json.loads(path.read_text())
            assert row["pass"] and len(row["generated_output_counts"]) == 12
            assert all(n == 1024 for n in row["generated_output_counts"])
            cohort_rows.append(dict(cohort=cid, cycles=row["cycles"],
                                    decode_wall_s=row["wall_seconds"],
                                    output_tokens=sum(row["generated_output_counts"]),
                                    useful_tokens_per_slot_cycle=1024 / row["cycles"]))
        r = formal["runs"][run_index]
        duration = r["duration_s"]
        assert r["success"] == 48 and r["output_tps"] > 0
        assert abs(OUTPUT_TOKENS / duration - r["output_tps"]) < 1e-6
        current.append(dict(run=run_index + 1, duration_s=duration,
                            tps=r["output_tps"], cycles=sum(c["cycles"] for c in cohort_rows),
                            summed_rank0_decode_wall_s=sum(c["decode_wall_s"] for c in cohort_rows),
                            cohorts=cohort_rows))
    predictions = {}
    for name, s in SCENARIOS.items():
        rows = []
        for base in current:
            decode_saving_s = base["cycles"] * (s["target_ms"] + s["other_decode_ms"]) / 1000
            prefill_saving_s = 4 * s["prefill_s_per_cohort"]
            predicted_duration = base["duration_s"] - decode_saving_s - prefill_saving_s
            assert predicted_duration > 0
            rows.append(dict(run=base["run"], cycles=base["cycles"],
                             decode_saving_s=decode_saving_s, prefill_saving_s=prefill_saving_s,
                             duration_s=predicted_duration, tps=OUTPUT_TOKENS / predicted_duration))
        predictions[name] = dict(assumptions=s, runs=rows,
                                 median_tps=sorted(r["tps"] for r in rows)[1])
    out = dict(schema_version=1, contract="48x32K->1024 c12 warm DP1TP8 DSpark7",
               status="conditional_counterfactual_not_identified_hardware_bound",
               current=current, current_median_tps=sorted(r["tps"] for r in current)[1],
               predictions=predictions,
               limits=["Formal client duration is the only E2E anchor; rank0 cohort wall is a consistency check, not additive decomposition.",
                       "Scenario milliseconds are assumed exposed savings and remain unverified by intervention.",
                       "Cycles and useful tokens are held fixed; changed acceptance, queueing, cache hit, or prefill scheduling require a new replay.",
                       "Run98/107 kernel/stage timing and one-card bandwidth are diagnostic constraints, never summed into E2E."],
               sources=[str(RUN99 / "summary.json"), str(RUN99 / "runtime/rank0_cohort*.json")])
    Path(args.output).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"current_median_tps": out["current_median_tps"],
                      "cycles": [x["cycles"] for x in current],
                      "scenario_medians": {k: v["median_tps"] for k, v in predictions.items()}}, indent=2))


if __name__ == "__main__":
    main()
