#!/usr/bin/env python3
"""Run630: paired formal E2E scope accounting and fixed-work saving hurdles.

This is an arithmetic sensitivity, not a Resource, Scheduling, or Product bound.
It never turns an unclassified scope difference into removable Host time.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path("/data/wio/Inference_Foundry")
SOURCES = {
    "formal_scope": ROOT / "evidence/20260925_loop044_target/run153/e2e_accounting.json",
    "formal_wave": ROOT / "evidence/20260926_loop058_bound/run238/wave_audit.json",
    "bound_model": ROOT / "evidence/20260928_loop081_bound/run625/bound_calibration_v3_44.json",
}
OUT = ROOT / "evidence/20260928_loop081_bound/run630/product_hurdle.json"
N = 49152
LANDMARKS = (581, 607, 616, 682)


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a: float, b: float, tol: float = 1e-7) -> None:
    if abs(a - b) > tol:
        raise AssertionError((a, b))


def main() -> None:
    s = load(SOURCES["formal_scope"])
    w = load(SOURCES["formal_wave"])
    m = load(SOURCES["bound_model"])
    by_wave = {r["formal_run"]: r for r in w["runs"]}
    assert len(s["repeats"]) == len(by_wave) == 3
    assert [r["repeat"] for r in s["repeats"]] == [1, 2, 3]
    assert [r["formal_run"] for r in w["runs"]] == [1, 2, 3]
    assert s["source"].startswith("Run99") and w["source"].startswith("Run99")
    assert m["current"]["accepted_formal_tps"] == 571.681

    paired = []
    for row in s["repeats"]:
        idx = row["repeat"]
        wave = by_wave[idx]
        close(row["client_duration_s"], wave["duration_s"])
        close(row["client_duration_s"], m["current"]["formal_repeat_client_wall_s"][idx - 1])
        close(sum(c["runtime_wall_s"] for c in row["cohorts"]), row["runtime_sum_s"])
        close(row["client_duration_s"] - row["runtime_sum_s"], row["difference_s"])
        assert len(row["cohorts"]) == len(wave["waves"]) == 4
        for c, q in zip(row["cohorts"], wave["waves"]):
            assert c["runtime_cohort_id"] == q["cohort"]
            close(c["request_envelope_s"], q["client_wall_s"])
            close(c["runtime_wall_s"], q["rank0_decode_wall_s"])
            assert c["cycles"] == q["cycles"]
        assert sum(c["cycles"] for c in row["cohorts"]) == m["current"]["formal_repeat_cycles"][idx - 1]
        paired.append({
            "repeat": idx,
            "client_wall_s": row["client_duration_s"],
            "rank0_runtime_scope_sum_s": row["runtime_sum_s"],
            "arithmetic_remainder_s": row["difference_s"],
            "arithmetic_remainder_fraction": row["difference_s"] / row["client_duration_s"],
            "runtime_cycles": sum(c["cycles"] for c in row["cohorts"]),
            "wave_cohort_ids": [c["runtime_cohort_id"] for c in row["cohorts"]],
            "sum_client_wave_envelopes_minus_full_client_s": (
                sum(c["request_envelope_s"] for c in row["cohorts"])
                - row["client_duration_s"]
            ),
        })

    median = sorted(paired, key=lambda r: r["client_wall_s"])[1]
    rounded_tps_implied_wall = N / m["current"]["accepted_formal_tps"]
    close(rounded_tps_implied_wall, m["current"]["implied_wall_s_for_49152_tokens"])
    # The reported formal TPS is rounded; its implied wall differs slightly
    # from the saved median repeat. Use the exact paired repeat in hurdles.
    baseline_wall = median["client_wall_s"]
    assert abs(rounded_tps_implied_wall - baseline_wall) < 0.001
    landmarks = []
    for p in LANDMARKS:
        desired = N / p
        saved = baseline_wall - desired
        assert saved > 0
        landmarks.append({
            "tps_landmark_not_bound": p,
            "required_product_wall_s": desired,
            "hypothetical_total_wall_saving_from_formal_median": saved,
            "saving_as_fraction_of_paired_median_runtime_scope": saved / median["rank0_runtime_scope_sum_s"],
            "saving_as_fraction_of_paired_median_arithmetic_remainder": saved / median["arithmetic_remainder_s"],
            "uniform_if_all_paired_median_repeat_cycles_ms": 1000 * saved / median["runtime_cycles"],
            "scope": "counterfactual wall saving, not an identified available phase or attainable TPS",
        })

    delta = {
        "client_wall_range_s": max(r["client_wall_s"] for r in paired) - min(r["client_wall_s"] for r in paired),
        "runtime_scope_range_s": max(r["rank0_runtime_scope_sum_s"] for r in paired) - min(r["rank0_runtime_scope_sum_s"] for r in paired),
        "arithmetic_remainder_range_s": max(r["arithmetic_remainder_s"] for r in paired) - min(r["arithmetic_remainder_s"] for r in paired),
        "scope": "three formal samples only; ranges are not a variance model and the remainder is not named Host overhead",
    }
    result = {
        "status": "formal_paired_scope_arithmetic_only_no_bound",
        "contract": "fixed DSpark7 acceptance/cycles/output/model work; 8x910B3 DP1TP8 48x32K->1024 c12",
        "source_sha256": {k: sha(v) for k, v in SOURCES.items()},
        "formal_current_tps": m["current"]["accepted_formal_tps"],
        "paired_median_repeat": median["repeat"],
        "paired_median_cycles": median["runtime_cycles"],
        "paired_median_exact_client_wall_s": baseline_wall,
        "reported_rounded_tps_implied_wall_s": rounded_tps_implied_wall,
        "rounding_wall_difference_s": rounded_tps_implied_wall - baseline_wall,
        "paired_repeats": paired,
        "sample_range": delta,
        "scenario_landmark_hurdles": landmarks,
        "interpretation": [
            "Run153 and Run238 pair the same Run99 formal repeat/cohort records; each row is checked against V3.44 current cycles and wall.",
            "The arithmetic remainder is client wall minus rank0 FixedCohortServing Runtime scope sum. It is not a clock-certified interval complement: Run99 has no server absolute timestamps or client request-ID token join. Candidate unclassified contributors include admission, prefill, Runtime construction, synchronization and publication, with possible overlap/clock-boundary effects.",
            "Four client wave envelopes do not partition full client wall: their summed durations exceed it by roughly 7 ms per repeat. Do not add wave-wise residuals as mutually exclusive stages.",
            "The 11–17 s remainder is not a measured removable Host gap or an achievable counterfactual; likewise the approximately 69 s scoped Runtime is not a compulsory floor.",
            "Run602/610 stage timestamps are observer-perturbed different W0s and cannot fill Run99 remainder or per-node costs.",
            "LANDMARKS are old scenario rates used solely to express required whole-Product wall savings. They are neither Hardware ceilings nor stopping targets.",
        ],
        "bound_endpoints": {
            "resource_hardware_floor_s": None,
            "scheduling_execution_floor_s": None,
            "product_e2e_ceiling_tps": None,
            "numeric_current_to_credible_limit_gap": None,
        },
        "next_measurement": "Build an exact same-W0 all8/client low-overhead Product phase completion ledger with common clock/causal markers for admission, prefill/seed, Runtime construction, cycles, existing sync and publication; retain W0/Basis/Product joins and OFF/ON/OFF transfer. Classify the remainder before ranking a selected KV row packet as the main whole-Product bound experiment.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": result["status"], "output": str(OUT), "sha256": sha(OUT), "repeats": len(paired)}))


if __name__ == "__main__":
    main()
