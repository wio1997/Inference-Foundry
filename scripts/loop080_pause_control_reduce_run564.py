#!/usr/bin/env python3
"""Compare admitted Run564 A0/B/A1 traces without claiming complete W0."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


FIELDS = (
    "num_computed_before", "last_token_before", "draft_before",
    "target_input_ids", "target_positions", "target_argmax",
    "accepted", "counts", "next_draft",
)


def first_difference(left, right):
    for cycle in range(max(len(left), len(right))):
        if cycle >= len(left) or cycle >= len(right):
            return {"cycle": cycle, "field": "trace_length"}
        for field in FIELDS:
            if left[cycle][field] != right[cycle][field]:
                return {"cycle": cycle, "field": field}
    return None


def reduce_run(root: Path, cohort: int):
    arms = {name: root / name for name in ("a0", "b", "a1")}
    admissions = {
        name: json.loads((arm / "arm_admission.json").read_text())
        for name, arm in arms.items()
    }
    assert all(row["status"] == "diagnostic_arm_admitted" for row in admissions.values())
    assert [admissions[name]["mode"] for name in arms] == ["off", "on", "off"]
    assert all(row["cohort"] == cohort for row in admissions.values())
    for name, arm in arms.items():
        for field, size in (("trace_file_sha256", 32),
                            ("runtime_file_sha256", 64),
                            ("pause_file_sha256", 8 if name == "b" else 0)):
            hashes = admissions[name][field]
            assert len(hashes) == size
            for relative, expected in hashes.items():
                path = arm / relative
                assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, \
                    f"{name} {field} changed after admission: {relative}"
        for filename, field in (("client_admission.json", "client_admission_sha256"),
                                ("server_post_count.txt", "server_post_count_sha256")):
            assert hashlib.sha256((arm / filename).read_bytes()).hexdigest() == admissions[name][field]
    target_keys = admissions["b"]["request_semantic_keys_by_cohort"][str(cohort)]
    matched = {"b": cohort}
    for name in ("a0", "a1"):
        options = [int(key) for key, keys in
                   admissions[name]["request_semantic_keys_by_cohort"].items()
                   if keys == target_keys]
        matched[name] = options[0] if len(options) == 1 else None
    semantic_match = all(matched[name] is not None for name in arms)
    differences = {}
    if semantic_match:
        for rank in range(8):
            traces = {
                name: json.loads((arm / "trace" /
                                  f"trace_rank{rank}_cohort{matched[name]}.json").read_text())
                for name, arm in arms.items()
            }
            differences[str(rank)] = {
                "a0_vs_a1": first_difference(traces["a0"], traces["a1"]),
                "a0_vs_b": first_difference(traces["a0"], traces["b"]),
                "a1_vs_b": first_difference(traces["a1"], traces["b"]),
            }
    a_controls_equal = semantic_match and all(
        item["a0_vs_a1"] is None for item in differences.values())
    b_observed_equal = a_controls_equal and all(
        item["a0_vs_b"] is None and item["a1_vs_b"] is None
        for item in differences.values())
    wall_seconds = {}
    if semantic_match:
        for name, arm in arms.items():
            wall_seconds[name] = [
                json.loads((arm / "runtime" /
                            f"rank{rank}_cohort{matched[name]}.json").read_text())["wall_seconds"]
                for rank in range(8)
            ]
    wall_medians = {name: statistics.median(values)
                    for name, values in wall_seconds.items()}
    control_median = statistics.median((wall_medians["a0"], wall_medians["a1"])) \
        if semantic_match else None
    b_minus_control_seconds = wall_medians["b"] - control_median \
        if semantic_match else None
    return {
        "status": "diagnostic_reduced",
        "cohort": cohort,
        "semantic_request_order_match": semantic_match,
        "matched_cohorts": matched,
        "a0_a1_observed_trace_equal": a_controls_equal,
        "b_observed_trace_equal_to_controls": b_observed_equal,
        "first_differences": differences,
        "matched_cohort_wall_seconds_by_rank": wall_seconds,
        "matched_cohort_wall_median_seconds": wall_medians,
        "b_minus_control_median_seconds": b_minus_control_seconds,
        "timing_interpretation": "descriptive A0/B/A1 only; compare paired workload, noise, and full fixed-W0 witness before any bound update",
        "scope": "existing per-cycle observed fields only; equality does not certify complete KV/Draft-context/state fixed-W0 or a time bound",
        "bound_promotion": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cohort", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = reduce_run(args.root, args.cohort)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
