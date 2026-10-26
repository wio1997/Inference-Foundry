#!/usr/bin/env python3
"""Summarize the eight-rank real Extreme Runtime handoff run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    rows = []
    errors = []
    for path in sorted(args.run_dir.glob("rank*.json")):
        value = json.loads(path.read_text())
        (errors if path.name.endswith("_error.json") else rows).append(value)
    state_keys = (
        "cycles",
        "cycle_index",
        "stage_order",
        "handoff_before_model_forward",
        "oracle_target_calls_after_handoff",
        "model_runner_retained",
        "proposer_runner_type",
        "cache_tensors",
        "snapshotted_cache_tensors",
        "accepted_counts",
        "emitted_token_count",
        "final_num_computed_tokens",
        "state_advance_exact",
        "acceptance_counts_in_contract",
        "target_validation",
        "first_acceptance_equal",
        "pass",
    )
    rank_state_equal = bool(rows) and all(
        {key: row.get(key) for key in state_keys}
        == {key: rows[0].get(key) for key in state_keys}
        for row in rows
    )
    summary = {
        "rank_files": len(rows),
        "error_files": len(errors),
        "all_pass": len(rows) == 8 and not errors and all(row["pass"] for row in rows),
        "all_rank_state_equal": rank_state_equal,
        "cycles": sorted({row.get("cycles") for row in rows}),
        "handoff_before_model_forward": sorted(
            {row.get("handoff_before_model_forward") for row in rows}
        ),
        "oracle_target_calls_after_handoff": sorted(
            {row.get("oracle_target_calls_after_handoff") for row in rows}
        ),
        "model_runner_retained": sorted(
            {row.get("model_runner_retained") for row in rows}
        ),
        "proposer_runner_types": sorted(
            {row.get("proposer_runner_type") for row in rows}
        ),
        "cache_tensor_counts": sorted(
            {row.get("cache_tensors") for row in rows}
        ),
        "state_advance_exact": sorted(
            {row.get("state_advance_exact") for row in rows}
        ),
        "errors": errors,
    }
    if rows:
        summary["accepted_total"] = sum(
            sum(cycle) for cycle in rows[0]["accepted_counts"]
        )
        summary["emitted_total"] = sum(rows[0]["emitted_token_count"])
    output = args.run_dir / "summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
