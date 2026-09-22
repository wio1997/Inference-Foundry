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
    summary = {
        "rank_files": len(rows),
        "error_files": len(errors),
        "all_pass": len(rows) == 8 and not errors and all(row["pass"] for row in rows),
        "cycles": sorted({row.get("cycles") for row in rows}),
        "model_runner_retained": sorted(
            {row.get("model_runner_retained") for row in rows}
        ),
        "proposer_runner_types": sorted(
            {row.get("proposer_runner_type") for row in rows}
        ),
        "cache_tensor_counts": sorted(
            {row.get("cache_tensors") for row in rows}
        ),
        "snapshotted_cache_tensor_counts": sorted(
            {row.get("snapshotted_cache_tensors") for row in rows}
        ),
        "errors": errors,
    }
    output = args.run_dir / "summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
