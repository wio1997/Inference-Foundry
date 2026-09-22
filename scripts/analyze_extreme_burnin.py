#!/usr/bin/env python3
"""Validate and summarize a long fixed-cycle Extreme Runtime window."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = [json.loads((args.run_dir / f"rank{rank}.json").read_text())
            for rank in range(8)]
    cycles = {row["cycles"] for row in rows}
    final_state = {tuple(row["final_num_computed_tokens"]) for row in rows}
    emitted_state = {tuple(row["emitted_token_count"]) for row in rows}
    walls = [float(row["profiled_wall_seconds"]) for row in rows]
    rates = [float(row["emitted_tokens_per_second"]) for row in rows]
    report = {
        "pass": (
            all(row["pass"] for row in rows)
            and all(row["state_advance_exact"] for row in rows)
            and all(row["host_mirror_exact"] for row in rows)
            and len(cycles) == len(final_state) == len(emitted_state) == 1
        ),
        "rank_count": len(rows),
        "cycles": sorted(cycles),
        "all_rank_state_equal": len(final_state) == len(emitted_state) == 1,
        "host_mirror_exact": all(row["host_mirror_exact"] for row in rows),
        "wall_seconds": {
            "median": statistics.median(walls),
            "min": min(walls),
            "max": max(walls),
        },
        "cycle_wall_ms_median": 1000 * statistics.median(walls) / next(iter(cycles)),
        "emitted_tokens": sum(rows[0]["emitted_token_count"]),
        "emitted_tokens_per_second": {
            "median": statistics.median(rates),
            "min": min(rates),
            "max": max(rates),
        },
        "comparability": (
            "Decode-window metric after one-time vLLM bootstrap. It excludes prefill, HTTP, "
            "request refill and response publication; do not compare it directly to the frozen "
            "543.65 tok/s 48x32K-to-1024 E2E baseline."
        ),
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
