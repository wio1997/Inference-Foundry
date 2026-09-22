#!/usr/bin/env python3
"""Compare two Extreme Runtime profile summaries without mixing overlaps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before = json.loads(args.before.read_text())
    after = json.loads(args.after.read_text())
    names = sorted(
        set(before["scope_steady_cycles"]) | set(after["scope_steady_cycles"])
    )
    scopes = {}
    for name in names:
        old = before["scope_steady_cycles"].get(name, {})
        new = after["scope_steady_cycles"].get(name, {})
        old_median = old.get("median_ms")
        new_median = new.get("median_ms")
        scopes[name] = {
            "before_median_ms": old_median,
            "after_median_ms": new_median,
            "median_delta_ms": (
                new_median - old_median
                if old_median is not None and new_median is not None
                else None
            ),
            "before_p90_ms": old.get("p90_ms"),
            "after_p90_ms": new.get("p90_ms"),
        }
    old_cycle = before["scope_steady_cycles"]["extreme::cycle"]["median_ms"]
    new_cycle = after["scope_steady_cycles"]["extreme::cycle"]["median_ms"]
    report = {
        "before": str(args.before),
        "after": str(args.after),
        "comparability": (
            "Same model/hardware/fixed c12/8 cycles and profiler configuration; "
            "prompt-to-slot ordering and accepted-token counts differ, so this is "
            "a structural attribution comparison, not a formal E2E throughput A/B."
        ),
        "cycle_median_ms": {
            "before": old_cycle,
            "after": new_cycle,
            "delta": new_cycle - old_cycle,
            "percent": 100 * (new_cycle / old_cycle - 1),
        },
        "scopes": scopes,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
