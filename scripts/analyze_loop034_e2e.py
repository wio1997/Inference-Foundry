#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def main() -> None:
    root = Path(sys.argv[1])
    runs = [
        json.loads((root / f"bench48_{index}.json").read_text())
        for index in range(1, 4)
    ]
    summaries = [run["summary"] for run in runs]
    rank_files = sorted((root / "runtime").glob("rank*_cohort*.json"))
    rank_rows = [json.loads(path.read_text()) for path in rank_files]
    tps = [float(row["output_tps"]) for row in summaries]
    result = {
        "pass": all(
            row["success"] == 48
            and row["fail"] == 0
            and all(
                req["output_tokens"] == 1024 and req["error"] is None
                for req in run["requests"]
            )
            for row, run in zip(summaries, runs)
        )
        and bool(rank_rows)
        and all(row["pass"] for row in rank_rows),
        "runs": summaries,
        "output_tps": {
            "samples": tps,
            "median": statistics.median(tps),
            "stock_median": 543.6546,
            "delta_percent": (
                statistics.median(tps) / 543.6546 - 1.0
            ) * 100.0,
        },
        "runtime_rank_files": len(rank_rows),
        "runtime_cohorts": sorted(
            {int(row["cohort"]) for row in rank_rows}
        ),
        "runtime_ranks": sorted({int(row["rank"]) for row in rank_rows}),
        "all_runtime_rows_pass": bool(rank_rows)
        and all(row["pass"] for row in rank_rows),
    }
    (root / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
