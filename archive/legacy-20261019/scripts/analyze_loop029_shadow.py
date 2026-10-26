#!/usr/bin/env python3
"""Summarize consecutive-cycle standalone state predictions by TP rank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


FIELDS = (
    "input_ids_equal",
    "positions_equal",
    "query_start_equal",
    "seq_lens_equal",
    "slot_mapping_equal",
)


def prefix_len(rows, predicate):
    count = 0
    for row in rows:
        if not predicate(row):
            break
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("shadow_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    ranks = []
    for path in sorted(args.shadow_dir.glob("rank*.jsonl")):
        rows = [
            json.loads(line)
            for line in path.read_text().splitlines()
            if line.strip()
        ]
        ranks.append(
            {
                "rank": rows[0]["rank"] if rows else None,
                "comparisons": len(rows),
                "exact": sum(bool(row["exact"]) for row in rows),
                "exact_prefix": prefix_len(rows, lambda row: bool(row["exact"])),
                "core_state_prefix": prefix_len(
                    rows,
                    lambda row: all(
                        row[field] is not False
                        for field in (
                            "input_ids_equal",
                            "positions_equal",
                            "query_start_equal",
                            "seq_lens_equal",
                        )
                    ),
                ),
                "field_matches": {
                    field: sum(row[field] is not False for row in rows)
                    for field in FIELDS
                },
                "first_failure": next(
                    (row for row in rows if not row["exact"]), None
                ),
            }
        )

    acceptance_ranks = []
    for path in sorted(args.shadow_dir.glob("accept_rank*.jsonl")):
        rows = [
            json.loads(line)
            for line in path.read_text().splitlines()
            if line.strip()
        ]
        acceptance_ranks.append(
            {
                "rank": rows[0]["rank"] if rows else None,
                "comparisons": len(rows),
                "accepted_tokens_equal": sum(
                    bool(row["accepted_tokens_equal"]) for row in rows
                ),
                "accepted_counts_equal": sum(
                    bool(row["accepted_counts_equal"]) for row in rows
                ),
                "first_failure": next(
                    (
                        row
                        for row in rows
                        if not row["accepted_tokens_equal"]
                        or not row["accepted_counts_equal"]
                    ),
                    None,
                ),
            }
        )

    result = {
        "rank_count": len(ranks),
        "total_comparisons": sum(rank["comparisons"] for rank in ranks),
        "all_exact": bool(ranks)
        and all(rank["comparisons"] > 0 and rank["exact"] == rank["comparisons"] for rank in ranks),
        "ranks": ranks,
        "acceptance_all_exact": bool(acceptance_ranks)
        and all(
            rank["comparisons"] > 0
            and rank["accepted_tokens_equal"] == rank["comparisons"]
            and rank["accepted_counts_equal"] == rank["comparisons"]
            for rank in acceptance_ranks
        ),
        "acceptance_ranks": acceptance_ranks,
        "scope": "standalone fixed-state prediction of next oracle target input; real weights/operators remain oracle-owned",
        "performance_use": "diagnostic only",
    }
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
