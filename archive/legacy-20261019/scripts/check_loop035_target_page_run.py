"""Check Run80 eight-rank target page snapshot candidate coverage."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def check(root: Path) -> dict:
    bench = json.loads((root / "bench.json").read_text())
    if bench["summary"]["success"] != 12 or bench["summary"]["fail"] != 0:
        raise ValueError("client completion gate failed")
    if any(row["output_tokens"] != 1024 for row in bench["requests"]):
        raise ValueError("client output length gate failed")
    rows = []
    runtime = []
    for rank in range(8):
        page = json.loads((root / "pages" / f"rank{rank}.json").read_text())
        if page["rank"] != rank or len(page["rows"]) != 2:
            raise ValueError(f"rank {rank} page rows incomplete")
        if [record["cycle"] for record in page["rows"]] != [0, 1]:
            raise ValueError(f"rank {rank} page cycle order invalid")
        for record in page["rows"]:
            if (record["source_layers"] != 170 or record["cache_views"] != 67
                    or record["cache_views_snapshotted"] != 67
                    or record["cache_views_certified_no_write"] != 0
                    or record["skipped"]):
                raise ValueError(f"rank {rank} page coverage incomplete")
        rows.append(page["rows"])
        record = json.loads((root / "runtime" / f"rank{rank}_cohort1.json").read_text())
        if (not record["pass"] or not record["host_mirror_exact"]
                or record["target_graph_mode"] != "FULL"
                or record["oracle_target_calls_after_handoff"] != 0
                or record["model_runner_cycles_after_handoff"] != 0):
            raise ValueError(f"rank {rank} runtime gate failed")
        runtime.append(record)
    structural = [
        [{key: value for key, value in row.items() if key != "rank"}
         for row in rank_rows]
        for rank_rows in rows
    ]
    if any(item != structural[0] for item in structural[1:]):
        raise ValueError("eight rank page structures differ")
    kinds = Counter(source["kind"] for source in rows[0][0]["sources"])
    zero_sources = [
        sum(source["physical_pages"] == 0 for source in row["sources"])
        for row in rows[0]
    ]
    return {
        "rank_page_records": 8,
        "cycles_checked": [0, 1],
        "rank_structure_equal": True,
        "source_layers": 170,
        "source_kinds": dict(kinds),
        "zero_write_source_layers_by_cycle": zero_sources,
        "cache_views_snapshotted_by_cycle": [row["cache_views_snapshotted"]
                                             for row in rows[0]],
        "candidate_page_entries_by_cycle": [row["page_count_sum"]
                                             for row in rows[0]],
        "skipped": 0,
        "clients_exact_1024": 12,
        "runtime_rank_gates": len(runtime),
        "rank0_cycles": runtime[0]["cycles"],
        "scope": (
            "source-derived pre-target page candidates and snapshot execution; "
            "no post-target restore, KV value parity, Stock token oracle, or formal E2E comparison"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = check(args.root)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

