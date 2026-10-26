#!/usr/bin/env python3
"""Reduce Run656's same-W0 all-rank Host-submit skew without a live service."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

LABELS = (
    "cycle_begin", "target_before", "target_after",
    "proposer_before", "proposer_after",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def median_p95(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "median": statistics.median(ordered),
        "p95": ordered[int(0.95 * (len(ordered) - 1))],
    }


def reduce(root: Path) -> dict:
    files = sorted(root.glob("rank*_cohort*.json"))
    if len(files) != 32:
        raise ValueError(f"expected 32 packet files, got {len(files)}")
    packets = {}
    hashes = {}
    for path in files:
        packet = json.loads(path.read_text())
        ident = packet["identity"]
        key = (ident["cohort"], ident["rank"])
        if key in packets or not (5 <= key[0] <= 8 and 0 <= key[1] < 8):
            raise ValueError(f"duplicate or invalid rank/cohort: {key}")
        if packet["status"] != "instrumented_current_stream_full_cohort":
            raise ValueError(f"status drift: {path}")
        cycles = ident["cycles"]
        if len(packet["class_rows"]) != cycles:
            raise ValueError(f"class rows drift: {path}")
        events = packet["events"]
        if len(events) != 5 * cycles + 1:
            raise ValueError(f"event count drift: {path}")
        by_cycle = []
        for c in range(cycles):
            window = events[5*c:5*c+5]
            if [e["label"] for e in window] != list(LABELS):
                raise ValueError(f"label order drift: {path} cycle {c}")
            expected_generation = 2 + sum(
                packets[(prior, ident["rank"])][0]["identity"]["cycles"] > c
                for prior in range(5, ident["cohort"])
            )
            if any(e["cycle"] != c or e["generation"] != expected_generation for e in window):
                raise ValueError(f"generation/cycle drift: {path} cycle {c}")
            times = [e["host_submit_ns"] for e in window]
            if times != sorted(times):
                raise ValueError(f"Host time reversed: {path} cycle {c}")
            by_cycle.append({e["label"]: e["host_submit_ns"] for e in window})
        packets[key] = (packet, by_cycle)
        hashes[str(path)] = sha(path)
    spreads = {label: [] for label in LABELS}
    latest = {label: Counter() for label in LABELS}
    deltas = {"target_phase": [], "proposer_phase": [], "intercycle": []}
    latest_switches = {"proposer_to_next_target": 0, "target_to_proposer": 0}
    cohorts = {}
    for cohort in range(5, 9):
        group = [packets[(cohort, rank)] for rank in range(8)]
        ids = [p[0]["identity"] for p in group]
        keys = (
            "time_namespace", "run_ts", "req_ids", "cycles",
            "canonical_effective_staged_trajectory_sha256",
            "count_history_sha256", "raw_padded_token_history_sha256",
            "runtime_bulk_output_sha256",
        )
        for key in keys:
            if any(row[key] != ids[0][key] for row in ids[1:]):
                raise ValueError(f"all-rank identity mismatch: cohort {cohort} {key}")
        cycles = ids[0]["cycles"]
        cohorts[str(cohort)] = {"cycles": cycles, "run_ts": ids[0]["run_ts"]}
        for c in range(cycles):
            if c < 8 or c + 1 >= cycles:
                continue
            if any(any(row["parked_before"] or row["parked_after"]
                       for row in p[0]["class_rows"][c:c+2]) for p in group):
                continue
            selected = {}
            for label in LABELS:
                times = [p[1][c][label] for p in group]
                spread = (max(times) - min(times)) / 1e6
                spreads[label].append(spread)
                rank = max(range(8), key=lambda r: times[r])
                latest[label][rank] += 1
                selected[label] = (spread, rank)
            next_times = [p[1][c+1]["target_before"] for p in group]
            next_spread = (max(next_times) - min(next_times)) / 1e6
            next_rank = max(range(8), key=lambda r: next_times[r])
            deltas["target_phase"].append(
                selected["target_after"][0] - selected["target_before"][0])
            deltas["proposer_phase"].append(
                selected["proposer_after"][0] - selected["proposer_before"][0])
            deltas["intercycle"].append(next_spread - selected["proposer_after"][0])
            latest_switches["target_to_proposer"] += (
                selected["target_after"][1] != selected["proposer_after"][1])
            latest_switches["proposer_to_next_target"] += (
                selected["proposer_after"][1] != next_rank)
    n = len(deltas["intercycle"])
    if n < 500:
        raise ValueError(f"too few steady unparked adjacent cycles: {n}")
    return {
        "status": "same_w0_host_submit_skew_only",
        "source": "Run656 ON full48 32 admitted packets",
        "input_sha256": hashes,
        "cohorts": cohorts,
        "steady_unparked_adjacent_cycles": n,
        "spread_ms": {k: median_p95(v) for k, v in spreads.items()},
        "spread_change_ms": {k: median_p95(v) for k, v in deltas.items()},
        "latest_rank_counts": {
            k: {str(rank): counts[rank] for rank in range(8)}
            for k, counts in latest.items()
        },
        "latest_rank_switches": latest_switches,
        "limits": (
            "Cross-rank monotonic Host-submit markers only. They precede Event "
            "execution and do not establish device producer readiness, collective "
            "completion, intrinsic service, legal schedule savings, or formal E2E."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = reduce(args.packet_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "input_sha256"}))


if __name__ == "__main__":
    main()
