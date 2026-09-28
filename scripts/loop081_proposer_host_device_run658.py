#!/usr/bin/env python3
"""Run656 same-W0 proposer Host-call versus current-stream duration spread."""
from __future__ import annotations
import argparse
import hashlib
import json
import statistics
from pathlib import Path


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[int(fraction * (len(ordered) - 1))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packet-dir", type=Path, required=True)
    ap.add_argument("--pin", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    pin = json.loads(args.pin.read_text())["input_sha256"]
    packets = {}
    for path in args.packet_dir.glob("rank*_cohort*.json"):
        if str(path) not in pin or hashlib.sha256(path.read_bytes()).hexdigest() != pin[str(path)]:
            raise ValueError(f"Run657 input pin mismatch: {path}")
        data = json.loads(path.read_text())
        ident = data["identity"]
        packets[(ident["cohort"], ident["rank"])] = data
    if len(packets) != len(pin) != 32:
        raise ValueError("incomplete all8 full48 packets")
    rows = []
    for cohort in range(5, 9):
        ncycles = packets[cohort, 0]["identity"]["cycles"]
        for cycle in range(8, ncycles - 1):
            group = [packets[cohort, rank] for rank in range(8)]
            if any(any(row["parked_before"] or row["parked_after"]
                       for row in packet["class_rows"][cycle:cycle+2])
                   for packet in group):
                continue
            values = []
            for packet in group:
                segment = packet["events"][5*cycle:5*cycle+5]
                marks = {row["label"]: row for row in segment}
                a, b = marks["proposer_before"], marks["proposer_after"]
                host_duration = (b["host_submit_ns"] - a["host_submit_ns"]) / 1e6
                event_duration = b["elapsed_ms"] - a["elapsed_ms"]
                if host_duration < 0 or event_duration < 0:
                    raise ValueError("negative proposer duration")
                values.append((host_duration, event_duration, b["host_submit_ns"]))
            latest = max(range(8), key=lambda rank: values[rank][2])
            longest_event = max(value[1] for value in values)
            rows.append({
                "host_duration_range_ms": max(x[0] for x in values) - min(x[0] for x in values),
                "event_duration_range_ms": longest_event - min(x[1] for x in values),
                "latest_host_event_gap_ms": longest_event - values[latest][1],
            })
    if len(rows) != 714:
        raise ValueError(f"Run657 stratum drift: {len(rows)}")
    result = {
        "status": "same_w0_proposer_host_vs_current_stream_diagnostic",
        "source_pin_sha256": hashlib.sha256(args.pin.read_bytes()).hexdigest(),
        "n": len(rows),
        "host_duration_range_ms": {
            "median": statistics.median(x["host_duration_range_ms"] for x in rows),
            "p95": percentile([x["host_duration_range_ms"] for x in rows], .95),
        },
        "event_duration_range_ms": {
            "median": statistics.median(x["event_duration_range_ms"] for x in rows),
            "p95": percentile([x["event_duration_range_ms"] for x in rows], .95),
        },
        "latest_host_rank_has_longest_event_duration_exact": sum(
            x["latest_host_event_gap_ms"] == 0 for x in rows),
        "latest_host_rank_within_5us_of_longest_event_duration": sum(
            x["latest_host_event_gap_ms"] <= .005 for x in rows),
        "limits": (
            "Per-rank Event durations are on each rank current stream and include queue/wait. "
            "This is not cross-rank device timeline alignment, all-stream completion, "
            "intrinsic DSpark service, exposed wall, removable Host time or formal E2E."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
