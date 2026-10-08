#!/usr/bin/env python3
"""Analyze completed CG_DISPATCH_DIAG logs without importing vLLM.

The input files should be stable completed captures. This script streams each
file and aligns records by the per-process `step` counter and DP rank.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

MARKER = "CG_DISPATCH_DIAG "
SCALARS = (
    "ts_ns",
    "pid",
    "dp",
    "tp",
    "step",
    "tokens",
    "local_padded",
    "reqs",
    "max_sched",
    "uniform_query_len",
    "is_all_decode",
    "uniform",
    "local_mode",
    "dp_state",
    "final_mode",
)
LITERALS = ("req_ids", "computed_tokens", "dp_tokens", "dp_modes")
SCALAR_RE = {
    key: re.compile(rf"(?:^|\s){key}=([^\s]+)") for key in SCALARS
}


def literal_value(line: str, key: str) -> Any:
    marker = f"{key}="
    start = line.find(marker)
    if start < 0:
        return None
    value_start = start + len(marker)
    if value_start >= len(line) or line[value_start] != "[":
        return None

    depth = 0
    quote: str | None = None
    escaped = False
    for end in range(value_start, len(line)):
        char = line[end]
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in ("'", '"'):
            quote = char
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return ast.literal_eval(line[value_start : end + 1])
    return None


def parse_record(line: str, source: str, line_number: int) -> dict[str, Any] | None:
    if MARKER not in line:
        return None
    record: dict[str, Any] = {"source": source, "line": line_number}
    for key, pattern in SCALAR_RE.items():
        match = pattern.search(line)
        if match is None:
            record[key] = None
            continue
        value = match.group(1)
        if key in {"is_all_decode", "uniform"}:
            record[key] = value == "True"
        elif key in {"local_mode", "dp_state", "final_mode"}:
            record[key] = value
        elif key == "uniform_query_len" and value == "None":
            record[key] = None
        else:
            try:
                record[key] = int(value)
            except ValueError:
                record[key] = value
    for key in LITERALS:
        try:
            record[key] = literal_value(line, key)
        except (ValueError, SyntaxError):
            record[key] = None
    if any(record.get(key) is None for key in ("dp", "tp", "step", "local_mode", "final_mode")):
        record["parse_error"] = "missing required field"
    return record


def classify(record: dict[str, Any]) -> str:
    req_ids = record.get("req_ids")
    req_count = record.get("reqs")
    ids_present = isinstance(req_ids, list) and len(req_ids) == req_count and all(
        isinstance(req_id, str) and req_id for req_id in req_ids
    )
    dummy_markers = ("dummy", "warmup", "profile", "capture", "__dummy__")
    if ids_present and any(any(mark in req_id.lower() for mark in dummy_markers) for req_id in req_ids):
        return "dummy_id_likely"
    if record.get("uniform") is True and record.get("is_all_decode") is False:
        return "forced_uniform_likely"
    if req_count == 0 or not req_ids:
        return "no_request_ids_dummy_or_empty"
    if ids_present:
        return "request_ids_present_phase_unverified"
    return "request_ids_incomplete_or_unavailable"


def read_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for line_number, line in enumerate(stream, start=1):
                record = parse_record(line, str(path), line_number)
                if record is not None:
                    record["batch_class"] = classify(record)
                    records.append(record)
    return records


def analyze(records: list[dict[str, Any]], expected_dp: int) -> dict[str, Any]:
    by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("tp") == 0 and isinstance(record.get("step"), int):
            by_step[record["step"]].append(record)

    steps = []
    for step, group in sorted(by_step.items()):
        group = sorted(group, key=lambda row: row["dp"])
        local_none = [row["dp"] for row in group if row.get("local_mode") == "NONE"]
        final_modes = Counter(row.get("final_mode") for row in group)
        dp_mode_vectors = [row.get("dp_modes") for row in group if isinstance(row.get("dp_modes"), list)]
        dp_token_vectors = [row.get("dp_tokens") for row in group if isinstance(row.get("dp_tokens"), list)]
        vector = dp_mode_vectors[0] if dp_mode_vectors else None
        token_vector = dp_token_vectors[0] if dp_token_vectors else None
        vector_disagreement = any(item != vector for item in dp_mode_vectors[1:])
        local_vector_mismatch = []
        if isinstance(vector, list):
            for row in group:
                dp = row["dp"]
                if dp < len(vector) and row.get("local_mode") != vector[dp]:
                    local_vector_mismatch.append(dp)
        any_none_in_vector = isinstance(vector, list) and "NONE" in vector
        expected_final = "NONE" if any_none_in_vector else None
        final_none_all = len(group) == expected_dp and all(row.get("final_mode") == "NONE" for row in group)
        forced_uniform = [
            row["dp"]
            for row in group
            if row.get("uniform") is True and row.get("is_all_decode") is False
        ]
        nonuniform_local_none = [
            row["dp"]
            for row in group
            if row.get("local_mode") == "NONE" and row.get("uniform") is False
        ]
        batch_classes = Counter(row.get("batch_class") for row in group)
        steps.append(
            {
                "step": step,
                "dp_records": len(group),
                "dp_complete": len(group) == expected_dp and len({row["dp"] for row in group}) == expected_dp,
                "local_none_dp": local_none,
                "nonuniform_local_none_dp": nonuniform_local_none,
                "forced_uniform_likely_dp": forced_uniform,
                "dp_modes": vector,
                "dp_tokens": token_vector,
                "dp_mode_vectors_disagree": vector_disagreement,
                "local_mode_vs_dp_vector_mismatch": local_vector_mismatch,
                "any_none_in_dp_vector": any_none_in_vector,
                "all_dp_final_none": final_none_all,
                "final_modes": dict(final_modes),
                "batch_classes": dict(batch_classes),
                "ts_ns_min": min((r["ts_ns"] for r in group if isinstance(r.get("ts_ns"), int)), default=None),
                "ts_ns_max": max((r["ts_ns"] for r in group if isinstance(r.get("ts_ns"), int)), default=None),
                "records": group,
            }
        )

    intervals = []
    for dp in range(expected_dp):
        rows = sorted((r for r in records if r.get("tp") == 0 and r.get("dp") == dp and isinstance(r.get("ts_ns"), int)), key=lambda r: r["step"])
        for previous, current in zip(rows, rows[1:]):
            intervals.append(
                {
                    "dp": dp,
                    "from_step": previous["step"],
                    "to_step": current["step"],
                    "source_final_mode": previous["final_mode"],
                    "destination_final_mode": current["final_mode"],
                    "interval_ms": (current["ts_ns"] - previous["ts_ns"]) / 1_000_000,
                }
            )

    interval_transitions = Counter(
        (item["source_final_mode"], item["destination_final_mode"])
        for item in intervals
    )

    summary = {
        "record_count": len(records),
        "steps": len(steps),
        "complete_dp_steps": sum(s["dp_complete"] for s in steps),
        "steps_any_local_none": sum(bool(s["local_none_dp"]) for s in steps),
        "steps_nonuniform_local_none": sum(bool(s["nonuniform_local_none_dp"]) for s in steps),
        "steps_any_none_in_dp_vector": sum(bool(s["any_none_in_dp_vector"]) for s in steps),
        "steps_all_dp_final_none": sum(bool(s["all_dp_final_none"]) for s in steps),
        "steps_with_vector_mismatch": sum(s["dp_mode_vectors_disagree"] or bool(s["local_mode_vs_dp_vector_mismatch"]) for s in steps),
        "batch_class_counts": dict(Counter(r.get("batch_class") for r in records)),
        "step_details": steps,
        "record_intervals": intervals,
        "interval_transition_counts": {
            f"{source}->{destination}": count
            for (source, destination), count in sorted(interval_transitions.items())
        },
        "interval_note": "Each interval is ts_ns(to_step)-ts_ns(from_step), so attribute elapsed model execution to the source step/mode; it also includes next-step scheduling/preparation/coordination. It is not a DFC or model-forward duration. Group intervals by source_final_mode for source-step cost, not destination_final_mode.",
        "classification_note": "forced_uniform_likely is inferred from uniform=True with is_all_decode=False under the observed MTP formula. The patched log does not record force_uniform_decode or an explicit scheduler phase, so request_ids_present_phase_unverified is not proof of a real request.",
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path, help="completed D170/D171 diagnostic logs")
    parser.add_argument("--expected-dp", type=int, default=16)
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    args = parser.parse_args()
    result = analyze(read_records(args.logs), args.expected_dp)
    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()
