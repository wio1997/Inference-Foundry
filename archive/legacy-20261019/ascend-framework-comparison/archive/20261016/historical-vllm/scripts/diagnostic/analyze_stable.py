#!/usr/bin/env python3
"""Read-only replay of AISBench StablePerfMetricCalculator + six SLA gates.

Usage: analyze_stable.py RUN_DIR CONCURRENCY EXPECTED_N [--output FILE]

RUN_DIR may contain outputs/full/<case>/gsm8k_details.jsonl and db_data/*.db,
or may point directly at that case directory. The SQLite DB is opened with
mode=ro&immutable=1. Output is JSON on stdout unless --output is specified.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import pathlib
import sqlite3
import sys
from typing import Any

import numpy as np


AISBENCH_SOURCE_REVISION = "19018e9cc95dd517528221e01b03d1fd14336a14"
SELECTOR_DESCRIPTION = (
    "Faithful offline replay of StablePerfMetricCalculator._get_requests_id: "
    "stable-sort all start/end events by time (start/end insertion order follows "
    "request order), increment/decrement concurrency, set the stable start at "
    "the second start event that reaches configured concurrency, retain the "
    "last event time at concurrency, then select every request whose start is "
    "within the inclusive interval."
)
QUANTILE_DESCRIPTION = "numpy.percentile default linear method (p*(n-1))"
GATES = {
    "ttft_p50_ms": 4000.0,
    "ttft_p75_ms": 8000.0,
    "ttft_p90_ms": 12000.0,
    "ttft_p99_ms": 30000.0,
    "tpot_p50_ms": 18.0,
    "tpot_p90_ms": 40.0,
}


def find_inputs(root: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    search_root = root / "outputs" / "full" if (root / "outputs" / "full").is_dir() else root
    detail_files = sorted(search_root.glob("**/gsm8k_details.jsonl"))
    if not detail_files and root.is_file() and root.name == "gsm8k_details.jsonl":
        detail_files = [root]
    if len(detail_files) != 1:
        raise ValueError(f"expected exactly one gsm8k_details.jsonl below {root}, found {len(detail_files)}")
    detail = detail_files[0]
    db_files = sorted(detail.parent.glob("db_data/*.db"))
    if len(db_files) != 1:
        raise ValueError(f"expected exactly one db_data/*.db beside {detail}, found {len(db_files)}")
    return detail, db_files[0]


def load_rows(details_path: pathlib.Path, db_path: pathlib.Path) -> tuple[list[dict], dict]:
    uri = db_path.resolve().as_uri() + "?mode=ro&immutable=1"
    con = sqlite3.connect(uri, uri=True)
    try:
        blobs = {int(row_id): bytes(blob) for row_id, blob in con.execute("SELECT id, arr_blob FROM numpy_store")}
    finally:
        con.close()

    all_rows: list[dict] = []
    failed = 0
    missing_timing = 0
    with details_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            record = json.loads(line)
            if not record.get("success"):
                failed += 1
                continue
            points = record.get("time_points")
            if isinstance(points, dict) and "__db_ref__" in points:
                points = blobs.get(int(points["__db_ref__"]))
                if points is not None:
                    points = np.load(io.BytesIO(points), allow_pickle=False)
            if points is None:
                missing_timing += 1
                continue
            t = np.asarray(points).reshape(-1)
            out_tokens = int(record.get("output_tokens") or 0)
            in_tokens = int(record.get("input_tokens") or 0)
            if len(t) < 2 or out_tokens < 1:
                missing_timing += 1
                continue
            start = float(t[0])
            end = float(t[-1])
            latency = end - start
            ttft = float(t[1] - t[0])
            tpot = (latency - ttft) / (out_tokens - 1) if out_tokens > 1 else 0.0
            if not all(math.isfinite(v) for v in (start, end, latency, ttft, tpot)) or end < start:
                missing_timing += 1
                continue
            all_rows.append({
                "id": record.get("id", line_no - 1), "input_tokens": in_tokens,
                "output_tokens": out_tokens, "start_time": start, "end_time": end,
                "latency_s": latency, "ttft_s": ttft, "tpot_s": tpot,
            })
    if not all_rows:
        raise ValueError("no successful requests with valid timing found")
    return all_rows, {"detail_rows": line_no if 'line_no' in locals() else 0,
                      "failed": failed, "missing_timing": missing_timing,
                      "timed_success": len(all_rows)}


def native_stable_selection(rows: list[dict], cap: int) -> tuple[list[int] | None, dict]:
    # Match source event creation and stable sort semantics exactly.
    events = []
    for i, row in enumerate(rows):
        events.append({"id": i, "attr": "start", "time": row["start_time"]})
        events.append({"id": i, "attr": "end", "time": row["end_time"]})
    events.sort(key=lambda event: event["time"])
    concurrency = 0
    max_start_count = 0
    first_max_time = None
    last_max_time = None
    for event in events:
        if event["attr"] == "start":
            concurrency += 1
        else:
            concurrency -= 1
        if concurrency == cap:
            if event["attr"] == "start":
                max_start_count += 1
                if max_start_count == 2:
                    first_max_time = event["time"]
            last_max_time = event["time"]
    if first_max_time is None:
        return None, {"native_error": "CAN_NOT_FIND_STABLE_STAGE", "stage_start_s": None, "stage_end_s": None}
    selected = [i for i, row in enumerate(rows) if first_max_time <= row["start_time"] <= last_max_time]
    if not selected:
        return None, {"native_error": "CAN_NOT_FIND_STABLE_STAGE", "stage_start_s": first_max_time, "stage_end_s": last_max_time}
    return selected, {"native_error": None, "stage_start_s": first_max_time,
                      "stage_end_s": last_max_time, "window_s": last_max_time-first_max_time}


def pct(values: list[float], p: float) -> float | None:
    return float(np.percentile(np.asarray(values, dtype=np.float64), p)) if values else None


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantiles_ms(rows: list[dict], field: str, ps: tuple[int, ...]) -> dict[str, float | None]:
    values = [float(row[field]) * 1000.0 for row in rows]
    return {f"p{p}": pct(values, p) for p in ps}


def run(root: pathlib.Path, cap: int, expected_n: int) -> dict[str, Any]:
    detail_path, db_path = find_inputs(root)
    rows, counts = load_rows(detail_path, db_path)
    selected_ids, selection_meta = native_stable_selection(rows, cap)
    output: dict[str, Any] = {
        "run_dir": str(root.resolve()),
        "inputs": {"details": str(detail_path.resolve()), "sqlite": str(db_path.resolve()),
                   "sqlite_open_mode": "ro&immutable", "details_sha256": sha256_file(detail_path),
                   "sqlite_sha256": sha256_file(db_path)},
        "configured_concurrency": cap,
        "expected_n": expected_n,
        "counts": counts,
        "integrity_pass": counts["timed_success"] == expected_n and counts["failed"] == 0 and counts["missing_timing"] == 0,
        "selector": {"source_revision": AISBENCH_SOURCE_REVISION,
                     "class": "ais_bench.benchmark.calculators.stable_perf_metric_calculator.StablePerfMetricCalculator",
                     "algorithm": SELECTOR_DESCRIPTION,
                     "selected_by_start_time_inclusive": True,
                     **selection_meta},
        "quantile_method": QUANTILE_DESCRIPTION,
        "sla_limits": GATES,
    }
    if selected_ids is None:
        output.update({"selected_n": None, "window_condition": None, "all_gate_pass": None,
                       "stable_ttft_ms": None, "stable_tpot_ms": None,
                       "stable_input_token_throughput_token_s": None,
                       "stable_output_token_throughput_token_s": None,
                       "stable_gates": None})
        return output

    selected = [rows[i] for i in selected_ids]
    window = float(selection_meta["window_s"])
    max_all_e2e = max(row["latency_s"] for row in rows)
    max_selected_e2e = max(row["latency_s"] for row in selected)
    window_condition = max_all_e2e <= window / 3.0 if window > 0 else False
    ttft = quantiles_ms(selected, "ttft_s", (50, 75, 90, 99))
    tpot = quantiles_ms(selected, "tpot_s", (50, 75, 90, 99))
    measured = {f"ttft_p{p}_ms": ttft[f"p{p}"] for p in (50, 75, 90, 99)}
    measured.update({f"tpot_p{p}_ms": tpot[f"p{p}"] for p in (50, 90)})
    gates = {}
    for name, limit in GATES.items():
        value = measured[name]
        gates[name] = {"observed": value, "limit": limit,
                       "strict_pass": bool(value < limit), "difference": value - limit}
    output.update({
        "selected_n": len(selected),
        "selected_request_ids": [row["id"] for row in selected],
        "window_s": window,
        "window_condition": {
            "rule": "max E2E over all timed successful requests <= selected window / 3",
            "max_e2e_all_s": max_all_e2e,
            "max_e2e_selected_s": max_selected_e2e,
            "window_third_s": window / 3.0,
            "max_all_over_window_third": max_all_e2e / (window / 3.0) if window > 0 else None,
            "pass": window_condition,
        },
        "stable_ttft_ms": ttft,
        "stable_tpot_ms": tpot,
        "stable_input_token_throughput_token_s": sum(r["input_tokens"] for r in selected) / window if window > 0 else None,
        "stable_output_token_throughput_token_s": sum(r["output_tokens"] for r in selected) / window if window > 0 else None,
        "stable_total_token_throughput_token_s": sum(r["input_tokens"] + r["output_tokens"] for r in selected) / window if window > 0 else None,
        "stable_gates": gates,
        "all_gate_pass": all(g["strict_pass"] for g in gates.values()),
    })
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=pathlib.Path)
    parser.add_argument("concurrency", type=int)
    parser.add_argument("expected_n", type=int)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    try:
        result = run(args.run_dir, args.concurrency, args.expected_n)
    except Exception as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
