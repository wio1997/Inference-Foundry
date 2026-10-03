#!/usr/bin/env python3
"""
SLO analyser for the GLM-5.2 PD AISBench prefix test.

Reads the AISBench per-request performance table (gsm8k.csv) which AISBench
derives from the *successful* requests only (its `N` column is the success
count), and reports the required TTFT/TPOT percentiles plus an SLO verdict.

SLO targets (from the task):
  TTFT  P50 < 4 s   P75 < 8 s   P90 < 12 s   P99 < 30 s
  TPOT  P50 < 18 ms             P90 < 40 ms

The AISBench table already contains Median (=P50), P75, P90, P99 - these are
percentiles of the successful requests, not means, so no substitution happens.
The raw per-request detail file is used to sanity check success counts and the
real output-token count (so a short EOS-terminated answer cannot pass as a
61440-token answer).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys

UNIT_RE = re.compile(r"^\s*(-?[\d.]+)\s*([a-zA-Z/%]*)\s*$")


def parse_value(cell: str):
    """'439.1 ms' -> (439.1, 'ms');  '2.0' -> (2.0, ''). Returns (None,'') if unparsable."""
    if cell is None:
        return None, ""
    m = UNIT_RE.match(cell)
    if not m:
        return None, ""
    try:
        return float(m.group(1)), m.group(2)
    except ValueError:
        return None, ""


def ms(value: float, unit: str):
    """Normalise a latency to milliseconds."""
    if value is None:
        return None
    if unit == "s":
        return value * 1000.0
    if unit == "us":
        return value / 1000.0
    return value  # 'ms' or unitless


def read_perf_table(path: str):
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or len(row) < 3:
                continue
            if row[0].strip() == "Performance Parameters":
                header = [c.strip() for c in row]
                continue
            metric = row[0].strip()
            rows[metric] = row
    return rows


def metric_stats(rows, metric):
    """Return dict of stat_name -> (value, unit)."""
    row = rows.get(metric)
    if not row:
        return None
    # row = [metric, stage, Average, Min, Max, Median, P75, P90, P99, N]
    out = {}
    names = ["Average", "Min", "Max", "Median", "P75", "P90", "P99", "N"]
    for i, name in enumerate(names):
        idx = i + 2
        cell = row[idx] if idx < len(row) else None
        v, u = parse_value(cell) if cell is not None else (None, "")
        out[name] = (v, u)
    return out


def pct(stats, name, latency=True):
    if not stats or name not in stats:
        return None
    v, u = stats[name]
    if v is None:
        return None
    return ms(v, u) if latency else v


def load_details(path):
    if not path or not os.path.exists(path):
        return None
    n_total = n_success = 0
    out_tokens = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_total += 1
            if d.get("success"):
                n_success += 1
                ot = d.get("output_tokens")
                if isinstance(ot, (int, float)):
                    out_tokens.append(int(ot))
    return {
        "details_total": n_total,
        "details_success": n_success,
        "output_tokens_count": len(out_tokens),
        "output_tokens_min": min(out_tokens) if out_tokens else None,
        "output_tokens_max": max(out_tokens) if out_tokens else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perf_csv", required=True)
    ap.add_argument("--details_jsonl", default="")
    ap.add_argument("--expect_output_len", type=int, default=61440)
    ap.add_argument("--concurrency", type=int, default=0)
    ap.add_argument("--expected_requests", type=int, default=None, help="Actual declared request count; default retains legacy 2*concurrency")
    ap.add_argument("--out_json", default="")
    ap.add_argument("--out_md", default="")
    args = ap.parse_args()
    if args.expected_requests is not None and args.expected_requests <= 0:
        ap.error("expected_requests must be positive")

    # Explicit mid-run scope reduction: retain originals; score predetermined IDs only.
    scope_path = os.path.join(os.path.dirname(args.details_jsonl), "request_scope_amendment.json")
    if args.details_jsonl and os.path.isfile(scope_path):
        scope = json.load(open(scope_path, encoding="utf-8"))
        selected = set(scope["selected_ids"])
        filtered = args.details_jsonl + ".selected.jsonl"
        with open(args.details_jsonl, encoding="utf-8") as src, open(filtered, "w", encoding="utf-8") as dst:
            for line in src:
                try: item = json.loads(line)
                except json.JSONDecodeError: continue
                if item.get("id") in selected: dst.write(line)
        args.details_jsonl = filtered

    rows = read_perf_table(args.perf_csv)
    ttft = metric_stats(rows, "TTFT")
    tpot = metric_stats(rows, "TPOT")
    e2el = metric_stats(rows, "E2EL")
    otok = metric_stats(rows, "OutputTokens")
    itok = metric_stats(rows, "InputTokens")

    res = {
        "perf_csv": args.perf_csv,
        "details_jsonl": args.details_jsonl,
        "concurrency": args.concurrency,
        "n_success": int(pct(ttft, "N", latency=False) or 0),
        "ttft_ms": {
            "p50": pct(ttft, "Median"),
            "p75": pct(ttft, "P75"),
            "p90": pct(ttft, "P90"),
            "p99": pct(ttft, "P99"),
            "avg": pct(ttft, "Average"),
            "min": pct(ttft, "Min"),
            "max": pct(ttft, "Max"),
        },
        "tpot_ms": {
            "p50": pct(tpot, "Median"),
            "p90": pct(tpot, "P90"),
            "p99": pct(tpot, "P99"),
            "avg": pct(tpot, "Average"),
            "min": pct(tpot, "Min"),
            "max": pct(tpot, "Max"),
        },
        "e2el_ms": {"p50": pct(e2el, "Median"), "avg": pct(e2el, "Average")},
        "output_tokens": {
            "median": pct(otok, "Median", latency=False),
            "min": pct(otok, "Min", latency=False),
            "max": pct(otok, "Max", latency=False),
        },
        "input_tokens": {"median": pct(itok, "Median", latency=False)},
    }

    det = load_details(args.details_jsonl)
    if det:
        res.update(det)

    # ---- SLO verdict -------------------------------------------------------
    checks = {
        "TTFT_P50_lt_4000ms":  (res["ttft_ms"]["p50"], 4000.0, "<"),
        "TTFT_P75_lt_8000ms":  (res["ttft_ms"]["p75"], 8000.0, "<"),
        "TTFT_P90_lt_12000ms": (res["ttft_ms"]["p90"], 12000.0, "<"),
        "TTFT_P99_lt_30000ms": (res["ttft_ms"]["p99"], 30000.0, "<"),
        "TPOT_P50_lt_18ms":    (res["tpot_ms"]["p50"], 18.0, "<"),
        "TPOT_P90_lt_40ms":    (res["tpot_ms"]["p90"], 40.0, "<"),
    }
    verdict = {}
    for name, (val, thr, op) in checks.items():
        if val is None:
            verdict[name] = "NO_DATA"
        else:
            verdict[name] = "PASS" if val < thr else "FAIL"
    res["slo"] = verdict
    res["slo_all_pass"] = all(v == "PASS" for v in verdict.values())

    # ---- output length integrity ------------------------------------------
    exp = args.expect_output_len
    ot_med = res["output_tokens"]["median"]
    res["output_len_ok"] = (ot_med is not None and abs(ot_med - exp) < 1.0)
    if det and det.get("output_tokens_min") is not None:
        res["output_len_ok"] = res["output_len_ok"] and det["output_tokens_min"] >= exp - 1
    if det and det.get("details_success") is not None and det.get("details_total"):
        res["all_requests_succeeded"] = det["details_success"] == det["details_total"]
    else:
        res["all_requests_succeeded"] = None

    expected_requests = args.expected_requests if args.expected_requests is not None else args.concurrency * 2
    res["expected_requests"] = expected_requests
    res["output_len_ok"] = bool(
        det and det.get("details_success", 0) > 0
        and det.get("output_tokens_count") == det["details_success"]
        and det.get("output_tokens_min") == exp
        and det.get("output_tokens_max") == exp
    )
    res["all_requests_succeeded"] = bool(
        det and expected_requests > 0
        and det["details_total"] == expected_requests
        and det["details_success"] == expected_requests
        and res["n_success"] == expected_requests
    )
    res["latency_slo_all_pass"] = res["slo_all_pass"]
    res["slo_all_pass"] = bool(res["latency_slo_all_pass"]
        and res["output_len_ok"] and res["all_requests_succeeded"])

    # P99 with few samples is not robust -> flag it.
    n = res["n_success"] or 0
    res["p99_note"] = (
        f"N={n}: P99 is based on fewer than 100 requests and is not a robust tail estimate"
        if n < 100 else f"N={n}"
    )

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)

    def fmt(v):
        return "N/A" if v is None else f"{v:.1f}"

    md = []
    md.append(f"### SLO result — concurrency {args.concurrency}")
    md.append("")
    md.append(f"- successful requests N = {n}   (output_len_ok={res['output_len_ok']}, "
              f"all_succeeded={res['all_requests_succeeded']})")
    md.append(f"- P99 sample note: {res['p99_note']}")
    md.append("")
    md.append("| metric | P50 | P75 | P90 | P99 | target | verdict |")
    md.append("|---|---|---|---|---|---|---|")
    md.append(f"| TTFT (ms) | {fmt(res['ttft_ms']['p50'])} | {fmt(res['ttft_ms']['p75'])} | "
              f"{fmt(res['ttft_ms']['p90'])} | {fmt(res['ttft_ms']['p99'])} | "
              f"P50<4000 P75<8000 P90<12000 P99<30000 | "
              f"{verdict['TTFT_P50_lt_4000ms']}/{verdict['TTFT_P75_lt_8000ms']}/"
              f"{verdict['TTFT_P90_lt_12000ms']}/{verdict['TTFT_P99_lt_30000ms']} |")
    md.append(f"| TPOT (ms) | {fmt(res['tpot_ms']['p50'])} | - | {fmt(res['tpot_ms']['p90'])} | "
              f"{fmt(res['tpot_ms']['p99'])} | P50<18 P90<40 | "
              f"{verdict['TPOT_P50_lt_18ms']}/{verdict['TPOT_P90_lt_40ms']} |")
    md.append("")
    md.append(f"**ALL SLO PASS: {res['slo_all_pass']}**")
    text = "\n".join(md)

    print(text)
    if args.out_md:
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write(text + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
