#!/usr/bin/env python3
"""Recheck Run246/247 target task overlap; observation, not a lower bound."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = "evidence/20260926_loop060_resource"


def load(path):
    return json.loads((ROOT / path).read_bytes())


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def ns(value):
    decimal = Decimal(str(value).strip())
    assert decimal.is_finite()
    scaled = decimal * 1000
    assert scaled == scaled.to_integral_value()
    return int(scaled)


def classify_core(core, name, op_type):
    if core in ("AI_CORE", "MIX_AIC"):
        return 1
    if core in ("AI_VECTOR_CORE", "MIX_AIV"):
        return 2
    if core == "COMMUNICATION":
        if name == "AivKernel" and op_type.startswith("hcom_"):
            return 2
        if name.startswith("hcom_"):
            return 4
        raise AssertionError((core, name, op_type))
    raise AssertionError((core, name, op_type))


def categories(scope, task_intervals):
    lo, hi = scope
    events = [(lo, 0, 0), (hi, 0, 0)]
    for start, end, mask in task_intervals:
        start, end = max(lo, start), min(hi, end)
        if end <= start or not mask:
            continue
        events.append((start, mask, 1))
        events.append((end, mask, -1))
    events.sort(key=lambda x: x[0])
    counts = [0, 0, 0]
    duration = Counter()
    prev = lo
    for t, mask, delta in events:
        if t > prev:
            state = sum((1 << i) for i, count in enumerate(counts) if count > 0)
            duration[state] += t - prev
            prev = t
        for i in range(3):
            if mask & (1 << i):
                counts[i] += delta
                assert counts[i] >= 0
    assert all(c == 0 for c in counts)
    assert sum(duration.values()) == hi - lo
    return duration


def main(out):
    export = load(f"{P}/run247/export_index.json")
    prior = load(f"{P}/run247/analysis.json")
    assert export["export_exit"] == 0 and prior["status"] == "valid"
    assert prior["latest_capture"] == 4 and prior["latest_valid_windows"] == 16
    index = {}
    for item in export["rows"]:
        path = item["path"]
        name = Path(path).parent.name
        rank = int(name.split("_")[0].removeprefix("rank"))
        index.setdefault(rank, []).append(item)
    assert set(index) == set(range(8))
    assert all(len(v) == 5 for v in index.values())
    rows = []
    source_sha = {}
    for rank in range(8):
        item = sorted(index[rank], key=lambda x: x["path"])[4]
        trace_path = item["path"] + "/trace_view.json"
        kernel_path = item["path"] + "/kernel_details.csv"
        assert sha(trace_path) == item["trace_sha256"]
        assert sha(kernel_path) == item["kernel_csv_sha256"]
        source_sha[trace_path] = item["trace_sha256"]
        source_sha[kernel_path] = item["kernel_csv_sha256"]
        events = load(trace_path)
        scopes = sorted((e for e in events if e.get("name") == "extreme::target" and e.get("cat") == "cpu_op"),
                        key=lambda e: ns(e["ts"]))
        assert len(scopes) == 2
        with (ROOT / kernel_path).open(newline="") as handle:
            kernels = list(csv.DictReader(handle))
        for cycle, event in enumerate(scopes):
            lo = ns(event["ts"])
            hi = lo + ns(event["dur"])
            assert hi > lo
            task_intervals = []
            selected_count = 0
            carry_in_count = 0
            cross_end_count = 0
            family_count = Counter()
            for kernel in kernels:
                start = ns(kernel["Start Time(us)"])
                end = start + ns(kernel["Duration(us)"])
                assert end > start
                if not start < hi or not end > lo:
                    continue
                mask = classify_core(kernel["Accelerator Core"],kernel["Name"],kernel["Type"])
                assert mask
                task_intervals.append((start, end, mask))
                if end > hi:
                    cross_end_count += 1
                if start < lo:
                    carry_in_count += 1
                    continue
                selected_count += 1
                name = kernel["Name"]
                if name.startswith("aclnnGroupedMatmulSwigluQuantWeightNzV2_"):
                    family_count["gmm1"] += 1
                if name.startswith("aclnnGroupedMatmulWeightNz_"):
                    family_count["gmm2"] += 1
                if name.startswith("hcom_"):
                    family_count["hcom"] += 1
                if name == "AivKernel" and kernel["Type"].startswith("hcom_"):
                    assert float(kernel["aiv_time(us)"]) > 0
                    assert float(kernel["aiv_read_main_memory_datas(KB)"]) > 0
                    family_count["hccl_aiv"] += 1
            assert family_count == {"gmm1": 43, "gmm2": 43, "hcom": 265,"hccl_aiv":265}
            assert carry_in_count == 0 and cross_end_count == 0
            duration = categories((lo, hi), task_intervals)
            prior_row = next(w for w in prior["windows"] if (w["rank"], w["capture"], w["cycle"]) == (rank, 4, cycle))
            assert math.isclose((hi - lo) / 1e6, prior_row["scope_ms"], abs_tol=1e-3)
            row = dict(rank=rank, capture=4, cycle=cycle, scope_ms=(hi-lo)/1e6,
                       exported_ai_core_interval_union_ms=sum(v for mask, v in duration.items() if mask & 1)/1e6,
                       exported_vector_interval_union_including_hccl_aiv_ms=sum(v for mask, v in duration.items() if mask & 2)/1e6,
                       exported_communication_pseudo_envelope_union_ms=sum(v for mask, v in duration.items() if mask & 4)/1e6,
                       exported_communication_pseudo_envelope_without_exported_ai_ms=sum(v for mask, v in duration.items() if mask == 4)/1e6,
                       any_exported_task_ms=sum(v for mask, v in duration.items() if mask)/1e6,
                       no_exported_task_ms=duration[0]/1e6,
                       selected_tasks=selected_count, carry_in_tasks=carry_in_count,cross_end_tasks=cross_end_count,
                       family_count=dict(family_count),
                       source_trace_sha256=sha(trace_path), source_kernel_sha256=sha(kernel_path))
            assert math.isclose(row["any_exported_task_ms"]+row["no_exported_task_ms"],row["scope_ms"],abs_tol=1e-5)
            rows.append(row)
    assert len(rows) == 16
    summary = {name: {"median": statistics.median(r[name] for r in rows),
                      "min": min(r[name] for r in rows),
                      "max": max(r[name] for r in rows)}
               for name in ("scope_ms","exported_ai_core_interval_union_ms",
                            "exported_vector_interval_union_including_hccl_aiv_ms",
                            "exported_communication_pseudo_envelope_union_ms",
                            "exported_communication_pseudo_envelope_without_exported_ai_ms",
                            "any_exported_task_ms","no_exported_task_ms")}
    result = dict(status="instrumented_current_overlap_only", source_sha256=source_sha,
                  index_sha256=sha(f"{P}/run247/export_index.json"),
                  prior_analysis_sha256=sha(f"{P}/run247/analysis.json"),
                  rows=rows, summary=summary, cross_rank_physical_overlap=None,
                  limitation="Profiler synchronized/marked original FULL Graph; 265 hcom pseudo envelopes and 265 HCCL AivKernel tasks/window are distinct. Pseudo envelopes are not link service or exposed collective latency. No-exported-task intervals are not removable idle. Rank-local interval unions are not a dependency DAG, capacity floor or passive formal Current. Cross-rank clock domains are unvalidated, so no all8 concurrent time or makespan is calculated.",
                  numeric_bound_update=False, formal_current_tps=571.681,
                  finite_resource_floor_s=None, finite_scheduling_floor_s=None,
                  finite_product_tps_ceiling=None)
    out.mkdir(parents=True, exist_ok=False)
    (out / "overlap.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(summary=summary, cross_rank_physical_overlap=None), sort_keys=True))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        assert ns("1790356535568869.700") == 1790356535568869700
        for value in ("NaN", "Infinity", "1.0001"):
            try: ns(value)
            except (AssertionError, ValueError, OverflowError): pass
            else: raise AssertionError("invalid/unsupported clock precision accepted")
        assert categories((0, 100), [(10, 40, 1), (20, 50, 2), (20, 50, 4)]) == {0: 60, 1: 10, 7: 20, 6: 10}
        for value in (-1, 0):
            try:
                assert 0 + ns(value) > 0
            except AssertionError: pass
            else: raise AssertionError("nonpositive duration accepted")
        try: classify_core("UNKNOWN", "x", "y")
        except AssertionError: pass
        else: raise AssertionError("unknown core accepted")
        print("Run572 interval scope gate PASS")
    else:
        assert args.out_dir is not None
        main(args.out_dir)
