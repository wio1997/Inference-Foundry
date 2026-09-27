#!/usr/bin/env python3
"""Summarize clean B same-device ordering; no uninstrumented-schedule inference."""
import argparse
import json
import statistics
from pathlib import Path
from loop078_count_markers_analyze import record


def describe(vals):
    return {"min": min(vals), "median": statistics.median(vals), "max": max(vals)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capture-dir", type=Path, required=True)
    p.add_argument("--timing", type=Path, required=True)
    p.add_argument("--gate", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    preflight = json.loads(a.timing.read_text())
    gate = json.loads(a.gate.read_text())
    assert preflight["passed"] and preflight["devices"] == list(range(8))
    assert gate["arm"] == "B" and gate["http_posts"] == 60 and gate["source_restored"]
    raw = [json.loads(path.read_text()) for path in sorted(a.capture_dir.glob("rank*_cohort*.json"))]
    assert len(raw) == 40
    assert {(x["cohort"], x["rank"]) for x in raw} == {(c,r) for c in range(1,6) for r in range(8)}
    assert all(x["run_id"] == gate["run_id"] for x in raw)
    for cohort in range(1,6):
        part = [x for x in raw if x["cohort"] == cohort]
        assert all(x["accepted_counts"] == part[0]["accepted_counts"] for x in part)
    rows = [record(x, preflight["timing"]) for x in raw]
    assert all(x["observed_ordering"] for x in rows)
    assert all(x["original_schedule_slack_supported"] for x in rows)
    assert all(not x["downstream_observed"] for x in rows)
    out = {
        "status": "accepted_instrumented_same_device_local_read_before_overwrite",
        "rank_cohort_pairs": len(rows),
        "direct_margin_ms": describe([x["margin_ms"] for x in rows]),
        "read_event_envelope_ms": describe([x["read_envelope_ms"] for x in rows]),
        "write_expression_envelope_ms": describe([x["write_envelope_ms"] for x in rows]),
        "host_existing_copy_event_sync_us": describe([x["sync_host_ns"]/1000 for x in rows]),
        "timing_empirical_uncertainty_ms": preflight["timing"],
        "first_host_numeric_consumers": "count_add and seq_mirror_add after existing production-event synchronize; validated pointers and monotonic Host order in every record",
        "later_draft_metadata_consumer": "unobserved: fixed padded/nonasync branch bypasses hooked utils clone/add and drafting-specific DSA builder was not hooked",
        "original_schedule_extrapolation": None,
        "storage_happens_before_source_proof": False,
        "scheduling_latency_floor_s": None,
        "product_tps_ceiling": None,
        "scope": "Instrumented original normal unparked c64->c65 observations on eight ranks/five cohorts only; not a removable gap or legal accelerated schedule.",
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2)+"\n")
    print(json.dumps({"status": out["status"], "pairs": len(rows), "margin_ms": out["direct_margin_ms"]}))


if __name__ == "__main__":
    main()
