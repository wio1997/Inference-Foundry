#!/usr/bin/env python3
"""Run631: exact Host-marker coverage for two separate diagnostic W0s.

The output is a current execution ledger, not an attainable interval or a
cross-run timed DAG. It detects prior-cohort work inside the next cohort file.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path("/data/wio/Inference_Foundry")
OUT = ROOT / "evidence/20260928_loop081_bound/run631/product_coverage.json"


def read(p: Path):
    return json.loads(p.read_text())


def sha(p: Path):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def marker(xs: list[dict], name: str, reducer):
    t = [m["t_ns"] for x in xs for m in x["marks"] if m["kind"] == name]
    if name != "execute_entry" and len(t) != 8:
        raise AssertionError((name, len(t)))
    if not t:
        raise AssertionError(name)
    return reducer(t)


def cohort(base: Path, k: int, mapped: dict, client: dict,
           admission: dict) -> dict:
    paths = [base / f"product/rank{r}_cohort{k}.json" for r in range(8)]
    for path in paths:
        if sha(path) != admission["raw_sha256"].get(str(path)):
            raise AssertionError("raw product SHA mismatch")
    xs = [read(path) for path in paths]
    if any(x["rank"] != r or x["cohort"] != k for r, x in enumerate(xs)):
        raise AssertionError("rank/cohort ownership")
    if any(x["run_ts"] != admission["run_ts"] for x in xs):
        raise AssertionError("run tag mismatch")
    if len({x["time_namespace"] for x in xs}) != 1:
        raise AssertionError("time namespace mismatch")
    owned = {rid for rid, label in mapped.items() if label == k}
    assert len(owned) == 12
    for x in xs:
        if any(m.get("rank", x["rank"]) != x["rank"] for m in x["marks"]):
            raise AssertionError("marker rank mismatch")
        built = [m for m in x["marks"] if m["kind"] == "runtime_built_host"]
        serve = [m for m in x["marks"] if m["kind"] == "serve_start_host"]
        sync = [m for m in x["marks"] if m["kind"] == "serve_synced_host"]
        if not (len(built) == len(serve) == len(sync) == 1):
            raise AssertionError("marker cardinality")
        if (len(built[0]["req_ids"]) != 12
                or set(built[0]["req_ids"]) != owned):
            raise AssertionError("built request ownership")
        if not built[0]["t_ns"] <= serve[0]["t_ns"] <= sync[0]["t_ns"]:
            raise AssertionError("rank-local marker order")
    starts = [client[rid]["start_monotonic_ns"] for rid in owned]
    ends = [client[rid]["end_monotonic_ns"] for rid in owned]
    entries = [(m["t_ns"], x["rank"], m) for x in xs for m in x["marks"] if m["kind"] == "execute_entry"]
    labels = Counter()
    own_entries = []
    first_entries = []
    for t, rank, m in entries:
        ids = [q["req_id"] for q in m["new"] + m["cached"]]
        if not ids:
            labels["no_request_operand"] += 1
            first_entries.append(t)
            continue
        seen = {mapped.get(rid) for rid in ids}
        if None in seen or seen - {k - 1, k}:
            raise AssertionError((k, rank, seen))
        labels.update(seen)
        if k in seen:
            own_entries.append(t)
        first_entries.append(t)
    if not own_entries:
        raise AssertionError("no own work")
    first_record = min(first_entries)
    first_own = min(own_entries)
    first_client = min(starts)
    last_client = max(ends)
    built = marker(xs, "runtime_built_host", max)
    serve_start = marker(xs, "serve_start_host", max)
    synced = marker(xs, "serve_synced_host", max)
    if not first_record <= first_own <= built <= serve_start <= synced <= last_client:
        raise AssertionError("marker order")
    if not first_client <= first_own:
        raise AssertionError("client request starts after own execute")
    return {
        "cohort": k,
        "request_count": len(owned),
        "time_namespace": xs[0]["time_namespace"],
        "execute_record_count_all8": len(entries),
        "execute_record_label_presence_all8": {str(a): b for a, b in sorted(labels.items(), key=lambda x: str(x[0]))},
        "timestamps_monotonic_ns": {
            "first_client_start": first_client,
            "first_recorded_execute": first_record,
            "first_own_cohort_execute": first_own,
            "last_allrank_runtime_built": built,
            "last_allrank_serve_start": serve_start,
            "last_allrank_existing_sync": synced,
            "last_client_end": last_client,
        },
        "host_marker_durations_s": {
            "first_recorded_to_first_own": (first_own - first_record) / 1e9,
            "first_client_start_to_first_own": (first_own - first_client) / 1e9,
            "first_own_to_allrank_built": (built - first_own) / 1e9,
            "allrank_built_to_existing_sync": (synced - built) / 1e9,
            "existing_sync_to_last_client_end": (last_client - synced) / 1e9,
            "first_recorded_to_last_client_end": (last_client - first_record) / 1e9,
        },
        "warning": "Recorded first execute can belong to previous cohort; same-cohort intervals may overlap neighboring cohorts and are not device work or removable time.",
    }


def union_coverage(rows: list[dict], low: int, high: int) -> dict:
    spans = []
    for row in rows:
        t = row["timestamps_monotonic_ns"]
        points = (
            t["first_recorded_execute"],
            t["first_own_cohort_execute"],
            t["last_allrank_runtime_built"],
            t["last_allrank_existing_sync"],
            t["last_client_end"],
        )
        labels = ("recorded_prior_or_empty_to_own", "own_to_built_host",
                  "built_to_existing_sync_host", "sync_to_client_end_host")
        for label, begin, end in zip(labels, points, points[1:]):
            if end < begin:
                raise AssertionError("negative segment")
            spans.append((max(low, begin), min(high, end), row["cohort"], label))
    cuts = sorted({low, high} | {v for a, b, _, _ in spans for v in (a, b)})
    occupancy_ns = Counter()
    uncovered_ns = 0
    covered_ns = 0
    overlap_ns = 0
    for begin, end in zip(cuts, cuts[1:]):
        active = [(k, label) for a, b, k, label in spans if a <= begin and end <= b and b > a]
        dur = end - begin
        if not active:
            uncovered_ns += dur
        else:
            covered_ns += dur
            if len(active) > 1:
                overlap_ns += dur
            # Mutually exclusive occupancy signatures, not additive work.
            occupancy_ns["|".join(f"{k}:{label}" for k, label in sorted(active))] += dur
    if covered_ns + uncovered_ns != high - low:
        raise AssertionError("union conservation")
    return {
        "client_window_ns": high - low,
        "covered_by_any_host_marker_envelope_ns": covered_ns,
        "uncovered_by_these_envelopes_ns": uncovered_ns,
        "multiple_envelopes_active_ns": overlap_ns,
        "exclusive_occupancy_ns": dict(occupancy_ns),
        "scope": "Host marker envelope union over this diagnostic client clock only; coverage is not physical device occupancy, necessary work, removable time, or a formal-run transfer",
    }


def one_run(run: int) -> dict:
    base = ROOT / f"evidence/20260928_loop081_bound/run{run}/live/b"
    a = read(base / "client_admission.json")
    j = read(base / "product_admission.json")
    if sha(base / "client_admission.json") != j["raw_sha256"].get(str(base / "client_admission.json")):
        raise AssertionError("raw client admission SHA mismatch")
    summary_name = "summary.json" if run == 602 else "product_summary.json"
    summary_path = base.parents[1] / summary_name
    s = read(summary_path)
    if (a["status"] != "client_two_phase_admitted"
            or j["status"] != "product_identity_pass"):
        raise AssertionError("admission failure")
    if len(j["join_rows"]) != 48:
        raise AssertionError("incomplete join")
    by_response = {r["response_id"]: r for r in a["request_index"] if r["phase"] == "measured"}
    if len(by_response) != 48:
        raise AssertionError("incomplete client")
    mapped = {r["req_id"]: r["cohort"] for r in j["join_rows"]}
    client = {r["req_id"]: by_response[r["response_id"]] for r in j["join_rows"]}
    if len(mapped) != 48:
        raise AssertionError("duplicate request")
    if (len({r["response_id"] for r in j["join_rows"]}) != 48
            or {r["response_id"] for r in j["join_rows"]} != set(by_response)):
        raise AssertionError("response/client bijection")
    rows = [cohort(base, k, mapped, client, j) for k in range(5, 9)]
    low = a["measured_summary"]["clock"]["begin"]["monotonic_ns"]
    high = a["measured_summary"]["clock"]["end"]["monotonic_ns"]
    if not low <= min(r["timestamps_monotonic_ns"]["first_client_start"] for r in rows):
        raise AssertionError("client lower boundary")
    if not max(r["timestamps_monotonic_ns"]["last_client_end"] for r in rows) <= high:
        raise AssertionError("client upper boundary")
    # The prior summary used its client DONE marker. request_index end is a
    # later client endpoint (typically ~0.2 ms), so retain the difference.
    for row, prior in zip(rows, s["cohorts"]):
        t = row["host_marker_durations_s"]
        delta = t["first_recorded_to_last_client_end"] - prior["first_execute_to_client_done_s"]
        if not 0 <= delta < 0.001:
            raise AssertionError("summary mismatch")
        t["request_end_minus_prior_client_done_s"] = delta
    paths = [base / "client_admission.json", base / "product_admission.json", summary_path]
    wall_low = a["measured_summary"]["wall_start_monotonic_ns"]
    wall_high = a["measured_summary"]["wall_end_monotonic_ns"]
    if not low <= wall_low < wall_high <= high:
        raise AssertionError("clock envelope")
    return {
        "run": run,
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in paths},
        "client_boot_id": a["measured_summary"]["clock"]["kernel_boot_id"],
        "client_time_namespace": a["measured_summary"]["clock"]["time_namespace"],
        "client_outer_clock_span_s": (high - low) / 1e9,
        "client_measured_wall_monotonic_span_s": (wall_high - wall_low) / 1e9,
        "client_reported_perf_counter_duration_s": a["measured_summary"]["duration_s"],
        "outer_minus_measured_wall_s": ((high - low) - (wall_high - wall_low)) / 1e9,
        "product_output_ids": s["totals"]["client_total_usage"],
        "cohorts": rows,
        "union_coverage": union_coverage(rows, low, high),
        "scope": "one observer-perturbed W0; Host marker ordering and request ownership only, not complete device readiness, compulsory work, formal TPS or a bound",
    }


def main():
    runs = [one_run(x) for x in (602, 606)]
    for r in runs:
        if r["product_output_ids"] != 49152:
            raise AssertionError("product count")
        if any(c["time_namespace"] != r["client_time_namespace"] for c in r["cohorts"]):
            raise AssertionError("cross-process time namespace")
    result = {
        "status": "two_separate_diagnostic_host_coverage_ledgers",
        "contract": "fixed DSpark7 acceptance/cycles/output/model work; 8x910B3 DP1TP8 48x32K->1024 c12",
        "runs": runs,
        "findings": [
            "Run602 and Run606 use distinct W0 and observer implementations; their durations may be compared as diagnostic envelope classes but not stitched or subtracted as a causal intervention.",
            "For cohorts6–8, each recorded next-cohort file begins with previous-cohort execute entries on all8 ranks; first-recorded-to-built is not a pure next-cohort prefill/seed phase.",
            "Client cohort request envelopes can overlap predecessor completion; per-cohort phase durations must not be summed as mutually exclusive whole-Product work.",
            "Allrank built is Host construction completion, not seed/KV/device readiness. Existing sync is a current barrier, not a compulsory or removable floor.",
        ],
        "formal_current_tps": 571.681,
        "strict_resource_floor_s": None,
        "strict_scheduling_floor_s": None,
        "strict_product_tps_ceiling": None,
        "numeric_current_to_credible_limit_gap": None,
        "next_gap": "Need exact same-W0 Host/Device completion and output-publication coverage with causal stream waits, observer OFF/ON/OFF and formal repeated E2E before assigning exposure or attainability.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "sha256": sha(OUT)}))


if __name__ == "__main__":
    main()
