#!/usr/bin/env python3
"""Validate c12 release chronology without inventing semaphore parent edges."""
import argparse
import hashlib
import json
from pathlib import Path


def validate(rows, concurrency):
    if concurrency != 12 or len(rows) != 48:
        raise ValueError("frozen c12/48 client scope changed")
    indices = [r["i"] for r in rows]
    if len(set(indices)) != 48:
        raise ValueError("duplicate dataset index")
    for r in rows:
        times = [r[k] for k in (
            "c12_acquire_attempt_monotonic_ns", "c12_acquired_monotonic_ns",
            "start_monotonic_ns", "sse_done_monotonic_ns",
            "end_monotonic_ns", "c12_release_completed_by_monotonic_ns")]
        if not all(type(t) is int and t > 0 for t in times) or times != sorted(times):
            raise ValueError("client c12 chronology invalid")
        if r["c12_unique_predecessor_certified"] is not False:
            raise ValueError("anonymous semaphore has no certified unique parent")
        if r["http_status"] != 200 or r["output_tokens"] != 1024 or r["error"] is not None:
            raise ValueError("client request incomplete")
    # Every concurrent request interval certainly includes acquisition through
    # stream end. Permit release occurs somewhere in [end, release_completed_by].
    # The resulting overlap must respect c12; it cannot prove exact parentage.
    events = sorted([(r["c12_acquired_monotonic_ns"], 1) for r in rows] +
                    [(r["end_monotonic_ns"], -1) for r in rows],
                    key=lambda x: (x[0], x[1]))
    active = peak = 0
    for _, delta in events:
        active += delta
        if active < 0 or active > 12:
            raise ValueError("c12 guaranteed-active interval violated")
        peak = max(peak, active)
    if active != 0:
        raise ValueError("unclosed client interval")
    ordered = sorted(rows, key=lambda r: r["c12_acquired_monotonic_ns"])
    eligible = {}
    for place, r in enumerate(ordered):
        if place < 12:
            continue
        # At least one earlier request must have finished the SSE stream before
        # this acquisition. This is a candidate set, not a transferred permit.
        candidates = [p["i"] for p in ordered[:place]
                      if p["end_monotonic_ns"] <= r["c12_acquired_monotonic_ns"]]
        if not candidates:
            raise ValueError("successor has no eligible earlier completed stream")
        eligible[str(r["i"])] = candidates
    return {"guaranteed_active_peak": peak,
            "successor_count": len(eligible),
            "eligible_release_set_size_min": min(map(len, eligible.values())),
            "eligible_release_set_size_max": max(map(len, eligible.values())),
            "unique_parent_edges_certified": 0,
            "eligible_release_sets": eligible,
            "scope": "Host-clock and anonymous-semaphore candidate sets only; actual permit release is bracketed by end and release_completed_by"}


def self_test():
    base = 10**12
    rows = []
    for i in range(48):
        # Four complete waves with c12 slots, each record has all six markers.
        wave, slot = divmod(i, 12)
        acquired = base + wave * 10000 + slot
        end = acquired + 1000
        rows.append({"i": i, "c12_acquire_attempt_monotonic_ns": acquired - 1,
                     "c12_acquired_monotonic_ns": acquired,
                     "start_monotonic_ns": acquired + 1,
                     "sse_done_monotonic_ns": end - 1,
                     "end_monotonic_ns": end,
                     "c12_release_completed_by_monotonic_ns": end + 1,
                     "c12_unique_predecessor_certified": False,
                     "http_status": 200, "output_tokens": 1024, "error": None})
    positive = validate(rows, 12)
    negative = 0
    def reject(mutator):
        nonlocal negative
        changed = [dict(x) for x in rows]
        mutator(changed)
        try:
            validate(changed, 12)
        except (ValueError, KeyError, TypeError):
            negative += 1
            return
        raise AssertionError("invalid c12 lineage admitted")
    reject(lambda x: x[0].update(sse_done_monotonic_ns=None))
    reject(lambda x: x[0].update(c12_unique_predecessor_certified=True))
    reject(lambda x: x[0].update(c12_release_completed_by_monotonic_ns=x[0]["end_monotonic_ns"] - 1))
    reject(lambda x: x[0].update(output_tokens=1023))
    reject(lambda x: x[0].update(i=x[1]["i"]))
    reject(lambda x: x[12].update(c12_acquired_monotonic_ns=x[0]["c12_acquired_monotonic_ns"] + 2))
    try:
        validate(rows, 13)
    except ValueError:
        negative += 1
    else:
        raise AssertionError("wrong c12 limit admitted")
    return {"positive": 1, "negative": negative, "pass": negative == 7,
            "synthetic_successor_count": positive["successor_count"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase-dir", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.self_test:
        result = self_test()
    else:
        if a.phase_dir is None or a.output is None:
            p.error("--phase-dir and --output required")
        path = a.phase_dir / "summary.json"
        source = json.loads(path.read_text())
        if source["summary"]["success"] != 48 or source["summary"]["concurrency"] != 12:
            raise ValueError("incomplete frozen client phase")
        rows = source["requests"]
        result = validate(rows, 12)
        result["summary_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        result["status"] = "c12_eligible_release_sets_admitted"
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "eligible_release_sets"}))


if __name__ == "__main__":
    main()
