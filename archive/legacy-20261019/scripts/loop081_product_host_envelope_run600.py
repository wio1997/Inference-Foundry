#!/usr/bin/env python3
"""Reconcile Run341 same-W0 Host marks without claiming a device DAG bound."""

import hashlib
import json
from pathlib import Path


ROOT = Path("evidence/20260926_loop074_refill/run341")
OUT = Path("evidence/20260928_loop081_bound/run600")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path):
    return json.loads(path.read_text())


def merged_ns(intervals):
    ordered = sorted(intervals)
    if not ordered:
        return 0
    end = ordered[0][1]
    total = 0
    start = ordered[0][0]
    for lo, hi in ordered[1:]:
        if lo > end:
            total += end - start
            start, end = lo, hi
        else:
            end = max(end, hi)
    return total + end - start


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    input_hashes = {}
    scheduler_path = ROOT / "scheduler_analysis.json"
    input_hashes[str(scheduler_path)] = digest(scheduler_path)
    scheduler = load_json(scheduler_path)
    phase_path = ROOT / "phase_analysis.json"
    input_hashes[str(phase_path)] = digest(phase_path)
    phases = load_json(phase_path)
    if not scheduler["all8_semantic_row_parity"]:
        raise AssertionError("scheduler all8 semantic parity")
    rows = []
    for rank in range(8):
        mark_path = ROOT / "marks" / f"rank{rank}.jsonl"
        input_hashes[str(mark_path)] = digest(mark_path)
        marks = [json.loads(line) for line in mark_path.read_text().splitlines()]
        if len(marks) != 46 or any(x["rank"] != rank for x in marks):
            raise AssertionError((rank, "mark count/rank"))
        groups = []
        group = []
        for mark in marks:
            group.append(mark)
            if mark["kind"] == "runtime_built":
                groups.append(group)
                group = []
        if group or len(groups) != 4:
            raise AssertionError((rank, "cohort grouping"))
        for cohort, group in zip(range(5, 9), groups):
            executions = [m for m in group if m["kind"] == "execute_entry"]
            handoff = group[-1]
            if not executions or handoff["kind"] != "runtime_built":
                raise AssertionError((rank, cohort, "mark kinds"))
            last_exec = executions[-1]
            if last_exec["total_scheduled_tokens"] != 96 or last_exec["new"]:
                raise AssertionError((rank, cohort, "final handoff mark"))
            if handoff["req_ids"] != [
                item["req_id"] for item in last_exec["cached"]
            ]:
                raise AssertionError((rank, cohort, "handoff IDs"))
            call_path = ROOT / "marks" / f"rank{rank}_cohort{cohort}_calls.json"
            input_hashes[str(call_path)] = digest(call_path)
            capture = load_json(call_path)
            if capture["rank"] != rank or capture["cohort"] != cohort:
                raise AssertionError((rank, cohort, "call identity"))
            calls = capture["calls"]
            expected_calls = phases["cohorts"][cohort - 5]["calls"]
            if len(calls) != len(expected_calls):
                raise AssertionError((rank, cohort, "phase call count"))
            lo, hi = executions[0]["t_ns"], handoff["t_ns"]
            intervals = []
            last_index = 0
            for call, expected_call in zip(calls, expected_calls):
                index = call["execute_mark_count"]
                if not last_index < index < len(executions):
                    raise AssertionError((rank, cohort, "execute ordinal"))
                last_index = index
                if call["num_tokens_padded"] != expected_call["num_tokens_padded"]:
                    raise AssertionError((rank, cohort, "padded shape"))
                start, end = call["host_start_ns"], call["host_end_ns"]
                if not lo <= start <= end <= hi:
                    raise AssertionError((rank, cohort, "call outside envelope"))
                if not executions[index - 1]["t_ns"] <= start <= end <= executions[index]["t_ns"]:
                    raise AssertionError((rank, cohort, "call outside execute ordinal"))
                intervals.append((start, end))
            fwd_union = merged_ns(intervals)
            if fwd_union != sum(end - start for start, end in intervals):
                raise AssertionError((rank, cohort, "overlapping Host calls"))
            if not intervals or intervals[-1][1] > last_exec["t_ns"]:
                raise AssertionError((rank, cohort, "last call vs final execute"))
            rows.append({
                "rank": rank,
                "cohort": cohort,
                "request_count": len(handoff["req_ids"]),
                "handoff_req_ids": handoff["req_ids"],
                "execute_count": len(executions),
                "forward_call_count": len(calls),
                "first_execute_ns": lo,
                "last_execute_ns": last_exec["t_ns"],
                "last_forward_host_end_ns": intervals[-1][1],
                "runtime_built_ns": hi,
                "first_execute_to_runtime_built_ms": (hi - lo) / 1e6,
                "forward_host_union_ms": fwd_union / 1e6,
                "outside_forward_host_envelope_ms": (hi - lo - fwd_union) / 1e6,
                "last_forward_end_to_last_execute_ms": (last_exec["t_ns"] - intervals[-1][1]) / 1e6,
                "last_execute_to_runtime_built_ms": (hi - last_exec["t_ns"]) / 1e6,
                "pre_handoff_scheduler_output_counts": [
                    item["num_output_tokens"] for item in last_exec["cached"]
                ],
            })
    cohorts = []
    for cohort in range(5, 9):
        part = [r for r in rows if r["cohort"] == cohort]
        if len(part) != 8 or any(r["request_count"] != 12 for r in part):
            raise AssertionError((cohort, "rank/slot count"))
        if any(r["handoff_req_ids"] != part[0]["handoff_req_ids"] for r in part[1:]):
            raise AssertionError((cohort, "all8 handoff ID/order mismatch"))
        if part[0]["handoff_req_ids"] != scheduler["cohorts"][cohort - 5]["runtime_req_ids"]:
            raise AssertionError((cohort, "scheduler admission IDs"))
        counts = [r["pre_handoff_scheduler_output_counts"] for r in part]
        if any(x != counts[0] for x in counts[1:]):
            raise AssertionError((cohort, "all8 pre-handoff count mismatch"))
        first = min(r["first_execute_ns"] for r in part)
        last = max(r["runtime_built_ns"] for r in part)
        cohorts.append({
            "cohort": cohort,
            "allrank_host_envelope_ms": (last - first) / 1e6,
            "rank_local_forward_host_union_ms": [r["forward_host_union_ms"] for r in part],
            "rank_local_outside_forward_host_envelope_ms": [
                r["outside_forward_host_envelope_ms"] for r in part
            ],
            "rank_local_last_execute_to_runtime_built_ms": [
                r["last_execute_to_runtime_built_ms"] for r in part
            ],
            "pre_handoff_scheduler_output_tokens_sum": sum(counts[0]),
            "pre_handoff_scheduler_output_tokens_range": [min(counts[0]), max(counts[0])],
        })
    result = {
        "status": "same_W0_Run341_Host_envelope_only",
        "scope": "Per-rank Host entry→runtime_built split by nonoverlapping _model_forward Host spans. Includes scheduler, DSpark seed, waits, build, and other work in outside-forward remainder; does not classify them or measure device completion. Cached num_output_tokens are scheduler state counts that may include async placeholders, not certified generated or client-published outputs.",
        "not_numeric_bound": True,
        "input_sha256": input_hashes,
        "rows": rows,
        "cohorts": cohorts,
    }
    (OUT / "host_envelope.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(cohorts, indent=2))


if __name__ == "__main__":
    main()
