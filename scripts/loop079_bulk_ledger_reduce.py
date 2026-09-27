#!/usr/bin/env python3
"""Validate one frozen Host bulk-ledger diagnostic without promoting formal TPS."""
import argparse
import hashlib
import json
from pathlib import Path


def peak_concurrency(rows):
    events = []
    for row in rows:
        events += [(row["start"], 1), (row["end"], -1)]
    n = peak = 0
    for _, delta in sorted(events):
        n += delta
        peak = max(peak, n)
    assert n == 0
    return peak


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    d = a.run_dir
    assert (d / "source_before.sha256").read_bytes() == (d / "source_after.sha256").read_bytes()
    cleanup = dict(line.split("=", 1) for line in (d / "cleanup_status.txt").read_text().splitlines())
    assert all(cleanup[k] == "0" for k in ("run_exit", "stop_exit", "stop_verify_exit", "restore_exit", "sha_exit", "sha_compare_exit", "final_exit"))
    assert int((d / "server_post_count.txt").read_text()) == 60
    assert int((d / "ledger_row_count.txt").read_text()) == 60
    clients = []
    for name, count in (("warmup48.json", 48), ("bench.json", 12)):
        obj = json.loads((d / name).read_text())
        assert len(obj["requests"]) == count and obj["summary"]["success"] == count and obj["summary"]["fail"] == 0
        assert all(x["output_tokens"] == 1024 and x["error"] is None for x in obj["requests"])
        assert peak_concurrency(obj["requests"]) <= 12
        clients.append(obj)
    reports = [json.loads(x.read_text()) for x in sorted((d / "runtime").glob("rank*_cohort*.json"))]
    assert len(reports) == 40
    assert {(x["rank"], x["cohort"]) for x in reports} == {(r,c) for r in range(8) for c in range(1,6)}
    assert all(x["pass"] and x["host_mirror_exact"] and x["target_graph_mode"] == "FULL"
               and x["oracle_target_calls_after_handoff"] == x["model_runner_cycles_after_handoff"] == 0
               and x["generated_output_counts"] == [1024]*12 for x in reports)
    ids = set()
    cohort_ids = {}
    for cohort in range(1,6):
        rows = [x for x in reports if x["cohort"] == cohort]
        assert all(x["req_ids"] == rows[0]["req_ids"] and x["cycles"] == rows[0]["cycles"] for x in rows)
        cohort_ids[cohort] = rows[0]["req_ids"]
        ids.update(rows[0]["req_ids"])
    assert len(ids) == 60
    files = sorted((d / "ledger").glob("pid*.jsonl"))
    assert files
    ledger = [json.loads(line) for f in files for line in f.read_text().splitlines()]
    assert len(ledger) == 60 and {x["request_id"] for x in ledger} == ids
    assert len({x["request_id"] for x in ledger}) == 60
    assert all(x["run_id"] == "LOOP079-RUN421" and x["max_tokens"] == 1024
               and x["g_after"] - x["g_before"] == x["append_delta"] == x["admitted_len"]
               and x["timestamp_before_ns"] <= x["timestamp_after_ns"] for x in ledger)
    before = sorted(set(x["g_before"] for x in ledger))
    incoming = sorted(set(x["incoming_bulk_len"] for x in ledger))
    admitted = sorted(set(x["admitted_len"] for x in ledger))
    classes = sorted(set(x["scheduler_class"] for x in ledger))
    reset_risk = [x["request_id"] for x in ledger if x["resumable"] or x["num_stale_output_tokens"] or x["drop_stale_output"] or x["output_is_stale"]]
    by_id = {x["request_id"]: x for x in ledger}
    assert all(x["incoming_bulk_len"] == 1024 and x["g_after"] == 1024 and x["stopped"]
               and x["g_before"] + x["admitted_len"] == 1024 for x in ledger)
    cohort_accounting = [
        {"cohort": cohort, "requests": len(req_ids),
         "scheduler_generated_before_bulk": sum(by_id[r]["g_before"] for r in req_ids),
         "runtime_incoming_bulk": sum(by_id[r]["incoming_bulk_len"] for r in req_ids),
         "scheduler_admitted_from_bulk": sum(by_id[r]["admitted_len"] for r in req_ids)}
        for cohort, req_ids in sorted(cohort_ids.items())
    ]
    zero_ready = before == [0] and not reset_risk
    result = {
        "status": "zero_prior_generated_candidate_for_independent_source_review" if zero_ready else "positive_prior_generated_requires_api_publication_ledger",
        "run_id": "LOOP079-RUN421", "http_posts": 60, "client_requests": 60,
        "max_client_concurrency": max(peak_concurrency(x["requests"]) for x in clients),
        "rank_cohort_reports": 40, "unique_runtime_request_ids": len(ids),
        "scheduler_ledger_rows": len(ledger), "scheduler_classes": classes,
        "g_before_values": before, "incoming_bulk_len_values": incoming,
        "admitted_len_values": admitted, "reset_or_stale_request_ids": reset_risk,
        "scheduler_generated_before_bulk_total": sum(x["g_before"] for x in ledger),
        "runtime_incoming_bulk_total": sum(x["incoming_bulk_len"] for x in ledger),
        "scheduler_admitted_from_bulk_total": sum(x["admitted_len"] for x in ledger),
        "cohort_accounting": cohort_accounting,
        "all_bulk_append_accounting_exact": True,
        "all_external_1024_length": True,
        "zero_prior_generated_source_review_ready": zero_ready,
        "external_pre_handoff_p_i_verified": False,
        "external_p_i_scope": "Positive pre-bulk Scheduler generated counts do not identify already published tokens at handoff. Request-correlated API publication ledger is required for Run421; no retroactive Run403 or Run99 p_i inference.",
        "formal_tps_promotion": False,
        "ledger_file_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "scheduler_ledger_rows", "g_before_values", "incoming_bulk_len_values", "admitted_len_values")}))


if __name__ == "__main__":
    main()
