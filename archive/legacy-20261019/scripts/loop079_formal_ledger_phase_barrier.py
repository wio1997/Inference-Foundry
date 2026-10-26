#!/usr/bin/env python3
"""Require a complete warmup48 Host/client barrier before phase transition."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from loop079_formal_ledger_server import verify_flushes


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger-dir", required=True, type=Path)
    ap.add_argument("--client-report", required=True, type=Path)
    ap.add_argument("--phase-marker", required=True, type=Path)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    report = json.loads(args.client_report.read_text())
    if report["status"] != "warmup48_client_admitted" or report["request_count"] != 48 or report["unique_response_ids"] != 48:
        raise ValueError("warmup client admission incomplete")
    response_ids = {row["response_id"] for row in report["request_index"]}
    if len(response_ids) != 48:
        raise ValueError("warmup client response IDs not unique")
    marker_raw = args.phase_marker.read_bytes()
    marker = json.loads(marker_raw)
    if marker["run_id"] != args.run_id or marker["phase"] != "warmup" or marker["generation"] != 0:
        raise ValueError("wrong warmup marker")
    marker_sha = sha(marker_raw)
    files = sorted(p for p in args.ledger_dir.glob("pid*.jsonl")
                   if re.fullmatch(r"pid\d+\.jsonl", p.name))
    if not files:
        raise ValueError("no server ledger files")
    footer_files = sorted(args.ledger_dir.glob("pid*.flush.jsonl"))
    if len(footer_files) != len(files) or {
            p.name.replace(".flush.jsonl", ".jsonl") for p in footer_files
    } != {p.name for p in files}:
        raise ValueError("server ledger/footer file set mismatch")
    rows = []
    integrity = {}
    for file in files:
        match = re.fullmatch(r"pid(\d+)\.jsonl", file.name)
        if match is None:
            raise ValueError("unexpected ledger filename")
        pid = int(match.group(1))
        integrity[str(pid)] = verify_flushes(args.ledger_dir, pid)
        footers = [json.loads(line) for line in
                   (args.ledger_dir / f"pid{pid}.flush.jsonl").read_text().splitlines()]
        if not all(f.get("pid") == pid and f.get("run_id") == args.run_id
                   and f.get("committed_records", 0) > 0 for f in footers):
            raise ValueError(f"flush footer identity mismatch: {file}")
        process_rows = [json.loads(line) for line in file.read_text().splitlines()]
        if not process_rows or [r["seq"] for r in process_rows] != list(range(len(process_rows))):
            raise ValueError(f"noncontiguous process sequence: {file}")
        origin = process_rows[0]
        if origin["event"] != "clock_origin" or origin["pid"] != pid or origin["run_id"] != args.run_id:
            raise ValueError(f"clock origin missing/mismatched: {file}")
        for field in ("boot_id", "time_namespace", "time_namespace_offsets"):
            if not isinstance(origin.get(field), str) or not origin[field].strip():
                raise ValueError(f"clock identity missing: {file}/{field}")
        previous_ns = origin["monotonic_ns"]
        for row in process_rows[1:]:
            if row["pid"] != pid or row["monotonic_ns"] < previous_ns or row["event"] == "clock_origin":
                raise ValueError(f"process identity/order mismatch: {file}")
            previous_ns = row["monotonic_ns"]
            if (row["run_id"] != args.run_id or row["phase"] != "warmup"
                    or row["phase_generation"] != 0 or row["phase_marker_sha256"] != marker_sha
                    or row["phase_run_id_matches"] is not True):
                raise ValueError(f"wrong server event phase: {file}")
            rows.append(row)
    output_add = [r for r in rows if r["event"] == "output_add_request"]
    internal_to_external = {r["request_id"]: r["external_req_id"] for r in output_add}
    if (len(output_add) != 48 or len(internal_to_external) != 48
            or set(internal_to_external.values()) != response_ids):
        raise ValueError("Output request mapping is not a 48-request bijection")
    api_done = [r for r in rows if r["event"] == "api_serialized_yield" and r["payload"] == "data: [DONE]\n\n"]
    done_ids = [r["external_req_id"] for r in api_done]
    if len(done_ids) != 48 or set(done_ids) != response_ids:
        raise ValueError("API DONE/client response mismatch")
    output_done = [r for r in rows if r["event"] == "output_queue" and r["finished"]]
    output_ids = [r["external_req_id"] for r in output_done]
    if len(output_ids) != 48 or set(output_ids) != response_ids:
        raise ValueError("Output finished/client response mismatch")
    expected = {(rank, cohort) for rank in range(8) for cohort in range(1, 5)}
    by_event = {}
    for event in ("runtime_handoff", "runtime_post_drain", "runner_done"):
        matched = [r for r in rows if r["event"] == event]
        keys = [(r["rank"], r["cohort"]) for r in matched]
        if len(keys) != 32 or set(keys) != expected:
            raise ValueError(f"all8 warmup cohort coverage failed: {event}")
        if event == "runner_done" and not all(
                r["host_mirror_exact"] and "FULL" in r["target_graph_mode"]
                for r in matched):
            raise ValueError("Host mirror/FULL graph mismatch")
        by_event[event] = {(r["rank"], r["cohort"]): r for r in matched}
    all_internal_ids = []
    for cohort in range(1, 5):
        reference = by_event["runtime_handoff"][(0, cohort)]["req_ids"]
        if len(reference) != 12 or len(set(reference)) != 12:
            raise ValueError("cohort has wrong request cardinality")
        all_internal_ids.extend(reference)
        for rank in range(8):
            key = (rank, cohort)
            for event in by_event:
                if by_event[event][key]["req_ids"] != reference:
                    raise ValueError(f"all8/Host request join mismatch: {event}/{key}")
            drained = by_event["runtime_post_drain"][key]
            done = by_event["runner_done"][key]
            if (len(drained["retained_raw_ids"]) != 12
                    or any(len(tokens) != 1024 for tokens in drained["retained_raw_ids"])
                    or done["generated_output_counts"] != [1024] * 12):
                raise ValueError(f"Runtime output incomplete: {key}")
    if len(all_internal_ids) != 48 or set(all_internal_ids) != set(internal_to_external):
        raise ValueError("Runtime cohorts do not cover exactly client-mapped requests")
    bulk = [r for r in rows if r["event"] == "scheduler_append" and r["bulk"]]
    bulk_ids = [r["request_id"] for r in bulk]
    if (len(bulk_ids) != 48 or set(bulk_ids) != set(internal_to_external)
            or not all(r["prefix_equal"] and r["stopped"]
                       and not r["stale"] and not r["resumable"]
                       and r["max_tokens"] == 1024 and r["g_after"] == 1024
                       and r["g_after"] - r["g_before"] == r["append_delta"]
                       and r["append_delta"] == len(r["admitted_raw_ids"])
                       for r in bulk)):
        raise ValueError("Scheduler terminal bulk incomplete or non-prefix")
    for row in rows:
        event = row["event"]
        if event in ("scheduler_append", "output_receive") and row["request_id"] not in internal_to_external:
            raise ValueError(f"unmatched internal request: {event}")
        if event in ("output_add_request", "output_queue", "api_serialized_yield"):
            external = row.get("external_req_id")
            if external not in response_ids:
                raise ValueError(f"unmatched external request: {event}")
        if event == "output_queue" and (
                row["request_id"] not in internal_to_external
                or row["external_req_id"] != internal_to_external[row["request_id"]]
                or row["output_request_id"] != row["external_req_id"]):
            raise ValueError("Output queue identity mismatch")
        if event == "api_consume" and (
                row["output_request_id"] not in response_ids
                or row["output_request_id"] != row["api_request_id"]):
            raise ValueError("API consume external identity mismatch")
    result = {
        "status": "warmup48_phase_barrier_pass",
        "run_id": args.run_id,
        "client_report_sha256": sha(args.client_report.read_bytes()),
        "phase_marker_sha256": marker_sha,
        "server_ledger_files": len(files),
        "server_ledger_integrity": integrity,
        "server_event_rows": len(rows),
        "api_done_requests": 48,
        "output_finished_requests": 48,
        "all8_rank_cohorts": 32,
        "mapped_internal_requests": 48,
        "scheduler_terminal_bulk_requests": 48,
        "scope": "Host/client completion barrier; device-ready and fresh Target work remain unobserved",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "requests": 48, "rank_cohorts": 32}))


if __name__ == "__main__":
    main()
