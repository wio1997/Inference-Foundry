#!/usr/bin/env python3
"""Fail-closed Host lineage reducer for the instrumented 48+48 diagnostic.

This validates observed Host events, not fresh device work or a formal TPS run.
The client report must already have passed loop079_formal_ledger_client_validate.
"""
from __future__ import annotations

import argparse
import base64
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def unique(rows, key, count, label):
    keys = [key(row) for row in rows]
    require(len(rows) == count and len(set(keys)) == count, f"{label}: cardinality/duplicates")
    return dict(zip(keys, rows))


def read_processes(root, run_id):
    files = sorted(root.glob("pid*.jsonl"))
    data_files = [p for p in files if re.fullmatch(r"pid[0-9]+\.jsonl", p.name)]
    footer_files = [p for p in files if re.fullmatch(r"pid[0-9]+\.flush\.jsonl", p.name)]
    require(data_files and len(files) == len(data_files) + len(footer_files), "unexpected/missing ledger files")
    require({p.name.replace(".jsonl", "") for p in data_files} ==
            {p.name.replace(".flush.jsonl", "") for p in footer_files}, "PID/footer set mismatch")
    all_rows, summaries, clock = [], [], None
    for path in data_files:
        pid = int(path.name[3:-6])
        blob = path.read_bytes()
        footer_path = root / f"pid{pid}.flush.jsonl"
        footer_blob = footer_path.read_bytes()
        require(footer_blob.endswith(b"\n"), f"PID {pid}: incomplete footer line")
        footers = [json.loads(line) for line in footer_blob.splitlines()]
        require(footers, f"PID {pid}: no flush footer")
        offset, seq, process_rows = 0, 0, []
        for index, footer in enumerate(footers):
            require(footer.get("run_id") == run_id and footer.get("pid") == pid and
                    footer.get("flush_index") == index and footer.get("byte_offset_begin") == offset,
                    f"PID {pid}: footer identity/index/offset")
            stop = footer.get("byte_offset_end")
            require(type(stop) is int and stop > offset and stop <= len(blob), f"PID {pid}: bad footer end")
            segment = blob[offset:stop]
            require(segment.endswith(b"\n") and sha(segment) == footer.get("blob_sha256"),
                    f"PID {pid}: segment hash/framing")
            rows = [json.loads(line) for line in segment.splitlines()]
            require(len(rows) == footer.get("committed_records") and rows and
                    rows[0].get("seq") == footer.get("first_seq") == seq and
                    rows[-1].get("seq") == footer.get("last_seq") == seq + len(rows) - 1,
                    f"PID {pid}: footer count/sequence")
            require(all(row.get("pid") == pid and row.get("run_id") == run_id and
                        row.get("seq") == seq + j for j, row in enumerate(rows)),
                    f"PID {pid}: row identity/sequence")
            process_rows.extend(rows)
            seq += len(rows)
            offset = stop
        require(offset == len(blob), f"PID {pid}: uncommitted trailing bytes")
        origin = process_rows[0]
        require(origin.get("event") == "clock_origin" and origin.get("seq") == 0,
                f"PID {pid}: missing clock origin")
        identity = tuple(origin.get(k) for k in ("boot_id", "time_namespace", "time_namespace_offsets"))
        require(all(isinstance(v, str) and v for v in identity), f"PID {pid}: incomplete clock identity")
        if clock is None:
            clock = identity
        require(identity == clock, f"PID {pid}: different monotonic clock scope")
        previous_ns = origin["monotonic_ns"]
        for row in process_rows[1:]:
            require(row.get("event") != "clock_origin" and
                    type(row.get("monotonic_ns")) is int and row["monotonic_ns"] >= previous_ns,
                    f"PID {pid}: process clock/order")
            previous_ns = row["monotonic_ns"]
        all_rows.extend(process_rows[1:])
        summaries.append(dict(pid=pid, records=len(process_rows), flushes=len(footers),
                              ledger_sha256=sha(blob), footer_sha256=sha(footer_blob)))
    return all_rows, summaries, clock


def replay_runtime(row):
    n = len(row["req_ids"])
    width, cycles = row["width"], row["cycles"]
    require(n == 12 and type(width) is int and 1 <= width <= 8 and
            type(cycles) is int and cycles > 0, "Runtime shape/width/cycles")
    for field in ("initial_positions", "progress_baseline", "initial_output_counts",
                  "remaining", "retained_raw_ids", "staged_counts"):
        require(len(row[field]) == n, f"Runtime {field} shape")
    counts, tokens = row["count_history"], row["token_history"]
    require(len(counts) == len(tokens) == cycles, "Runtime history cycles")
    actual = [[] for _ in range(n)]
    staged = [0] * n
    for c in range(cycles):
        require(len(counts[c]) == len(tokens[c]) == n, "Runtime history slot shape")
        for slot in range(n):
            count = counts[c][slot]
            token_row = tokens[c][slot]
            require(type(count) is int and 0 <= count <= width and len(token_row) == width,
                    "Runtime count/token width")
            accepted = token_row[:count]
            require(all(type(t) is int and t >= 0 for t in accepted), "Runtime accepted token invalid")
            staged[slot] += count
            need = row["remaining"][slot] - len(actual[slot])
            if need > 0:
                actual[slot].extend(accepted[:need])
    require(actual == row["retained_raw_ids"] and staged == row["staged_counts"] and
            row["remaining"] == [1024] * n and all(len(x) == 1024 for x in actual),
            "Runtime replay/retained output mismatch")
    require(row.get("device_ready") == row.get("prefill_device_complete") ==
            row.get("seed_device_ready") == "unknown", "unclaimed device-ready field changed")
    return actual


def validate_phase(phase, rows, index_rows, client_dir):
    response_ids = [r["response_id"] for r in index_rows]
    require(len(response_ids) == len(set(response_ids)) == 48, f"{phase}: client IDs")
    expected_ids = set(response_ids)
    adds = unique([r for r in rows if r["event"] == "output_add_request"],
                  lambda r: r["request_id"], 48, f"{phase}: Output add")
    mapping = {internal: r["external_req_id"] for internal, r in adds.items()}
    require(set(mapping.values()) == expected_ids and len(set(mapping.values())) == 48,
            f"{phase}: internal/external mapping")
    cohorts = range(1, 5) if phase == "warmup" else range(5, 9)
    events = {}
    for name in ("runtime_handoff", "runtime_post_drain", "runner_done"):
        events[name] = unique([r for r in rows if r["event"] == name],
                              lambda r: (r["rank"], r["cohort"]), 32, f"{phase}: {name}")
        require(set(events[name]) == {(rank, cohort) for rank in range(8) for cohort in cohorts},
                f"{phase}: all8 cohort coverage")
    slot_map = {}
    cycles_by_key = defaultdict(list)
    for row in rows:
        if row["event"] == "serving_host_cycle":
            cycles_by_key[(row["rank"], row["cohort"])].append(row)
    require(set(cycles_by_key) == set(events["runtime_handoff"]),
            f"{phase}: Host cycle all8 coverage")
    for cohort in cohorts:
        ids = events["runtime_handoff"][(0, cohort)]["req_ids"]
        require(len(ids) == len(set(ids)) == 12, f"{phase}: cohort membership")
        for rank in range(8):
            key = rank, cohort
            hand, drain, done = (events[name][key] for name in
                                 ("runtime_handoff", "runtime_post_drain", "runner_done"))
            require(hand["pid"] == drain["pid"] == done["pid"] and
                    all(r["pid"] == hand["pid"] for r in cycles_by_key[key]),
                    f"{phase}: rank/cohort process ownership")
            require(hand["req_ids"] == drain["req_ids"] == done["req_ids"] == ids and
                    [s["request_id"] for s in hand["slots"]] == ids and
                    [s["slot"] for s in hand["slots"]] == list(range(12)),
                    f"{phase}: all8 slot/request join")
            retained = replay_runtime(drain)
            cycle_rows = sorted(cycles_by_key[key], key=lambda r: r["seq"])
            require(len(cycle_rows) == drain["cycles"] and
                    [r["cycle"] for r in cycle_rows] == list(range(drain["cycles"])),
                    f"{phase}: Host cycle history completeness")
            for cycle_row in cycle_rows:
                require(len(cycle_row["progress"]) == len(cycle_row["parked_before"]) ==
                        len(cycle_row["parked_after"]) == 12 and
                        all(type(x) is bool for x in cycle_row["parked_before"] + cycle_row["parked_after"]),
                        f"{phase}: Host cycle slot shape")
                require(all((not before) or after for before, after in
                            zip(cycle_row["parked_before"], cycle_row["parked_after"])),
                        f"{phase}: Host parked state regressed")
            require(all(cycle_rows[-1]["progress"][slot] - drain["progress_baseline"][slot]
                        >= drain["remaining"][slot] for slot in range(12)),
                    f"{phase}: final committed progress incomplete")
            require(done["host_mirror_exact"] is True and done["generated_output_counts"] == [1024] * 12 and
                    done["cycles"] == drain["cycles"], f"{phase}: runner completion/mirror")
            if phase == "measured":
                require(done["target_graph_mode"] == "CUDAGraphMode.FULL",
                        f"{phase}: Runner did not report FULL target graph")
            if rank == 0:
                for slot, request_id in enumerate(ids):
                    require(request_id not in slot_map, f"{phase}: repeated internal request")
                    slot_map[request_id] = retained[slot]
            else:
                reference = events["runtime_post_drain"][(0, cohort)]
                require(retained == reference["retained_raw_ids"] and
                        drain["count_history"] == reference["count_history"] and
                        drain["token_history"] == reference["token_history"],
                        f"{phase}: all8 Runtime parity")
        require(len({events["runtime_handoff"][(rank, cohort)]["pid"] for rank in range(8)}) == 8,
                f"{phase}: all8 ranks share a process")
    require(set(slot_map) == set(mapping), f"{phase}: Runtime/Output mapped requests")
    scheduler = unique([r for r in rows if r["event"] == "scheduler_append" and r["bulk"]],
                       lambda r: r["request_id"], 48, f"{phase}: Scheduler bulk")
    appends = defaultdict(list)
    for row in rows:
        if row["event"] == "scheduler_append":
            require(row["request_id"] in mapping, f"{phase}: Scheduler unknown request")
            appends[row["request_id"]].append(row)
    require(set(appends) == set(mapping), f"{phase}: Scheduler request coverage")
    formal_raw = {}
    for internal, entry in scheduler.items():
        ordered = sorted(appends[internal], key=lambda r: (r["pid"], r["seq"]))
        require(len({r["pid"] for r in ordered}) == 1 and ordered[-1] is entry and
                sum(int(r["bulk"]) for r in ordered) == 1,
                f"{phase}: terminal bulk placement/process")
        g = 0
        full = []
        generation_ids = {append["generation_object_id"] for append in ordered}
        require(len(generation_ids) == 1, f"{phase}: Scheduler generation object changed")
        for j, append in enumerate(ordered):
            admitted = append["admitted_raw_ids"]
            require(append["g_before"] == g and append["g_after"] == g + len(admitted) and
                    append["append_delta"] == len(admitted) and append["prefix_equal"] is True and
                    admitted == append["incoming_raw_ids"][:len(admitted)] and
                    append["max_tokens"] == 1024 and append["stale"] is False and
                    append["resumable"] is False and append["stale_tokens"] == 0 and
                    (j == len(ordered) - 1 or append["stopped"] is False),
                    f"{phase}: Scheduler G/prefix/status replay")
            if append["bulk"]:
                require(append["incoming_raw_ids"] == slot_map[internal] and
                        append["stopped"] is True and
                        admitted == slot_map[internal][:len(admitted)] and
                        len(admitted) == 1024 - g,
                        f"{phase}: terminal bulk retained-prefix ownership")
            full.extend(admitted)
            g += len(admitted)
        require(g == 1024, f"{phase}: Scheduler incomplete formal output")
        formal_raw[internal] = full
    for name in ("output_receive", "output_queue", "api_consume"):
        require(any(r["event"] == name for r in rows), f"{phase}: missing {name}")
    receives = defaultdict(list)
    queues = defaultdict(list)
    yields = defaultdict(list)
    consumes = defaultdict(list)
    for row in rows:
        event = row["event"]
        if event == "output_receive":
            require(row["request_id"] in mapping and row["external_req_id"] == mapping[row["request_id"]],
                    f"{phase}: Output receive identity")
            receives[row["request_id"]].append(row)
        elif event == "output_queue":
            require(row["request_id"] in mapping and row["external_req_id"] == mapping[row["request_id"]] and
                    row["output_request_id"] == row["external_req_id"], f"{phase}: Output queue identity")
            queues[row["request_id"]].append(row)
        elif event == "api_serialized_yield":
            require(row["external_req_id"] in expected_ids, f"{phase}: API yield identity")
            yields[row["external_req_id"]].append(row)
        elif event == "api_consume":
            require(row["output_request_id"] in expected_ids and row["api_request_id"] == row["output_request_id"],
                    f"{phase}: API consume identity")
            consumes[row["output_request_id"]].append(row)
    require(set(receives) == set(queues) == set(mapping) and set(yields) == set(consumes) == expected_ids,
            f"{phase}: Output/API coverage")
    for internal, raw in formal_raw.items():
        recv = sorted(receives[internal], key=lambda r: (r["pid"], r["seq"]))
        require(len({r["pid"] for r in recv}) == 1 and
                sum((r["raw_ids"] for r in recv), []) == raw and
                sum(int(r["finished"]) for r in recv) == 1 and recv[-1]["finished"],
                f"{phase}: Output receive raw sequence")
        queue = sorted(queues[internal], key=lambda r: (r["pid"], r["seq"]))
        require(len({r["pid"] for r in queue}) == 1 and
                sum(int(r["finished"]) for r in queue) == 1 and queue[-1]["finished"],
                f"{phase}: Output queue completion")
        choice_chunks = [choice for q in queue for choice in q["choices"]]
        require(all(x["index"] == 0 for x in choice_chunks) and
                sum((x["raw_ids"] for x in choice_chunks), []) == raw,
                f"{phase}: Output queue raw sequence")
        external = mapping[internal]
        api_rows = sorted(consumes[external], key=lambda r: (r["pid"], r["seq"]))
        running = 0
        collected = []
        for consume in api_rows:
            require(consume["choice"] == 0 and
                    all(type(token) is int and token >= 0 for token in consume["raw_ids"]),
                    f"{phase}: API raw token/choice")
            running += len(consume["raw_ids"])
            require(consume["cumulative"] == running, f"{phase}: API cumulative mismatch")
            collected.extend(consume["raw_ids"])
        require(running == 1024 and collected == raw,
                f"{phase}: API raw output mismatch")
    for item in index_rows:
        external = item["response_id"]
        client = json.loads((client_dir / f"request_{item['dataset_index']:03d}.json").read_text())
        require(client["row"]["response_id"] == external, f"{phase}: client report/request mismatch")
        server = sorted(yields[external], key=lambda r: (r["pid"], r["seq"]))
        require(len({r["pid"] for r in server}) == 1 and len(server) == len(client["events"]),
                f"{phase}: API yield cardinality/process")
        for s, c in zip(server, client["events"]):
            payload = s["payload"].encode("utf-8")
            require(sha(payload) == s["payload_sha256"] and payload.startswith(b"data: ") and
                    payload.endswith(b"\n\n") and payload[6:-2] == base64.b64decode(c["payload_b64"], validate=True),
                    f"{phase}: exact API/client SSE bytes")
        require(server[-1]["payload"] == "data: [DONE]\n\n" and
                sum(s["payload"] == "data: [DONE]\n\n" for s in server) == 1,
                f"{phase}: API DONE")
        require(len({r["pid"] for r in consumes[external]}) == 1,
                f"{phase}: API consume process")
    return dict(requests=48, rank_cohorts=32, raw_retained_tokens=48 * 1024,
                api_exact_sse_requests=48, scheduler_bulk_requests=48,
                ordinary_append_requests=sum(len(appends[r]) > 1 for r in appends),
                bulk_clipped_requests=sum(scheduler[r]["g_before"] > 0 for r in scheduler),
                prebulk_g_max=max(scheduler[r]["g_before"] for r in scheduler))


def validate(ledger_dir, client_report, warmup_report, warmup_dir, measured_dir,
             transitions_path, run_id):
    report_raw = client_report.read_bytes()
    report = json.loads(report_raw)
    require(report.get("status") == "client_two_phase_admitted" and report.get("request_count") == 96 and
            report.get("unique_response_ids") == 96 and len(report.get("request_index", [])) == 96,
            "client two-phase admission missing")
    transition_bytes = transitions_path.read_bytes()
    require(transition_bytes.endswith(b"\n"), "phase transition file incomplete")
    transitions = [json.loads(line) for line in transition_bytes.splitlines()]
    require(len(transitions) == 2 and [r["phase"] for r in transitions] == ["warmup", "measured"] and
            [r["generation"] for r in transitions] == [0, 1] and
            all(r["run_id"] == run_id for r in transitions), "phase transitions wrong")
    marker_sha = [sha((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode())
                  for row in transitions]
    warmup_report_raw = warmup_report.read_bytes()
    warmup_admission = json.loads(warmup_report_raw)
    require(warmup_admission.get("status") == "warmup48_client_admitted" and
            warmup_admission.get("request_count") == 48 and
            warmup_admission.get("request_index") ==
            [r for r in report["request_index"] if r["phase"] == "warmup"] and
            transitions[1]["previous_marker_sha256"] == marker_sha[0] and
            transitions[1]["completed_client_report_sha256"] == sha(warmup_report_raw) and
            transitions[1]["completed_client_report"] == str(warmup_report),
            "measured transition lacks warmup barrier linkage")
    rows, process_summaries, server_clock = read_processes(ledger_dir, run_id)
    client_clock = report["warmup_summary"]["clock"]
    require((client_clock.get("kernel_boot_id"), client_clock.get("time_namespace"),
             (client_clock.get("timens_offsets") or "").strip()) == server_clock,
            "client/server monotonic clock identity mismatch")
    phase_rows = {name: [] for name in ("warmup", "measured")}
    for row in rows:
        phase = row.get("phase")
        require(phase in phase_rows and row.get("phase_generation") == (0 if phase == "warmup" else 1) and
                row.get("phase_marker_sha256") == marker_sha[0 if phase == "warmup" else 1] and
                row.get("phase_run_id_matches") is True,
                "unmatched server phase/marker")
        phase_rows[phase].append(row)
    for summary in process_summaries:
        pid_rows = sorted((r for r in rows if r["pid"] == summary["pid"]), key=lambda r: r["seq"])
        require([r["phase_generation"] for r in pid_rows] ==
                sorted(r["phase_generation"] for r in pid_rows), "phase regressed within process")
    result = {}
    for name, client_dir in (("warmup", warmup_dir), ("measured", measured_dir)):
        index = [r for r in report["request_index"] if r["phase"] == name]
        result[name] = validate_phase(name, phase_rows[name], index, client_dir)
    require(len(set(r["response_id"] for r in report["request_index"])) == 96,
            "cross-phase external ID reuse")
    warm_internal = {r["request_id"] for r in phase_rows["warmup"] if r["event"] == "output_add_request"}
    measured_internal = {r["request_id"] for r in phase_rows["measured"] if r["event"] == "output_add_request"}
    require(not warm_internal.intersection(measured_internal), "cross-phase internal ID reuse")
    return dict(status="server_two_phase_admitted", run_id=run_id,
                scope="instrumented_host_lineage_not_formal_tps_or_device_freshness",
                client_report_sha256=sha(report_raw), warmup_report_sha256=sha(warmup_report_raw),
                transitions_sha256=sha(transition_bytes),
                ledger_processes=process_summaries, server_event_rows=len(rows), phases=result,
                phase_response_ids={phase: sorted(r["response_id"] for r in report["request_index"]
                                                  if r["phase"] == phase)
                                    for phase in ("warmup", "measured")},
                unresolved=["fresh_retained_Target_row", "device_prefill_seed_ready",
                            "typed_collective_ready", "resource_critical_path", "memoization_policy",
                            "controller_cross_process_clock_identity", "Host_progress_count_alignment"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledger-dir", type=Path, required=True)
    ap.add_argument("--client-report", type=Path, required=True)
    ap.add_argument("--warmup-report", type=Path, required=True)
    ap.add_argument("--warmup-dir", type=Path, required=True)
    ap.add_argument("--measured-dir", type=Path, required=True)
    ap.add_argument("--phase-transitions", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    result = validate(a.ledger_dir, a.client_report, a.warmup_report, a.warmup_dir, a.measured_dir,
                      a.phase_transitions, a.run_id)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "events": result["server_event_rows"]}))


if __name__ == "__main__":
    main()
