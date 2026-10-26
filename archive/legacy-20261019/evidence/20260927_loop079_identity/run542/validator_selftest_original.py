#!/usr/bin/env python3
"""Synthetic full-48+48 positive and targeted negative Host-ledger cases."""
from __future__ import annotations

import base64
import copy
import json
from pathlib import Path
import tempfile

from scripts import loop079_formal_ledger_server_validate as v


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def make_fixture(root):
    run_id = "synthetic-run"
    ledger_dir = root / "ledger"
    ledger_dir.mkdir()
    warmup_dir, measured_dir = root / "warmup", root / "measured"
    warmup_dir.mkdir()
    measured_dir.mkdir()
    warmup_report = root / "warmup_report.json"
    client_report = root / "client_report.json"
    transitions_path = root / "transitions.jsonl"
    clock = dict(kernel_boot_id="boot", time_namespace="time:[1]", timens_offsets="monotonic 0 0\n")
    origin = dict(event="clock_origin", run_id=run_id, pid=111, seq=0, monotonic_ns=1,
                  boot_id="boot", time_namespace="time:[1]", time_namespace_offsets="monotonic 0 0")
    rows = [origin]
    report_rows = []
    markers = []
    for phase, generation in (("warmup", 0), ("measured", 1)):
        marker = dict(run_id=run_id, phase=phase, generation=generation,
                      previous_marker_sha256=(None if not markers else markers[0][1]),
                      completed_client_report=(None if generation == 0 else str(warmup_report)),
                      completed_client_report_sha256=(None if generation == 0 else v.sha(warmup_report.read_bytes())),
                      controller_monotonic_ns=100 if generation == 0 else 10000000,
                      controller_pid=222)
        marker_raw = (json.dumps(marker, sort_keys=True, separators=(",", ":")) + "\n").encode()
        markers.append((marker, v.sha(marker_raw)))
        for cohort in (range(1, 5) if phase == "warmup" else range(5, 9)):
            ids = [f"{phase}-internal-{(cohort - (1 if phase == 'warmup' else 5))*12 + slot}"
                   for slot in range(12)]
            token_row = [7] * 8
            histories = [[[token_row[:] for _ in range(12)] for _ in range(128)],
                         [[8] * 12 for _ in range(128)]]
            for rank in range(8):
                slots = [dict(slot=slot, request_id=rid) for slot, rid in enumerate(ids)]
                payloads = [
                    dict(event="runtime_handoff", rank=rank, cohort=cohort, req_ids=ids, slots=slots),
                ] + [
                    dict(event="serving_host_cycle", rank=rank, cohort=cohort, cycle=cycle,
                         progress=[8 * (cycle + 1)] * 12,
                         parked_before=[cycle > 0 and cycle == 127] * 12,
                         parked_after=[cycle == 127] * 12)
                    for cycle in range(128)
                ] + [
                    dict(event="runtime_post_drain", rank=rank, cohort=cohort, req_ids=ids,
                         width=8, cycles=128, initial_positions=[0] * 12,
                         progress_baseline=[0] * 12, initial_output_counts=[0] * 12,
                         remaining=[1024] * 12, token_history=histories[0], count_history=histories[1],
                         retained_raw_ids=[[7] * 1024 for _ in range(12)], staged_counts=[1024] * 12,
                         device_ready="unknown", prefill_device_complete="unknown", seed_device_ready="unknown"),
                    dict(event="runner_done", rank=rank, cohort=cohort, req_ids=ids,
                         cycles=128, host_mirror_exact=True, generated_output_counts=[1024] * 12,
                         target_graph_mode="CUDAGraphMode.FULL"),
                ]
                for item in payloads:
                    rows.append(dict(item, run_id=run_id, pid=111, seq=len(rows),
                                     monotonic_ns=(1000 if generation == 0 else 10000001) + len(rows),
                                     phase=phase, phase_generation=generation,
                                     phase_marker_sha256=markers[generation][1], phase_run_id_matches=True))
        for i in range(48):
            internal = f"{phase}-internal-{i}"
            external = f"{phase}-external-{i}"
            g = 1 if i == 0 else 0
            raw = ([99] if g else []) + [7] * (1024 - g)
            items = [
                dict(event="output_add_request", request_id=internal, external_req_id=external),
            ]
            if g:
                items.append(dict(event="scheduler_append", request_id=internal, bulk=False,
                                  prefix_equal=True, incoming_raw_ids=[99], admitted_raw_ids=[99],
                                  append_delta=1, g_before=0, g_after=1, max_tokens=1024,
                                  generation_object_id=100 + i,
                                  stale=False, resumable=False, stale_tokens=0, stopped=False))
            items += [
                dict(event="scheduler_append", request_id=internal, bulk=True, prefix_equal=True,
                     incoming_raw_ids=[7] * 1024, admitted_raw_ids=[7] * (1024 - g),
                     append_delta=1024 - g, g_before=g, g_after=1024, max_tokens=1024,
                     generation_object_id=100 + i,
                     stale=False, resumable=False, stale_tokens=0, stopped=True),
                dict(event="output_receive", request_id=internal, external_req_id=external,
                     raw_ids=raw, finished=True),
                dict(event="output_queue", request_id=internal, external_req_id=external,
                     output_request_id=external, choices=[dict(index=0, raw_ids=raw)], finished=True),
                dict(event="api_consume", output_request_id=external, api_request_id=external,
                     choice=0, raw_ids=raw, cumulative=1024),
            ]
            for item in items:
                rows.append(dict(item, run_id=run_id, pid=111, seq=len(rows),
                                 monotonic_ns=(1000 if generation == 0 else 10000001) + len(rows),
                                 phase=phase, phase_generation=generation,
                                 phase_marker_sha256=markers[generation][1], phase_run_id_matches=True))
            payloads = [f'data: {{"id":"{external}","choices":[]}}\n\n',
                        f'data: {{"id":"{external}","choices":[],"usage":{{"completion_tokens":1024}}}}\n\n',
                        "data: [DONE]\n\n"]
            events = []
            for payload in payloads:
                rows.append(dict(event="api_serialized_yield", external_req_id=external,
                                 payload=payload, payload_sha256=v.sha(payload.encode()), run_id=run_id,
                                 pid=111, seq=len(rows),
                                 monotonic_ns=(1000 if generation == 0 else 10000001) + len(rows),
                                 phase=phase, phase_generation=generation,
                                 phase_marker_sha256=markers[generation][1], phase_run_id_matches=True))
                events.append(dict(payload_b64=base64.b64encode(payload.encode()[6:-2]).decode()))
            dump((warmup_dir if phase == "warmup" else measured_dir) / f"request_{i:03d}.json",
                 dict(row=dict(response_id=external), events=events))
            report_rows.append(dict(phase=phase, response_id=external, dataset_index=i))
        if generation == 0:
            dump(warmup_report, dict(status="warmup48_client_admitted", request_count=48,
                                     request_index=copy.deepcopy(report_rows)))
    transitions_path.write_bytes(b"".join((json.dumps(m, sort_keys=True, separators=(",", ":")) + "\n").encode()
                                          for m, _ in markers))
    dump(client_report, dict(status="client_two_phase_admitted", request_count=96,
                             unique_response_ids=96, request_index=report_rows,
                             warmup_summary=dict(clock=clock, wall_end_monotonic_ns=9990000),
                             measured_summary=dict(wall_start_monotonic_ns=10000001)))
    by_pid = {111: []}
    for row in rows[1:]:
        pid = 1000 + row["rank"] if row["event"] in (
            "runtime_handoff", "serving_host_cycle", "runtime_post_drain", "runner_done") else 111
        by_pid.setdefault(pid, []).append(row)
    for pid, process_rows in by_pid.items():
        own_origin = dict(origin, pid=pid)
        own_rows = [own_origin]
        for row in process_rows:
            own_rows.append(dict(row, pid=pid, seq=len(own_rows)))
        blob = b"".join((json.dumps(r, separators=(",", ":")) + "\n").encode() for r in own_rows)
        (ledger_dir / f"pid{pid}.jsonl").write_bytes(blob)
        footer = dict(run_id=run_id, pid=pid, flush_index=0, reason="fixture",
                      committed_records=len(own_rows), first_seq=0, last_seq=len(own_rows) - 1,
                      byte_offset_begin=0, byte_offset_end=len(blob), blob_sha256=v.sha(blob))
        dump(ledger_dir / f"pid{pid}.flush.jsonl", footer)
    return dict(ledger_dir=ledger_dir, client_report=client_report,
                warmup_report=warmup_report, warmup_dir=warmup_dir,
                measured_dir=measured_dir, transitions_path=transitions_path, run_id=run_id)


def expect_reject(call, needle):
    try:
        call()
    except ValueError as exc:
        assert needle in str(exc), (needle, str(exc))
    else:
        raise AssertionError(f"negative case accepted: {needle}")


def mutate_ledger(kw, pid, predicate, change):
    path = kw["ledger_dir"] / f"pid{pid}.jsonl"
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    target = next(r for r in rows if predicate(r))
    change(target)
    blob = b"".join((json.dumps(r, separators=(",", ":")) + "\n").encode() for r in rows)
    path.write_bytes(blob)
    footer_path = kw["ledger_dir"] / f"pid{pid}.flush.jsonl"
    footer = json.loads(footer_path.read_text())
    footer["byte_offset_end"] = len(blob)
    footer["blob_sha256"] = v.sha(blob)
    dump(footer_path, footer)


def main():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        kw = make_fixture(root)
        good = v.validate(**kw)
        assert good["status"] == "server_two_phase_admitted" and good["phases"]["measured"]["requests"] == 48
        assert good["phases"]["warmup"]["bulk_clipped_requests"] == 1
        negatives = 0
        ledger = kw["ledger_dir"] / "pid111.jsonl"
        intact = ledger.read_bytes()
        ledger.write_bytes(intact[:-1])
        expect_reject(lambda: v.validate(**kw), "bad footer end")
        negatives += 1
        ledger.write_bytes(intact)
        footer = kw["ledger_dir"] / "pid111.flush.jsonl"
        intact_footer = footer.read_bytes()
        footer.write_bytes(intact_footer[:-1])
        expect_reject(lambda: v.validate(**kw), "incomplete footer")
        negatives += 1
        footer.write_bytes(intact_footer)
        warm = kw["warmup_dir"] / "request_000.json"
        intact_client = warm.read_bytes()
        client = json.loads(intact_client)
        client["events"][0]["payload_b64"] = base64.b64encode(b"changed").decode()
        dump(warm, client)
        expect_reject(lambda: v.validate(**kw), "exact API/client SSE bytes")
        negatives += 1
        warm.write_bytes(intact_client)
        trans = kw["transitions_path"]
        intact_trans = trans.read_bytes()
        transitions = [json.loads(line) for line in intact_trans.splitlines()]
        transitions[1]["completed_client_report_sha256"] = "0" * 64
        trans.write_bytes(b"".join((json.dumps(m, sort_keys=True, separators=(",", ":")) + "\n").encode()
                                   for m in transitions))
        expect_reject(lambda: v.validate(**kw), "warmup barrier linkage")
        negatives += 1
        trans.write_bytes(intact_trans)
        intact_ledger = ledger.read_bytes()
        intact_footer = footer.read_bytes()
        mutate_ledger(kw, 111, lambda r: r.get("event") == "api_consume" and
                      r.get("api_request_id") == "warmup-external-0",
                      lambda r: r.update(raw_ids=[7] * 1024))
        expect_reject(lambda: v.validate(**kw), "API raw output mismatch")
        negatives += 1
        ledger.write_bytes(intact_ledger)
        footer.write_bytes(intact_footer)
        mutate_ledger(kw, 111, lambda r: r.get("event") == "api_consume" and
                      r.get("api_request_id") == "warmup-external-0",
                      lambda r: r.update(cumulative=1023))
        expect_reject(lambda: v.validate(**kw), "API cumulative mismatch")
        negatives += 1
        ledger.write_bytes(intact_ledger)
        footer.write_bytes(intact_footer)
        mutate_ledger(kw, 111, lambda r: r.get("event") == "scheduler_append" and
                      r.get("request_id") == "warmup-internal-0" and r.get("bulk"),
                      lambda r: r.update(g_before=0))
        expect_reject(lambda: v.validate(**kw), "Scheduler G/prefix/status replay")
        negatives += 1
        ledger.write_bytes(intact_ledger)
        footer.write_bytes(intact_footer)
        rank_path = kw["ledger_dir"] / "pid1000.jsonl"
        rank_footer = kw["ledger_dir"] / "pid1000.flush.jsonl"
        intact_rank, intact_rank_footer = rank_path.read_bytes(), rank_footer.read_bytes()
        mutate_ledger(kw, 1000, lambda r: r.get("event") == "runtime_post_drain" and
                      r.get("cohort") == 1,
                      lambda r: r["retained_raw_ids"][0].__setitem__(0, 8))
        expect_reject(lambda: v.validate(**kw), "Runtime replay/retained output mismatch")
        negatives += 1
        rank_path.write_bytes(intact_rank)
        rank_footer.write_bytes(intact_rank_footer)
        mutate_ledger(kw, 111, lambda r: r.get("event") == "scheduler_append" and
                      r.get("request_id") == "warmup-internal-0" and r.get("bulk"),
                      lambda r: r.update(generation_object_id=-1))
        expect_reject(lambda: v.validate(**kw), "Scheduler generation object changed")
        negatives += 1
        ledger.write_bytes(intact_ledger)
        footer.write_bytes(intact_footer)
        mutate_ledger(kw, 1000, lambda r: r.get("event") == "runner_done" and
                      r.get("cohort") == 5,
                      lambda r: r.update(target_graph_mode="CUDAGraphMode.NONE"))
        expect_reject(lambda: v.validate(**kw), "Runner did not report FULL target graph")
        negatives += 1
        rank_path.write_bytes(intact_rank)
        rank_footer.write_bytes(intact_rank_footer)
        print(json.dumps(dict(status="pass", positives=1, negatives=negatives,
                              phase_requests=96, all8_rank_cohorts=64,
                              clipped_G_positive=good["phases"]["warmup"]["prebulk_g_max"])))


if __name__ == "__main__":
    main()
