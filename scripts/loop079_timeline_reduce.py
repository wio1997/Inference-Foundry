#!/usr/bin/env python3
"""Validate the Run427 Host timeline and report interval-valued H cutoffs."""
import argparse
import collections
import hashlib
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def peak_concurrency(rows):
    active = peak = 0
    for _, delta in sorted((t, change) for row in rows for t, change in
                           ((row["start"], 1), (row["end"], -1))):
        active += delta
        peak = max(peak, active)
    assert active == 0
    return peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    d = a.run_dir
    assert (d / "source_before.sha256").read_bytes() == (d / "source_after.sha256").read_bytes()
    cleanup = dict(x.split("=", 1) for x in (d / "cleanup_status.txt").read_text().splitlines())
    assert all(cleanup[k] == "0" for k in
               ("run_exit", "stop_exit", "stop_verify_exit", "restore_exit", "sha_exit", "sha_compare_exit", "final_exit"))
    assert int((d / "server_post_count.txt").read_text()) == 60
    clients = [load(d / x) for x in ("warmup48.json", "bench.json")]
    assert [len(x["requests"]) for x in clients] == [48, 12]
    assert all(x["summary"]["fail"] == 0 and all(y["output_tokens"] == 1024 and y["error"] is None
               for y in x["requests"]) for x in clients)
    assert peak_concurrency([row for client in clients for row in client["requests"]]) <= 12
    reports = [load(p) for p in sorted((d / "runtime").glob("rank*_cohort*.json"))]
    assert len(reports) == 40
    assert {(x["rank"], x["cohort"]) for x in reports} == {(r,c) for r in range(8) for c in range(1,6)}
    assert all(x["pass"] and x["host_mirror_exact"] and x["target_graph_mode"] == "FULL"
               and x["model_runner_cycles_after_handoff"] == x["oracle_target_calls_after_handoff"] == 0
               and x["generated_output_counts"] == [1024]*12 for x in reports)
    cohorts = {}
    for c in range(1,6):
        rs = [x for x in reports if x["cohort"] == c]
        assert all(x["req_ids"] == rs[0]["req_ids"] and x["cycles"] == rs[0]["cycles"] for x in rs)
        cohorts[c] = rs[0]["req_ids"]
    all_ids = {r for ids in cohorts.values() for r in ids}
    assert len(all_ids) == 60

    files = sorted((d / "timeline").glob("pid*.jsonl"))
    assert files
    events = []
    for f in files:
        file_pid = int(f.stem[3:])
        rows = [json.loads(line) for line in f.read_text().splitlines()]
        assert rows and all(x["pid"] == file_pid for x in rows)
        events.extend(rows)
    assert len(events) == int((d / "timeline_row_count.txt").read_text())
    assert all(x["run_id"] == "LOOP079-RUN427" for x in events)
    origins = [x for x in events if x["event"] == "clock_origin"]
    assert origins and {x["pid"] for x in origins} == {x["pid"] for x in events}
    assert all(x["boot_id"] and x["time_namespace"] and x["time_namespace_offsets"]
               and x["clock_implementation"] for x in origins)
    assert len({(x["boot_id"], x["time_namespace"], x["time_namespace_offsets"],
                             x["clock_implementation"]) for x in origins}) == 1
    by_type = collections.defaultdict(list)
    for event in events:
        by_type[event["event"]].append(event)

    hands = by_type["runtime_handoff"]
    assert len(hands) == 40
    hand_by_cohort = {}
    for c, ids in cohorts.items():
        matched = [x for x in hands if [r["request_id"] for r in x["request_rows"]] == ids]
        assert len(matched) == 8 and {x["tp_rank"] for x in matched} == set(range(8))
        assert all(x["total_scheduled"] == 96 and x["handoff_entry_ns"] <= x["monotonic_ns"] for x in matched)
        hand_by_cohort[c] = dict(lo=min(x["handoff_entry_ns"] for x in matched),
                                 hi=max(x["monotonic_ns"] for x in matched))
    bulk = [x for x in by_type["scheduler_append"] if x["bulk"]]
    normal = [x for x in by_type["scheduler_append"] if not x["bulk"]]
    assert len(bulk) == 60 and {x["request_id"] for x in bulk} == all_ids
    assert all(x["incoming_len"] == 1024 and x["g_after"] == 1024
               and x["admitted_len"] == 1024 - x["g_before"] and x["stopped"]
               and not x["stale"] for x in bulk)
    assert {x["request_id"] for x in normal} <= all_ids

    op_recv = by_type["output_processor_receive"]
    op_queue = by_type["output_processor_queue_submitted"]
    api_consume = by_type["api_consume"]
    api_yield = by_type["api_generator_yield"]
    op_map = {}
    for x in op_recv:
        assert x["request_id"] in all_ids
        old = op_map.setdefault(x["request_id"], x["external_req_id"])
        assert old == x["external_req_id"]
    assert set(op_map) == all_ids and len(set(op_map.values())) == 60
    assert {x["request_id"] for x in op_queue} == all_ids
    assert {x["external_req_id"] for x in api_consume} == set(op_map.values())
    assert {x["external_req_id"] for x in api_yield} == set(op_map.values())

    per_request = []
    for c, ids in cohorts.items():
        h = hand_by_cohort[c]
        for req_id in ids:
            ne = sorted((x for x in normal if x["request_id"] == req_id), key=lambda x:x["before_ns"])
            be = next(x for x in bulk if x["request_id"] == req_id)
            assert ne and ne[0]["g_before"] == 0
            series = ne + [be]
            assert all(x["g_before"] <= x["g_after"] and x["admitted_len"] == x["g_after"]-x["g_before"]
                       and x["before_ns"] <= x["after_ns"] <= x["monotonic_ns"] for x in series)
            assert all(a["g_after"] == b["g_before"] and a["after_ns"] <= b["before_ns"]
                       for a,b in zip(series, series[1:]))
            assert be["before_ns"] > h["hi"]
            lower = sum(x["admitted_len"] for x in ne if x["after_ns"] < h["lo"])
            upper = sum(x["admitted_len"] for x in ne if x["before_ns"] <= h["hi"])
            assert 0 <= lower <= upper <= be["g_before"]
            external = op_map[req_id]
            oe = sorted((x for x in op_recv if x["request_id"] == req_id), key=lambda x:x["monotonic_ns"])
            qe = [x for x in op_queue if x["request_id"] == req_id]
            qe.sort(key=lambda x:x["monotonic_ns"])
            ae = sorted((x for x in api_consume if x["external_req_id"] == external), key=lambda x:x["monotonic_ns"])
            ye = sorted((x for x in api_yield if x["external_req_id"] == external), key=lambda x:x["monotonic_ns"])
            assert oe and ae and ye
            assert len(series) == len(oe) == len(qe) == len(ae)
            assert all(x["choice"] == 0 and x["request_id"] == external for x in ae+ye)
            assert all(x["request_output_id"] == external and len(x["choice_token_lens"]) == 1
                       and len(x["choice_token_sha256"]) == 1 for x in qe)
            assert sum(x["new_token_len"] for x in oe) == 1024
            assert sum(x["new_token_len"] for x in ae) == 1024
            assert ae[-1]["raw_consumed_cumulative"] == 1024
            assert all(a["raw_consumed_cumulative"] <= b["raw_consumed_cumulative"] for a,b in zip(ae,ae[1:]))
            assert all(x["request_id"] == external for x in ae+ye)
            sched_admitted = [(x["admitted_len"], x["admitted_token_sha256"]) for x in series]
            op_received = [(x["new_token_len"], x["new_token_sha256"]) for x in oe]
            assert sched_admitted == op_received
            queue_submitted = [(x["choice_token_lens"][0], x["choice_token_sha256"][0]) for x in qe]
            api_consumed = [(x["new_token_len"], x["new_token_sha256"]) for x in ae]
            assert sched_admitted == queue_submitted == api_consumed
            assert all(s["after_ns"] <= r["monotonic_ns"] <= q["monotonic_ns"] <= a["monotonic_ns"]
                       for s,r,q,a in zip(series,oe,qe,ae))
            assert all((0 if n == 0 else ae[n-1]["raw_consumed_cumulative"]) + x["new_token_len"]
                       == x["raw_consumed_cumulative"] for n,x in enumerate(ae))
            consumed_post_marker_lower = max((x["raw_consumed_cumulative"] for x in ae
                                              if x["monotonic_ns"] < h["lo"]), default=0)
            # An API consume marker is after parser work. The first marker
            # after the H envelope may still have consumed before H.
            first_after = next((x["raw_consumed_cumulative"] for x in ae
                                if x["monotonic_ns"] > h["hi"]), 1024)
            consumed_possible_by_marker = max(first_after, max((x["raw_consumed_cumulative"] for x in ae
                                                               if x["monotonic_ns"] <= h["hi"]), default=0))
            # API raw consumption cannot precede its Scheduler append. The
            # verified normal+bulk -> OutputProcessor digest chain and actual
            # source path bound A(H_probe) by committed G(H_probe).
            consumed_possible = min(consumed_possible_by_marker, upper)
            assert consumed_post_marker_lower <= consumed_possible
            pre_yield_possible = [x for x in ye if x["monotonic_ns"] <= h["hi"]]
            progress = sorted(ae + ye, key=lambda x:x["monotonic_ns"])
            # A later same-request parser/yield marker proves that the prior
            # generator yield completed and the generator resumed.
            yield_definitely_before = [x for x in ye if x["monotonic_ns"] < h["lo"]
                                       and any(z["monotonic_ns"] > x["monotonic_ns"]
                                               and z["monotonic_ns"] < h["lo"] for z in progress)]
            assert all(x["has_reasoning"] and not x["has_content"]
                       and not x["has_tool_calls"] and not x["finished"]
                       for x in yield_definitely_before)
            per_request.append(dict(cohort=c, request_id=req_id, external_req_id=external,
                                    terminal_prebulk=be["g_before"],
                                    scheduler_committed_at_global_H_interval=[lower,upper],
                                    api_raw_consumed_at_global_H_interval=[consumed_post_marker_lower,consumed_possible],
                                    api_raw_consume_upper_from_marker_only=consumed_possible_by_marker,
                                    api_generator_yield_count_at_H_interval=[len(yield_definitely_before),len(pre_yield_possible)],
                                    confirmed_reasoning_delta_yields_before_H=len(yield_definitely_before),
                                    api_pre_yield_raw_consumption_watermark_possible_at_H=max((x["raw_consumed_cumulative"] for x in pre_yield_possible),default=0),
                                    output_processor_receive_events=len(oe),
                                    output_processor_queue_events=len(qe)))
    assert len(per_request) == 60
    result = dict(schema=1, run_id="LOOP079-RUN427", status="host_timeline_valid_pending_independent_review",
                  http_posts=60, runtime_rank_cohort_reports=40, request_generations=60,
                  clock_origin={k:origins[0][k] for k in ("boot_id","time_namespace","time_namespace_offsets","clock_implementation")},
                  event_counts={k:len(v) for k,v in sorted(by_type.items())},
                  handoff_global_envelopes_ns={str(k):v for k,v in hand_by_cohort.items()},
                  scheduler_terminal_prebulk_total=sum(x["terminal_prebulk"] for x in per_request),
                  scheduler_committed_at_global_H_total_interval=[sum(x["scheduler_committed_at_global_H_interval"][i] for x in per_request) for i in (0,1)],
                  api_raw_consumed_at_global_H_total_interval=[sum(x["api_raw_consumed_at_global_H_interval"][i] for x in per_request) for i in (0,1)],
                  api_generator_yield_count_at_global_H_interval=[sum(x["api_generator_yield_count_at_H_interval"][i] for x in per_request) for i in (0,1)],
                  requests_with_confirmed_reasoning_delta_yield_before_H=sum(x["confirmed_reasoning_delta_yields_before_H"] > 0 for x in per_request),
                  per_request=per_request,
                  interpretation="H is a pre-run Host probe envelope. Scheduler G(H_probe) has true append brackets. API consumption markers occur after parser work; generator markers occur before yield, so their intervals are conservative. Raw-token publication, ASGI send/client receipt, and device-completed D(H) remain unproved. Timing perturbed by Host hooks; no formal TPS promotion.",
                  formal_tps_promotion=False,
                  timeline_file_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps({k:result[k] for k in ("status","scheduler_terminal_prebulk_total",
                                              "scheduler_committed_at_global_H_total_interval",
                                              "api_raw_consumed_at_global_H_total_interval",
                                              "api_generator_yield_count_at_global_H_interval")}))


if __name__ == "__main__":
    main()
