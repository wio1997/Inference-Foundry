"""Mechanical H1 trace reduction; upper bounds cannot prove late readiness.

Only observed RPC publication before an extra schedule is positive evidence.
The proposed removable time remains a bound, not an E2E gain or device bubble.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def identity(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=h.hexdigest())


def reduce(run):
    files, rows, raw_index = [], [], []
    for path in sorted((run / "host_trace").glob("*.jsonl")):
        entries = [json.loads(line) for line in path.read_text().splitlines()]
        files.append(dict(**identity(path), rows=len(entries),
                          events=dict(Counter(e["event"] for e in entries))))
        for line, entry in enumerate(entries, 1):
            entry = dict(entry, raw_path=str(path), raw_line=line)
            rows.append(entry)
    core = sorted((e for e in rows if e["role"] == "core"), key=lambda e: e["monotonic_ns"])
    publications = defaultdict(list)
    for entry in rows:
        if entry["event"] == "reply.published_upper_bound" and entry.get("batch", 0) > 0:
            publications[(entry["batch"], entry.get("rpc"))].append(entry)
    steps, active = [], []
    for entry in core:
        if entry["event"] == "step.begin":
            active = [entry]
        elif active:
            active.append(entry)
            if entry["event"] == "step.end":
                steps.append(active)
                active = []
    cases = []
    for step in steps:
        schedules = [e for e in step if e["event"] == "schedule.begin"]
        outputs = [e for e in step if e["event"] == "schedule.output"]
        commits = [e for e in step if e["event"] == "update_from_output.begin"]
        if not schedules or not outputs or not commits:
            continue
        schedule, new_batch, commit = schedules[0], outputs[0]["batch"], commits[0]
        if commit["batch"] >= new_batch:
            continue
        needed = [e for e in step if e["event"] == "get_response.begin"
                  and e["monotonic_ns"] < commit["monotonic_ns"]]
        # One selected output-rank response per RPC on this no-KV native path.
        proof, missing = [], []
        for consume in needed:
            replies = publications[(consume["batch"], consume["rpc"])]
            roles = {e["role"] for e in replies}
            if len(roles) != 1:
                missing.append(dict(batch=consume["batch"], rpc=consume["rpc"], roles=sorted(roles)))
                continue
            proof.append(min(replies, key=lambda e: e["monotonic_ns"]))
        ready_ub = max((e["monotonic_ns"] for e in proof), default=None)
        proven = bool(needed) and not missing and ready_ub <= schedule["monotonic_ns"]
        case = dict(committed_batch=commit["batch"], new_batch=new_batch,
                    required_rpc_count=len(needed), missing_or_ambiguous=missing,
                    all_required_published_before_extra_schedule=proven,
                    reply_ready_upper_bound_ns=ready_ub,
                    schedule_begin_ns=schedule["monotonic_ns"], commit_begin_ns=commit["monotonic_ns"],
                    enqueue_priority_to_commit_ms=(commit["monotonic_ns"]-schedule["monotonic_ns"])/1e6,
                    proof=proof, core_raw=step)
        cases.append(case)
    clients = []
    for path in sorted(run.glob("request*.events.json")):
        value = json.loads(path.read_text())
        clients.append(dict(**identity(path), begin_monotonic_ns=value["begin_monotonic_ns"],
                            events=value["events"]))
    devices = [dict(path=str(p), bytes=p.stat().st_size) for p in sorted((run / "device_trace").rglob("*")) if p.is_file()]
    summary = dict(host_files=files, worker_roles=sorted({e["role"] for e in rows if e["role"] != "core"}),
                   core_events=len(core), committed_batches=sum(e["event"] == "commit.output" for e in core),
                   enqueue_before_older_commit_cases=len(cases),
                   proven_ready_before_extra_schedule_cases=sum(c["all_required_published_before_extra_schedule"] for c in cases),
                   trace_limit_hit=any(f["rows"] >= 16384 for f in files),
                   device_artifacts=devices,
                   conclusions="Mechanical observations only; absence of early publication does not falsify early readiness. Device dependence and E2E Gain require Sol review.")
    return summary, dict(cases=cases, clients=clients)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir()
    summary, decisive = reduce(args.run)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "decisive.json").write_text(json.dumps(decisive, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("host_files", "device_artifacts")}))
