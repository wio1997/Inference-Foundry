#!/usr/bin/env python3
"""Post-cleanup integrity gate for the selected 48+48 frontier diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def gate(root):
    status = dict(line.split("=", 1) for line in
                  (root / "cleanup_status.txt").read_text().splitlines())
    for key in ("run_exit", "stop_exit", "stop_verify_exit", "restore_exit",
                "frontier_restore_exit", "sha_exit", "sha_compare_exit",
                "script_sha_exit", "script_compare_exit", "pre_admission_exit"):
        need(status.get(key) == "0", f"cleanup {key}: {status.get(key)}")
    need((root / "source_before.sha256").read_bytes() ==
         (root / "source_after.sha256").read_bytes(), "source restore mismatch")
    need((root / "scripts_before.sha256").read_bytes() ==
         (root / "scripts_after.sha256").read_bytes(), "script drift")
    client = load(root / "client_admission.json")
    server = load(root / "server_admission.json")
    warm = load(root / "warmup_phase_barrier.json")
    releases = load(root / "frontier_client_admission.json")
    need(client["status"] == "client_two_phase_admitted" and
         client["request_count"] == 96, "full client protocol")
    need(server["status"] == "server_two_phase_admitted", "full server ledger")
    need(warm["status"] == "warmup48_phase_barrier_pass", "warmup barrier")
    need(releases["status"] == "c12_eligible_release_sets_admitted" and
         releases["successor_count"] == 36 and
         releases["unique_parent_edges_certified"] == 0,
         "c12 release candidate sets")
    need((root / "server_post_count.txt").read_text().strip() == "96",
         "server POST count")
    ledger_handoffs = {}
    ledger_done = {}
    ledger_drains = {}
    for path in (root / "ledger").glob("pid*.jsonl"):
        with path.open() as handle:
            for line in handle:
                if ('"event":"runtime_handoff"' not in line and
                        '"event":"runner_done"' not in line and
                        '"event":"runtime_post_drain"' not in line):
                    continue
                row = json.loads(line)
                if row["cohort"] in (5, 6):
                    key = (row["rank"], row["cohort"])
                    need(row["phase"] == "measured" and
                         row["phase_run_id_matches"] is True,
                         f"measured selected phase {key}")
                    destination = {"runtime_handoff": ledger_handoffs,
                                   "runner_done": ledger_done,
                                   "runtime_post_drain": ledger_drains}[row["event"]]
                    need(key not in destination, f"duplicate selected event {key}")
                    destination[key] = row["req_ids"]
    need(len(ledger_handoffs) == len(ledger_done) == len(ledger_drains) == 16 and
         ledger_handoffs == ledger_done == ledger_drains,
         "all8 selected measured handoff/drain/done join")
    first_install, first_restore = load(root / "install.json"), load(root / "restore.json")
    second_install = load(root / "frontier_install.json")
    second_restore = load(root / "frontier_restore.json")
    need(first_install["installed"] is True and first_restore["restored"] is True and
         first_restore["helper_sha_drift"] is False and
         first_install["files"] == first_restore["files"], "formal patch restore")
    need(second_install["installed"] is True and second_restore["restored"] is True and
         second_install["files"] == second_restore["files"], "frontier patch restore")
    frontier = root / "frontier"
    accepted = []
    optional_unresolved = []
    for cohort in (5, 6):
        for rank in range(8):
            path = frontier / f"rank{rank}_cohort{cohort}.json"
            row = load(path)
            need(row["rank"] == rank and row["cohort"] == cohort,
                 f"frontier rank/cohort {rank}/{cohort}")
            handoffs = [x for x in row["rows"] if x["event"] == "frontier_handoff"]
            need(len(handoffs) == 1 and handoffs[0]["rank"] == rank and
                 handoffs[0]["cohort"] == cohort and len(handoffs[0]["req_ids"]) == 12,
                 f"frontier handoff {rank}/{cohort}")
            need(handoffs[0]["req_ids"] == ledger_handoffs[(rank, cohort)],
                 f"frontier/formal handoff identity {rank}/{cohort}")
            runtime = load(root / "runtime" / f"rank{rank}_cohort{cohort}.json")
            need(runtime["req_ids"] == handoffs[0]["req_ids"] and
                 runtime["target_graph_mode"] == "FULL" and runtime["pass"] is True,
                 f"frontier/runtime request and FULL join {rank}/{cohort}")
            group_keys = [(x["group"], x["generation"])
                          for x in row["event_groups"]]
            need(len(group_keys) == len(set(group_keys)),
                 f"duplicate event group/generation rank{rank} cohort{cohort}")
            required = "history" if cohort == 5 else "first_target"
            required_groups = [x for x in row["event_groups"] if x["group"] == required]
            need(len(required_groups) == 1, f"missing/duplicate {required} events rank{rank}")
            required_group = required_groups[0]
            need(required_group["generation"] == cohort and
                 required_group["scope"] ==
                 "same-rank current-stream event relation only",
                 f"event generation/scope rank{rank} cohort{cohort}")
            points = required_group["points"]
            need(points and points[0]["elapsed_ms"] == 0.0 and
                 all(math.isfinite(p["elapsed_ms"]) and p["elapsed_ms"] >= 0
                     for p in points) and
                 all(a["elapsed_ms"] <= b["elapsed_ms"] for a, b in
                     zip(points, points[1:])), f"event order rank{rank} cohort{cohort}")
            if cohort == 5:
                need(points[0]["label"] == "before_cycle_1" and
                     points[-1]["label"] == f"after_cycle_{runtime['cycles']}",
                     f"history coverage rank{rank}")
                storage = [x for x in row["rows"] if x["event"] == "history_after_loop"]
                need(len(storage) == 1 and storage[0]["cycles"] == runtime["cycles"] and
                     storage[0]["storage"]["token_shape"][1:] == [12, 8],
                     f"history storage rank{rank}")
            else:
                need([p["label"] for p in points] ==
                     ["before_prepare", "before_target_consumer",
                      "after_acceptance_current_stream"],
                     f"first target marks rank{rank}")
                target_rows = [x for x in row["rows"] if x["event"] == "first_target_mark"]
                need(len(target_rows) == 3 and
                     [x["label"] for x in target_rows] == [p["label"] for p in points] and
                     all(x["cohort"] == 6 and x["cycle_index"] == 0
                         for x in target_rows) and
                     not any(x["target_self_replay"] for x in target_rows),
                     f"first Target branch rank{rank}")
                if not any(x["event"] == "prepare_ids_before" for x in row["rows"]):
                    optional_unresolved.append(f"rank{rank}:successor_prepare_branch")
                if not any(x["event"] == "forward_after" for x in row["rows"]):
                    optional_unresolved.append(f"rank{rank}:ordinary_forward_current_stream")
            need(row["device_ready"] == "unknown_if_private_stream_or_HCCL_unjoined" and
                 row["safe_publication"] == "unknown" and
                 row["client_permit_parent"] == "unknown", "false frontier promotion")
            accepted.append(dict(rank=rank, cohort=cohort, req_ids=handoffs[0]["req_ids"],
                                 event_groups=sorted(set(x["group"] for x in row["event_groups"])),
                                 sha256=sha(path)))
    for cohort in (5, 6):
        ids = {tuple(x["req_ids"]) for x in accepted if x["cohort"] == cohort}
        need(len(ids) == 1, f"all8 request join cohort{cohort}")
    measured_client = load(root / "measured_client" / "summary.json")
    by_response = {x["response_id"]: x["i"] for x in measured_client["requests"]}
    need(len(by_response) == 48, "measured response-ID uniqueness")
    selected_client_indices = {}
    for cohort in (5, 6):
        req_ids = next(x["req_ids"] for x in accepted
                       if x["rank"] == 0 and x["cohort"] == cohort)
        indices = []
        for request_id in req_ids:
            matches = [i for response_id, i in by_response.items()
                       if request_id.startswith(response_id + "-")]
            need(len(matches) == 1, f"client/server response identity {request_id}")
            indices.extend(matches)
        need(len(set(indices)) == 12, f"selected client index uniqueness cohort{cohort}")
        selected_client_indices[str(cohort)] = indices
    need(all(str(i) in releases["eligible_release_sets"]
             for i in selected_client_indices["6"]),
         "selected successor has no c12 candidate set")
    return dict(status="selected_frontier_diagnostic_integrity_admitted",
                scope="instrumented source/current-stream lineage only; no formal TPS or finite Bound endpoint",
                current_formal_tps_unchanged=571.681,
                files=accepted, optional_unresolved=optional_unresolved,
                selected_client_indices=selected_client_indices,
                c12_release_admission_sha256=sha(root / "frontier_client_admission.json"),
                source_restored=True, service_stopped=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    result = gate(args.run_dir)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps(dict(status=result["status"], files=len(result["files"]),
                          optional_unresolved=result["optional_unresolved"])))


if __name__ == "__main__":
    main()
