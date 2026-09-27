#!/usr/bin/env python3
"""Reduce one admitted selected frontier without inventing device/Host joins."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


def need(ok, message):
    if not ok:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def admitted_ledger_paths(root):
    server = load(root / "server_admission.json")
    need(server["status"] == "server_two_phase_admitted", "server admission")
    processes = server["ledger_processes"]
    need(len(processes) == len({x["pid"] for x in processes}),
         "duplicate admitted ledger pid")
    expected = {root / "ledger" / f"pid{x['pid']}.jsonl" for x in processes}
    expected_footers = {root / "ledger" / f"pid{x['pid']}.flush.jsonl"
                        for x in processes}
    files = set((root / "ledger").glob("pid*.jsonl"))
    data = {p for p in files if re.fullmatch(r"pid[0-9]+\.jsonl", p.name)}
    footers = {p for p in files if re.fullmatch(r"pid[0-9]+\.flush\.jsonl", p.name)}
    need(files == data | footers and data == expected and
         footers == expected_footers, "ledger/footer file sets differ from server admission")
    for row in processes:
        path = root / "ledger" / f"pid{row['pid']}.jsonl"
        need(sha(path) == row["ledger_sha256"],
             f"ledger bytes differ from server admission pid{row['pid']}")
        footer = root / "ledger" / f"pid{row['pid']}.flush.jsonl"
        need(sha(footer) == row["footer_sha256"],
             f"ledger footer differs from server admission pid{row['pid']}")
    return sorted(expected)


def selected_drains(ledger_paths, cohort):
    rows = {}
    for path in ledger_paths:
        with path.open() as handle:
            for line in handle:
                if '"event":"runtime_post_drain"' not in line:
                    continue
                row = json.loads(line)
                if row["cohort"] == cohort and row["phase"] == "measured":
                    rank = row["rank"]
                    need(rank not in rows, f"duplicate drain rank{rank}")
                    rows[rank] = row
    need(set(rows) == set(range(8)), "selected all8 drain missing")
    return rows


def first_full_cycle(history, slot, target=1024):
    count = 0
    for cycle, row in enumerate(history, 1):
        count += int(row[slot])
        if count >= target:
            return cycle
    raise ValueError(f"slot{slot} never reaches {target}")


def bracket(points, crossing_cycle, total_cycles):
    markers = []
    for point in points:
        label = point["label"]
        if label == "before_cycle_1":
            cycle = 0
        elif label.startswith("after_cycle_"):
            cycle = int(label.removeprefix("after_cycle_"))
        else:
            raise ValueError(f"unknown history marker {label}")
        markers.append((cycle, float(point["elapsed_ms"])))
    expected = [0] + list(range(8, total_cycles + 1, 8))
    if total_cycles % 8:
        expected.append(total_cycles)
    need([cycle for cycle, _ in markers] == expected,
         "history marker coverage/stride differs from source rule")
    before = max((x for x in markers if x[0] < crossing_cycle), default=None)
    after = min((x for x in markers if x[0] >= crossing_cycle), default=None)
    need(before is not None and after is not None, "history crossing not bracketed")
    return dict(before_cycle=before[0], after_cycle=after[0],
                same_rank_current_stream_elapsed_ms_interval=[before[1], after[1]],
                bracket_width_cycles=after[0] - before[0],
                bracket_width_ms=after[1] - before[1])


def reduce(root):
    admission = load(root / "final_admission.json")
    need(admission["status"] == "selected_frontier_diagnostic_integrity_admitted",
         "Run has no admitted frontier integrity")
    need(admission["source_restored"] is True and admission["service_stopped"] is True,
         "final environment gate")
    admitted_files = {(x["rank"], x["cohort"]): x for x in admission["files"]}
    need(len(admitted_files) == 16, "final frontier file manifest")
    for (rank, cohort), entry in admitted_files.items():
        path = root / "frontier" / f"rank{rank}_cohort{cohort}.json"
        need(sha(path) == entry["sha256"], f"frontier file changed {rank}/{cohort}")
    release_path = root / "frontier_client_admission.json"
    need(sha(release_path) == admission["c12_release_admission_sha256"],
         "c12 release report changed")
    release_report = load(release_path)
    need(sha(root / "measured_client" / "summary.json") ==
         release_report["summary_sha256"], "measured client summary changed")
    ledger_paths = admitted_ledger_paths(root)
    predecessor = selected_drains(ledger_paths, 5)
    successor = selected_drains(ledger_paths, 6)
    need(len({tuple(x["req_ids"]) for x in predecessor.values()}) == 1 and
         len({tuple(x["req_ids"]) for x in successor.values()}) == 1,
         "selected all8 request mapping")
    for cohort, drains in ((5, predecessor), (6, successor)):
        for rank, drain in drains.items():
            runtime = load(root / "runtime" / f"rank{rank}_cohort{cohort}.json")
            need(drain["req_ids"] == runtime["req_ids"] ==
                 admitted_files[(rank, cohort)]["req_ids"],
                 f"admitted request join {rank}/{cohort}")
            need(drain["cycles"] == runtime["cycles"] == len(drain["count_history"]) and
                 all(len(row) == 12 for row in drain["count_history"]),
                 f"count/cycle dimensions {rank}/{cohort}")
    first = predecessor[0]
    crossing = [first_full_cycle(first["count_history"], slot)
                for slot in range(12)]
    chosen_slot = min(range(12), key=lambda slot: (crossing[slot], slot))
    rank_brackets = []
    for rank in range(8):
        drain = predecessor[rank]
        need(drain["req_ids"][chosen_slot] == first["req_ids"][chosen_slot],
             "selected slot request mismatch")
        rank_cross = first_full_cycle(drain["count_history"], chosen_slot)
        frontier = load(root / "frontier" / f"rank{rank}_cohort5.json")
        history = next(x for x in frontier["event_groups"]
                       if x["group"] == "history" and x["generation"] == 5)
        rank_brackets.append(dict(rank=rank, full_1024_runtime_history_crossing_cycle=rank_cross,
                                  **bracket(history["points"], rank_cross,
                                            drain["cycles"])))
    target = []
    seed = []
    selected_successors = set(successor[0]["req_ids"])
    for rank in range(8):
        frontier = load(root / "frontier" / f"rank{rank}_cohort6.json")
        group = next(x for x in frontier["event_groups"]
                     if x["group"] == "first_target" and x["generation"] == 6)
        points = {x["label"]: float(x["elapsed_ms"]) for x in group["points"]}
        target.append(dict(rank=rank,
                           before_prepare_to_before_target_current_stream_ms=
                           points["before_target_consumer"] - points["before_prepare"],
                           before_target_to_after_acceptance_current_stream_ms=
                           points["after_acceptance_current_stream"] -
                           points["before_target_consumer"],
                           scope="instrumented same-rank current stream; does not certify completion of unjoined side work"))
        rows = frontier["rows"]
        calls = {}
        for row in rows:
            call = row.get("call")
            if call is not None:
                calls.setdefault(call, []).append(row)
        matched = []
        for call, call_rows in sorted(calls.items()):
            events = [x["event"] for x in call_rows]
            need(len(events) == len(set(events)),
                 f"duplicate call/event rank{rank} call{call}")
            prep = next((x for x in call_rows if x["event"] == "prepare_ids_before"), None)
            if prep is None:
                continue
            hits = sorted(set(prep["req_ids"]) & selected_successors)
            if not hits:
                continue
            forward = next((x for x in call_rows if x["event"] == "forward_before"), None)
            if forward is not None:
                need(forward["req_ids"] == prep["req_ids"],
                     f"forward/prepare request mismatch rank{rank} call{call}")
            dcp = next((x for x in call_rows if x["event"] == "dcp_result"), None)
            copy = next((x for x in call_rows if x["event"] == "draft_copy_return"), None)
            matched.append(dict(call=call, successor_req_ids=hits,
                                scheduled={k: prep["scheduled"][k] for k in hits},
                                input_branch=prep["branch"],
                                prompt_upload=prep["prompt_upload"],
                                draft_scatter_possible=prep["draft_scatter_possible"],
                                dcp_rebuilt=None if dcp is None else dcp["rebuilt"],
                                dcp_positions_ready_on_device=None if dcp is None
                                else dcp["positions_ready_on_device"],
                                ordinary_forward_actual_tokens=None if forward is None
                                else forward["actual_tokens"],
                                ordinary_forward_padded_tokens=None if forward is None
                                else forward["padded_tokens"],
                                draft_copy_source_predicate=None if copy is None
                                else copy["copy_branch_source_predicate"]))
        seed.append(dict(rank=rank, matched_calls=matched,
                         unjoined_edges=["DCP CPU callback preparation generation",
                                         "actual side-stream/HCCL and KV writer completion",
                                         "first Target storage value generation"]))
    client = load(root / "measured_client" / "summary.json")
    by_index = {x["i"]: x for x in client["requests"]}
    selected_indices = admission["selected_client_indices"]
    predecessor_release = [by_index[i]["c12_release_completed_by_monotonic_ns"]
                           for i in selected_indices["5"]]
    successor_acquire = [by_index[i]["c12_acquired_monotonic_ns"]
                         for i in selected_indices["6"]]
    return dict(status="selected_frontier_reduced_conditional",
                current_formal_tps=571.681,
                bound_endpoints=dict(algorithm_resource_tps=None,
                                     hardware_resource_tps=None,
                                     scheduling_execution_tps=None,
                                     product_e2e_tps=None),
                selected_predecessor=dict(cohort=5, slot=chosen_slot,
                    request_id=first["req_ids"][chosen_slot],
                    full_1024_runtime_history_crossing_cycle_rank0=crossing[chosen_slot],
                    rank_local_history_completion_brackets=rank_brackets,
                    interpretation="conservative complete Runtime history prefix; no early API publication or KV retirement"),
                successor=dict(cohort=6, req_ids=successor[0]["req_ids"],
                               first_target_current_stream_by_rank=target,
                               branch_calls_by_rank=seed),
                c12_observed_client=dict(predecessor_release_completed_by_ns_range=
                                         [min(predecessor_release), max(predecessor_release)],
                                         successor_acquired_ns_range=
                                         [min(successor_acquire), max(successor_acquire)],
                                         unique_permit_parent=False,
                                         counterfactual_early_release_legal=None,
                                         scope="descriptive timestamps in one client clock; release_completed_by is an upper bracket, not a parent edge"),
                unresolved=["safe individual output publication and ASGI/client receipt",
                            "slot/KV/state lifetime and release under continuing cohort",
                            "residual prefill, seed and Target producer/consumer joins",
                            "all8 resource-contended mixed service under a legal window",
                            "A0/A1 matched controls before Current timing transfer",
                            "strict compulsory work/traffic and exact-board cumulative capacity"],
                evidence_scope="same-run diagnostic; no cross-rank device clock subtraction, no formal TPS transfer")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = reduce(args.run_dir)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(status=result["status"],
                          slot=result["selected_predecessor"]["slot"],
                          rank_brackets=len(result["selected_predecessor"]["rank_local_history_completion_brackets"]))))


if __name__ == "__main__":
    main()
