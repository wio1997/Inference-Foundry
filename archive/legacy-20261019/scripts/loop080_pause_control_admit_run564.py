#!/usr/bin/env python3
"""Admit one Run564 A or B arm without promoting a fixed-W0 conclusion."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from check_loop035_trace import check as check_trace


FIELDS = (
    "num_computed_before", "last_token_before", "draft_before",
    "target_input_ids", "target_positions", "target_argmax",
    "accepted", "counts", "next_draft",
)


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def file_digest(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def admit(arm: Path, mode: str, cohort: int):
    client = json.loads((arm / "client_admission.json").read_text())
    assert client["status"] == "client_two_phase_admitted"
    assert client["request_count"] == 96 and client["unique_response_ids"] == 96
    assert client["same_48_request_bodies_across_phases"] is True
    by_response = {row["response_id"]: row for row in client["request_index"]}
    assert len(by_response) == 96
    post_count = int((arm / "server_post_count.txt").read_text().strip())
    assert post_count == 96

    cohort_rows = []
    trace_shas = []
    pause_shas = []
    segment_signatures = []
    semantic_keys_by_cohort = {}
    trace_file_shas = {}
    runtime_file_shas = {}
    pause_file_shas = {}
    for rank in range(8):
        runtime_files = sorted((arm / "runtime").glob(f"rank{rank}_cohort*.json"))
        assert len(runtime_files) == 8, (rank, len(runtime_files))
        assert {path.name for path in runtime_files} == {
            f"rank{rank}_cohort{index}.json" for index in range(1, 9)}
        for path in runtime_files:
            row = json.loads(path.read_text())
            assert row["rank"] == rank and row["pass"] is True
            assert row["host_mirror_exact"] is True
            assert row["generated_output_counts"] == [1024] * 12
            runtime_file_shas[str(path.relative_to(arm))] = file_digest(path)
        for measured_cohort in range(5, 9):
            runtime_path = arm / "runtime" / f"rank{rank}_cohort{measured_cohort}.json"
            row = json.loads(runtime_path.read_text())
            assert row["cohort"] == measured_cohort and row["cycles"] > 0
            ids = row["req_ids"]
            assert len(ids) == len(set(ids)) == 12
            keys = []
            for request_id in ids:
                matches = [row for response_id, row in by_response.items()
                           if request_id.startswith(response_id + "-")]
                assert len(matches) == 1, "client/worker request identity is not unique"
                request = matches[0]
                assert request["phase"] == "measured"
                keys.append((request["dataset_index"], request["dataset_row_sha256"],
                             request["request_body_sha256"]))
            known_keys = semantic_keys_by_cohort.setdefault(str(measured_cohort), keys)
            assert known_keys == keys, "cross-rank request/slot order drift"
            trace_path = arm / "trace" / f"trace_rank{rank}_cohort{measured_cohort}.json"
            trace = json.loads(trace_path.read_text())
            assert len(trace) == row["cycles"], "trace truncated or extra cycles"
            assert all(set(FIELDS).issubset(cycle) for cycle in trace)
            check_trace(trace_path, runtime_path)
            trace_file_shas[str(trace_path.relative_to(arm))] = file_digest(trace_path)
            if measured_cohort == cohort:
                trace_shas.append(digest(trace))
                cohort_rows.append(dict(rank=rank, cycles=row["cycles"],
                                        runtime_wall_seconds=row["wall_seconds"],
                                        trace_sha256=trace_shas[-1]))

        row = json.loads((arm / "runtime" / f"rank{rank}_cohort{cohort}.json").read_text())
        ids = row["req_ids"]
        trace = json.loads((arm / "trace" / f"trace_rank{rank}_cohort{cohort}.json").read_text())

        pause_path = arm / "trace" / f"pause_rank{rank}_cohort{cohort}.json"
        if mode == "on":
            pause = json.loads(pause_path.read_text())
            pause_file_shas[str(pause_path.relative_to(arm))] = file_digest(pause_path)
            segment, trajectory = pause["segment"], pause["trajectory"]
            assert segment["generation"] == trajectory["generation"] == cohort
            assert segment["request_ids"] == trajectory["request_ids"] == ids
            assert len(segment["slots"]) == len(segment["token_ids"]) == 1
            assert 0 <= segment["slots"][0] < 12
            assert len(segment["token_ids"][0]) == 1024
            assert 0 < segment["cycle"] < row["cycles"]
            assert math.isfinite(pause["resume_pause_ms"]) and pause["resume_pause_ms"] >= 0
            assert trajectory["cycles"] == row["cycles"]
            counts, sampled = trajectory["counts"], trajectory["sampled"]
            assert len(counts) == len(sampled) == row["cycles"]
            staged = [0] * 12
            zeroed = [False] * 12
            for cycle, (cycle_counts, cycle_sampled) in enumerate(zip(counts, sampled)):
                assert len(cycle_counts) == len(cycle_sampled) == 12
                for slot, (count, tokens) in enumerate(zip(cycle_counts, cycle_sampled)):
                    assert isinstance(count, int) and 0 <= count <= 8
                    assert len(tokens) == count
                    if count:
                        assert not zeroed[slot], "parked slot resumed positive output"
                        assert count == trace[cycle]["counts"][slot]
                        assert tokens == trace[cycle]["accepted"][slot][:count]
                        staged[slot] += count
                    else:
                        assert staged[slot] >= 1024, "zero count before slot completion"
                        zeroed[slot] = True
            assert all(value >= 1024 for value in staged)
            reconstructed = []
            for slot in range(12):
                output = [token for cycle_sampled in sampled
                          for token in cycle_sampled[slot]]
                reconstructed.append(output[:1024])
                assert len(reconstructed[-1]) == 1024
            early_slot = segment["slots"][0]
            early_prefix = [token for cycle_sampled in sampled[:segment["cycle"]]
                            for token in cycle_sampled[early_slot]][:1024]
            assert len(early_prefix) == 1024
            assert early_prefix == segment["token_ids"][0] == reconstructed[early_slot]
            assert trajectory["sha256"] == digest({
                "generation": cohort, "request_ids": ids,
                "counts": counts, "sampled": sampled,
            })
            assert segment["output_digest"] == digest({
                "generation": cohort, "request_ids": ids,
                "slots": segment["slots"], "cycle": segment["cycle"],
                "token_ids": segment["token_ids"],
            })
            pause_shas.append(trajectory["sha256"])
            segment_signatures.append((segment["cycle"], tuple(segment["slots"]),
                                       segment["output_digest"]))
        else:
            assert not pause_path.exists(), "control arm contains a pause"

    assert len(list((arm / "trace").glob("trace_rank*_cohort*.json"))) == 32
    assert len(list((arm / "trace").glob("pause_rank*_cohort*.json"))) == (8 if mode == "on" else 0)
    assert len({tuple(key) for keys in semantic_keys_by_cohort.values() for key in keys}) == 48
    assert {key[0] for keys in semantic_keys_by_cohort.values() for key in keys} == set(range(48))
    if mode == "on":
        assert len(set(pause_shas)) == 1, "all8 sampled/count ledger mismatch"
        assert len(set(segment_signatures)) == 1, "all8 pause cycle/slot/output mismatch"
    return {
        "status": "diagnostic_arm_admitted",
        "mode": mode, "cohort": cohort,
        "request_semantic_keys_by_cohort": semantic_keys_by_cohort,
        "request_semantic_sha256_by_cohort": {
            key: digest(value) for key, value in semantic_keys_by_cohort.items()
        },
        "ranks": cohort_rows, "pause_trajectory_shas": pause_shas,
        "segment_signatures": segment_signatures,
        "trace_file_sha256": trace_file_shas,
        "runtime_file_sha256": runtime_file_shas,
        "pause_file_sha256": pause_file_shas,
        "client_admission_sha256": file_digest(arm / "client_admission.json"),
        "server_post_count_sha256": file_digest(arm / "server_post_count.txt"),
        "scope": "all8 client/Runtime/trace completeness and B sampled/count join; not fixed-W0 or formal TPS",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--mode", choices=("off", "on"), required=True)
    parser.add_argument("--cohort", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = admit(args.arm, args.mode, args.cohort)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "mode": args.mode,
                      "cohort": args.cohort, "ranks": len(report["ranks"])}))


if __name__ == "__main__":
    main()
