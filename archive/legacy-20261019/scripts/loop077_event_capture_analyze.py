#!/usr/bin/env python3
"""Validate sparse original-path eight-rank event and acceptance capture."""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "evidence/20260927_loop077_bound/run386"
RUNTIME_LABELS = ["begin", "prepare_target", "derived_target_metadata", "target",
                  "acceptance", "state_advance", "proposer", "draft_commit"]
DRAFT_LABELS = ["begin", "host_mirror", "refresh_common", "prepare_inputs", "pack_hidden", "model"]


def source_check():
    before = (BASE / "source_before.sha256").read_text().splitlines()
    after = (BASE / "source_after.sha256").read_text().splitlines()
    assert before == after
    assert len(before) == 3
    return {"source_before_after_sha_match": True, "files": len(before)}


def stages(markers, labels):
    assert [x["label"] for x in markers] == labels
    before = [x["ms_before_anchor"] for x in markers]
    assert all(x >= 0 for x in before)
    assert all(before[i] >= before[i + 1] for i in range(len(before) - 1))
    return {labels[i]: before[i - 1] - before[i] for i in range(1, len(before))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    clients = {}
    for name, n in (("warmup48.json", 48), ("bench.json", 12)):
        data = json.loads((BASE / name).read_text())
        assert len(data["requests"]) == n
        assert all(x["output_tokens"] == 1024 and x["error"] is None for x in data["requests"])
        clients[name] = {"requests": n, "exact_1024": True}
    rows = []
    anchor_widths = []
    for cohort in range(1, 6):
        accepted_reference = None
        for rank in range(8):
            obj = json.loads((BASE / "events" / f"rank{rank}_cohort{cohort}.json").read_text())
            runtime = json.loads((BASE / "runtime" / f"rank{rank}_cohort{cohort}.json").read_text())
            assert obj["rank"] == runtime["rank"] == rank
            assert obj["cohort"] == runtime["cohort"] == cohort
            assert obj["cycles"] == runtime["cycles"] and runtime["pass"]
            assert [x["cycle"] for x in obj["runtime_events"]] == [64, 65]
            assert [x["cycle"] for x in obj["dspark_events"]] == [64, 65]
            counts = obj["accepted_counts_by_cycle_slot"]
            assert len(counts) == obj["cycles"] and all(len(x) == 12 and all(0 <= v <= 8 for v in x) for x in counts)
            if accepted_reference is None:
                accepted_reference = counts
            else:
                assert counts == accepted_reference
            needed = obj["remaining"]
            output = [0] * 12
            useful = []
            for count in counts:
                accepted = [min(value, max(0, needed[i] - output[i])) for i, value in enumerate(count)]
                output = [output[i] + accepted[i] for i in range(12)]
                useful.append(sum(accepted))
            assert output == needed
            width_ns = obj["anchor_after_ns"] - obj["anchor_before_ns"]
            assert 0 <= width_ns < 10_000_000_000
            anchor_widths.append(width_ns / 1e6)
            for idx, cycle in enumerate((64, 65)):
                runtime_markers = obj["runtime_events"][idx]["markers"]
                draft_markers = obj["dspark_events"][idx]["markers"]
                rs = stages(runtime_markers, RUNTIME_LABELS)
                ds = stages(draft_markers, DRAFT_LABELS)
                rows.append({"cohort": cohort, "rank": rank, "cycle": cycle,
                             "useful_clipped_tokens": useful[cycle],
                             "raw_accepted_tokens": sum(counts[cycle]),
                             "runtime_stage_ms": rs, "dspark_stage_ms": ds,
                             "cycle_selected_span_ms": runtime_markers[0]["ms_before_anchor"] - runtime_markers[-1]["ms_before_anchor"],
                             "draft_model_event_ms_before_anchor": draft_markers[-1]["ms_before_anchor"],
                             "next_draft_commit_event_ms_before_anchor": runtime_markers[-1]["ms_before_anchor"],
                             "anchor_bracket_ms": width_ns / 1e6,
                             "anchor_after_ns": obj["anchor_after_ns"]})
    assert len(rows) == 80
    by_stage = defaultdict(list)
    for row in rows:
        for name, value in row["runtime_stage_ms"].items():
            by_stage[name].append(value)
    stage_summary = {name: {"min": min(values), "median": statistics.median(values), "max": max(values)}
                     for name, values in by_stage.items()}
    output = {"status": "original_path_sparse_device_event_and_acceptance_ledger",
              "source_integrity": source_check(), "client_gates": clients, "rows": rows,
              "summary": {"rank_cycles": len(rows), "cohorts": 5, "ranks": 8,
                          "selected_cycles": [64, 65],
                          "anchor_sync_bracket_ms_range": [min(anchor_widths), max(anchor_widths)],
                          "runtime_stage_ms_across_rank_cycles": stage_summary,
                          "selected_cycle_span_ms_range": [min(x["cycle_selected_span_ms"] for x in rows), max(x["cycle_selected_span_ms"] for x in rows)]},
              "limits": ["Eight rank event durations are not an automatically aligned cross-rank makespan; Host anchor bracket and stream semantics must be checked before timestamp joins.",
                         "Markers record the runtime current stream; Target Graph has internal streams and the proposer can enqueue asynchronous tasks. Validate final producer/consumer stream joins before calling any marker a complete device endpoint.",
                         "Only two cycles per cohort have sparse event operations, still an instrumentation overhead; do not convert stage gaps to removable Product wall-time without an uninstrumented control and dependency DAG.",
                         "Useful counts are Runtime clipped, and pre-handoff/client output attribution plus prefill/seed/parking remain separate."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["summary"]))


if __name__ == "__main__":
    main()
