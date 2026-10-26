#!/usr/bin/env python3
"""Run570: admitted A0 physical-vs-active Target-row census, no timing claim."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import loop080_fixed_w0_census_run569 as prior

ROOT = Path(__file__).resolve().parents[1]
SERVING = ROOT / "runtime/fixed_serving.py"
SERVING_SHA = "137f1cc6a460700e6f26e5837aa062022eb621198e33099b85988f8b7ef6743a"
DECODE = ROOT / "runtime/extreme_decode.py"
DECODE_SHA = "eb4b5142cc475a51e06c6eec8238e9cfeb918b568e8a5f27b0a4bc7d26963499"
STATE = ROOT / "runtime/fixed_decode.py"
STATE_SHA = "7d74f4cbba380ff0a9296cc06d86e0cb9c9c921b6213b7965700831a29a6bafa"


def derive_slot(position: list[int], counts: list[int], staged: int,
                *, required: int = 1024, park_offset: int = 960) -> dict:
    assert len(position) == len(counts) and position
    assert all(isinstance(x, int) for x in position + counts)
    assert all(0 <= x <= 8 for x in counts)
    diffs = [b - a for a, b in zip(position, position[1:])]
    negative = [i for i, value in enumerate(diffs) if value < 0]
    assert len(negative) <= 1
    stop = negative[0] + 1 if negative else len(position)
    assert all(1 <= counts[i] <= 8 and diffs[i] == counts[i]
               for i in range(stop-1))
    assert 1 <= counts[stop-1] <= 8
    if negative:
        assert position[stop] == position[0] + park_offset
        assert all(value == 0 for value in diffs[stop:])
    active_sampled = sum(counts[:stop])
    assert active_sampled >= required
    assert active_sampled == staged
    return dict(active_cycles=stop, first_parked_cycle=stop if negative else None,
                parking_rewind=diffs[stop-1] if negative else None,
                current_parked_cycles=len(position)-stop,
                active_sampled_tokens=active_sampled)


def run(out: Path):
    assert hashlib.sha256(SERVING.read_bytes()).hexdigest() == SERVING_SHA
    assert hashlib.sha256(DECODE.read_bytes()).hexdigest() == DECODE_SHA
    assert hashlib.sha256(STATE.read_bytes()).hexdigest() == STATE_SHA
    admission = prior.read(prior.P["a0_admission"])
    _, trace_hashes, cohort_cycles, req_ids = prior.read_a0_trace(admission)
    rows = []
    for rank in range(8):
        for cohort in range(5, 9):
            trace_path = ROOT / prior.TRACE_DIR / f"trace_rank{rank}_cohort{cohort}.json"
            runtime_path = ROOT / prior.RUNTIME_DIR / f"rank{rank}_cohort{cohort}.json"
            trace = json.loads(trace_path.read_text())
            runtime = json.loads(runtime_path.read_text())
            assert runtime["generated_output_counts"] == [1024] * 12
            assert len(runtime["staged_output_counts"]) == 12
            for slot in range(12):
                positions = [cycle["num_computed_before"][slot] for cycle in trace]
                counts = [cycle["counts"][slot] for cycle in trace]
                evidence = derive_slot(positions, counts, runtime["staged_output_counts"][slot])
                rows.append(dict(rank=rank, cohort=cohort, slot=slot,
                    cohort_cycles=len(trace), current_physical_target_rows=8*len(trace),
                    active_target_rows_current_evaluation_class=8*evidence["active_cycles"],
                    parked_target_rows_current_physical=8*evidence["current_parked_cycles"],
                    **evidence))
    assert len(rows) == 8 * 4 * 12
    by_cohort = {}
    for cohort in range(5, 9):
        cohort_rows = [r for r in rows if r["cohort"] == cohort]
        reference = tuple((r["slot"], r["active_cycles"], r["active_sampled_tokens"])
                          for r in cohort_rows if r["rank"] == 0)
        assert len(reference) == 12
        for rank in range(1, 8):
            assert tuple((r["slot"], r["active_cycles"], r["active_sampled_tokens"])
                         for r in cohort_rows if r["rank"] == rank) == reference
        rank0 = [r for r in cohort_rows if r["rank"] == 0]
        physical = 96 * cohort_cycles[cohort]
        active = sum(r["active_target_rows_current_evaluation_class"] for r in rank0)
        parked = sum(r["parked_target_rows_current_physical"] for r in rank0)
        assert active + parked == physical
        by_cohort[str(cohort)] = dict(cycles=cohort_cycles[cohort],
            physical_target_rows_per_rank=physical,
            active_target_rows_current_evaluation_class_per_rank=active,
            parked_target_rows_current_physical_per_rank=parked,
            rank_consensus=True)
    physical = sum(x["physical_target_rows_per_rank"] for x in by_cohort.values())
    active = sum(x["active_target_rows_current_evaluation_class_per_rank"] for x in by_cohort.values())
    parked = sum(x["parked_target_rows_current_physical_per_rank"] for x in by_cohort.values())
    assert physical == active + parked
    result = dict(status="a0_diagnostic_current_row_census_not_strict_bound",
        source_sha256=dict(fixed_serving=SERVING_SHA,
            extreme_decode=DECODE_SHA, fixed_decode=STATE_SHA,
            run569_generator=hashlib.sha256(Path(prior.__file__).read_bytes()).hexdigest(),
            run566_a0_admission=prior.sha(prior.P["a0_admission"])),
        trace_sha256={f"rank{r}_cohort{c}": h for (r,c),h in sorted(trace_hashes.items())},
        request_ids_sha256_by_cohort={str(c): hashlib.sha256(json.dumps(ids).encode()).hexdigest()
            for c,ids in sorted(req_ids.items())},
        cohorts=by_cohort,
        total_per_rank=dict(physical_target_rows=physical,
            active_target_rows_current_evaluation_class=active,
            parked_target_rows_current_physical=parked,
            parked_row_fraction_of_current_physical=parked/physical),
        all8_exact_consensus=True,
        interpretation="Current fixed96 Graph executes parked slots; active-row count follows admitted same-arm position rewind/freeze and staged-count join. It is conditional on current Target-8 evaluation class, not a mathematical compulsory FLOP/HBM count or removable wall time.",
        formal_Run99_W0_certified=False, numeric_resource_bound_update=False,
        numeric_scheduling_bound_update=False, formal_e2e_update=False,
        current_formal_tps=571.681)
    out.mkdir(parents=True, exist_ok=False)
    with (out / "slot_rows.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    (out / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps(result["total_per_rank"], sort_keys=True))


def self_test():
    assert derive_slot([10,11,13,12,12],[1,2,1,5,5],4,
                       required=4,park_offset=2)["first_parked_cycle"] == 3
    assert derive_slot([10,11,13],[1,2,1],4,
                       required=4,park_offset=2)["first_parked_cycle"] is None
    for position, counts, staged in (
        ([10,10,13,12,12], [1,2,1,5,5],4),
        ([10,11,13,12,13], [1,2,1,5,5],4),
        ([10,11,13,12,12], [1,2,1,5,5],5),
        ([10,11,13,12,11], [1,2,1,5,5],4),
        ([10,12,14,12,12], [1,2,1,5,5],4),
        ([10,11,13,11,11], [1,2,1,5,5],4),
    ):
        try: derive_slot(position, counts, staged, required=4, park_offset=2)
        except AssertionError: pass
        else: raise AssertionError("accepted malformed park trajectory")
    print("Run570 active-row source gate PASS")


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--out-dir");p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    if a.self_test:self_test()
    else:
        assert a.out_dir
        run(Path(a.out_dir))
