"""Check a full fixed-cohort trace without replaying model operators."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def check(trace_path: Path, rank_path: Path, *, prefix: bool = False) -> dict:
    trace = json.loads(trace_path.read_text())
    rank = json.loads(rank_path.read_text())
    if not trace or (len(trace) != rank["cycles"] and not prefix):
        raise ValueError("trace must cover every rank0 cohort cycle")
    cumulative = [0] * 12
    verified = 0
    active_transitions = 0
    parked_transitions = 0
    for cycle, row in enumerate(trace):
        for name in ("num_computed_before", "last_token_before", "draft_before",
                     "target_input_ids", "target_positions", "target_argmax",
                     "accepted", "counts", "next_draft"):
            if name not in row:
                raise ValueError(f"cycle {cycle} missing {name}")
        for slot in range(12):
            computed = row["num_computed_before"][slot]
            last = row["last_token_before"][slot]
            draft = row["draft_before"][slot]
            ids = row["target_input_ids"][slot]
            pos = row["target_positions"][slot]
            predicted = row["target_argmax"][slot]
            actual = row["accepted"][slot]
            count = row["counts"][slot]
            if len(draft) != 7 or len(ids) != 8 or len(pos) != 8:
                raise ValueError(f"cycle {cycle} slot {slot} ABI width")
            if ids != [last] + draft or pos != list(range(computed, computed + 8)):
                raise ValueError(f"cycle {cycle} slot {slot} target input state")
            matches = 0
            while matches < 7 and draft[matches] == predicted[matches]:
                matches += 1
            expected = draft[:matches] + [predicted[matches]] + [-1] * (7 - matches)
            if actual != expected or count != matches + 1:
                raise ValueError(f"cycle {cycle} slot {slot} acceptance")
            verified += 1
            if cumulative[slot] < 1024:
                cumulative[slot] += min(count, 1024 - cumulative[slot])
            if cycle + 1 == len(trace):
                continue
            nxt = trace[cycle + 1]
            if nxt["draft_before"][slot] != row["next_draft"][slot]:
                raise ValueError(f"cycle {cycle} slot {slot} draft commit")
            if cumulative[slot] < 1024:
                if nxt["num_computed_before"][slot] != computed + count:
                    raise ValueError(f"cycle {cycle} slot {slot} computed advance")
                if nxt["last_token_before"][slot] != actual[count - 1]:
                    raise ValueError(f"cycle {cycle} slot {slot} last token advance")
                active_transitions += 1
            elif cumulative[slot] == 1024:
                if nxt["num_computed_before"][slot] > computed + count:
                    raise ValueError(f"cycle {cycle} slot {slot} parked position escaped")
                parked_transitions += 1
    if not prefix and cumulative != [1024] * 12:
        raise ValueError(f"client token coverage incomplete: {cumulative}")
    if rank["generated_output_counts"] != [1024] * 12 or not rank["pass"]:
        raise ValueError("rank0 serving gate failed")
    return {"cycles": len(trace), "verified_slot_cycles": verified,
            "active_transitions": active_transitions,
            "completion_or_park_transitions": parked_transitions,
            "client_token_coverage": cumulative,
            "scope": "product trace arithmetic and greedy acceptance only; no Stock token oracle or KV value parity"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--rank", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prefix", action="store_true")
    args = parser.parse_args()
    result = check(args.trace, args.rank, prefix=args.prefix)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
