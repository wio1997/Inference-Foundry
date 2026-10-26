#!/usr/bin/env python3
import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

TENSOR_FIELDS = (
    "block_table", "draft_token_ids", "hidden_states_buffer",
    "input_ids_buffer", "positions_buffer", "query_start_loc",
    "sample_indices", "seq_lens", "slot_mapping", "target_hidden_states",
    "target_positions", "target_token_ids",
)


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("evidence_dir", type=Path)
    ap.add_argument("--batch-size", type=int, default=12)
    args = ap.parse_args()
    root = args.evidence_dir
    rank_files = sorted((root / "raw").glob("rank*.jsonl"))
    replay_files = sorted((root / "replay").glob("rank*.json"))
    ranks = []
    for path in rank_files:
        all_rows = load_jsonl(path)
        rows = [r for r in all_rows if r["batch_size"] == args.batch_size]
        tensors = {}
        for field in TENSOR_FIELDS:
            ptrs = [r[field]["data_ptr"] for r in rows]
            shapes = [tuple(r[field]["shape"]) for r in rows]
            strides = [tuple(r[field]["stride"]) for r in rows]
            tensors[field] = {
                "observations": len(rows),
                "unique_data_ptrs": len(set(ptrs)),
                "stable_data_ptr": len(set(ptrs)) == 1,
                "unique_shapes": len(set(shapes)),
                "unique_strides": len(set(strides)),
                "shape": list(shapes[0]) if shapes and len(set(shapes)) == 1 else None,
                "stride": list(strides[0]) if strides and len(set(strides)) == 1 else None,
                "dtype": rows[0][field]["dtype"] if rows else None,
                "device": rows[0][field]["device"] if rows else None,
            }
        ranks.append({
            "rank": all_rows[0]["rank"],
            "all_observations": len(all_rows),
            "batch_histogram": {str(k): v for k, v in sorted(Counter(r["batch_size"] for r in all_rows).items())},
            "selected_batch_size": args.batch_size,
            "selected_observations": len(rows),
            "selected_num_tokens": sorted(set(r["num_tokens"] for r in rows)),
            "selected_metadata_steps": sorted(set(r["metadata_steps"] for r in rows)),
            "tensors": tensors,
        })
    fields_all_ranks_stable = [
        field for field in TENSOR_FIELDS
        if ranks and all(rank["tensors"][field]["stable_data_ptr"] for rank in ranks)
    ]
    fields_dynamic = [field for field in TENSOR_FIELDS if field not in fields_all_ranks_stable]
    pointer_summary = {
        "scope": "legacy DSpark proposer entry, warm pure-decode observations",
        "batch_size": args.batch_size,
        "rank_count": len(ranks),
        "observations_per_rank": sorted(set(r["selected_observations"] for r in ranks)),
        "all_ranks_fixed_shape_and_stride": all(
            t["unique_shapes"] == 1 and t["unique_strides"] == 1
            for rank in ranks for t in rank["tensors"].values()
        ),
        "fields_stable_address_on_all_ranks": fields_all_ranks_stable,
        "fields_with_changing_address": fields_dynamic,
        "interpretation": {
            "supported": "At c12, all observed tensor shapes/strides are fixed. Ten owner/view fields retain one address per rank across 172 proposer calls; target token and position inputs change allocation address.",
            "limit": "Address stability is process-local observational evidence, not proof of graph safety or lifetime stability. Tensor values were intentionally not copied to host.",
        },
        "ranks": ranks,
    }
    (root / "pointer_stability_summary.json").write_text(json.dumps(pointer_summary, indent=2) + "\n")

    replay = [json.loads(path.read_text()) for path in replay_files]
    first = [r["first_ms"] for r in replay]
    second = [r["replay_ms"] for r in replay]
    replay_summary = {
        "scope": "one exact in-process replay of the already-materialized c12 DSpark proposer closure on each TP rank",
        "rank_count": len(replay),
        "draft_equal_ranks": sum(bool(r["draft_equal"]) for r in replay),
        "all_ranks_draft_equal": bool(replay) and all(r["draft_equal"] for r in replay),
        "draft_shape": sorted({tuple(r["draft_shape"]) for r in replay}),
        "batch_size": sorted({r["batch_size"] for r in replay}),
        "num_tokens": sorted({r["num_tokens"] for r in replay}),
        "first_ms": {"min": min(first), "median": statistics.median(first), "max": max(first)},
        "replay_ms": {"min": min(second), "median": statistics.median(second), "max": max(second)},
        "correctness": {
            "functional_check_pass": json.loads((root / "run2" / "functional_check.json").read_text())["pass"],
            "proposer_draft_tokens_exact": bool(replay) and all(r["draft_equal"] for r in replay),
            "accepted_tokens_compared": False,
            "full_cycle_state_mutation_compared": False,
        },
        "performance_use": "diagnostic only; JSON tracing and one inserted duplicate proposer call invalidate the decode sample for baseline comparison",
        "decision": "The already-materialized proposer closure is an executable exact-replay boundary. Loop028 did not yet capture target verification, acceptance, or end-of-cycle state mutation, so it does not satisfy the full-cycle standalone-runtime gate.",
        "ranks": replay,
    }
    (root / "fixed_cycle_replay_summary.json").write_text(json.dumps(replay_summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
