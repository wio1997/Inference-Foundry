#!/usr/bin/env python3
"""Validate Run299 and classify only address risks supported by captured metadata."""

import argparse
import collections
import json
from pathlib import Path


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def owners(rank):
    return {i for i in range(12)
            if max(0, min((rank + 1) * 12, (i + 1) * 8) - max(rank * 12, i * 8))}


def source_kind(name):
    return name.split("self_attn.", 1)[1]


def analyze(root):
    bench = json.loads((root / "bench12.json").read_text())
    if bench["summary"]["success"] != 12 or any(x["output_tokens"] != 1024 for x in bench["requests"]):
        raise RuntimeError("client contract failed")
    runtime = sorted((root / "runtime").glob("rank*_cohort*.json"))
    if len(runtime) != 8 or not all(json.loads(x.read_text())["pass"] for x in runtime):
        raise RuntimeError("all-eight Runtime pass missing")
    if not all(json.loads(x.read_text())["target_graph_mode"] == "FULL" for x in runtime):
        raise RuntimeError("not all Runtime reports use FULL Graph")
    registry = [json.loads((root / "census" / f"rank{r}_registry.json").read_text()) for r in range(8)]
    samples = [rows(root / "census" / f"rank{r}.jsonl") for r in range(8)]
    expected_cycles = [0, 1, 63, 64, 127, 128, 192, 193, 255, 256]
    for rank, (reg, rr) in enumerate(zip(registry, samples)):
        if reg["rank"] != rank or reg["all_typed_leaf_count"] != 196 or reg["draft_typed_leaf_count"] != 3:
            raise RuntimeError(f"rank{rank} incomplete typed leaf registry")
        if len(reg["layer2_storage_ptrs"]) != 3 or len(reg["selected_typed_views"]) != 15:
            raise RuntimeError(f"rank{rank} layer2 alias registry changed")
        if reg["draft_overlap_selected_storage"]:
            raise RuntimeError(f"rank{rank} draft overlaps selected storage")
        if [x["cycle"] for x in rr] != expected_cycles:
            raise RuntimeError(f"rank{rank} selected cycle mismatch")
        if [x["cycle"] for x in rr if x["first_parking"]] != [192]:
            raise RuntimeError(f"rank{rank} first parking mismatch")
        for view in reg["selected_typed_views"]:
            a, b = view["byte_envelope"]
            if a < 0 or b <= a or b > view["storage_nbytes"]:
                raise RuntimeError(f"rank{rank} invalid typed view envelope")
    storage_shapes = [[sorted((x["storage_nbytes"], x["path"], tuple(x["shape"]),
                               tuple(x["stride"]), x["storage_offset"])
                              for x in reg["selected_typed_views"])] for reg in registry]
    if len({json.dumps(x) for x in storage_shapes}) != 1:
        raise RuntimeError("eight-rank typed layout differs")

    backing = {"attn": "main", "compressor.state_cache": "main",
               "indexer.compressor.state_cache": "indexer", "indexer.k_cache": "indexer",
               "swa_cache": "swa"}
    matrix = collections.Counter()
    current_and_future = collections.Counter()
    swa_prefix_page_overlaps = 0
    swa_exact_slot_overlaps = 0
    for rank, rr in enumerate(samples):
        own = owners(rank)
        for sample_index, sample in enumerate(rr):
            src = {source_kind(x["layer"]): x for x in sample["sources"]}
            if set(src) != set(backing):
                raise RuntimeError("layer2 source registry incomplete")
            writes, reads = {}, {}
            for name, s in src.items():
                write, read = set(), set()
                ratio, block = s["ratio"], s["block_size"]
                table = s["prefix_physical_pages_by_request"]
                for req, start in enumerate(s["start_pos"]):
                    if req not in own:
                        if s["slot_mapping"] is not None:
                            write.update(int(x[0]) for x in s["slot_mapping"][req * 8:(req + 1) * 8])
                        else:
                            write.update(int(table[req][pos // (ratio * block)])
                                         for pos in range(start, start + 8))
                    elif name.endswith("state_cache"):
                        read.update(int(table[req][pos // block])
                                    for pos in range(max(0, start - 8), start))
                    elif name == "swa_cache":
                        read.update(int(table[req][pos // block])
                                    for pos in range(max(0, start - 127), start + 8))
                    else:
                        read.update(int(x) for x in table[req])
                writes[name], reads[name] = write, read
            for producer in backing:
                for consumer in backing:
                    if backing[producer] != backing[consumer]:
                        continue
                    matrix[f"{producer} -> {consumer}: samples"] += 1
                    overlap = writes[producer] & reads[consumer]
                    matrix[f"{producer} -> {consumer}: page_overlap_samples"] += bool(overlap)
                    matrix[f"{producer} -> {consumer}: page_overlap_count"] += len(overlap)

            next_sample = (rr[sample_index + 1] if sample_index + 1 < len(rr)
                           and rr[sample_index + 1]["cycle"] == sample["cycle"] + 1 else None)
            next_src = ({source_kind(x["layer"]): x for x in next_sample["sources"]}
                        if next_sample is not None else None)
            owner_current_by_name = {}
            next_read_by_name = {}
            for name, s in src.items():
                ratio, block = s["ratio"], s["block_size"]
                owner_current = set()
                for req in own:
                    start = s["start_pos"][req]
                    if s["slot_mapping"] is not None:
                        owner_current.update(int(x[0]) for x in s["slot_mapping"][req * 8:(req + 1) * 8])
                    else:
                        owner_current.update(int(s["prefix_physical_pages_by_request"][req]
                                                 [pos // (ratio * block)])
                                             for pos in range(start, start + 8))
                current_and_future[f"{name}: current_owner_overlap_samples"] += bool(writes[name] & owner_current)
                owner_current_by_name[name] = owner_current
                if next_src is None:
                    continue
                ns = next_src[name]
                future_read = set()
                for req in own:
                    start = ns["start_pos"][req]
                    table = ns["prefix_physical_pages_by_request"][req]
                    if name.endswith("state_cache"):
                        future_read.update(int(table[pos // block])
                                           for pos in range(max(0, start - 8), start))
                    elif name == "swa_cache":
                        future_read.update(int(table[pos // block])
                                           for pos in range(max(0, start - 127), start + 8))
                    else:
                        future_read.update(int(x) for x in table)
                current_and_future[f"{name}: adjacent_pairs"] += 1
                current_and_future[f"{name}: next_owner_read_overlap_pairs"] += bool(writes[name] & future_read)
                next_read_by_name[name] = future_read
            for producer in backing:
                for consumer in backing:
                    if backing[producer] != backing[consumer]:
                        continue
                    current_and_future[f"{producer} -> {consumer}: current_owner_page_overlap_samples"] += bool(
                        writes[producer] & owner_current_by_name[consumer])
                    if next_src is not None:
                        current_and_future[f"{producer} -> {consumer}: next_owner_read_page_overlap_pairs"] += bool(
                            writes[producer] & next_read_by_name[consumer])

            swa = src["swa_cache"]
            prefix_write = set()
            prefix_read = set()
            exact_write = set()
            exact_read = set()
            for req, start in enumerate(swa["start_pos"]):
                table = swa["prefix_physical_pages_by_request"][req]
                if req in own:
                    prefix_read.update(table)
                    exact_read.update((int(table[pos // 32]), pos % 32)
                                      for pos in range(max(0, start - 127), start + 8))
                else:
                    slots = swa["slot_mapping"][req * 8:(req + 1) * 8]
                    prefix_write.update(int(x[0]) for x in slots)
                    exact_write.update((int(x[0]), int(x[1])) for x in slots)
            swa_prefix_page_overlaps += len(prefix_write & prefix_read)
            swa_exact_slot_overlaps += len(exact_write & exact_read)
    summary = {
        "status": "read_only_diagnostic_pass",
        "runner_scope": "one c12x1024 cohort; diagnostic Host sync, no formal TPS comparison",
        "client_success": 12, "client_output_tokens_each": 1024,
        "runtime_rank_pass": 8, "target_graph_mode": "full",
        "rank_count": 8, "samples_per_rank": 10, "sample_cycles": expected_cycles,
        "first_parking_cycle": 192,
        "typed_leaves_per_rank": 196, "draft_typed_leaves_per_rank": 3,
        "layer2_related_typed_views_per_rank": 15,
        "layer2_related_unique_backings_per_rank": 3,
        "draft_overlap_selected_backings": 0,
        "backing_nbytes_per_rank": sorted({x["storage_ptr"]: x["storage_nbytes"]
                                             for x in registry[0]["selected_typed_views"]}.values()),
        "same_layer2_backing_page_matrix": dict(sorted(matrix.items())),
        "current_owner_and_next_sample_page_checks": dict(sorted(current_and_future.items())),
        "swa_full_prefix_page_overlap_count": swa_prefix_page_overlaps,
        "swa_window_exact_slot_overlap_count": swa_exact_slot_overlaps,
        "classification": "same-layer2 nonowner current-write versus owner bounded read: disjoint in sampled source-derived page/slot domains; cross-layer alias consumers and native read trace unknown",
        "limits": [
            "Only 10 selected cycles per rank of one cohort, including five adjacent pairs/rank; no continuous byte trace or page epoch.",
            "Compressed writes and QLI/Sparse prefix reads are conservative source-derived envelopes, not native observed addresses.",
            "Neighboring layers that share two backings lack live metadata in Run299; their read/write liveness is unknown.",
            "Disjoint sampled addresses do not prove no semantic work or HBM traffic is required.",
        ],
    }
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run_dir)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "rank_count", "samples_per_rank",
                                            "first_parking_cycle", "swa_window_exact_slot_overlap_count")}))
