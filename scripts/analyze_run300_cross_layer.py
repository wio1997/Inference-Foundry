#!/usr/bin/env python3
"""Fail-closed post-run validator and compact cross-layer address-risk summary."""

import argparse
import collections
import json
from pathlib import Path


def overlap(a, b):
    return max(a[0], b[0]) < min(a[1], b[1])


def analyze(root):
    bench = json.loads((root / "bench12.json").read_text())
    if bench["summary"]["success"] != 12 or any(x["output_tokens"] != 1024 for x in bench["requests"]):
        raise RuntimeError("client 12x1024 correctness gate failed")
    runtime_files = sorted((root / "runtime").glob("rank*_cohort*.json"))
    if len(runtime_files) != 8:
        raise RuntimeError(f"expected eight Runtime reports, got {len(runtime_files)}")
    runtime = [json.loads(x.read_text()) for x in runtime_files]
    if {x["rank"] for x in runtime} != set(range(8)) or not all(x["pass"] and x["target_graph_mode"] == "FULL" for x in runtime):
        raise RuntimeError("all-eight FULL Graph Runtime gate failed")
    regs = []
    rows = []
    expected_layers = {0: (0, None), 1: (0, None), 2: (4, 2),
                       3: (128, 1), 4: (4, 2), 5: (128, 1)}
    for rank in range(8):
        reg = json.loads((root / "census" / f"rank{rank}_registry.json").read_text())
        rr = [json.loads(x) for x in (root / "census" / f"rank{rank}.jsonl").read_text().splitlines()]
        if reg["rank"] != rank or reg["all_typed_leaf_count"] != 196 or reg["selected_target_alias_count"] != 14:
            raise RuntimeError(f"rank{rank} registry incomplete")
        if len(reg["draft_typed_views"]) != 3 or len(reg["selected_target_views"]) != 15:
            raise RuntimeError(f"rank{rank} typed view inventory incomplete")
        for draft in reg["draft_typed_views"]:
            if any(overlap(draft["absolute_byte_envelope"], target["absolute_byte_envelope"])
                   for target in reg["selected_target_views"]):
                raise RuntimeError(f"rank{rank} draft absolute-byte interval overlaps target")
        for source in reg["sources"]:
            ratio, coff = expected_layers[source["layer_index"]]
            if (source["producer_compress_ratio"], source["producer_coff"]) != (ratio, coff):
                raise RuntimeError(f"rank{rank} producer ratio/coff wrong: {source['name']}")
            expected_position = ratio if source["name"].endswith(".attn") or source["name"].endswith("indexer.k_cache") else 1
            if source["cache_position_ratio"] != expected_position:
                raise RuntimeError(f"rank{rank} address ratio wrong: {source['name']}")
        if len(rr) < 8 or rr[0]["cycle"] != 0 or rr[1]["cycle"] != 1:
            raise RuntimeError(f"rank{rank} adjacent first cycles absent")
        if len({x["cycle"] for x in rr}) != len(rr):
            raise RuntimeError(f"rank{rank} duplicate sample cycle")
        if len([x for x in rr if x["first_parking"]]) != 1:
            raise RuntimeError(f"rank{rank} first parking sample absent")
        first = next(x["cycle"] for x in rr if x["first_parking"])
        if first + 1 not in {x["cycle"] for x in rr}:
            raise RuntimeError(f"rank{rank} parking successor absent")
        source_names = {x["name"] for x in reg["sources"]}
        for sample in rr:
            if sample["rank"] != rank or len(sample["sources"]) != 14:
                raise RuntimeError(f"rank{rank} source sample incomplete")
            if {x["name"] for x in sample["sources"]} != source_names:
                raise RuntimeError(f"rank{rank} sample source names differ")
            if sample["owner_request_slots"] != [i for i in range(12)
                                                  if max(0, min((rank + 1) * 12, (i + 1) * 8)
                                                         - max(rank * 12, i * 8))]:
                raise RuntimeError(f"rank{rank} owner geometry changed")
            for source in sample["sources"]:
                if len(source["current_write_slots_by_request"]) != 12:
                    raise RuntimeError(f"rank{rank} source write slots incomplete")
                if any(len(x) == 0 for x in source["current_write_slots_by_request"]):
                    raise RuntimeError(f"rank{rank} empty source write slots")
        regs.append(reg)
        rows.append(rr)
    cycles = [[x["cycle"] for x in rr] for rr in rows]
    if len({tuple(x) for x in cycles}) != 1:
        raise RuntimeError("rank sample cycles differ")

    counts = collections.Counter()
    examples = {}
    later_state = {}
    for rank, rr in enumerate(rows):
        for row in rr:
            for source in row["sources"]:
                name = source["name"]
                layer = source["layer_index"]
                for collision in source["read_overlap"]:
                    edge = collision["edge"]
                    key = ("read_overlap", edge, layer, name)
                    counts[key] += len(collision["hits"])
                    if key not in examples:
                        examples[key] = {"rank": rank, "cycle": row["cycle"],
                                         "request_slot": collision["request_slot"],
                                         "precision": collision["precision"],
                                         "first_hit": collision["hits"][0]}
                    if edge == "same_cycle" and layer > 2:
                        counts[("later_layer_read_hazard", layer, name)] += len(collision["hits"])
                        if name.endswith("state_cache"):
                            detail = later_state.setdefault(name, {"layer": layer, "hits": 0,
                                "owner_hits": 0, "nonowner_hits": 0,
                                "minimum_age_lower_bound_tokens": None,
                                "maximum_age_lower_bound_tokens": None,
                                "example": None})
                            for hit in collision["hits"]:
                                age_lb = (source["start_pos"][collision["request_slot"]]
                                          - (hit["logical_page"] + 1) * source["block_size"] + 1)
                                detail["hits"] += 1
                                detail["owner_hits" if collision["request_slot"] in row["owner_request_slots"]
                                       else "nonowner_hits"] += 1
                                if (detail["minimum_age_lower_bound_tokens"] is None or
                                        age_lb < detail["minimum_age_lower_bound_tokens"]):
                                    detail["minimum_age_lower_bound_tokens"] = age_lb
                                    detail["example"] = {"rank": rank, "cycle": row["cycle"],
                                        "request_slot": collision["request_slot"],
                                        "start_pos": source["start_pos"][collision["request_slot"]],
                                        "block_size": source["block_size"], **hit}
                                if (detail["maximum_age_lower_bound_tokens"] is None or
                                        age_lb > detail["maximum_age_lower_bound_tokens"]):
                                    detail["maximum_age_lower_bound_tokens"] = age_lb
                    if edge == "previous_sample_cycle" and layer < 2:
                        counts[("next_cycle_early_layer_read_hazard", layer, name)] += len(collision["hits"])
                for item in source["same_backing_write_overlap"]:
                    if layer != 2:
                        counts[("allocator_write_overlap", layer, name)] += len(item["physical_pages"])
    compact = {
        "status": "read_only_cross_layer_census_pass",
        "runner_scope": "one diagnostic c12x1024 cohort, not a formal TPS comparison",
        "client_success": 12, "client_output_tokens_each": 1024,
        "runtime_rank_pass": 8, "target_graph_mode": "FULL",
        "typed_leaves_per_rank": 196, "target_alias_sources_per_rank": 14,
        "target_alias_views_per_rank": 15, "draft_views_per_rank": 3,
        "draft_absolute_byte_overlap_selected_target": 0,
        "sample_cycles": cycles[0], "sample_count_per_rank": len(cycles[0]),
        "first_parking_cycle": next(x["cycle"] for x in rows[0] if x["first_parking"]),
        "counts": [{"kind": k[0], "edge_or_layer": k[1], "layer_or_source": k[2],
                    "source": k[3] if len(k) > 3 else None, "count": v}
                   for k, v in sorted(counts.items(), key=lambda item: str(item[0]))],
        "examples": [{"category": list(k), **v} for k, v in list(examples.items())[:24]],
        "later_state_full_prefix_hits": list(later_state.values()),
        "limits": [
            "Read hits are page-level source-derived envelopes, not native addresses or read-from values.",
            "State read uses a conservative full-prefix table; a hit may lie outside the actual recurrent live window.",
            "State hit age is measured from the end of its logical page to the current start; the native live state window is not established here.",
            "Current-cycle layer0/1 reads precede layer2 writes and are not same-cycle successor hazards.",
            "Allocator WAR/WAW may be removable with different storage/lifetime and is not semantic RAW.",
            "Only selected cycles of one cohort; no compulsory bytes, critical-path gain or Product bound is inferred.",
        ],
    }
    return compact


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = analyze(a.run_dir)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "target_alias_sources_per_rank",
                                            "sample_count_per_rank", "first_parking_cycle")}))
