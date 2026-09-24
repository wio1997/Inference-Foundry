"""Summarize one-time target/DSpark cache alias manifests for snapshot planning."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def check(root: Path) -> dict:
    summaries = []
    for rank in range(8):
        path = root / f"rank{rank}.json"
        manifest = json.loads(path.read_text())
        if manifest["rank"] != rank or len(manifest["groups"]) != 6:
            raise ValueError(f"rank {rank} group manifest incomplete")
        if sorted(group["gid"] for group in manifest["groups"]) != list(range(6)):
            raise ValueError(f"rank {rank} group IDs incomplete")
        cache_counts = Counter(cache["group"] for cache in manifest["target_caches"])
        source_ratios = defaultdict(set)
        source_layers = defaultdict(list)
        for source in manifest["target_sources"]:
            source_ratios[source["table_group"]].add(source["ratio"])
            source_layers[source["table_group"]].append(source["layer"])
        draft = manifest["draft_groups"]
        if sorted(row["gid"] for row in draft) != [2, 3]:
            raise ValueError(f"rank {rank} draft binding incomplete")
        if any(row["table_group"] != row["gid"] or row["mapping_group"] != row["gid"] for row in draft):
            raise ValueError(f"rank {rank} draft group alias mismatch")
        summaries.append({"rank": rank,
                          "target_cache_tensors": len(manifest["target_caches"]),
                          "target_cache_tensors_by_group": {str(k): v for k, v in sorted(cache_counts.items(), key=lambda pair: str(pair[0]))},
                          "unmapped_cache_tensors": cache_counts[None],
                          "target_source_layers": len(manifest["target_sources"]),
                          "target_ratios_by_table_group": {str(k): sorted(v) for k, v in source_ratios.items()},
                          "target_source_layers_by_table_group": {str(k): sorted(v) for k, v in source_layers.items()},
                          "target_sources_without_slot_tensor": sum(source["slot_shape"] is None for source in manifest["target_sources"]),
                          "draft_groups": [2, 3]})
    structural = [{k: v for k, v in row.items() if k != "rank"} for row in summaries]
    if any(row != structural[0] for row in structural[1:]):
        raise ValueError("rank cache manifests differ structurally")
    return {"rank_manifests": 8, "rank_structure_equal": True,
            "structure": structural[0],
            "scope": "one-time pointer and metadata inventory; actual compressed write rows and KV values are not covered"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = check(args.root)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
