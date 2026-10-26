"""Audit physical cache aliases; never infer write ownership from positional slot specs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def check(root: Path) -> dict:
    manifests = [json.loads((root / f"rank{rank}.json").read_text())
                 for rank in range(8)]
    structures = []
    for rank, manifest in enumerate(manifests):
        if manifest["rank"] != rank:
            raise ValueError(f"rank mismatch: {rank}")
        caches = manifest["target_caches"]
        if len(caches) != 67:
            raise ValueError(f"rank {rank} has {len(caches)} cache tensor views")
        structure = []
        for cache in caches:
            aliases = cache["layer_aliases"]
            if not aliases or any(alias["table_group"] is None for alias in aliases):
                raise ValueError(f"rank {rank} unresolved alias: {cache['name']}")
            structure.append((
                cache["name"], tuple(cache["shape"]), tuple(cache["stride"]),
                cache["dtype"], cache["group"],
                tuple(sorted((alias["layer"], alias["table_group"])
                             for alias in aliases)),
            ))
        structures.append(structure)
    if any(structure != structures[0] for structure in structures[1:]):
        raise ValueError("eight rank cache alias structures differ")

    cache_rows = manifests[0]["target_caches"]
    actual_groups = [set(alias["table_group"] for alias in row["layer_aliases"])
                     for row in cache_rows]
    storage_to_names = defaultdict(list)
    for row in cache_rows:
        storage_to_names[row["storage_ptr"]].append(row["name"])
    mismatched = [row["name"] for row, groups in zip(cache_rows, actual_groups)
                  if row["group"] not in groups]
    mixed = [row["name"] for row, groups in zip(cache_rows, actual_groups)
             if len(groups) > 1]
    histogram = Counter(",".join(map(str, sorted(groups))) for groups in actual_groups)
    aliases = sum(len(row["layer_aliases"]) for row in cache_rows)
    return {
        "rank_manifests": 8,
        "rank_structure_equal": True,
        "cache_tensor_views": len(cache_rows),
        "cache_data_ptrs": len({row["name"] for row in cache_rows}),
        "storage_allocations": len(storage_to_names),
        "storage_allocations_with_multiple_views":
            sum(len(names) > 1 for names in storage_to_names.values()),
        "layer_alias_records": aliases,
        "positional_group_label_outside_actual_alias_groups": len(mismatched),
        "positional_group_label_mismatch_examples": mismatched[:12],
        "cache_views_shared_across_groups": len(mixed),
        "actual_group_set_histogram": dict(sorted(histogram.items())),
        "conclusion": (
            "The positional cache_slot_specs group labels are not a physical "
            "write-set certificate. Some cache views have aliases in multiple "
            "KV groups and some views share storage. Derive snapshot page "
            "coverage from each actual layer alias and its block table."
        ),
        "scope": (
            "one-time cache alias ownership only; request-cycle write pages, "
            "KV values, target logits, and acceptance were not compared"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = check(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
