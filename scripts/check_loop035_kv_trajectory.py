"""Validate streamed continuous DSpark context slot ownership evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def check(root: Path) -> dict:
    ranks = []
    for rank in range(8):
        runtime_path = root / "runtime" / f"rank{rank}_cohort1.json"
        stream_path = root / "slots" / f"rank{rank}.jsonl"
        runtime = json.loads(runtime_path.read_text())
        rows = [json.loads(line) for line in stream_path.read_text().splitlines()]
        cycles = runtime["cycles"]
        if not runtime["pass"] or not runtime["host_mirror_exact"]:
            raise ValueError(f"rank {rank} serving gate failed")
        if runtime["generated_output_counts"] != [1024] * 12:
            raise ValueError(f"rank {rank} client lengths failed")
        if len(rows) != cycles * 2:
            raise ValueError(f"rank {rank} stream incomplete: {len(rows)} != {cycles * 2}")
        target_mismatch_cycles = {gid: 0 for gid in range(6)}
        target_zero_cycles = {gid: 0 for gid in range(6)}
        for cycle in range(cycles):
            before, after = rows[2 * cycle : 2 * cycle + 2]
            if before["rank"] != rank or after["rank"] != rank:
                raise ValueError(f"rank {rank} record identity")
            if before["phase"] != "pre_target" or after["phase"] != "post_proposer":
                raise ValueError(f"rank {rank} cycle {cycle} phase order")
            if before["cycle"] != cycle or after["cycle"] != cycle:
                raise ValueError(f"rank {rank} cycle {cycle} discontinuity")
            if len(before["groups"]) != 6 or len(after["dspark_context"]) != 2:
                raise ValueError(f"rank {rank} cycle {cycle} group coverage")
            for group in before["groups"]:
                if group["supported"]:
                    target_mismatch_cycles[group["group"]] += int(group["mismatched"] > 0)
                    target_zero_cycles[group["group"]] += int(group["zero_blocks"] > 0)
                if group["supported"] and group["group"] != 0 and (group["negative_blocks"] or group["zero_blocks"]):
                    raise ValueError(f"rank {rank} cycle {cycle} target physical ownership")
            for group in after["dspark_context"]:
                if group["gid"] not in (2, 3):
                    raise ValueError(f"rank {rank} cycle {cycle} unexpected draft group")
                for field in ("context_vs_expected", "context_vs_source",
                              "source_vs_expected", "context_negative", "context_zero",
                              "same_cycle_duplicate_slots", "cross_request_owner_conflicts"):
                    if group[field] != 0:
                        gid = group["gid"]
                        raise ValueError(f"rank {rank} cycle {cycle} gid{gid} {field}={group[field]}")
        ranks.append({"rank": rank, "cycles": cycles, "stream_records": len(rows),
                      "draft_groups_exact": [2, 3],
                      "target_mismatch_cycles": target_mismatch_cycles,
                      "target_zero_cycles": target_zero_cycles,
                      "supported_target_groups_nonzero_except_group0": True,
                      "unclassified_target_groups": sorted({g["group"] for row in rows[::2]
                                                          for g in row["groups"] if not g["supported"] or g["group"] == 0})})
    if len({r["cycles"] for r in ranks}) != 1:
        raise ValueError("rank cycle counts differ")
    return {"rank_pass": 8, "cycles": ranks[0]["cycles"],
            "total_phase_records": sum(r["stream_records"] for r in ranks),
            "ranks": ranks,
            "scope": "complete current physical slot ownership for draft gid2/3; target groups0/1 consumption and KV values are not certified"}


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
