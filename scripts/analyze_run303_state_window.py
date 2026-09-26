#!/usr/bin/env python3
"""Audit Run301 full-prefix page hits against the packaged Compressor read bound.

This is a source-derived screen, not a native address trace or binary attestation.
"""

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_SHA256 = "ebe567933371481de1067cfd78d88ad84900ed51cfc9fddf68de47dcb5d73823"
WINDOW_BY_LAYER = {3: 128, 4: 8, 5: 128}


def audit(run_dir: Path, source: Path):
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise RuntimeError(f"Compressor source changed: {digest}")
    summary = json.loads((run_dir / "analysis.json").read_text())
    if summary["status"] != "read_only_cross_layer_census_pass":
        raise RuntimeError("Run302 offline validation did not pass")
    per_layer = {layer: {"full_prefix_hits": 0, "inside_conservative_window": 0,
                         "minimum_age_lower_bound_tokens": None,
                         "conservative_window_tokens": window}
                 for layer, window in WINDOW_BY_LAYER.items()}
    for rank in range(8):
        rows = [json.loads(line) for line in
                (run_dir / "census" / f"rank{rank}.jsonl").read_text().splitlines()]
        for row in rows:
            for item in row["sources"]:
                layer = item["layer_index"]
                if layer not in per_layer or not item["name"].endswith("state_cache"):
                    continue
                for collision in item["read_overlap"]:
                    if collision["edge"] != "same_cycle":
                        continue
                    start = item["start_pos"][collision["request_slot"]]
                    for hit in collision["hits"]:
                        age_lb = start - (hit["logical_page"] + 1) * item["block_size"] + 1
                        result = per_layer[layer]
                        result["full_prefix_hits"] += 1
                        prior = result["minimum_age_lower_bound_tokens"]
                        result["minimum_age_lower_bound_tokens"] = (
                            age_lb if prior is None else min(prior, age_lb))
                        if age_lb <= result["conservative_window_tokens"]:
                            result["inside_conservative_window"] += 1
    expected = {3: (1422, 1126), 4: (638, 222), 5: (1034, 1326)}
    for layer, (hits, age) in expected.items():
        result = per_layer[layer]
        if (result["full_prefix_hits"], result["minimum_age_lower_bound_tokens"],
                result["inside_conservative_window"]) != (hits, age, 0):
            raise RuntimeError(f"layer{layer} native-window screen changed: {result}")
    return {
        "status": "source_bound_sampled_state_window_pass",
        "source_path": str(source),
        "source_sha256": digest,
        "source_derive": "ReadState: compressTcSize==0 returns; ordinary read current group to s; OVERLAP coff=2 also previous group. Conservative history [s-r*coff,s).",
        "source_to_loaded_binary_proven": False,
        "sampled_all_rank_state_full_prefix_hits": sum(x["full_prefix_hits"] for x in per_layer.values()),
        "sampled_hits_inside_source_bound": sum(x["inside_conservative_window"] for x in per_layer.values()),
        "per_layer": per_layer,
        "limits": ["Source-derived, selected cycles only; not observed native addresses or a binary attestation.",
                   "No persistent cache value, entire lifetime, resource bound or E2E gain follows from this screen."],
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = audit(a.run_dir, a.source)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"],
                      "sampled_hits_inside_source_bound": result["sampled_hits_inside_source_bound"]}))
