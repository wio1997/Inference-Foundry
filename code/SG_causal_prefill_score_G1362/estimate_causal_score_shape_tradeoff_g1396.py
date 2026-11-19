"""Compare logical score work and operand shapes under native Prefill tiling."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from host_geometry import RequestRows


ALIGNMENTS = {
    "full_score_width": 0,
    "exact_causal_suffix": 1,
    "round_width_to_128": 128,
    "round_width_to_512": 512,
    "round_width_to_1024": 1024,
}


def score_scenarios(prompt, rows, heads, ratio, budget):
    cells = dict.fromkeys(ALIGNMENTS, 0)
    operand_shapes = {key: set() for key in ALIGNMENTS}
    for prefix in range(0, prompt, rows):
        keys = (prefix + rows) // ratio
        geometry = RequestRows(0, prefix, rows)
        tiles = geometry.score_tiles(compress_ratio=ratio, heads=heads, key_count=keys, budget_bytes=budget)
        for tile in tiles:
            queries = tile.end - tile.start
            for key, alignment in ALIGNMENTS.items():
                width = keys if alignment == 0 else min(
                    keys, (tile.key_width + alignment - 1) // alignment * alignment
                )
                assert tile.key_width <= width <= keys
                cells[key] += queries * heads * width
                operand_shapes[key].add((queries, width))
    full = cells["full_score_width"]
    return {key: {"score_cells_per_rank": value, "distinct_query_and_key_width_pairs": len(operand_shapes[key]),
                  "fraction_less_than_dense": (full - value) / full}
            for key, value in cells.items()}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("score-source-receipt", "sg-warmup", "sg-cadence", "sg-timing", "vllm-warmup", "vllm-command"):
        cli.add_argument(f"--{name}", type=Path, required=True)
    args = cli.parse_args()
    inputs = {}
    pins = {}
    for name, path in vars(args).items():
        raw = path.read_bytes()
        inputs[name] = json.loads(raw)
        pins[name] = sha256(raw).hexdigest()
    score = inputs["score_source_receipt"]
    observed = score["observed_score_operand_shapes_per_rank"]
    rows, heads, depth = observed["query"]
    k_rows, k_depth = observed["key"]
    budget = score["observed_index_score_cap_bytes"]
    assert (rows, heads, depth, k_rows, k_depth, budget) == (8192, 32, 128, 4096, 128, 128 << 20)
    cadence = inputs["sg_cadence"]["rank0_events"]
    assert len(cadence) == 64 and all(row["new_tokens"] == rows and row["cached_tokens"] == 0 for row in cadence)
    sg, native = inputs["sg_warmup"], inputs["vllm_warmup"]
    timing = inputs["sg_timing"]["timeline"]
    native_command = inputs["vllm_command"]
    assert sg["classification"] == "terminal_verified" and sg["exact_shape_successes"] == sg["request_count"] == 16
    assert sg["request_IDs_exactly_once"] and sg["raw_details_SHA256"] == timing["detail_SHA256"]
    assert native["fully_validated"] and native["requests"] == native["successful_requests"] == 16
    assert timing["peak_concurrency"] == native["observed_max_concurrency"] == 16
    assert timing["shape"] == {"requests": 16, "input": 32768, "output": 1024}
    assert (native["input_tokens"], native["output_tokens"]) == (32768, 1024)
    assert sg["dataset_SHA256"] == native_command["dataset_sha256"] and native_command["requests"] == 16
    sglang_seconds, vllm_seconds = sg["metrics"]["duration_s"], native["duration_s"]
    scenarios = {str(prompt): score_scenarios(prompt, rows, heads, rows // k_rows, budget)
                 for prompt in (32768, 65536, 131072, 262144)}
    trim = scenarios["32768"]["exact_causal_suffix"]["fraction_less_than_dense"]
    print(json.dumps({
        "input_artifact_SHA256": pins,
        "actual_unscored_first_warmups": {
            "shared_dataset_SHA256": sg["dataset_SHA256"], "concurrency": 16,
            "requests": 16, "prompt_tokens": 32768, "generated_tokens": 1024,
            "SGLang_duration_s": sglang_seconds, "vLLM_duration_s": vllm_seconds,
            "SGLang_to_vLLM_duration_ratio": sglang_seconds / vllm_seconds,
        },
        "native_first_query_shape_per_rank": observed,
        "native_row_tiling_budget_bytes": budget,
        "score_work_scenarios": scenarios,
        "wall_time_fraction_reduction_to_match_first_warmup": 1 - vllm_seconds / sglang_seconds,
        "hypothetical_32K_SGLang_s_if_all_time_scaled_linearly_with_score_cells": sglang_seconds * (1 - trim),
        "scope": "Contiguous zero-cache requests and native query tiling. Operand shapes are not compiler cache entries."
                 " Logical cells and the linear thought experiment do not measure device or end-to-end speedup.",
    }, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
