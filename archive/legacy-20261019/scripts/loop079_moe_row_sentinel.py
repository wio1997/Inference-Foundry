#!/usr/bin/env python3
"""Isolated CANN 9.1 MoE V3 partial-EP row/abs/mask semantic sentinel."""
import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import torch
import torch_npu


def run_case(name, ids, lo, hi, hidden):
    rows, topk = len(ids), len(ids[0])
    assert all(len(row) == topk for row in ids)
    x = torch.ones((rows, hidden), dtype=torch.bfloat16)
    for t in range(rows):
        x[t, t] = 16
    x = x.npu()
    experts = torch.tensor(ids, dtype=torch.int32).npu()
    packed, indices, counts, scales = torch_npu.npu_moe_init_routing_v2(
        x, experts, active_num=rows * topk, expert_num=256,
        expert_tokens_num_type=1, expert_tokens_num_flag=True,
        active_expert_range=[lo, hi], quant_mode=1, row_idx_type=0)
    idx = indices.cpu().to(torch.int64).tolist()
    count = counts.cpu().tolist()
    q = packed.cpu()
    scale = scales.cpu()
    valid = [(t, k, idx[t * topk + k]) for t in range(rows)
             for k in range(topk) if lo <= ids[t][k] < hi]
    excluded = [(t, k, idx[t * topk + k]) for t in range(rows)
                for k in range(topk) if not (lo <= ids[t][k] < hi)]
    assert sum(count) == len(valid)
    assert count == [sum(expert == e for row in ids for expert in row)
                     for e in range(lo, hi)]
    assert all(0 <= pos < len(valid) for _, _, pos in valid)
    assert len({pos for _, _, pos in valid}) == len(valid)
    assert all(pos == -1 for _, _, pos in excluded)
    starts = [sum(count[:e - lo]) for e in range(lo, hi)]
    for t, k, pos in valid:
        expert = ids[t][k]
        assert starts[expert - lo] <= pos < starts[expert - lo] + count[expert - lo], (
            name, t, k, expert, pos)
        assert int(torch.argmax(q[pos, :rows].abs())) == t, (name, t, k, pos)
    # Independent token/route signatures at the packed locations. Excluded
    # slots are deliberately zero-masked before abs(-1) aliases a valid row.
    payload = torch.zeros((rows * topk, hidden), dtype=torch.bfloat16)
    probs = torch.zeros((rows, topk), dtype=torch.bfloat16)
    expected = [0.0] * rows
    for t, k, pos in valid:
        signature = 1.0 + 4.0 * k + t / 128.0
        weight = 2.0 ** -(k + 1)
        payload[pos, 0] = signature
        probs[t, k] = weight
        expected[t] += signature * weight
    out = torch_npu.npu_moe_token_unpermute(
        payload.npu(), torch.abs(indices), probs=probs.npu()).cpu()
    got = out[:, 0].float().tolist()
    assert all(abs(a - b) <= 0.015 for a, b in zip(got, expected)), (
        name, max(abs(a - b) for a, b in zip(got, expected)))
    assert all(v == 0 for v in out[:, 1:].float().flatten().tolist())
    return {
        "name": name, "shape": [rows, topk, hidden], "active_expert_range": [lo, hi],
        "quant_mode": 1, "row_idx_type": 0,
        "valid_routes": len(valid), "excluded_routes": len(excluded),
        "valid_indices_unique_and_in_packed_prefix": True,
        "valid_indices_in_matching_expert_count_segment": True,
        "excluded_indices_all_minus_one": True,
        "packed_int8_row_identity": True,
        "masked_abs_unpermute_max_abs_error": max(abs(a - b) for a, b in zip(got, expected)),
        "scales_finite_on_valid_prefix": bool(torch.isfinite(scale[:len(valid)]).all()),
        "first_indices": idx[:min(20, len(idx))],
        "counts_total": sum(count),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.npu.set_device(0)
    small = [[0, 33], [63, 1], [32, 64], [2, 31]]
    source = Path("/data/wio/Inference_Foundry/evidence/20260927_loop078_bound/run403/capture/rank0_cohort1.json")
    route = json.loads(source.read_text())["records"]["64"]["target"][0]["ids"]
    assert len(route) == 96 and all(len(row) == 6 for row in route)
    result = {"schema": 1, "status": "running", "pid": os.getpid(),
              "torch_npu": torch_npu.__version__, "cases": []}
    try:
        result["cases"].append(run_case("small_partial_0_32", small, 0, 32, 4096))
        result["cases"].append(run_case("actual_route_96x6_rank0_range", route, 0, 32, 4096))
        result["cases"].append(run_case("actual_route_96x6_nonzero_range", route, 96, 128, 4096))
        result["status"] = "pass"
    except Exception:
        result["status"] = "fail"
        result["error"] = traceback.format_exc()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "cases": len(result["cases"]),
                      "error": result.get("error", "")[:600]}))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
