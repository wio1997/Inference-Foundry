#!/usr/bin/env python3
"""Two-generation graph replay of the installed partial-EP MoE ABI."""
import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import torch
import torch_npu


def validate(name, ids, marker_shift, q, idx, count, out, payload, probs, lo, hi):
    rows, topk = len(ids), len(ids[0])
    g = idx.cpu().tolist()
    q = q.cpu()
    c = count.cpu().tolist()
    y = out[:, 0].float().cpu().tolist()
    p = payload[:, 0].float().cpu().tolist()
    w = probs.float().cpu().tolist()
    valid = []
    expected = [0.0] * rows
    starts = [sum(c[:e - lo]) for e in range(lo, hi)]
    for t in range(rows):
        for k in range(topk):
            pos = g[t * topk + k]
            if lo <= ids[t][k] < hi:
                assert 0 <= pos < sum(c), (name, t, k, pos)
                e = ids[t][k]
                assert starts[e - lo] <= pos < starts[e - lo] + c[e - lo], (
                    name, t, k, e, pos)
                valid.append(pos)
                assert int(q[pos, :rows].abs().argmax()) == (t + marker_shift) % rows, (name, t, k)
                expected[t] += p[pos] * w[t][k]
            else:
                assert pos == -1 and w[t][k] == 0, (name, t, k, pos)
    assert len(set(valid)) == len(valid) == sum(c)
    assert c == [sum(e == expert for row in ids for e in row) for expert in range(lo, hi)]
    error = max(abs(a - b) for a, b in zip(y, expected))
    assert error < 0.015, (name, error)
    assert bool(torch.count_nonzero(out[:, 1:]).item() == 0)
    return {"name": name, "valid_routes": len(valid), "excluded_routes": rows * topk - len(valid),
            "indices_sha256": __import__("hashlib").sha256(bytes(str(g), "utf8")).hexdigest(),
            "max_unpermute_abs_error": error}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.npu.set_device(0)
    route = json.loads(Path("/data/wio/Inference_Foundry/evidence/20260927_loop078_bound/run403/capture/rank0_cohort1.json").read_text())["records"]["64"]["target"][0]["ids"]
    rows, topk, hidden, lo, hi = 96, 6, 4096, 0, 32
    assert len(route) == rows
    x = torch.ones((rows, hidden), dtype=torch.bfloat16).npu()
    ids = torch.zeros((rows, topk), dtype=torch.int32).npu()
    probs = torch.zeros((rows, topk), dtype=torch.bfloat16).npu()
    payload = torch.zeros((rows * topk, hidden), dtype=torch.bfloat16).npu()
    payload[:, 0].copy_(torch.arange(1, rows * topk + 1, dtype=torch.float32).to(torch.bfloat16).npu() / 128)

    def stage(route_ids, shift):
        host_x = torch.ones((rows, hidden), dtype=torch.bfloat16)
        for t in range(rows):
            host_x[t, (t + shift) % rows] = 16
        host_p = torch.tensor([[2.0 ** -(k + 1) if lo <= e < hi else 0.0
                               for k, e in enumerate(row)] for row in route_ids], dtype=torch.bfloat16)
        x.copy_(host_x.npu())
        ids.copy_(torch.tensor(route_ids, dtype=torch.int32).npu())
        probs.copy_(host_p.npu())
        torch.npu.synchronize()

    stage(route, 0)

    def forward():
        packed, indices, counts, scales = torch_npu.npu_moe_init_routing_v2(
            x, ids, active_num=rows * topk, expert_num=256,
            expert_tokens_num_type=1, expert_tokens_num_flag=True,
            active_expert_range=[lo, hi], quant_mode=1, row_idx_type=0)
        out = torch_npu.npu_moe_token_unpermute(payload, torch.abs(indices), probs=probs)
        return packed, indices, counts, scales, out

    forward()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        graph_output = forward()
    torch.npu.synchronize()

    result = {"schema": 1, "status": "running", "pid": os.getpid(),
              "torch_npu": torch_npu.__version__, "shape": [rows, topk, hidden],
              "active_expert_range": [lo, hi], "cases": []}
    try:
        for label, actual, shift in (
            ("generation_1_actual_route", route, 0),
            ("generation_2_shifted_experts_and_row_sentinel", [[(e + 17) % 256 for e in row] for row in route], 1),
            ("generation_3_restored_route", route, 0),
        ):
            stage(actual, shift)
            graph.replay()
            torch.npu.synchronize()
            result["cases"].append(validate(label, actual, shift, *graph_output[:3],
                                            graph_output[4], payload, probs, lo, hi))
        assert result["cases"][0]["indices_sha256"] != result["cases"][1]["indices_sha256"]
        assert result["cases"][0]["indices_sha256"] == result["cases"][2]["indices_sha256"]
        result["status"] = "pass"
    except Exception:
        result["status"] = "fail"
        result["error"] = traceback.format_exc()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "cases": len(result["cases"]),
                      "error": result.get("error", "")[:800]}))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
