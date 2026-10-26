#!/usr/bin/env python3
"""Official Level1 MemoryAccess trace for one GMM2 Graph bank condition."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path("/data/wio/Inference_Foundry")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--route-ordinal", type=int, default=64)
    parser.add_argument("--weight-banks", type=int, choices=(1, 8), required=True)
    args = parser.parse_args()
    import torch
    import torch_npu
    from torch_npu.profiler import profile, ProfilerActivity, ProfilerLevel, AiCMetrics, _ExperimentalConfig, tensorboard_trace_handler
    from vllm_ascend.utils import enable_custom_op
    assert enable_custom_op()
    torch.npu.config.allow_internal_format = True
    torch.npu.set_device(args.rank)
    args.out.mkdir(parents=True, exist_ok=True)
    snap_path = ROOT / f"evidence/20260925_loop039_gmm/run121/counts/rank{args.rank}_cycle64.json"
    snap = json.loads(snap_path.read_text())
    assert snap["rows"][args.route_ordinal]["ordinal"] == args.route_ordinal
    counts = snap["rows"][args.route_ordinal]["counts"]
    group = torch.tensor(counts, device=f"npu:{args.rank}", dtype=torch.int64)
    x1 = torch.zeros((576, 4096), device=f"npu:{args.rank}", dtype=torch.int8)
    w1 = torch_npu.npu_format_cast(torch.zeros((32, 4096, 512), device=f"npu:{args.rank}", dtype=torch.int32), 29)
    ws1 = torch.ones((32, 4096), device=f"npu:{args.rank}", dtype=torch.int64)
    xs1 = torch.ones((576,), device=f"npu:{args.rank}", dtype=torch.float32)
    assist = torch.zeros((32, 4096), device=f"npu:{args.rank}", dtype=torch.float32)
    hidden, scale = torch.ops._C_ascend.grouped_matmul_swiglu_quant_v2(
        x=x1, weight=[w1], weight_scale=[ws1], x_scale=xs1,
        group_list=group, group_list_type=1, dequant_mode=0,
        swiglu_limit=0.0, weight_assist_matrix=[assist])
    w2_banks = [torch_npu.npu_format_cast(torch.zeros((32, 2048, 512), device=f"npu:{args.rank}", dtype=torch.int32), 29)
                for _ in range(args.weight_banks)]
    assert len({w.data_ptr() for w in w2_banks}) == args.weight_banks
    ws2 = torch.ones((32, 1, 4096), device=f"npu:{args.rank}", dtype=torch.int64)
    bias2 = torch.zeros((32, 4096), device=f"npu:{args.rank}", dtype=torch.float32)

    def gmm2(bank):
        return torch_npu.npu_grouped_matmul(
            x=[hidden], weight=[w2_banks[bank]], scale=[ws2], bias=[bias2],
            per_token_scale=[scale], split_item=2, group_list_type=1,
            group_type=0, group_list=group, output_dtype=torch.bfloat16)[0]

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        outputs = [gmm2(i % args.weight_banks) for i in range(16)]
    for _ in range(5):
        graph.replay()
    torch.npu.synchronize()
    experimental = _ExperimentalConfig(profiler_level=ProfilerLevel.Level1, aic_metrics=AiCMetrics.MemoryAccess)
    trace_dir = args.out / "profile"
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
                 schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=2, repeat=1),
                 on_trace_ready=tensorboard_trace_handler(str(trace_dir)),
                 experimental_config=experimental) as prof:
        for _ in range(3):
            graph.replay()
            prof.step()
    torch.npu.synchronize()
    assert all(list(t.shape) == [576, 4096] and bool(torch.isfinite(t).all().item()) for t in outputs)
    record = {"status": "profile_completed", "rank": args.rank,
              "route_ordinal": args.route_ordinal, "route_counts": counts,
              "route_count_sha256": hashlib.sha256(json.dumps(counts).encode()).hexdigest(),
              "weight_banks": args.weight_banks,
              "weight_bank_bytes": sum(w.numel() * w.element_size() for w in w2_banks),
              "graph_captured_calls_per_replay": 16,
              "profile_active_replays": 2,
              "post_replay_all_outputs_finite": True,
              "timing_use": "none; profile perturbation excludes this from service latency"}
    (args.out / "profile_request.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
