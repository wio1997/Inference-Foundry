#!/usr/bin/env python3
"""All-eight concurrent product-shape GMM Graph service calibration.

This measures isolated operators under all-card concurrency. It does not
measure a FULL Graph or establish compulsory product work/physical BW.
"""
import argparse
import json
import multiprocessing as mp
import os
import statistics
import time
import traceback
from pathlib import Path

ROOT = Path("/data/wio/Inference_Foundry")


def worker(rank, barrier, output_dir, samples, warmups, graph_ops):
    try:
        import torch
        import torch_npu
        from vllm_ascend.utils import enable_custom_op

        assert enable_custom_op()
        torch.npu.config.allow_internal_format = True
        torch.npu.set_device(rank)
        snap = json.loads((ROOT / f"evidence/20260925_loop039_gmm/run121/counts/rank{rank}_cycle64.json").read_text())
        routes = [row["counts"] for row in snap["rows"][43:]]
        counts = min(routes, key=lambda c: abs(sum(x > 0 for x in c) - 15) + abs(sum(c) - 69) / 10)
        assert len(counts) == 32 and 0 < sum(counts) <= 576
        group = torch.tensor(counts, device=f"npu:{rank}", dtype=torch.int64)
        x1 = torch.zeros((576, 4096), device=f"npu:{rank}", dtype=torch.int8)
        w1 = torch_npu.npu_format_cast(torch.zeros((32, 4096, 512), device=f"npu:{rank}", dtype=torch.int32), 29)
        assert torch_npu.get_npu_format(w1) == 29
        ws1 = torch.ones((32, 4096), device=f"npu:{rank}", dtype=torch.int64)
        xs1 = torch.ones((576,), device=f"npu:{rank}", dtype=torch.float32)
        assist = torch.zeros((32, 4096), device=f"npu:{rank}", dtype=torch.float32)
        w2 = torch_npu.npu_format_cast(torch.zeros((32, 2048, 512), device=f"npu:{rank}", dtype=torch.int32), 29)
        assert torch_npu.get_npu_format(w2) == 29
        ws2 = torch.ones((32, 1, 4096), device=f"npu:{rank}", dtype=torch.int64)
        bias2 = torch.zeros((32, 4096), device=f"npu:{rank}", dtype=torch.float32)

        def gmm1():
            return torch.ops._C_ascend.grouped_matmul_swiglu_quant_v2(
                x=x1, weight=[w1], weight_scale=[ws1], x_scale=xs1,
                group_list=group, group_list_type=1, dequant_mode=0,
                swiglu_limit=0.0, weight_assist_matrix=[assist])

        hidden, scale = gmm1()
        assert list(hidden.shape) == [576, 2048]

        def gmm2():
            return torch_npu.npu_grouped_matmul(
                x=[hidden], weight=[w2], scale=[ws2], bias=[bias2],
                per_token_scale=[scale], split_item=2, group_list_type=1,
                group_type=0, group_list=group, output_dtype=torch.bfloat16)[0]

        output = gmm2()
        assert list(output.shape) == [576, 4096]
        assert bool(torch.isfinite(output).all().item())
        torch.npu.synchronize()
        result = {"rank": rank, "route_tokens": sum(counts), "active_experts": sum(x > 0 for x in counts),
                  "w1_shape": list(w1.shape), "w2_shape": list(w2.shape), "weight_format": 29,
                  "cases": {}}
        for name, invoke in (("gmm1", gmm1), ("gmm2", gmm2)):
            for _ in range(warmups):
                invoke()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                captured_outputs = [invoke() for _ in range(graph_ops)]
            assert captured_outputs
            torch.npu.synchronize()
            for _ in range(warmups):
                graph.replay()
            torch.npu.synchronize()
            barrier.wait(timeout=120)
            host_start = time.perf_counter_ns()
            events = []
            for _ in range(samples):
                start = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                start.record()
                graph.replay()
                end.record()
                events.append((start, end))
            torch.npu.synchronize()
            host_end = time.perf_counter_ns()
            duration_us = [start.elapsed_time(end) * 1000 / graph_ops for start, end in events]
            assert all(d > 0 for d in duration_us)
            result["cases"][name] = {
                "host_start_ns": host_start, "host_end_ns": host_end,
                "graph_ops": graph_ops,
                "device_event_us_per_op": duration_us,
                "median_us": statistics.median(duration_us),
                "p10_us": sorted(duration_us)[int(0.1 * (samples - 1))],
                "p90_us": sorted(duration_us)[int(0.9 * (samples - 1))],
            }
            barrier.wait(timeout=120)
        (output_dir / f"rank{rank}.json").write_text(json.dumps(result, indent=2) + "\n")
    except Exception:
        (output_dir / f"rank{rank}.error.txt").write_text(traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=80)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--graph-ops", type=int, default=16)
    parser.add_argument("--world-size", type=int, choices=(1, 8), default=8)
    args = parser.parse_args()
    assert args.samples >= 20 and args.warmups >= 3 and args.graph_ops >= 8
    args.out.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(args.world_size)
    jobs = [ctx.Process(target=worker, args=(rank, barrier, args.out, args.samples, args.warmups, args.graph_ops)) for rank in range(args.world_size)]
    for job in jobs:
        job.start()
    for job in jobs:
        job.join(timeout=240)
    failures = []
    for rank, job in enumerate(jobs):
        if job.is_alive():
            job.terminate()
            failures.append((rank, "timeout"))
        elif job.exitcode != 0:
            failures.append((rank, job.exitcode))
    if failures:
        raise RuntimeError(f"all-rank service invalid: {failures}")
    rows = [json.loads((args.out / f"rank{rank}.json").read_text()) for rank in range(args.world_size)]
    summary = {"status": "isolated_graph_service", "ranks": args.world_size,
               "samples_per_rank_case": args.samples, "warmups": args.warmups,
               "graph_ops_per_replay": args.graph_ops,
               "cases": {}, "limits": ["operator-only, no FULL Graph, Target/Draft or HCCL contention",
                                    "same operator repeated may benefit from weight cache; no MemoryAccess counter in this run",
                                    "graph replay event includes a bounded graph submission gap; per-op division amortizes but does not remove it",
                                    "event elapsed times are not a hardware physical lower bound or Product critical-path cost"]}
    for name in ("gmm1", "gmm2"):
        starts = [row["cases"][name]["host_start_ns"] for row in rows]
        ends = [row["cases"][name]["host_end_ns"] for row in rows]
        medians = [row["cases"][name]["median_us"] for row in rows]
        assert max(starts) < min(ends), f"no measured all8 concurrency for {name}"
        summary["cases"][name] = {"all_rank_start_spread_ms": (max(starts) - min(starts)) / 1e6,
                                  "all_rank_overlap_ms": (min(ends) - max(starts)) / 1e6,
                                  "rank_median_us": medians,
                                  "slowest_rank_median_us": max(medians)}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
