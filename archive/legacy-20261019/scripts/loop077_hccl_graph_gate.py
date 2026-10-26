#!/usr/bin/env python3
"""TP8 small collective Graph semantic and timing gate on actual decode dtypes."""
import json
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "evidence/20260927_loop077_bound/run381/gate.json"
RANK = int(os.environ["RANK"])
LOCAL = int(os.environ["LOCAL_RANK"])
WORLD = int(os.environ["WORLD_SIZE"])
assert WORLD == 8
torch.npu.set_device(LOCAL)
dist.init_process_group("hccl", rank=RANK, world_size=WORLD)

bf = torch.full((49152,), RANK + 1, dtype=torch.bfloat16, device=f"npu:{LOCAL}")
fp = torch.full((3072,), RANK + 1, dtype=torch.float32, device=f"npu:{LOCAL}")
ag_bf = torch.empty((WORLD * 49152,), dtype=bf.dtype, device=bf.device)
ag_fp = torch.empty((WORLD * 3072,), dtype=fp.dtype, device=fp.device)
rs_in = torch.full((WORLD * 49152,), RANK + 1, dtype=bf.dtype, device=bf.device)
rs_out = torch.empty_like(bf)
a2a_out = torch.empty_like(rs_in)

def chain():
    dist.all_gather_into_tensor(ag_bf, bf)
    dist.all_gather_into_tensor(ag_fp, fp)
    dist.reduce_scatter_tensor(rs_out, rs_in)
    dist.all_to_all_single(a2a_out, rs_in)

for _ in range(5):
    chain()
torch.npu.synchronize()
dist.barrier()
torch.npu.synchronize()
graph = torch.npu.NPUGraph()
with torch.npu.graph(graph):
    chain()
for _ in range(5):
    graph.replay()
torch.npu.synchronize()
for tensor, count in ((ag_bf, 49152), (ag_fp, 3072), (a2a_out, 49152)):
    expected = torch.arange(1, WORLD + 1, dtype=tensor.dtype).repeat_interleave(count)
    assert torch.equal(tensor.cpu(), expected)
assert torch.equal(rs_out.cpu(), torch.full((49152,), 36, dtype=torch.bfloat16))
dist.barrier()
samples = []
for _ in range(30):
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    samples.append(start.elapsed_time(end))
rows = [None] * WORLD
dist.all_gather_object(rows, {"rank": RANK, "samples_ms": samples})
if RANK == 0:
    latest = [max(row["samples_ms"][i] for row in rows) for i in range(30)]
    result = {"status": "tp8_small_hccl_graph_semantic_gate", "world": WORLD,
              "collectives": ["AG49152BF16", "AG3072FP32", "RS49152BF16", "A2A49152BF16"],
              "eager_warmup": 5, "graph_replay_warmup": 5, "graph_result_checks_passed": True,
              "rank_samples": rows, "latest_rank_median_ms": statistics.median(latest),
              "latest_rank_range_ms": [min(latest), max(latest)],
              "limits": ["Four-operation synthetic graph only, no model producers, compute/HBM contention or full265-task order.",
                         "Graph semantic success permits a full-chain fixture, not a Product or strict Hardware bound."]}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "median_ms": result["latest_rank_median_ms"]}), flush=True)
dist.destroy_process_group()
