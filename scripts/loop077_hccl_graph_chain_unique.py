#!/usr/bin/env python3
"""Exact-order TP8 Graph HCCL chain with distinct buffers and per-call checks."""
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "evidence/20260927_loop077_bound/run378/ledger.json"
OUTPUT = Path(os.getenv("EXTREME_HCCL_GRAPH_CHAIN_OUTPUT", str(ROOT / "evidence/20260927_loop077_bound/run387/chain.json")))
RANK = int(os.environ["RANK"])
LOCAL = int(os.environ["LOCAL_RANK"])
WORLD = int(os.environ["WORLD_SIZE"])
assert WORLD == 8
torch.npu.set_device(LOCAL)
dist.init_process_group("hccl", rank=RANK, world_size=WORLD)
ordered = json.loads(LEDGER.read_text())["ordered_tasks"]
assert len(ordered) == 265
DTYPES = {"BFP16": torch.bfloat16, "FP32": torch.float32}
buffers = []
for i, row in enumerate(ordered):
    count = row["elements"]
    td = DTYPES[row["dtype"]]
    value = RANK + 1 + (i % 100)
    chunk = torch.full((count,), value, dtype=td, device=f"npu:{LOCAL}")
    full = torch.full((count * WORLD,), value, dtype=td, device=f"npu:{LOCAL}")
    out_chunk = torch.empty_like(chunk)
    out_full = torch.empty_like(full)
    buffers.append((chunk, full, out_chunk, out_full))


def chain():
    for row, (chunk, full, out_chunk, out_full) in zip(ordered, buffers):
        if row["kind"] == "hcom_allGather":
            dist.all_gather_into_tensor(out_full, chunk)
        elif row["kind"] == "hcom_reduceScatter":
            dist.reduce_scatter_tensor(out_chunk, full)
        elif row["kind"] == "hcom_alltoall":
            dist.all_to_all_single(out_full, full)
        else:
            raise AssertionError(row["kind"])

for _ in range(3):
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
for i, (row, (_, _, out_chunk, out_full)) in enumerate(zip(ordered, buffers)):
    value = i % 100
    if row["kind"] == "hcom_reduceScatter":
        expected = torch.full((row["elements"],), 36 + 8 * value, dtype=DTYPES[row["dtype"]])
        assert torch.equal(out_chunk.cpu(), expected), (i, row)
    else:
        expected = (torch.arange(1, WORLD + 1, dtype=DTYPES[row["dtype"]]) + value).repeat_interleave(row["elements"])
        assert torch.equal(out_full.cpu(), expected), (i, row)
dist.barrier()
samples = []
for _ in range(20):
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    host = time.perf_counter()
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    samples.append({"device_ms": start.elapsed_time(end),
                    "host_submit_and_wait_ms": (time.perf_counter() - host) * 1000})
rows = [None] * WORLD
dist.all_gather_object(rows, {"rank": RANK, "samples": samples, "checked_operations": len(ordered)})
if RANK == 0:
    latest = [max(row["samples"][i]["device_ms"] for row in rows) for i in range(20)]
    out = {"status": "isolated_exact_order_HCCL_Graph_distinct_buffer_capacity",
           "world": WORLD, "collectives_per_chain": len(ordered),
           "all_rank_each_operation_output_checks_passed": True,
           "distinct_buffers_per_operation": True, "rank_samples": rows,
           "latest_rank_device_median_ms": statistics.median(latest),
           "latest_rank_device_range_ms": [min(latest), max(latest)],
           "limits": ["Synthetic rank-valued buffers and distinct per-call storage; no model compute/HBM/producer contention or Product overlap.",
                      "Current TP8 task order/count/dtype is mirrored from Run378, but Python collective Graph capture may choose different CANN internals than original Target Graph.",
                      "Attained synthetic Graph service, not strict hardware floor or Product exposed communication."]}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"status": out["status"], "median_ms": out["latest_rank_device_median_ms"]}), flush=True)
dist.destroy_process_group()
