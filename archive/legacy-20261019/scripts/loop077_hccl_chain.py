#!/usr/bin/env python3
"""Isolated exact-order TP8 HCCL chain capacity, distinct from Product DAG."""
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
OUTPUT = ROOT / "evidence/20260927_loop077_bound/run379/chain.json"
RANK = int(os.environ["RANK"])
LOCAL = int(os.environ["LOCAL_RANK"])
WORLD = int(os.environ["WORLD_SIZE"])
assert WORLD == 8

torch.npu.set_device(LOCAL)
dist.init_process_group("hccl", rank=RANK, world_size=WORLD)
ordered = json.loads(LEDGER.read_text())["ordered_tasks"]
assert len(ordered) == 265
DTYPES = {"BFP16": torch.bfloat16, "FP32": torch.float32}
buffers = {}
for row in ordered:
    key = (row["kind"], row["elements"], row["dtype"])
    if key in buffers:
        continue
    kind, count, dtype = key
    td = DTYPES[dtype]
    chunk = torch.full((count,), RANK + 1, dtype=td, device=f"npu:{LOCAL}")
    full = torch.full((count * WORLD,), RANK + 1, dtype=td, device=f"npu:{LOCAL}")
    out_chunk = torch.empty_like(chunk)
    out_full = torch.empty_like(full)
    buffers[key] = (chunk, full, out_chunk, out_full)


def chain():
    for row in ordered:
        key = (row["kind"], row["elements"], row["dtype"])
        chunk, full, out_chunk, out_full = buffers[key]
        if row["kind"] == "hcom_allGather":
            dist.all_gather_into_tensor(out_full, chunk)
        elif row["kind"] == "hcom_reduceScatter":
            dist.reduce_scatter_tensor(out_chunk, full)
        elif row["kind"] == "hcom_alltoall":
            dist.all_to_all_single(out_full, full)
        else:
            raise AssertionError(row["kind"])

for _ in range(5):
    chain()
torch.npu.synchronize()
for (kind, count, dtype), (_, _, out_chunk, out_full) in buffers.items():
    if kind == "hcom_reduceScatter":
        assert torch.equal(out_chunk.cpu(), torch.full((count,), 36, dtype=DTYPES[dtype]))
    else:
        expected = torch.arange(1, 9, dtype=DTYPES[dtype]).repeat_interleave(count)
        assert torch.equal(out_full.cpu(), expected), (kind, count, dtype)
dist.barrier()
torch.npu.synchronize()
samples = []
for _ in range(20):
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    host = time.perf_counter()
    start.record()
    chain()
    end.record()
    end.synchronize()
    samples.append({"device_ms": start.elapsed_time(end),
                    "host_submit_and_wait_ms": (time.perf_counter() - host) * 1000})
rows = [None] * WORLD
dist.all_gather_object(rows, {"rank": RANK, "samples": samples})
if RANK == 0:
    latest = [max(row["samples"][i]["device_ms"] for row in rows) for i in range(20)]
    output = {"status": "isolated_exact_order_hccl_chain_capacity", "source": str(LEDGER.relative_to(ROOT)),
              "world": WORLD, "result_checks_passed": True, "warmup_chains": 5, "measured_chains": 20,
              "collectives_per_chain": len(ordered), "rank_samples": rows,
              "latest_rank_device_ms": latest,
              "latest_rank_device_median_ms": statistics.median(latest),
              "latest_rank_device_range_ms": [min(latest), max(latest)],
              "limits": ["Synthetic rank-valued buffers, no model compute/HBM/KV contention, no real producer arrivals or consumer dependencies.",
                         "Device event interval on the current stream is an attained isolated chain observation, not a strict hardware floor or additive Product HCCL cost.",
                         "Correct task order/count/dtype is from Run378; exact borrowed CANN kernel/Graph scheduling may differ from torch.distributed replay."]}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"status": output["status"], "median_ms": output["latest_rank_device_median_ms"]}), flush=True)
dist.destroy_process_group()
