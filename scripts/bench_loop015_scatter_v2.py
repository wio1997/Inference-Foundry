#!/usr/bin/env python3
import json
import statistics
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from vllm_ascend.utils import enable_custom_op

enable_custom_op()
torch.npu.set_device(0)
root = Path("/data/wio/Inference_Foundry")
out = root / "evidence/20260920_loop015_cold_kernels/scatter_v2_screen.json"
indices_cpu = torch.load(root / "evidence/20260920_loop015_cold_kernels/probe/scatter_pid803128.pt", map_location="cpu", weights_only=True)
indices = indices_cpu.npu()
shape = (34091, 32, 1, 512)
torch.manual_seed(15)
updates = torch.randn((8096, 1, 512), dtype=torch.float32).npu()
a = torch.zeros(shape, dtype=torch.float32, device="npu:0")
b = torch.zeros_like(a)
ops = {
    "sk": torch.ops._C_ascend.npu_scatter_nd_update_sk,
    "v2": torch.ops._C_ascend.npu_scatter_nd_update_v2,
}
ops["sk"](a, indices, updates)
ops["v2"](b, indices, updates)
torch.npu.synchronize()
same = bool(torch.equal(a, b))
if not same:
    raise RuntimeError("V2 output differs from SK")
results = {"same_output": same, "shape": shape, "updates_shape": list(updates.shape), "indices_shape": list(indices.shape), "runs": 25}
for name, cache in (("sk", a), ("v2", b)):
    op = ops[name]
    for _ in range(5):
        op(cache, indices, updates)
    torch.npu.synchronize()
    device_ms = []
    wall_ms = []
    for _ in range(25):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        t0 = time.perf_counter()
        start.record()
        op(cache, indices, updates)
        end.record()
        torch.npu.synchronize()
        wall_ms.append((time.perf_counter() - t0) * 1000)
        device_ms.append(start.elapsed_time(end))
    results[name] = {"device_ms_median": statistics.median(device_ms), "device_ms_min": min(device_ms), "wall_ms_median": statistics.median(wall_ms), "device_ms": device_ms, "wall_ms": wall_ms}
out.write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps({k:v for k,v in results.items() if k in ("same_output","sk","v2")}, indent=2)[:1200])
