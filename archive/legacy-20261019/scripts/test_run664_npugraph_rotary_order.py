#!/usr/bin/env python3
"""Run664 isolated NPU Graph dynamic index and post-replay ordering diagnostic."""
import json
import torch
import torch_npu

torch.npu.set_device(0)
table = torch.arange(256 * 16, dtype=torch.float32, device="npu:0").view(256, 16)
indices = torch.arange(96, dtype=torch.int64, device="npu:0")
output = torch.empty((96, 16), dtype=torch.float32, device="npu:0")
output.copy_(table.index_select(0, indices))
torch.npu.synchronize()
graph = torch.npu.NPUGraph()
with torch.npu.graph(graph):
    output.copy_(table.index_select(0, indices))
torch.npu.synchronize()
rows = []
for base in (8, 40, 80):
    indices.copy_(torch.arange(base, base + 96, dtype=torch.int64, device="npu:0"))
    graph.replay()
    torch.npu.synchronize()
    expected = table.index_select(0, indices)
    rows.append({"base": base, "graph_dynamic_equal": bool(torch.equal(output, expected))})
    output.copy_(torch.full_like(output, -1))
    graph.replay()
    output.copy_(expected)
    torch.npu.synchronize()
    rows[-1]["post_replay_eager_no_wait_equal"] = bool(torch.equal(output, expected))
    graph.replay()
    torch.npu.synchronize()
    output.copy_(expected)
    torch.npu.synchronize()
    rows[-1]["post_replay_eager_with_wait_equal"] = bool(torch.equal(output, expected))
print(json.dumps({"torch_npu": torch_npu.__version__, "rows": rows}, indent=2))
