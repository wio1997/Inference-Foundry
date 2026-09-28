#!/usr/bin/env python3
"""One-card isolated CANN Graph native-task tagging feasibility probe."""
from __future__ import annotations

import ctypes
import hashlib
import json
from pathlib import Path

import torch
import torch_npu  # noqa: F401

ROOT = Path('/data/wio/Inference_Foundry')
OUT = ROOT / 'evidence/20260928_loop081_bound/run587'
OUT.mkdir(parents=True, exist_ok=False)
api = ctypes.CDLL('libascendcl.so').aclrtCacheLastTaskExtendInfo
api.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
api.restype = ctypes.c_int

def tag(label: str) -> int:
    payload = f'RUN587:{label}'.encode()
    rc = api(payload, len(payload))
    if rc != 0:
        raise RuntimeError(f'tag {label} failed: {rc}')
    return rc

torch.npu.set_device(0)
x = torch.ones((64, 64), device='npu:0', dtype=torch.bfloat16)
w = torch.eye(64, device='npu:0', dtype=torch.bfloat16)
dest = torch.empty_like(x)
for _ in range(2):
    warm = torch.matmul(x, w)
    dest.copy_(warm)
torch.npu.synchronize()

graph = torch.npu.NPUGraph()
with torch.npu.graph(graph):
    partial = torch.matmul(x, w)
    tag('AFTER_MATMUL')
    dest.copy_(partial)
    tag('AFTER_COPY')
    final = torch.add(dest, x)
    tag('AFTER_ADD')
graph.replay()
torch.npu.synchronize()
if not torch.all(final == 2).item():
    raise RuntimeError('Graph output mismatch')
dump = OUT / 'graph.json'
graph.debug_dump(str(dump))
rows = json.loads(dump.read_text())
tagged = []
for row in rows:
    info = row.get('args', {}).get('ExtendInfo', '')
    if info.startswith('RUN587:'):
        tagged.append({'tag':info, 'name':row['name'], 'model':row['args']['Model Id'],
                       'stream':row['args']['Stream Id'], 'task':row['args']['Task Id'],
                       'task_type':row['args']['Task Type']})
result = {
    'status':'probe_complete',
    'torch_npu_version':torch_npu.__version__,
    'graph_sha256':hashlib.sha256(dump.read_bytes()).hexdigest(),
    'task_count':len(rows),
    'tagged':tagged,
    'expected_tags':['RUN587:AFTER_MATMUL','RUN587:AFTER_COPY','RUN587:AFTER_ADD'],
    'can_tag_copy_task':any(x['tag']=='RUN587:AFTER_COPY' and x['task_type']=='MEMCPY_ASYNC'
                            for x in tagged),
    'can_tag_matmul_task':any(x['tag']=='RUN587:AFTER_MATMUL' and 'MatMul' in x['name']
                              for x in tagged),
    'scope':'isolated one-card Graph only; no production identity/time/bound',
}
(OUT/'probe.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('graph_sha256','scope')}))
