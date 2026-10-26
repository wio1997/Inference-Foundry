#!/usr/bin/env python3
"""Test queue1 same-thread native Graph task tags on one isolated NPU."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from torch.utils.cpp_extension import load

ROOT=Path('/data/wio/Inference_Foundry')
OUT=ROOT/'evidence/20260928_loop081_bound/run588'
OUT.mkdir(parents=True,exist_ok=False)
if os.environ.get('TASK_QUEUE_ENABLE')!='1':
    raise RuntimeError('queue1 environment required')

SOURCE=ROOT/'scripts/loop081_native_tag_sidecar_run588.cpp'
NPU=Path(torch_npu.__file__).parent
CANN=Path('/usr/local/Ascend/cann-9.1.0/aarch64-linux/lib64')
sidecar=load(name='bound_native_tag_run588',sources=[str(SOURCE)],
             build_directory=str(OUT),verbose=True,
             extra_include_paths=[str(NPU/'include'),str(CANN.parent/'include')],
             extra_ldflags=[f'-L{NPU / "lib"}','-ltorch_npu',
                            f'-L{CANN}','-lascendcl',
                            f'-Wl,-rpath,{NPU / "lib"}',f'-Wl,-rpath,{CANN}'])

torch.npu.set_device(0)
x=torch.ones((64,64),device='npu:0',dtype=torch.bfloat16)
w=torch.eye(64,device='npu:0',dtype=torch.bfloat16)
dest=torch.empty_like(x)
for _ in range(2):
    warm=torch.matmul(x,w)
    dest.copy_(warm)
torch.npu.synchronize()

graph=torch.npu.NPUGraph()
with torch.npu.graph(graph):
    partial=torch.matmul(x,w)
    sidecar.enqueue_tag('RUN588:AFTER_MATMUL')
    dest.copy_(partial)
    sidecar.enqueue_tag('RUN588:AFTER_COPY')
    final=torch.add(dest,x)
    sidecar.enqueue_tag('RUN588:AFTER_ADD')
graph.replay()
torch.npu.synchronize()
if not torch.all(final==2).item():
    raise RuntimeError('Graph output mismatch')
dump=OUT/'graph.json'
graph.debug_dump(str(dump))
rows=json.loads(dump.read_text())
tagged=[]
for row in rows:
    info=row.get('args',{}).get('ExtendInfo','')
    if info.startswith('RUN588:'):
        tagged.append({'tag':info,'name':row['name'],
                       'model':row['args']['Model Id'],
                       'stream':row['args']['Stream Id'],
                       'task':row['args']['Task Id'],
                       'type':row['args']['Task Type']})
tags={x['tag']:x for x in tagged}
checks={
    'all_three_exact':set(tags)=={'RUN588:AFTER_MATMUL','RUN588:AFTER_COPY','RUN588:AFTER_ADD'} and len(tagged)==3,
    'matmul_native': 'RUN588:AFTER_MATMUL' in tags and 'MatMul' in tags['RUN588:AFTER_MATMUL']['name'],
    'copy_native':'RUN588:AFTER_COPY' in tags and tags['RUN588:AFTER_COPY']['type']=='MEMCPY_ASYNC',
    'add_native':'RUN588:AFTER_ADD' in tags and 'Add' in tags['RUN588:AFTER_ADD']['name'] and
                 tags['RUN588:AFTER_ADD']['type'] in ('KERNEL_AIVEC','KERNEL_AICORE'),
    'same_model_stream_and_order':len(tagged)==3 and
        len({(x['model'],x['stream']) for x in tagged})==1 and
        tags['RUN588:AFTER_MATMUL']['task']<tags['RUN588:AFTER_COPY']['task']<tags['RUN588:AFTER_ADD']['task'],
}
result={'status':'native_tag_preflight_pass' if all(checks.values()) else 'native_tag_preflight_fail',
        'queue':1,'torch_npu_version':torch_npu.__version__,
        'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'dump_sha256':hashlib.sha256(dump.read_bytes()).hexdigest(),
        'task_count':len(rows),'tagged':tagged,'checks':checks,
        'scope':'isolated one-card queue1 Graph only; no production task or Bound'}
(OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('source_sha256','dump_sha256','scope')}))
if not all(checks.values()):
    raise RuntimeError('native tag task identity preflight did not pass')
