#!/usr/bin/env python3
"""Deciding native ownership probe: pending H2D vs CPU staging overwrite."""
import hashlib
import json
from pathlib import Path
import time
import torch
import torch_npu

torch.npu.set_device(0)
torch.set_num_threads(1)
device=torch.device('npu:0')
blocker=torch.npu.Stream(device=device)
copier=torch.npu.Stream(device=device)
a=torch.ones((4096,4096),dtype=torch.bfloat16,device=device)
b=torch.ones_like(a);out=torch.empty_like(a)
torch.mm(a,b,out=out);torch.npu.synchronize()
cases=[]
for dtype,n in [(torch.int32,24),(torch.int64,16)]:
    for ownership in ['shared_staging','independent_generation']:
        original=torch.arange(n,dtype=dtype)+101
        replacement=torch.arange(n,dtype=dtype)+10001
        src=torch.empty(n,dtype=dtype,pin_memory=True);src.copy_(original)
        assert src.is_pinned()
        stable=None
        if ownership=='independent_generation':
            stable=torch.empty_like(src,pin_memory=True);stable.copy_(src)
        read_source=src if stable is None else stable
        dst=torch.empty(n,dtype=dtype,device=device)
        gate=torch.npu.Event();done=torch.npu.Event()
        with torch.npu.stream(blocker):
            for _ in range(64):torch.mm(a,b,out=out)
            gate.record()
        with torch.npu.stream(copier):
            copier.wait_event(gate)
            dst.copy_(read_source,non_blocking=True)
            done.record()
        before=time.monotonic_ns();pending_before=not gate.query() and not done.query()
        src.copy_(replacement)
        pending_after=not gate.query();after=time.monotonic_ns()
        done.synchronize();got=dst.cpu()
        assert pending_before and pending_after,'INVALID: gate was not pending throughout CPU overwrite'
        cases.append(dict(dtype=str(dtype),n=n,ownership=ownership,pinned=src.is_pinned(),gate_pending_before=pending_before,gate_pending_after=pending_after,CPU_overwrite_ns=after-before,
            old_value_differences=int((got!=original).sum()),replacement_value_differences=int((got!=replacement).sum()),
            got_bytes_sha256=hashlib.sha256(got.numpy().tobytes()).hexdigest(),old_bytes_sha256=hashlib.sha256(original.numpy().tobytes()).hexdigest(),replacement_bytes_sha256=hashlib.sha256(replacement.numpy().tobytes()).hexdigest()))
        torch.npu.synchronize()
result=dict(status='COMPLETE_NATIVE_OWNERSHIP_DISCRIMINATOR',torch=torch.__version__,torch_npu=torch_npu.__version__,device_name=torch.npu.get_device_name(0),cases=cases,
    inference_requests=0,model_or_weight_loads=0,new_profiling=False,scope='INT32[24] and INT64[16] pinned H2D copy_ under a pending stream event, compared with independent-generation staging. BF16 GEMM is bounded gate work only, not a speed benchmark or model qualification.')
Path('result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
