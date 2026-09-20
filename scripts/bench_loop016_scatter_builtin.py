#!/usr/bin/env python3
import json, statistics, time
from pathlib import Path
import torch
import torch_npu
from vllm_ascend.utils import enable_custom_op
enable_custom_op()
torch.npu.set_device(0)
root=Path('/data/wio/Inference_Foundry')
idx=torch.load(root/'evidence/20260920_loop015_cold_kernels/probe/scatter_pid803128.pt',map_location='cpu',weights_only=True).npu()
updates=torch.randn((8096,1,512),dtype=torch.float32,device='npu:0')
shape=(34091,32,1,512)
a=torch.zeros(shape,dtype=torch.float32,device='npu:0')
b=torch.zeros_like(a)
sk=torch.ops._C_ascend.npu_scatter_nd_update_sk
builtin=torch_npu.npu_scatter_nd_update_
result={'shape':shape,'variant':'torch_npu.npu_scatter_nd_update_'}
try:
    sk(a,idx,updates)
    builtin(b,idx,updates)
    torch.npu.synchronize()
    result['same_output']=bool(torch.equal(a,b))
    if not result['same_output']:
        raise RuntimeError('output mismatch')
    for name,op,cache in (('sk',sk,a),('builtin',builtin,b)):
        for _ in range(5): op(cache,idx,updates)
        torch.npu.synchronize()
        samples=[]
        for _ in range(25):
            start=torch.npu.Event(enable_timing=True); end=torch.npu.Event(enable_timing=True)
            start.record(); op(cache,idx,updates); end.record()
            torch.npu.synchronize()
            samples.append(start.elapsed_time(end))
        result[name]={'device_ms_median':statistics.median(samples),'samples':samples}
    result['pass']=True
except Exception as exc:
    result['pass']=False
    result['error']=repr(exc)
out=root/'evidence/20260920_loop016_direct_scatter'
out.mkdir(exist_ok=True)
(out/'builtin_screen.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('sk','builtin')}))
if 'sk' in result: print('medians',result['sk']['device_ms_median'],result['builtin']['device_ms_median'])
