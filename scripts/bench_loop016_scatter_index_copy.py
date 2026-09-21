#!/usr/bin/env python3
import json, statistics
from pathlib import Path
import torch
import torch_npu  # noqa: F401
from vllm_ascend.utils import enable_custom_op
enable_custom_op(); torch.npu.set_device(0)
root=Path('/data/wio/Inference_Foundry')
idx_cpu=torch.load(root/'evidence/20260920_loop015_cold_kernels/probe/scatter_pid803128.pt',map_location='cpu',weights_only=True)
idx=idx_cpu.npu()
updates=torch.randn((8096,1,512),dtype=torch.float32,device='npu:0')
a=torch.zeros((34091,32,1,512),dtype=torch.float32,device='npu:0')
b=torch.zeros_like(a)
lin32=(idx[:,0]*32+idx[:,1]).contiguous()
sk=torch.ops._C_ascend.npu_scatter_nd_update_sk
result={'shape':[34091,32,1,512], 'indices':8096}
def time_op(fn):
    for _ in range(5): fn()
    torch.npu.synchronize()
    t=[]
    for _ in range(25):
        s=torch.npu.Event(enable_timing=True); e=torch.npu.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.npu.synchronize(); t.append(s.elapsed_time(e))
    return {'device_ms_median':statistics.median(t),'samples':t}
try:
    sk(a,idx,updates)
    b.view(-1,512).index_copy_(0,lin32,updates.view(-1,512))
    torch.npu.synchronize()
    result['same_output']=bool(torch.equal(a,b))
    if not result['same_output']: raise RuntimeError('output mismatch')
    result['sk']=time_op(lambda: sk(a,idx,updates))
    result['index_copy_int32']=time_op(lambda: b.view(-1,512).index_copy_(0,lin32,updates.view(-1,512)))
    result['pass']=True
except Exception as exc:
    result['pass']=False; result['error']=repr(exc)
out=root/'evidence/20260920_loop016_direct_scatter/index_copy_screen.json'
out.write_text(json.dumps(result,indent=2)+'\n')
print({k:v for k,v in result.items() if k not in ('sk','index_copy_int32')})
if 'sk' in result: print('medians',result['sk']['device_ms_median'],result['index_copy_int32']['device_ms_median'])
