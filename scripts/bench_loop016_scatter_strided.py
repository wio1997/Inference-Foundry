#!/usr/bin/env python3
import json,statistics
from pathlib import Path
import torch
import torch_npu  # noqa: F401
from vllm_ascend.utils import enable_custom_op
enable_custom_op(); torch.npu.set_device(0)
root=Path('/data/wio/Inference_Foundry')
idx=torch.load(root/'evidence/20260920_loop015_cold_kernels/probe/scatter_pid803128.pt',map_location='cpu',weights_only=True).npu()
x=torch.randn((8096,1,512),dtype=torch.float32,device='npu:0')
base_shape=(68182,32,1,512)
a_base=torch.zeros(base_shape,dtype=torch.float32,device='npu:0')
b_base=torch.zeros_like(a_base)
a=a_base[::2]; b=b_base[::2]
span=(b.shape[0]-1)*b.stride(0)//512+(b.shape[1]-1)*b.stride(1)//512+1
flat=b.as_strided((span,512),(512,1))
sk=torch.ops._C_ascend.npu_scatter_nd_update_sk
def fast():
    linear=(idx[:,0]*(b.stride(0)//512)+idx[:,1]*(b.stride(1)//512)).contiguous()
    flat.index_copy_(0,linear,x.view(-1,512))
sk(a,idx,x); fast(); torch.npu.synchronize()
same=bool(torch.equal(a,b))
if not same: raise RuntimeError('strided output mismatch')
def bench(fn):
    for _ in range(4): fn()
    torch.npu.synchronize()
    values=[]
    for _ in range(20):
        s=torch.npu.Event(enable_timing=True);e=torch.npu.Event(enable_timing=True)
        s.record();fn();e.record();torch.npu.synchronize();values.append(s.elapsed_time(e))
    return {'median_ms':statistics.median(values),'samples_ms':values}
result={'same_output':same,'cache_shape':list(a.shape),'cache_stride':list(a.stride()),'span_rows':span,'sk':bench(lambda:sk(a,idx,x)),'index_copy_with_flatten':bench(fast)}
out=root/'evidence/20260920_loop016_direct_scatter/strided_screen.json'
out.write_text(json.dumps(result,indent=2)+'\n')
print('same',same,'stride',a.stride(),'medians',result['sk']['median_ms'],result['index_copy_with_flatten']['median_ms'])
