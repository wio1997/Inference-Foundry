#!/usr/bin/env python3
"""One-NPU shape-matched Compressor screen; operator timing is not E2E TTFT."""
import json
import statistics
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from vllm_ascend.utils import enable_custom_op

enable_custom_op()
torch.npu.set_device(0)
root=Path("/data/wio/Inference_Foundry")
out=root/"evidence/20260920_loop017_compressor/shape_matched_operator.json"
device="npu:0"
torch.manual_seed(1701)
x=torch.randn((8096,4096),dtype=torch.bfloat16,device=device)*0.02
wkv=torch.randn((1024,4096),dtype=torch.bfloat16,device=device)*0.02
wgate=torch.randn((1024,4096),dtype=torch.bfloat16,device=device)*0.02
state=torch.zeros((34091,2,2048),dtype=torch.float32,device=device)
ape=torch.randn((4,1024),dtype=torch.float32,device=device)*0.02
norm=torch.ones((512,),dtype=torch.bfloat16,device=device)
sin=torch.zeros((2025,64),dtype=torch.bfloat16,device=device)
cos=torch.ones((2025,64),dtype=torch.bfloat16,device=device)
blocks=torch.zeros((1,524288),dtype=torch.int32,device=device)
blocks[0,:4048]=torch.arange(1,4049,dtype=torch.int32,device=device)
cu=torch.tensor([0,8096],dtype=torch.int32,device=device)
start_pos=torch.tensor([0],dtype=torch.int32,device=device)
def op():
    return torch.ops._C_ascend.compressor(
        x,wkv,wgate,state,ape,norm,sin,cos,
        state_block_table=blocks,cu_seqlens=cu,seqused=None,start_pos=start_pos,
        rope_head_dim=64,cmp_ratio=4,coff=2,norm_eps=1e-6,rotary_mode=2,cache_mode=1)
for _ in range(5):
    y=op()
torch.npu.synchronize()
device_ms=[]
wall_ms=[]
for _ in range(25):
    begin=torch.npu.Event(enable_timing=True)
    end=torch.npu.Event(enable_timing=True)
    t0=time.perf_counter()
    begin.record()
    y=op()
    end.record()
    torch.npu.synchronize()
    device_ms.append(begin.elapsed_time(end))
    wall_ms.append((time.perf_counter()-t0)*1000)
result={
 "shape":{"x":[8096,4096],"w":[1024,4096],"state":[34091,2,2048],
          "ape":[4,1024],"norm":[512],"rope":[2025,64],"blocks":[1,524288]},
 "config":{"cmp_ratio":4,"coff":2,"rope_head_dim":64,"cache_mode":1,"block_size":2,"start_pos":0},
 "output_shape":list(y.shape),"output_finite":bool(torch.isfinite(y).all().item()),
 "device_ms_median":statistics.median(device_ms),"device_ms_min":min(device_ms),
 "wall_ms_median":statistics.median(wall_ms),
 "device_ms":device_ms,"wall_ms":wall_ms}
out.write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({k:v for k,v in result.items() if k not in ("device_ms","wall_ms")},indent=2))
