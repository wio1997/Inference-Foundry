#!/usr/bin/env python3
"""One-NPU shape-matched Compressor screen; operator timing is not E2E TTFT."""
import json
import os
import hashlib
import statistics
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from vllm_ascend.utils import enable_custom_op

if os.environ.get("LOOP017_ISOLATED_VENDOR") == "1":
    import vllm_ascend.vllm_ascend_C  # noqa: F401
else:
    enable_custom_op()
torch.npu.set_device(0)
root=Path("/data/wio/Inference_Foundry")
out=Path(os.environ.get("LOOP017_OUT", str(root/"evidence/20260920_loop017_compressor/shape_matched_operator.json")))
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
fingerprint=hashlib.sha256(x[:4].view(torch.uint16).cpu().numpy().tobytes()).hexdigest()
result={
 "input_sha256_first_four_rows":fingerprint,
 "isolated_vendor":os.environ.get("ASCEND_CUSTOM_OPP_PATH") if os.environ.get("LOOP017_ISOLATED_VENDOR") == "1" else None,
 "shape":{"x":[8096,4096],"w":[1024,4096],"state":[34091,2,2048],
          "ape":[4,1024],"norm":[512],"rope":[2025,64],"blocks":[1,524288]},
 "config":{"cmp_ratio":4,"coff":2,"rope_head_dim":64,"cache_mode":1,"block_size":2,"start_pos":0},
 "output_shape":list(y.shape),"output_finite":bool(torch.isfinite(y).all().item()),
 "device_ms_median":statistics.median(device_ms),"device_ms_min":min(device_ms),
 "wall_ms_median":statistics.median(wall_ms),
 "device_ms":device_ms,"wall_ms":wall_ms}
reference_out=os.environ.get("LOOP017_REFERENCE_OUT")
if reference_out:
    torch.save({"fingerprint":fingerprint,"y":y.cpu(),"state":state[:4096].cpu()},reference_out)
reference_path=os.environ.get("LOOP017_REFERENCE")
if reference_path:
    reference=torch.load(reference_path,map_location="cpu",weights_only=True)
    result["reference_input_equal"]=fingerprint==reference["fingerprint"]
    output=y.cpu()
    cache=state[:4096].cpu()
    result["reference_y_allclose"]=bool(torch.allclose(output,reference["y"],rtol=0.01,atol=0.01,equal_nan=False))
    result["reference_state_allclose"]=bool(torch.allclose(cache,reference["state"],rtol=0.01,atol=0.01,equal_nan=False))
    result["reference_y_max_abs"]=float((output.float()-reference["y"].float()).abs().max())
    result["reference_state_max_abs"]=float((cache-reference["state"]).abs().max())
out.write_text(json.dumps(result,indent=2)+"\n")
if reference_path and not (result["reference_input_equal"] and result["reference_y_allclose"] and result["reference_state_allclose"]):
    raise RuntimeError("isolated operator differs from reference")
print(json.dumps({k:v for k,v in result.items() if k not in ("device_ms","wall_ms")},indent=2))
