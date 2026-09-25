#!/usr/bin/env python3
"""One-card product-shape GMM1 memory-access counter profile."""
import json
from pathlib import Path
import torch,torch_npu
from torch_npu.profiler import profile,ProfilerActivity,ProfilerLevel,AiCMetrics,_ExperimentalConfig,tensorboard_trace_handler
from vllm_ascend.utils import enable_custom_op
assert enable_custom_op()
torch.npu.config.allow_internal_format=True
torch.npu.set_device(0)
root=Path('/data/wio/Inference_Foundry')
out=root/'evidence/20260925_loop044_target/run148/profile'
out.mkdir(parents=True,exist_ok=True)
snap=json.loads((root/'evidence/20260925_loop039_gmm/run121/counts/rank0_cycle64.json').read_text())
rows=[row['counts'] for row in snap['rows'][43:]]
real=min(rows,key=lambda c:(abs(sum(x>0 for x in c)-15)+abs(sum(c)-69)/10))
assert len(real)==32 and 0<sum(real)<=576
x=torch.zeros((576,4096),device='npu:0',dtype=torch.int8)
w=torch_npu.npu_format_cast(torch.zeros((32,4096,512),device='npu:0',dtype=torch.int32),29)
assert torch_npu.get_npu_format(w)==29
ws=torch.ones((32,4096),device='npu:0',dtype=torch.int64)
xs=torch.ones((576,),device='npu:0',dtype=torch.float32)
assist=torch.zeros((32,4096),device='npu:0',dtype=torch.float32)
group=torch.tensor(real,device='npu:0',dtype=torch.int64)
def invoke():
 return torch.ops._C_ascend.grouped_matmul_swiglu_quant_v2(
   x=x,weight=[w],weight_scale=[ws],x_scale=xs,
   group_list=group,group_list_type=1,dequant_mode=0,
   swiglu_limit=0.0,weight_assist_matrix=[assist])
for _ in range(10):invoke()
torch.npu.synchronize()
config=_ExperimentalConfig(profiler_level=ProfilerLevel.Level1,aic_metrics=AiCMetrics.MemoryAccess)
with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.NPU],
             schedule=torch_npu.profiler.schedule(wait=0,warmup=1,active=4,repeat=1),
             on_trace_ready=tensorboard_trace_handler(str(out)),
             experimental_config=config) as prof:
 for _ in range(5):
  invoke()
  prof.step()
torch.npu.synchronize()
summary={'run':'run148','device':'npu:0','kernel':'grouped_matmul_swiglu_quant_v2','counts':real,'tokens':sum(real),'active_experts':sum(c>0 for c in real),'input_shape':[576,4096],'weight_shape':[32,4096,512],'weight_format':29,'profiler_level':'Level1','aic_metrics':'MemoryAccess'}
(root/'evidence/20260925_loop044_target/run148/profile_request.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary),flush=True)
