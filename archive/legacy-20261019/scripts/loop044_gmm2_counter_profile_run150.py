#!/usr/bin/env python3
"""One-card product-shape GMM2 MemoryAccess counter profile."""
import json
from pathlib import Path
import torch,torch_npu
from torch_npu.profiler import profile,ProfilerActivity,ProfilerLevel,AiCMetrics,_ExperimentalConfig,tensorboard_trace_handler
from vllm_ascend.utils import enable_custom_op
assert enable_custom_op()
torch.npu.config.allow_internal_format=True
torch.npu.set_device(0)
root=Path('/data/wio/Inference_Foundry')
out=root/'evidence/20260925_loop044_target/run150/profile'
out.mkdir(parents=True,exist_ok=True)
snap=json.loads((root/'evidence/20260925_loop039_gmm/run121/counts/rank0_cycle64.json').read_text())
rows=[row['counts'] for row in snap['rows'][43:]]
real=min(rows,key=lambda c:(abs(sum(x>0 for x in c)-15)+abs(sum(c)-69)/10))
assert len(real)==32 and 0<sum(real)<=576
x=torch.zeros((576,2048),device='npu:0',dtype=torch.int8)
w=torch_npu.npu_format_cast(torch.zeros((32,2048,512),device='npu:0',dtype=torch.int32),29)
assert torch_npu.get_npu_format(w)==29
ws=torch.ones((32,1,4096),device='npu:0',dtype=torch.int64)
xs=torch.ones((576,),device='npu:0',dtype=torch.float32)
group=torch.tensor(real,device='npu:0',dtype=torch.int64)
w1=torch_npu.npu_format_cast(torch.zeros((32,4096,512),device='npu:0',dtype=torch.int32),29)
w1_scale=torch.ones((32,4096),device='npu:0',dtype=torch.int64)
w1_input=torch.zeros((576,4096),device='npu:0',dtype=torch.int8)
w1_xscale=torch.ones((576,),device='npu:0',dtype=torch.float32)
assist=torch.zeros((32,4096),device='npu:0',dtype=torch.float32)
hidden,product_scale=torch.ops._C_ascend.grouped_matmul_swiglu_quant_v2(
    x=w1_input,weight=[w1],weight_scale=[w1_scale],x_scale=w1_xscale,
    group_list=group,group_list_type=1,dequant_mode=0,
    swiglu_limit=0.0,weight_assist_matrix=[assist])
bias2=torch.zeros((32,4096),device='npu:0',dtype=torch.float32)
def invoke():
 return torch_npu.npu_grouped_matmul(x=[hidden],weight=[w],scale=[ws],bias=[bias2],
  per_token_scale=[product_scale],split_item=2,group_list_type=1,group_type=0,
  group_list=group,output_dtype=torch.bfloat16)[0]
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
summary={'run':'run150','device':'npu:0','kernel':'npu_grouped_matmul GMM2 with bias and GMM1-produced input','counts':real,'tokens':sum(real),'active_experts':sum(c>0 for c in real),'input_shape':[576,2048],'weight_shape':[32,2048,512],'weight_format':29,'profiler_level':'Level1','aic_metrics':'MemoryAccess'}
(root/'evidence/20260925_loop044_target/run150/profile_request.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary),flush=True)
