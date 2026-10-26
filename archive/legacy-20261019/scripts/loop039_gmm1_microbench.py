#!/usr/bin/env python3
"""Single-910B3 diagnostic route-sensitivity microbenchmark for product GMM1."""
import json,statistics,time
from pathlib import Path
import torch,torch_npu
from vllm_ascend.utils import enable_custom_op
assert enable_custom_op()
torch.npu.set_device(0)
root=Path('/data/wio/Inference_Foundry')
snap=json.loads((root/'evidence/20260925_loop039_gmm/run121/counts/rank0_cycle64.json').read_text())
rows=[row['counts'] for row in snap['rows'][43:]]
real=min(rows,key=lambda c:(abs(sum(x>0 for x in c)-15)+abs(sum(c)-69)/10))
m=sum(real)
assert len(real)==32 and 0<m<=576
dense=[m//32+(i<m%32) for i in range(32)]
single=[m]+[0]*31
print(json.dumps({'product_x_shape':[576,4096],'product_w1_shape':[32,4096,512],
                  'real_counts':real,'real_m':m,'active_real':sum(x>0 for x in real),
                  'active_dense':sum(x>0 for x in dense),'active_single':1}),flush=True)
x=torch.zeros((576,4096),device='npu:0',dtype=torch.int8)
w=torch.zeros((32,4096,512),device='npu:0',dtype=torch.int32)
ws=torch.ones((32,4096),device='npu:0',dtype=torch.int64)
xs=torch.ones((576,),device='npu:0',dtype=torch.float32)
def invoke(counts):
    group=torch.tensor(counts,device='npu:0',dtype=torch.int64)
    out,scale=torch.ops._C_ascend.grouped_matmul_swiglu_quant_v2(
      x=x,weight=[w],weight_scale=[ws],x_scale=xs,group_list=group,
      group_list_type=1,dequant_mode=0,swiglu_limit=0.0)
    torch.npu.synchronize()
    return tuple(out.shape),tuple(scale.shape)
for label,counts in [('real',real),('dense',dense),('single',single)]:
    t=time.perf_counter()
    shapes=invoke(counts)
    print(json.dumps({'smoke':label,'shapes':shapes,'wall_s':time.perf_counter()-t}),flush=True)
