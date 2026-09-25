#!/usr/bin/env python3
"""Single-910B3 diagnostic route-sensitivity microbenchmark for product GMM1."""
import json,statistics,time
from pathlib import Path
import torch,torch_npu
from vllm_ascend.utils import enable_custom_op
assert enable_custom_op()
torch.npu.config.allow_internal_format=True
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
w=torch_npu.npu_format_cast(w,29)
print(json.dumps({'w_format':torch_npu.get_npu_format(w),'w_shape':list(w.shape)}),flush=True)
assert torch_npu.get_npu_format(w)==29
ws=torch.ones((32,4096),device='npu:0',dtype=torch.int64)
xs=torch.ones((576,),device='npu:0',dtype=torch.float32)
assist=torch.zeros((32,4096),device='npu:0',dtype=torch.float32)
groups={label:torch.tensor(counts,device='npu:0',dtype=torch.int64)
        for label,counts in [('real',real),('dense',dense),('single',single)]}
def invoke(label):
    return torch.ops._C_ascend.grouped_matmul_swiglu_quant_v2(
      x=x,weight=[w],weight_scale=[ws],x_scale=xs,
      group_list=groups[label],group_list_type=1,dequant_mode=0,
      swiglu_limit=0.0,weight_assist_matrix=[assist])
for label in groups:
    for _ in range(20): invoke(label)
torch.npu.synchronize()
result={'run':'run128','kind':'single_npu_gmm1_route_sensitivity',
        'M_active':m,'M_capacity':576,'counts':{'real':real,'dense':dense,'single':single},
        'samples_per_block':50,'blocks':[]}
for label in ('real','dense','single','real'):
    samples=[]
    for _ in range(50):
        start=torch.npu.Event(enable_timing=True)
        end=torch.npu.Event(enable_timing=True)
        start.record()
        output,scale=invoke(label)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    result['blocks'].append({'label':label,'median_ms':statistics.median(samples),
                             'min_ms':min(samples),'max_ms':max(samples),
                             'p10_ms':sorted(samples)[5],'p90_ms':sorted(samples)[45],
                             'samples_ms':samples})
p=root/'evidence/20260925_loop039_gmm/run128_result.json'
p.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='blocks'}),flush=True)
for block in result['blocks']:
    print(json.dumps({k:v for k,v in block.items() if k!='samples_ms'}),flush=True)
