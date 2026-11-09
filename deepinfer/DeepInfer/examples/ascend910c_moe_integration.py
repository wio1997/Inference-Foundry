"""Offline two-rank routed-expert integration using real checkpoint matrices.

This exercises gate -> HCCL dispatch -> FP4 expert decode/SwiGLU/down-projection
-> HCCL combine. It is a BF16 reference module, not a complete model runner or
an FP8-activation accuracy/performance result. No standalone HCCL test is run.
"""
import argparse
import datetime
import json
import os
from pathlib import Path

import torch
import torch_npu
import torch.distributed as dist
import torch.nn.functional as F

from deepinfer import PLATFORM
from deepinfer.npu_ops.checkpoint910c import ByteCheckpoint
from deepinfer.npu_ops.moe910c import dispatch_combine,route,swiglu


def cpu_matrix(reader,name):
    data,dtype=reader.tensor(name)
    if dtype in ('BF16','F32'):return data
    scales,sf_dtype=reader.tensor(name.removesuffix('.weight')+'.scale')
    if sf_dtype!='F8_E8M0':raise ValueError('unexpected checkpoint scale format')
    if dtype=='F8_E4M3':values=data.view(torch.float8_e4m3fn).float();rows=32
    elif dtype=='I8' and '.experts.' in name:
        table=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.])
        values=torch.stack((table[(data.int()&15).long()],table[(data.int()>>4).long()]),-1).flatten(-2);rows=1
    else:raise ValueError('unsupported CPU golden storage')
    sf=torch.pow(2.,scales.float()-127.).repeat_interleave(rows,0).repeat_interleave(32,1)
    return (values*sf[:values.shape[0],:values.shape[1]]).bfloat16()


def expert(reader,expert_id,x,weight,layer,device,clamp,cache,cpu=False):
    prefix=f'layers.{layer}.ffn.experts.{expert_id}'
    if expert_id not in cache:
        loader=(lambda n:cpu_matrix(reader,n)) if cpu else (lambda n:reader.matrix(n,device))
        cache[expert_id]=tuple(loader(prefix+f'.w{i}.weight') for i in (1,3,2))
    w1,w3,w2=cache[expert_id]
    gate=(x.float()@w1.float().T).bfloat16();up=(x.float()@w3.float().T).bfloat16()
    if cpu:
        a=gate.float();b=up.float()
        if clamp is not None:a=a.clamp_max(clamp);b=b.clamp(-clamp,clamp)
        hidden=(F.silu(a)*b*weight[:,None]).bfloat16()
    else:hidden=swiglu(gate,up,weight,clamp)
    return (hidden.float()@w2.float().T).bfloat16()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model-dir',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--layer',type=int,default=0);args=parser.parse_args()
    rank=int(os.environ.get('RANK',0));local=int(os.environ.get('LOCAL_RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    if world!=2:raise ValueError('integration is bounded at two ranks')
    torch.npu.set_device(local)
    if not PLATFORM.is_910c_reference:raise RuntimeError('explicit 910C reference profile required')
    dist.init_process_group('hccl',timeout=datetime.timedelta(seconds=180))
    torch.set_num_threads(4)
    reader=ByteCheckpoint(args.model_dir);config=json.loads((args.model_dir/'config.json').read_text())['text_config']
    hidden=config['hidden_size'];num_experts=config['n_routed_experts'];topk=config['num_experts_per_tok'];scale=config['routed_scaling_factor'];clamp=config.get('swiglu_limit')
    gate=cpu_matrix(reader,f'layers.{args.layer}.ffn.gate.weight');bias,dtype=reader.tensor(f'layers.{args.layer}.ffn.gate.bias')
    # One-hot probes give exact, deterministic BF16 gate logits on both devices.
    x=torch.zeros((1,hidden),dtype=torch.bfloat16);x[0,rank]=1
    expected_logits=(x.float()@gate.float().T).bfloat16()
    logits=(x.npu().float()@gate.npu().float().T).bfloat16()
    torch.testing.assert_close(logits.cpu(),expected_logits,rtol=0,atol=0)
    indices,weights=route(logits,bias.npu(),topk,scale)
    scores=F.softplus(expected_logits.float()).sqrt();cpu_indices=torch.topk(scores+bias,topk,dim=-1,sorted=True).indices
    cpu_weights=scores.gather(-1,cpu_indices);cpu_weights=cpu_weights/(cpu_weights.sum(-1,keepdim=True)+1e-20)*scale
    torch.testing.assert_close(indices.cpu(),cpu_indices,rtol=0,atol=0);torch.testing.assert_close(weights.cpu(),cpu_weights,rtol=3e-5,atol=3e-6)
    gpu_cache={};cpu_cache={}
    callback=lambda i,t,w:expert(reader,i,t,w,args.layer,t.device,clamp,gpu_cache)
    output=dispatch_combine(x.npu(),indices,weights,num_experts,callback).cpu()
    golden=torch.zeros_like(x,dtype=torch.float32)
    for position,i in enumerate(cpu_indices[0].tolist()):
        golden+=expert(reader,i,x,cpu_weights[:,position],args.layer,torch.device('cpu'),clamp,cpu_cache,cpu=True).float()
        cpu_cache.clear()
    golden=golden.bfloat16()
    torch.testing.assert_close(output,golden,rtol=.02,atol=2e-4)
    result={'status':'pass','rank':rank,'world_size':world,'layer':args.layer,'input_shape':list(x.shape),'routing_topk':topk,'selected_experts':cpu_indices[0].tolist(),'local_experts_executed':sorted(gpu_cache),'max_abs_error':float((output.float()-golden.float()).abs().max()),'golden_output_rms':float(golden.float().square().mean().sqrt()),'output_finite':bool(output.isfinite().all()),'boundary':'Real gate and selected FP4 routed experts on two ranks, HCCL dispatch/combine, BF16 intermediate activations. No shared expert, mHC, full decoder layer, FP8 activation equivalence or full-model inference.'}
    args.output_dir.mkdir(parents=True,exist_ok=True);(args.output_dir/f'moe-rank{rank}.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
    dist.destroy_process_group()

if __name__=='__main__':main()
