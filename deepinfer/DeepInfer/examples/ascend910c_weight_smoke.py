"""Bounded offline module check; this does not run a language model.

Reads one real attention projection and one expert matrix from an explicitly
provided official checkpoint. No downloads and no ModelSlim format inference.
"""
import argparse
import json
import struct
from pathlib import Path

import torch
import torch_npu

from deepinfer import PLATFORM
from deepinfer.model_executor.layers.linear import ReplicatedLinear
from deepinfer.model_executor.layers.tilelang_ops.norm import norm
from deepinfer.npu_ops.compat910c import decode_block_weight


def read_tensor_bytes(root, mapping, name):
    with (root/mapping[name]).open('rb') as f:
        header_size=struct.unpack('<Q',f.read(8))[0]
        header=json.loads(f.read(header_size))
        desc=header[name];begin,end=desc['data_offsets']
        if end-begin>32*1024*1024:
            raise ValueError('smoke test refuses tensor reads over 32 MiB')
        f.seek(8+header_size+begin)
        value=torch.frombuffer(bytearray(f.read(end-begin)),dtype=torch.uint8).clone().reshape(desc['shape'])
    return value,desc


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model-dir',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.model_dir
    if not PLATFORM.is_910c_reference:raise RuntimeError('explicit 910C reference profile is required')
    config=json.loads((root/'config.json').read_text())
    quant=config.get('quantization_config',{})
    if quant.get('quant_method')!='fp8' or quant.get('expert_dtype')!='fp4':
        raise ValueError('requires official FP8/FP4 checkpoint; other quantization formats need separate loaders')
    index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
    results=[]
    for name,fmt,block in [('layers.0.attn.wq_a','e4m3',(32,32)),('layers.0.ffn.experts.0.w1','e2m1',(1,32))]:
        data,desc=read_tensor_bytes(root,index,name+'.weight')
        scales,sdesc=read_tensor_bytes(root,index,name+'.scale')
        if sdesc['dtype']!='F8_E8M0':raise ValueError('expected unpacked HF UE8M0 scales')
        if fmt=='e4m3':
            if desc['dtype']!='F8_E4M3':raise ValueError('unexpected FP8 storage')
            decoded=data.view(torch.float8_e4m3fn).float()
        else:
            if desc['dtype']!='I8':raise ValueError('unexpected FP4 packed storage')
            lut=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.])
            decoded=torch.stack((lut[(data.int()&15).long()],lut[(data.int()>>4).long()]),-1).flatten(-2)
        sf=torch.pow(2.,scales.float()-127.)
        golden=(decoded*sf.repeat_interleave(block[0],0).repeat_interleave(block[1],1)).bfloat16()
        actual=decode_block_weight(data.npu(),scales.npu(),fmt=fmt,block_shape=block)
        torch.testing.assert_close(actual.cpu(),golden,rtol=0,atol=0)
        n,k=golden.shape;gen=torch.Generator().manual_seed(23)
        x=torch.randn((3,k),generator=gen).bfloat16();w=torch.ones(k,dtype=torch.bfloat16)
        normalized=norm(x.npu(),w.npu())
        fp=x.float();normalized_cpu=(fp*torch.rsqrt(fp.square().mean(-1,keepdim=True)+1e-6)).bfloat16()
        layer=ReplicatedLinear(k,n,use_fp32_output=True,weight=torch.nn.Parameter(actual,requires_grad=False))
        output=layer(normalized).cpu();reference=normalized_cpu.float()@golden.float().T
        torch.testing.assert_close(output,reference,rtol=.006,atol=.01)
        results.append({'tensor':name,'logical_shape':[n,k],'storage_dtype':desc['dtype'],'scale_dtype':sdesc['dtype'],'block':list(block),'decoded_bf16_exact':True,'module_output_finite':bool(output.isfinite().all()),'max_abs_error':float((output-reference).abs().max()),'output_rms':float(reference.square().mean().sqrt())})
        del layer,actual,golden,decoded,data,scales
        torch.npu.empty_cache()
    args.output.write_text(json.dumps({'status':'pass','boundary':'Two real full-size weight matrices plus RMSNorm/linear; no attention/cache/EP/full-model inference','device':torch.npu.get_device_name(0),'results':results},indent=2))
    print(args.output.read_text())


if __name__=='__main__':main()
