"""Real ModelSlim INT8 linear/expert probes, independently checked on CPU."""
import argparse
import json
from pathlib import Path

import torch

from deepinfer.hugging_face.modelslim910c import ModelSlimCheckpoint


def cpu_quant(x):
    maximum=x.float().abs().amax(-1)
    scale=maximum/127.
    # 910C diagnostics with exact half-maximum BF16 values distinguish the
    # operation order: (x * 127) / max preserves ties; x / (max / 127)
    # can move a tie by one FP32 ULP. This ideal reference chooses nearest-even;
    # different CANN tilings may choose the other nearest integer at ties.
    codes=torch.round(x.float()*127./maximum.clamp_min(1e-30)[:,None]).clamp(-128,127).to(torch.int8)
    return codes,scale


def cpu_linear(x,weight,scale):
    codes,sx=cpu_quant(x)
    # This is integer accumulation, independent of NPU implementation, without
    # ever dequantizing the checkpoint weight matrix into BF16 or FP32.
    accum=codes.int()@weight.int().T
    return (accum.float()*sx[:,None]*scale.float()[None]).bfloat16()


def cpu_expert(x,payload,limit=10.):
    (w1,s1),(w3,s3),(w2,s2)=payload
    combined=cpu_linear(x,torch.cat((w1,w3)),torch.cat((s1,s3)))
    a,b=combined.chunk(2,-1)
    hidden=(torch.nn.functional.silu(a.float().clamp_max(limit))*b.float().clamp(-limit,limit)).bfloat16()
    return cpu_linear(hidden,w2,s2)


def compare(actual,expected,name):
    actual=actual.cpu();error=actual.float()-expected.float()
    relative=float(error.norm()/expected.float().norm().clamp_min(1e-15));maximum=float(error.abs().max())
    torch.testing.assert_close(actual,expected,rtol=.008,atol=5e-4)
    if relative>.006:raise AssertionError(f'{name}: relative L2 {relative}')
    return {'relative_l2':relative,'max_abs_error':maximum}


def main():
    import torch_npu
    from deepinfer.npu_ops.w8a8 import AscendW8A8Expert
    p=argparse.ArgumentParser();p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.npu.set_device(0);torch.set_num_threads(4)
    reader=ModelSlimCheckpoint(args.model_dir);results={};g=torch.Generator().manual_seed(20261026)
    # Quantization is validated independently, including zero, extrema and
    # nonsymmetric inputs. All scale comparisons use the observed CANN output.
    x=torch.randn((7,1280),generator=g).bfloat16();x[0].zero_();x[1,0]=-128.;x[2,0]=127.
    actual_q,actual_s=torch_npu.npu_dynamic_quant(x.npu(),dst_type=torch.int8);q,s=cpu_quant(x)
    torch.testing.assert_close(actual_q.cpu(),q,rtol=0,atol=0)
    torch.testing.assert_close(actual_s.cpu()[1:],s[1:],rtol=2e-6,atol=1e-8)
    results['quantization']={'codes_exact':True,'zero_input_scale':float(actual_s.cpu()[0])}
    for prefix in ['layers.0.attn.wq_b','layers.0.ffn.experts.0.w1','layers.0.ffn.experts.0.w2','layers.0.ffn.shared_experts.w1']:
        weight,scale=reader.linear_payload(prefix)
        x=torch.randn((3,weight.shape[1]),generator=g).bfloat16()
        golden=cpu_linear(x,weight,scale)
        for nz in (False,True):
            op=reader.load_linear(prefix,'npu:0',nz=nz);actual=op(x.npu())
            results[prefix+('/NZ' if nz else '/ND')]=compare(actual,golden,prefix)
            assert op.weight.dtype==torch.int8
            if nz:
                assert torch_npu.get_npu_format(op.weight)==29
                results[prefix+'/NZ']['npu_format']=29
            args.output.with_suffix('.partial.json').write_text(json.dumps(results,indent=2))
        del op
    for expert_id in (0,17):
        payload=[reader.linear_payload(f'layers.0.ffn.experts.{expert_id}.w{i}') for i in (1,3,2)]
        args_gpu=[item.npu() for pair in payload for item in pair]
        op=AscendW8A8Expert(*args_gpu,limit=10.)
        x=torch.randn((4,5120),generator=g).bfloat16()
        actual,stages=op(x.npu(),return_stages=True)
        stages={name:value.cpu() for name,value in stages.items()}
        golden=cpu_expert(x,payload)
        first=cpu_linear(x,torch.cat((payload[0][0],payload[1][0])),torch.cat((payload[0][1],payload[1][1])))
        a,b=first.float().chunk(2,-1)
        activated=(torch.nn.functional.silu(a.clamp_max(10.))*b.clamp(-10.,10.)).bfloat16()
        iq,isf=cpu_quant(x);sq,ssf=cpu_quant(activated)
        detail={'input_code_mismatches':int((stages['input_codes']!=iq).sum()),
            'first_max_abs_error':float((stages['combined'].float()-first.float()).abs().max()),
            'first_relative_l2':float((stages['combined'].float()-first.float()).norm()/first.float().norm()),
            'activation_max_abs_error':float((stages['activated'].float()-activated.float()).abs().max()),
            'second_code_mismatches':int((stages['second_codes']!=sq).sum()),
            'output_relative_l2':float((actual.cpu().float()-golden.float()).norm()/golden.float().norm())}
        args.output.with_name(f'expert-{expert_id}-diagnostic.json').write_text(json.dumps(detail,indent=2))
        torch.save({'x':x,'payload':payload,'actual':actual.cpu(),'stages':stages,'golden':golden,'golden_first':first,'golden_activated':activated},args.output.with_name(f'expert-{expert_id}-diagnostic.pt'))
        results[f'expert-{expert_id}']=compare(actual,golden,f'expert-{expert_id}')
        del op,args_gpu
    result={'status':'pass','checkpoint':args.model_dir.name,'weight_compute_dtype':'int8','operators':['npu_dynamic_quant','npu_quant_matmul','npu_grouped_matmul'],'runtime_weight_scale_dtype':'bfloat16 (checkpoint F32 rounded as installed vLLM model parameter)','results':results,'boundary':'Real complete checkpoint matrices and experts in INT8, ND and NZ layouts; independent integer CPU golden. No complete layer or full engine generation claim.'}
    args.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
