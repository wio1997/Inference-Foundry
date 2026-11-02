# SPDX-License-Identifier: Apache-2.0
"""Prepared future NPU probe; NOT authorized by merely having this script.

The caller must first obtain current user authorization and exclusive resource
ownership outside this program. This program never starts/stops a container,
service or occupying job. No full-model generation or performance claim.
"""
import argparse
import datetime
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint',type=Path,required=True)
parser.add_argument('--output-dir',type=Path,required=True)
parser.add_argument('--authorized-npu',action='store_true',help='Require later user authorization and pre-established resource ownership')
options=parser.parse_args()
if not options.authorized_npu:
    parser.error('NPU execution is disabled; later explicit user authorization is required')
# Keep every torch/NPU import after the execution gate.
import torch
import torch_npu
from deepinfer.hugging_face.modelslim910c import ModelSlimCheckpoint
from deepinfer.model_executor.models.glm5_next.decoder import Glm5NextDenseDecoder
from deepinfer.model_executor.models.glm5_next.mhc import rms_norm

versions={name:metadata.version(name) for name in ['torch','torch-npu','flash-linear-attention-npu-a3']}
if not versions['torch'].startswith('2.10.0') or versions['torch-npu']!='2.10.0.post7' or versions['flash-linear-attention-npu-a3']!='26.9.1':
    raise RuntimeError('This prepared probe requires the previously exercised 2.10/26.9.1 dependency tuple')
torch.set_num_threads(2);torch.manual_seed(5302)
torch_npu.npu.set_device(0);torch_npu.npu.set_compile_mode(jit_compile=False)
options.output_dir.mkdir(parents=True,exist_ok=True)
reader=ModelSlimCheckpoint(options.checkpoint)
source_root=Path(__file__).resolve().parents[2]
report={'status':'in_progress','NPU_execution':True,'versions':versions,'checks':{},
        'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'workload':'one real dense KDA decoder layer, 5 BF16 synthetic hidden tokens; no generation/benchmark',
        'source_sha256':{name:hashlib.sha256((source_root/name).read_bytes()).hexdigest() for name in [
            'deepinfer/model_executor/models/glm5_next/kda.py','deepinfer/model_executor/models/glm5_next/mhc.py',
            'deepinfer/model_executor/models/glm5_next/decoder.py','tests/ascend/probe_glm5_next_dense_npu.py']}}


def save():
    p=options.output_dir/'dense-npu-result.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(p)


def compare(name,actual,expected,limit=.001):
    a=actual.detach().float().cpu();e=expected.detach().float().cpu()
    assert a.shape==e.shape and bool(torch.isfinite(a).all())
    diff=a-e;rel=float(diff.square().mean().sqrt()/e.square().mean().sqrt().clamp_min(1e-12))
    maximum=float(diff.abs().max()) if diff.numel() else 0.
    bound=.03*float(e.abs().max())+1e-5
    report['checks'][name]={'shape':list(a.shape),'relative_rmse':rel,'relative_rmse_limit':limit,
                           'max_abs':maximum,'max_abs_limit':bound,'passed':rel<=limit and maximum<=bound}
    save();assert report['checks'][name]['passed'],(name,report['checks'][name])


save()
try:
    layer=Glm5NextDenseDecoder.from_checkpoint(reader,0,'npu',kda_backend='ascendc')
    assert all(getattr(layer.mlp,name).weight.dtype==torch.int8 for name in ['gate_proj','up_proj','down_proj'])
    x=torch.randn(5,4096).bfloat16().npu()
    with torch.inference_mode():
        complete,complete_state=layer(x)
        first,first_state=layer(x[:4]);before=first_state.recurrent.clone()
        last,last_state=layer(x[4:],first_state)
        compare('decoder_segmented_output',torch.cat((first,last)),complete)
        compare('decoder_segmented_recurrent_state',last_state.recurrent,complete_state.recurrent)
        compare('caller_recurrent_state_unchanged',first_state.recurrent,before,limit=0.)
        for i,(full,segmented) in enumerate(zip(complete_state.conv,last_state.conv,strict=True)):
            compare(f'decoder_segmented_conv_{i}',segmented,full,limit=0.)
        fresh,_=layer(x);compare('fresh_request_isolation',fresh,complete,limit=0.)
        assert (complete_state.tokens_seen,first_state.tokens_seen,last_state.tokens_seen)==(5,4,5)
        # Reconstruct the FFN input from explicit mHC seams.
        residual=layer.attn_hc.expand(x);attn_input,mix=layer.attn_hc.pre(residual)
        attention,_=layer.attention(rms_norm(attn_input,layer.input_norm,layer.norm_eps))
        residual=layer.attn_hc.post(attention,residual,mix)
        ffn_input,_=layer.ffn_hc.pre(residual);ffn_input=rms_norm(ffn_input,layer.ffn_norm,layer.norm_eps)
        gate=layer.mlp.gate_proj(ffn_input);up=layer.mlp.up_proj(ffn_input)
        activated=(torch.nn.functional.silu(gate.float().clamp_max(10.))*up.float().clamp(-10.,10.)).bfloat16()
        # Independent CPU integer dot products for selected actual output rows.
        # Quantization codes come from the NPU op: this checks INT8 GEMM/scales,
        # not an independent implementation of the dynamic-quant rounding rule.
        for name,values,actual in [('gate_proj',ffn_input,gate),('up_proj',ffn_input,up),
                                  ('down_proj',activated,layer.mlp.down_proj(activated))]:
            codes,token_scale=torch_npu.npu_dynamic_quant(values,dst_type=torch.int8)
            codes=codes.cpu().to(torch.int32);token_scale=token_scale.float().cpu().reshape(-1,1)
            count=actual.shape[1];rows=[0,1,17,count-1];weights=[];scales=[]
            for row in rows:
                w,s=reader.linear_payload(f'model.language_model.layers.0.mlp.{name}',output_slice=slice(row,row+1))
                assert w.dtype==torch.int8 and s.dtype==torch.bfloat16
                weights.append(w);scales.append(s)
            weight=torch.cat(weights).to(torch.int32);scale=torch.cat(scales).float().reshape(1,-1)
            expected=((codes@weight.T).float()*scale*token_scale).bfloat16()
            compare(name+'_sampled_INT8_cpu_dot',actual[:,rows],expected,limit=.01)
        compare('dense_forward_vs_staged',layer.mlp(ffn_input),layer.mlp.down_proj(activated),limit=0.)
    report['status']='passed'
except BaseException as exc:
    report['status']='failed';report['failure']=repr(exc);raise
finally:
    report['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();save();print(json.dumps(report,indent=2))
