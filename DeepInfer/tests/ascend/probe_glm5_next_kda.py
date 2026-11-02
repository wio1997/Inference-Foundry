# SPDX-License-Identifier: Apache-2.0
"""Bounded 910C correctness probe; explicit opt-in, no serving/benchmark."""
import argparse
import datetime
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import time
from pathlib import Path

import torch
import torch_npu
from deepinfer.hugging_face.modelslim910c import ModelSlimCheckpoint
from deepinfer.model_executor.models.glm5_next.kda import (
    Glm5NextKDA, causal_conv, recurrent_kda, ascendc_recurrent_kda,
)

torch.set_num_threads(2)
torch.manual_seed(53)
torch_npu.npu.set_device(0)
torch_npu.npu.set_compile_mode(jit_compile=False)
torch_npu.npu.config.allow_internal_format=True
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint',type=Path,required=True)
parser.add_argument('--output-dir',type=Path,required=True)
cli_options=parser.parse_args()
root=cli_options.output_dir
root.mkdir(parents=True,exist_ok=True)
source_root=Path(__file__).resolve().parents[2]
report={'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'architecture':platform.machine(),'device':torch_npu.npu.get_device_name(0),
        'visible_devices':os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
        'versions':{p:metadata.version(p) for p in ['torch','torch-npu','flash-linear-attention-npu-a3']},
        'checks':{},'status':'in_progress'}
report['source_sha256']={str(p.relative_to(source_root)):hashlib.sha256(p.read_bytes()).hexdigest()
 for p in [source_root/'deepinfer/model_executor/models/glm5_next/kda.py',Path(__file__)]}

def save():
    p=root/'npu-result.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(p)

def compare(name,actual,expected,*,rrmse_limit,atol):
    a=actual.detach().float().cpu();e=expected.detach().float().cpu()
    assert a.shape==e.shape, (name,a.shape,e.shape)
    assert bool(torch.isfinite(a).all()),name
    diff=a-e;rmse=float(diff.square().mean().sqrt())
    rel=rmse/max(float(e.square().mean().sqrt()),1e-12)
    max_abs=float(diff.abs().max()) if diff.numel() else 0
    report['checks'][name]={'shape':list(a.shape),'max_abs':max_abs,'rmse':rmse,'relative_rmse':rel,
                           'relative_rmse_limit':rrmse_limit,'absolute_limit':atol,
                           'passed':rel<=rrmse_limit and max_abs<=atol}
    save();assert report['checks'][name]['passed'],(name,report['checks'][name])

save()
try:
    # Full GLM head geometry, nonzero initial state, >8 tokens exercises chunks.
    t,h,d=11,64,128
    q,k,v,g=[torch.randn(t,h,d).bfloat16() for _ in range(4)]
    beta=torch.randn(t,h).bfloat16();a=torch.randn(h);bias=torch.randn(h*d)
    initial=torch.randn(h,d,d)*0.01
    expected,expected_state=recurrent_kda(q,k,v,g,beta,initial,a,bias)
    args=[x.npu() for x in (q,k,v,g,beta,initial,a,bias)]
    original=args[5].clone()
    actual,actual_state=ascendc_recurrent_kda(*args)
    torch_npu.npu.synchronize()
    compare('native_recurrent_output_11x64x128',actual,expected,rrmse_limit=.01,atol=.001)
    compare('native_recurrent_state_vk',actual_state,expected_state,rrmse_limit=.01,atol=.003)
    compare('native_input_state_immutable',args[5],original,rrmse_limit=0,atol=0)
    first,mid=ascendc_recurrent_kda(*(x[:3] if i<5 else x for i,x in enumerate(args)))
    tail,final=ascendc_recurrent_kda(*[x[3:] for x in args[:5]],mid,*args[6:])
    compare('native_segmented_output',torch.cat((first,tail)),actual,rrmse_limit=.01,atol=.001)
    compare('native_segmented_state',final,actual_state,rrmse_limit=.01,atol=.003)
    # The installed fused-normalization path diverged ~2% at real GLM
    # projection amplitudes. This test must keep rejecting that path.
    small_args=[x.clone() for x in args]
    small_args[0]=small_args[0]*.001;small_args[1]=small_args[1]*.001
    small_cpu=[x.cpu() for x in small_args]
    small_ref,small_ref_state=recurrent_kda(*small_cpu)
    small_out,small_state=ascendc_recurrent_kda(*small_args)
    # BF16 output may differ by one ULP at a rounding boundary; retain 0.1% RMS guard.
    compare('small_qk_epsilon_regression',small_out,small_ref,rrmse_limit=.001,atol=.0001)
    compare('small_qk_epsilon_state',small_state,small_ref_state,rrmse_limit=.001,atol=.00001)
    # Independent vendor short-convolution comparison at actual 8192 channels.
    from fla_npu.ops.ascendc import causal_conv1d_fn
    x=torch.randn(5,8192).bfloat16();w=torch.randn(8192,4)
    hist=torch.randn(3,8192).bfloat16()
    reference,history=causal_conv(x,w.bfloat16(),hist)
    cache=torch.stack((torch.zeros_like(hist),hist)).npu()
    conv=causal_conv1d_fn(x.npu(),w.T.bfloat16().contiguous().npu(),None,conv_states=cache,
        query_start_loc=torch.tensor([0,5],dtype=torch.int32,device='npu'),
        cache_indices=torch.ones(1,dtype=torch.int32,device='npu'),
        has_initial_state=torch.ones(1,dtype=torch.bool,device='npu'),activation='silu',null_block_id=0)
    compare('vendor_conv_output',conv,reference,rrmse_limit=.01,atol=.0625)
    compare('vendor_conv_history',cache[1],history,rrmse_limit=0,atol=0)
    # Actual layer 0 weights. No weight dequantization or full-model allocation.
    reader=ModelSlimCheckpoint(cli_options.checkpoint)
    report['checkpoint_index_sha256']=hashlib.sha256((reader.root/'quant_model_weights.safetensors.index.json').read_bytes()).hexdigest()
    report['checkpoint_config_sha256']=hashlib.sha256((reader.root/'config.json').read_bytes()).hexdigest()
    # Unsupported layer/backend contracts must reject before loading projections.
    for layer_id,backend in [(3,'torch'),(0,'unknown')]:
        try: Glm5NextKDA.from_checkpoint(reader,layer_id,'cpu',backend=backend)
        except ValueError: pass
        else: raise AssertionError('unsupported KDA contract accepted')
    report['contract_rejections']=['DSA layer 3','unknown backend']
    cpu=Glm5NextKDA.from_checkpoint(reader,0,'cpu',backend='torch')
    report['loaded_layer0_state_dict']={name:{'shape':list(value.shape),'dtype':str(value.dtype),
        'sha256':hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
        for name,value in cpu.state_dict().items()}
    layer=Glm5NextKDA.from_checkpoint(reader,0,'npu',backend='ascendc')
    hidden=torch.randn(5,4096).bfloat16()*0.1
    with torch.inference_mode():
        ref,ref_state=cpu(hidden)
        out,state=layer(hidden.npu())
        torch_npu.npu.synchronize()
        compare('real_layer0_output',out,ref,rrmse_limit=.03,atol=.1)
        layer.backend='torch'
        npu_ref,npu_ref_state=layer(hidden.npu())
        layer.backend='ascendc'
        compare('real_layer0_native_vs_npu_torch',out,npu_ref,rrmse_limit=.001,atol=.0001)
        compare('real_layer0_native_vs_npu_torch_state',state.recurrent,npu_ref_state.recurrent,rrmse_limit=.001,atol=.0001)
        compare('real_layer0_state',state.recurrent,ref_state.recurrent,rrmse_limit=.03,atol=.01)
        one,s1=layer(hidden[:4].npu())
        two,s2=layer(hidden[4:].npu(),s1)
        compare('real_layer0_segmented_output',torch.cat((one,two)),out,rrmse_limit=.03,atol=.1)
        compare('real_layer0_segmented_state',s2.recurrent,state.recurrent,rrmse_limit=.03,atol=.01)
        assert (state.tokens_seen,s1.tokens_seen,s2.tokens_seen)==(5,4,5)
        fresh,_=layer(hidden.npu())
        compare('real_layer0_request_isolation',fresh,out,rrmse_limit=0,atol=0)
    report['status']='passed'
except BaseException as exc:
    report['status']='failed';report['failure']=repr(exc);raise
finally:
    report['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
    print(json.dumps(report,indent=2),flush=True)
