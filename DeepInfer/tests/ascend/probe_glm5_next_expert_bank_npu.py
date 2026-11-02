# SPDX-License-Identifier: Apache-2.0
"""Prepared future one-token selected INT8 bank probe; not authorized by this file.

Requires later user NPU permission and externally established exclusive resource
ownership. Never controls services, cards or containers; caller supplies timeout.
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
parser.add_argument('--authorized-npu',action='store_true')
options=parser.parse_args()
if not options.authorized_npu:
    parser.error('NPU execution is disabled; later explicit user authorization is required')
# No device/backend imports precede the execution gate.
import torch
import torch_npu
from deepinfer.hugging_face.modelslim910c import ModelSlimCheckpoint
from deepinfer.model_executor.models.glm5_next.moe import GlmMoERouter,MoERoutes
from deepinfer.model_executor.models.glm5_next.expert_bank import load_selected_expert_bank,selected_expert_manifest
from deepinfer.npu_ops.w8a8 import AscendW8A8Expert

versions={name:metadata.version(name) for name in ['torch','torch-npu','flash-linear-attention-npu-a3']}
if not versions['torch'].startswith('2.10.0') or versions['torch-npu']!='2.10.0.post7' or versions['flash-linear-attention-npu-a3']!='26.9.1':
    raise RuntimeError('Prepared probe requires the previously exercised 2.10/26.9.1 dependency tuple')
torch.set_num_threads(2);torch.manual_seed(5316)
torch_npu.npu.set_device(0);torch_npu.npu.set_compile_mode(jit_compile=False)
options.output_dir.mkdir(parents=True,exist_ok=True)
reader=ModelSlimCheckpoint(options.checkpoint);src=Path(__file__).resolve().parents[2]
report={'status':'in_progress','NPU_execution':True,'versions':versions,'checks':{},
        'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'workload':'layer4 MoE, one BF16 hidden tokenx4096, top8 routed experts plus one shared expert; no generation/benchmark',
        'source_sha256':{name:hashlib.sha256((src/name).read_bytes()).hexdigest() for name in [
            'deepinfer/model_executor/models/glm5_next/expert_bank.py','deepinfer/model_executor/models/glm5_next/moe.py',
            'deepinfer/npu_ops/w8a8.py','tests/ascend/probe_glm5_next_expert_bank_npu.py']},
        'config_sha256':hashlib.sha256((options.checkpoint/'config.json').read_bytes()).hexdigest()}


def save():
    p=options.output_dir/'expert-bank-npu-result.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(p)


def compare(name,actual,expected,limit=.01):
    a=actual.detach().float().cpu();e=expected.detach().float().cpu()
    assert a.shape==e.shape and bool(torch.isfinite(a).all())
    relative=float((a-e).square().mean().sqrt()/e.square().mean().sqrt().clamp_min(1e-12))
    absolute=float((a-e).abs().max()) if a.numel() else 0.
    bound=.03*float(e.abs().max())+1e-5
    report['checks'][name]={'relative_rmse':relative,'relative_rmse_limit':limit,'max_abs':absolute,
                           'max_abs_limit':bound,'passed':relative<=limit and absolute<=bound}
    save();assert report['checks'][name]['passed'],(name,report['checks'][name])


def one_expert(spec):
    values=[]
    for prefix in spec.prefixes:
        weight,scale=reader.linear_payload(prefix)
        assert weight.dtype==torch.int8 and scale.dtype==torch.bfloat16
        values.extend((weight.npu(),scale.npu()))
    return AscendW8A8Expert(*values,limit=10.)


save()
try:
    layer=4;hidden=torch.randn(1,4096).bfloat16().npu()*.1
    router=GlmMoERouter.from_checkpoint(reader,layer,'npu');routes=router(hidden)
    ids=sorted(routes.expert_ids.cpu().flatten().tolist());assert len(ids)==8 and len(set(ids))==8
    wrapper=load_selected_expert_bank(reader,layer,ids,'npu',max_experts=8,include_shared=True)
    assert wrapper.bank.first.dtype==torch.int8 and wrapper.bank.second.dtype==torch.int8
    assert wrapper.shared.first.dtype==torch.int8 and wrapper.shared.second.dtype==torch.int8
    report['bank_global_ids']=ids
    with torch.inference_mode():
        actual=wrapper(hidden,routes)
        manifest=selected_expert_manifest(reader,layer,ids)
        serial=torch.zeros_like(hidden,dtype=torch.float32)
        weights={int(e):w for e,w in zip(routes.expert_ids[0].cpu().tolist(),routes.weights[0],strict=True)}
        # Separately read/load/run one native expert at a time. The persistent
        # compact bank has8 routed experts+shared; reference peaks at one extra.
        for spec in manifest.routed:
            expert=one_expert(spec);serial+=expert(hidden).float()*weights[spec.global_id];del expert
        shared=one_expert(manifest.shared);serial+=shared(hidden).float();del shared
        compare('compact_bank_vs_serial_native_experts',actual,serial.bfloat16())
        repeated=wrapper(hidden,routes);compare('repeated_request',repeated,actual,limit=0.)
        empty=MoERoutes(routes.expert_ids[:0],routes.weights[:0])
        assert wrapper(hidden[:0],empty).shape==(0,4096)
        try:load_selected_expert_bank(reader,layer,list(range(9)),'npu',max_experts=8)
        except ValueError as exc:
            assert 'exceeds max_experts' in str(exc);report['checks']['allocation_bound']={'passed':True}
        else:raise AssertionError('allocation bound was ignored')
    torch_npu.npu.synchronize();report['status']='passed'
except BaseException as exc:
    report['status']='failed';report['failure']=repr(exc);raise
finally:
    report['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();save();print(json.dumps(report,indent=2))
