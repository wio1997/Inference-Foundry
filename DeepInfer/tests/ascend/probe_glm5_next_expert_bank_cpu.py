# SPDX-License-Identifier: Apache-2.0
"""CPU compact-bank preparation with reused routes and bounded INT8 row reads.

Actual expert FFNs/native loading are NOT executed. Synthetic affine expert
outputs test the mapping and weighted combine independently of weight compute.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import struct
import sys

import numpy as np
import torch
from torch import nn
from deepinfer.hugging_face.modelslim910c import ModelSlimCheckpoint
from deepinfer.model_executor.models.glm5_next.moe import MoERoutes,pack_routes
from deepinfer.model_executor.models.glm5_next.expert_bank import (
    compact_expert_plan,SelectedExpertBank,selected_expert_manifest,load_selected_expert_bank,
)

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint',type=Path,required=True)
parser.add_argument('--routes-evidence',type=Path,required=True)
parser.add_argument('--output-dir',type=Path,required=True)
options=parser.parse_args();options.output_dir.mkdir(parents=True,exist_ok=True)
assert not list(Path('/dev').glob('davinci[0-9]*'))
assert os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0'
torch.set_num_threads(2);torch.manual_seed(5316)
reader=ModelSlimCheckpoint(options.checkpoint,max_read_bytes=64*1024)
previous=json.loads(options.routes_evidence.read_text());assert previous['status']=='passed' and previous['NPU_execution'] is False
src=Path(__file__).resolve().parents[2]
assert hashlib.sha256((src/'deepinfer/model_executor/models/glm5_next/moe.py').read_bytes()).hexdigest()==previous['source_sha256']['deepinfer/model_executor/models/glm5_next/moe.py']
assert hashlib.sha256((options.checkpoint/'config.json').read_bytes()).hexdigest()==previous['config_sha256']
report={'status':'in_progress','native_round':16,'NPU_execution':False,'expert_FFN_executed':False,
        'architecture':platform.machine(),'torch':torch.__version__,'per_read_bound_bytes':reader.max_read_bytes,
        'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'reused_routes_sha256':hashlib.sha256(options.routes_evidence.read_bytes()).hexdigest(),
        'config_sha256':previous['config_sha256'],'weight_slice_bytes':0,'checks':{},'layers':{},'payload_slices':[],
        'source_sha256':{name:hashlib.sha256((src/name).read_bytes()).hexdigest() for name in [
            'deepinfer/model_executor/models/glm5_next/expert_bank.py','tests/ascend/probe_glm5_next_expert_bank_cpu.py']}}


def check(name,a,b,*,atol=0.,rtol=0.):
    torch.testing.assert_close(a,b,atol=atol,rtol=rtol)
    diff=(a.float()-b.float()).abs()
    report['checks'][name]={'passed':True,'shape':list(a.shape),'max_abs':float(diff.max()) if diff.numel() else 0.,'atol':atol,'rtol':rtol}


def source_bytes(name,size):
    path,start,d=reader.descriptor(name);begin,end=d['data_offsets']
    assert size<=end-begin and size<=reader.max_read_bytes
    with path.open('rb') as handle:handle.seek(start+begin);result=handle.read(size)
    assert len(result)==size
    return result


class AffineBank(nn.Module):
    def __init__(self,ids):super().__init__();self.ids=ids;self.count=len(ids);self.groups=None
    def forward_grouped(self,x,groups,*,group_list_type):
        assert group_list_type==0 and len(groups)==self.count
        self.groups=groups.clone();result=torch.empty_like(x);start=0
        for expert,end in zip(self.ids,groups.tolist(),strict=True):
            result[start:end]=x[start:end]*(expert+1)+.125;start=end
        assert start==len(x)
        return result


class Shared(nn.Module):
    def __init__(self):super().__init__();self.calls=0
    def forward(self,x):self.calls+=1;return x*.125


try:
    for layer_id in (3,4):
        old=previous['layers'][str(layer_id)]
        routes=MoERoutes(torch.tensor(old['expert_ids'],dtype=torch.int32),torch.tensor(old['route_weights'],dtype=torch.float32))
        packed=pack_routes(routes,num_experts=288);active=compact_expert_plan(packed)
        ids=sorted(set(active.global_expert_ids.tolist()+[0,287]))
        compact=compact_expert_plan(packed,ids);prefix=f'layer{layer_id}'
        check(prefix+'_global_local_mapping',compact.global_expert_ids[compact.local_expert_ids],packed.expert_ids)
        check(prefix+'_compact_counts',compact.group_ends,torch.bincount(compact.local_expert_ids,minlength=len(ids)).cumsum(0))
        assert int(compact.group_ends[-1])==40
        manifest=selected_expert_manifest(reader,layer_id,ids)
        assert [s.global_id for s in manifest.routed]==ids and manifest.shared.global_id is None
        assert manifest.num_experts==288 and manifest.hidden_size==4096 and manifest.swiglu_limit==10.
        report['checks'][prefix+'_metadata_manifest']={'passed':True,'routed_experts':len(ids),'shared_experts':1}
        # Reused actual router decisions, NEW synthetic F32 hidden rows solely
        # for adapter verification. These rows are not a rerun of the router.
        hidden=torch.randn(5,4096);bank=AffineBank(ids);shared=Shared()
        wrapper=SelectedExpertBank(ids,bank,num_experts=288,shared=shared);actual=wrapper(hidden,routes)
        expected=hidden.numpy().astype(np.float64)*.125
        for t in range(5):
            for expert,weight in zip(routes.expert_ids[t].tolist(),routes.weights[t].tolist()):
                expected[t]+=weight*(hidden[t].numpy().astype(np.float64)*(expert+1)+.125)
        check(prefix+'_serial_numpy_combine',actual,torch.from_numpy(expected).float(),atol=1e-4,rtol=2e-6)
        check(prefix+'_bank_group_ends',bank.groups,compact.group_ends)
        assert shared.calls==1
        sampled=[manifest.routed[ids.index(expert)] for expert in active.global_expert_ids[:2].tolist()]+[manifest.shared]
        for spec in sampled:
            for projection,shape in zip(spec.prefixes,spec.shapes,strict=True):
                weight,scale=reader.linear_payload(projection,output_slice=slice(0,2))
                assert weight.dtype==torch.int8 and scale.dtype==torch.bfloat16
                raw=source_bytes(projection+'.weight',2*shape[1])
                assert weight.contiguous().view(torch.uint8).numpy().tobytes()==raw
                scales=struct.unpack('<2f',source_bytes(projection+'.weight_scale',8))
                check(projection+'_scale_rounding',scale,torch.tensor(scales).bfloat16())
                report['checks'][projection+'_INT8_bytes']={'passed':True,'bytes':len(raw)}
                report['weight_slice_bytes']+=len(raw)
                report['payload_slices'].append({'prefix':projection,'expert_global_id':spec.global_id,'shape':list(weight.shape),
                    'weight_sha256':hashlib.sha256(raw).hexdigest(),'scale_BF16':scale.float().tolist()})
        report['layers'][str(layer_id)]={'bank_global_ids':ids,'local_ids':compact.local_expert_ids.tolist(),
            'group_ends':compact.group_ends.tolist(),'zero_token_experts':int((torch.bincount(compact.local_expert_ids,minlength=len(ids))==0).sum())}
    try:load_selected_expert_bank(reader,3,[0],'cpu')
    except ValueError as exc:
        assert 'no CPU dequantization fallback' in str(exc)
        report['checks']['native_factory_CPU_guard']={'passed':True}
    else:raise AssertionError('native factory did not reject CPU')
    for layer in (0,45):
        try:selected_expert_manifest(reader,layer,[0])
        except ValueError:report['checks'][f'unsupported_layer_{layer}']={'passed':True}
        else:raise AssertionError('unsupported expert layer accepted')
    name='model.language_model.layers.3.mlp.experts.0.gate_proj.weight';old=reader.quant[name];reader.quant[name]='FLOAT'
    try:
        try:selected_expert_manifest(reader,3,[0],include_shared=False)
        except ValueError:report['checks']['wrong_expert_recipe_rejected']={'passed':True}
        else:raise AssertionError('FLOAT expert accepted by INT8 bank manifest')
    finally:reader.quant[name]=old
    assert 'torch_npu' not in sys.modules
    report['status']='passed'
except BaseException as exc:
    report['status']='failed';report['failure']=repr(exc);raise
finally:
    report['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    p=options.output_dir/'cpu-expert-bank-result.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(p)
    print(json.dumps(report,indent=2))
