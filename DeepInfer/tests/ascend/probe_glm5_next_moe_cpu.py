# SPDX-License-Identifier: Apache-2.0
"""Bounded CPU router preparation with real small router tensors, no expert FFN."""
import argparse
import ast
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from types import SimpleNamespace

import numpy as np
import torch
from deepinfer.hugging_face.modelslim910c import ModelSlimCheckpoint
from deepinfer.model_executor.models.glm5_next.moe import GlmMoERouter,pack_routes,dispatch_hidden,combine_packed

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint',type=Path,required=True)
parser.add_argument('--reference-router',type=Path,required=True)
parser.add_argument('--output-dir',type=Path,required=True)
options=parser.parse_args();options.output_dir.mkdir(parents=True,exist_ok=True)
assert not list(Path('/dev').glob('davinci[0-9]*'))
assert os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0'
torch.set_num_threads(2);torch.manual_seed(5315)
# Extract only the pure CPU function, dropping its torch.compile decorator.
# Do not import vLLM, custom ops, the platform runtime or any device backend.
node=next(n for n in ast.parse(options.reference_router.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='grouped_topk')
node.decorator_list=[]
namespace={'torch':torch,'envs':SimpleNamespace(VLLM_USE_FUSED_MOE_GROUPED_TOPK=False,VLLM_BATCH_INVARIANT=False)}
exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'reference_grouped_topk','exec'),namespace)
reference=namespace['grouped_topk']
reader=ModelSlimCheckpoint(options.checkpoint);src=Path(__file__).resolve().parents[2]
report={'status':'in_progress','native_round':15,'NPU_execution':False,'expert_FFN_executed':False,
        'architecture':platform.machine(),'torch':torch.__version__,
        'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'reference_sha256':hashlib.sha256(options.reference_router.read_bytes()).hexdigest(),
        'config_sha256':hashlib.sha256((options.checkpoint/'config.json').read_bytes()).hexdigest(),
        'source_sha256':{name:hashlib.sha256((src/name).read_bytes()).hexdigest() for name in [
            'deepinfer/model_executor/models/glm5_next/moe.py','tests/ascend/probe_glm5_next_moe_cpu.py']},
        'checks':{},'layers':{}}

def check(name,actual,expected,*,atol=0.,rtol=0.):
    assert actual.shape==expected.shape
    diff=(actual.float()-expected.float()).abs()
    report['checks'][name]={'shape':list(actual.shape),'max_abs':float(diff.max()) if diff.numel() else 0.,'atol':atol,'rtol':rtol}
    torch.testing.assert_close(actual,expected,atol=atol,rtol=rtol)
    report['checks'][name]['passed']=True

try:
    for layer_id in (3,4):
        router=GlmMoERouter.from_checkpoint(reader,layer_id)
        hidden=torch.randn(5,4096).bfloat16();logits=router.logits(hidden);routes=router(hidden)
        ref_weights,ref_ids=reference(hidden,logits,8,True,1,1,'sigmoid',2.5,router.correction_bias)
        prefix=f'layer{layer_id}'
        check(prefix+'_reference_ids',routes.expert_ids,ref_ids)
        check(prefix+'_reference_weights',routes.weights,ref_weights)
        numpy_logits=hidden.float().numpy().astype(np.float64)@router.weight.numpy().astype(np.float64).T
        check(prefix+'_FP64_projection',logits,torch.from_numpy(numpy_logits).float(),atol=2e-5,rtol=2e-5)
        scores=np.exp(-np.logaddexp(0.,-numpy_logits));choices=scores+router.correction_bias.numpy().astype(np.float64)
        order=np.argsort(-choices,axis=-1,kind='stable');selected=order[:,:8]
        min_gap=float(np.min(np.take_along_axis(choices,order[:,7:8],1)-np.take_along_axis(choices,order[:,8:9],1)))
        assert min_gap>1e-6,'FP64/FP32 selection boundary too close; no exact selection claim'
        expected_scores=np.take_along_axis(scores,routes.expert_ids.long().numpy(),axis=1)
        expected_weights=expected_scores/expected_scores.sum(-1,keepdims=True)*2.5
        check(prefix+'_numpy_selected_sets',routes.expert_ids.sort(-1).values,torch.from_numpy(np.sort(selected,axis=-1)).int())
        check(prefix+'_numpy_weights',routes.weights,torch.from_numpy(expected_weights).float(),atol=2e-6,rtol=2e-5)
        check(prefix+'_routed_mass',routes.weights.sum(-1),torch.full((5,),2.5),atol=3e-7,rtol=3e-7)
        plan=pack_routes(routes,num_experts=288);packed=dispatch_hidden(hidden,plan)
        assert plan.group_ends.numel()==288 and int(plan.group_ends[-1])==40
        # Small synthetic expert outputs exercise only route alignment/combination.
        values=packed[:,:7].float()*(plan.expert_ids+1).unsqueeze(-1).float()
        shared=hidden[:,:7].float()*.125
        combined=combine_packed(values,plan,shared_output=shared)
        expected=shared.double().numpy().copy()
        for t in range(5):
            for e,w in zip(routes.expert_ids[t].tolist(),routes.weights[t].tolist()):
                expected[t]+=w*hidden[t,:7].float().numpy().astype(np.float64)*(e+1)
        check(prefix+'_dispatch_combine_shared',combined,torch.from_numpy(expected).float(),atol=1e-4,rtol=2e-6)
        params={}
        for suffix in ('weight','e_score_correction_bias'):
            name=f'model.language_model.layers.{layer_id}.mlp.gate.{suffix}';value,dtype=reader.tensor(name)
            params[suffix]={'shape':list(value.shape),'storage':dtype,'sha256':hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
        contracts={}
        for family in ('shared_experts','experts.0','experts.287'):
            contracts[family]={}
            for proj in ('gate_proj','up_proj','down_proj'):
                name=f'model.language_model.layers.{layer_id}.mlp.{family}.{proj}.weight';d=reader.descriptor(name)[2]
                expected_shape=[4096,2048] if proj=='down_proj' else [2048,4096]
                assert d['shape']==expected_shape and d['dtype']=='I8' and reader.quant[name]=='W8A8_DYNAMIC'
                contracts[family][proj]={'shape':d['shape'],'dtype':d['dtype'],'recipe':reader.quant[name]}
        report['layers'][str(layer_id)]={'router_params':params,'minimum_selection_gap_FP64':min_gap,
            'expert_ids':routes.expert_ids.tolist(),'route_weights':routes.weights.tolist(),'FFN_metadata_only':contracts}
    for layer_id in (0,45):
        try:GlmMoERouter.from_checkpoint(reader,layer_id)
        except ValueError:report['checks'][f'unsupported_layer_{layer_id}']={'passed':True}
        else:raise AssertionError('unsupported backbone layer accepted')
    name='model.language_model.layers.3.mlp.gate.weight';old=reader.quant[name]
    reader.quant[name]='W8A8_DYNAMIC'
    try:
        try:GlmMoERouter.from_checkpoint(reader,3)
        except ValueError:report['checks']['quantized_router_rejected']={'passed':True}
        else:raise AssertionError('non-FLOAT router accepted')
    finally:reader.quant[name]=old
    assert 'torch_npu' not in sys.modules
    report['status']='passed'
except BaseException as exc:
    report['status']='failed';report['failure']=repr(exc);raise
finally:
    report['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    p=options.output_dir/'cpu-moe-result.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(p)
    print(json.dumps(report,indent=2))
