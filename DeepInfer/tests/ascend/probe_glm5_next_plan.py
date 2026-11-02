# SPDX-License-Identifier: Apache-2.0
"""Actual-config backbone metadata assembly with allocation-free fake factories.

No torch import, model layers/weights/tensors, operators, generation or devices.
Factory outputs are metadata objects, never a runnable45-layer model.
"""
import argparse
import ast
from dataclasses import replace
import datetime
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
from types import SimpleNamespace

from deepinfer.hugging_face.glm5_next_config import Glm5NextConfig
from deepinfer.hugging_face.glm5_next_plan import GlmBackbonePlan,BackboneOptions

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--prior-evidence',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
assert 'torch' not in sys.modules and not list(Path('/dev').glob('davinci[0-9]*'))
src=Path(__file__).resolve().parents[2];config=Glm5NextConfig.from_checkpoint(a.checkpoint)
old=json.loads(a.prior_evidence.read_text());assert old['status']=='passed' and old['NPU_execution'] is False
assert old['config_sha256']==hashlib.sha256((a.checkpoint/'config.json').read_bytes()).hexdigest()
assert old['source_sha256']['deepinfer/model_executor/models/glm5_next/hybrid_decoder.py']==hashlib.sha256((src/'deepinfer/model_executor/models/glm5_next/hybrid_decoder.py').read_bytes()).hexdigest()
plan=GlmBackbonePlan.from_config(config);calls=[]
report={'status':'in_progress','native_round':23,'NPU_execution':False,'torch_imported':False,'tensor_allocation':False,'model_layer_allocation':False,
        'factory_output':'metadata-only SimpleNamespace objects','weight_payload_bytes_read':0,'model_generation':False,
        'architecture':platform.machine(),'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'config_sha256':old['config_sha256'],'reused_evidence_sha256':hashlib.sha256(a.prior_evidence.read_bytes()).hexdigest(),
        'plan':plan.to_dict(),'checks':{},'logical_state_bytes':{},'source_sha256':{n:hashlib.sha256((src/n).read_bytes()).hexdigest() for n in [
            'deepinfer/hugging_face/glm5_next_plan.py','tests/ascend/test_glm5_next_plan.py','tests/ascend/probe_glm5_next_plan.py']}}


def check(name,condition):
    assert condition,name
    report['checks'][name]={'passed':True}


def attention(spec):
    calls.append(('attention',spec.layer_id,spec.attention,spec.checkpoint_prefix))
    return SimpleNamespace(kind=spec.attention,layer_id=spec.layer_id)


def ffn(spec):
    calls.append(('FFN',spec.layer_id,spec.mlp,spec.checkpoint_prefix))
    return SimpleNamespace(kind=spec.mlp,layer_id=spec.layer_id)


def decoder(spec,attn,mlp):
    calls.append(('decoder',spec.layer_id,spec.attention,spec.checkpoint_prefix))
    hc=lambda:SimpleNamespace(hidden_size=config.hidden_size,streams=config.hf_config['text_config']['hc_mult'])
    return SimpleNamespace(layer_id=spec.layer_id,attention_kind=attn.kind,mlp_kind=mlp.kind,attention=attn,mlp=mlp,attn_hc=hc(),ffn_hc=hc())


bound={'attention_factories':{'linear_attention':attention,'deepseek_sparse_attention':attention},'ffn_factories':{'dense':ffn,'sparse':ffn},'decoder_factory':decoder}
try:
    check('actual45_layer_counts',plan.to_dict()['counts']=={'linear_attention':34,'deepseek_sparse_attention':11,'dense':3,'sparse':42})
    check('DSA_layers3_7_to43',[s.layer_id for s in plan.layers if s.attention=='deepseek_sparse_attention']==list(range(3,44,4)))
    check('vision_MTP_present_but_not_enabled',plan.vision_config_present and plan.declared_mtp_layers==1 and not plan.options.vision and plan.options.mtp_tokens==0)
    layers=plan.assemble(config,**bound)
    check('all45_metadata_layers_bound',list(layers)==list(range(45)))
    check('135_factory_calls_order',[(s,i) for s,i,*rest in calls]==[(stage,i) for i in range(45) for stage in ('attention','FFN','decoder')])
    check('all_checkpoint_prefixes',[prefix for stage,i,kind,prefix in calls]==[f'model.language_model.layers.{i}' for i in range(45) for _ in range(3)])
    check('all_component_identities_preserved',all(layers[i].attention.layer_id==i and layers[i].mlp.layer_id==i for i in layers))
    try:layers[0]=None
    except TypeError:check('assembled_map_immutable',True)
    else:raise AssertionError('assembled layer map is writable')
    # Known missing bindings must prevent ANY factory side effect.
    calls.clear();bad={**bound,'attention_factories':{'linear_attention':attention}}
    try:plan.assemble(config,**bad)
    except ValueError:check('missing_DSA_rejected_before_callbacks',calls==[])
    else:raise AssertionError('missing DSA binding accepted')
    changed=replace(config,hidden_size=config.hidden_size+1)
    try:plan.assemble(changed,**bound)
    except ValueError:check('stale_config_rejected_before_callbacks',calls==[])
    else:raise AssertionError('stale config accepted')
    for label,kwargs in [('TP2',{'tensor_parallel_size':2}),('EP2',{'expert_parallel_size':2}),('requests2',{'requests_per_step':2}),('vision',{'vision':True}),('MTP',{'mtp_tokens':1}),('graphs',{'graphs':True}),('prefix',{'prefix_cache':True}),('block16',{'logical_block_size':16})]:
        try:GlmBackbonePlan.from_config(config,BackboneOptions(**kwargs))
        except ValueError:check(f'unsupported_{label}_rejected',calls==[])
        else:raise AssertionError('unsupported option accepted: '+label)
    for tokens in (0,1,7,1024):
        for lookahead in (0,3):
            actual=plan.logical_state_bytes(tokens,lookahead=lookahead);total=0
            for layer in config.layers:
                if layer.attention=='linear_attention':
                    linear=config.linear_attn_config;h=linear['num_heads'];d=linear['head_dim'];w=linear['short_conv_kernel_size']
                    total+=3*math.prod((w-1,h*d))*2+math.prod((h,d,d))*4
                else:
                    total+=math.prod((tokens,512))*2+math.prod((tokens//4,128))*2+math.prod((2,4+lookahead,128))*4
            tag=f'tokens{tokens}_look{lookahead}';check(tag+'_independent_logical_bytes',actual['logical_total']==total)
            report['logical_state_bytes'][tag]=actual
    registry=src/'deepinfer/model_executor/models/__init__.py';tree=ast.parse(registry.read_bytes())
    mapping=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_MODELS' for t in n.targets))
    check('GLM_not_falsely_registered_ready','Glm5NextForConditionalGeneration' not in mapping)
    report['registry']={'sha256':hashlib.sha256(registry.read_bytes()).hexdigest(),'supported_architectures':list(mapping),'modified':False}
    versions={}
    for package in ('torch','torch-npu','transformers','flash-linear-attention-npu-a3'):
        try:versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:versions[package]=None
    report['development_package_metadata']=versions
    check('no_torch_or_native_import','torch' not in sys.modules and 'torch_npu' not in sys.modules)
    report['status']='passed'
except Exception as exc:
    report['status']='failed';report['failure']=repr(exc);raise
finally:
    report['ended_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (a.output_dir/'backbone-plan-result.json').write_text(json.dumps(report,indent=2)+'\n')
