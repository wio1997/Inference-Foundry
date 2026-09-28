#!/usr/bin/env python3
"""CPU negative controls for selected FULL96 structural admission."""
from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
from pathlib import Path

ROOT=Path('/data/wio/Inference_Foundry')
path=ROOT/'scripts/loop081_slice_validate_run582.py'
spec=importlib.util.spec_from_file_location('validate582',path)
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)

def tensor(ptr,shape):
    n=1
    for x in shape:n*=x
    stride=[shape[-1],1]
    return dict(shape=shape,stride=stride,dtype='torch.bfloat16',device='npu:0',
                data_ptr=ptr,storage_ptr=ptr,storage_offset=0,
                storage_nbytes=n*2,element_size=2)

def row(rank):
    partial=tensor(100000,[96,4096]); reduced=tensor(200000,[12,4096])
    dst=tensor(300000,[12,4096])
    return dict(rank=rank,replay_submitted=True,replay_completed=False,
        capture_serial=1,replay_submission_ordinal=1,
        runtime_phase='target',runtime_cycle=64,
        descriptor='BatchDescriptor(num_tokens=96, num_reqs=12, uniform=True, has_lora=False, num_active_loras=0)',
        descriptor_fields=copy.deepcopy(v.EXPECTED_DESCRIPTOR),
        wrapper_type='vllm_ascend.compilation.acl_graph.ACLGraphWrapper',
        replay_owner_type='vllm_ascend.compilation.acl_graph.ACLGraphWrapper',
        runnable_type='vllm_ascend.models.deepseek_v4.DeepseekV4Model',
        layer0={
          'attention_before':dict(layer_name='model.layers.0.self_attn.attn',
             custom_op='vllm_ascend.ops.linear_op.SequenceRowParallelOp',
             quant_method='vllm_ascend.quantization.method_adapters.AscendLinearMethod',
             inner_quant_method=None,full_weight=False,oproj_tp=False,sp=True,dsa_cp=True,
             input_is_parallel=True,
             flash_comm=True,mmrs=False,pad=0,tp_rank=rank,tp_size=8,
             tp_members=list(range(8)),proj_input=tensor(400000,[96,1536]),
             output_destination=copy.deepcopy(dst)),
          'sequence_rs':dict(prefix='model.layers.0.self_attn.wo_b',flash_comm=True,
             mmrs=False,input=tensor(400000,[96,1536]),partial=partial,reduced=reduced),
          'projected':copy.deepcopy(reduced),
          'attention_after':dict(output=copy.deepcopy(dst)),
          'hc_consumer':dict(hidden=copy.deepcopy(dst),residual=tensor(500000,[12,4096]),
                             post=tensor(600000,[12,4096]),comb=tensor(700000,[12,4096])),
        })

base=row(0)
v.validate_one(base,0)
mutations={
 'missing_projected':lambda x:x['layer0'].pop('projected'),
 'wrong_descriptor':lambda x:x['descriptor_fields'].__setitem__('num_tokens',84),
 'wrong_class':lambda x:x['layer0']['attention_before'].__setitem__('custom_op','fake.DSV4OProjRowParallelOp'),
 'wrong_rank':lambda x:x.__setitem__('rank',1),
 'wrong_dtype':lambda x:x['layer0']['sequence_rs']['reduced'].__setitem__('dtype','torch.float16'),
 'same_storage_wrong_offset':lambda x:x['layer0']['projected'].__setitem__('storage_offset',1),
 'false_completion':lambda x:x.__setitem__('replay_completed',True),
 'wrong_shape':lambda x:x['layer0']['sequence_rs']['partial'].__setitem__('shape',[95,4096]),
 'wrong_members':lambda x:x['layer0']['attention_before'].__setitem__('tp_members',[0,1,2,3,4,5,6,9]),
 'wrong_projected_ptr':lambda x:x['layer0']['projected'].__setitem__('data_ptr',200002),
 'wrong_hc_alias':lambda x:x['layer0']['hc_consumer']['hidden'].__setitem__('data_ptr',300002),
 'full_weight_branch':lambda x:x['layer0']['attention_before'].__setitem__('full_weight',True),
 'fused_mmrs_branch':lambda x:x['layer0']['sequence_rs'].__setitem__('mmrs',True),
 'wrong_owner':lambda x:x.__setitem__('wrapper_type','fake.PIECEWISE'),
 'wrong_runtime_phase':lambda x:x.__setitem__('runtime_phase','draft'),
 'disconnected_input':lambda x:x['layer0']['sequence_rs']['input'].__setitem__('data_ptr',400002),
 'invalid_partial_storage':lambda x:x['layer0']['sequence_rs']['partial'].__setitem__('storage_nbytes',1),
 'float_destination':lambda x:x['layer0']['attention_before']['output_destination'].__setitem__('dtype','torch.float32'),
}
for name,fn in mutations.items():
    x=copy.deepcopy(base);fn(x)
    try:v.validate_one(x,0)
    except ValueError:pass
    else:raise AssertionError(f'negative {name} admitted')

with tempfile.TemporaryDirectory() as tmp:
    d=Path(tmp)
    for rank in range(8):
        (d/f'rank{rank}_captures.jsonl').write_text(json.dumps(row(rank))+'\n')
    assert len([v.select_rank(d/f'rank{rank}_captures.jsonl',rank) for rank in range(8)])==8
    p=d/'rank7_captures.jsonl'
    p.write_text(p.read_text()*2)
    try:v.select_rank(p,7)
    except ValueError:pass
    else:raise AssertionError('duplicate selected graph admitted')
print(json.dumps(dict(status='pass',negative_count=len(mutations)+1,positive_all8=True)))
