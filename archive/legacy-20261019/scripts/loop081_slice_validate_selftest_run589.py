#!/usr/bin/env python3
"""CPU negative controls for selected FULL96 structural admission."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path

ROOT=Path('/data/wio/Inference_Foundry')
path=ROOT/'scripts/loop081_slice_validate_run589.py'
spec=importlib.util.spec_from_file_location('validate589',path)
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
        runtime_phase='target',runtime_cycle=64,cohort_start_cycle=64,
        relative_cycle=0,cohort=5,run_tag='LOOP081-RUN589-B',
        request_ids=[f'req{i}' for i in range(12)],
        entry_id=123,graph_object_id=456,
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
          'partial_pre_rs':copy.deepcopy(partial),
          'native_tags':{
              'partial':f'RUN589|r={rank}|g=1|role=partial|ptr=0x{100000:016x}|bytes=786432',
              'copy':f'RUN589|r={rank}|g=1|role=copy|src=0x{200000:016x}|dst=0x{300000:016x}|bytes=98304'},
          'projected':copy.deepcopy(reduced),
          'attention_after':dict(output=copy.deepcopy(dst)),
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
 'full_weight_branch':lambda x:x['layer0']['attention_before'].__setitem__('full_weight',True),
 'fused_mmrs_branch':lambda x:x['layer0']['sequence_rs'].__setitem__('mmrs',True),
 'wrong_owner':lambda x:x.__setitem__('wrapper_type','fake.PIECEWISE'),
 'wrong_runtime_phase':lambda x:x.__setitem__('runtime_phase','draft'),
 'disconnected_input':lambda x:x['layer0']['sequence_rs']['input'].__setitem__('data_ptr',400002),
 'invalid_partial_storage':lambda x:x['layer0']['sequence_rs']['partial'].__setitem__('storage_nbytes',1),
 'float_destination':lambda x:x['layer0']['attention_before']['output_destination'].__setitem__('dtype','torch.float32'),
 'wrong_cohort':lambda x:x.__setitem__('cohort',4),
 'missing_tag_role':lambda x:x['layer0']['native_tags'].pop('copy'),
 'partial_alias_mismatch':lambda x:x['layer0']['partial_pre_rs'].__setitem__('data_ptr',100002),
}
for name,fn in mutations.items():
    x=copy.deepcopy(base);fn(x)
    try:v.validate_one(x,0)
    except ValueError:pass
    else:raise AssertionError(f'negative {name} admitted')

source=ROOT/'evidence/20260927_loop079_identity/run502/b_candidate/graph_dump/rank0_cohort5_acl_graph.json'
nodes=json.loads(source.read_text())
for task in nodes:
    key=(task['args']['Stream Id'],task['args']['Task Id'])
    if key==(1,40):task['args']['ExtendInfo']=base['layer0']['native_tags']['partial']
    if key==(1,43):task['args']['ExtendInfo']=base['layer0']['native_tags']['copy']
meta=dict(rank=0,cohort=5,request_ids=base['request_ids'],capture_serial=1,
          entry_id=123,graph_object_id=456,completion='post cohort torch.npu.synchronize',
          replay_submissions=2,cycles=300,graph_task_count=len(nodes))
assert v.graph_task_identity(base,meta,nodes,0)['rs_task']==[0,13]
graph_negatives={
    'wrong_graph_rank':lambda r,m,n:m.__setitem__('rank',1),
    'wrong_graph_request':lambda r,m,n:m.__setitem__('request_ids',['wrong']+m['request_ids'][1:]),
    'partial_tag_on_copy':lambda r,m,n:next(x for x in n if x['args']['Stream Id']==1 and x['args']['Task Id']==40)['args'].__setitem__('Task Type','MEMCPY_ASYNC'),
    'copy_tag_wrong_ptr':lambda r,m,n:r['layer0']['native_tags'].__setitem__('copy',r['layer0']['native_tags']['copy'].replace('dst=','wrong=')),
    'event_pair_broken':lambda r,m,n:next(x for x in n if x['args']['Stream Id']==0 and x['args']['Task Id']==12).__setitem__('name','EVENT_WAIT_0'),
    'group_overwritten':lambda r,m,n:next(x for x in n if x['args']['Stream Id']==0 and x['args']['Task Id']==13)['args'].__setitem__('ExtendInfo','lost_group'),
    'false_completion':lambda r,m,n:m.__setitem__('completion','only submitted'),
}
for name,fn in graph_negatives.items():
    r,m,n=copy.deepcopy((base,meta,nodes));fn(r,m,n)
    try:v.graph_task_identity(r,m,n,0)
    except ValueError:pass
    else:raise AssertionError(f'graph negative {name} admitted')
with tempfile.TemporaryDirectory() as tmp:
    d=Path(tmp)
    p=d/'rank0_captures.jsonl';p.write_text(json.dumps(base)+'\n')
    gp=d/'rank0_cohort5_acl_graph.json';gp.write_text(json.dumps(nodes))
    meta['graph_dump_sha256']=hashlib.sha256(gp.read_bytes()).hexdigest()
    (d/'rank0_cohort5_graph_meta.json').write_text(json.dumps(meta))
    assert v.select_rank(p,0)['native']['rs_task']==[0,13]
    p.write_text(p.read_text()*2)
    try:v.select_rank(p,0)
    except ValueError:pass
    else:raise AssertionError('duplicate selected graph admitted')
print(json.dumps(dict(status='pass',negative_count=len(mutations)+len(graph_negatives)+1,
                      positive_same_run_fixture=True)))
