#!/usr/bin/env python3
"""Strict structural admission for all8 selected FULL96 layer0 capture rows.

Even PASS means capture-time tensor/branch identity only. It never certifies
native last writer, collective completion, service time or fixed-W0 parity.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

EXPECTED_DESCRIPTOR=dict(num_tokens=96,num_reqs=12,uniform=True,
                         has_lora=False,num_active_loras=0)
REQUIRED={'attention_before','sequence_rs','projected','attention_after','hc_consumer'}

def fail(message):
    raise ValueError(message)

def geometry(t,label):
    shape=t['shape']; stride=t['stride']; size=t['element_size']
    if len(shape)!=len(stride) or any(x<=0 for x in shape) or any(s<0 for s in stride):
        fail(f'{label}: unsupported/empty view geometry')
    start=t['storage_offset']*size
    end=start+(1+sum((n-1)*s for n,s in zip(shape,stride)))*size
    if start<0 or end>t['storage_nbytes']:
        fail(f'{label}: view exceeds backing storage')
    if t['data_ptr']!=t['storage_ptr']+start:
        fail(f'{label}: data pointer versus offset mismatch')

def exact_tensor(a,b,label):
    fields=('shape','stride','dtype','device','data_ptr','storage_ptr',
            'storage_offset','storage_nbytes','element_size')
    if any(a.get(k)!=b.get(k) for k in fields):
        fail(f'{label}: tensor descriptor mismatch')
    for t in (a,b):
        geometry(t,label)

def validate_one(row,rank):
    if row.get('rank')!=rank or row.get('replay_submitted') is not True:
        fail('rank/replay submission identity')
    if row.get('replay_completed') is not False:
        fail('capture row falsely claims replay completion')
    if row.get('descriptor_fields')!=EXPECTED_DESCRIPTOR:
        fail('not selected FULL96 descriptor')
    if row.get('capture_serial',0)<=0 or row.get('replay_submission_ordinal',0)<=0:
        fail('missing capture/replay generation')
    if (row.get('wrapper_type')!='vllm_ascend.compilation.acl_graph.ACLGraphWrapper'
        or row.get('replay_owner_type')!=row.get('wrapper_type')
        or not row.get('runnable_type')
        or row.get('runtime_phase')!='target'
        or type(row.get('runtime_cycle')) is not int or row['runtime_cycle']<0):
        fail('missing wrapper owner identity')
    layer=row.get('layer0',{})
    if set(layer)!=REQUIRED:
        fail(f'incomplete layer0 slice: {sorted(REQUIRED-set(layer))}')
    before=layer['attention_before']; rs=layer['sequence_rs']
    if re.search(r'(?:^|\.)layers\.0\.self_attn(?:\.attn)?$',before['layer_name']) is None:
        fail('wrong Target layer0 source prefix')
    if not rs['prefix'].endswith('layers.0.self_attn.wo_b'):
        fail('wrong wo_b RS prefix')
    if before['custom_op']!='vllm_ascend.ops.linear_op.SequenceRowParallelOp':
        fail('loaded custom op is not SequenceRowParallelOp')
    if not before['quant_method']:
        fail('missing loaded quant method')
    if (before['full_weight'] or before['oproj_tp'] or not before['sp']
        or not before['dsa_cp'] or before['input_is_parallel'] is not True):
        fail('wrong DSA/SP/OTP branch')
    if not before['flash_comm'] or before['mmrs'] or before['pad']!=0:
        fail('wrong FlashComm/MMRS/pad branch')
    if not rs['flash_comm'] or rs['mmrs']:
        fail('RS source branch differs from attention context')
    if before['tp_rank']!=rank or before['tp_size']!=8 or before['tp_members']!=list(range(8)):
        fail('TP8 rank membership mismatch')
    partial,reduced=rs['partial'],rs['reduced']
    geometry(partial,'RS partial')
    geometry(reduced,'RS reduced')
    exact_tensor(before['proj_input'],rs['input'],'wo_b input to RS input')
    if partial['shape']!=[96,4096] or reduced['shape']!=[12,4096]:
        fail('unexpected fixed TP output shape')
    if partial['dtype']!='torch.bfloat16' or reduced['dtype']!='torch.bfloat16':
        fail('unexpected TP output dtype')
    exact_tensor(reduced,layer['projected'],'RS result to projection return')
    exact_tensor(before['output_destination'],layer['attention_after']['output'],
                 'attention copy destination')
    exact_tensor(layer['attention_after']['output'],layer['hc_consumer']['hidden'],
                 'attention return to HC-post input')
    if before['output_destination']['shape']!=[12,4096]:
        fail('unexpected attention destination shape')
    if (before['output_destination']['dtype']!='torch.bfloat16'
        or before['output_destination']['element_size']!=2):
        fail('unexpected attention destination dtype')
    return dict(rank=rank,capture_serial=row['capture_serial'],
                replay_submission_ordinal=row['replay_submission_ordinal'],
                descriptor=row['descriptor'],custom_op=before['custom_op'],
                quant_method=before['quant_method'],
                partial_bytes=96*4096*2,reduced_bytes=12*4096*2)

def select_rank(path,rank):
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    selected=[x for x in rows if x.get('descriptor_fields')==EXPECTED_DESCRIPTOR]
    if len(selected)!=1:fail(f'rank{rank}: expected one selected FULL96 row, got {len(selected)}')
    return validate_one(selected[0],rank)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--capture-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    rows=[select_rank(a.capture_dir/f'rank{rank}_captures.jsonl',rank) for rank in range(8)]
    result=dict(status='capture_branch_and_alias_pass',rows=rows,
                unresolved=['native HCCL ordinal and actual last writer',
                    'producer ready and collective completion time',
                    'same-W0 fixed trajectory and marker perturbation',
                    'all8 mixed service and finite Resource/Scheduling/Product Bound'])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],rank_count=len(rows))))

if __name__=='__main__':main()
