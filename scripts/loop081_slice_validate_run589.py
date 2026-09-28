#!/usr/bin/env python3
"""Strict structural admission for all8 selected FULL96 layer0 capture rows.

Even PASS means capture-time tensor/branch identity only. It never certifies
native last writer, collective completion, service time or fixed-W0 parity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

EXPECTED_DESCRIPTOR=dict(num_tokens=96,num_reqs=12,uniform=True,
                         has_lora=False,num_active_loras=0)
REQUIRED={'attention_before','partial_pre_rs','native_tags','sequence_rs','projected','attention_after'}

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
    if (row.get('cohort')!=5 or row.get('run_tag')!='LOOP081-RUN589-B'
        or not isinstance(row.get('request_ids'),list) or len(row['request_ids'])!=12
        or len(set(row['request_ids']))!=12 or row.get('relative_cycle')!=0
        or row.get('runtime_cycle')!=row.get('cohort_start_cycle')):
        fail('selected measured cohort/request/first-cycle identity')
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
    exact_tensor(layer['partial_pre_rs'],partial,'local matmul to RS input')
    if partial['shape']!=[96,4096] or reduced['shape']!=[12,4096]:
        fail('unexpected fixed TP output shape')
    if partial['dtype']!='torch.bfloat16' or reduced['dtype']!='torch.bfloat16':
        fail('unexpected TP output dtype')
    exact_tensor(reduced,layer['projected'],'RS result to projection return')
    exact_tensor(before['output_destination'],layer['attention_after']['output'],
                 'attention copy destination')
    if before['output_destination']['shape']!=[12,4096]:
        fail('unexpected attention destination shape')
    if (before['output_destination']['dtype']!='torch.bfloat16'
        or before['output_destination']['element_size']!=2):
        fail('unexpected attention destination dtype')
    if set(layer['native_tags'])!={'partial','copy'}:
        fail('native tag source roles missing')
    return dict(rank=rank,capture_serial=row['capture_serial'],
                replay_submission_ordinal=row['replay_submission_ordinal'],
                cohort=row['cohort'],request_ids=row['request_ids'],
                runtime_cycle=row['runtime_cycle'],
                descriptor=row['descriptor'],custom_op=before['custom_op'],
                quant_method=before['quant_method'],
                partial_bytes=96*4096*2,reduced_bytes=12*4096*2)

def graph_task_identity(row,meta,nodes,rank):
    if (meta.get('rank')!=rank or meta.get('cohort')!=5 or
        meta.get('request_ids')!=row['request_ids'] or
        meta.get('capture_serial')!=row['capture_serial'] or
        meta.get('entry_id')!=row['entry_id'] or
        meta.get('graph_object_id')!=row['graph_object_id'] or
        meta.get('completion')!='post cohort torch.npu.synchronize' or
        meta.get('replay_submissions',0)<row['replay_submission_ordinal'] or
        meta.get('cycles',0)<=0 or meta.get('graph_task_count')!=len(nodes)):
        fail('Graph/capture generation and post-cohort completion join')
    if len(nodes)!=5412:
        fail('FULL96 task count differs from prior production Graph')
    ix={}
    for task in nodes:
        args=task['args'];key=(args['Model Id'],args['Stream Id'],args['Task Id'])
        if key in ix:fail('duplicate native task')
        ix[key]=task
    models={k[0] for k in ix}
    if len(models)!=1:fail('selected Graph contains multiple models')
    tagged=[x for x in nodes if x['args'].get('ExtendInfo','').startswith('RUN589|')]
    if len(tagged)!=2:fail('expected exactly two native labels')
    byrole={}
    for task in tagged:
        info=task['args']['ExtendInfo']
        parts=dict(p.split('=',1) for p in info.split('|')[1:])
        if parts.get('r')!=str(rank) or parts.get('g')!=str(row['capture_serial']):
            fail('native tag rank/capture generation')
        role=parts.get('role')
        if role in byrole or role not in ('partial','copy'):
            fail('duplicate/unknown native role')
        byrole[role]=(task,parts)
        if info!=row['layer0']['native_tags'].get(role):
            fail('native task label differs from source capture')
    if set(byrole)!={'partial','copy'}:fail('missing native roles')
    partial=row['layer0']['sequence_rs']['partial']
    reduced=row['layer0']['sequence_rs']['reduced']
    dst=row['layer0']['attention_before']['output_destination']
    ptask,p=byrole['partial'];ctask,c=byrole['copy']
    if (ptask['args']['Task Type']!='KERNEL_AICORE' or 'MatMul' not in ptask['name']
        or p.get('ptr')!=f"0x{partial['data_ptr']:016x}" or p.get('bytes')!='786432'):
        fail('partial tag not on exact local matmul output')
    if (ctask['args']['Task Type']!='MEMCPY_ASYNC' or ctask['name']!='MEMCPY_ASYNC'
        or c.get('src')!=f"0x{reduced['data_ptr']:016x}"
        or c.get('dst')!=f"0x{dst['data_ptr']:016x}"
        or c.get('bytes')!='98304'):
        fail('copy tag not on typed projected-to-attention copy')
    model=next(iter(models));ps=ptask['args']['Stream Id'];cs=ctask['args']['Stream Id']
    pt=ptask['args']['Task Id'];ct=ctask['args']['Task Id']
    if ps!=cs or pt>=ct:fail('producer and copy stream/order')
    def tasks(stream,start,end,kind):
        return [x for (m,s,t),x in ix.items() if m==model and s==stream and
                start<t<end and x['args']['Task Type']==kind]
    def eid(x):return x['name'].split('_')[-1]
    precords=tasks(ps,pt,ct,'EVENT_RECORD')
    candidates=[]
    for precord in precords:
        pe=eid(precord)
        waits=[x for x in nodes if x['args']['Model Id']==model and
               x['args']['Task Type']=='EVENT_WAIT' and eid(x)==pe and
               x['args']['Stream Id']!=ps]
        for pwait in waits:
            rsstream=pwait['args']['Stream Id'];wt=pwait['args']['Task Id']
            rss=[x for (m,s,t),x in ix.items() if m==model and s==rsstream and t>wt and
                 x['name']=='aiv_reduce_scatter_bfloat16_t']
            for rs in rss:
                rt=rs['args']['Task Id']
                records=[x for (m,s,t),x in ix.items() if m==model and s==rsstream and
                         t>rt and x['args']['Task Type']=='EVENT_RECORD']
                for rrecord in records:
                    reid=eid(rrecord)
                    rwaits=[x for x in tasks(ps,precord['args']['Task Id'],ct,'EVENT_WAIT')
                            if eid(x)==reid]
                    if len(rwaits)==1 and rwaits[0]['args']['Task Id']<ct:
                        candidates.append((precord,pwait,rs,rrecord,rwaits[0]))
    if len(candidates)!=1:fail(f'native event/RS chain ambiguity: {len(candidates)}')
    precord,pwait,rs,rrecord,rwait=candidates[0]
    if not rs['args'].get('ExtendInfo','').startswith('group_name_'):
        fail('HCCL original group metadata missing/overwritten')
    hcs=[x for (m,s,t),x in ix.items() if m==model and s==cs and t>ct and
         x['name'].startswith('HcPost_')]
    if not hcs:fail('no HC-post candidate after copy')
    hc=min(hcs,key=lambda x:x['args']['Task Id'])
    return dict(model_id=model,partial_task=[ps,pt],
                producer_event=eid(precord),rs_task=[rs['args']['Stream Id'],rs['args']['Task Id']],
                rs_group=rs['args']['ExtendInfo'],return_event=eid(rrecord),
                copy_task=[cs,ct],hc_post_candidate=[cs,hc['args']['Task Id']],
                native_copy_src_dst_count_from_capture_label=True,
                hc_post_typed_input_proven=False)


def select_rank(path,rank):
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    selected=[x for x in rows if x.get('descriptor_fields')==EXPECTED_DESCRIPTOR]
    if len(selected)!=1:fail(f'rank{rank}: expected one selected FULL96 row, got {len(selected)}')
    row=selected[0]
    result=validate_one(row,rank)
    parent=path.parent
    graph_path=parent/f'rank{rank}_cohort5_acl_graph.json'
    meta_path=parent/f'rank{rank}_cohort5_graph_meta.json'
    meta=json.loads(meta_path.read_text())
    if hashlib.sha256(graph_path.read_bytes()).hexdigest()!=meta['graph_dump_sha256']:
        fail('same-run Graph dump SHA')
    native=graph_task_identity(row,meta,json.loads(graph_path.read_text()),rank)
    return dict(**result,native=native,
                graph_dump_sha256=meta['graph_dump_sha256'])

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--capture-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    rows=[select_rank(a.capture_dir/f'rank{rank}_captures.jsonl',rank) for rank in range(8)]
    result=dict(status='same_run_native_partial_rs_copy_identity_pass',rows=rows,
                unresolved=['compiled layer0 HC-post typed consumer input',
                    'producer ready and collective completion time',
                    'same-W0 fixed trajectory and marker perturbation',
                    'all8 mixed service and finite Resource/Scheduling/Product Bound'])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],rank_count=len(rows))))

if __name__=='__main__':main()
