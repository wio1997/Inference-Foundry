#!/usr/bin/env python3
"""Synthetic parser gates derived from admitted Run403 routes, not live evidence."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import loop080_postpark_validate_run576 as v

ROOT=Path('/data/wio/Inference_Foundry')
SOURCE=ROOT/'evidence/20260927_loop078_bound/run403/capture'


def meta(rows):
    return dict(shape=[rows,7168],stride=[7168,1],dtype='torch.bfloat16',
                device='npu:0',data_ptr=123456,storage_ptr=123456,
                storage_offset=0,element_size=2,storage_nbytes=rows*7168*2)


def weight():
    x=meta(32)
    x['npu_format']=29
    return x


def fixture():
    rows=[]
    for rank in range(8):
        old=json.loads((SOURCE/f'rank{rank}_cohort1.json').read_text())
        rec=copy.deepcopy(old['records']['64'])
        rec['target_query_start_loc']=list(range(0,97,8))
        rec['target_logits_indices']=list(range(96))
        rec['target_seq_lens']=[32768]*12
        rec['active_mask'][4]=False
        rec['replay_meta']=[dict(key=k,capture_serial=i+1,graph_object_id=1000+rank*10+i)
                            for i,k in enumerate(rec['replayed_graphs'])]
        prefix=v.local_prefix(rank)
        rec['attention']=[dict(ordinal=i,graph_key=rec['replayed_graphs'][0],
                               capture_serial=1,
                               q=meta(12),q_prefix=prefix,key_lengths=[32768]*12,
                               ratio=1 if i<2 else 128 if i<22 else 4,
                               full_gather=False) for i in range(43)]
        for model in ('target','draft'):
            for layer in rec[model]:
                n=len(layer['ids'])
                layer['hidden']=meta(n)
                layer['router_logits']=meta(n)
                layer['router_logits']['shape']=[n,256]
                layer['quant_type']='W4A8'
                layer['mega_moe']=False
                layer['capture_serial']=1 if model=='target' else None
                layer['weights']={name:[weight()] for name in
                                  ('w1','w2','w1_scale','w2_scale')}
                layer['weights']['w1_scale_bias']=None
                layer['weights']['w2_scale_bias']=None
        old['schema']=2
        old['cohort']=5
        old['run_tag']='LOOP080-RUN576-SYNTHETIC'
        old['request_ids']=[f'req{i}' for i in range(12)]
        old['selected_cycle']=64
        old['park']=dict(completed_cycle=63,next_target_cycle=64,
                         newly_parked_slots=[4],parked=[i==4 for i in range(12)],
                         progress=[1000]*12)
        old['counts'][64][4]=0
        old['records']={'64':rec}
        rows.append(old)
    return rows


def reject(rows,label):
    try:v.validate(rows)
    except (KeyError,TypeError,ValueError,AssertionError):return label
    raise AssertionError(f'validator accepted mutation: {label}')


def main():
    base=fixture()
    result=v.validate(copy.deepcopy(base))
    assert result['status']=='scoped_diagnostic_accepted' and result['cycle']==64
    mutations=[]
    def mutate(label,fn):
        rows=copy.deepcopy(base);fn(rows);mutations.append(reject(rows,label))
    mutate('wrong_rank',lambda x:x[7].update(rank=6))
    mutate('wrong_request',lambda x:x[7]['request_ids'].__setitem__(0,'wrong'))
    mutate('stale_run',lambda x:x[7].update(run_tag='LOOP080-RUN576-OTHER'))
    mutate('wrong_absolute_cycle',lambda x:x[7]['records']['64'].update(absolute_cycle=999))
    mutate('wrong_park',lambda x:x[7]['park'].update(newly_parked_slots=[-1]))
    mutate('duplicate_target_layer',lambda x:x[7]['records']['64']['target'][1].update(ordinal=0))
    mutate('wrong_graph_generation',lambda x:x[7]['records']['64']['replay_meta'][0].update(capture_serial=0))
    mutate('stale_route_generation',lambda x:x[7]['records']['64']['target'][0].update(capture_serial=2))
    mutate('wrong_dsa_prefix',lambda x:x[7]['records']['64']['attention'][0]['q_prefix'].__setitem__(1,99))
    mutate('wrong_draft_route',lambda x:x[7]['records']['64']['draft'][0]['ids'][0].__setitem__(0,255))
    mutate('wrong_expert_owner',lambda x:x[7]['records']['64']['target'][0]['group'].__setitem__(0,999))
    mutate('missing_weight_format',lambda x:x[7]['records']['64']['target'][0]['weights']['w1'][0].pop('npu_format'))
    report=dict(status='synthetic_validator_gate_pass',source='Run403 real route matrices plus synthetic metadata/park',
                positive=1,negative_mutations=mutations,not_live_evidence=True)
    out=ROOT/'evidence/20260928_loop080_bound/run576/validator_gate.json'
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],negatives=len(mutations))))


if __name__=='__main__':main()
