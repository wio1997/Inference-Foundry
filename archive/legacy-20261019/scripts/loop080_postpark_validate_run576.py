#!/usr/bin/env python3
"""Fail-closed all8 scope validator for one Run576 diagnostic cohort."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def need(ok, why):
    if not ok:
        raise ValueError(why)


def local_prefix(rank: int) -> list[int]:
    prefix = [0]
    lo, hi = 12*rank, 12*(rank+1)
    for slot in range(12):
        a, b = 8*slot, 8*(slot+1)
        prefix.append(prefix[-1] + max(0, min(hi,b)-max(lo,a)))
    return prefix


def check_tensor_meta(m, name, *, physical=False):
    need(isinstance(m, dict), f'{name}: tensor metadata missing')
    shape = m['shape']
    need(isinstance(shape,list) and all(type(x) is int and x>0 for x in shape),
         f'{name}: invalid shape')
    need(len(m['stride']) == len(shape) and all(type(x) is int for x in m['stride']),
         f'{name}: invalid stride')
    need(m['device'].startswith('npu') and m['data_ptr']>0 and m['storage_ptr']>0,
         f'{name}: storage identity')
    need(m['storage_offset']>=0 and m['element_size']>0 and m['storage_nbytes']>0,
         f'{name}: storage size')
    if physical:
        need(type(m.get('npu_format')) is int, f'{name}: physical format absent')


def validate(rows):
    need(len(rows)==8 and sorted(x['rank'] for x in rows)==list(range(8)),
         'exact TP8 rank set')
    rows.sort(key=lambda x:x['rank'])
    base=rows[0]
    need(all(x['schema']==2 and x['cohort']==5 for x in rows), 'schema/cohort')
    for name in ('cohort','run_tag','request_ids','cycles','start_cycle','selected_cycle','park','remaining',
                 'initial_output_counts','initial_positions','counts'):
        need(all(x[name]==base[name] for x in rows), f'all8 {name} mismatch')
    need(base['run_tag'].startswith('LOOP080-RUN576-') and
         len(base['request_ids'])==12 and len(set(base['request_ids']))==12,
         'run tag/request identities')
    cycle=base['selected_cycle']; park=base['park']
    need(type(cycle) is int and 0<cycle<base['cycles'], 'selected cycle range')
    need(park['next_target_cycle']==cycle and park['completed_cycle']==cycle-1,
         'park/target cycle linkage')
    need(bool(park['newly_parked_slots']) and 0<len(park['newly_parked_slots'])<12,
         'first park partial slots')
    need(len(set(park['newly_parked_slots']))==len(park['newly_parked_slots']) and
         all(type(s) is int and 0<=s<12 for s in park['newly_parked_slots']),
         'parked slot identity')
    need(len(park['progress'])==12 and
         all(type(x) is int and x>=0 for x in park['progress']), 'park progress')
    need(len(park['parked'])==12 and all(type(v) is bool for v in park['parked']),
         'parked mask')
    need(all(park['parked'][s] for s in park['newly_parked_slots']),
         'newly parked slot not parked')
    need(len(base['counts'])==base['cycles'] and all(len(x)==12 for x in base['counts']),
         'full acceptance ledger')
    need(all(type(v) is int and 0<=v<=8 for row in base['counts'] for v in row),
         'acceptance count range')
    need(all(set(x['records'])=={str(cycle)} for x in rows), 'selected record only')
    recs=[x['records'][str(cycle)] for x in rows]
    first=recs[0]
    for name in ('target_positions','target_input_ids','target_query_start_loc',
                 'target_logits_indices','target_seq_lens','active_mask',
                 'absolute_cycle',
                 'raw_acceptance_counts','draft_metadata','draft_output'):
        need(all(r[name]==first[name] for r in recs), f'all8 {name} mismatch')
    need(first['active_mask']==[not v for v in park['parked']],
         'parked/active mask linkage')
    need(first['absolute_cycle']==base['start_cycle']+cycle,
         'absolute cycle mismatch')
    need(len(first['target_input_ids'])==96 and len(first['target_positions'])==96,
         'Target input geometry')
    need(first['target_query_start_loc']==list(range(0,97,8)),
         'Target query prefix')
    need(first['target_logits_indices']==list(range(96)),
         'Target logits selector')
    for slot in range(12):
        p=first['target_positions'][8*slot:8*(slot+1)]
        need(p==list(range(p[0],p[0]+8)), 'Target position stride')
    need(base['counts'][cycle]==[raw if active else 0 for raw,active in zip(
         first['raw_acceptance_counts'],first['active_mask'])],
         'raw/masked acceptance linkage')
    q=first['draft_metadata']['q']
    need(q in (7,8) and len(first['draft_metadata']['input_ids'])==12*q,
         'Draft query geometry')
    need(first['draft_metadata']['query_start_loc']==[q*i for i in range(13)],
         'Draft query prefix')
    need(len(first['draft_output'])==12 and all(len(x)==7 for x in first['draft_output']),
         'Draft output width')

    summary=dict(status='scoped_diagnostic_accepted', cohort=5, cycle=cycle,
                 total_cycles=base['cycles'],
                 run_tag=base['run_tag'],request_ids=base['request_ids'],
                 parked_slots=[i for i,v in enumerate(park['parked']) if v],
                 active_slots=[i for i,v in enumerate(first['active_mask']) if v],
                 target_layers=43,draft_layers=3,attention_layers=43,
                 physical_target_rows_per_rank=96, local_query_rows_per_rank=12,
                 conditional_work_not_compulsory=True, timing_eligible=False,
                 formal_W0_transfer=False,numeric_bound_update=False,
                 target_distinct_experts_per_layer=[])
    for rank,r in enumerate(recs):
        need(bool(r['replayed_graphs']) and len(r['replayed_graphs'])==len(set(
             json.dumps(x) for x in r['replayed_graphs'])), 'Graph replay identity')
        need(len(r['replay_meta'])==len(r['replayed_graphs']) and
             all(m['key'] in r['replayed_graphs'] and
                 type(m['capture_serial']) is int and m['capture_serial']>0 and
                 type(m['graph_object_id']) is int and m['graph_object_id']>0
                 for m in r['replay_meta']), 'Graph capture/replay generation')
        graph_serial={json.dumps(m['key']):m['capture_serial'] for m in r['replay_meta']}
        need(len(graph_serial)==len(r['replay_meta']), 'duplicate Graph generation key')
        target=sorted(r['target'],key=lambda x:x['ordinal'])
        draft=sorted(r['draft'],key=lambda x:x['ordinal'])
        attn=sorted(r['attention'],key=lambda x:x['ordinal'])
        need([x['ordinal'] for x in target]==list(range(43)), 'Target layer census')
        need([x['ordinal'] for x in draft]==[43,44,45], 'Draft layer census')
        need([x['ordinal'] for x in attn]==list(range(43)), 'DSA layer census')
        for a in attn:
            need(a['graph_key'] in r['replayed_graphs'], 'DSA selected Graph join')
            need(a['capture_serial']==graph_serial[json.dumps(a['graph_key'])],
                 'DSA capture generation join')
            check_tensor_meta(a['q'], 'DSA query')
            need(a['q']['shape'][0]==12, 'DSA local q rows')
            need(a['q_prefix']==local_prefix(rank), 'DSA captured-reference prefix snapshot')
            need(isinstance(a['key_lengths'],list) and len(a['key_lengths'])==12 and
                 all(type(v) is int and v>=0 for v in a['key_lengths']),
                 'DSA key-length reference snapshot')
            need(a['full_gather'] is False, 'DSA frozen SpecDecoding branch')
            need(a['ratio'] in (0,1,4,128), 'DSA unexpected ratio')
        for model,items,expected_rows in (('target',target,96),('draft',draft,12*q)):
            for item in items:
                need(item['model']==model, 'model identity')
                need(item['graph_key'] in r['replayed_graphs'] if model=='target'
                     else item['graph_key'] is None, 'route selected Graph join')
                need(item['capture_serial']==graph_serial[json.dumps(item['graph_key'])]
                     if model=='target' else item['capture_serial'] is None,
                     'route capture generation join')
                need(len(item['ids'])==expected_rows and all(len(x)==6 for x in item['ids']),
                     'route row/width')
                need(all(len(set(row))==6 and all(type(e) is int and 0<=e<256
                         for e in row) for row in item['ids']),
                     'route ID range/uniqueness')
                need(item['ep_rank']==rank, 'EP relative-rank ownership')
                emap=[e-32*rank if 32*rank<=e<32*(rank+1) else -1 for e in range(256)]
                need(item['expert_map']==emap, 'EP map')
                need(item['quant_type'] and not item['mega_moe'], 'quant/mega branch')
                check_tensor_meta(item['hidden'], 'MoE hidden')
                check_tensor_meta(item['router_logits'], 'router logits')
                need(item['hidden']['shape'][0]==expected_rows and
                     item['router_logits']['shape'][0]==expected_rows,
                     'MoE logical row count')
                for name,weights in item['weights'].items():
                    if weights is None:
                        continue
                    need(bool(weights), f'{name}: empty list')
                    for w in weights: check_tensor_meta(w,name,physical=True)
                need(item['weights']['w1'] is not None and
                     item['weights']['w2'] is not None and
                     item['weights']['w1_scale'] is not None and
                     item['weights']['w2_scale'] is not None,
                     'missing W4A8 operand')
                ref=sorted(recs[0][model],key=lambda x:x['ordinal'])[item['ordinal']-(43 if model=='draft' else 0)]
                need(item['ids']==ref['ids'], f'all8 {model} pre-dispatch route mismatch')
                hist=Counter(e for row in item['ids'] for e in row)
                need(item['group']==[hist[e] for e in range(32*rank,32*(rank+1))],
                     f'{model} native dispatcher count-mode group mismatch')
        if rank==0:
            for item in target:
                ids=item['ids']
                need(all(len(set(row))==6 and all(0<=e<256 for e in row) for row in ids),
                     'topk ID range/uniqueness')
                summary['target_distinct_experts_per_layer'].append(
                    len(set(e for row in ids for e in row)))
    summary['limits']=[
        'Same-rank and all-rank route parity do not prove an arbitrary common row permutation absent.',
        'One diagnostic cycle cannot be transferred to Run99 formal W0.',
        'Actual operand footprint is not compulsory HBM traffic or exposed E2E time.',
        'Collector diagnostics may perturb schedule; no TPS claim.',
        'DSA prefix/key-length values are capture-reference snapshots after Target completion, not proven native replay arguments.',
    ]
    return summary


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('capture_dir',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    paths=sorted(a.capture_dir.glob('rank*_cohort5.json'))
    need(len(paths)==8,'exactly eight capture files')
    rows=[json.loads(p.read_text()) for p in paths]
    result=validate(rows)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','cohort','cycle')}))


if __name__=='__main__':
    main()
