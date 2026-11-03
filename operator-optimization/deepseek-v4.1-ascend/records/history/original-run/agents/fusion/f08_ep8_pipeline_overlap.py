#!/usr/bin/env python3
"""Offline EP8 L14 recorded-task interval/shape check; not tensor binding."""
import hashlib, json
from decimal import Decimal as D
from pathlib import Path
p = Path(__file__).resolve().parents[2] / 'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl'
selected = None
for line in p.open():
    row = json.loads(line)
    if row.get('kind') != 'rank' or row['data']['ep'] != 8 or row['data']['pid'] != 737:
        continue
    for layer in row['data']['layers']:
        if layer['layer_scope_id'] == '737-20':
            if selected is not None:
                raise ValueError('scope not unique')
            selected = layer
if selected is None:
    raise ValueError('scope missing')
items = {}
for group in selected['operator_groups']:
    for t in group['tasks']:
        c = t['csv']
        k = (int(c['Stream ID']), int(c['Task ID']))
        if k in items:
            raise ValueError(f'duplicate {k}')
        items[k] = t
keys = [(35,13274),(35,13275),(35,13276),(47,44167),(47,44194),(47,44196)]
names = ['norm','rope','swa_scatter','q_b_candidate','indexer','attention']
rows = {}
for name,key in zip(names,keys):
    t = items[key]
    c = t['csv']
    start, dur = D(t['device_start_us']), D(t['trace_duration_us'])
    rows[name] = {'stream': key[0], 'task': key[1], 'flow_id': t['flow_id'],
                  'name': c['Name'], 'start_us': str(start), 'duration_us': str(dur),
                  'end_us': str(start+dur), 'input_shapes':c['Input Shapes'],
                  'input_dtypes': c['Input Data Types']}
def overlap(a,b):
    a,b=rows[a],rows[b]
    return max(D(0),min(D(a['end_us']),D(b['end_us']))-max(D(a['start_us']),D(b['start_us'])))
assert rows['swa_scatter']['name'].find('ScatterNdUpdateSk') >= 0
assert rows['q_b_candidate']['name'].find('QuantBatchMatmul') >= 0
assert rows['indexer']['name']=='VllmAscendQuantLightningIndexerV2'
assert rows['attention']['name']=='SparseFlashMla'
result = {'scope':'EP8 L14 737-20, selected matched task intervals only',
          'input_sha256':hashlib.sha256(p.read_bytes()).hexdigest(), 'tasks':rows,
          'overlap_with_q_b_candidate_us':{n:str(overlap(n,'q_b_candidate')) for n in ('norm','rope','swa_scatter')},
          'attention_start_minus_scatter_end_us':str(D(rows['attention']['start_us'])-D(rows['swa_scatter']['end_us'])),
          'indexer_start_minus_scatter_end_us':str(D(rows['indexer']['start_us'])-D(rows['swa_scatter']['end_us'])),
          'limits':['Recorded intervals only; same scope and shape do not establish tensor identity or event edge.',
                    'The candidate Q_b task is compatible with source shape/order but not runtime-call bound.',
                    'Overlap does not show resource utilization, speedup, absence of waits, or effects in unprofiled runs.']}
print(json.dumps(result,ensure_ascii=False,indent=2))
