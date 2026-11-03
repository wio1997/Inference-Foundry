#!/usr/bin/env python3
"""Recompute paired F16 projection records from frozen JSONL; no NPU access."""
import collections
import hashlib
import json
import statistics
from decimal import Decimal
from pathlib import Path

root = Path(__file__).resolve().parents[2]
source = root / 'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl'
assert hashlib.sha256(source.read_bytes()).hexdigest() == '0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4'
groups = {0: [], 8: []}
for line in source.open():
    item = json.loads(line)
    if item.get('kind') != 'rank':
        continue
    rank = item['data']
    ep = rank['ep']
    assert rank['status'] == 'completed'
    scopes = {}
    for layer in rank['layers']:
        n = layer['layer']
        candidates = []
        for group in layer['operator_groups']:
            for task in group['tasks']:
                c = task['csv']
                m = 4760 if ep < 8 else 5144
                for rows in (m, 8184):
                    si = c['Input Shapes'].strip('"')
                    so = c['Output Shapes'].strip('"')
                    dtype_ok = c['Input Data Types'] == 'DT_BF16;DT_BF16' and c['Output Data Types'] == 'DT_BF16'
                    first = c['Type'] == 'TransposeBatchMatMul' and si == f'{rows},1,4096;1,4096,1024' and so == f'{rows},1,1024'
                    second = c['Type'] in ('MatMulV2','MatMulV3') and si == f'{rows},1024;5120,1024' and so == f'{rows},5120'
                    if dtype_ok and (first or second):
                        candidates.append((rows, 'first' if first else 'second', task))
        assert len(candidates) == 2, (ep, n, layer['scheduler_digest'], len(candidates))
        rows = candidates[0][0]
        assert all(x[0] == rows for x in candidates)
        first = [x[2] for x in candidates if x[1] == 'first']; second = [x[2] for x in candidates if x[1] == 'second']
        assert len(first) == len(second) == 1
        first, second = first[0], second[0]
        assert first['csv']['Stream ID'] == second['csv']['Stream ID']
        assert first['device_track_pid'] == second['device_track_pid'] and first['device_track_tid'] == second['device_track_tid']
        assert int(second['csv']['Task ID']) == int(first['csv']['Task ID']) + 1
        assert second['csv']['Type'] == 'MatMulV3'
        for task in (first, second):
            c=task['csv']
            assert c['Input Formats'] == 'ND;ND' and c['Output Formats'] == 'ND'
            assert c['Accelerator Core'] == 'AI_CORE' and c['Block Num'] == '24'
        assert Decimal(first['device_start_us']) + Decimal(first['trace_duration_us']) <= Decimal(second['device_start_us'])
        key = (n, rows)
        assert key not in scopes
        a,b = Decimal(first['trace_duration_us']),Decimal(second['trace_duration_us'])
        assert a > 0 and b > 0
        counters = [Decimal(t['csv']['aic_mac_ratio']) for t in (first,second)]
        cycles = [int(t['csv']['aic_total_cycles']) for t in (first,second)]
        assert all(x > 0 for x in counters) and all(x > 0 for x in cycles)
        scopes[key] = {'ep': ep, 'layer': n, 'rows': rows, 'digest': layer['scheduler_digest'], 'first_us': a, 'second_us': b, 'rate_ratio': Decimal('0.8')*b/a,'aic_ratios':counters,'aic_cycles':cycles}
    assert len(scopes) == 80
    for layer in range(40):
        small = 4760 if ep < 8 else 5144
        a,b = scopes[(layer,small)],scopes[(layer,8184)]
        assert a['digest'] != b['digest']
        groups[0 if ep < 8 else 8].append((a,b))

result = {}
by_rows=collections.defaultdict(list)
for group, pairs in groups.items():
    assert len(pairs) == 320
    def med(name, which):
        return float(statistics.median(p[which][name] for p in pairs))
    result[str(group)] = {'pairs': len(pairs), 'small_rows': pairs[0][0]['rows'], 'large_rows': 8184,
      'small_first_second_median_us': [med('first_us',0),med('second_us',0)],
      'large_first_second_median_us': [med('first_us',1),med('second_us',1)],
      'small_large_rate_ratio_medians': [med('rate_ratio',0),med('rate_ratio',1)],
      'rate_ratio_large_greater_equal_less': [sum(b['rate_ratio'] > a['rate_ratio'] for a,b in pairs),sum(b['rate_ratio'] == a['rate_ratio'] for a,b in pairs),sum(b['rate_ratio'] < a['rate_ratio'] for a,b in pairs)]}
    for idx, label in enumerate(('first','second')):
        diffs=[b['aic_ratios'][idx]-a['aic_ratios'][idx] for a,b in pairs]
        intercepts=[]; slopes=[]
        for a,b in pairs:
            slope=(b[label+'_us']-a[label+'_us'])/Decimal(b['rows']-a['rows'])
            slopes.append(slope);intercepts.append(a[label+'_us']-slope*a['rows'])
        result[str(group)][label+'_counter']={'small_large_ratio_medians':[float(statistics.median(a['aic_ratios'][idx] for a,b in pairs)),float(statistics.median(b['aic_ratios'][idx] for a,b in pairs))], 'delta_median':float(statistics.median(diffs)), 'delta_positive_zero_negative':[sum(x>0 for x in diffs),sum(x==0 for x in diffs),sum(x<0 for x in diffs)], 'negative_intercepts':sum(x<0 for x in intercepts), 'negative_slopes':sum(x<0 for x in slopes), 'intercept_median_us':float(statistics.median(intercepts))}
    for pair in pairs:
        for sample in pair:
            c0,c1=sample['aic_cycles'];r0,r1=sample['aic_ratios']
            assert r0<r1
            by_rows[sample['rows']].append(Decimal(c0)*r0/(Decimal(c1)*r1*Decimal('0.8')))
    if group == 0:
        selected = [p for p in pairs if p[0]['ep']==3 and p[0]['layer']==36]
        assert len(selected)==1
        result[str(group)]['ep3_layer36_us'] = [[float(x['first_us']),float(x['second_us'])] for x in selected[0]]
normalized={str(m):{'n':len(v),'min_median_max':[float(min(v)),float(statistics.median(v)),float(max(v))]} for m,v in sorted(by_rows.items())}
assert [len(by_rows[m]) for m in (4760,5144,8184)]==[320,320,640]
print(json.dumps({'source_sha256':'0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4','groups':result,'normalized_cycle_proxy_first_over_second':normalized,'limitations':'Profiled scopes: same rank/layer, distinct digests. Equal effective work, layout, tensor identity and causal effect are not established. Dense logical rate and cycle proxy are not physical utilization.'},indent=2))
