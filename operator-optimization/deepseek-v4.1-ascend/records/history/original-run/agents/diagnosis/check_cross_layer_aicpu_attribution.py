#!/usr/bin/env python3
"""Compare saved host-layer-attributed tasks in device-export time, no hardware access."""
import hashlib, json
from decimal import Decimal
from pathlib import Path
root = Path(__file__).resolve().parents[2]
p = root / 'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl'
expected = '0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4'
h = hashlib.sha256(p.read_bytes()).hexdigest()
assert h == expected, (h, expected)
selected = {}
for line in p.open():
    record = json.loads(line)
    if record.get('kind') != 'rank':
        continue
    rank = record['data']
    for layer in rank['layers']:
        if layer['layer'] not in (14, 23):
            continue
        key = (layer['scheduler_digest'], rank['ep'])
        for group in layer['operator_groups']:
            if (layer['layer'], group['operator_name']) not in ((14, 'DispatchFFNCombine'), (23, 'reduce_scatterAicpuKernel')):
                continue
            assert len(group['tasks']) == 1, (key, layer['layer'])
            t = group['tasks'][0]
            c = t['csv']
            assert t['host_pid'] == str(rank['pid']) and t['host_tid'] == str(rank['pid'])
            assert c['Accelerator Core'] == ('MIX_AIC' if layer['layer'] == 14 else 'AI_CPU')
            start = Decimal(t['device_start_us'])
            duration = Decimal(c['Duration(us)'])
            assert duration > 0
            row = {'layer_scope_id':layer['layer_scope_id'], 'host_pid':t['host_pid'], 'host_tid':t['host_tid'],
                   'host_start_us':t['host_start_us'], 'device_start_us':str(start), 'duration_us':str(duration),
                   'stream':c['Stream ID'], 'task':c['Task ID'], 'csv_line':t['csv_line'],
                   'accelerator_core':c['Accelerator Core'], 'start':start, 'end':start+duration}
            d=selected.setdefault(key,{})
            assert layer['layer'] not in d
            d[layer['layer']]=row
results=[]
for (digest,ep), pair in sorted(selected.items()):
    assert set(pair)=={14,23}, (digest,ep,pair.keys())
    a,b=pair[14],pair[23]
    assert a['stream'] != b['stream']
    overlap=max(Decimal(0),min(a['end'],b['end'])-max(a['start'],b['start']))
    def serial(v):
        return {k:w for k,w in v.items() if k not in ('start','end')}
    results.append({'digest':digest,'ep':ep,'layer14_mc2':serial(a),'layer23_reduce_scatter_aicpu':serial(b),
                    'rs_start_minus_mc2_us':str(b['start']-a['start']),'overlap_us':str(overlap),
                    'overlap_fraction_of_mc2':str(overlap/(a['end']-a['start']))})
assert len(results)==32
summaries=[]
for digest in sorted({x['digest'] for x in results}):
    rows=[x for x in results if x['digest']==digest]
    assert len(rows)==8
    summaries.append({'digest':digest,'ep':[r['ep'] for r in rows],
      'full_count':sum(Decimal(r['overlap_fraction_of_mc2'])==1 for r in rows),
      'partial_count':sum(0<Decimal(r['overlap_fraction_of_mc2'])<1 for r in rows),
      'zero_count':sum(Decimal(r['overlap_fraction_of_mc2'])==0 for r in rows)})
result={'input_sha256':h,'scope':'saved host-selected L14 MC2 versus L23 AICPU RS, same rank and digest, exported device timestamps',
        'unit':'us; starts from trace and durations from matched CSV, rounded to CSV precision',
        'summary':summaries,'pairs':results,
        'limitations':['host layer scope labels submission flow, not exclusive device occupancy','same digest or coincident interval does not prove collective identity or tensor dependency','AICPU interval does not prove continuous link transfer, resource contention or causation','not unprofiled benchmark, overall model or layer-exclusive time; hardware clock calibration/completeness unproven']}
print(json.dumps(result,ensure_ascii=False,indent=2))
