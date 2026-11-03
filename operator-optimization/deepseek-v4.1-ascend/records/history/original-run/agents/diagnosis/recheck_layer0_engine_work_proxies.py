#!/usr/bin/env python3
"""Recheck existing MC2 counter proxies per rank; no device or new capture."""
import csv, hashlib, json, statistics
from pathlib import Path

T = Path(__file__).resolve().parents[2]
paths = {
    'timeline': T/'mc2-preceding-analysis/rank_layer0_timeline.csv',
    'calls': T/'mc2-arrival-analysis-v2/rank_target_calls.csv',
    'stage159': T/'agents/diagnosis/layer0-dp-type-counters.json',
    'self': Path(__file__).resolve(),
}
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
if digest(paths['stage159']) != '0482575955229199d9b2c25e1cff2590d82a27d8a91f89e0d8f3418e7be37fad':
    raise RuntimeError('stage159 identity mismatch')
with paths['timeline'].open() as f:
    timeline = [r for r in csv.DictReader(f) if r['interior_runs_1_2_3']=='True']
with paths['calls'].open() as f: calls = list(csv.DictReader(f))
lookup = {(r['rank'],int(r['target_call_ordinal'])):r for r in calls}
if len(lookup)!=len(calls) or len(timeline)!=48: raise RuntimeError('coverage mismatch')
stage = json.loads(paths['stage159'].read_text())
old = {(r['rank'],r['run']):r for r in stage['mc2_by_window']}
if len(old)!=48: raise RuntimeError('stage159 coverage mismatch')
data=[]
for t in timeline:
    rank,run=t['rank'],int(t['captured_run_in_model'])
    row=lookup[(rank,int(t['target_forward_ordinal'])*40)]
    ref=old[(rank,run)]
    if (row['source_layer']!='0' or row['model_id']!='49' or row['task_id']!=ref['task_id']
        or row['stream_id']!=ref['stream'] or abs(float(row['duration_us'])*1000-ref['duration_ns'])>0.001
        or row['mac_ratio']!=ref['aic_mac_ratio']): raise RuntimeError('event identity mismatch')
    cyc,mac,mte = float(ref['aic_cycles']),float(row['mac_ratio']),float(row['mte2_ratio'])
    if cyc<=0 or not 0<=mac<=1 or not 0<=mte<=1: raise RuntimeError('invalid counter')
    data.append({'rank':rank,'run':run,'dp':rank.split('_')[0],
                 'duration_us':float(row['duration_us']), 'aic_cycles':cyc,
                 'mac_ratio':mac,'mte2_ratio':mte,
                 'mac_cycle_proxy':cyc*mac,'mte2_cycle_proxy':cyc*mte})
groups=[]
for run in (1,2,3):
    for dp in ('dp0','dp1'):
        r=[z for z in data if z['run']==run and z['dp']==dp]
        if len(r)!=8 or len({x['rank'] for x in r})!=8: raise RuntimeError('group mismatch')
        groups.append({'run':run,'dp':dp,'n':8,'median':{k:statistics.median(x[k] for x in r) for k in
            ('duration_us','aic_cycles','mac_ratio','mte2_ratio','mac_cycle_proxy','mte2_cycle_proxy')}})
result={'scope':'48 previously attributed layer0 MC2 samples, per-rank counters and grouped medians',
        'input_sha256':{k:digest(v) for k,v in paths.items()},'groups':groups,'per_rank':data,
        'limits':['Ratios are rounded exported counters; cycle-weighted numerators are proxies, not measured MAC operations or transfer bytes.',
                  'MTE2 definition/exporter version and cycle meaning are not fully established for this capture; not a bandwidth saturation measure.',
                  'Groups are sample summaries and do not prove same request, routing, work, synchronized arrival, or causality.',
                  'Cycles are neither wall time nor additive across AIC/AIV; overlap and profiler coverage remain unknown.',
                  'No NPU activity or implementation changes.']}
print(json.dumps(result,ensure_ascii=False,indent=2))
