#!/usr/bin/env python3
"""Recheck F16 48 paired CSV rows; preserve missing/zero counters."""
import collections,hashlib,json,statistics
from pathlib import Path
T=Path(__file__).resolve().parents[2]
def load(relative, expected):
    raw=(T/relative).read_bytes(); assert hashlib.sha256(raw).hexdigest()==expected
    return json.loads(raw)
x=load('agents/fusion/f16-layer24-fixed-counters.json','70f3d84b682e54766bc0b5e5ba8b04cd4fec8bee944029167e1f5e9e84f8b0b0')
a=load('agents/fusion/f16-layer24-all-rank-run-pairs.json','26d0ab676539995816c26f697d0702941e7284983f020ff9f221f86c3d81840f')
w=load('captured-layer-attribution/module-call-metrics.json','d98aac57fbf07885e5bfe405e9f50c2e27bc22d4d5aba27e80b95e1aecf4de5a')
m=load('evidence/TRACE-EXPORT-SHA256.json','45ac6c7d22cc20601df2592ce0d78406df24a612f4aa56045566354d9ff46139')
assert x['anchors_sha256']=='26d0ab676539995816c26f697d0702941e7284983f020ff9f221f86c3d81840f'
assert x['csv_hashes']=={r['rank']:r['kernel_csv_sha256'] for r in m['exports']}
windows={(r['rank'],r['run_in_model'],r['layer']):r for r in w}
anchors={(r['rank'],r['run']):r for r in a['rows']}
pairs=collections.defaultdict(dict)
for r in x['samples']:
    key=(r['rank'],r['run']); assert r['label'] not in pairs[key]; pairs[key][r['label']]=r
    entry=anchors[key]; win=windows[(r['rank'],r['run'],24)]; original=entry[r['label']][0]; f=r['fields']
    assert r['model']==entry['model']=='49' and r['stream']==entry['main_stream']==str(win['main_stream_id'])
    assert r['start_ns']==win['window_end_ns']-(win['window_end_ns']-win['window_start_ns'])+original['start_rel_ns']
    assert r['end_ns']==win['window_start_ns']+original['end_rel_ns']
    assert f['Task ID']==original['task_id'] and f['Type']==original['type']
    assert f['Input Shapes'].strip('"')==original['input_shapes'] and f['Output Shapes'].strip('"')==original['output_shapes']
    assert f['Input Data Types']==original['input_dtypes'] and f['Output Data Types']==original['output_dtype']
assert len(pairs)==48 and all(set(p)=={'first','second'} for p in pairs.values())
meta=['Name','Accelerator Core','Block Num','Mix Block Num','OP State','Input Formats','Output Formats','Input Data Types','Output Data Types']
metadata={label:{field:dict(collections.Counter(p[label]['fields'][field] for p in pairs.values())) for field in meta} for label in ('first','second')}
fields=['aic_total_cycles','aic_mac_ratio','aic_scalar_ratio','aic_mte1_ratio','aic_mte2_ratio','aic_fixpipe_ratio','aic_icache_miss_rate','aiv_total_cycles','cube_utilization(%)']
def parse(value):
    try:return float(value)
    except (ValueError,TypeError):return None
summary=[]
for run in (1,2,3):
    group=[p for (rank,r),p in pairs.items() if r==run];assert len(group)==16
    for field in fields:
        one=[parse(p['first']['fields'].get(field)) for p in group];two=[parse(p['second']['fields'].get(field)) for p in group]
        def s(vals):
            valid=[v for v in vals if v is not None]
            return {'valid':len(valid),'missing':len(vals)-len(valid),'zeros':sum(v==0 for v in valid),
                    'min':min(valid) if valid else None,'median':statistics.median(valid) if valid else None,'max':max(valid) if valid else None}
        paired=[(u,v) for u,v in zip(one,two) if u is not None and v is not None]
        summary.append({'run':run,'field':field,'first':s(one),'second':s(two),'paired_valid':len(paired),
                        'first_greater':sum(u>v for u,v in paired),'first_equal':sum(u==v for u,v in paired),'first_less':sum(u<v for u,v in paired)})
print(json.dumps({'scope':'96 previously anchored F16 rows, rank-local 48 pairs; zero and missing preserved',
 'metadata':metadata,'summary':summary,'missing_column_samples':sum(bool(r['missing_columns']) for r in x['samples']),
 'limits':['CSV fields are reported values; the exporter cycle denominator, multi-core aggregation, formula and replay mapping are not fully established.',
 'Zero reported counters do not establish no actual device activity; ratios are rounded, not a direct bottleneck, bytes or whole-chip utilization.',
 'The same rank pair and physical shapes do not prove tensor identity, request, effective rows, kernel implementation or identical work.']},ensure_ascii=False,indent=2))
