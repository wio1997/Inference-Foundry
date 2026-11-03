#!/usr/bin/env python3
"""Compare two existing candidate BF16 matmul rows within each rank/run."""
import collections,hashlib,json,statistics
from decimal import Decimal
from pathlib import Path
T=Path(__file__).resolve().parents[2]
def read(path,expected):
    raw=(T/path).read_bytes(); assert hashlib.sha256(raw).hexdigest()==expected
    return json.loads(raw)
x=read('agents/diagnosis/16rank-engram-matmul-metrics.json','35ff1b007a471903324d96de5eeb67b80a13251a4c4dadd8619eead77c23db1d')
w=read('captured-layer-attribution/module-call-metrics.json','d98aac57fbf07885e5bfe405e9f50c2e27bc22d4d5aba27e80b95e1aecf4de5a')
manifest=read('evidence/TRACE-EXPORT-SHA256.json','45ac6c7d22cc20601df2592ce0d78406df24a612f4aa56045566354d9ff46139')
assert x['kernel_sha256_by_rank']=={e['rank']:e['kernel_csv_sha256'] for e in manifest['exports']}
windows={(i['rank'],i['run_in_model'],i['layer']):i for i in w}
pairs=collections.defaultdict(dict)
for r in x['samples']:
    t=r['target'];f=r['csv_fields'];key=(t['rank'],t['run']);layer=t['after_layer'];assert layer in (0,13) and layer not in pairs[key]
    first=windows[(t['rank'],t['run'],layer)];nxt=windows[(t['rank'],t['run'],layer+1)]
    assert str(first['model_id'])==str(nxt['model_id'])==t['model']=='49'
    assert str(first['main_stream_id'])==str(nxt['main_stream_id'])==t['stream']
    assert first['window_end_ns']<=t['start_ns']<t['end_ns']<=nxt['window_start_ns']
    assert int(Decimal(f['Start Time(us)'].strip())*1000)==t['start_ns']
    assert int(Decimal(f['Duration(us)'].strip())*1000)==t['end_ns']-t['start_ns']
    assert f['Type']=='MatMulV2' and f['Model ID'].strip()=='49' and f['Stream ID'].strip()==t['stream']
    assert f['Input Shapes'].strip('"')=='3,6144;25600,6144' and f['Output Shapes'].strip('"')=='3,25600'
    assert f['Input Data Types']=='DT_BF16;DT_BF16' and f['Output Data Types']=='DT_BF16'
    pairs[key][layer]=r
assert len(pairs)==48 and len({k[0] for k in pairs})==16 and all(set(v)=={0,13} for v in pairs.values())
rows=[]
for (rank,run),pair in sorted(pairs.items()):
    a,b=pair[0],pair[13];fa,fb=a['csv_fields'],b['csv_fields'];assert a['target']['stream']==b['target']['stream']
    assert a['target']['end_ns']<=b['target']['start_ns']
    row={'rank':rank,'run':run,'stream':a['target']['stream'],'duration_ns_after0':a['target']['end_ns']-a['target']['start_ns'],
         'duration_ns_after13':b['target']['end_ns']-b['target']['start_ns']}
    row['duration_difference_ns']=row['duration_ns_after0']-row['duration_ns_after13']
    for name in ('aic_total_cycles','aic_mac_ratio','aic_mte2_ratio','cube_utilization(%)'):
        row[name]={'after0':fa[name],'after13':fb[name]}
    rows.append(row)
summary=[]
for run in (1,2,3):
    subset=[r for r in rows if r['run']==run];assert len(subset)==16
    dif=[r['duration_difference_ns']/1000 for r in subset]
    s={'run':run,'pair_count':16,'first_duration_difference_us':{'min':min(dif),'median':statistics.median(dif),'max':max(dif)},
       'first_longer':sum(d>0 for d in dif),'first_equal':sum(d==0 for d in dif),'first_shorter':sum(d<0 for d in dif)}
    for field in ('aic_total_cycles','aic_mac_ratio','aic_mte2_ratio','cube_utilization(%)'):
        vs=[(float(r[field]['after0']),float(r[field]['after13'])) for r in subset]
        s[field]={'after0_median':statistics.median(a for a,b in vs),'after13_median':statistics.median(b for a,b in vs),
                  'after0_greater':sum(a>b for a,b in vs),'equal':sum(a==b for a,b in vs),'after0_less':sum(a<b for a,b in vs)}
    summary.append(s)
print(json.dumps({'scope':'same rank/run two shape-compatible boundary candidates; no cross-rank timestamps',
                  'input_sha256':{'metrics':'35ff1b007a471903324d96de5eeb67b80a13251a4c4dadd8619eead77c23db1d','windows':'d98aac57fbf07885e5bfe405e9f50c2e27bc22d4d5aba27e80b95e1aecf4de5a','manifest':'45ac6c7d22cc20601df2592ce0d78406df24a612f4aa56045566354d9ff46139'},'summary':summary,'rows':rows,
                  'limits':['Window labels are source-attributed candidates; same rank/run is not request, tensor, binary or equal-work proof.',
                            'Ratios and cycles are exporter fields with incomplete semantics and rounded displayed precision; they are not bytes, FLOPs, independent waits or causal proof.',
                            'No NPU or new profile.']},ensure_ascii=False,indent=2))
