#!/usr/bin/env python3
"""Scope-matched INT64 padding and gating records; CPU-only offline analysis."""
import hashlib,json,statistics
from decimal import Decimal
from pathlib import Path
p=Path('diagnostic-20261029/target-layer-all-kernels-16rank.jsonl')
h='0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4'
assert hashlib.sha256(p.read_bytes()).hexdigest()==h
names=('aclnnConstantPadNd_PadV3AiCpu_PadV3','MoeGatingTopKHash')
fields=('Type','Input Shapes','Output Shapes','Input Data Types','Output Data Types','Input Formats','Output Formats','Accelerator Core','Block Num','Mix Block Num')
rows=[]
with p.open() as f:
 for line in f:
  obj=json.loads(line)
  if obj.get('kind')!='rank' or obj['data']['ep'] not in range(8):continue
  rank=obj['data']['ep'];assert obj['data']['status']=='completed'
  ls=[l for l in obj['data']['layers'] if l['scheduler_digest'].startswith('7ddf7ae98a00')]
  assert len(ls)==40 and {l['layer'] for l in ls}==set(range(40))
  for l in ls:
   ts=[]
   for name in names:
    groups=[g for g in l['operator_groups'] if g['operator_name']==name]
    assert len(groups)==1 and len(groups[0]['tasks'])==1
    t=groups[0]['tasks'][0];c=t['csv']
    ts.append({'start_us':t['device_start_us'],'duration_us':t['trace_duration_us'],'stream':c['Stream ID'],'task':c['Task ID'],'flow':t['flow_id'],'signature':{k:c[k] for k in fields}})
   a,z=ts
   assert a['stream']==z['stream']
   gap=Decimal(z['start_us'])-Decimal(a['start_us'])-Decimal(a['duration_us']);assert gap>=0
   rows.append({'ep':rank,'layer':l['layer'],'scope':l['layer_scope_id'],'gap_us':str(gap),'pad':a,'gate':z})
assert len(rows)==320
for k in ('pad','gate'):assert len({json.dumps(x[k]['signature'],sort_keys=True) for x in rows})==1
assert all('INT64' in x['pad']['signature']['Input Data Types'] for x in rows)
def desc(xs):return {'min':str(min(xs)),'median':str(statistics.median(xs)),'max':str(max(xs))}
pad=[Decimal(x['pad']['duration_us']) for x in rows];gate=[Decimal(x['gate']['duration_us']) for x in rows];gap=[Decimal(x['gap_us']) for x in rows]
rank_summaries={}
for rank in range(8):
 xs=sorted((x for x in rows if x['ep']==rank),key=lambda x:Decimal(x['pad']['start_us']))
 assert len(xs)==40 and [x['layer'] for x in xs]==list(range(40))
 ends=[Decimal(x['pad']['start_us'])+Decimal(x['pad']['duration_us']) for x in xs]
 assert all(ends[i]<=Decimal(xs[i+1]['pad']['start_us']) for i in range(39))
 rank_summaries[rank]={'records':len(xs),'selected_pad_union_us':str(sum(Decimal(x['pad']['duration_us']) for x in xs)),'same_rank_intervals_nonoverlap':True}
out={'input_sha256':h,'scope':'one profiled capture, digest 7ddf, EP0-7, 40 layers each','n':len(rows),'pad_duration_us':desc(pad),'gate_duration_us':desc(gate),'gap_us':desc(gap),'pad_longer_count':sum(x>y for x,y in zip(pad,gate)),'rank_summaries':rank_summaries,'rows':rows}
Path('agents/diagnosis/int64-pad-gating-scope.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))
