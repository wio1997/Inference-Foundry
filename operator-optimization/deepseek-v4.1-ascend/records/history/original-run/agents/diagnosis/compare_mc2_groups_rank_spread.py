#!/usr/bin/env python3
"""In-group rank spread of previously extracted MC2 records; CPU only."""
import hashlib,json,statistics
from decimal import Decimal
from pathlib import Path
p=Path('diagnostic-20261029/target-layer-all-kernels-16rank.jsonl')
h='0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4'
assert hashlib.sha256(p.read_bytes()).hexdigest()==h
fields=('Type','Input Shapes','Output Shapes','Input Data Types','Output Data Types','Input Formats','Output Formats','Accelerator Core','Block Num','Mix Block Num')
patterns=[('7ddf7ae98a00',0),('db852bff987d',0),('1b826883b782',8),('8c18c293afc6',8)]
rows={}
with p.open() as f:
 for line in f:
  x=json.loads(line)
  if x.get('kind')!='rank':continue
  d=x['data'];ep=d['ep'];assert d['status']=='completed'
  for l in d['layers']:
   match=[(prefix,start) for prefix,start in patterns if l['scheduler_digest'].startswith(prefix) and start<=ep<start+8]
   if not match:continue
   assert len(match)==1
   prefix,start=match[0];key=(prefix,l['layer']);gs=[g for g in l['operator_groups'] if g['operator_name']=='DispatchFFNCombine']
   assert len(gs)==1 and len(gs[0]['tasks'])==1 and ep not in rows.setdefault(key,{})
   t=gs[0]['tasks'][0];c=t['csv'];v=Decimal(t['trace_duration_us']);assert v>0
   rows[key][ep]={'scope':l['layer_scope_id'],'flow':t['flow_id'],'stream':c['Stream ID'],'task':c['Task ID'],'duration_us':str(v),'signature':{n:c[n] for n in fields}}
assert len(rows)==160
out=[]
for prefix,start in patterns:
 groups=[]
 for layer in range(40):
  rs=rows[(prefix,layer)];assert set(rs)==set(range(start,start+8))
  assert len({json.dumps(r['signature'],sort_keys=True) for r in rs.values()})==1
  vals={ep:Decimal(v['duration_us']) for ep,v in rs.items()};lo=min(vals.values());hi=max(vals.values())
  winners=[ep for ep in vals if vals[ep]==lo];assert len(winners)==1
  groups.append({'layer':layer,'min_rank':winners[0],'max_min_ratio':str(hi/lo),'records':rs})
 counts={str(ep):sum(g['min_rank']==ep for g in groups) for ep in range(start,start+8)}
 ratios=[Decimal(g['max_min_ratio']) for g in groups]
 print(prefix,'winner counts',counts,'max/min median',statistics.median(ratios),'range',min(ratios),max(ratios))
 out.append({'digest_prefix':prefix,'rank_start':start,'winner_counts':counts,'median_ratio':str(statistics.median(ratios)),'groups':groups})
Path('agents/diagnosis/mc2-groups-rank-spread.json').write_text(json.dumps({'source_sha256':h,'groups':out},indent=2)+'\n')
