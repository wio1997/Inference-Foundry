#!/usr/bin/env python3
"""Recompute EP3 versus peers for 80 scoped MC2 sets, without NPU."""
import hashlib,json,statistics
from decimal import Decimal
from pathlib import Path
p=Path('diagnostic-20261029/target-layer-all-kernels-16rank.jsonl')
h='0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4'
assert hashlib.sha256(p.read_bytes()).hexdigest()==h
fields=('Type','Input Shapes','Output Shapes','Input Data Types','Output Data Types','Input Formats','Output Formats','Accelerator Core','Block Num','Mix Block Num')
rows={}
comparisons={}
names=('DispatchFFNCombine','SparseFlashMla','MoeGatingTopKHash')
with p.open() as f:
 for line in f:
  x=json.loads(line)
  if x.get('kind')!='rank' or x['data']['ep'] not in range(8): continue
  ep=x['data']['ep']
  for l in x['data']['layers']:
   digest=l['scheduler_digest']
   if not digest.startswith(('db852bff987d','7ddf7ae98a00')): continue
   k=(digest,l['layer']); gs=[g for g in l['operator_groups'] if g['operator_name']=='DispatchFFNCombine']
   assert len(gs)==1 and len(gs[0]['tasks'])==1 and ep not in rows.setdefault(k,{})
   t=gs[0]['tasks'][0]; c=t['csv']
   rows[k][ep]={'scope':l['layer_scope_id'],'flow_id':t['flow_id'],'stream':c['Stream ID'],'task_id':c['Task ID'],'duration_us':t['trace_duration_us'],'signature':{n:c[n] for n in fields}}
   if ep < 7:
    selected={}
    for name in names:
     gg=[g for g in l['operator_groups'] if g['operator_name']==name]
     assert len(gg)==1 and len(gg[0]['tasks'])==1
     tt=gg[0]['tasks'][0]; cc=tt['csv']
     selected[name]={'duration_us':tt['trace_duration_us'],'stream':cc['Stream ID'],'signature':{n:cc[n] for n in fields}}
    assert len({v['stream'] for v in selected.values()})==1
    comparisons.setdefault(k,{})[ep]=selected
assert len(rows)==80
out=[]
for (digest,layer),rs in sorted(rows.items()):
 assert set(rs)==set(range(8))
 assert len({json.dumps(r['signature'],sort_keys=True) for r in rs.values()})==1
 d=Decimal(rs[3]['duration_us']);peers=[Decimal(rs[i]['duration_us']) for i in (0,1,2,4,5,6,7)]
 out.append({'digest':digest,'layer':layer,'ep3_fastest':all(d<x for x in peers),'ep3_us':str(d),'peer_min_us':str(min(peers)),'peer_median_us':str(statistics.median(peers)),'peer_max_us':str(max(peers)),'records':rs})
assert sum(x['ep3_fastest'] for x in out)==79
other=[]
for (digest,layer), rs in sorted(comparisons.items()):
 assert set(rs)==set(range(7))
 for name in names:
  assert len({json.dumps(rs[i][name]['signature'],sort_keys=True) for i in rs})==1
  d=Decimal(rs[3][name]['duration_us']); peers=[Decimal(rs[i][name]['duration_us']) for i in (0,1,2,4,5,6)]
  other.append({'digest':digest,'layer':layer,'name':name,'ep3_fastest':all(d<x for x in peers),'ep3_to_peer_median':str(d/statistics.median(peers)),'records':{i:rs[i][name] for i in rs}})
assert len(other)==240
result={'input_sha256':h,'scope':'one profiled acquisition; EP0-7; two selected digests and 40 layers each','groups':out,'cross_module_ep0_6':other}
Path('agents/diagnosis/mc2-rank-pattern-80layers.json').write_text(json.dumps(result,indent=2)+'\n')
for prefix in ('db852bff987d','7ddf7ae98a00'):
 xs=[x for x in out if x['digest'].startswith(prefix)]
 print(prefix,'fastest',sum(x['ep3_fastest'] for x in xs),'of',len(xs))
 for x in xs:
  if not x['ep3_fastest'] or x['layer']==15: print('layer',x['layer'],'ep3',x['ep3_us'],'peer min/median/max',x['peer_min_us'],x['peer_median_us'],x['peer_max_us'])
 for name in names:
  ys=[x for x in other if x['digest'].startswith(prefix) and x['name']==name]
  ratios=[Decimal(x['ep3_to_peer_median']) for x in ys]
  print(' ',name,'fastest',sum(y['ep3_fastest'] for y in ys),'median',str(statistics.median(ratios)),'range',str(min(ratios)),str(max(ratios)))
