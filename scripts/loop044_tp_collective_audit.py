#!/usr/bin/env python3
"""HCCL adjacency and payload audit in 15 valid Run107 target windows."""
import glob,json,statistics
from collections import Counter
from pathlib import Path
root=Path(__file__).resolve().parents[1]
prior=json.loads((root/'evidence/20260924_loop038_cycle/run107/target_window.json').read_text())
excluded={(x['rank'],x['cycle']) for x in prior['excluded_windows']}
rows=[]
for rank in range(8):
 paths=glob.glob(str(root/f'evidence/20260924_loop038_cycle/run107/profile/rank{rank}_*/ASCEND_PROFILER_OUTPUT/trace_view.json'))
 assert len(paths)==1
 events=json.loads(Path(paths[0]).read_text())
 scopes=sorted((e for e in events if e.get('name')=='extreme::target' and e.get('cat')=='cpu_op'),key=lambda e:float(e['ts']))
 details=sorted((e for e in events if e.get('name','').startswith('hcom_') and '__' in e.get('name','') and 'count' in e.get('args',{})),key=lambda e:float(e['ts']))
 for cycle,scope in zip((64,65),scopes):
  if (rank,cycle) in excluded:continue
  start=float(scope['ts']);end=start+float(scope['dur'])
  kernels=sorted((e for e in events if start<=float(e.get('ts',-1))<end and e.get('args',{}).get('Task Type','').startswith('KERNEL_')),key=lambda e:float(e['ts']))
  h=[(i,e) for i,e in enumerate(kernels) if e['name'].startswith('hcom_')]
  assert len(h)==260
  adjacent=[]
  for (i,e),(j,f) in zip(h,h[1:]):
   if j!=i+1 or e['name']!='hcom_allGather_' or f['name']!='hcom_allGather_':continue
   pair=[]
   for g in (e,f):
    t=float(g['ts'])
    matches=[d for d in details if d['name'].startswith('hcom_allGather__') and abs(float(d['ts'])-t)<1000]
    assert matches
    d=min(matches,key=lambda x:abs(float(x['ts'])-t))
    pair.append({'data_type':d['args']['data_type'],'count':d['args']['count'],'duration_ms':float(g['dur'])/1000})
   adjacent.append(pair)
  assert 0<len(adjacent)<=43,(rank,cycle,len(adjacent))
  rows.append({'rank':rank,'cycle':cycle,'pairs':adjacent,
   'pair_duration_ms':sum(v['duration_ms'] for pair in adjacent for v in pair)})
assert len(rows)==15
shapes=Counter((p[0]['data_type'],p[0]['count'],p[1]['data_type'],p[1]['count']) for r in rows for p in r['pairs'])
def desc(v):return {'median':statistics.median(v),'min':min(v),'max':max(v)}
out={'run':'run152','source':'Run107 15 valid target windows, HCCL KERNEL events and detailed payload events',
 'adjacent_allgather_pairs_per_window':desc([len(r['pairs']) for r in rows]),'pair_payload_shapes':[{ 'first_dtype':a,'first_count':b,'second_dtype':c,'second_count':d,'occurrences':n} for (a,b,c,d),n in shapes.items()],
 'pair_duration_ms_per_window':desc([r['pair_duration_ms'] for r in rows]),
 'second_fp32_duration_ms_per_window':desc([sum(pair[1]['duration_ms'] for pair in r['pairs']) for r in rows]),
 'windows':rows,
 'decision':'Most layers have an adjacent BF16 allGather of 49152 elements and FP32 allGather of 3072 elements; other layer pairs are separated by compute. Different dtypes prevent trivial buffer coalescing; the small FP32 collective is ~0.3ms/window, not a >=5ms opportunity. The other collectives are interleaved with dependent compute/attention/MoE. No safe TP8 collective removal established.'}
p=root/'evidence/20260925_loop044_target/run152/tp_collective_audit.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='windows'},indent=2))
