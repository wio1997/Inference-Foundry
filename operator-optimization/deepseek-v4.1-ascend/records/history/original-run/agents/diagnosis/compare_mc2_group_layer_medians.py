#!/usr/bin/env python3
"""CPU-only conditional comparison of existing MC2 exported groups."""
import collections,hashlib,json,statistics
from pathlib import Path
root=Path(__file__).resolve().parents[2]
src=root/'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl'
sha=hashlib.sha256(src.read_bytes()).hexdigest()
assert sha=='0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4'
keys=('Name','Type','Input Shapes','Input Data Types','Input Formats','Output Shapes','Output Data Types','Output Formats','Accelerator Core','Block Num','Mix Block Num')
rows=collections.defaultdict(lambda:collections.defaultdict(list))
selected={'1b826883b782ba77814d5bd40fcd3e46c759e380069db2e066ff0e9cb4bce89d','db852bff987d','8c18c293afc6','7ddf7ae98a00'}
for line in src.open():
 d=json.loads(line)
 if d.get('kind')!='rank':continue
 ep=d['data']['ep']
 for layer in d['data']['layers']:
  digest=layer['scheduler_digest']; matches=[x for x in selected if digest.startswith(x)]
  if not matches:continue
  for g in layer['operator_groups']:
   if g['operator_name']!='DispatchFFNCombine':continue
   for t in g['tasks']:
    c=t['csv'];duration=float(t['trace_duration_us']);assert duration>0
    rows[matches[0]][layer['layer']].append({'ep':ep,'duration_us':duration,'signature':[c[k] for k in keys]})
comparisons=[]
for label,high,low in [('1023','1b826883b782ba77814d5bd40fcd3e46c759e380069db2e066ff0e9cb4bce89d','db852bff987d'),('643','8c18c293afc6','7ddf7ae98a00')]:
 result=[]
 for layer in range(40):
  a,b=rows[high][layer],rows[low][layer]
  assert len(a)==len(b)==8 and {x['ep'] for x in a}==set(range(8,16)) and {x['ep'] for x in b}==set(range(8))
  assert len({json.dumps(x['signature'],sort_keys=True) for x in a+b})==1
  assert a[0]['signature'][2].strip('"').startswith(label+',5120;')
  ma,mb=statistics.median(x['duration_us'] for x in a),statistics.median(x['duration_us'] for x in b)
  result.append({'layer':layer,'group_high_ep_median_us':ma,'group_low_ep_median_us':mb,'ratio':ma/mb})
 v=[x['ratio'] for x in result]
 comparisons.append({'physical_input_rows':int(label),'layer_count':40,'ratio_min':min(v),'ratio_median':statistics.median(v),'ratio_max':max(v),'count_high_group_longer':sum(x>1 for x in v),'layers':result})
p=root/'agents/diagnosis/mc2-group-layer-median-comparison.json'
p.write_text(json.dumps({'source_sha256':sha,'signature_fields':keys,'comparisons':comparisons,'limits':['Different digest, different 8-rank group; these are group medians, not matched same-rank pairs or a controlled experiment.','Same physical signature does not establish the same valid expert workload, routing, implementation or runtime conditions.','Only these profiled MC2 records; no claim of all task types, online latency, causality or speedup.']},indent=2)+'\n')
for c in comparisons:print(c['physical_input_rows'],c['layer_count'],c['count_high_group_longer'],round(c['ratio_min'],3),round(c['ratio_median'],3),round(c['ratio_max'],3))
print('output sha256',hashlib.sha256(p.read_bytes()).hexdigest())
