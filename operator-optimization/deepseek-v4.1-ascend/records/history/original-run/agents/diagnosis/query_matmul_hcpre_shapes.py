#!/usr/bin/env python3
"""Read 48 fixed target runs; classify exported matmul/HcPre rows by physical metadata, not by inferred semantics."""
import argparse,csv,hashlib,json,statistics
from collections import defaultdict,Counter
from decimal import Decimal,InvalidOperation
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--task-root',required=True);p.add_argument('--trace-root',required=True);a=p.parse_args()
T=Path(a.task_root);R=Path(a.trace_root)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(rel,want):
 p=T/rel
 if sha(p)!=want:raise ValueError('input SHA mismatch '+rel)
 return json.loads(p.read_text())
wins=load('captured-layer-attribution/module-call-metrics.json','d98aac57fbf07885e5bfe405e9f50c2e27bc22d4d5aba27e80b95e1aecf4de5a')
manifest=load('evidence/TRACE-EXPORT-SHA256.json','45ac6c7d22cc20601df2592ce0d78406df24a612f4aa56045566354d9ff46139')
expected={x['rank']:x['kernel_csv_sha256'] for x in manifest['exports']}; windows=defaultdict(list)
for w in wins:
 if w['run_in_model'] in (1,2,3):windows[w['rank']].append(w)
if len(expected)!=16 or set(expected)!=set(windows) or any(len(v)!=120 for v in windows.values()):raise ValueError('unexpected window dimensions')
fields=['Input Shapes','Input Data Types','Input Formats','Output Shapes','Output Data Types','Output Formats','Block Num','Mix Block Num','Accelerator Core']
types={'QuantBatchMatmulV3','HcPre'}; samples=[]; issues=[]; actual={}; selected=Counter()
for rank in sorted(expected):
 path=R/rank/'ASCEND_PROFILER_OUTPUT/kernel_details.csv';actual[rank]=sha(path)
 if actual[rank]!=expected[rank]:raise ValueError('trace sha mismatch '+rank)
 ws=windows[rank]
 if any(str(w['model_id'])!='49' for w in ws):raise ValueError('unexpected model')
 streams={str(w['main_stream_id']) for w in ws}
 if len(streams)!=1:raise ValueError('unexpected streams')
 with path.open(newline='') as f:
  reader=csv.DictReader(f);required=set(fields+['Name','Type','Model ID','Stream ID','Start Time(us)','Duration(us)'])
  if not required.issubset(reader.fieldnames):raise ValueError('missing columns '+str(required-set(reader.fieldnames)))
  for z in reader:
   typ=z['Type'].strip()
   if typ not in types or z['Model ID']!='49' or z['Stream ID'] not in streams:continue
   try:
    start=Decimal(z['Start Time(us)'])*1000; dur=Decimal(z['Duration(us)'])*1000
    if not(start.is_finite() and dur.is_finite() and dur>0 and start==start.to_integral_value() and dur==dur.to_integral_value()):raise ValueError('bad timestamp')
   except (InvalidOperation,ValueError):issues.append({'rank':rank,'type':typ,'reason':'invalid time'});continue
   start=int(start);dur=int(dur);hits=[w for w in ws if w['window_start_ns']<=start and start+dur<=w['window_end_ns']]
   if len(hits)>1:raise ValueError('overlapping windows')
   if not hits:continue
   w=hits[0]; selected[(rank,w['run_in_model'],w['layer'],typ)]+=1
   sig={k:z[k] for k in fields};sig['kernel_name_sha256']=hashlib.sha256(z['Name'].encode()).hexdigest()
   samples.append({'rank':rank,'run':w['run_in_model'],'layer':w['layer'],'type':typ,'start_ns':start,'duration_ns':dur,'signature':sig})
for rank,ws in windows.items():
 for w in ws:
  for typ in types:
   n=selected[(rank,w['run_in_model'],w['layer'],typ)]
   if typ=='HcPre' and n!=2:issues.append({'rank':rank,'run':w['run_in_model'],'layer':w['layer'],'type':typ,'count':n})
   if typ=='QuantBatchMatmulV3' and n not in (3,4):issues.append({'rank':rank,'run':w['run_in_model'],'layer':w['layer'],'type':typ,'count':n})
groups=defaultdict(list)
for s in samples:
 key=json.dumps({'type':s['type'],'signature':s['signature']},sort_keys=True,ensure_ascii=False)
 groups[key].append(s)
out=[]
for key,rows in groups.items():
 d=sorted(s['duration_ns']/1000 for s in rows);bylayer=Counter(s['layer'] for s in rows)
 out.append({**json.loads(key),'count':len(rows),'duration_sum_us':sum(d),'duration_median_us':statistics.median(d),'duration_min_us':d[0],'duration_max_us':d[-1],'layers':dict(sorted(bylayer.items())),'run_counts':dict(Counter(s['run'] for s in rows)),'rank_count':len({s['rank'] for s in rows})})
out.sort(key=lambda z:z['duration_sum_us'],reverse=True)
print(json.dumps({'status':'COMPLETE' if not issues else 'INCOMPLETE','scope':'16 ranks x 3 interior runs x 40 exact target layer windows, exported main stream model49; fully contained rows only','kernel_sha256_by_rank':actual,'input_sha256':{'windows':'d98aac57fbf07885e5bfe405e9f50c2e27bc22d4d5aba27e80b95e1aecf4de5a','manifest':'45ac6c7d22cc20601df2592ce0d78406df24a612f4aa56045566354d9ff46139'},'groups':out,'row_count':len(samples),'issues':issues,'limits':['Exported shape/layout/kernel hash is not a proven operator role, tensor identity, weight residency, or dynamic valid tokens.','Layer-contained task sum mixes rank and run and cannot be interpreted as wall time, critical path, NPU utilization or achievable savings.','No cross-rank temporal comparisons. Profiles differ from unprofiled baseline.','Fully contained rows selected; type counts and validation establish coverage only for these fixed windows.','No device work.']},ensure_ascii=False,indent=2))
