#!/usr/bin/env python3
import argparse,csv,hashlib,json,statistics
from collections import defaultdict,Counter
from decimal import Decimal
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--task-root',required=True);ap.add_argument('--trace-root',required=True);a=ap.parse_args();T=Path(a.task_root);R=Path(a.trace_root)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p,s):
 p=T/p
 if sha(p)!=s:raise ValueError('input hash mismatch '+str(p))
 return json.loads(p.read_text())
ws=read('captured-layer-attribution/module-call-metrics.json','d98aac57fbf07885e5bfe405e9f50c2e27bc22d4d5aba27e80b95e1aecf4de5a');manifest=read('evidence/TRACE-EXPORT-SHA256.json','45ac6c7d22cc20601df2592ce0d78406df24a612f4aa56045566354d9ff46139');summary=read('agents/diagnosis/matmul-hcpre-shapes.json','2d09b0226e8882039d9b866848ab3a9b7625b66f67787b3eb44cf81d0439028e')
g=[g for g in summary['groups'] if g['type']=='QuantBatchMatmulV3' and g['signature']['Output Shapes']=='"3,32768"'];assert len(g)==1 and g[0]['count']==1920;sig=g[0]['signature'];expected={e['rank']:e['kernel_csv_sha256'] for e in manifest['exports']};windows=defaultdict(list)
for w in ws:
 if w['run_in_model'] in (1,2,3):windows[w['rank']].append(w)
if len(expected)!=16 or set(expected)!=set(windows) or any(len(v)!=120 for v in windows.values()):raise ValueError('unexpected windows')
rows=[];hashes={}
for rank in sorted(expected):
 p=R/rank/'ASCEND_PROFILER_OUTPUT/kernel_details.csv';hashes[rank]=sha(p)
 if hashes[rank]!=expected[rank]:raise ValueError('csv hash mismatch')
 w=windows[rank];ev=defaultdict(list)
 if any(str(q['model_id'])!='49' for q in w) or len({str(q['main_stream_id']) for q in w})!=1:raise ValueError('model/stream mismatch')
 with p.open(newline='') as f:
  for z in csv.DictReader(f):
   if z['Model ID']!='49' or z['Type'] not in ('QuantBatchMatmulV3','ScatterNdUpdateSk'):continue
   start=Decimal(z['Start Time(us)'].strip())*1000;dur=Decimal(z['Duration(us)'].strip())*1000
   if not(start.is_finite() and dur.is_finite() and start==start.to_integral_value() and dur==dur.to_integral_value() and dur>0):continue
   start=int(start);end=start+int(dur);hits=[x for x in w if x['window_start_ns']<=start and end<=x['window_end_ns']]
   if len(hits)>1:raise ValueError('overlapping target windows')
   if not hits:continue
   x=hits[0];main=str(x['main_stream_id']);key=(x['run_in_model'],x['layer'])
   isq=z['Type']=='QuantBatchMatmulV3' and z['Stream ID']==main and all(z[k]==v for k,v in sig.items() if k!='kernel_name_sha256') and hashlib.sha256(z['Name'].encode()).hexdigest()==sig['kernel_name_sha256']
   iss=z['Type']=='ScatterNdUpdateSk' and z['Stream ID']!=main and z['Stream ID'].isdigit() and z['Input Shapes'].strip('"').endswith(';24,2;24,512')
   if isq or iss:ev[key].append({'kind':'q' if isq else 'scatter','stream':z['Stream ID'],'task':z['Task ID'],'start_ns':start,'end_ns':end})
 for x in w:
  v=ev[(x['run_in_model'],x['layer'])];q=[y for y in v if y['kind']=='q'];s=[y for y in v if y['kind']=='scatter']
  if len(q)!=1:raise ValueError('Q signature ambiguous')
  r={'rank':rank,'run':x['run_in_model'],'layer':x['layer'],'q_task':q[0]['task'],'q_stream':q[0]['stream'],'q_duration_ns':q[0]['end_ns']-q[0]['start_ns'],'scatter_count':len(s)}
  if len(s)==1:
   z=s[0];over=max(0,min(q[0]['end_ns'],z['end_ns'])-max(q[0]['start_ns'],z['start_ns']));r.update(scatter_task=z['task'],scatter_stream=z['stream'],scatter_duration_ns=z['end_ns']-z['start_ns'],overlap_ns=over,scatter_tail_after_q_ns=z['end_ns']-q[0]['end_ns'])
  rows.append(r)
if len(rows)!=1920:raise ValueError('incorrect total')
def summarise(v):
 unique=[x for x in v if x['scatter_count']==1];over=[x for x in unique if x['overlap_ns']>0]; med=lambda vals:statistics.median(vals) if vals else None
 return {'windows':len(v),'scatter_counts':dict(sorted(Counter(str(x['scatter_count']) for x in v).items())),'unique_scatter_overlap_windows':len(over),'q_duration_ns_median_in_unique':med([x['q_duration_ns'] for x in unique]),'overlap_ns_median_in_unique':med([x['overlap_ns'] for x in unique]),'scatter_duration_ns_median_in_unique':med([x['scatter_duration_ns'] for x in unique]),'scatter_tail_after_q_ns_min_median_max':([min(x['scatter_tail_after_q_ns'] for x in unique),med([x['scatter_tail_after_q_ns'] for x in unique]),max(x['scatter_tail_after_q_ns'] for x in unique)] if unique else None)}
out={'scope':'16 rank x 3 interior runs x 40 attributed windows; same rank/model49; main stream exact Q signature, other numeric stream scatter whose input ends ;24,2;24,512, both fully window-contained','input_csv_sha256':hashes,'samples':rows,'overall':summarise(rows),'by_rank':{rank:summarise([r for r in rows if r['rank']==rank]) for rank in sorted(expected)},'by_run':{str(run):summarise([r for r in rows if r['run']==run]) for run in (1,2,3)},'limitations':['Selected records only; no physical tensor identity, source role, event dependency, runtime configuration, cross-stream clock calibration, or end-to-end attribution proven.','Incomplete export and profiling perturbation; non-overlap does not prove idle hardware.','Three runs from one profile, different ranks are not independent experiments or necessarily same request.','Span intersection does not establish simultaneous execution, no contention or achievable speedup.']}
print(json.dumps(out,ensure_ascii=False,indent=2))
