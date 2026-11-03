#!/usr/bin/env python3
"""Read existing kernel CSVs only; correlate four window markers without joins on task ID alone."""
import argparse,bisect,csv,hashlib,json
from collections import Counter,defaultdict
from decimal import Decimal
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--task-root',required=True);ap.add_argument('--trace-root',required=True);a=ap.parse_args()
t=Path(a.task_root);wp=t/'captured-layer-attribution/module-call-metrics.json';mp=t/'evidence/TRACE-EXPORT-SHA256.json'
windows=[x for x in json.loads(wp.read_text()) if x['run_in_model'] in (1,2,3)]
assert len(windows)==1920 and len({x['rank'] for x in windows})==16
by=defaultdict(list)
for w in windows:by[(w['rank'],str(w['model_id']),str(w['main_stream_id']))].append(w)
for g in by.values():
 g.sort(key=lambda x:x['window_start_ns'])
 assert all(x['window_end_ns']<=y['window_start_ns'] for x,y in zip(g,g[1:]))
expected={x['rank']:x['kernel_csv_sha256'] for x in json.loads(mp.read_text())['exports']}
assert len(expected)==16
seen=defaultdict(list);hashes={}
for rank in sorted({w['rank'] for w in windows}):
 p=Path(a.trace_root)/rank/'ASCEND_PROFILER_OUTPUT/kernel_details.csv'
 if p.stat().st_size>50*1024*1024:raise RuntimeError('CSV exceeds per-rank bound')
 sh=hashlib.sha256(p.read_bytes()).hexdigest();assert sh==expected[rank];hashes[rank]=sh
 with p.open(newline='') as f:
  for z in csv.DictReader(f):
   if z['Type'] not in ('HcPre','HcPost'):continue
   key=(rank,z['Model ID'],z['Stream ID']);spans=by.get(key,[])
   if not spans:continue
   start=int(Decimal(z['Start Time(us)'].strip())*1000);end=start+int(Decimal(z['Duration(us)'].strip())*1000)
   i=bisect.bisect_right([w['window_start_ns'] for w in spans],start)-1
   if i<0:continue
   w=spans[i]
   if not(w['window_start_ns']<=start and end<=w['window_end_ns']):continue
   unit=(rank,w['run_in_model'],w['layer']);seen[unit].append({'type':z['Type'],'id':z['Task ID'],'start':start,'end':end,
    'input_shapes':z['Input Shapes'],'input_dtypes':z['Input Data Types'],
    'output_shapes':z['Output Shapes'],'output_dtypes':z['Output Data Types']})
results=[];unknown=[]
for w in windows:
 unit=(w['rank'],w['run_in_model'],w['layer']);ev=seen[unit];markers=[]
 for idx,(typ,ident) in enumerate(zip(('HcPre','HcPost','HcPre','HcPost'),w['marker_task_ids'])):
  hit=[e for e in ev if e['type']==typ and e['id']==str(ident)]
  if len(hit)!=1:
   unknown.append({'rank':w['rank'],'run':w['run_in_model'],'layer':w['layer'],'marker':idx,'matching_rows':len(hit)})
   break
  markers.append(hit[0])
 else:
  if not all(x['end']<=y['start'] for x,y in zip(markers,markers[1:])):
   unknown.append({'rank':w['rank'],'run':w['run_in_model'],'layer':w['layer'],'reason':'marker order inconsistent'});continue
  for phase,index in (('attention_pre',0),('ffn_pre',2)):
   e=markers[index];inputs=e['input_shapes'].strip('"').split(';');first=inputs[0].strip() if inputs else ''
   dims=first.split(',') if first else []
   d=[int(x.strip()) for x in dims] if dims and all(x.strip().isdigit() for x in dims) else None
   bs=(d[0] if len(d)==3 else d[0]*d[1] if len(d)==4 else None) if d else None
   results.append({'rank':w['rank'],'run':w['run_in_model'],'layer':w['layer'],'phase':phase,
    'marker_id':e['id'],'input_shapes':e['input_shapes'],'input_dtypes':e['input_dtypes'],
    'output_shapes':e['output_shapes'],'output_dtypes':e['output_dtypes'],'parsed_first_input_dims':d,'candidate_bs':bs})
groups=Counter((r['phase'],r['input_shapes'],r['input_dtypes'],r['output_shapes'],r['output_dtypes'],r['candidate_bs']) for r in results)
out={'scope':'16 ranks x three existing interior target replay ordinals x 40 source-derived layer brackets',
 'window_sha256':hashlib.sha256(wp.read_bytes()).hexdigest(),'manifest_sha256':hashlib.sha256(mp.read_bytes()).hexdigest(),
 'csv_hashes':hashes,'window_count':len(windows),'complete_correlations':len(results)//2,'unknown_windows':unknown,
 'shape_groups':[{'phase':k[0],'input_shapes':k[1],'input_dtypes':k[2],'output_shapes':k[3],'output_dtypes':k[4],'candidate_bs':k[5],'count':v} for k,v in sorted(groups.items(),key=lambda p:str(p[0]))],
 'records':results,'limitations':['Marker attribution based on source sequence, model/stream/window and role invariants, not direct tensor identity.',
 'Graph task IDs repeat; unique matching is within same rank/model/stream/nonoverlapping time window only.',
 'CSV shapes are physical reported profiler metadata, not proof of input values, actual tiling data or bound source/binary.',
 'No AIV core count, tiling key, curRowFactor or runtime output numerical result measured.']}
print(json.dumps(out,ensure_ascii=False,indent=2))
