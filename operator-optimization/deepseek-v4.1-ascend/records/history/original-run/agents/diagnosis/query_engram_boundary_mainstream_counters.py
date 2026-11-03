#!/usr/bin/env python3
"""Read frozen CSV counters for three pre-existing layer gaps; no NPU operations."""
import argparse,csv,hashlib,json
from collections import defaultdict,Counter
from decimal import Decimal,InvalidOperation
from pathlib import Path

ap=argparse.ArgumentParser();ap.add_argument('--task-root',required=True);ap.add_argument('--trace-root',required=True);args=ap.parse_args()
t=Path(args.task_root); trace=Path(args.trace_root)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
inputs={'windows':('captured-layer-attribution/module-call-metrics.json','d98aac57fbf07885e5bfe405e9f50c2e27bc22d4d5aba27e80b95e1aecf4de5a'),'controls':('agents/diagnosis/engram-boundary-controls.json','b88c60f030b6994934aed5cfd37a50b5ab2a0e23060149a27c66415ea4d4e765'),'manifest':('evidence/TRACE-EXPORT-SHA256.json','45ac6c7d22cc20601df2592ce0d78406df24a612f4aa56045566354d9ff46139')}
for label,(path,digest) in inputs.items():
 if sha(t/path)!=digest:raise RuntimeError('input SHA mismatch: '+label)
read=lambda name:json.loads((t/inputs[name][0]).read_text())
controls=read('controls'); manifest=read('manifest'); win=defaultdict(dict)
for w in read('windows'):
 if w['run_in_model'] in (1,2,3):
  k=(w['rank'],w['run_in_model']); layer=w['layer']
  if layer in win[k]:raise RuntimeError('duplicate layer')
  win[k][layer]=w
cmap={(x['rank'],x['run'],x['after_layer']):x for x in controls['samples']}
expected={x['rank']:x['kernel_csv_sha256'] for x in manifest['exports']}
if len(expected)!=16 or expected!=controls['kernel_sha256_by_rank'] or len(win)!=48 or len(cmap)!=144:raise RuntimeError('input scope mismatch')
byrank=defaultdict(list)
for (rank,run), layers in win.items():
 if set(layers)!=set(range(40)):raise RuntimeError('layer coverage')
 for l in (0,12,13):
  a,b=layers[l],layers[l+1]; c=cmap[(rank,run,l)]
  start,end=a['window_end_ns'],b['window_start_ns']; model=str(a['model_id']);stream=str(a['main_stream_id'])
  if end<start or end-start!=c['gap_ns'] or model!='49' or str(b['model_id'])!=model or str(b['main_stream_id'])!=stream or c['main_stream']!=stream:raise RuntimeError('bad boundary')
  byrank[rank].append({'rank':rank,'run':run,'after_layer':l,'start_ns':start,'end_ns':end,'model':model,'stream':stream,'den':Decimal(0),'num':Decimal(0),'positive':0,'skipped':0,'type_counts':Counter(),'crossing':0})
if set(byrank)!=set(expected):raise RuntimeError('rank mismatch')
def dec(v):
 try:
  a=Decimal(v.strip());return a if a.is_finite() else None
 except (InvalidOperation,AttributeError):return None
for rank in sorted(byrank):
 p=trace/rank/'ASCEND_PROFILER_OUTPUT/kernel_details.csv'
 if sha(p)!=expected[rank]:raise RuntimeError('CSV SHA mismatch: '+rank)
 ws=byrank[rank]
 with p.open(newline='') as f:
  reader=csv.DictReader(f); req={'Type','Model ID','Stream ID','Start Time(us)','Duration(us)','aic_total_cycles','aic_mac_ratio'}
  if not req.issubset(reader.fieldnames or ()):raise RuntimeError('missing CSV columns')
  for z in reader:
   if z['Model ID']!='49' or 'metadata' in z['Type'].lower():continue
   relevant=[w for w in ws if z['Stream ID']==w['stream']]
   if not relevant:continue
   a=dec(z['Start Time(us)']);d=dec(z['Duration(us)'])
   if a is None or d is None or d<0:raise RuntimeError('bad timestamp on relevant model/stream')
   start=a*1000;end=(a+d)*1000
   if start!=start.to_integral_value() or end!=end.to_integral_value():raise RuntimeError('subnanosecond event')
   for w in relevant:
    contained = w['start_ns']<=start and end<=w['end_ns']
    if not contained:
     if start<w['end_ns'] and end>w['start_ns']:w['crossing']+=1
     continue
    w['type_counts'][z['Type']]+=1
    c=dec(z['aic_total_cycles']);ratio=dec(z['aic_mac_ratio'])
    if c is None or c<=0:w['skipped']+=1;continue
    if ratio is None or not Decimal(0)<=ratio<=Decimal(1):raise RuntimeError('invalid positive-cycle ratio')
    w['num']+=c*ratio;w['den']+=c;w['positive']+=1
rows=[]
for rank in sorted(byrank):
 for w in byrank[rank]:
  w['ratio']=str(w['num']/w['den']) if w['den'] else None
  w['num']=str(w['num']);w['den']=str(w['den']);w['type_counts']=dict(w['type_counts']);rows.append(w)
print(json.dumps({'scope':'144 pre-existing layer-boundary intervals, model49 main stream, complete-contained tasks only','inputs_sha256':{k:v[1] for k,v in inputs.items()},'kernel_sha256_by_rank':expected,'rows':rows,'limits':['Each boundary interval is inferred from layer markers, not a verified Engram module; it may include other work.','Only selected model and main stream; unknown or nonpositive cycles omitted, overlapping records not deduplicated; incomplete captured work and counter rounding remain.','Positive cycle-weighted MAC proxy is not whole NPU utilization, throughput, idle or time; empty denominator is unknown, never zero.','Different groups or ranks need not contain identical workload/request; no causal inference or speedup.','Boundary crossing excluded and counted, cannot assume no relevant work.'],'no_device_work':True},ensure_ascii=False,indent=2))
